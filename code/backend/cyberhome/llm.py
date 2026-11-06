"""LLM Provider 抽象层 —— 单一 ChatModel 接口 + 调用护栏（F04 / 决策 35、36）。

三条红线钉在这里：
1. **无 key 也能跑**：工厂在无密钥时返回确定性 `StubChatModel`，不碰网络、同输入同
   输出——本地 / CI / golden test 的地基。
2. **SDK 不泄漏到业务层**：reactor/prompt 层只见 `ChatModel.chat()`，看不到 openai。
   `from openai import OpenAI` **延迟**到 adapter 真正建连时才 import——stub 路径完全
   不 import openai，环境没装也不崩。
3. **脱敏**：日志/异常绝不打印 api_key；prompt 过长只记长度摘要，不落原文。

DeepSeek 与 MiMo 都走 OpenAI-compatible，且都用标准 Bearer 鉴权，故**共用基类默认
`_client_kwargs()`**；唯一差异是 `_max_tokens_param`（DeepSeek=`max_tokens`，
MiMo=`max_completion_tokens`）。仍保留独立 adapter 类（决策 36）以承载 provider 语义
与未来可能的分化，但当前差异已收敛到一处可覆盖点。
超时与有限重试**直接交给 openai SDK**（`timeout` / `max_retries`）——SDK 自带指数退避，
自己重写一遍是不必要的架构税（优雅原则）。
"""

from __future__ import annotations

import json
import logging
import os
import time
import importlib.util
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import threading
from typing import Iterator

from .config import LLMConfig, load_llm_config

logger = logging.getLogger("cyberhome.llm")

# 日志里 prompt 摘要的截断上限：只记长度与前缀，绝不落全文（脱敏 + 防日志爆炸）。
_MAX_LOG_PROMPT = 160

_DEFAULT_QUEUE_RPM = 10
_DEFAULT_QUEUE_CONCURRENCY = 1
_QUEUE_LOCK = threading.Lock()
_QUEUE_CACHE: dict[tuple[int, int], "_LLMRequestLimiter"] = {}

# stub 在「提示要求 JSON」时返回的默认占位结构。键对齐 F05 文档口径
# （是否开口 / 理由 / 台词 / 情绪微调 / 记忆摘要），让 F06 无 key 也能解析跑通。
_DEFAULT_JSON_TEMPLATE: dict = {
    "speak": True,
    "reason": "stub 占位：无 key 降级路径，无真实推理。",
    "line": "",
    "mood": "calm",
    "memory_note": "",
}


def openai_sdk_available() -> bool:
    """Whether the OpenAI SDK needed by OpenAI-compatible providers is importable."""
    return importlib.util.find_spec("openai") is not None


def llm_runtime_status(config: LLMConfig | None = None) -> str:
    """Human-readable startup status without exposing secrets."""
    config = config or load_llm_config()
    if config.provider == "stub":
        return "stub provider"
    if not config.has_api_key:
        return "missing api key -> StubChatModel"
    if not openai_sdk_available():
        return "missing openai SDK -> StubChatModel"
    return "ready"


@dataclass(frozen=True)
class LLMResponse:
    """一次 chat 的结构化结果。frozen：调用方拿到即定，不被下游悄改。

    raw 保留 provider 原始返回（dict），便于排查；stub 无原始返回记 None。
    usage 记 token 估算 / 耗时等观测量（F14 trace 用），可为 None。
    """

    text: str
    provider: str
    model: str
    raw: dict | None = None
    usage: dict | None = None


class ChatModel(ABC):
    """LLM 调用的唯一抽象面。业务层只依赖它，不依赖任何 provider SDK。

    messages 是 OpenAI 风格 `[{"role": "system"/"user"/"assistant", "content": "..."}]`。
    """

    provider: str = "abstract"

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """发起一次对话补全。`max_output_tokens` 是**对外统一名**，adapter 内部各自映射。"""


    def quick_chat(
        self,
        messages: list[dict],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> LLMResponse:
        return self.chat(
            messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    def stream_chat(
        self,
        messages: list[dict],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        text = self.chat(
            messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        ).text
        if text:
            yield text


class StubChatModel(ChatModel):
    """无网络、确定性的 stub —— 无 key 降级路径 + golden test 基线。

    同输入必得同输出（无随机、无时间戳进正文）。检测到提示要求 JSON 时，返回**结构合法**
    的占位 JSON 字符串（默认模板可经 `json_template` 覆盖），F05/F06 据此在零 key 下解析跑通。
    """

    provider = "stub"

    def __init__(self, *, model: str = "stub", json_template: dict | None = None) -> None:
        self._model = model
        # 允许注入「期望 JSON schema」：F05/F06 想要特定键时传入，否则用默认模板。
        self._json_template = json_template

    def chat(
        self,
        messages: list[dict],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        text = self._synthesize(messages)
        return LLMResponse(text=text, provider="stub", model=self._model, raw=None, usage=None)

    def _synthesize(self, messages: list[dict]) -> str:
        last_user = _last_content(messages, "user")
        if _wants_json(messages):
            template = self._json_template if self._json_template is not None else _DEFAULT_JSON_TEMPLATE
            payload = dict(template)
            # 把最后一条 user 摘要塞进 line（若模板留了该键），让占位输出可辨识来源。
            if "line" in payload and not payload["line"]:
                payload["line"] = _gist(last_user)
            # sort_keys 保证同输入同输出（dict 插入序在跨版本下不该被依赖）。
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        # 非 JSON：确定性回显最后一条 user 摘要，前缀标明这是 stub（避免被误当真实文案）。
        return f"[stub:{self._model}] {_gist(last_user)}"


class _LLMRequestLimiter:
    """Small process-local gate for real provider calls.

    The external provider used in demos can reject bursts with a 10 requests/minute
    limit. This limiter spaces request starts and caps concurrent in-flight calls,
    so independent subsystems do not stampede the same key.
    """

    def __init__(
        self,
        *,
        requests_per_minute: int,
        max_concurrency: int,
        monotonic=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self._min_interval = 60.0 / max(1, requests_per_minute)
        self._gate = threading.Semaphore(max(1, max_concurrency))
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0
        self._monotonic = monotonic
        self._sleep = sleep

    @contextmanager
    def slot(self, *, provider: str, model: str):
        queued_at = time.perf_counter()
        self._gate.acquire()
        wait_ms = 0
        try:
            with self._lock:
                now = self._monotonic()
                delay = max(0.0, self._next_allowed_at - now)
                if delay > 0:
                    self._sleep(delay)
                started_at = self._monotonic()
                self._next_allowed_at = started_at + self._min_interval
                wait_ms = int((time.perf_counter() - queued_at) * 1000)
            if wait_ms > 0:
                logger.info(
                    "llm queue released provider=%s model=%s wait_ms=%d",
                    provider,
                    model,
                    wait_ms,
                )
            yield
        finally:
            self._gate.release()


def _real_provider_slot(config: LLMConfig, *, provider: str, injected_client: bool):
    if injected_client or not _queue_enabled():
        return nullcontext()
    settings = (_queue_rpm(), _queue_concurrency())
    with _QUEUE_LOCK:
        limiter = _QUEUE_CACHE.get(settings)
        if limiter is None:
            limiter = _LLMRequestLimiter(
                requests_per_minute=settings[0],
                max_concurrency=settings[1],
            )
            _QUEUE_CACHE[settings] = limiter
    return limiter.slot(provider=provider, model=config.model)


def _queue_enabled() -> bool:
    raw = (os.getenv("CYBERHOME_LLM_QUEUE") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _queue_rpm() -> int:
    raw = os.getenv("CYBERHOME_LLM_REQUESTS_PER_MINUTE")
    try:
        return max(1, int(float(raw))) if raw else _DEFAULT_QUEUE_RPM
    except (TypeError, ValueError):
        return _DEFAULT_QUEUE_RPM


def _queue_concurrency() -> int:
    raw = os.getenv("CYBERHOME_LLM_MAX_CONCURRENCY")
    try:
        return max(1, int(float(raw))) if raw else _DEFAULT_QUEUE_CONCURRENCY
    except (TypeError, ValueError):
        return _DEFAULT_QUEUE_CONCURRENCY


class _OpenAICompatibleChatModel(ChatModel):
    """DeepSeek / MiMo 共用的 OpenAI-compatible 基类。差异下放到两个可覆盖点。

    client 可注入（构造传 `client=...`）——测试据此捕获请求参数、验证映射，**不联网**。
    """

    # 子类覆盖：token 参数名。DeepSeek=max_tokens；MiMo=max_completion_tokens（决策 36）。
    _max_tokens_param: str = "max_tokens"

    def __init__(self, config: LLMConfig, *, client=None) -> None:
        self._config = config
        self._client = client  # 注入优先；为 None 时首次调用惰性建连
        self._client_injected = client is not None

    # ---- 两个可覆盖点：鉴权与 token 参数名 ----
    def _client_kwargs(self) -> dict:
        """构造 openai.OpenAI(**kwargs)。默认走 Bearer（DeepSeek）；MiMo 覆盖成 api-key header。

        timeout / max_retries 直接交给 SDK：SDK 自带超时与指数退避重试，无需自研（决策：不加架构税）。
        """
        return {
            "api_key": self._config.api_key,
            "base_url": self._config.base_url,
            "timeout": self._config.timeout,
            "max_retries": self._config.max_retries,
        }

    def _build_request(
        self,
        messages: list[dict],
        max_output_tokens: int | None,
        temperature: float | None,
    ) -> dict:
        """把统一入参映射成 provider 的请求 kwargs（**纯函数**，无副作用，便于测试映射正确性）。"""
        params: dict = {"model": self._config.model, "messages": messages}
        # 显式入参 > config 缺省；两者都无则不传该字段（用 provider 自身默认）。
        tokens = max_output_tokens if max_output_tokens is not None else self._config.max_output_tokens
        if tokens is not None:
            params[self._max_tokens_param] = tokens  # ← DeepSeek/MiMo 差异点
        temp = temperature if temperature is not None else self._config.temperature
        if temp is not None:
            params["temperature"] = temp
        return params

    def _ensure_client(self):
        if self._client is None:
            # 延迟 import：stub 路径永不到这，环境没装 openai 也不影响 stub 闭环。
            from openai import OpenAI

            self._client = OpenAI(**self._client_kwargs())
        return self._client

    def chat(
        self,
        messages: list[dict],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        return self._chat_with_options(
            messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    def quick_chat(
        self,
        messages: list[dict],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> LLMResponse:
        return self._chat_with_options(
            messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _chat_with_options(
        self,
        messages: list[dict],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> LLMResponse:
        params = self._build_request(messages, max_output_tokens, temperature)
        client = self._ensure_client()
        if timeout is not None or max_retries is not None:
            options = {}
            if timeout is not None:
                options["timeout"] = timeout
            if max_retries is not None:
                options["max_retries"] = max_retries
            client = client.with_options(**options)
        logger.debug("llm chat provider=%s model=%s %s", self.provider, self._config.model, _summarize(messages))
        started = time.perf_counter()
        try:
            with _real_provider_slot(
                self._config,
                provider=self.provider,
                injected_client=self._client_injected,
            ):
                completion = client.chat.completions.create(**params)
        except Exception as exc:  # 只记脱敏摘要后抛出——降级由 F06 reactor 层决定（本层不越界）
            logger.warning(
                "llm chat failed provider=%s model=%s err=%s: %s",
                self.provider, self._config.model, type(exc).__name__, exc,
            )
            raise
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return self._to_response(completion, elapsed_ms)

    def stream_chat(
        self,
        messages: list[dict],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        params = self._build_request(messages, max_output_tokens, temperature)
        params["stream"] = True
        client = self._ensure_client()
        logger.debug("llm stream provider=%s model=%s %s", self.provider, self._config.model, _summarize(messages))
        started = time.perf_counter()
        try:
            with _real_provider_slot(
                self._config,
                provider=self.provider,
                injected_client=self._client_injected,
            ):
                stream = client.chat.completions.create(**params)
                for chunk in stream:
                    text = _extract_stream_delta(chunk)
                    if text:
                        yield text
        except Exception as exc:
            logger.warning(
                "llm stream failed provider=%s model=%s err=%s: %s",
                self.provider, self._config.model, type(exc).__name__, exc,
            )
            raise
        finally:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.debug(
                "llm stream done provider=%s model=%s elapsed_ms=%d",
                self.provider, self._config.model, elapsed_ms,
            )

    def _to_response(self, completion, elapsed_ms: int) -> LLMResponse:
        text = ""
        choices = getattr(completion, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            text = (getattr(message, "content", None) or "") if message is not None else ""
        raw = completion.model_dump() if hasattr(completion, "model_dump") else None
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self._config.model,
            raw=raw,
            usage=_extract_usage(completion, elapsed_ms),
        )


class DeepSeekChatModel(_OpenAICompatibleChatModel):
    """DeepSeek：标准 Bearer 鉴权 + 标准 `max_tokens`（基类默认即是，无需覆盖）。"""

    provider = "deepseek"
    _max_tokens_param = "max_tokens"


class MiMoChatModel(_OpenAICompatibleChatModel):
    """MiMo（小米开放平台，base_url=api.xiaomimimo.com/v1）：唯一偏离 OpenAI 惯例处是
    `max_completion_tokens`（非 `max_tokens`）。鉴权走标准 Bearer——官方明确同时支持
    `api-key` header 与 `Authorization: Bearer`，取后者即可复用基类默认 `_client_kwargs`，
    无需 placeholder + default_headers 特例（决策 36 修订：少一个特例，低熵）。"""

    provider = "mimo"
    _max_tokens_param = "max_completion_tokens"


def build_chat_model(config: LLMConfig | None = None) -> ChatModel:
    """工厂：按 config.provider 返回实现；无 key / provider=stub 一律降级 StubChatModel。

    这里是「无 key 也能跑」红线的落点——config.provider 记意图，实际有没有 key 由这里判。
    """
    config = config or load_llm_config()
    if config.provider == "stub" or not config.has_api_key:
        return StubChatModel(model=config.model)
    if not openai_sdk_available():
        logger.warning(
            "real llm provider configured but openai SDK is missing provider=%s model=%s; "
            "install backend requirements; falling back to StubChatModel",
            config.provider, config.model,
        )
        return StubChatModel(model=config.model)
    if config.provider == "deepseek":
        return DeepSeekChatModel(config)
    if config.provider == "mimo":
        return MiMoChatModel(config)
    # config.provider 已被 LLMConfig 限定为枚举，理论不可达；兜底 stub 求稳。
    return StubChatModel(model=config.model)


def parse_json_response(text: str | None) -> dict | None:
    """把模型返回的文本尽力解析成 dict，失败返回 None（护栏：JSON 解析容错）。

    容忍三种脏：① 直接就是 JSON；② ```json ... ``` 代码块包裹；③ 前后夹带说明文字
    （取首个 `{` 到末个 `}` 的切片）。只认对象（dict），数组/标量不算结构化响应。
    """
    if not text or not text.strip():
        return None
    for candidate in _json_candidates(text):
        obj = _try_json(candidate)
        if isinstance(obj, dict):
            return obj
    return None


def redact_prompt(text: str, limit: int = _MAX_LOG_PROMPT) -> str:
    """把可能过长的 prompt 压成「前 N 字 + 总长」的日志安全摘要（脱敏辅助，供各层复用）。"""
    if text is None:
        return "<none>"
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…(+{len(text) - limit} chars)"


# ---------------------------------------------------------------------------
# 内部纯函数：无状态、可独立测试
# ---------------------------------------------------------------------------
def _last_content(messages: list[dict], role: str) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == role:
            return str(msg.get("content") or "")
    return ""


def _wants_json(messages: list[dict]) -> bool:
    """朴素判定：任一 system/user 提示含 'json'（大小写不敏感）即认为期望 JSON 输出。"""
    for msg in messages or []:
        content = str(msg.get("content") or "")
        if "json" in content.lower():
            return True
    return False


def _gist(text: str, limit: int = 80) -> str:
    """确定性摘要：strip 后截断到 limit 字，用于 stub 回显（同输入同输出）。"""
    text = (text or "").strip()
    return text[:limit]


def _summarize(messages: list[dict]) -> str:
    """日志用的脱敏摘要：只报角色序列、条数与总字数，绝不落 prompt 全文。"""
    roles = "/".join(str(m.get("role") or "?") for m in (messages or []))
    total = sum(len(str(m.get("content") or "")) for m in (messages or []))
    return f"[{roles}] {len(messages or [])} msgs ~{total} chars"


def _extract_usage(completion, elapsed_ms: int) -> dict:
    usage: dict = {"elapsed_ms": elapsed_ms}
    usage_obj = getattr(completion, "usage", None)
    if usage_obj is not None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage_obj, key, None)
            if value is not None:
                usage[key] = value
    # finish_reason 上抛给 reactor 层观测：推理模型（如 deepseek-v4-flash）token 预算过小时
    # 全被 reasoning_content 吃掉、content 为空且 finish_reason='length'——不暴露则上层只见空
    # 文本、无从判因（决策：观测截断，不在本层改写 text 语义）。
    choices = getattr(completion, "choices", None) or []
    if choices:
        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason is not None:
            usage["finish_reason"] = finish_reason
    return usage


def _extract_stream_delta(chunk) -> str:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""
    return getattr(delta, "content", None) or ""


def _json_candidates(text: str):
    """产出待解析候选串，从「最可能正确」到「最宽容」依序 yield。"""
    stripped = text.strip()
    yield stripped                       # 1. 原文直解
    unfenced = _strip_code_fence(stripped)
    if unfenced != stripped:
        yield unfenced                   # 2. 去掉 ```json 代码围栏
    # 3. 花括号切片：容忍前后夹带的说明文字
    start = unfenced.find("{")
    end = unfenced.rfind("}")
    if start != -1 and end > start:
        yield unfenced[start:end + 1]


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):   # 去首行 ``` 或 ```json
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):  # 去尾行 ```
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _try_json(candidate: str):
    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        return None
