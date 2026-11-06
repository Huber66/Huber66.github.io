"""LLM 配置装载 —— 从环境变量解析结构化配置（F04 / 决策 35、36）。

一件事在这里定死：**无 key 也能跑**（红线）。所以 provider 未显式指定时按
「有 DEEPSEEK_API_KEY→deepseek；否则有 MIMO_API_KEY→mimo；都没有→stub」自动降级，
让本地 / CI / 测试在零密钥下走确定性 stub 闭环，不碰网络。

两个刻意设计：
1. `load_llm_config(env=...)` 允许**注入 env 映射**——对齐本项目「随机源/时间显式注入」
   的可测试性风格（reactor.rng、engine.now），测试不必污染 os.environ。
2. `api_key` 用 `field(repr=False)`：从**类型层**堵死密钥进日志（脱敏红线）。frozen
   dataclass 的默认 repr 会打印所有字段，若不排除，一次 `print(config)` 就泄漏 key。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Mapping

LLMProvider = Literal["deepseek", "mimo", "stub"]

# 各 provider 的接入默认值（决策 36：base_url / 默认模型 / key 环境变量）。
# stub 无网络、无 key，base_url=None。
_PROVIDER_DEFAULTS: dict[str, dict] = {
    "deepseek": {
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "mimo": {
        "model": "mimo-v2.5-pro",
        "base_url": "https://api.xiaomimimo.com/v1",
        "api_key_env": "MIMO_API_KEY",
    },
    "stub": {
        "model": "stub",
        "base_url": None,
        "api_key_env": None,
    },
}

_DEFAULT_TIMEOUT = 30.0    # 秒；LLM 首字节可能较慢，给足余量又不至于卡死请求
_DEFAULT_MAX_RETRIES = 1   # 有限重试（护栏红线：最多 1-2 次），默认 1 次补偿瞬时抖动


@dataclass(frozen=True)
class LLMConfig:
    """一次 LLM 接入的完整配置。frozen：装载后不可变，避免运行期被悄悄改写。

    provider 记录**用户意图**（显式指定则尊重，未指定则自动选择的结果）；
    是否真的降级到 stub 由工厂 `build_chat_model` 依据 `api_key` 是否就位决定——
    config 只读意图，工厂管现实降级，职责分明。
    """

    provider: LLMProvider
    model: str
    base_url: str | None
    timeout: float
    max_retries: int
    max_output_tokens: int | None = None
    temperature: float | None = None
    # 密钥绝不进 repr（脱敏红线）；stub 路径为 None。
    api_key: str | None = field(default=None, repr=False)

    @property
    def has_api_key(self) -> bool:
        """真实 provider 是否拿到了可用密钥。工厂据此决定是否降级 stub。"""
        return bool(self.api_key)


def load_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    """从环境变量装载 LLMConfig。`env` 可注入（测试友好），默认取 os.environ。

    provider 选择：显式 `CYBERHOME_LLM_PROVIDER` 优先且被尊重；缺省或非法值时，
    按密钥可用性自动降级 deepseek→mimo→stub（无 key 也能跑的地基）。
    """
    src = os.environ if env is None else env

    deepseek_key = _clean(src.get("DEEPSEEK_API_KEY"))
    mimo_key = _clean(src.get("MIMO_API_KEY"))

    requested = (_clean(src.get("CYBERHOME_LLM_PROVIDER")) or "").lower()
    if requested in _PROVIDER_DEFAULTS:
        provider: LLMProvider = requested  # type: ignore[assignment]
    else:
        # 未指定或非法值 → 按密钥自动选择（非法值不报错，宽容降级更稳）
        provider = _auto_provider(deepseek_key, mimo_key)

    defaults = _PROVIDER_DEFAULTS[provider]
    api_key = {"deepseek": deepseek_key, "mimo": mimo_key}.get(provider)

    model = _clean(src.get("CYBERHOME_LLM_MODEL")) or defaults["model"]
    base_url = _clean(src.get("CYBERHOME_LLM_BASE_URL")) or defaults["base_url"]

    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        timeout=_env_float(src, "CYBERHOME_LLM_TIMEOUT", _DEFAULT_TIMEOUT),
        max_retries=max(0, _env_int(src, "CYBERHOME_LLM_MAX_RETRIES", _DEFAULT_MAX_RETRIES)),
        max_output_tokens=_env_int(src, "CYBERHOME_LLM_MAX_TOKENS", None),
        temperature=_env_float(src, "CYBERHOME_LLM_TEMPERATURE", None),
        api_key=api_key,
    )


def _auto_provider(deepseek_key: str | None, mimo_key: str | None) -> LLMProvider:
    """无显式 provider 时的降级链：deepseek → mimo → stub（有哪个 key 用哪个）。"""
    if deepseek_key:
        return "deepseek"
    if mimo_key:
        return "mimo"
    return "stub"


def _clean(value: str | None) -> str | None:
    """去空白后为空串则视为未设置——把 '' 和 '   ' 一律当作 None，避免空 key 蒙混。"""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _env_float(src: Mapping[str, str], key: str, default: float | None) -> float | None:
    raw = _clean(src.get(key))
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default  # 非法数值宽容回退，不因一个环境变量拼错而崩整个后端


def _env_int(src: Mapping[str, str], key: str, default: int | None) -> int | None:
    raw = _clean(src.get(key))
    if raw is None:
        return default
    try:
        return int(float(raw))  # 容忍 '512.0' 这类写法
    except (TypeError, ValueError):
        return default
