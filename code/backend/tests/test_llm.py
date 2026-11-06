"""LLM Provider 抽象层 golden tests（F04 · 5.4 验收）。

红线：**不联网、不依赖真实 key**。真实 provider 调用用 skipUnless 跳过。
验证点：stub 确定性、工厂无 key 降级、config 自动选择、DeepSeek/MiMo 参数映射与
鉴权 header 组装（用注入的 fake client 捕获，不真连）、parse_json_response 容错、脱敏。

标准库 unittest，运行：backend/ 下 python3 -m unittest discover tests
"""

import os
import unittest
from unittest.mock import patch

from cyberhome.config import LLMConfig, load_llm_config
from cyberhome.llm import (
    ChatModel, DeepSeekChatModel, LLMResponse, MiMoChatModel, StubChatModel,
    _LLMRequestLimiter, build_chat_model, llm_runtime_status, parse_json_response, redact_prompt,
)


# ---------------------------------------------------------------------------
# 测试替身：模仿 openai client.chat.completions.create，捕获请求参数、不联网
# ---------------------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeStreamChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeStreamChunk:
    def __init__(self, content):
        self.choices = [_FakeStreamChoice(content)]


class _FakeUsage:
    def __init__(self):
        self.prompt_tokens = 11
        self.completion_tokens = 7
        self.total_tokens = 18


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()

    def model_dump(self):
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


class _FakeCompletions:
    def __init__(self, parent):
        self._parent = parent

    def create(self, **kwargs):
        self._parent.captured = kwargs  # ← 捕获请求参数，供断言映射正确性
        return _FakeCompletion("回声：收到")


class _FakeChat:
    def __init__(self, parent):
        self.completions = _FakeCompletions(parent)


class _FakeOpenAIClient:
    """注入 adapter 的假 client：不建真连，只记录调用参数。"""

    def __init__(self):
        self.captured = None
        self.chat = _FakeChat(self)


# ---------------------------------------------------------------------------
# config：自动选择与降级、脱敏
# ---------------------------------------------------------------------------
class TestConfig(unittest.TestCase):
    def test_defaults_to_stub_without_keys(self):
        """无任何 key、无显式 provider → stub（无 key 也能跑的地基）。"""
        cfg = load_llm_config(env={})
        self.assertEqual(cfg.provider, "stub")
        self.assertFalse(cfg.has_api_key)

    def test_auto_selects_deepseek_when_key_present(self):
        cfg = load_llm_config(env={"DEEPSEEK_API_KEY": "sk-deep"})
        self.assertEqual(cfg.provider, "deepseek")
        self.assertEqual(cfg.model, "deepseek-v4-flash")
        self.assertEqual(cfg.base_url, "https://api.deepseek.com")
        self.assertTrue(cfg.has_api_key)

    def test_auto_prefers_deepseek_over_mimo(self):
        """两个 key 都在、无显式 provider → 按降级链优先 deepseek。"""
        cfg = load_llm_config(env={"DEEPSEEK_API_KEY": "sk-deep", "MIMO_API_KEY": "sk-mimo"})
        self.assertEqual(cfg.provider, "deepseek")

    def test_auto_falls_back_to_mimo(self):
        cfg = load_llm_config(env={"MIMO_API_KEY": "sk-mimo"})
        self.assertEqual(cfg.provider, "mimo")
        self.assertEqual(cfg.model, "mimo-v2.5-pro")
        self.assertEqual(cfg.base_url, "https://api.xiaomimimo.com/v1")

    def test_explicit_provider_is_respected(self):
        """显式 provider 优先于自动选择——即便有 deepseek key 也尊重 mimo 意图。"""
        cfg = load_llm_config(env={"CYBERHOME_LLM_PROVIDER": "mimo", "DEEPSEEK_API_KEY": "sk-deep"})
        self.assertEqual(cfg.provider, "mimo")

    def test_blank_key_treated_as_absent(self):
        """空串 / 纯空白 key 视为未设置，不蒙混成有 key。"""
        cfg = load_llm_config(env={"DEEPSEEK_API_KEY": "   "})
        self.assertEqual(cfg.provider, "stub")

    def test_overrides_applied(self):
        cfg = load_llm_config(env={
            "DEEPSEEK_API_KEY": "sk-deep",
            "CYBERHOME_LLM_MODEL": "deepseek-v4-pro",
            "CYBERHOME_LLM_BASE_URL": "https://proxy.internal",
            "CYBERHOME_LLM_TIMEOUT": "12.5",
            "CYBERHOME_LLM_MAX_TOKENS": "512",
            "CYBERHOME_LLM_TEMPERATURE": "0.3",
        })
        self.assertEqual(cfg.model, "deepseek-v4-pro")
        self.assertEqual(cfg.base_url, "https://proxy.internal")
        self.assertEqual(cfg.timeout, 12.5)
        self.assertEqual(cfg.max_output_tokens, 512)
        self.assertEqual(cfg.temperature, 0.3)

    def test_invalid_numeric_override_falls_back(self):
        """环境变量数值拼错不该崩后端，宽容回退默认。"""
        cfg = load_llm_config(env={"CYBERHOME_LLM_TIMEOUT": "not-a-number"})
        self.assertEqual(cfg.timeout, 30.0)

    def test_api_key_not_in_repr(self):
        """脱敏红线：api_key 绝不出现在 repr（一次 print(config) 就泄漏是不可接受的）。"""
        cfg = LLMConfig(
            provider="deepseek", model="m", base_url="u",
            timeout=30.0, max_retries=1, api_key="sk-super-secret",
        )
        self.assertNotIn("sk-super-secret", repr(cfg))


# ---------------------------------------------------------------------------
# StubChatModel：确定性、JSON 降级
# ---------------------------------------------------------------------------
class TestStub(unittest.TestCase):
    def test_stub_output_is_deterministic(self):
        """同输入必得同输出（golden test 基线）。"""
        stub = StubChatModel()
        msgs = [{"role": "user", "content": "老爸今天心情怎么样"}]
        r1 = stub.chat(msgs)
        r2 = stub.chat(msgs)
        self.assertEqual(r1.text, r2.text)
        self.assertEqual(r1.provider, "stub")
        self.assertIsNone(r1.raw)

    def test_stub_echoes_last_user_gist(self):
        stub = StubChatModel()
        r = stub.chat([
            {"role": "system", "content": "你是老爸"},
            {"role": "user", "content": "下雨了"},
        ])
        self.assertIn("下雨了", r.text)

    def test_stub_returns_valid_json_when_prompted(self):
        """提示要求 JSON → 返回结构合法可解析的占位 JSON（F05/F06 无 key 可跑通）。"""
        stub = StubChatModel()
        r = stub.chat([
            {"role": "system", "content": "只输出 JSON。"},
            {"role": "user", "content": "老爸要不要开口"},
        ])
        parsed = parse_json_response(r.text)
        self.assertIsInstance(parsed, dict)
        self.assertIn("speak", parsed)

    def test_stub_json_is_deterministic(self):
        stub = StubChatModel()
        msgs = [{"role": "user", "content": "请用 json 回答"}]
        self.assertEqual(stub.chat(msgs).text, stub.chat(msgs).text)

    def test_stub_custom_json_template(self):
        """F05/F06 可注入期望 schema，占位输出即按该 schema 结构合法。"""
        stub = StubChatModel(json_template={"decision": "yes", "score": 1})
        r = stub.chat([{"role": "user", "content": "output json"}])
        parsed = parse_json_response(r.text)
        self.assertEqual(parsed, {"decision": "yes", "score": 1})


# ---------------------------------------------------------------------------
# 工厂：无 key 降级 stub，有 key 返回对应 adapter
# ---------------------------------------------------------------------------
class TestFactory(unittest.TestCase):
    def test_no_key_returns_stub(self):
        """无 key 红线：工厂必返回 StubChatModel。"""
        model = build_chat_model(load_llm_config(env={}))
        self.assertIsInstance(model, StubChatModel)

    def test_explicit_stub_returns_stub(self):
        model = build_chat_model(load_llm_config(env={"CYBERHOME_LLM_PROVIDER": "stub"}))
        self.assertIsInstance(model, StubChatModel)

    def test_provider_requested_but_no_key_degrades_to_stub(self):
        """显式要 deepseek 但没给 key → 仍降级 stub（意图 vs 现实，工厂管现实）。"""
        cfg = load_llm_config(env={"CYBERHOME_LLM_PROVIDER": "deepseek"})
        self.assertEqual(cfg.provider, "deepseek")  # 意图记录
        self.assertIsInstance(build_chat_model(cfg), StubChatModel)  # 现实降级

    def test_deepseek_key_returns_deepseek_adapter(self):
        model = build_chat_model(load_llm_config(env={"DEEPSEEK_API_KEY": "sk-deep"}))
        self.assertIsInstance(model, DeepSeekChatModel)

    def test_real_provider_missing_openai_sdk_degrades_to_stub_with_status(self):
        cfg = load_llm_config(env={"DEEPSEEK_API_KEY": "sk-deep"})
        with patch("cyberhome.llm.importlib.util.find_spec", return_value=None):
            self.assertIsInstance(build_chat_model(cfg), StubChatModel)
            self.assertIn("missing openai SDK", llm_runtime_status(cfg))

    def test_mimo_key_returns_mimo_adapter(self):
        model = build_chat_model(load_llm_config(env={"MIMO_API_KEY": "sk-mimo"}))
        self.assertIsInstance(model, MiMoChatModel)

    def test_all_models_are_chatmodel(self):
        for env in ({}, {"DEEPSEEK_API_KEY": "k"}, {"MIMO_API_KEY": "k"}):
            self.assertIsInstance(build_chat_model(load_llm_config(env=env)), ChatModel)


class TestLLMRequestLimiter(unittest.TestCase):
    def test_spaces_consecutive_provider_calls(self):
        now = [100.0]
        sleeps = []

        def monotonic():
            return now[0]

        def sleep(delay):
            sleeps.append(delay)
            now[0] += delay

        limiter = _LLMRequestLimiter(
            requests_per_minute=60,
            max_concurrency=1,
            monotonic=monotonic,
            sleep=sleep,
        )

        with limiter.slot(provider="test", model="m"):
            pass
        with limiter.slot(provider="test", model="m"):
            pass

        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 1.0)


# ---------------------------------------------------------------------------
# DeepSeek adapter：参数映射（max_tokens）、Bearer 鉴权、响应解析。用 fake client。
# ---------------------------------------------------------------------------
class TestDeepSeekAdapter(unittest.TestCase):
    def _adapter(self, fake, **over):
        env = {"DEEPSEEK_API_KEY": "sk-deep"}
        env.update(over)
        return DeepSeekChatModel(load_llm_config(env=env), client=fake)

    def test_uses_max_tokens_param(self):
        """DeepSeek 用标准 max_tokens（非 max_completion_tokens）。"""
        fake = _FakeOpenAIClient()
        self._adapter(fake).chat([{"role": "user", "content": "hi"}], max_output_tokens=256)
        self.assertEqual(fake.captured["max_tokens"], 256)
        self.assertNotIn("max_completion_tokens", fake.captured)

    def test_passes_model_and_messages_and_temperature(self):
        fake = _FakeOpenAIClient()
        msgs = [{"role": "user", "content": "hi"}]
        self._adapter(fake).chat(msgs, temperature=0.5)
        self.assertEqual(fake.captured["model"], "deepseek-v4-flash")
        self.assertEqual(fake.captured["messages"], msgs)
        self.assertEqual(fake.captured["temperature"], 0.5)

    def test_omits_token_and_temp_when_unset(self):
        """未指定 max token / temperature 时不塞该字段，交给 provider 默认。"""
        fake = _FakeOpenAIClient()
        self._adapter(fake).chat([{"role": "user", "content": "hi"}])
        self.assertNotIn("max_tokens", fake.captured)
        self.assertNotIn("temperature", fake.captured)

    def test_response_parsed(self):
        fake = _FakeOpenAIClient()
        resp = self._adapter(fake).chat([{"role": "user", "content": "hi"}])
        self.assertIsInstance(resp, LLMResponse)
        self.assertEqual(resp.text, "回声：收到")
        self.assertEqual(resp.provider, "deepseek")
        self.assertEqual(resp.usage["total_tokens"], 18)
        self.assertIn("elapsed_ms", resp.usage)
        # finish_reason 上抛供 reactor 观测截断（推理模型 token 不足会 'length'）。
        self.assertEqual(resp.usage["finish_reason"], "stop")

    def test_stream_chat_uses_stream_param_and_yields_deltas(self):
        class _StreamCompletions:
            def __init__(self, parent):
                self._parent = parent

            def create(self, **kwargs):
                self._parent.captured = kwargs
                return [_FakeStreamChunk("回"), _FakeStreamChunk("声")]

        class _StreamChat:
            def __init__(self, parent):
                self.completions = _StreamCompletions(parent)

        class _StreamClient:
            def __init__(self):
                self.captured = None
                self.chat = _StreamChat(self)

        fake = _StreamClient()
        chunks = list(self._adapter(fake).stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks, ["回", "声"])
        self.assertTrue(fake.captured["stream"])
        self.assertEqual(fake.captured["model"], "deepseek-v4-flash")

    def test_client_kwargs_use_bearer_style(self):
        """DeepSeek 走 openai SDK 默认 Bearer：api_key 直传，无自定义 header。"""
        adapter = DeepSeekChatModel(load_llm_config(env={"DEEPSEEK_API_KEY": "sk-deep"}))
        kwargs = adapter._client_kwargs()
        self.assertEqual(kwargs["api_key"], "sk-deep")
        self.assertEqual(kwargs["base_url"], "https://api.deepseek.com")
        self.assertNotIn("default_headers", kwargs)


# ---------------------------------------------------------------------------
# MiMo adapter：max_output_tokens→max_completion_tokens 映射 + 标准 Bearer 鉴权
# ---------------------------------------------------------------------------
class TestMiMoAdapter(unittest.TestCase):
    def _adapter(self, fake=None):
        return MiMoChatModel(load_llm_config(env={"MIMO_API_KEY": "sk-mimo"}), client=fake)

    def test_maps_max_output_tokens_to_max_completion_tokens(self):
        """核心差异点：对外 max_output_tokens → MiMo 请求 max_completion_tokens。"""
        fake = _FakeOpenAIClient()
        self._adapter(fake).chat([{"role": "user", "content": "hi"}], max_output_tokens=128)
        self.assertEqual(fake.captured["max_completion_tokens"], 128)
        self.assertNotIn("max_tokens", fake.captured)

    def test_client_kwargs_use_bearer_style(self):
        """MiMo 走标准 Bearer（官方同时支持 api-key header 与 Bearer，取后者复用基类）：
        api_key 直传、无 default_headers 特例，base_url 指向小米官方端点。"""
        adapter = self._adapter()
        kwargs = adapter._client_kwargs()
        self.assertEqual(kwargs["api_key"], "sk-mimo")           # 真实 key 直传，非占位
        self.assertEqual(kwargs["base_url"], "https://api.xiaomimimo.com/v1")
        self.assertNotIn("default_headers", kwargs)              # 无 api-key header 特例

    def test_config_max_tokens_used_when_call_arg_absent(self):
        """未在调用点传 token 时回退 config.max_output_tokens，且仍映射为 MiMo 参数名。"""
        fake = _FakeOpenAIClient()
        adapter = MiMoChatModel(
            load_llm_config(env={"MIMO_API_KEY": "sk-mimo", "CYBERHOME_LLM_MAX_TOKENS": "64"}),
            client=fake,
        )
        adapter.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(fake.captured["max_completion_tokens"], 64)

    def test_response_provider_is_mimo(self):
        fake = _FakeOpenAIClient()
        self.assertEqual(self._adapter(fake).chat([{"role": "user", "content": "hi"}]).provider, "mimo")


# ---------------------------------------------------------------------------
# parse_json_response：容错代码块与脏前后缀
# ---------------------------------------------------------------------------
class TestParseJson(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_json_response('{"a": 1}'), {"a": 1})

    def test_json_code_fence(self):
        text = '```json\n{"speak": true, "line": "你好"}\n```'
        self.assertEqual(parse_json_response(text), {"speak": True, "line": "你好"})

    def test_bare_code_fence(self):
        self.assertEqual(parse_json_response('```\n{"a": 1}\n```'), {"a": 1})

    def test_dirty_prefix_suffix(self):
        """模型爱在 JSON 前后夹话——取首 { 到末 } 切片救回。"""
        text = '好的，这是结果：{"a": 1, "b": [2, 3]} 希望有用！'
        self.assertEqual(parse_json_response(text), {"a": 1, "b": [2, 3]})

    def test_returns_none_on_garbage(self):
        self.assertIsNone(parse_json_response("这里根本没有 json"))

    def test_returns_none_on_empty(self):
        self.assertIsNone(parse_json_response(""))
        self.assertIsNone(parse_json_response(None))

    def test_array_is_not_accepted(self):
        """只认对象；数组不算结构化响应（F05/F06 期望 dict）。"""
        self.assertIsNone(parse_json_response("[1, 2, 3]"))


# ---------------------------------------------------------------------------
# 脱敏辅助
# ---------------------------------------------------------------------------
class TestRedaction(unittest.TestCase):
    def test_short_prompt_unchanged(self):
        self.assertEqual(redact_prompt("短提示"), "短提示")

    def test_long_prompt_truncated_with_length(self):
        long = "x" * 500
        out = redact_prompt(long, limit=100)
        self.assertTrue(out.startswith("x" * 100))
        self.assertIn("+400 chars", out)
        self.assertLess(len(out), len(long))


# ---------------------------------------------------------------------------
# 真实 provider：仅在给了 key 时跑，默认跳过（不联网红线）
# ---------------------------------------------------------------------------
class TestRealProvider(unittest.TestCase):
    @unittest.skipUnless(os.getenv("DEEPSEEK_API_KEY"), "需 DEEPSEEK_API_KEY，无 key 跳过")
    def test_deepseek_live_chat(self):
        model = build_chat_model(load_llm_config())
        # deepseek-v4-flash 是推理模型：先写 reasoning_content 再写 content。token 上限太小
        # （如 16）会全被思考吃掉、content 为空且 finish_reason='length'。给足 256 留输出预算。
        resp = model.chat([{"role": "user", "content": "用一个字回答：你好"}], max_output_tokens=256)
        self.assertTrue(resp.text)
        self.assertEqual(resp.provider, "deepseek")

    @unittest.skipUnless(os.getenv("MIMO_API_KEY"), "需 MIMO_API_KEY，无 key 跳过")
    def test_mimo_live_chat(self):
        model = build_chat_model(load_llm_config(env={
            "CYBERHOME_LLM_PROVIDER": "mimo",
            "MIMO_API_KEY": os.getenv("MIMO_API_KEY", ""),
        }))
        resp = model.chat([{"role": "user", "content": "用一个字回答：你好"}], max_output_tokens=16)
        self.assertTrue(resp.text)
        self.assertEqual(resp.provider, "mimo")


if __name__ == "__main__":
    unittest.main()
