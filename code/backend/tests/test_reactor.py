"""LLMReactor / StubReactor 测试（F06 步骤 8.5 验收）。

重点证明：bidding capped、oracle 差异化反应、全体沉默返回 []、mood 被解析、
LLM 抛异常时 react 不抛且不损坏（降级到 fallback 或 []），以及无 key（StubChatModel）
闭环可跑通。零第三方依赖，标准库 unittest。运行：cd backend && python3 -m unittest discover tests
"""

import json
import random
import unittest

from cyberhome.llm import ChatModel, LLMResponse, StubChatModel
from cyberhome.memory import InMemoryStore, SQLiteMemoryStore
from cyberhome.models import Trigger
from cyberhome.reactor import AgentReaction, LLMReactor, StubReactor
from cyberhome.seed import make_initial_world


# ---------------------------------------------------------------------------
# 测试用假 ChatModel（不联网、确定性）
# ---------------------------------------------------------------------------
class FixedChatModel(ChatModel):
    """对所有 prompt 返回同一段固定 JSON。"""

    provider = "fake"

    def __init__(self, payload: dict) -> None:
        self._text = json.dumps(payload, ensure_ascii=False)

    def chat(self, messages, *, max_output_tokens=None, temperature=None) -> LLMResponse:
        return LLMResponse(text=self._text, provider="fake", model="fake")


class PerAgentChatModel(ChatModel):
    """按 system prompt 里的角色名（「名字」）返回不同 JSON，验证差异化反应。"""

    provider = "fake"

    def __init__(self, by_name: dict, default: dict) -> None:
        self._by_name = by_name
        self._default = default

    def chat(self, messages, *, max_output_tokens=None, temperature=None) -> LLMResponse:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        payload = self._default
        for name, p in self._by_name.items():
            if f"「{name}」" in system:
                payload = p
                break
        return LLMResponse(text=json.dumps(payload, ensure_ascii=False), provider="fake", model="fake")


class RaisingChatModel(ChatModel):
    """chat 一律抛异常，用于验证降级红线。"""

    provider = "fake"

    def chat(self, messages, *, max_output_tokens=None, temperature=None) -> LLMResponse:
        raise RuntimeError("simulated network error")


def _oracle() -> Trigger:
    return Trigger(kind="oracle", text="要下雨了", ts=2_000_000)


class TestLLMReactorOracle(unittest.TestCase):
    def setUp(self):
        self.world = make_initial_world(now=1_000_000)
        self.memory = InMemoryStore()

    def test_oracle_differentiated_reactions(self):
        """两个角色给出不同台词 + 各自 mood 被解析（差异化涌现）。"""
        model = PerAgentChatModel(
            by_name={
                "老爸": {"speak": True, "line": "哼，早说啊", "mood": "grumpy"},
                "老妈": {"speak": True, "line": "那我去炖个汤", "mood": "happy"},
            },
            default={"speak": False, "line": ""},
        )
        reactor = LLMReactor(model, self.memory)
        out = reactor.react(self.world, _oracle(), self.memory)
        by_id = {r.agent_id: r for r in out}
        self.assertIn("agent_dad", by_id)
        self.assertIn("agent_mom", by_id)
        self.assertNotEqual(by_id["agent_dad"].line, by_id["agent_mom"].line)
        self.assertEqual(by_id["agent_dad"].mood, "grumpy")
        self.assertEqual(by_id["agent_mom"].mood, "happy")

    def test_bidding_capped(self):
        """全体开口时 bidding 截断到 max_speakers。"""
        model = FixedChatModel({"speak": True, "line": "我也说一句"})
        reactor = LLMReactor(model, self.memory, max_speakers=2)
        out = reactor.react(self.world, _oracle(), self.memory)
        speakers = [r for r in out if r.speak and r.line]
        self.assertEqual(len(speakers), 2)  # 4 角色全开口 → capped 2

    def test_all_silent_returns_empty(self):
        """全体合法沉默（speak=false 且无状态变化）→ 返回 []。"""
        model = FixedChatModel({"speak": False, "line": ""})
        reactor = LLMReactor(model, self.memory)
        out = reactor.react(self.world, _oracle(), self.memory)
        self.assertEqual(out, [])

    def test_mood_parsed(self):
        model = FixedChatModel({"speak": True, "line": "记下了", "mood": "tired"})
        reactor = LLMReactor(model, self.memory, max_speakers=2)
        out = reactor.react(self.world, _oracle(), self.memory)
        self.assertTrue(out)
        self.assertTrue(all(r.mood == "tired" for r in out))


class TestLLMReactorDegrade(unittest.TestCase):
    """降级红线（决策 8.3）：react 永不抛异常给引擎。"""

    def setUp(self):
        self.world = make_initial_world(now=1_000_000)
        self.memory = InMemoryStore()

    def test_llm_exception_does_not_raise_and_delegates_fallback(self):
        """LLM 全崩 → 不抛异常，委托 fallback（StubReactor）产出兜底反应。"""
        reactor = LLMReactor(
            RaisingChatModel(), self.memory, fallback=StubReactor(rng=random.Random(42))
        )
        out = reactor.react(self.world, _oracle(), self.memory)  # 不应抛
        self.assertIsInstance(out, list)
        self.assertTrue(out)  # fallback 有开口者

    def test_partial_failure_skips_only_failing_agent(self):
        """单个 agent 失败只跳过它，其余正常反应（不牵连）。"""
        model = PerAgentChatModel(
            by_name={"老爸": {"speak": True, "line": "老爸说话"}},
            default={"speak": True, "line": "别人说话"},
        )
        # 让老爸这条走「解析失败」路径：返回非 JSON 由 default 覆盖不了，这里改用抛异常模型不便，
        # 故用一个只对老爸返回坏文本的模型验证「解析失败=跳过」。
        class DadBrokenModel(ChatModel):
            provider = "fake"

            def chat(self, messages, *, max_output_tokens=None, temperature=None):
                system = next((m["content"] for m in messages if m.get("role") == "system"), "")
                if "「老爸」" in system:
                    return LLMResponse(text="这不是 JSON", provider="fake", model="fake")
                return LLMResponse(
                    text=json.dumps({"speak": True, "line": "别人说话"}, ensure_ascii=False),
                    provider="fake", model="fake",
                )

        reactor = LLMReactor(DadBrokenModel(), self.memory, max_speakers=4)
        out = reactor.react(self.world, _oracle(), self.memory)
        ids = {r.agent_id for r in out}
        self.assertNotIn("agent_dad", ids)  # 老爸解析失败被跳过
        self.assertTrue(out)  # 其余角色仍有反应

class TestNoKeyClosedLoop(unittest.TestCase):
    """无 key 闭环：StubChatModel（build_chat_model 无 key 时的降级）能被解析跑通。"""

    def test_stub_chat_model_oracle_closed_loop(self):
        world = make_initial_world(now=1_000_000)
        memory = InMemoryStore()
        reactor = LLMReactor(StubChatModel(), memory, max_speakers=2)
        out = reactor.react(world, _oracle(), memory)
        self.assertEqual(len(out), 2)  # StubChatModel 默认 speak=true → capped 2
        self.assertTrue(all(r.speak and r.line for r in out))

    def test_stub_chat_model_move_closed_loop(self):
        world = make_initial_world(now=1_000_000)
        memory = InMemoryStore()
        reactor = LLMReactor(StubChatModel(), memory)
        trig = Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=2_000_000)
        out = reactor.react(world, trig, memory)
        # 移动者应在候选首位并开口（StubChatModel 默认 speak=true）
        self.assertTrue(out)
        self.assertEqual(out[0].agent_id, "agent_dad")


class TestLLMReactorMemoryWriteback(unittest.TestCase):
    """F09 记忆写入闭环：LLMReactor 产出非空 memory_note 后，记忆库能检索到。

    用 FixedChatModel 造确定性 memory_note（不联网），证明「写入端把记忆喂进去」这一环
    真正闭合——读端 assemble_prompt 早已在读 memory.search（prompting.py），此处只补写端。
    """

    def setUp(self):
        self.world = make_initial_world(now=1_000_000)

    def _model_with_note(self, note: str) -> FixedChatModel:
        # 全员沉默但都带同一条 memory_note：证明写入以「产出 note」为准，与开口/bidding 正交。
        return FixedChatModel({"speak": False, "line": "", "memory_note": note})

    def test_memory_note_written_and_retrievable(self):
        """核心闭环：产出 memory_note → 写入 agent:{id} 经历层 → search 命中。"""
        memory = InMemoryStore()
        reactor = LLMReactor(self._model_with_note("我记住了要下雨，得收衣服"), memory)
        reactor.react(self.world, _oracle(), memory)  # trigger.ts=2_000_000（见 _oracle）
        got = memory.search("agent:agent_dad", "下雨", top_k=5, token_cap=100_000)
        self.assertTrue(any("下雨" in m.text for m in got))
        # 可复现红线：写入 ts 取自 trigger.ts，不是 wall clock。
        self.assertEqual(got[0].ts, 2_000_000)

    def test_memory_note_closed_loop_with_sqlite_backend(self):
        """写入端对 SQLite 后端同样贯通（换后端零漂移的写侧证据）。"""
        memory = SQLiteMemoryStore(db_path=":memory:")
        reactor = LLMReactor(self._model_with_note("香菜这事我记下了"), memory)
        reactor.react(self.world, _oracle(), memory)
        got = memory.search("agent:agent_mom", "香菜", top_k=5, token_cap=100_000)
        self.assertTrue(any("香菜" in m.text for m in got))
        memory.close()

    def test_relation_note_written_for_copresent_peer(self):
        """关系层闭环：oracle 下同处一室的对端也各得一份 note（rel:{id}:{other}）。

        种子布局老爸与老空调同在客厅——老爸记住的事应同时落进 rel:agent_dad:agent_ac，
        日后两者再同室时可被 prompting._retrieve_relations 按此 key 检索到（读写路径对称）。
        这是 spec F09「写入经历与关系摘要」里「关系」那一半的可执行证据。
        """
        memory = InMemoryStore()
        reactor = LLMReactor(self._model_with_note("老空调又在客厅嫌吵，我记下了"), memory)
        reactor.react(self.world, _oracle(), memory)  # trigger.ts=2_000_000
        # 经历层照旧命中（两层都写）
        exp = memory.search("agent:agent_dad", "老空调", top_k=5, token_cap=100_000)
        self.assertTrue(any("老空调" in m.text for m in exp))
        # 关系层：对同室对端老空调也写了一份，且 ts 取自 trigger（可复现）
        rel = memory.search("rel:agent_dad:agent_ac", "老空调", top_k=5, token_cap=100_000)
        self.assertTrue(any("老空调" in m.text for m in rel))
        self.assertEqual(rel[0].ts, 2_000_000)

    def test_relation_scoping_excludes_non_copresent(self):
        """关系层 scoping：note 只写给同室对端，不写给不在场者。

        老爸在客厅、老妈在厨房，本轮并未同处一室，故不该产生 rel:agent_dad:agent_mom——
        否则日后会把「从未共处时记的事」错误地当成两人的关系记忆检索出来。
        """
        memory = InMemoryStore()
        reactor = LLMReactor(self._model_with_note("这条只该进客厅侧的关系层"), memory)
        reactor.react(self.world, _oracle(), memory)
        cross = memory.search("rel:agent_dad:agent_mom", "关系层", top_k=5, token_cap=100_000)
        self.assertEqual(cross, [])

    def test_no_note_writes_nothing(self):
        """不硬塞：反应无 memory_note 时，记忆库保持为空。"""
        memory = InMemoryStore()
        reactor = LLMReactor(FixedChatModel({"speak": True, "line": "说句话但不记"}), memory)
        reactor.react(self.world, _oracle(), memory)
        got = memory.search("agent:agent_dad", "说句话", top_k=5, token_cap=100_000)
        self.assertEqual(got, [])

    def test_none_ts_trigger_does_not_write(self):
        """可复现红线：trigger.ts 为 None 时不落库（不取 wall clock，宁可漏记不污染重放）。"""
        memory = InMemoryStore()
        reactor = LLMReactor(self._model_with_note("这条不该被写入"), memory)
        no_ts = Trigger(kind="oracle", text="要下雨了")  # ts 缺省 None
        reactor.react(self.world, no_ts, memory)
        got = memory.search("agent:agent_dad", "写入", top_k=5, token_cap=100_000)
        self.assertEqual(got, [])

    def test_memory_write_failure_does_not_raise(self):
        """降级红线：add 抛异常也不传导给引擎——react 正常返回，反应不受损。"""

        class BoomStore(InMemoryStore):
            def add(self, namespace, text, ts, importance=0.5):
                raise RuntimeError("simulated memory backend down")

        memory = BoomStore()
        reactor = LLMReactor(
            FixedChatModel({"speak": True, "line": "照常开口", "memory_note": "写入会炸"}), memory
        )
        out = reactor.react(self.world, _oracle(), memory)  # 不应抛
        self.assertTrue(out)
        self.assertTrue(all(r.speak and r.line for r in out))


class TestStubReactor(unittest.TestCase):
    """StubReactor 新 react 接口：可复现 + 台词无「名字：」前缀。"""

    def test_oracle_lines_have_no_name_prefix(self):
        world = make_initial_world(now=1_000_000)
        name_by_id = {a.id: a.name for a in world.agents}
        reactor = StubReactor(rng=random.Random(42))
        out = reactor.react(world, _oracle(), InMemoryStore())
        self.assertTrue(out)
        for r in out:
            self.assertTrue(r.speak)
            # 纯台词：不以「<名字>：」开头（归属由 engine 的 event.agentId 承载）
            self.assertFalse(r.line.startswith(f"{name_by_id[r.agent_id]}："))

    def test_reproducible(self):
        world = make_initial_world(now=1_000_000)
        a = StubReactor(rng=random.Random(42)).react(world, _oracle(), InMemoryStore())
        b = StubReactor(rng=random.Random(42)).react(world, _oracle(), InMemoryStore())
        self.assertEqual([r.line for r in a], [r.line for r in b])


if __name__ == "__main__":
    unittest.main()
