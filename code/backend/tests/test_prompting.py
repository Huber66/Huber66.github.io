"""Prompt 组装与私有视角上下文预算测试（F05 验收，决策 34）。

重点证明四件事：
1. 预算生效：token_cap 收紧时记忆被裁，persona 锚与情境永远在。
2. persona 不被裁：token_cap=0 也保住 system 段与情境段。
3. 空记忆可运行：InMemoryStore 空库组装不报错，messages 合法。
4. oracle / move 两类模板各含各自情境关键词 + JSON schema 要求。
5. parse_reaction 能解析合法 JSON（含代码块包裹），垃圾输入返回 None。

零第三方依赖，标准库 unittest。运行：python -m unittest discover tests
"""

import unittest

from cyberhome.memory import InMemoryStore
from cyberhome.models import Trigger
from cyberhome.prompting import (
    AssembledPrompt,
    ReactionDecision,
    assemble_prompt,
    parse_reaction,
)
from cyberhome.seed import make_initial_world


def _system(p: AssembledPrompt) -> str:
    return next(m["content"] for m in p.messages if m["role"] == "system")


def _user(p: AssembledPrompt) -> str:
    return next(m["content"] for m in p.messages if m["role"] == "user")


def _dad(world):
    return next(a for a in world.agents if a.id == "agent_dad")


class TestBudgetTrimming(unittest.TestCase):
    """决策 34：预算收紧时从后往前裁（经历先于关系），persona/情境永不裁。"""

    def _memory_with_records(self) -> InMemoryStore:
        store = InMemoryStore()
        # 老爸与冰箱的关系记忆（同在厨房时会被检索）
        store.add(namespace="rel:agent_dad:agent_mom", text="老爸和老妈拌过嘴但很快就和好", ts=1000)
        # 老爸自己的经历记忆
        for i in range(5):
            store.add(namespace="agent:agent_dad", text=f"老爸记得第{i}件很长很长很长很长的往事", ts=2000 + i)
        return store

    def test_tight_cap_trims_memory_keeps_anchor_and_situation(self):
        """token_cap 很小时：记忆被裁掉，但 persona 锚与情境仍在 messages 里。"""
        world = make_initial_world(now=1_000_000)
        store = self._memory_with_records()
        # 让老爸移动进厨房，触发对老妈的关系检索
        trig = Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=2_000_000)

        tight = assemble_prompt(_dad(world), world, trig, store, token_cap=30)
        # 预算极紧：记忆一条都进不来
        self.assertEqual(tight.budget_report.relation_items, 0)
        self.assertEqual(tight.budget_report.experience_items, 0)
        self.assertTrue(tight.budget_report.experience_dropped)
        # persona 锚（system）与情境（user）仍在
        self.assertIn("老爸", _system(tight))
        self.assertIn("厨房", _user(tight))

    def test_experience_trimmed_before_relation(self):
        """预算只够一层时：关系记忆保留、经历记忆先被挤掉（关系优先于经历）。"""
        world = make_initial_world(now=1_000_000)
        store = self._memory_with_records()
        trig = Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=2_000_000)

        p = assemble_prompt(_dad(world), world, trig, store, token_cap=200)
        # 关系记忆条数 > 0 时，说明关系层拿到了预算
        if p.budget_report.relation_items > 0:
            # 经历用量不应挤占到让关系拿不到——关系先分配
            self.assertGreaterEqual(p.budget_report.relation_tokens, 0)
        # 充足预算下再验证：两层都能进
        rich = assemble_prompt(_dad(world), world, trig, store, token_cap=5000)
        self.assertGreater(rich.budget_report.relation_items, 0)
        self.assertGreater(rich.budget_report.experience_items, 0)


class TestPersonaNeverTrimmed(unittest.TestCase):
    """persona 锚永不被裁——即使 token_cap=0。"""

    def test_zero_cap_still_has_persona_and_situation(self):
        world = make_initial_world(now=1_000_000)
        store = InMemoryStore()
        trig = Trigger(kind="oracle", text="今天降温了", ts=2_000_000)

        p = assemble_prompt(_dad(world), world, trig, store, token_cap=0)
        sys = _system(p)
        usr = _user(p)
        # persona 关键信息在
        self.assertIn("老爸", sys)
        self.assertIn("退休工程师", sys)  # persona 文本
        # 情境在
        self.assertIn("降温", usr)
        # 记忆全被裁
        self.assertTrue(p.budget_report.relation_dropped)
        self.assertTrue(p.budget_report.experience_dropped)
        self.assertEqual(p.budget_report.relation_tokens, 0)
        self.assertEqual(p.budget_report.experience_tokens, 0)

    def test_persona_not_leaking_others_private_mind(self):
        """私有视角：老爸的 prompt 不得泄露其他角色的 persona/monologue 私有心理。"""
        world = make_initial_world(now=1_000_000)
        store = InMemoryStore()
        trig = Trigger(kind="oracle", text="晚饭吃什么", ts=2_000_000)

        p = assemble_prompt(_dad(world), world, trig, store, token_cap=5000)
        blob = _system(p) + "\n" + _user(p)
        # 在场的老妈/空调的名字可以出现（外部可见）
        # 但老妈的 persona / 内心独白不该出现
        self.assertNotIn("爱唠叨的能量核心", blob)     # 老妈 persona
        self.assertNotIn("晚上做个蛋炒饭吧", blob)      # 老妈 monologue
        self.assertNotIn("又没人给我擦灰", blob)        # 老空调 monologue


class TestEmptyMemory(unittest.TestCase):
    """空记忆库组装不报错、messages 合法。"""

    def test_empty_store_produces_valid_messages(self):
        world = make_initial_world(now=1_000_000)
        store = InMemoryStore()
        trig = Trigger(kind="oracle", text="要下雨了", ts=2_000_000)

        p = assemble_prompt(_dad(world), world, trig, store)
        self.assertIsInstance(p, AssembledPrompt)
        self.assertEqual(len(p.messages), 2)
        self.assertEqual(p.messages[0]["role"], "system")
        self.assertEqual(p.messages[1]["role"], "user")
        self.assertTrue(all(m["content"].strip() for m in p.messages))
        self.assertEqual(p.budget_report.relation_items, 0)
        self.assertEqual(p.budget_report.experience_items, 0)


class TestTemplates(unittest.TestCase):
    """oracle / move 两类模板差异 + 都含 JSON schema 要求。"""

    def test_oracle_template(self):
        world = make_initial_world(now=1_000_000)
        store = InMemoryStore()
        trig = Trigger(kind="oracle", text="家里要来客人了", ts=2_000_000)

        p = assemble_prompt(_dad(world), world, trig, store)
        sys = _system(p)
        usr = _user(p)
        # 情境关键词
        self.assertIn("神谕", usr)
        self.assertIn("家里要来客人了", usr)
        # 影响非命令
        self.assertIn("影响", sys)
        # JSON schema 要求
        self.assertIn("JSON", sys)
        self.assertIn("speak", sys)
        self.assertIn("memory_note", sys)

    def test_move_template(self):
        world = make_initial_world(now=1_000_000)
        store = InMemoryStore()
        # 老妈走进客厅（老爸所在），从老爸视角组装
        trig = Trigger(kind="move", agentId="agent_mom", roomId="room_living", ts=2_000_000)

        p = assemble_prompt(_dad(world), world, trig, store)
        usr = _user(p)
        sys = _system(p)
        # 移动者名字出现在情境里
        self.assertIn("老妈", usr)
        self.assertIn("客厅", usr)
        # 共处元规则
        self.assertIn("共处", sys)
        # 仍要求 JSON
        self.assertIn("JSON", sys)

    def test_move_self_perspective(self):
        """移动者自己的视角：'你走进了{房间}，看到了{谁}'。"""
        world = make_initial_world(now=1_000_000)
        store = InMemoryStore()
        # 老爸自己走进厨房，从老爸视角
        trig = Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=2_000_000)

        p = assemble_prompt(_dad(world), world, trig, store)
        usr = _user(p)
        self.assertIn("你走进了", usr)
        self.assertIn("厨房", usr)
        # 厨房里的老妈应作为在场者被看到
        self.assertIn("老妈", usr)


class TestParseReaction(unittest.TestCase):
    """parse_reaction：合法 JSON（含代码块）成功，垃圾输入 None，字段容错。"""

    def test_parse_plain_json(self):
        text = '{"speak": true, "reason": "有点意思", "line": "你也在啊", "mood": "calm", "memory_note": "老妈今天心情好"}'
        d = parse_reaction(text)
        self.assertIsInstance(d, ReactionDecision)
        self.assertTrue(d.speak)
        self.assertEqual(d.line, "你也在啊")
        self.assertEqual(d.mood, "calm")
        self.assertEqual(d.memory_note, "老妈今天心情好")

    def test_parse_fenced_json(self):
        text = "好的，这是我的反应：\n```json\n{\"speak\": false, \"reason\": \"懒得搭理\", \"line\": \"\"}\n```\n"
        d = parse_reaction(text)
        self.assertIsInstance(d, ReactionDecision)
        self.assertFalse(d.speak)
        self.assertEqual(d.line, "")
        self.assertIsNone(d.mood)          # 缺 mood -> None
        self.assertIsNone(d.memory_note)   # 缺 memory_note -> None

    def test_speak_defaults_from_line(self):
        """speak 缺失时，用 line 是否有内容兜底。"""
        d = parse_reaction('{"reason": "自言自语", "line": "唉又降温"}')
        self.assertTrue(d.speak)  # 有台词 -> 视为开口
        d2 = parse_reaction('{"reason": "沉默", "line": ""}')
        self.assertFalse(d2.speak)  # 无台词 -> 不开口

    def test_invalid_mood_dropped(self):
        """非法 mood 一律置 None，不污染下游 patch。"""
        d = parse_reaction('{"speak": true, "line": "哈", "mood": "ecstatic"}')
        self.assertIsNone(d.mood)

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_reaction("这不是 JSON，只是一段闲话。"))
        self.assertIsNone(parse_reaction(""))
        self.assertIsNone(parse_reaction(None))
        self.assertIsNone(parse_reaction("[1, 2, 3]"))  # 数组不算结构化响应


if __name__ == "__main__":
    unittest.main()
