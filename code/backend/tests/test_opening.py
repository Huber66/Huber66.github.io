"""开场故事生成测试（F08 · Task 12 验收）。

标准库 unittest + FastAPI TestClient，与 test_api / test_history 同一套运行：
    cd backend && python3 -m unittest discover tests

核心验收线（brief）：
  · stub_opening 确定性；换角色→文本明显变；换布局→房间名体现变化；
  · 3-5 条区间；无「欢迎/使用/系统」式话术；
  · parse_opening：合法 beats→解析；beats 缺失/空→None；agentId 不在 world→置 None；
  · synthesize_opening 无 key（StubChatModel）→ 走 stub 不抛，3-5 条；
  · driver.open_world：source=="opening"、写进 world.events、走溯源；空 beats→();
  · POST /world/opening 返回 {events, world}，events 非空且 source==opening。
"""

import random
import unittest

from fastapi.testclient import TestClient

from cyberhome.api import create_app
from cyberhome.driver import WorldDriver
from cyberhome.llm import StubChatModel
from cyberhome.opening import (
    OpeningBeat,
    assemble_opening_prompt,
    parse_opening,
    stub_opening,
    synthesize_opening,
)
from cyberhome.reactor import StubReactor
from cyberhome.seed import make_initial_world

from dataclasses import replace

# 严禁出现在开场里的「产品引导 / 系统话术」——它是「生活」不是「引导」（红线 6）。
_BANNED_PHRASES = ("欢迎", "使用", "系统", "点击", "按钮", "开始体验", "提示")


def _fixed_reactor() -> StubReactor:
    return StubReactor(rng=random.Random(42))


def _counter_event_id():
    n = {"i": 0}

    def gen():
        n["i"] += 1
        return f"evt_test_{n['i']}"

    return gen


def _texts(beats) -> list[str]:
    return [b.text for b in beats]


class TestStubOpening(unittest.TestCase):
    """确定性兜底开场：可复现、换角色/布局必变、条数区间、无系统话术。"""

    def test_deterministic(self):
        """同一 (world, seed) 必得同一结果（可复现红线）。"""
        world = make_initial_world(now=1_000_000)
        self.assertEqual(_texts(stub_opening(world, 0)), _texts(stub_opening(world, 0)))
        self.assertEqual(stub_opening(world, 3), stub_opening(world, 3))

    def test_count_in_range(self):
        """开场条数落在 [3, 5]。"""
        world = make_initial_world(now=1_000_000)
        beats = stub_opening(world, 0)
        self.assertGreaterEqual(len(beats), 3)
        self.assertLessEqual(len(beats), 5)

    def test_agent_change_alters_opening(self):
        """换角色（改 name / monologue）→ 开场文本明显变化（F08 验收线核心）。"""
        world = make_initial_world(now=1_000_000)
        before = _texts(stub_opening(world, 0))
        changed = replace(
            world,
            agents=tuple(
                replace(a, name=a.name + "·改", monologue="全新的心事" + a.id)
                for a in world.agents
            ),
        )
        after = _texts(stub_opening(changed, 0))
        self.assertNotEqual(before, after)
        # 新名字确实进入了开场文本（不是仅仅顺序抖动）
        self.assertTrue(any("·改" in t for t in after))

    def test_layout_change_reflected(self):
        """换布局（改房间名）→ 房间名体现在开场里。"""
        world = make_initial_world(now=1_000_000)
        renamed = replace(world, rooms=tuple(replace(r, name="新" + r.name) for r in world.rooms))
        after = _texts(stub_opening(renamed, 0))
        self.assertTrue(any("新" in t for t in after))

    def test_room_reassignment_reflected(self):
        """换布局（把角色挪到别的房间）→ 该角色开场句里的房间名随之变。"""
        world = make_initial_world(now=1_000_000)
        # 老爸原在客厅，挪到卧室
        moved = replace(
            world,
            agents=tuple(
                replace(a, roomId="room_bedroom") if a.id == "agent_dad" else a
                for a in world.agents
            ),
        )
        dad_beat = next(b for b in stub_opening(moved, 0) if b.agentId == "agent_dad")
        self.assertIn("卧室", dad_beat.text)

    def test_no_system_phrases(self):
        """开场不含「欢迎/使用/系统」式产品话术（红线 6）。"""
        world = make_initial_world(now=1_000_000)
        for seed in range(4):
            joined = "".join(_texts(stub_opening(world, seed)))
            for banned in _BANNED_PHRASES:
                self.assertNotIn(banned, joined, f"seed={seed} 泄漏系统话术: {banned}")

    def test_seed_variant_changes_wording(self):
        """seed>0 换一套措辞变体（「换一种讲法」）。"""
        world = make_initial_world(now=1_000_000)
        self.assertNotEqual(_texts(stub_opening(world, 0)), _texts(stub_opening(world, 1)))

    def test_fills_to_min_when_few_agents(self):
        """角色不足 3 个时用世界级氛围句补足到 3 条（agentId 为 None）。"""
        world = make_initial_world(now=1_000_000)
        one = replace(world, agents=(world.agents[0],))
        beats = stub_opening(one, 0)
        self.assertGreaterEqual(len(beats), 3)
        # 至少一条世界级氛围句（无 agentId）
        self.assertTrue(any(b.agentId is None for b in beats))
        # 有角色的那条 agentId 指向真实角色
        self.assertTrue(any(b.agentId == world.agents[0].id for b in beats))

    def test_beat_agent_ids_point_to_real_agents(self):
        """每条角色 beat 的 agentId 都指向在册角色。"""
        world = make_initial_world(now=1_000_000)
        valid = {a.id for a in world.agents}
        for b in stub_opening(world, 0):
            if b.agentId is not None:
                self.assertIn(b.agentId, valid)


class TestAssemblePrompt(unittest.TestCase):
    """prompt 组装：system 必含 JSON（无 key 闭环机关）+ user 列出房间与角色。"""

    def test_system_contains_json_schema(self):
        world = make_initial_world(now=1_000_000)
        messages = assemble_opening_prompt(world, seed=0)
        system = next(m["content"] for m in messages if m["role"] == "system")
        self.assertIn("JSON", system)
        self.assertIn("beats", system)

    def test_user_lists_rooms_and_agents(self):
        world = make_initial_world(now=1_000_000)
        user = next(m["content"] for m in assemble_opening_prompt(world, seed=0) if m["role"] == "user")
        self.assertIn("客厅", user)
        self.assertIn("老妈", user)
        self.assertIn("agent_mom", user)  # 带 id 供 LLM 关联

    def test_seed_appends_reroll_hint(self):
        world = make_initial_world(now=1_000_000)
        user0 = next(m["content"] for m in assemble_opening_prompt(world, seed=0) if m["role"] == "user")
        user1 = next(m["content"] for m in assemble_opening_prompt(world, seed=1) if m["role"] == "user")
        self.assertNotEqual(user0, user1)  # seed>0 追加「换一种讲法」


class TestParseOpening(unittest.TestCase):
    """响应解析：合法→tuple；缺 beats→None；瞎编 id→None（不信 LLM）。"""

    def test_parse_valid_beats(self):
        world = make_initial_world(now=1_000_000)
        text = (
            '{"beats": [{"text": "厨房飘来饭香", "agentId": "agent_mom"}, '
            '{"text": "客厅安静下来", "agentId": ""}]}'
        )
        beats = parse_opening(text, world)
        self.assertIsNotNone(beats)
        self.assertEqual(len(beats), 2)
        self.assertEqual(beats[0].agentId, "agent_mom")
        self.assertIsNone(beats[1].agentId)  # 空字符串 → None（世界级叙述）

    def test_missing_beats_returns_none(self):
        """无 beats 字段（StubChatModel 默认占位 JSON 的形状）→ None（走 stub 的判定点）。"""
        world = make_initial_world(now=1_000_000)
        self.assertIsNone(parse_opening('{"speak": true, "line": ""}', world))

    def test_empty_beats_returns_none(self):
        world = make_initial_world(now=1_000_000)
        self.assertIsNone(parse_opening('{"beats": []}', world))

    def test_garbage_returns_none(self):
        world = make_initial_world(now=1_000_000)
        self.assertIsNone(parse_opening("这不是 JSON", world))
        self.assertIsNone(parse_opening(None, world))

    def test_unknown_agent_id_nulled(self):
        """LLM 瞎编不存在的 agentId → 置 None，不当作角色关联。"""
        world = make_initial_world(now=1_000_000)
        text = '{"beats": [{"text": "有人在低语", "agentId": "ghost_999"}]}'
        beats = parse_opening(text, world)
        self.assertEqual(len(beats), 1)
        self.assertIsNone(beats[0].agentId)

    def test_empty_text_beats_dropped(self):
        """text 为空的 beat 被丢弃；全空 → None。"""
        world = make_initial_world(now=1_000_000)
        self.assertIsNone(parse_opening('{"beats": [{"text": "", "agentId": "agent_mom"}]}', world))

    def test_clamped_to_max(self):
        """超过 5 条被截断到 5 条。"""
        world = make_initial_world(now=1_000_000)
        items = ",".join(f'{{"text": "第{i}句", "agentId": ""}}' for i in range(8))
        beats = parse_opening(f'{{"beats": [{items}]}}', world)
        self.assertEqual(len(beats), 5)


class TestSynthesizeOpening(unittest.TestCase):
    """主入口：无 key 走 stub 不抛；注入含 beats 的 stub 走 LLM 解析路径。"""

    def test_no_key_falls_back_to_stub(self):
        """无 key（默认 StubChatModel，占位 JSON 不含 beats）→ 走 stub，3-5 条不抛。"""
        world = make_initial_world(now=1_000_000)
        beats = synthesize_opening(StubChatModel(), world, seed=0)
        self.assertGreaterEqual(len(beats), 3)
        self.assertLessEqual(len(beats), 5)
        # 与直接 stub 一致（确认确实走了 stub 分支）
        self.assertEqual(_texts(beats), _texts(stub_opening(world, 0)))

    def test_no_key_deterministic(self):
        world = make_initial_world(now=1_000_000)
        m = StubChatModel()
        self.assertEqual(
            _texts(synthesize_opening(m, world, seed=0)),
            _texts(synthesize_opening(m, world, seed=0)),
        )

    def test_uses_llm_beats_when_valid(self):
        """注入含 beats 的 StubChatModel（模拟真 LLM 给了有效开场）→ 用 LLM 结果。"""
        world = make_initial_world(now=1_000_000)
        template = {
            "beats": [
                {"text": "老妈端出一盘蛋炒饭", "agentId": "agent_mom"},
                {"text": "窗外的天暗了些", "agentId": ""},
                {"text": "冰箱哼了一声", "agentId": "agent_fridge"},
            ]
        }
        model = StubChatModel(json_template=template)
        beats = synthesize_opening(model, world, seed=0)
        self.assertEqual(beats[0].text, "老妈端出一盘蛋炒饭")
        self.assertEqual(beats[0].agentId, "agent_mom")
        self.assertIsNone(beats[1].agentId)

    def test_synth_never_raises_on_bad_model(self):
        """chat 抛异常时吞掉退回 stub（绝不 500）。"""
        class BoomModel:
            def chat(self, messages, **kwargs):
                raise RuntimeError("boom")

        world = make_initial_world(now=1_000_000)
        beats = synthesize_opening(BoomModel(), world, seed=0)
        self.assertEqual(_texts(beats), _texts(stub_opening(world, 0)))


class TestDriverOpenWorld(unittest.TestCase):
    """driver.open_world：落 source=opening 事件 + 空 patch 走溯源，不走 step。"""

    def _driver(self) -> WorldDriver:
        return WorldDriver(
            make_initial_world(now=1_000_000),
            reactor=_fixed_reactor(),
            event_id=_counter_event_id(),
        )

    def test_open_world_appends_opening_events(self):
        d = self._driver()
        before_events = len(d.world.events)
        before_log = len(d.store)
        beats = (
            OpeningBeat(text="老妈系上围裙", agentId="agent_mom"),
            OpeningBeat(text="灯影晃了晃", agentId=None),
        )
        events = d.open_world(beats, now=2_000_000)
        # 返回的事件全是 opening 来源
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e.source == "opening" for e in events))
        self.assertEqual(events[0].agentId, "agent_mom")
        self.assertIsNone(events[1].agentId)
        # 写进 world.events（追加，不改结构）
        self.assertEqual(len(d.world.events), before_events + 2)
        # 走了溯源：日志长度 +1（一次 commit）
        self.assertEqual(len(d.store), before_log + 1)

    def test_open_world_does_not_change_structure(self):
        """opening 只追加叙事：agents / rooms 不变。"""
        d = self._driver()
        agents_before = d.world.agents
        rooms_before = d.world.rooms
        d.open_world((OpeningBeat(text="家里热闹起来", agentId=None),), now=2_000_000)
        self.assertEqual(d.world.agents, agents_before)
        self.assertEqual(d.world.rooms, rooms_before)

    def test_empty_beats_no_commit(self):
        """空 beats → 返回 ()，不 commit（无叙事噪声）。"""
        d = self._driver()
        before_events = len(d.world.events)
        before_log = len(d.store)
        self.assertEqual(d.open_world((), now=2_000_000), ())
        self.assertEqual(len(d.world.events), before_events)
        self.assertEqual(len(d.store), before_log)

    def test_open_world_events_are_replayable(self):
        """opening 事件进溯源日志，可被回放（与 add_agent 同源）。"""
        d = self._driver()
        d.open_world((OpeningBeat(text="厨房飘出葱花香", agentId="agent_mom"),), now=2_000_000)
        replayed = d.store.replay_events()
        self.assertTrue(any(e.text == "厨房飘出葱花香" and e.source == "opening" for e in replayed))


class TestOpeningApi(unittest.TestCase):
    """POST /world/opening：无 key 也返回 {events, world}，events 非空且 source==opening。"""

    def _client(self) -> TestClient:
        # 显式注入 StubChatModel 走无 key 路径（确定性、不联网）。
        return TestClient(create_app(chat_model=StubChatModel()))

    def test_post_opening_returns_events_and_world(self):
        client = self._client()
        resp = client.post("/world/opening", json={"seed": 0})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("events", data)
        self.assertIn("world", data)
        self.assertGreater(len(data["events"]), 0)
        self.assertTrue(all(e["source"] == "opening" for e in data["events"]))
        self.assertGreaterEqual(len(data["events"]), 3)
        self.assertLessEqual(len(data["events"]), 5)

    def test_post_opening_no_system_phrases(self):
        client = self._client()
        data = client.post("/world/opening", json={"seed": 0}).json()
        joined = "".join(e["text"] for e in data["events"])
        for banned in _BANNED_PHRASES:
            self.assertNotIn(banned, joined)

    def test_post_opening_default_seed(self):
        """seed 可省略（默认 0），不 500。"""
        client = self._client()
        resp = client.post("/world/opening", json={})
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(len(resp.json()["events"]), 0)

    def test_post_opening_appended_to_world_events(self):
        """开场事件被追加进世界事件流（前端可直接用返回 world 覆盖）。"""
        client = self._client()
        before = len(client.get("/world").json()["events"])
        data = client.post("/world/opening", json={"seed": 0}).json()
        self.assertGreater(len(data["world"]["events"]), before)


if __name__ == "__main__":
    unittest.main()
