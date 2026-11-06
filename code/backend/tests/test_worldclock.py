"""世界时钟后端化测试（F11 验收）。

标准库 unittest，与 test_engine/test_api 同一套运行方式：
    cd backend && python3 -m unittest discover tests

覆盖 F11 五条：
  1. worldclock 映射：next_phase 环形轮转、phase_label/phase_short 逐字对齐前端 PHASE_META。
  2. step time 分支：第一条 source="time" 事件、事件 clockLabel=新 phase 的 short、
     patch.world_updates 含 phase(下一档) + clockLabel(新 phase 完整 label)，并可追加轻量自主生活事件。
  3. apply_patch 折叠 time patch 后 phase 与 clockLabel 一致（F11 验收第一条）。
  4. ambientInventory round-trip + 旧快照（无该键）兼容（向后兼容红线）。
  5. advance-time 端点：注入 driver，POST 后 world.phase 前进、事件 source="time"。
"""

import random
import unittest
from dataclasses import replace

from fastapi.testclient import TestClient

from cyberhome import (
    StubReactor, Trigger, WorldDriver,
    advance_phase_label, apply_patch, make_initial_world, next_phase, phase_label, phase_short, step,
)
from cyberhome.api import create_app
from cyberhome.schemas import world_from_dict, world_to_dict
from cyberhome.seed import AMBIENT_INVENTORY


def _counter_event_id():
    n = {"i": 0}
    def gen():
        n["i"] += 1
        return f"evt_test_{n['i']}"
    return gen


class TestWorldClockMapping(unittest.TestCase):
    def test_next_phase_is_cyclic(self):
        """环形轮转：day→dusk→night→lateNight→day。"""
        self.assertEqual(next_phase("day"), "dusk")
        self.assertEqual(next_phase("dusk"), "night")
        self.assertEqual(next_phase("night"), "lateNight")
        self.assertEqual(next_phase("lateNight"), "day")

    def test_phase_label_and_short(self):
        """默认 label/short 仍逐字对齐初始世界与前端 PHASE_META。"""
        self.assertEqual(phase_label("day"), "周三 午后 14:20")
        self.assertEqual(phase_short("day"), "午后")
        self.assertEqual(phase_label("dusk"), "周三 傍晚 18:40")
        self.assertEqual(phase_short("dusk"), "傍晚")
        self.assertEqual(phase_label("night"), "周三 夜晚 21:30")
        self.assertEqual(phase_short("night"), "夜晚")
        self.assertEqual(phase_label("lateNight"), "周四 深夜 01:10")
        self.assertEqual(phase_short("lateNight"), "深夜")

    def test_advance_phase_label_keeps_real_week_cycle(self):
        """推进时星期不再固定周三/周四，而是按真实一周滚动。"""
        self.assertEqual(advance_phase_label("周三 夜晚 21:30", "night", "lateNight"), "周四 深夜 01:10")
        self.assertEqual(advance_phase_label("周四 深夜 01:10", "lateNight", "day"), "周四 午后 14:20")
        self.assertEqual(advance_phase_label("周日 夜晚 21:30", "night", "lateNight"), "周一 深夜 01:10")


class TestStepTime(unittest.TestCase):
    def test_time_step_emits_time_event_first_and_patch(self):
        """第一条 source=time 事件；事件 clockLabel=新 phase short；patch 含 phase+完整 label。"""
        world = make_initial_world(now=1_000_000)  # phase=dusk
        events, patch = step(
            world, Trigger(kind="time", ts=2_000_000),
            event_id=_counter_event_id(),
        )
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0].source, "time")
        # dusk 的下一档是 night
        self.assertEqual(events[0].clockLabel, phase_short("night"))  # 事件用 short「夜晚」
        self.assertEqual(patch.world_updates["phase"], "night")
        self.assertEqual(patch.world_updates["clockLabel"], "周三 夜晚 21:30")  # world 用完整 label
        # 事件 clockLabel(short) 与 world_updates.clockLabel(label) 口径不同——刻意如此（对齐前端）
        self.assertNotEqual(events[0].clockLabel, patch.world_updates["clockLabel"])

    def test_time_step_may_emit_autonomous_life_events(self):
        """time 分支推进世界节律时，也允许顺手补一片自主生活。"""
        world = make_initial_world(now=1_000_000)
        events, patch = step(
            world, Trigger(kind="time", ts=2_000_000),
            reactor=StubReactor(rng=random.Random(42)), event_id=_counter_event_id(),
        )
        self.assertEqual(events[0].source, "time")
        self.assertTrue(all(e.source in {"time", "move"} for e in events))
        self.assertTrue(any(e.source == "move" for e in events))
        self.assertFalse(any(e.source == "catchup" for e in events))

    def test_night_wander_moves_dad_to_study_not_bathroom(self):
        """夜晚低频自然走动时，老爸去书房，不再去尚未完善的卫生间。"""
        world = make_initial_world(now=1_000_000)  # dusk -> night
        events, patch = step(
            world, Trigger(kind="time", ts=2_000_000),
            reactor=StubReactor(rng=random.Random(42)), event_id=_counter_event_id(),
        )

        move = next(event for event in events if event.source == "move")
        self.assertEqual(move.agentId, "agent_dad")
        self.assertIn("书房", move.text)
        self.assertNotIn("卫生间", move.text)
        self.assertEqual(patch.agent_updates["agent_dad"]["roomId"], "room_study")

    def test_time_step_does_not_wait_for_injected_reactor(self):
        """推进时间是 UI 时钟动作，不应等待生产 LLMReactor 的联网响应。"""
        class SlowReactor:
            def react(self, world, trigger, memory):
                raise AssertionError("time step should not call the injected reactor")

        world = make_initial_world(now=1_000_000)
        events, _patch = step(
            world, Trigger(kind="time", ts=2_000_000),
            reactor=SlowReactor(), event_id=_counter_event_id(),
        )
        self.assertEqual(events[0].source, "time")

    def test_late_night_advance_moves_humans_to_bedroom(self):
        """深夜手动推进时，所有该睡的人类都应先去卧室，而不是拖到白天补账。"""
        world = replace(
            make_initial_world(now=1_000_000),
            phase="lateNight",
            clockLabel=phase_label("lateNight"),
        )
        events, patch = step(
            world, Trigger(kind="time", ts=2_000_000),
            reactor=StubReactor(rng=random.Random(42)), event_id=_counter_event_id(),
        )
        self.assertEqual(events[0].source, "time")
        move_events = [e for e in events if e.source == "move"]
        self.assertEqual(len(move_events), 1)
        self.assertIsNone(move_events[0].agentId)
        self.assertIn("老爸和老妈", move_events[0].text)
        self.assertEqual(patch.agent_updates["agent_dad"]["roomId"], "room_bedroom")
        self.assertEqual(patch.agent_updates["agent_mom"]["roomId"], "room_bedroom")

    def test_night_to_late_night_moves_humans_to_bedroom(self):
        """从夜晚进入深夜时，老爸老妈应在这一档就去卧室。"""
        world = replace(
            make_initial_world(now=1_000_000),
            phase="night",
            clockLabel=phase_label("night"),
        )
        events, patch = step(
            world, Trigger(kind="time", ts=2_000_000),
            reactor=StubReactor(rng=random.Random(42)), event_id=_counter_event_id(),
        )
        self.assertEqual(patch.world_updates["phase"], "lateNight")
        move_events = [e for e in events if e.source == "move"]
        self.assertEqual(len(move_events), 1)
        self.assertIsNone(move_events[0].agentId)
        self.assertIn("老爸和老妈", move_events[0].text)
        self.assertEqual(patch.agent_updates["agent_dad"]["roomId"], "room_bedroom")
        self.assertEqual(patch.agent_updates["agent_mom"]["roomId"], "room_bedroom")

    def test_time_step_is_pure(self):
        """step 是纯函数：time 分支不改动入参 world。"""
        world = make_initial_world(now=1_000_000)
        before_phase = world.phase
        step(world, Trigger(kind="time", ts=2_000_000), event_id=_counter_event_id())
        self.assertEqual(world.phase, before_phase)  # 入参未变（dusk）


class TestApplyPatchConsistency(unittest.TestCase):
    def test_folded_world_phase_and_label_consistent(self):
        """折叠 time patch 后，星期随真实日期向前走，而不是回到固定周三。"""
        world = make_initial_world(now=1_000_000)  # dusk / 周三 傍晚 18:40
        labels = []
        for _ in range(4):
            events, patch = step(
                world, Trigger(kind="time", ts=2_000_000),
                event_id=_counter_event_id(),
            )
            world = apply_patch(world, patch, events)
            labels.append(world.clockLabel)
        self.assertEqual(world.phase, "dusk")  # 四档环形回到起点
        self.assertEqual(labels, [
            "周三 夜晚 21:30",
            "周四 深夜 01:10",
            "周四 午后 14:20",
            "周四 傍晚 18:40",
        ])

    def test_weekday_cycles_from_monday_to_sunday(self):
        """连续推进会经过完整一周，并在周日之后回到周一。"""
        world = replace(make_initial_world(now=1_000_000), phase="night", clockLabel="周日 夜晚 21:30")
        events, patch = step(world, Trigger(kind="time", ts=2_000_000), event_id=_counter_event_id())
        world = apply_patch(world, patch, events)
        self.assertEqual(world.clockLabel, "周一 深夜 01:10")

        world = replace(make_initial_world(now=1_000_000), phase="dusk", clockLabel="周一 傍晚 18:40")
        seen = set()
        for _ in range(28):
            events, patch = step(world, Trigger(kind="time", ts=2_000_000), event_id=_counter_event_id())
            world = apply_patch(world, patch, events)
            seen.add(world.clockLabel.split()[0])
        self.assertEqual(seen, {"周一", "周二", "周三", "周四", "周五", "周六", "周日"})


class TestAmbientInventoryRoundTrip(unittest.TestCase):
    def test_roundtrip_preserves_inventory(self):
        """序列化↔反序列化对称，含 ambientInventory（前后端字段对齐红线）。"""
        world = make_initial_world(now=1_000_000)
        self.assertEqual(world.ambientInventory, AMBIENT_INVENTORY)
        restored = world_from_dict(world_to_dict(world))
        self.assertEqual(restored, world)
        self.assertEqual(restored.ambientInventory, AMBIENT_INVENTORY)

    def test_legacy_snapshot_without_inventory(self):
        """旧快照（无 ambientInventory 键）不报错，得空 tuple（F03 向后兼容）。"""
        data = world_to_dict(make_initial_world(now=1_000_000))
        del data["ambientInventory"]  # 模拟 F11 之前落库的旧世界快照
        restored = world_from_dict(data)  # 不应 KeyError
        self.assertEqual(restored.ambientInventory, ())


class TestAdvanceTimeEndpoint(unittest.TestCase):
    def _client_with_driver(self) -> TestClient:
        driver = WorldDriver(
            make_initial_world(now=1_000_000),
            reactor=StubReactor(rng=random.Random(42)),
            event_id=_counter_event_id(),
        )
        return TestClient(create_app(driver=driver))

    def test_advance_time_moves_phase(self):
        """POST /world/advance-time：返回 {events, world}，phase 前进、事件 source=time。"""
        resp = self._client_with_driver().post("/world/advance-time")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("events", data)
        self.assertIn("world", data)
        # dusk 的下一档是 night；world.clockLabel 亦随之为完整 label
        self.assertEqual(data["world"]["phase"], "night")
        self.assertEqual(data["world"]["clockLabel"], "周三 夜晚 21:30")
        self.assertEqual(data["events"][0]["source"], "time")
        self.assertEqual(data["events"][0]["clockLabel"], phase_short("night"))
        self.assertTrue(all(e["source"] in {"time", "catchup", "move"} for e in data["events"]))
        # world 随 GET /world 也应带 ambientInventory（asdict 自动含）
        self.assertIn("ambientInventory", data["world"])


if __name__ == "__main__":
    unittest.main()
