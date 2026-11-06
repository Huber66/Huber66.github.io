"""惰性补演 catchup 测试（F10 · 验收）。

标准库 unittest + 临时文件 DB，与 test_engine/test_persistence 同一套运行：
    cd backend && python3 -m unittest discover tests

覆盖 spec F10 四条验收：
  1. 模拟离开 6 小时打开 → 生成 1-5 条 source="catchup" 事件，且落在时间窗 (last_seen, now] 内。
  2. 连续刷新不重复补演同一段（第二次 last_seen 已推进到 now，gap 归零被阈值挡下）。
  3. now 可注入 → 同 (world, since, now, 固定 rng) 必得同一批事件（可复现，事件溯源前提）。
  4. LLM 失败降级不崩：LLMReactor 的 chat 全崩 → 委托 fallback StubReactor，仍产出补演事件。

另覆盖三条空返回护栏（无 persistence / 无 last_seen / gap 不足阈值），佐证「后端未运行
期间无常驻任务」——补演只在被显式调用时跑一次。用固定 seed reactor + 计数 event_id，
保证 step() 可复现。
"""

import os
import random
import tempfile
import unittest
from dataclasses import replace

from cyberhome import (
    LLMReactor, SQLitePersistenceStore, StubReactor, Trigger, WorldDriver,
    make_initial_world,
    step,
)
from cyberhome.memory import InMemoryStore

WORLD_ID = "world_home"
_SIX_HOURS_MS = 6 * 60 * 60 * 1000
# 固定的「现在」（真实量级毫秒戳），注入 catchup 以脱离 wall clock（可复现红线）。
_NOW = 1_700_000_000_000


def _fixed_reactor() -> StubReactor:
    """固定 seed 的 reactor，保证 catchup 分片文案可复现（事件溯源前提）。"""
    return StubReactor(rng=random.Random(42))


def _counter_event_id():
    n = {"i": 0}
    def gen():
        n["i"] += 1
        return f"evt_test_{n['i']}"
    return gen


class _BoomChatModel:
    """chat 必抛的假模型：模拟 LLM 端点全崩，用于验证 LLMReactor 降级到 fallback 不崩。"""

    def chat(self, messages):
        raise RuntimeError("boom: llm endpoint down")


class TestCatchup(unittest.TestCase):
    def setUp(self):
        # 临时文件 DB（非 :memory:）：catchup 依赖 persistence 的 last_seen，需真实读写。
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._store = SQLitePersistenceStore(self._db_path)

    def tearDown(self):
        self._store.close()
        if os.path.exists(self._db_path):
            os.remove(self._db_path)

    def _driver_after_leaving(self, gap_ms: int, *, reactor=None, now: int = _NOW) -> WorldDriver:
        """建一个已「离开 gap_ms」的 driver：落库快照 + set_last_seen(now-gap)。

        刻意不走 bootstrap（那会把 last_seen 设成 now，抹掉「离开」语义），而是手动把
        last_seen 拨回 now-gap，忠实模拟「上次看是 gap 之前，现在 now 才回来」。
        """
        world = make_initial_world(now=0)
        self._store.save_snapshot(world)
        self._store.set_last_seen(world.id, now - gap_ms)
        return WorldDriver(
            world,
            reactor=reactor or _fixed_reactor(),
            event_id=_counter_event_id(),
            persistence=self._store,
        )

    # ---- 验收 1：离开 6 小时 → 1-5 条 catchup，落在时间窗内 ----
    def test_leaving_six_hours_generates_catchup_in_window(self):
        now = _NOW
        since = now - _SIX_HOURS_MS
        driver = self._driver_after_leaving(_SIX_HOURS_MS, now=now)

        events = driver.catchup(now=now)

        # 生成若干条生活事件。每个 catchup slice 可能包含移动 + 多人同屋对话，
        # 因此事件条数不再等于 slice 条数。
        self.assertGreaterEqual(len(events), 1)
        self.assertLessEqual(len(events), 20)
        self.assertTrue(all(e.source in {"catchup", "move"} for e in events))
        self.assertTrue(any(e.source == "catchup" for e in events))
        # 全部落在时间窗 (since, now] 内：严格晚于上次离开、不晚于现在
        for e in events:
            self.assertGreater(e.ts, since)
            self.assertLessEqual(e.ts, now)
        # ts 单调不减（时间轴稳定有序，便于事件流顺序渲染）
        self.assertEqual([e.ts for e in events], sorted(e.ts for e in events))
        # 事件已写通进世界事件流（catchup 走 _commit 落库 + 折叠）
        self.assertTrue(any(e.source == "catchup" for e in driver.world.events))

    # ---- 验收 2：连续刷新不重复补演同一段 ----
    def test_consecutive_refresh_does_not_repeat(self):
        now = _NOW
        driver = self._driver_after_leaving(_SIX_HOURS_MS, now=now)

        first = driver.catchup(now=now)
        self.assertGreaterEqual(len(first), 1)
        # 第一次补演后 last_seen 已被 _commit 推进到 now（trigger.ts=now）
        self.assertEqual(self._store.get_last_seen(WORLD_ID), now)

        # 第二次同一 now：gap 归零 < 阈值 → 不再补（第一道闸：last_seen 推进）
        self.assertEqual(driver.catchup(now=now), ())
        # 稍后再刷但不足 1 分钟：仍不补（第二道闸：最小间隔阈值）
        self.assertEqual(driver.catchup(now=now + 30_000), ())

    # ---- 验收 3：now 可注入 → 可复现 ----
    def test_now_injectable_and_reproducible(self):
        now = _NOW
        # 两个各自独立 DB 的 driver，同样「离开 6h」、同固定 rng → 必得同一批文案与时刻
        events_a = self._run_isolated_catchup(now)
        events_b = self._run_isolated_catchup(now)
        self.assertEqual([e.text for e in events_a], [e.text for e in events_b])
        self.assertEqual([e.ts for e in events_a], [e.ts for e in events_b])

    def _run_isolated_catchup(self, now: int):
        """在一个隔离的临时 DB 上跑一次 6h 补演，返回事件。用于可复现对拍（两次互不共享 last_seen）。"""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        store = SQLitePersistenceStore(path)
        try:
            world = make_initial_world(now=0)
            store.save_snapshot(world)
            store.set_last_seen(world.id, now - _SIX_HOURS_MS)
            driver = WorldDriver(
                world, reactor=_fixed_reactor(),
                event_id=_counter_event_id(), persistence=store,
            )
            return driver.catchup(now=now)
        finally:
            store.close()
            if os.path.exists(path):
                os.remove(path)

    # ---- 验收 4：LLM 失败降级不崩 ----
    def test_llm_failure_degrades_without_crash(self):
        now = _NOW
        # LLMReactor 的 chat 全崩 → 每片候选全部硬失败 → 委托 fallback StubReactor（固定 rng）
        llm_reactor = LLMReactor(
            _BoomChatModel(), InMemoryStore(),
            fallback=StubReactor(rng=random.Random(42)),
        )
        driver = self._driver_after_leaving(_SIX_HOURS_MS, reactor=llm_reactor, now=now)

        # 不抛异常（降级红线：LLM 崩不许把异常抛给引擎/driver）
        events = driver.catchup(now=now)

        # 降级后仍有补演事件（fallback 顶上，世界不至于彻底哑火）。
        self.assertGreaterEqual(len(events), 1)
        self.assertLessEqual(len(events), 20)
        self.assertTrue(all(e.source in {"catchup", "move"} for e in events))
        self.assertTrue(any(e.source == "catchup" for e in events))

    def test_autonomous_intent_moves_mom_to_kitchen(self):
        world = make_initial_world(now=0)
        agents = tuple(
            replace(agent, roomId="room_living") if agent.id == "agent_mom" else agent
            for agent in world.agents
        )
        world = replace(world, agents=agents)

        events, patch = step(
            world,
            Trigger(kind="catchup", since=0, ts=60_000),
            reactor=_fixed_reactor(),
            event_id=_counter_event_id(),
        )

        self.assertEqual(patch.agent_updates["agent_mom"]["roomId"], "room_kitchen")
        self.assertTrue(
            any(e.source == "move" and e.agentId == "agent_mom" for e in events)
        )

    def test_copresent_agents_dialogue_is_recorded_in_event_stream(self):
        world = make_initial_world(now=0)

        events, patch = step(
            world,
            Trigger(kind="catchup", since=0, ts=60_000),
            reactor=_fixed_reactor(),
            event_id=_counter_event_id(),
        )

        speaker_ids = {e.agentId for e in events if e.source == "catchup"}
        self.assertGreaterEqual(len(speaker_ids), 2)
        self.assertTrue(speaker_ids.issubset({agent.id for agent in world.agents}))
        self.assertEqual(patch.agent_updates, {})

    # ---- 护栏：三种空返回情形（佐证「无常驻任务」，只在被调用时才可能补） ----
    def test_returns_empty_on_guards(self):
        now = _NOW
        # a) 无 persistence：纯内存 driver 无 last_seen，无从知离开多久（向后兼容）
        plain = WorldDriver(
            make_initial_world(now=0), reactor=_fixed_reactor(), event_id=_counter_event_id(),
        )
        self.assertEqual(plain.catchup(now=now), ())

        # b) 有 persistence 但无 last_seen 记录：世界还没被「看过」，谈不上补演
        world = make_initial_world(now=0)
        self._store.save_snapshot(world)  # 仅落快照，不写 last_seen
        self.assertIsNone(self._store.get_last_seen(WORLD_ID))
        driver = WorldDriver(
            world, reactor=_fixed_reactor(),
            event_id=_counter_event_id(), persistence=self._store,
        )
        self.assertEqual(driver.catchup(now=now), ())

        # c) 离开时长不足阈值：刚看过就刷新，不补
        self._store.set_last_seen(WORLD_ID, now - 10_000)  # 仅离开 10s
        self.assertEqual(driver.catchup(now=now), ())


if __name__ == "__main__":
    unittest.main()
