"""历史回放/回溯/what-if 只读能力测试（F12 · Task 15 验收，决策 32/33）。

标准库 unittest + FastAPI TestClient，与 test_api / test_persistence 同一套运行：
    cd backend && python3 -m unittest discover tests

核心红线（只读）：三个端点都不得改变主线世界/日志。测试分两层：
  · 单元：直接测 driver.store 的溯源原语（replay_events / world_at / fork_readonly）；
  · 集成：走 HTTP 端点（GET /world/events、GET /world/at、POST /world/what-if）。

用固定 seed reactor + 计数 event_id 造独立 driver，保证 step() 可复现（事件溯源前提）；
再用 create_app(driver=...) 注入该 driver，端点与单元测的是同一条日志，断言才有意义。
"""

import random
import unittest

from fastapi.testclient import TestClient

from cyberhome.api import create_app
from cyberhome.driver import WorldDriver
from cyberhome.models import Trigger
from cyberhome.reactor import StubReactor
from cyberhome.seed import make_initial_world


def _fixed_reactor() -> StubReactor:
    """固定随机源：同一 (world, trigger) 重放必得同一结果（可复现红线）。"""
    return StubReactor(rng=random.Random(42))


def _counter_event_id():
    """计数 event_id 生成器：避免依赖模块级计数器的跨用例漂移。"""
    n = {"i": 0}

    def gen():
        n["i"] += 1
        return f"evt_test_{n['i']}"

    return gen


class TestHistory(unittest.TestCase):
    def _driver(self) -> WorldDriver:
        """每个用例一个独立内存 driver（不挂 persistence），状态互不污染。"""
        return WorldDriver(
            make_initial_world(now=1_000_000),
            reactor=_fixed_reactor(),
            event_id=_counter_event_id(),
        )

    def _client(self, driver: WorldDriver) -> TestClient:
        """把独立 driver 注入 app：端点与单元测共用同一条溯源日志。"""
        return TestClient(create_app(driver))

    # ---- 15.1 GET /world/events：返回事件时间线 ----
    def test_events_timeline_grows_with_dispatch(self):
        driver = self._driver()
        client = self._client(driver)

        # 初始时间线 = 创世 opening 事件（seed 有 5 条）
        initial = client.get("/world/events").json()["events"]
        self.assertEqual(len(initial), len(make_initial_world().events))
        self.assertTrue(all(e["source"] == "opening" for e in initial))

        # 投一个 oracle + 移动一次，时间线应增长
        driver.dispatch(Trigger(kind="oracle", text="要下雨了", ts=2_000_000))
        driver.dispatch(Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=3_000_000))

        after = client.get("/world/events").json()["events"]
        self.assertGreater(len(after), len(initial))  # 严格增长（≥ 初始 + 新增）
        # 时间线正序（旧→新），含 oracle 与 move 两种来源
        sources = {e["source"] for e in after}
        self.assertIn("oracle", sources)
        self.assertIn("move", sources)

    def test_events_until_seq_truncates(self):
        """until_seq 只回放到第 seq 步（含）——回放语义的边界。"""
        driver = self._driver()
        client = self._client(driver)
        driver.dispatch(Trigger(kind="oracle", text="第一步", ts=2_000_000))
        driver.dispatch(Trigger(kind="oracle", text="第二步", ts=3_000_000))

        only_first = client.get("/world/events", params={"until_seq": 0}).json()["events"]
        full = client.get("/world/events").json()["events"]
        # 到第 0 步为止的事件数 < 全量（第二步的事件被截断）
        self.assertLess(len(only_first), len(full))
        # 端点与原语一致
        self.assertEqual(len(only_first), len(driver.store.replay_events(until_seq=0)))

    # ---- 15.2 GET /world/at?seq=：得到对应时刻 world 快照 ----
    def test_world_at_returns_past_snapshot(self):
        driver = self._driver()
        client = self._client(driver)

        # 第 0 步：老爸移入厨房
        driver.dispatch(Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=2_000_000))
        # 第 1 步：老爸再移入卧室（覆盖上一步位置）
        driver.dispatch(Trigger(kind="move", agentId="agent_dad", roomId="room_bedroom", ts=3_000_000))

        # 当前世界：老爸在卧室
        now_dad = next(a for a in client.get("/world").json()["agents"] if a["id"] == "agent_dad")
        self.assertEqual(now_dad["roomId"], "room_bedroom")

        # 回溯到第 0 步之后：老爸应仍在厨房（那一刻的快照，不受后续步影响）
        at0 = client.get("/world/at", params={"seq": 0}).json()
        dad_at0 = next(a for a in at0["agents"] if a["id"] == "agent_dad")
        self.assertEqual(dad_at0["roomId"], "room_kitchen")

        # 回溯到创世（seq=-1）：老爸在初始的客厅
        creation = client.get("/world/at", params={"seq": -1}).json()
        dad_creation = next(a for a in creation["agents"] if a["id"] == "agent_dad")
        self.assertEqual(dad_creation["roomId"], "room_living")

        # 端点与原语一致（seq=None → 最新）
        self.assertEqual(
            client.get("/world/at").json()["clockLabel"],
            driver.store.world_at().clockLabel,
        )

    def test_world_at_seq_out_of_range_is_latest(self):
        """seq 越界不报错：折叠到不超过 seq 的所有步，等价于最新。"""
        driver = self._driver()
        client = self._client(driver)
        driver.dispatch(Trigger(kind="oracle", text="一句", ts=2_000_000))

        big = client.get("/world/at", params={"seq": 999}).json()
        latest = client.get("/world").json()
        self.assertEqual(big, latest)

    def test_world_at_rejects_non_int_seq(self):
        """seq 非数字由 FastAPI query 类型校验挡下（422），不 500。"""
        client = self._client(self._driver())
        self.assertEqual(client.get("/world/at", params={"seq": "abc"}).status_code, 422)

    # ---- 15.3 POST /world/what-if：只读预演，绝不改主线 ----
    def test_what_if_does_not_mutate_mainline(self):
        driver = self._driver()
        client = self._client(driver)
        driver.dispatch(Trigger(kind="oracle", text="奠基", ts=2_000_000))

        # 预演前记录主线状态：日志长度 + 当前世界快照（dataclass 级）+ 端点输出（HTTP 级）
        log_len_before = len(driver.store)
        world_before = driver.store.world_at()  # 当前最新（等于 driver.world）
        world_json_before = client.get("/world").json()

        resp = client.post("/world/what-if", json={"kind": "oracle", "text": "假如天塌了"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # 主线绝不变：日志长度不变、当前世界逐字不变（fork_readonly 不写 record）
        self.assertEqual(len(driver.store), log_len_before)
        self.assertEqual(driver.store.world_at(), world_before)  # dataclass 级相等（最强）
        self.assertEqual(client.get("/world").json(), world_json_before)  # 端点输出前后一致

        # 但返回的假想世界反映了预演 trigger 的效果：多出「神谕已投放」事件
        self.assertIn("events", data)
        self.assertIn("world", data)
        self.assertGreater(len(data["events"]), 0)
        self.assertEqual(data["events"][0]["source"], "oracle")
        # 假想世界事件数 > 预演前主线事件数（预演把新事件折进了假想世界）
        self.assertGreater(len(data["world"]["events"]), len(world_before.events))

    def test_what_if_from_past_seq(self):
        """what-if 可从历史某 seq 岔开：预演基于那一刻的世界，而非最新。"""
        driver = self._driver()
        client = self._client(driver)
        driver.dispatch(Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=2_000_000))

        log_len_before = len(driver.store)
        # 从创世（seq=-1）预演一次移动：假想世界里老爸到卧室，但主线不受影响
        resp = client.post(
            "/world/what-if",
            json={"kind": "move", "agentId": "agent_dad", "roomId": "room_bedroom", "seq": -1},
        )
        self.assertEqual(resp.status_code, 200)
        hypo_dad = next(a for a in resp.json()["world"]["agents"] if a["id"] == "agent_dad")
        self.assertEqual(hypo_dad["roomId"], "room_bedroom")  # 假想：从创世岔开去卧室

        # 主线不变：日志长度不变，且真实老爸仍在厨房（第 0 步的结果）
        self.assertEqual(len(driver.store), log_len_before)
        real_dad = next(a for a in client.get("/world").json()["agents"] if a["id"] == "agent_dad")
        self.assertEqual(real_dad["roomId"], "room_kitchen")

    def test_what_if_unit_level_fork_readonly(self):
        """单元层：直接测 store.fork_readonly 不写日志（端点只是它的薄封装）。"""
        driver = self._driver()
        driver.dispatch(Trigger(kind="oracle", text="奠基", ts=2_000_000))
        len_before = len(driver.store)

        events, patch, hypothetical = driver.store.fork_readonly(
            at_seq=None,
            trigger=Trigger(kind="oracle", text="假设", ts=9_000_000),
            reactor=_fixed_reactor(),
            event_id=_counter_event_id(),
            now=9_000_000,
        )
        # 不写日志（A 档「推演完即弃」）
        self.assertEqual(len(driver.store), len_before)
        # 预演有产出：至少一条 oracle 事件，假想世界比基线多事件
        self.assertGreater(len(events), 0)
        self.assertGreater(len(hypothetical.events), len(driver.store.world_at().events))


if __name__ == "__main__":
    unittest.main()
