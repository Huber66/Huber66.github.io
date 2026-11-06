"""SQLite 持久化测试（F03 · 4.4 验收）。

标准库 unittest + 临时文件 DB（tearDown 删除），与 test_engine/test_api 同一套运行：
    cd backend && python3 -m unittest discover tests

覆盖：创建（无快照→seed 建库）、追加（dispatch 后 event_log 增长 + 快照更新）、
恢复（新建 driver 从同库读回，角色位置/事件流一致）、reset（清库回到无快照）。
核心验收（决策 32）：从 event_log 读全部日志、从 initial 依次折叠 apply_patch，
结果必与持久化的当前快照一致——证明事件溯源可回放到当前 world。

用固定 seed reactor + 计数 event_id，保证 step() 可复现（事件溯源前提）。
"""

import os
import random
import shutil
import tempfile
import unittest

from cyberhome import (
    SQLitePersistenceStore, StubReactor, Trigger, WorldDriver,
    apply_patch, make_initial_world,
)
from cyberhome.persistence import resolve_db_path

WORLD_ID = "world_home"


def _fixed_reactor() -> StubReactor:
    return StubReactor(rng=random.Random(42))


def _counter_event_id():
    n = {"i": 0}
    def gen():
        n["i"] += 1
        return f"evt_test_{n['i']}"
    return gen


class TestPersistence(unittest.TestCase):
    def setUp(self):
        # 临时文件 DB：真实文件路径（非 :memory:）才能验证「新建 driver 从同一 DB 读回」，
        # 这是跨进程/重启恢复的忠实代理。tearDown 删文件，测试间零残留。
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._store = SQLitePersistenceStore(self._db_path)

    def tearDown(self):
        self._store.close()
        if os.path.exists(self._db_path):
            os.remove(self._db_path)

    def _fresh_store(self) -> SQLitePersistenceStore:
        """开一个指向同一文件的新 store，模拟「重启后另起进程」。用例负责 close。"""
        return SQLitePersistenceStore(self._db_path)

    # ---- 创建：无快照 → bootstrap 用 seed 建库 ----
    def test_bootstrap_seeds_when_empty(self):
        self.assertIsNone(self._store.load_snapshot(WORLD_ID))  # 起初空
        driver = WorldDriver.bootstrap(
            self._store, reactor=_fixed_reactor(), event_id=_counter_event_id(), now=1_000_000,
        )
        # 世界已创世且落库
        self.assertEqual(driver.world.id, WORLD_ID)
        restored = self._store.load_snapshot(WORLD_ID)
        self.assertIsNotNone(restored)
        self.assertEqual(restored, driver.world)
        # last_seen 已写入（供 F10 补演）
        self.assertEqual(self._store.get_last_seen(WORLD_ID), 1_000_000)
        # 未 dispatch 前日志为空
        self.assertEqual(len(self._store.replay_log(WORLD_ID)), 0)

    # ---- 追加：每次 dispatch 后 event_log 增长 + 快照更新 ----
    def test_dispatch_appends_log_and_updates_snapshot(self):
        driver = WorldDriver.bootstrap(
            self._store, reactor=_fixed_reactor(), event_id=_counter_event_id(), now=1_000_000,
        )
        self.assertEqual(len(self._store.replay_log(WORLD_ID)), 0)

        driver.dispatch(Trigger(kind="oracle", text="要下雨了", ts=2_000_000))
        self.assertEqual(len(self._store.replay_log(WORLD_ID)), 1)

        driver.dispatch(Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=3_000_000))
        entries = self._store.replay_log(WORLD_ID)
        self.assertEqual(len(entries), 2)  # ADD-only：只增不减
        # seq 从 0 连续自增
        self.assertEqual([e.seq for e in entries], [0, 1])
        # 快照已随最后一次 dispatch 更新：老爸到厨房
        snap = self._store.load_snapshot(WORLD_ID)
        dad = next(a for a in snap.agents if a.id == "agent_dad")
        self.assertEqual(dad.roomId, "room_kitchen")
        # last_seen 推进到最后一次 trigger.ts
        self.assertEqual(self._store.get_last_seen(WORLD_ID), 3_000_000)

    # ---- 恢复：新建 driver 从同一 DB 读回，角色位置 + 事件流一致 ----
    def test_recover_from_same_db(self):
        d1 = WorldDriver.bootstrap(
            self._store, reactor=_fixed_reactor(), event_id=_counter_event_id(), now=1_000_000,
        )
        d1.dispatch(Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=2_000_000))
        dad_room_before = next(a for a in d1.world.agents if a.id == "agent_dad").roomId
        events_before = [e.id for e in d1.world.events]

        # 另起 store + driver（模拟重启进程），从同一文件恢复
        store2 = self._fresh_store()
        try:
            d2 = WorldDriver.bootstrap(store2, reactor=_fixed_reactor(), event_id=_counter_event_id())
            # 角色位置一致
            dad_room_after = next(a for a in d2.world.agents if a.id == "agent_dad").roomId
            self.assertEqual(dad_room_after, dad_room_before)
            self.assertEqual(dad_room_after, "room_kitchen")
            # 事件流一致（move 事件已随快照持久化）
            self.assertEqual([e.id for e in d2.world.events], events_before)
            # 恢复后不重复创世：快照即恢复前的世界
            self.assertEqual(d2.world, d1.world)
        finally:
            store2.close()

    # ---- 核心验收：事件溯源可回放到当前 world（决策 32）----
    def test_event_log_replays_to_current_snapshot(self):
        driver = WorldDriver.bootstrap(
            self._store, reactor=_fixed_reactor(), event_id=_counter_event_id(), now=1_000_000,
        )
        driver.dispatch(Trigger(kind="oracle", text="要下雨了", ts=2_000_000))
        driver.dispatch(Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=3_000_000))
        driver.dispatch(Trigger(kind="move", agentId="agent_mom", roomId="room_bedroom", ts=4_000_000))

        # 从创世 initial 依次折叠 event_log 的每条 (patch, events)。
        # 用 driver 的真实创世快照（store.initial），而非重新 make_initial_world——
        # 后者的 opening 事件 id 走模块级计数器，二次调用会得到不同 id，非同一创世。
        folded = driver.store.initial
        for entry in self._store.replay_log(WORLD_ID):
            folded = apply_patch(folded, entry.patch, entry.events)

        # 折叠结果必与持久化的当前快照逐字一致——F03 的核心：日志能重演出当前世界
        snapshot = self._store.load_snapshot(WORLD_ID)
        self.assertEqual(folded, snapshot)
        self.assertEqual(folded, driver.world)
        # 具体断言：两次移动都在折叠结果里体现
        self.assertEqual(next(a for a in folded.agents if a.id == "agent_dad").roomId, "room_kitchen")
        self.assertEqual(next(a for a in folded.agents if a.id == "agent_mom").roomId, "room_bedroom")

    # ---- reset：清库后回到无快照 ----
    def test_reset_clears_everything(self):
        driver = WorldDriver.bootstrap(
            self._store, reactor=_fixed_reactor(), event_id=_counter_event_id(), now=1_000_000,
        )
        driver.dispatch(Trigger(kind="oracle", text="要下雨了", ts=2_000_000))
        # 清库前有快照、有日志、有 last_seen
        self.assertIsNotNone(self._store.load_snapshot(WORLD_ID))
        self.assertGreater(len(self._store.replay_log(WORLD_ID)), 0)
        self.assertIsNotNone(self._store.get_last_seen(WORLD_ID))

        self._store.reset()

        # 清库后全空——回到「无快照」初态，bootstrap 会重新 seed
        self.assertIsNone(self._store.load_snapshot(WORLD_ID))
        self.assertEqual(len(self._store.replay_log(WORLD_ID)), 0)
        self.assertIsNone(self._store.get_last_seen(WORLD_ID))

    def test_reset_scoped_to_world_id(self):
        """带 world_id 的 reset 只清该世界，为多世界预留（MVP 单世界也应正确）。"""
        WorldDriver.bootstrap(
            self._store, reactor=_fixed_reactor(), event_id=_counter_event_id(), now=1_000_000,
        )
        self._store.reset("other_world")  # 清一个不存在的世界，不应误伤 world_home
        self.assertIsNotNone(self._store.load_snapshot(WORLD_ID))
        self._store.reset(WORLD_ID)
        self.assertIsNone(self._store.load_snapshot(WORLD_ID))

    # ---- 向后兼容：纯内存 driver（persistence=None）行为不变 ----
    def test_in_memory_driver_writes_nothing(self):
        """不挂 persistence 的 driver 绝不碰 DB——守 F03 前行为完全一致的红线。"""
        driver = WorldDriver(
            make_initial_world(now=1_000_000), reactor=_fixed_reactor(), event_id=_counter_event_id(),
        )
        driver.dispatch(Trigger(kind="oracle", text="要下雨了", ts=2_000_000))
        # 独立空库不因内存 driver 的 dispatch 而产生任何行
        self.assertIsNone(self._store.load_snapshot(WORLD_ID))
        self.assertEqual(len(self._store.replay_log(WORLD_ID)), 0)


class TestResolveDbPath(unittest.TestCase):
    """路径解析测试。因 resolve_db_path 会创建父目录（相对路径基于 cwd），
    整类在临时 cwd 下运行——任何 `./data/` 落进 tempdir，tearDown 一并清除，
    绝不在仓库里留下 backend/data/ 空目录（测试不该有文件系统副作用）。"""

    def setUp(self):
        self._prev_cwd = os.getcwd()
        self._tmp = tempfile.mkdtemp()
        os.chdir(self._tmp)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_env_override(self):
        # 用一个不经符号链接规范化的目录做断言——resolve() 会把 macOS 的 /tmp 规范成
        # /private/tmp，故不硬编码前缀，只验证 basename 与绝对性（跨平台稳定）。
        p = resolve_db_path(env={"CYBERHOME_DB_PATH": "./data/custom_name.db"})
        self.assertTrue(os.path.isabs(p))
        self.assertTrue(p.endswith("custom_name.db"))

    def test_memory_path_passthrough(self):
        """:memory: 不落盘、不建目录，原样透传。"""
        self.assertEqual(resolve_db_path(env={"CYBERHOME_DB_PATH": ":memory:"}), ":memory:")

    def test_default_is_absolute(self):
        """缺省路径应绝对化（父目录随之创建），避免相对 cwd 漂移。"""
        p = resolve_db_path(env={})
        self.assertTrue(os.path.isabs(p))
        self.assertTrue(p.endswith("cyberhome.db"))


if __name__ == "__main__":
    unittest.main()
