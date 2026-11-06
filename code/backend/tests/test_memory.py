"""记忆层接口契约测试（F13 / 6.2·6.3，决策 29/34）。

一套共享断言（MemoryContractMixin）同时跑在 InMemoryStore 与 Mem0MemoryStore 上，
证明两个后端**外部契约完全一致**——这是「换后端零成本」的可执行证据。
Mem0 用例在未装 mem0/chromadb 时优雅 skip（不 fail），装了即真跑。

零/轻依赖、标准库 unittest（与 test_engine 同风格）。
运行：cd backend && python3 -m unittest discover tests
"""

import os
import tempfile
import unittest

from cyberhome.memory import (
    InMemoryStore,
    Mem0MemoryStore,
    MemoryItem,
    MemoryStore,
    SQLiteMemoryStore,
    _mem0_available,
    build_memory_store,
    resolve_memory_db_path,
)


class MemoryContractMixin:
    """MemoryStore 接口契约。子类实现 _make_store() 返回被测实现。

    这些断言对**任何** MemoryStore 实现都必须成立——所以放在 mixin 里共享，
    换后端时同一套断言复跑一遍即验证契约未破。
    """

    def _make_store(self) -> MemoryStore:  # 子类覆盖
        raise NotImplementedError

    def test_add_only_count_grows(self):
        """ADD-only：连续 add 多条后条数只增（决策 29）。"""
        store = self._make_store()
        for i in range(5):
            item = store.add(namespace="agent:dad", text=f"老爸经历第{i}件事", ts=1000 + i)
            self.assertIsInstance(item, MemoryItem)
        got = store.search("agent:dad", "老爸", top_k=100, token_cap=100_000)
        self.assertEqual(len(got), 5)

    def test_search_does_not_consume(self):
        """ADD-only：search 不「消费」记忆，多次检索结果稳定。"""
        store = self._make_store()
        for i in range(4):
            store.add(namespace="agent:dad", text=f"老爸记得第{i}件事", ts=2000 + i)
        first = store.search("agent:dad", "老爸", top_k=100, token_cap=100_000)
        second = store.search("agent:dad", "老爸", top_k=100, token_cap=100_000)
        self.assertEqual(len(first), len(second))
        self.assertEqual({m.id for m in first}, {m.id for m in second})

    def test_top_k_caps_result_count(self):
        """top_k：token_cap 放开时，返回条数 ≤ top_k。"""
        store = self._make_store()
        for i in range(8):
            store.add(namespace="agent:dad", text=f"老爸做了第{i}件事情", ts=3000 + i)
        got = store.search("agent:dad", "老爸", top_k=3, token_cap=100_000)
        self.assertLessEqual(len(got), 3)

    def test_token_cap_trims_hard(self):
        """token_cap：预算收紧到只够 1 条（甚至 0）时，返回条数 ≤ 1。"""
        store = self._make_store()
        for i in range(8):
            store.add(namespace="agent:dad", text=f"老爸记得很长很长很长很长的第{i}件事情", ts=4000 + i)
        tight = store.search("agent:dad", "老爸", top_k=100, token_cap=15)
        self.assertLessEqual(len(tight), 1)

    def test_namespace_isolation_agents(self):
        """namespace 隔离：agent:a 的记忆不出现在 agent:b 的检索里。"""
        store = self._make_store()
        store.add(namespace="agent:a", text="A 喜欢吃辣", ts=5000)
        store.add(namespace="agent:b", text="B 不吃辣", ts=5001)
        got_a = store.search("agent:a", "吃辣", top_k=100, token_cap=100_000)
        self.assertTrue(all(m.namespace == "agent:a" for m in got_a))
        self.assertTrue(all("B 不吃辣" not in m.text for m in got_a))

    def test_namespace_isolation_experience_vs_relation(self):
        """namespace 隔离：关系层 rel:a:b 与经历层 agent:a 分开互不串味。"""
        store = self._make_store()
        store.add(namespace="agent:a", text="A 自己的经历记忆", ts=6000)
        store.add(namespace="rel:a:b", text="A 对 B 的关系记忆", ts=6001)
        got_exp = store.search("agent:a", "记忆", top_k=100, token_cap=100_000)
        got_rel = store.search("rel:a:b", "记忆", top_k=100, token_cap=100_000)
        self.assertTrue(all(m.namespace == "agent:a" for m in got_exp))
        self.assertTrue(all(m.namespace == "rel:a:b" for m in got_rel))
        self.assertTrue(all("关系" not in m.text for m in got_exp))

    def test_spike_add_search_hit(self):
        """最小 spike（6.3）：存一条经历，按 namespace 检索命中，验证预算裁剪生效。"""
        store = self._make_store()
        store.add(namespace="agent:dad", text="我不吃香菜", ts=7000)
        hit = store.search("agent:dad", "香菜", top_k=5, token_cap=100_000)
        self.assertGreaterEqual(len(hit), 1)
        self.assertTrue(any("香菜" in m.text for m in hit))
        # 同一次 spike 顺带验证 token_cap 能裁到空（预算不足一条时）
        starved = store.search("agent:dad", "香菜", top_k=5, token_cap=1)
        self.assertEqual(len(starved), 0)

    def test_recency_orders_ties(self):
        """时序优先：同等相关（子串命中数相同）时，新记忆（ts 大）排在前（决策 29 检索时时序）。

        这是「换后端零漂移」的可执行断言之一——InMemory / SQLite 共用 _naive_rank，
        故此契约对两者必须同时成立。"""
        store = self._make_store()
        store.add(namespace="agent:dad", text="老爸曾住北京", ts=1000)
        store.add(namespace="agent:dad", text="老爸现居上海", ts=2000)  # 更新的事实
        got = store.search("agent:dad", "老爸", top_k=5, token_cap=100_000)
        self.assertEqual(len(got), 2)
        # 两条都命中「老爸」一次（命中数持平），时序推理让更新的「现居上海」排最前。
        self.assertEqual(got[0].ts, 2000)
        self.assertIn("上海", got[0].text)


class TestInMemoryStoreContract(MemoryContractMixin, unittest.TestCase):
    """InMemoryStore 必须真跑通全部契约（零依赖，不 skip）。"""

    def _make_store(self) -> MemoryStore:
        return InMemoryStore()


class TestSQLiteMemoryStoreContract(MemoryContractMixin, unittest.TestCase):
    """SQLiteMemoryStore 跑同一套契约（F09，不 skip，必须真跑通）。

    每个用例用独立临时 DB 文件隔离（tearDown 清理），测完不污染 ./data/。
    用真实文件而非 `:memory:`：既验证落盘路径，又能在下面单独测「跨连接持久化」。
    """

    def setUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db", prefix="cyberhome_mem_test_")
        os.close(fd)  # 只借文件名，连接由 SQLiteMemoryStore 自建
        self._stores: list[SQLiteMemoryStore] = []

    def tearDown(self):
        for s in self._stores:
            s.close()
        if os.path.exists(self._db_path):
            os.remove(self._db_path)

    def _make_store(self) -> MemoryStore:
        store = SQLiteMemoryStore(db_path=self._db_path)
        self._stores.append(store)
        return store

    def test_add_only_never_updates_or_deletes(self):
        """ADD-only 表级验证：连续 add 后底表行数只增（等于写入次数），无就地更新/删除。"""
        store = self._make_store()
        for i in range(6):
            store.add(namespace="agent:dad", text=f"经历{i}", ts=1000 + i)
        # 直连底表数行数（绕过 search 的裁剪），证明写了几条就有几条。
        count = store._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
        self.assertEqual(count, 6)

    def test_persists_across_connections(self):
        """持久化：一个 store 写入后关闭，新 store 连同一文件仍能检索到（跨进程记忆连续性）。"""
        writer = SQLiteMemoryStore(db_path=self._db_path)
        writer.add(namespace="agent:dad", text="我不吃香菜", ts=7000)
        writer.close()
        reader = SQLiteMemoryStore(db_path=self._db_path)
        self._stores.append(reader)
        got = reader.search("agent:dad", "香菜", top_k=5, token_cap=100_000)
        self.assertTrue(any("香菜" in m.text for m in got))

    def test_memory_backend_falls_back_when_dir_missing(self):
        """resolve_memory_db_path 自建父目录：指向不存在的深层目录也能落库（不崩）。"""
        import shutil

        base = tempfile.mkdtemp(prefix="cyberhome_mem_dir_")
        try:
            nested = os.path.join(base, "a", "b", "memory.db")  # a/b 尚不存在
            store = SQLiteMemoryStore(db_path=resolve_memory_db_path(env={"CYBERHOME_MEMORY_DB_PATH": nested}))
            store.add(namespace="agent:x", text="深层目录也能写", ts=1)
            got = store.search("agent:x", "深层", top_k=5, token_cap=100_000)
            store.close()
            self.assertTrue(got)
        finally:
            shutil.rmtree(base, ignore_errors=True)


@unittest.skipUnless(_mem0_available(), "mem0/chromadb 未安装，Mem0 契约用例跳过（早验：装了即跑）")
class TestMem0MemoryStoreContract(MemoryContractMixin, unittest.TestCase):
    """Mem0+Chroma 后端跑同一套契约。未装依赖时整类 skip，不阻塞主线。

    每个测试新建独立 collection（用测试名做后缀），避免跨用例/跨运行的向量库串味。
    """

    def _make_store(self) -> MemoryStore:
        import uuid

        return Mem0MemoryStore(collection=f"cyberhome_test_{uuid.uuid4().hex[:8]}")


class TestBuildMemoryStoreFactory(unittest.TestCase):
    """回退工厂 build_memory_store：env 开关 + 逐级优雅回退（Mem0→SQLite→InMemory，永不抛异常）。

    统一注入 `CYBERHOME_MEMORY_DB_PATH=:memory:`——工厂 F09 后回退目标是持久化 SQLite，
    若不注入会落盘 ./data/memory.db 污染仓库。`:memory:` 让 SQLite 分支不落盘。
    """

    MEM_ENV = {"CYBERHOME_MEMORY_DB_PATH": ":memory:"}

    def test_explicit_memory_backend(self):
        """CYBERHOME_MEMORY_BACKEND=memory → 显式 InMemoryStore（不落盘）。"""
        store = build_memory_store(env={"CYBERHOME_MEMORY_BACKEND": "memory", **self.MEM_ENV})
        self.assertIsInstance(store, InMemoryStore)

    def test_explicit_sqlite_backend(self):
        """CYBERHOME_MEMORY_BACKEND=sqlite → SQLiteMemoryStore（F09 fallback 主力）。"""
        store = build_memory_store(env={"CYBERHOME_MEMORY_BACKEND": "sqlite", **self.MEM_ENV})
        self.assertIsInstance(store, SQLiteMemoryStore)
        store.close()

    def test_default_autodetect_never_raises(self):
        """默认（未设 backend）自动探测：无论装没装依赖都返回可用 MemoryStore，绝不抛异常。"""
        store = build_memory_store(env=dict(self.MEM_ENV))
        self.assertIsInstance(store, MemoryStore)
        # 冒烟：拿到的 store 立即可用（add→search 闭环），验证「就位即可用」。
        store.add(namespace="agent:x", text="工厂产出的 store 能用", ts=1)
        got = store.search("agent:x", "能用", top_k=5, token_cap=100_000)
        self.assertGreaterEqual(len(got), 1)

    def test_force_mem0_falls_back_to_sqlite_when_absent(self):
        """CYBERHOME_MEMORY_BACKEND=mem0 但依赖缺失时，回退 SQLiteMemoryStore 而非崩溃（F09 后回退目标）。"""
        if _mem0_available():
            self.skipTest("mem0 已安装，无法验证缺失回退路径")
        store = build_memory_store(env={"CYBERHOME_MEMORY_BACKEND": "mem0", **self.MEM_ENV})
        self.assertIsInstance(store, SQLiteMemoryStore)
        store.close()


if __name__ == "__main__":
    unittest.main()
