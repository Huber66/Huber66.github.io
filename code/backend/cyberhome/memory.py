"""记忆层 —— MemoryStore 抽象接口 + 内存 stub + Mem0/Chroma 早验（决策 29/31/34）。

三件事在这里定死：
1. search() 接口**必须**带 top_k / token_cap（决策 34 的硬约束）——
   ADD-only 存储只增不减，检索裁剪不是变轻、是更重要。接口签名先立住，
   免得未来切 Mem0 时才发现漏了上下文预算这个一等公民。
2. persona 锚**不进**记忆层（决策 34）——它钉在 prompt 里、永不被检索裁剪，
   所以这里根本没有写 persona 的入口。灵魂走锚，事实走记忆。
3. 双重预算裁剪抽成模块级 `_budget_clip`（先 top_k 截断、再 token_cap 累计封顶），
   InMemoryStore 与 Mem0MemoryStore **共用同一份**——换后端时检索语义零漂移。

本文件含三套实现：`InMemoryStore`（零依赖 stub，跑闭环/写测试用）、
`SQLiteMemoryStore`（ADD-only 单表持久化，F09 落地的 fallback 主力）与
`Mem0MemoryStore`（Mem0 OSS + Chroma 真向量后端，F13 早验）。三者**外部契约完全一致**
（add/search 同签名、共用 `_naive_rank` 排序与 `_budget_clip` 预算裁剪），换后端零漂移。
`build_memory_store()` 工厂按 env 选择，失败逐级优雅回退（Mem0 → SQLite → InMemory），
绝不阻塞主线。
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

logger = logging.getLogger("cyberhome.memory")

# Chroma 本地 persist 目录（对齐 persistence.py 的 ./data/ 约定）与默认 collection 名。
DEFAULT_CHROMA_PATH = "./data/chroma"
DEFAULT_COLLECTION = "cyberhome_memories"

# SQLite 记忆库默认路径（与 persistence.DEFAULT_DB_PATH 同 ./data/ 目录，独立文件避免混表）。
DEFAULT_MEMORY_DB_PATH = "./data/memory.db"


@dataclass(frozen=True)
class MemoryItem:
    """一条记忆。ADD-only：写入后不改不删（决策 29）。"""

    id: str
    namespace: str   # scoping：经历层=agent:{id}；关系层=rel:{id}:{target}
    text: str
    ts: int          # 写入时刻，检索端时序推理靠它取「当前」
    importance: float = 0.5


class MemoryStore(ABC):
    """记忆存储抽象。默认实现 Mem0 OSS（F13/步骤 7），可退回 InMemoryStore/SQLite。"""

    @abstractmethod
    def add(self, namespace: str, text: str, ts: int, importance: float = 0.5) -> MemoryItem:
        """追加一条记忆（ADD-only，绝不 update/delete）。"""

    @abstractmethod
    def search(
        self,
        namespace: str,
        query: str,
        *,
        top_k: int,
        token_cap: int,
    ) -> list[MemoryItem]:
        """检索并**裁剪进上下文预算**（决策 34）。

        top_k / token_cap 是**必填**，不是可选——检索同时背两个活：
        选时序上「当前」的事实 + 压进有限上下文窗口。
        """


class InMemoryStore(MemoryStore):
    """内存 stub。检索用朴素子串匹配 + 时序倒排 + 双重预算裁剪。

    够跑通涌现闭环、够写测试证明「接口带预算」这件事落地了。真语义检索留给 Mem0。
    """

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []
        self._seq = 0

    def add(self, namespace: str, text: str, ts: int, importance: float = 0.5) -> MemoryItem:
        self._seq += 1
        item = MemoryItem(id=f"mem_{self._seq}", namespace=namespace, text=text, ts=ts, importance=importance)
        self._items.append(item)  # 只追加
        return item

    def search(
        self,
        namespace: str,
        query: str,
        *,
        top_k: int,
        token_cap: int,
    ) -> list[MemoryItem]:
        # 1. 命名空间过滤
        pool = [m for m in self._items if m.namespace == namespace]
        # 2. 朴素相关性排序：与 SQLiteMemoryStore 共用同一份 _naive_rank（换后端零漂移）
        ranked = _naive_rank(pool, query)
        # 3. 双重预算裁剪：与 Mem0MemoryStore 共用同一份逻辑（决策 34）
        return _budget_clip(ranked, top_k=top_k, token_cap=token_cap)


def _naive_rank(pool: list[MemoryItem], query: str) -> list[MemoryItem]:
    """朴素相关性排序（stub 语义，Mem0 换语义检索）：子串命中数 > 新近度 ts > importance。

    抽成模块级函数，让 InMemoryStore 与 SQLiteMemoryStore **共用同一份**排序——
    SQLite 端只负责按 namespace 取出候选（DB 不掺排序），排序仍走这里，
    保证「换后端检索语义零漂移」（红线）。入参 pool 已按 namespace 过滤。
    """
    terms = [t for t in query if t.strip()]

    def score(m: MemoryItem) -> tuple:
        hits = sum(1 for t in terms if t in m.text)
        # 时序推理雏形：命中数 > 新近度 > importance，取「当前最相关」
        return (hits, m.ts, m.importance)

    return sorted(pool, key=score, reverse=True)


def _approx_tokens(text: str) -> int:
    """粗估 token 数（stub：中文约 1 字 1 token）。切真 tokenizer 时替换。"""
    return len(text)


def _budget_clip(ranked: list[MemoryItem], *, top_k: int, token_cap: int) -> list[MemoryItem]:
    """双重预算裁剪（决策 34）：先 top_k 截断，再按 token_cap 用 _approx_tokens 累计封顶。

    抽成模块级函数，让 InMemoryStore 与 Mem0MemoryStore 共用**同一份**裁剪——
    这样换向量后端时，「检索背上下文预算」的外部语义完全一致，F06/F09/F10 换实现零成本。
    入参 ranked 已按相关性倒序（子串命中数 / 向量相似度），本函数只负责预算封顶。
    """
    out: list[MemoryItem] = []
    budget = token_cap
    for m in ranked[:top_k]:
        cost = _approx_tokens(m.text)
        if cost > budget:
            break
        out.append(m)
        budget -= cost
    return out


def resolve_chroma_path(env: Mapping[str, str] | None = None) -> str:
    """解析 Chroma persist 目录：env `CYBERHOME_CHROMA_PATH` 优先，缺省 `./data/chroma`。

    绝对化并建目录（对齐 persistence.resolve_db_path 的「env 可注入 + 绝对化 + 建目录」
    风格；env 可注入让测试不必污染 os.environ）。Chroma path 是目录不是文件，故 mkdir 自身。
    """
    src = os.environ if env is None else env
    raw = (src.get("CYBERHOME_CHROMA_PATH") or "").strip() or DEFAULT_CHROMA_PATH
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_memory_db_path(env: Mapping[str, str] | None = None) -> str:
    """解析记忆库 DB 路径：env `CYBERHOME_MEMORY_DB_PATH` 优先，缺省 `./data/memory.db`。

    与 persistence.resolve_db_path **同款**（env 可注入 + 绝对化 + 建父目录，`:memory:`
    例外不落盘）——两处刻意不共用一个函数：记忆库与世界库是**独立文件**（不同 env、
    不同缺省），共用会把两个演进方向耦死。风格对齐、职责分开，是长期主义的取舍。
    """
    src = os.environ if env is None else env
    raw = (src.get("CYBERHOME_MEMORY_DB_PATH") or "").strip() or DEFAULT_MEMORY_DB_PATH
    if raw == ":memory:":
        return raw
    path = Path(raw).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


# ---- 建表 DDL：单表 ADD-only。IF NOT EXISTS 幂等，无迁移框架（MVP 不过度设计）----
# id 由 rowid 派生（AUTOINCREMENT 保证跨删无重用，但本表永不 DELETE，仅为语义清晰）。
# ns+ts 建索引：search 先按 namespace 过滤、时序推理靠 ts，命中检索热路径。
_MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace  TEXT NOT NULL,
    text       TEXT NOT NULL,
    ts         INTEGER NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5
);
CREATE INDEX IF NOT EXISTS idx_memories_ns_ts ON memories (namespace, ts);
"""


class SQLiteMemoryStore(MemoryStore):
    """sqlite3 记忆后端（ADD-only，F09）。与 InMemoryStore 外部契约完全一致。

    单表 `memories`，**只 INSERT，永不 UPDATE/DELETE**（决策 29 红线在存储层落地）：
    矛盾/新旧交给检索端时序推理消解（`_naive_rank` 的 ts 维度），写入端一律追加。

    检索分工（守「换后端零漂移」红线）：SQL 只做 namespace 过滤 + 取出候选，
    相关性排序走与 InMemoryStore **共用**的模块级 `_naive_rank`，预算裁剪走**共用**的
    `_budget_clip`——DB 不掺排序/裁剪语义，故 InMemory 与 SQLite 的 search 结果逐位一致。

    连接管理对齐 SQLitePersistenceStore：单连接（check_same_thread=False），让
    `:memory:` DB 跨调用存活；单用户单进程 MVP 不做池化（不过度设计红线）。
    """

    def __init__(self, db_path: str | None = None, *, env: Mapping[str, str] | None = None) -> None:
        self._path = db_path or resolve_memory_db_path(env)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_MEMORY_SCHEMA)
        self._conn.commit()

    @property
    def path(self) -> str:
        return self._path

    def add(self, namespace: str, text: str, ts: int, importance: float = 0.5) -> MemoryItem:
        # ADD-only：只 INSERT。id 用自增 rowid 派生（对齐 InMemory 的 mem_{seq} 语义）。
        cur = self._conn.execute(
            "INSERT INTO memories (namespace, text, ts, importance) VALUES (?, ?, ?, ?)",
            (namespace, text, ts, importance),
        )
        self._conn.commit()
        return MemoryItem(
            id=f"mem_{cur.lastrowid}", namespace=namespace, text=text, ts=ts, importance=importance
        )

    def search(
        self,
        namespace: str,
        query: str,
        *,
        top_k: int,
        token_cap: int,
    ) -> list[MemoryItem]:
        # 1. 命名空间过滤：DB 只负责取出候选，不掺排序（排序交给共用 _naive_rank）
        rows = self._conn.execute(
            "SELECT id, namespace, text, ts, importance FROM memories WHERE namespace = ?",
            (namespace,),
        ).fetchall()
        pool = [_row_to_item(r) for r in rows]
        # 2. 朴素相关性排序：与 InMemoryStore 共用同一份（零漂移）
        ranked = _naive_rank(pool, query)
        # 3. 双重预算裁剪：与 InMemory/Mem0 共用同一份逻辑（决策 34）
        return _budget_clip(ranked, top_k=top_k, token_cap=token_cap)

    def close(self) -> None:
        """关闭连接。测试 tearDown 用；生产随进程退出。重复调用安全。"""
        self._conn.close()


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    """sqlite3.Row → MemoryItem（id 派生自自增主键，对齐 InMemory 的 mem_{seq}）。"""
    return MemoryItem(
        id=f"mem_{row['id']}",
        namespace=row["namespace"],
        text=row["text"],
        ts=row["ts"],
        importance=row["importance"],
    )


def _mem0_available() -> bool:
    """轻量探测 mem0 + chromadb 是否可 import（供测试 skipUnless / 工厂自动探测）。

    只用 find_spec 查是否可导入，**不真 import、不实例化、不联网**——mem0 import 较重，
    且 embedder/网络类运行时失败交给 build_memory_store 的 try 兜底回退。
    """
    import importlib.util

    return (
        importlib.util.find_spec("mem0") is not None
        and importlib.util.find_spec("chromadb") is not None
    )


class Mem0MemoryStore(MemoryStore):
    """Mem0 OSS + Chroma 真向量后端的 MemoryStore 实现（F13 早验，决策 29/31/34）。

    与 InMemoryStore 的**外部契约完全一致**（add/search 同签名、同预算裁剪语义），
    故 F06/F09/F10 从 stub 换到这里零成本。

    —— 本地启动说明（随代码走，不易漂移）——
    依赖（backend/requirements.txt）：`mem0ai`（记忆层）+ `chromadb`（本地向量库）。
      未安装时：`import cyberhome.memory` 不受影响（本类惰性 import mem0），
      工厂 build_memory_store() 自动回退 InMemoryStore；契约测试对本类 skip。
    env：
      CYBERHOME_CHROMA_PATH  Chroma persist 目录，默认 ./data/chroma（绝对化+自建）。
      CYBERHOME_MEMORY_BACKEND  工厂后端开关：mem0 / memory（默认空=自动探测）。
      OPENAI_API_KEY  Mem0 OSS 默认 embedder 走 OpenAI，需要 embedding key
                      （或自行改配置换本地 ollama embedder）。无 key 时 add/search
                      会在运行时报错——由工厂/测试兜底回退或 skip，不阻塞主线。
    设计红线：
      · vector-only —— 只配 Chroma 向量存储，**不开 Neo4j / graph store**（不引图数据库）。
      · ADD-only —— add() 用 `infer=False` 存原文，跳过 Mem0 默认的 LLM 抽取/合并/删矛盾
        （infer=True 会改写、去重、删除旧记忆，直接违反决策 29 的「只增不改不删」）。
    """

    def __init__(
        self,
        *,
        collection: str = DEFAULT_COLLECTION,
        persist_path: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        # 惰性 import：没装 mem0 时，导入本模块本身不报错（早验的正确形态）。
        from mem0 import Memory

        path = persist_path or resolve_chroma_path(env)
        # vector-only：只配 Chroma，不配 graph_store（红线：不引 Neo4j）。
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {"collection_name": collection, "path": path},
            }
        }
        self._mem = Memory.from_config(config)
        self._collection = collection
        self._path = path
        self._seq = 0  # Mem0 取不到 id 时的兜底自增序号

    def add(self, namespace: str, text: str, ts: int, importance: float = 0.5) -> MemoryItem:
        self._seq += 1
        metadata = {"ts": ts, "importance": importance}
        # ADD-only：infer=False 存原文，不走 LLM 抽取/合并/删矛盾（决策 29 红线）。
        # namespace 作为 Mem0 的 user_id 承担隔离；ts/importance 塞 metadata 供回读。
        raw = self._mem.add(
            [{"role": "user", "content": text}],
            user_id=namespace,
            metadata=metadata,
            infer=False,
        )
        mem_id = self._first_id(raw) or f"mem_{self._seq}"
        # 返回原文构造的 MemoryItem（infer=False 存的就是原文，外部契约与 InMemory 一致）。
        return MemoryItem(id=str(mem_id), namespace=namespace, text=text, ts=ts, importance=importance)

    def search(
        self,
        namespace: str,
        query: str,
        *,
        top_k: int,
        token_cap: int,
    ) -> list[MemoryItem]:
        # 语义检索：user_id=namespace 保证只召回该命名空间（隔离靠它）；limit 对齐 top_k。
        raw = self._mem.search(query, user_id=namespace, limit=top_k)
        rows = self._rows(raw)
        ranked = [self._to_item(r, namespace) for r in rows]
        # 复用与 InMemoryStore 完全一致的双重预算裁剪（决策 34）。
        return _budget_clip(ranked, top_k=top_k, token_cap=token_cap)

    # ---- Mem0 返回格式适配（兼容 dict{"results":[...]} 与旧版直接 list）----
    @staticmethod
    def _rows(raw) -> list:
        if isinstance(raw, dict):
            return raw.get("results") or []
        return raw or []

    @classmethod
    def _first_id(cls, raw):
        rows = cls._rows(raw)
        if rows and isinstance(rows[0], dict):
            return rows[0].get("id")
        return None

    @staticmethod
    def _to_item(r: dict, namespace: str) -> MemoryItem:
        meta = r.get("metadata") or {}
        text = r.get("memory") or r.get("text") or ""
        raw_ts = meta.get("ts", 0)
        try:
            ts = int(raw_ts) if raw_ts is not None else 0
        except (TypeError, ValueError):
            ts = 0
        try:
            importance = float(meta.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        # namespace 用查询传入值（已按 user_id 过滤），与 InMemory 语义对齐。
        return MemoryItem(id=str(r.get("id") or ""), namespace=namespace, text=text, ts=ts, importance=importance)


def _fallback_store(env: Mapping[str, str]) -> MemoryStore:
    """最终回退目标（F09 落地后固定为 SQLite，决策 29/plan §F09/F13）。

    SQLiteMemoryStore 初始化理论上不该失败（纯本地文件），但仍 try 兜底到
    InMemoryStore 做最终保险——守「工厂永不抛异常」红线，任何环境都返回可用 store。
    """
    try:
        return SQLiteMemoryStore(env=env)
    except Exception as e:  # 磁盘/权限等极端情况 —— 最终兜底内存实现，绝不抛
        logger.warning(
            "SQLiteMemoryStore 初始化失败，回退 InMemoryStore：%s: %s",
            type(e).__name__, str(e)[:200],
        )
        return InMemoryStore()


def build_memory_store(*, env: Mapping[str, str] | None = None) -> MemoryStore:
    """记忆后端工厂：按 env 选实现；ImportError / 初始化失败一律**逐级优雅回退**。

    永不抛异常、永不阻塞主线；回退原因记 warning 脱敏日志。
    回退链（F09 落地后）：**Mem0 → SQLite → InMemory**——回退目标固定为持久化的
    SQLiteMemoryStore（跨交互记忆连续性），SQLite 再失败才最终兜底 InMemoryStore。

    env 开关 CYBERHOME_MEMORY_BACKEND（优先级）：
      · "memory" —— 直接 InMemoryStore（显式关闭持久化，测试/临时用，不落盘）。
      · "sqlite" —— 直接 SQLiteMemoryStore（F09 fallback 主力，失败兜底 InMemory）。
      · "mem0"   —— 强制试 Mem0+Chroma，失败回退 SQLite（Mem0 早验通过则设它为默认）。
      · ""（默认/未设）—— 自动探测：能 import mem0/chromadb 就试 mem0（失败回退 SQLite），
                         否则直接走 SQLite。

    注：本工厂只提供「接口就位」，**不**接进 driver/api（那是 F06/F09 的 wiring）。
    """
    src = os.environ if env is None else env
    backend = (src.get("CYBERHOME_MEMORY_BACKEND") or "").strip().lower()

    if backend == "memory":
        return InMemoryStore()
    if backend == "sqlite":
        return _fallback_store(src)

    want_mem0 = backend == "mem0" or (backend == "" and _mem0_available())
    if want_mem0:
        try:
            return Mem0MemoryStore(env=src)
        except Exception as e:  # ImportError / 配置 / 初始化失败 —— 回退 SQLite（非 InMemory）
            logger.warning(
                "Mem0MemoryStore 初始化失败，回退 SQLiteMemoryStore：%s: %s",
                type(e).__name__, str(e)[:200],
            )
    return _fallback_store(src)
