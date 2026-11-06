"""SQLite 持久化 —— 世界快照 + 事件溯源日志 + last_seen（F03，决策 32/35）。

WHY 三张表而非一张：
- `world_snapshots` 存**当前** world 的整快照（world_id 主键，UPSERT）——这是
  **快照优化**：恢复时 O(1) 读回，不必每次启动从创世重放整条日志。
- `event_log` 是**不可变追加日志**（ADD-only，只 INSERT），决策 32 的落地。它不存
  world_before——中间态是「initial 折叠 patch」的派生物，存了反而冗余。给定 initial，
  用各步 (events, patch) 依序 apply_patch 即可重演出任意时刻（F03 核心验收）。
  注：MVP 按「三张表」约束，genesis 不单独持久化；跨重启的完整回放（从创世重演）
  待 F12 再评估是否落 genesis 行，本轮不做。
- `world_meta.last_seen` 供惰性补演（F10）算「离开了多久」。

WHY 全表带 world_id：MVP 单用户单世界（固定 'world_home'），但表结构预留多世界，
免得未来加世界时改 schema——一次性结构成本换后续自由度（长期主义）。

WHY 不用 ORM：单进程单世界 + 三张表 + 无迁移需求，sqlite3 足矣。引入 SQLAlchemy
是把「轻状态机用重框架表达」的同类过度设计，与项目「优雅极简」红线冲突。

序列化：world 复用 schemas.world_to_dict/from_dict（守前后端字段对齐红线）；
trigger/events/patch 均 frozen dataclass，用 dataclasses.asdict 出，就地对称重建入。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional

from .models import Trigger, World, WorldEvent, WorldPatch
from .schemas import (
    _agent_from_dict,
    _event_from_dict,
    _room_from_dict,
    world_from_dict,
    world_to_dict,
)

DEFAULT_DB_PATH = "./data/cyberhome.db"
DEFAULT_WORLD_ID = "world_home"


def resolve_db_path(env: Mapping[str, str] | None = None) -> str:
    """解析 DB 路径：env `CYBERHOME_DB_PATH` 优先，缺省 `./data/cyberhome.db`。

    绝对化并建父目录（`:memory:` 例外，不落盘）。env 可注入（对齐本项目
    「时间/随机/环境显式注入」的可测试风格），测试不必污染 os.environ。
    """
    src = os.environ if env is None else env
    raw = (src.get("CYBERHOME_DB_PATH") or "").strip() or DEFAULT_DB_PATH
    if raw == ":memory:":
        return raw
    path = Path(raw).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _now_ms() -> int:
    """当前毫秒时间戳，对齐前端 Date.now()。"""
    return int(time.time() * 1000)


# ---- event_log 一行的解析结果（LogEntry-like，但刻意不含 world_before）----
@dataclass(frozen=True)
class PersistedEntry:
    """持久化日志的一条。与 eventstore.LogEntry 同构，但**无 world_before**——

    事件溯源不必存中间 world：从创世 initial 依次 apply_patch(patch, events) 折叠，
    即可重演出任意时刻乃至当前快照（F03 核心验收）。存 world_before 是冗余，违反
    ADD-only 的精简。
    """

    seq: int
    trigger: Trigger
    events: tuple[WorldEvent, ...]
    patch: WorldPatch


# ---- trigger / patch 的对称反序列化（events 直接复用 schemas._event_from_dict）----
def _trigger_from_dict(d: dict) -> Trigger:
    return Trigger(
        kind=d["kind"],
        text=d.get("text"),
        agentId=d.get("agentId"),
        roomId=d.get("roomId"),
        ts=d.get("ts"),
        since=d.get("since"),  # F10 catchup 时间窗起点；旧日志无此键时 .get 得 None（向后兼容）
    )


def _patch_from_dict(d: dict) -> WorldPatch:
    """asdict(patch) 会把 added_agents/rooms 递归成 dict，这里重建回 dataclass。"""
    return WorldPatch(
        agent_updates=d.get("agent_updates") or {},
        added_agents=tuple(_agent_from_dict(a) for a in (d.get("added_agents") or ())),
        agents=tuple(_agent_from_dict(a) for a in d["agents"]) if d.get("agents") is not None else None,
        added_rooms=tuple(_room_from_dict(r) for r in (d.get("added_rooms") or ())),
        rooms=tuple(_room_from_dict(r) for r in d["rooms"]) if d.get("rooms") is not None else None,
        world_updates=d.get("world_updates") or {},
    )


class PersistenceStore(ABC):
    """持久化抽象：世界快照 + 事件溯源日志 + last_seen。

    抽象出接口是为了 driver 不绑死 SQLite——测试可注入内存实现，未来若换存储
    （如恢复走 Postgres）也只替实现、不动 driver（同 MemoryStore/Reactor 的隔离思路）。
    """

    @abstractmethod
    def load_snapshot(self, world_id: str) -> Optional[World]:
        """读回当前 world 快照；无则 None（bootstrap 据此决定恢复 or seed）。"""

    @abstractmethod
    def save_snapshot(self, world: World) -> None:
        """UPSERT 当前 world 整快照（world.id 作 world_id）。"""

    @abstractmethod
    def append_log(
        self,
        world_id: str,
        trigger: Trigger,
        events: tuple[WorldEvent, ...],
        patch: WorldPatch,
    ) -> int:
        """追加一条溯源日志（ADD-only）。seq 由存储自算（DB max+1），返回该 seq。

        WHY 不让调用方传 seq：跨重启后内存 EventStore 的 seq 会从 0 重来，若用它写
        DB 会与已有行撞主键。DB seq 以 DB 自身为事实源，才能保证 ADD-only 链跨进程连续。
        """

    @abstractmethod
    def get_last_seen(self, world_id: str) -> Optional[int]:
        """读 last_seen（惰性补演 F10 用）；无则 None。"""

    @abstractmethod
    def set_last_seen(self, world_id: str, ts: int) -> None:
        """写 last_seen（UPSERT）。"""

    @abstractmethod
    def replay_log(self, world_id: str) -> list[PersistedEntry]:
        """按 seq 正序读出全部日志（供回放/折叠校验）。"""

    @abstractmethod
    def reset(self, world_id: Optional[str] = None) -> None:
        """开发用清库：world_id=None 清所有世界，否则只清该世界。"""


# ---- 建表 DDL：IF NOT EXISTS 幂等，无迁移框架（MVP 不需要，决策：不过度设计）----
_SCHEMA = """
CREATE TABLE IF NOT EXISTS world_snapshots (
    world_id      TEXT PRIMARY KEY,
    snapshot_json TEXT NOT NULL,
    updated_at    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS event_log (
    world_id     TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    trigger_json TEXT NOT NULL,
    events_json  TEXT NOT NULL,
    patch_json   TEXT NOT NULL,
    created_at   INTEGER NOT NULL,
    PRIMARY KEY (world_id, seq)
);
CREATE TABLE IF NOT EXISTS world_meta (
    world_id  TEXT PRIMARY KEY,
    last_seen INTEGER
);
"""


class SQLitePersistenceStore(PersistenceStore):
    """sqlite3 实现。持有单连接（check_same_thread=False）。

    WHY 单连接而非连接池/每操作短连接：MVP 单用户单进程，池化是部署化的事（红线：
    不过度设计）；单连接还让 `:memory:` DB 在多次调用间存活（短连接会丢内存库）。
    并发正确性非 MVP 目标——单用户下不会有真并发写。
    """

    def __init__(self, db_path: str | None = None, *, env: Mapping[str, str] | None = None) -> None:
        self._path = db_path or resolve_db_path(env)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def path(self) -> str:
        return self._path

    def load_snapshot(self, world_id: str) -> Optional[World]:
        row = self._conn.execute(
            "SELECT snapshot_json FROM world_snapshots WHERE world_id = ?",
            (world_id,),
        ).fetchone()
        if row is None:
            return None
        return world_from_dict(json.loads(row["snapshot_json"]))

    def save_snapshot(self, world: World) -> None:
        self._conn.execute(
            "INSERT INTO world_snapshots (world_id, snapshot_json, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(world_id) DO UPDATE SET "
            "snapshot_json = excluded.snapshot_json, updated_at = excluded.updated_at",
            (world.id, json.dumps(world_to_dict(world), ensure_ascii=False), _now_ms()),
        )
        self._conn.commit()

    def append_log(
        self,
        world_id: str,
        trigger: Trigger,
        events: tuple[WorldEvent, ...],
        patch: WorldPatch,
    ) -> int:
        seq = self._conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS next FROM event_log WHERE world_id = ?",
            (world_id,),
        ).fetchone()["next"]
        self._conn.execute(
            "INSERT INTO event_log "
            "(world_id, seq, trigger_json, events_json, patch_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                world_id,
                seq,
                json.dumps(asdict(trigger), ensure_ascii=False),
                json.dumps([asdict(e) for e in events], ensure_ascii=False),
                json.dumps(asdict(patch), ensure_ascii=False),
                _now_ms(),
            ),
        )
        self._conn.commit()
        return seq

    def get_last_seen(self, world_id: str) -> Optional[int]:
        row = self._conn.execute(
            "SELECT last_seen FROM world_meta WHERE world_id = ?",
            (world_id,),
        ).fetchone()
        return None if row is None else row["last_seen"]

    def set_last_seen(self, world_id: str, ts: int) -> None:
        self._conn.execute(
            "INSERT INTO world_meta (world_id, last_seen) VALUES (?, ?) "
            "ON CONFLICT(world_id) DO UPDATE SET last_seen = excluded.last_seen",
            (world_id, ts),
        )
        self._conn.commit()

    def replay_log(self, world_id: str) -> list[PersistedEntry]:
        rows = self._conn.execute(
            "SELECT seq, trigger_json, events_json, patch_json FROM event_log "
            "WHERE world_id = ? ORDER BY seq ASC",
            (world_id,),
        ).fetchall()
        return [
            PersistedEntry(
                seq=r["seq"],
                trigger=_trigger_from_dict(json.loads(r["trigger_json"])),
                events=tuple(_event_from_dict(e) for e in json.loads(r["events_json"])),
                patch=_patch_from_dict(json.loads(r["patch_json"])),
            )
            for r in rows
        ]

    def reset(self, world_id: Optional[str] = None) -> None:
        if world_id is None:
            self._conn.executescript(
                "DELETE FROM world_snapshots; DELETE FROM event_log; DELETE FROM world_meta;"
            )
        else:
            for table in ("world_snapshots", "event_log", "world_meta"):
                self._conn.execute(f"DELETE FROM {table} WHERE world_id = ?", (world_id,))
        self._conn.commit()

    def close(self) -> None:
        """关闭连接。测试 tearDown 用；生产随进程退出。重复调用安全。"""
        self._conn.close()
