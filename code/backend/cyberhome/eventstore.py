"""事件溯源 —— 不可变追加日志 + 回放/回溯/what-if（步骤 5，决策 32/33）。

本轮重心。核心论断（决策 32）：因 step() 是纯函数，三种重放是**免费午餐**——
不需要 LangGraph checkpointer 在有状态图里「模拟无状态重演」，我们本就无状态。

一条日志 = (world_before, trigger, events, patch)。把它们依时间追加，就得到：
  · 回放  replay_events —— 重播 events（前端时间轴回放的数据源）
  · 回溯  world_at      —— 从 initial 折叠 patch 到第 n 步，得当时快照
  · what-if fork_readonly —— 对某时刻 world 重新 step 预演分支
                            **A 档：推演完即弃、不落库、时间线单线（决策 33）**

B 档（分叉世界持久化、时间线成树）是 LangGraph 存废的开关，列 P2，本模块不做。
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Trigger, World, WorldEvent, WorldPatch, apply_patch
from .reactor import Reactor


@dataclass(frozen=True)
class LogEntry:
    """一次 step() 的不可变溯源记录。写入后永不修改（与 Mem0 ADD-only 同源）。"""

    seq: int
    world_before: World
    trigger: Trigger
    events: tuple[WorldEvent, ...]
    patch: WorldPatch


class EventStore:
    """世界状态的不可变追加日志。`initial` 是世界的创世快照。"""

    def __init__(self, initial: World) -> None:
        self._initial = initial
        self._log: list[LogEntry] = []

    @property
    def initial(self) -> World:
        return self._initial

    def __len__(self) -> int:
        return len(self._log)

    def record(
        self,
        world_before: World,
        trigger: Trigger,
        events: tuple[WorldEvent, ...],
        patch: WorldPatch,
    ) -> LogEntry:
        """追加一条溯源记录（ADD-only）。返回该条目。"""
        entry = LogEntry(
            seq=len(self._log),
            world_before=world_before,
            trigger=trigger,
            events=events,
            patch=patch,
        )
        self._log.append(entry)
        return entry

    # ---- 回放：重播 events（决策 32）----
    def replay_events(self, until_seq: int | None = None) -> tuple[WorldEvent, ...]:
        """按时间正序汇总 initial + 各步 events，供前端时间轴回放。

        until_seq=None 表示到最新；否则含到第 until_seq 步（含）。
        """
        out: list[WorldEvent] = list(self._initial.events)
        for entry in self._log:
            if until_seq is not None and entry.seq > until_seq:
                break
            out.extend(entry.events)
        return tuple(out)

    # ---- 回溯：折叠 patch 到某时刻（决策 32）----
    def world_at(self, seq: int | None = None) -> World:
        """把 initial 依次折叠各步的 (events, patch)，得到第 seq 步之后的世界快照。

        seq=None → 最新；seq=-1 → 创世（initial 原样）。因所有模型 frozen，
        折叠必产出新对象，绝不污染历史（what-if 只读安全的根基）。
        """
        world = self._initial
        for entry in self._log:
            if seq is not None and entry.seq > seq:
                break
            world = apply_patch(world, entry.patch, entry.events)
        return world

    # ---- what-if：对某时刻 world 重新 step（A 档只读，决策 33）----
    def fork_readonly(
        self,
        at_seq: int | None,
        trigger: Trigger,
        *,
        reactor: Reactor,
        event_id,
        now: int | None = None,
    ) -> tuple[tuple[WorldEvent, ...], WorldPatch, World]:
        """预演「若在第 at_seq 步后改走另一条 trigger 会怎样」。

        A 档语义（决策 33）：**只读**——回溯出历史 world 快照，对它跑一次 step，
        返回预演结果，**绝不写回 self._log、不改主线**。时间线保持单线。
        返回 (预演事件, 预演 patch, 预演后的假想世界)。
        """
        from .engine import step  # 局部导入避免与 engine 循环依赖

        base = self.world_at(at_seq)
        events, patch = step(base, trigger, reactor=reactor, now=now, event_id=event_id)
        hypothetical = apply_patch(base, patch, events)
        # 注意：此处不调用 self.record()——这正是 A 档「推演完即弃」的落地。
        return events, patch, hypothetical
