"""应用层 driver —— 把纯函数 step() 与事件溯源 EventStore 串起来。

step() 是纯的、无副作用；driver 负责「有状态」的那部分：持有当前世界、
每次 dispatch 后把 (world_before, trigger, events, patch) 追加进 EventStore，
并折叠出新的当前世界。FastAPI 端点（api.py）只是 driver 的薄封装。

这层刻意薄——真正的逻辑都在纯函数里，driver 只做「记录 + 折叠」的编排。

F03 起 driver 可选挂一个 `PersistenceStore`：内存 EventStore 仍是运行期唯一事实源
（回放/回溯/what-if 都走它），持久化只是**写通镜像**——每次 dispatch 后把日志追加、
快照更新落 SQLite。persistence=None 时行为与 F03 前**完全一致**（向后兼容红线）。
"""

from __future__ import annotations

import itertools
import re
import time
from dataclasses import replace

from .engine import step
from .eventstore import EventStore
from .memory import MemoryStore
from .models import Agent, Room, Trigger, World, WorldEvent, WorldPatch, apply_patch
from .opening import OpeningBeat
from .persistence import PersistenceStore
from .personas import coerce_mood
from .reactor import Reactor, StubReactor
from .seed import make_initial_world, new_agent_id, new_event_id

# 补演最小间隔（毫秒）：离开不足 1 分钟就刷新，视为「没真正离开」，不补演。
# WHY 设阈值：① 语义上「刚看过」没有故事可补；② 挡住频繁刷新刷出一堆 catchup（体验红线）。
# 60s 是保守下界——真实「离开再回来」远大于此，不会误伤正常补演。
_CATCHUP_MIN_GAP_MS = 60 * 1000


def _advance_id_counter(items, prefix: str, fallback):
    """Create an ID generator that starts past the max existing numeric suffix.

    The module-level itertools.count in seed.py resets to 1 on import (i.e., on
    every backend restart). Without this, new agents/events get IDs that collide
    with persisted ones — e.g., a restarted backend assigns agent_1 to a new agent
    while the SQLite snapshot already has agent_1, creating a duplicate.
    """
    max_num = 0
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for item in items:
        sid = item.id if hasattr(item, "id") else str(item)
        m = pattern.match(sid)
        if m:
            max_num = max(max_num, int(m.group(1)))
    if max_num > 0:
        counter = itertools.count(max_num + 1)
        return lambda: f"{prefix}{next(counter)}"
    return fallback


class WorldDriver:
    def __init__(
        self,
        world: World,
        *,
        reactor: Reactor | None = None,
        event_id=new_event_id,
        agent_id=new_agent_id,
        persistence: PersistenceStore | None = None,
        memory: MemoryStore | None = None,
    ) -> None:
        self._world = world
        self._store = EventStore(world)
        self._reactor = reactor or StubReactor()
        self._event_id = event_id
        self._agent_id = agent_id
        self._persistence = persistence
        # 反应生成的记忆检索来源。None → step 内部兜底空 InMemoryStore（保持向后兼容
        # 与可复现）；生产 wiring（api._default_driver）注入与 LLMReactor 共享的实例。
        self._memory = memory

    @classmethod
    def bootstrap(
        cls,
        persistence: PersistenceStore,
        *,
        reactor: Reactor | None = None,
        event_id=new_event_id,
        agent_id=new_agent_id,
        seed_factory=make_initial_world,
        now: int | None = None,
        memory: MemoryStore | None = None,
    ) -> "WorldDriver":
        """从持久化恢复，或首次用 seed 创世。

        WHY 独立类方法而非塞进 __init__：恢复 vs 创世是一次性启动决策，不该每次构造都
        碰 IO。有快照→load_snapshot 直接恢复当前世界（O(1)，不重放整条日志）；无快照→
        seed 创世并立即 save_snapshot + set_last_seen，让「第一次启动」也在库里留痕。

        注意：恢复分支下内存 EventStore 从**恢复后的 world 作 initial**重开（seq 从 0），
        它只服务本进程生命周期内的回放/what-if；跨重启的完整溯源链在 SQLite event_log 里
        （其 seq 由 DB 自增连续，见 SQLitePersistenceStore.append_log 的 WHY）。
        """
        world_id = seed_factory(now=0).id  # 单世界固定 id，取自 seed 保持单一来源
        restored = persistence.load_snapshot(world_id)
        if restored is not None:
            # Advance ID counters past existing IDs to avoid duplicates after restart.
            # seed.py's module-level itertools.count resets to 1 on import, so without
            # this, new agents/events would collide with persisted ones (e.g., agent_1
            # created again while the snapshot already has agent_1 → duplicate).
            agent_id = _advance_id_counter(restored.agents, "agent_", agent_id)
            event_id = _advance_id_counter(restored.events, "evt_", event_id)
            return cls(restored, reactor=reactor, event_id=event_id, agent_id=agent_id, persistence=persistence, memory=memory)

        world = seed_factory(now=now) if now is not None else seed_factory()
        persistence.save_snapshot(world)
        persistence.set_last_seen(world.id, now if now is not None else 0)
        return cls(world, reactor=reactor, event_id=event_id, agent_id=agent_id, persistence=persistence, memory=memory)

    @property
    def world(self) -> World:
        return self._world

    @property
    def store(self) -> EventStore:
        return self._store

    @property
    def memory(self) -> MemoryStore | None:
        """当前 driver 共享的记忆 store。端点层用于私聊等非 step 触发的记忆读写。"""
        return self._memory

    def dispatch(self, trigger: Trigger, *, now: int | None = None) -> tuple[WorldEvent, ...]:
        """执行一次触发：跑 step → 记录溯源 → 折叠出新世界。返回本次新事件。

        持久化（若挂了 store）在内存记录之后写通：append_log（ADD-only 溯源）+
        save_snapshot（当前快照）+ set_last_seen（最近活跃时刻，供 F10 补演）。
        返回值与 F03 前一致（tuple[WorldEvent, ...]），端点合同不受影响。
        """
        world_before = self._world
        events, patch = step(
            world_before, trigger,
            reactor=self._reactor, now=now, event_id=self._event_id, memory=self._memory,
        )
        self._commit(world_before, trigger, events, patch, now=now)
        return events

    def catchup(self, *, now: int) -> tuple[WorldEvent, ...]:
        """惰性补演（F10）：把「上次离开到 now」这段时间凝练成若干条 catchup 事件。

        **一次性批处理，非常驻 loop**（决策 3）：本方法只在被显式调用时跑一次（打开世界/
        GET catchup 时），后端未运行期间零后台任务。`now` 必传且可注入——测试据此模拟
        「离开 6 小时」，绝不在这里读 wall clock（可复现红线：补演窗口完全由入参决定）。

        返回空 tuple（不补演）的三种情形：
          1. 未挂 persistence —— 纯内存 driver 无 last_seen，无从知「离开了多久」（向后兼容）。
          2. 无 last_seen 记录 —— 世界还没被「看过」，谈不上补演离开。
          3. 离开时长 < _CATCHUP_MIN_GAP_MS —— 刚看过就刷新，没必要补（也避免频繁刷新刷出故事）。

        正常路径：构造 Trigger(kind="catchup", since=last_seen, ts=now) 走 step→_commit，
        复用 dispatch 的同一条溯源写通链路（record + apply_patch + append_log + save_snapshot）。
        _commit 里的 set_last_seen(now) 天然把 last_seen 推进到 now——**连续刷新不会重复补演
        同一段**（第二次的 since 已是第一次的 now，gap 归零被阈值挡下）。
        """
        if self._persistence is None:
            return ()
        last_seen = self._persistence.get_last_seen(self._world.id)
        if last_seen is None:
            return ()
        if now - last_seen < _CATCHUP_MIN_GAP_MS:
            return ()  # 刚看过，无需补演（也守住「连续刷新不重复」的第二道闸）

        trigger = Trigger(kind="catchup", since=last_seen, ts=now)
        world_before = self._world
        events, patch = step(
            world_before, trigger,
            reactor=self._reactor, now=now, event_id=self._event_id, memory=self._memory,
        )
        self._commit(world_before, trigger, events, patch, now=now)
        return events

    def add_agent(
        self,
        agent: Agent,
        *,
        now: int | None = None,
        arrival_text: str | None = None,
    ) -> tuple[WorldEvent, ...]:
        """创建一个新角色并写入世界（F07b：创建漏斗确认后落地）。

        **关键设计——不绕过溯源**：新角色经 `WorldPatch(added_agents=...)` + 一条 source
        ="opening" 的「第一次来到这个家」事件落地，与 dispatch 共用同一条溯源写通链路
        （store.record + apply_patch + persistence 写通）。这样事件溯源完整性不破：新增角色
        这一步同样可回放/回溯，与 Mem0 ADD-only 同源。

        WHY 不走 step()：新增角色是确定性的结构变更，不涉及「谁开口」的涌现决策，无需 reactor；
        且红线禁止改 step 纯函数签名。engine 对 opening trigger 本就返回空 patch，故这里直接
        在 driver 层构造 patch 是最小优雅方案——step 保持纯净，溯源日志照样记 (world_before,
        trigger=opening, events, patch)，语义完备。

        agent.id 若为空则由注入的 agent_id 生成器补齐（对齐 seed 的 'agent_xxx' 口径）。
        """
        ts = now if now is not None else int(time.time() * 1000)
        agent_id = agent.id or self._agent_id()
        agent = agent if agent.id else replace(agent, id=agent_id)

        text = arrival_text or f"{agent.name}第一次来到这个家。"
        event = WorldEvent(
            id=self._event_id(), ts=ts, text=text,
            source="opening", clockLabel=self._world.clockLabel, agentId=agent.id,
        )
        trigger = Trigger(kind="opening", agentId=agent.id, roomId=agent.roomId, ts=ts)
        patch = WorldPatch(added_agents=(agent,))

        world_before = self._world
        self._commit(world_before, trigger, (event,), patch, now=now)
        return (event,)

    def delete_agent(self, agent_id: str, *, now: int | None = None) -> tuple[WorldEvent, ...]:
        agent = next((a for a in self._world.agents if a.id == agent_id), None)
        if agent is None:
            return ()
        ts = now if now is not None else int(time.time() * 1000)
        event = WorldEvent(
            id=self._event_id(),
            ts=ts,
            text=f"{agent.name} 被收回了收纳箱。",
            source="system",
            clockLabel=self._world.clockLabel,
            agentId=agent.id,
        )
        trigger = Trigger(kind="floor_plan_update", text=event.text, agentId=agent.id, ts=ts)
        patch = WorldPatch(agents=tuple(a for a in self._world.agents if a.id != agent.id))
        self._commit(self._world, trigger, (event,), patch, now=now)
        return (event,)

    def open_world(
        self,
        beats: tuple[OpeningBeat, ...],
        *,
        now: int | None = None,
    ) -> tuple[WorldEvent, ...]:
        """把预生成的开场 beats 落成 source="opening" 事件写进事件溯源（F08 / Task 12）。

        **与 add_agent 对称**：driver 层直接构造事件 + 空 patch 走 _commit，**不走 step()**。
        opening 只往事件流追加叙事，不改世界结构（无角色/房间/状态变更），故 patch 为空——
        apply_patch 只把新事件并进 world.events。engine 对 opening trigger 本就返回空 patch，
        这里在 driver 层构造是最小优雅方案：step 保持纯净（红线），溯源日志照样记
        (world_before, trigger=opening, events, 空 patch)，可回放/回溯，与 add_agent 同源。

        beats 为空则直接返回 ()，不 commit——没有故事就不往事件流塞噪声（也不推进 last_seen）。
        每条 beat 的 ts 在 now 基础上微偏移，保证事件时序单调、不撞。
        """
        if not beats:
            return ()
        ts_base = now if now is not None else int(time.time() * 1000)
        events = tuple(
            WorldEvent(
                id=self._event_id(),
                ts=ts_base + i,
                text=beat.text,
                source="opening",
                clockLabel=self._world.clockLabel,
                agentId=beat.agentId,
            )
            for i, beat in enumerate(beats)
        )
        trigger = Trigger(kind="opening", ts=ts_base)
        patch = WorldPatch()  # opening 只追加叙事，不改结构

        world_before = self._world
        self._commit(world_before, trigger, events, patch, now=now)
        return events

    def append_life_beats(
        self,
        beats: tuple[tuple[str, str | None], ...],
        *,
        trigger: Trigger,
        now: int | None = None,
        anchor_event_id: str | None = None,
    ) -> tuple[WorldEvent, ...]:
        """Append narrative life beats without structural world changes.

        Background time enrichment uses this after the fast time advance has
        already moved the clock and any agents. The new prose still goes through
        the same event-sourcing path, so replay/history/persistence stay intact.
        """
        if not beats:
            return ()
        ts_base = now if now is not None else int(time.time() * 1000)
        valid_agent_ids = {agent.id for agent in self._world.agents}
        anchor_clock_label = next(
            (event.clockLabel for event in self._world.events if event.id == anchor_event_id),
            None,
        )
        events = tuple(
            WorldEvent(
                id=self._event_id(),
                ts=ts_base + i,
                text=text,
                source="catchup",
                clockLabel=anchor_clock_label or self._world.clockLabel,
                agentId=agent_id if agent_id in valid_agent_ids else None,
                anchorEventId=anchor_event_id,
            )
            for i, (text, agent_id) in enumerate(beats)
            if text.strip()
        )
        if not events:
            return ()
        world_before = self._world
        self._commit(world_before, trigger, events, WorldPatch(), now=now)
        return events

    def revise_agent_persona(
        self,
        agent_id: str,
        *,
        name: str,
        avatar: str,
        persona: str,
        monologue: str,
        mood: str,
        catchphrase: str,
        visibility: str,
        public_event_text: str = "",
        summary: str = "",
        room_id: str | None = None,
        move_with_revision: bool = False,
        now: int | None = None,
    ) -> tuple[WorldEvent, ...]:
        """确认人设修订：更新当前 persona 锚，并追加溯源记录（P0-1）。

        这不是普通 PATCH：旧事件与旧记忆保留，修订只从当前时刻影响未来。
        因此这里直接构造 `Trigger(kind="persona_revision") + WorldPatch(agent_updates=...)`
        走 `_commit`。若用户选择同时搬房间，再在修订落地后 dispatch 一次 move，
        让共处涌现基于新的 persona 锚发生，且不绕过 move 语义。
        """
        current = next((a for a in self._world.agents if a.id == agent_id), None)
        if current is None:
            return ()

        ts = now if now is not None else int(time.time() * 1000)
        event_text = (public_event_text or "").strip()
        if visibility == "public" and not event_text:
            display_name = (name or "").strip() or current.name
            event_text = f"这个家像是忽然更懂{display_name}了。"

        events: tuple[WorldEvent, ...] = ()
        if visibility == "public":
            events = (
                WorldEvent(
                    id=self._event_id(),
                    ts=ts,
                    text=event_text,
                    source="system",
                    clockLabel=self._world.clockLabel,
                    agentId=agent_id,
                ),
            )

        updates = {
            "name": (name or "").strip() or current.name,
            "avatar": (avatar or "").strip() or current.avatar,
            "persona": (persona or "").strip() or current.persona,
            "monologue": (monologue or "").strip() or current.monologue,
            "mood": coerce_mood(mood, default=current.mood),
            # 空字符串是合法含义：角色从此刻起不再有固定口头禅。
            "catchphrase": (catchphrase or "").strip() or None,
        }
        trigger = Trigger(
            kind="persona_revision",
            text=(summary or event_text or "Agent persona revised").strip(),
            agentId=agent_id,
            roomId=room_id,
            ts=ts,
        )
        patch = WorldPatch(agent_updates={agent_id: updates})

        world_before = self._world
        self._commit(world_before, trigger, events, patch, now=now)

        all_events = list(events)
        if move_with_revision and room_id and room_id != current.roomId:
            move_events = self.dispatch(
                Trigger(kind="move", agentId=agent_id, roomId=room_id, ts=ts + 1),
                now=now,
            )
            all_events.extend(move_events)
        return tuple(all_events)

    def update_floor_plan(
        self,
        floor_plan: dict,
        *,
        added_rooms: tuple[Room, ...] = (),
        now: int | None = None,
    ) -> tuple[WorldEvent, ...]:
        """保存自定义户型图，并把新增空间追加进世界。

        几何不进入 step()，因为模拟内核的空间事实仍是 Room + Agent.roomId。这里作为
        世界级结构变更走 `_commit`：事件溯源会记录一次 floor_plan_update，SQLite 快照
        也会写通，避免前端自定义户型刷新丢失。
        """
        ts = now if now is not None else int(time.time() * 1000)
        known_room_ids = {room.id for room in self._world.rooms}
        unique_rooms: list[Room] = []
        seen = set(known_room_ids)
        for room in added_rooms:
            if not room.id or room.id in seen:
                continue
            seen.add(room.id)
            unique_rooms.append(room)
        desired_room_ids = tuple(
            plan_room.get("roomId")
            for plan_room in floor_plan.get("rooms", ())
            if isinstance(plan_room, dict) and plan_room.get("roomId")
        )
        rooms_by_id = {room.id: room for room in (*self._world.rooms, *unique_rooms)}
        next_rooms = tuple(rooms_by_id[room_id] for room_id in desired_room_ids if room_id in rooms_by_id)
        fallback_room_id = next_rooms[0].id if next_rooms else None
        agent_updates = {
            agent.id: {"roomId": fallback_room_id}
            for agent in self._world.agents
            if fallback_room_id is not None and agent.roomId not in {room.id for room in next_rooms}
        }

        event = WorldEvent(
            id=self._event_id(),
            ts=ts,
            text="家里的户型被重新改造了一下。",
            source="system",
            clockLabel=self._world.clockLabel,
        )
        trigger = Trigger(kind="floor_plan_update", text=event.text, ts=ts)
        patch = WorldPatch(
            agent_updates=agent_updates,
            rooms=next_rooms,
            world_updates={"floorPlan": floor_plan},
        )

        world_before = self._world
        self._commit(world_before, trigger, (event,), patch, now=now)
        return (event,)

    def _commit(
        self,
        world_before: World,
        trigger: Trigger,
        events: tuple[WorldEvent, ...],
        patch: WorldPatch,
        *,
        now: int | None,
    ) -> None:
        """记录溯源 + 折叠新世界 + 持久化写通。dispatch 与 add_agent 共用（DRY）。

        持久化（若挂了 store）在内存记录之后写通：append_log（ADD-only 溯源）+
        save_snapshot（当前快照）+ set_last_seen（最近活跃时刻，供 F10 补演）。
        """
        self._store.record(world_before, trigger, events, patch)
        self._world = apply_patch(world_before, patch, events)

        if self._persistence is not None:
            self._persistence.append_log(self._world.id, trigger, events, patch)
            self._persistence.save_snapshot(self._world)
            ts = trigger.ts if trigger.ts is not None else now
            if ts is not None:
                self._persistence.set_last_seen(self._world.id, ts)
