"""引擎内核 step() —— 纯函数轻状态机（步骤 2，决策 24/30）。

`step(world, trigger) -> (events, patch)` 是整个后端的心脏。三功能都只是它的
调用方（决策 24）。它是一个 ~20 行主干的**纯函数轻状态机**——不套 StateGraph，
理由见决策 30：会做状态转移不构成采用重框架的理由，前端 worldReducer 用纯
useReducer 做完 World 状态转移就是对称旁证。

「纯」的准确含义：给定相同 (world, trigger, reactor, now, memory)，必得相同
(events, patch)。随机性收进可注入的 reactor.rng，时间由 trigger.ts / now 显式
传入，记忆由 memory 显式传入（缺省用空 InMemoryStore）——这几条保证 step() 可
复现，是事件溯源回放/what-if（决策 32）的地基。

反应生成职责下放给 reactor.react（F06）：engine 只把返回的 AgentReaction 落成
事件（开口者产 reaction 事件）与 patch（mood/monologue 折进 agent_updates），
不掺和「谁该说话」的 bidding 决策。
"""

from __future__ import annotations

import time
import random

from dataclasses import replace

from .memory import InMemoryStore, MemoryStore
from .models import Agent, Room, Trigger, World, WorldEvent, WorldPatch
from .reactor import AgentReaction, Reactor, StubReactor
from .worldclock import advance_phase_label, next_phase, phase_short


def agents_in_room(world: World, room_id: str) -> list[Agent]:
    """房间在场解析：按 roomId 聚合同处角色（决策 24 内核职责之一）。"""
    return [a for a in world.agents if a.roomId == room_id]


def _room_name(world: World, room_id: str) -> str:
    for r in world.rooms:
        if r.id == room_id:
            return r.name
    return room_id


def step(
    world: World,
    trigger: Trigger,
    *,
    reactor: Reactor | None = None,
    now: int | None = None,
    event_id,
    memory: MemoryStore | None = None,
) -> tuple[tuple[WorldEvent, ...], WorldPatch]:
    """驱动一次世界演化。返回 (新事件, 世界增量 patch)，**不改动入参 world**。

    event_id: 生成事件 id 的无参可调用（注入以对齐前端 newEventId 口径 + 可复现）。
    memory  : 反应生成时的记忆检索来源。缺省用空 InMemoryStore，保证纯函数可复现
              且不隐式依赖外部状态（生产 wiring 由 driver/api 注入共享实例）。
    """
    reactor = reactor or StubReactor()
    memory = memory if memory is not None else InMemoryStore()
    ts = trigger.ts if trigger.ts is not None else (now if now is not None else int(time.time() * 1000))

    if trigger.kind == "oracle":
        return _step_oracle(world, trigger, reactor, ts, event_id, memory)
    if trigger.kind == "move":
        return _step_move(world, trigger, reactor, ts, event_id, memory)
    if trigger.kind == "catchup":
        return _step_catchup(world, trigger, reactor, ts, event_id, memory)
    if trigger.kind == "time":
        return _step_time(world, trigger, reactor, ts, event_id, memory)
    # opening 预留：本轮不实现，返回空变更（不报错，便于渐进接入）
    return (), WorldPatch()


def _collect_state_updates(reactions: list[AgentReaction]) -> dict[str, dict]:
    """把每条 reaction 里非 None 的 mood/monologue 汇总成 agent_updates（只写有变化的字段）。

    沉默（speak=False）但心情/独白有变的反应也计入——状态涌现不以「是否开口」为前提。
    """
    updates: dict[str, dict] = {}
    for r in reactions:
        fields: dict = {}
        if r.mood is not None:
            fields["mood"] = r.mood
        if r.monologue is not None:
            fields["monologue"] = r.monologue
        if fields:
            updates[r.agent_id] = fields
    return updates


def _step_time(world, trigger, reactor, ts, event_id, memory):
    """世界时间推进（F11）：phase 前进一档，产一条 time 事件，phase/clockLabel 走 world_updates。

    事件 clockLabel 用 short，
    world.clockLabel 用完整 label（与前端 advancePhase / ADVANCE_PHASE 的口径逐字对齐）。
    推进节律后顺手跑一片轻量自主生活，让同屋对话和低频走动随时间自然发生。
    """
    phase = next_phase(world.phase)
    label = advance_phase_label(world.clockLabel, world.phase, phase)
    short = phase_short(phase)
    event = WorldEvent(
        id=event_id(), ts=ts,
        text=f"天色转向{short}，家里的光影悄悄变了。",
        source="time", clockLabel=short,
    )
    # If the user advances while it is already late night, the life slice should
    # still resolve late-night behavior (for example, someone finally going to
    # bed) before the world visually rolls into the next day.
    life_phase = world.phase if world.phase == "lateNight" else phase
    world_after_time = replace(world, phase=life_phase, clockLabel=label)
    life_events, life_updates = _autonomous_life_slice(
        world_after_time,
        ts=ts + 1,
        event_id=event_id,
        reactor=_time_slice_reactor(ts),
        memory=memory,
        slice_index=_time_life_slice_index(phase),
        allow_wander=True,
        include_dialogue=False,
    )
    patch = WorldPatch(
        agent_updates=life_updates,
        world_updates={"phase": phase, "clockLabel": label},
    )
    return (event, *life_events), patch


def _time_slice_reactor(ts: int) -> Reactor:
    """Use local deterministic prose for manual time advances.

    The advance-time button is a UI clock control and must feel instant. Real
    LLM calls still happen for oracle/private chat/opening/catchup, but time
    advancement should not wait on networked generation.
    """
    return StubReactor(rng=random.Random(ts))


def _time_life_slice_index(phase: str) -> int:
    """Map a clock phase to a deterministic life-slice slot.

    Slot 2 enables the low-priority wandering branch. Manual time advances are
    user-visible, so each advance may move one human Agent when no stronger
    intent rule applies.
    """
    return 2


def _step_oracle(world, trigger, reactor, ts, event_id, memory):
    """神谕：先落一条「已投放」事件，再由 reactor 决定谁反应（决策 7：影响非命令）。"""
    text = (trigger.text or "").strip()
    oracle_event = WorldEvent(
        id=event_id(), ts=ts,
        text=f"神谕已投放：「{text}」——一阵微风拂过整个家。",
        source="oracle", clockLabel=world.clockLabel,
    )

    move_intent = _oracle_move_intent(world, text)
    if move_intent is not None:
        agent, room = move_intent
        move_events, move_patch = _step_move(
            world,
            Trigger(
                kind="move",
                text=text,
                agentId=agent.id,
                roomId=room.id,
                ts=ts + 1,
            ),
            reactor,
            ts + 1,
            event_id,
            memory,
        )
        return (oracle_event, *move_events), move_patch

    events: list[WorldEvent] = [oracle_event]
    reactions = reactor.react(world, trigger, memory)
    # 开口者（speak 且台词非空）落成 reaction 事件；bidding/capped 已由 reactor 完成
    speakers = [r for r in reactions if r.speak and r.line]
    for i, r in enumerate(speakers):
        events.append(
            WorldEvent(
                id=event_id(), ts=ts + i + 1, text=r.line,
                source="reaction", clockLabel=world.clockLabel, agentId=r.agent_id,
            )
        )
    # mood/monologue 折进 patch（含沉默但状态有变者）
    return tuple(events), WorldPatch(agent_updates=_collect_state_updates(reactions))


_ORACLE_MOVE_MARKERS = ("去", "来", "到", "进", "回", "拿", "带", "送", "搬", "端")


def _oracle_move_intent(world: World, text: str) -> tuple[Agent, Room] | None:
    """Parse simple oracle movement commands like “老妈拿菜去客厅”."""
    normalized = text.strip()
    if not normalized or not any(marker in normalized for marker in _ORACLE_MOVE_MARKERS):
        return None

    agent = next((a for a in world.agents if a.name and a.name in normalized), None)
    if agent is None:
        return None

    room = next((r for r in world.rooms if r.name and r.name in normalized), None)
    if room is None or room.id == agent.roomId:
        return None
    return agent, room


def _step_move(world, trigger, reactor, ts, event_id, memory):
    """移动：engine 计算 roomId 位移，共处叙事/反应文案由 reactor 产出。"""
    agent_id = trigger.agentId
    room_id = trigger.roomId
    if not agent_id or not room_id:
        return (), WorldPatch()

    moved = next((a for a in world.agents if a.id == agent_id), None)
    if moved is None:
        return (), WorldPatch()

    reactions = reactor.react(world, trigger, memory)
    room_name = _room_name(world, room_id)

    # 移动者的共处叙事作为 move 事件；缺失则中性兜底——保证 ≥1 条 move 事件
    # （前端 moveAgent 依赖 events.length>0 判断是否显示气泡，空则提示「没挪过去」）。
    mover_reaction = next(
        (r for r in reactions if r.agent_id == agent_id and r.speak and r.line), None
    )
    move_line = mover_reaction.line if mover_reaction is not None else f"{moved.name} 走进了{room_name}。"
    events: list[WorldEvent] = [
        WorldEvent(
            id=event_id(), ts=ts, text=move_line,
            source="move", clockLabel=world.clockLabel, agentId=agent_id,
        )
    ]
    # 其他在场者的反应（speak）追加为 reaction 事件
    offset = 1
    for r in reactions:
        if r.agent_id == agent_id:
            continue
        if r.speak and r.line:
            events.append(
                WorldEvent(
                    id=event_id(), ts=ts + offset, text=r.line,
                    source="reaction", clockLabel=world.clockLabel, agentId=r.agent_id,
                )
            )
            offset += 1

    # patch：roomId（engine 职责）与 mood/monologue（reaction 折入）合并进同一 agent_updates，
    # 注意不互相覆盖——移动者可能同时有 roomId 位移与心情变化。
    agent_updates = _collect_state_updates(reactions)
    mover_update = dict(agent_updates.get(agent_id, {}))
    mover_update["roomId"] = room_id
    agent_updates[agent_id] = mover_update
    return tuple(events), WorldPatch(agent_updates=agent_updates)


# 补演分片上限：一次 catchup 最多补 5 片（决策 3：只表现「有故事发生过」，不还原每分钟）。
# 上限而非下限——避免离开越久事件越爆，也守住「一次批处理」轻量语义。
_CATCHUP_MAX_SLICES = 5
# 每片代表的世界时长（毫秒）：默认每 ~1 小时凝练成 1 片生活切片。delta 越大片数越多，
# 但被 _CATCHUP_MAX_SLICES 封顶。这是把「物理流逝时间」映射到「叙事密度」的唯一旋钮。
_CATCHUP_MS_PER_SLICE = 60 * 60 * 1000


def _catchup_slice_count(since: int, now: int) -> int:
    """按时间窗长度算补演片数，clamp 到 [1, _CATCHUP_MAX_SLICES]。

    窗口非正（now<=since，理论不该进到这——driver 已挡最小阈值）时返回 0（无补演）。
    否则至少 1 片（既然进来补演，就该有故事），每满一个 _CATCHUP_MS_PER_SLICE 多一片。
    纯算术、无随机/无 wall clock——同 (since, now) 必得同片数（可复现红线）。
    """
    delta = now - since
    if delta <= 0:
        return 0
    slices = delta // _CATCHUP_MS_PER_SLICE
    return max(1, min(_CATCHUP_MAX_SLICES, int(slices)))


def _rooms_with_agents(world: World) -> list[str]:
    """按 world.rooms 顺序取「有 ≥1 Agent 在场」的房间 id 列表（确定性轮询的候选池）。

    刻意按 world.rooms 顺序而非 agents 顺序——房间顺序是世界的稳定结构，轮询挑房间时
    同一世界必得同一序列（可复现）。空房间跳过：没人的房间没有「生活」可补。
    """
    return [r.id for r in world.rooms if agents_in_room(world, r.id)]


def _room_id_by_kind(world: World, kind: str) -> str | None:
    for room in world.rooms:
        if room.kind == kind:
            return room.id
    return None


def _conversation_rooms(world: World) -> list[str]:
    """按房间顺序返回有至少两个 Agent 同处的房间。"""
    return [r.id for r in world.rooms if len(agents_in_room(world, r.id)) >= 2]


def _merge_agent_update(updates: dict[str, dict], agent_id: str, fields: dict) -> None:
    if not fields:
        return
    cur = dict(updates.get(agent_id, {}))
    cur.update(fields)
    updates[agent_id] = cur


def _merge_agent_updates(dst: dict[str, dict], src: dict[str, dict]) -> None:
    for agent_id, fields in src.items():
        _merge_agent_update(dst, agent_id, fields)


def _world_with_agent_updates(world: World, updates: dict[str, dict]) -> World:
    if not updates:
        return world
    agents = tuple(replace(agent, **updates[agent.id]) if agent.id in updates else agent for agent in world.agents)
    return replace(world, agents=agents)


def _agent_wants_kitchen(agent: Agent, world: World) -> bool:
    text = f"{agent.name} {agent.persona} {agent.monologue} {agent.catchphrase or ''}"
    cooking_markers = ("做饭", "饭", "厨房", "鸡蛋", "围裙", "炒", "锅", "吃")
    return world.phase in ("day", "dusk") and any(marker in text for marker in cooking_markers)


def _agent_wants_rest(agent: Agent, world: World) -> bool:
    text = f"{agent.name} {agent.persona} {agent.monologue}"
    rest_markers = ("困", "睡", "打盹", "累", "歇")
    return world.phase == "lateNight" or agent.mood == "tired" or any(marker in text for marker in rest_markers)


def _choose_autonomous_move(world: World, slice_index: int, *, allow_wander: bool) -> tuple[Agent, str, str] | None:
    """挑一个自然走动。只移动 human，家电/家具作为物性 Agent 不自己满屋跑。"""
    humans = [agent for agent in world.agents if agent.kind == "human"]
    if not humans:
        return None

    kitchen = _room_id_by_kind(world, "kitchen")
    bedroom = _room_id_by_kind(world, "bedroom")
    living = _room_id_by_kind(world, "living")
    study = _room_id_by_kind(world, "study")
    rest_room = bedroom or living

    for agent in humans:
        if kitchen and agent.roomId != kitchen and _agent_wants_kitchen(agent, world):
            return agent, kitchen, "kitchen"

    if world.phase == "lateNight" and bedroom:
        for agent in humans:
            if agent.roomId != bedroom:
                return agent, bedroom, "rest"

    if rest_room:
        for agent in humans:
            if agent.roomId not in {rest_room, living} and _agent_wants_rest(agent, world):
                return agent, rest_room, "rest"

    if not allow_wander or slice_index % 3 != 2:
        return None

    rooms = [room.id for room in world.rooms if room.id]
    if len(rooms) < 2:
        return None
    agent = humans[slice_index % len(humans)]
    if world.phase == "night" and agent.id == "agent_dad" and study and agent.roomId != study:
        return agent, study, "study"
    current_index = rooms.index(agent.roomId) if agent.roomId in rooms else -1
    target = rooms[(current_index + 1 + slice_index) % len(rooms)]
    if target == agent.roomId:
        return None
    return agent, target, "wander"


def _autonomous_move_text(agent: Agent, room_name: str, reason: str) -> str:
    if reason == "kitchen":
        return f"{agent.name}惦记着锅灶和晚饭，自己去了{room_name}。"
    if reason == "rest":
        return f"{agent.name}看了看天色，挪去{room_name}歇一会儿。"
    if reason == "study":
        return f"{agent.name}把手边的事收了收，去{room_name}待一会儿。"
    return f"{agent.name}没说什么，慢慢溜达到{room_name}。"


def _autonomous_life_slice(
    world: World,
    *,
    ts: int,
    event_id,
    reactor: Reactor,
    memory: MemoryStore,
    slice_index: int,
    allow_wander: bool,
    include_dialogue: bool = True,
) -> tuple[tuple[WorldEvent, ...], dict[str, dict]]:
    """一次自主生活切片：偶尔走动，然后让同屋 Agent 自然搭话。"""
    events: list[WorldEvent] = []
    updates: dict[str, dict] = {}
    working_world = world

    conversation_room_id: str | None = None
    bedroom = _room_id_by_kind(working_world, "bedroom")
    if working_world.phase == "lateNight" and bedroom:
        sleepy_humans = [
            agent for agent in working_world.agents
            if agent.kind == "human" and agent.roomId != bedroom
        ]
        for agent in sleepy_humans:
            _merge_agent_update(updates, agent.id, {"roomId": bedroom})
        if sleepy_humans:
            room_name = _room_name(working_world, bedroom)
            if len(sleepy_humans) >= 2:
                names = "和".join(agent.name for agent in sleepy_humans[:2])
                text = f"{names}看了看天色，一起挪去{room_name}歇一会儿。"
                agent_id = None
            else:
                text = _autonomous_move_text(sleepy_humans[0], room_name, "rest")
                agent_id = sleepy_humans[0].id
            events.append(
                WorldEvent(
                    id=event_id(),
                    ts=ts + len(events),
                    text=text,
                    source="move",
                    clockLabel=working_world.clockLabel,
                    agentId=agent_id,
                )
            )
            working_world = _world_with_agent_updates(working_world, updates)
            if len(agents_in_room(working_world, bedroom)) >= 2:
                conversation_room_id = bedroom
    else:
        move = _choose_autonomous_move(working_world, slice_index, allow_wander=allow_wander)
        if move is not None:
            agent, target_room_id, reason = move
            _merge_agent_update(updates, agent.id, {"roomId": target_room_id})
            room_name = _room_name(working_world, target_room_id)
            events.append(
                WorldEvent(
                    id=event_id(),
                    ts=ts,
                    text=_autonomous_move_text(agent, room_name, reason),
                    source="move",
                    clockLabel=working_world.clockLabel,
                    agentId=agent.id,
                )
            )
            working_world = _world_with_agent_updates(working_world, updates)
            if len(agents_in_room(working_world, target_room_id)) >= 2:
                conversation_room_id = target_room_id

    if not include_dialogue:
        return tuple(events), updates

    if conversation_room_id is None:
        rooms = _conversation_rooms(working_world)
        if rooms:
            conversation_room_id = rooms[slice_index % len(rooms)]

    if conversation_room_id is None:
        return tuple(events), updates

    sub = Trigger(kind="catchup", roomId=conversation_room_id, ts=ts + len(events))
    reactions = reactor.react(working_world, sub, memory)
    _merge_agent_updates(updates, _collect_state_updates(reactions))
    speakers = [r for r in reactions if r.speak and r.line]
    offset = len(events)
    for r in speakers:
        events.append(
            WorldEvent(
                id=event_id(),
                ts=ts + offset,
                text=r.line,
                source="catchup",
                clockLabel=working_world.clockLabel,
                agentId=r.agent_id,
            )
        )
        offset += 1
    return tuple(events), updates


def _step_catchup(world, trigger, reactor, ts, event_id, memory):
    """惰性补演（F10）：把「你不在时」这段时间凝练成若干条生活切片事件。

    **一次批处理**，非常驻 loop（决策 3）：读 trigger 的时间窗 (since, ts]，按窗长算 1–5 片，
    每片挑一个有人的房间跑一次 reactor.react，把开口者的台词落成 source="catchup" 事件。

    可复现红线：房间按 `world.rooms` 顺序轮询（第 i 片挑第 i 个有人的房间，取模轮转），
    子 trigger 的 ts 在 (since, ts] 上**均匀插值**——不读 wall clock、不碰 DB、不随机挑房间。
    同一 (world, since, ts, 固定 rng reactor) 必得同一批事件，是事件溯源可回放的地基。

    mood/monologue 复用 _collect_state_updates 折进 patch（补演期间角色状态也会漂移）。
    最坏情况（无房间有人 / 全体沉默）产出空事件——降级不崩（决策 8.3）。
    """
    since = trigger.since
    # since 缺失或窗口非正：无从算窗，退化为「无事可补」（driver 正常会传合法 since）。
    if since is None or ts <= since:
        return (), WorldPatch()

    rooms = _rooms_with_agents(world)
    if not rooms:
        return (), WorldPatch()  # 空世界 / 全空房：没有生活可补

    n = _catchup_slice_count(since, ts)
    if n <= 0:
        return (), WorldPatch()

    events: list[WorldEvent] = []
    agent_updates: dict[str, dict] = {}
    working_world = world
    span = ts - since
    for i in range(n):
        # 子 trigger 的时刻：在 (since, ts] 上均匀插值。第 i 片落在 since + span*(i+1)/n，
        # 保证严格 > since 且最后一片正好 = ts（窗口终点），时序单调递增（可复现）。
        slice_ts = since + (span * (i + 1)) // n
        # A slice can emit a move plus several dialogue lines. Keep those
        # sub-events inside the catchup window even for the final slice.
        event_base_ts = max(since + 1, slice_ts - 3)
        slice_events, slice_updates = _autonomous_life_slice(
            working_world,
            ts=event_base_ts,
            event_id=event_id,
            reactor=reactor,
            memory=memory,
            slice_index=i,
            allow_wander=True,
        )
        events.extend(slice_events)
        _merge_agent_updates(agent_updates, slice_updates)
        working_world = _world_with_agent_updates(working_world, slice_updates)
    # 补演期间的走动、mood/monologue 漂移折进同一个 patch。
    return tuple(events), WorldPatch(agent_updates=agent_updates)
