"""后端领域模型 —— 单一事实源对齐 `frontend/src/types.ts`（决策 24）。

全部用 frozen dataclass。不可变不是洁癖：事件溯源（决策 32）要求把 patch
折叠到历史时刻时绝不就地改动 world，冻结从类型层面堵死了「误改历史」。
这也和 Mem0 ADD-only 是同一个不可变范式——时间有重量，写过的不改。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Optional

# ---- 字面类型：逐一对齐 frontend/src/types.ts，后端不得擅自扩集 ----
WorldPhase = Literal["day", "dusk", "night", "lateNight"]
RoomKind = Literal["living", "kitchen", "bedroom", "study", "entry", "bath", "balcony"]
AgentKind = Literal["human", "appliance"]
Mood = Literal["happy", "calm", "tired", "grumpy", "lonely"]

# 事件来源。对齐 frontend/src/types.ts 的 EventSource，并补 catchup（惰性补演，
# 前端 EventSource 尚未列出但 plan §3 的 Trigger 已含，后端提前预留）。
EventSource = Literal[
    "opening",   # 预置开场故事
    "oracle",    # 神谕投放本身
    "reaction",  # 角色对神谕的反应
    "move",      # 拖角色进房间触发共处涌现
    "heartbeat", # 空闲氛围心跳
    "time",      # 世界时间推进
    "catchup",   # 惰性补演追溯（后端专有）
    "system",    # 系统级生活化旁白（如人设修订公开事件）
]


@dataclass(frozen=True)
class GridPos:
    col: int
    row: int


@dataclass(frozen=True)
class Room:
    id: str
    kind: RoomKind
    name: str
    emoji: str
    pos: GridPos
    accent: str


@dataclass(frozen=True)
class Agent:
    id: str
    kind: AgentKind
    name: str
    avatar: str
    persona: str      # 一句话人设。persona 锚永不进记忆层（决策 34）
    monologue: str    # 常驻内心独白，保证「一进门就是活的」
    mood: Mood
    roomId: str
    catchphrase: Optional[str] = None


@dataclass(frozen=True)
class WorldEvent:
    id: str
    ts: int           # 毫秒时间戳，对齐前端 Date.now()
    text: str
    source: EventSource
    clockLabel: Optional[str] = None
    agentId: Optional[str] = None
    # Async LLM life beats finish later, but belong beside the event that caused them.
    anchorEventId: Optional[str] = None


@dataclass(frozen=True)
class World:
    id: str
    name: str
    phase: WorldPhase
    clockLabel: str
    gridCols: int
    gridRows: int
    rooms: tuple[Room, ...]
    agents: tuple[Agent, ...]
    events: tuple[WorldEvent, ...]  # 按时间正序（旧→新）
    # 自定义户型图。None 表示使用前端默认模板；保存后随 world 快照持久化。
    # 几何只服务渲染和命中，模拟内核仍只读 rooms + Agent.roomId。
    floorPlan: Optional[dict] = None
    # 空闲心跳氛围文案库存（F11）：打开时随 world 下发，前端只消费不实时烧 token。
    # 带默认值以免破坏现有全关键字构造（旧调用方 / 旧快照无此字段仍可构造）。
    ambientInventory: tuple[str, ...] = ()


# ---- Trigger：step() 的输入，对齐 plan §3（EventSource 的动作子集）----
@dataclass(frozen=True)
class Trigger:
    """驱动一次 step() 的触发器。

    kind 决定引擎走哪条分支：
      oracle  —— 神谕：全体各自反应（payload=神谕文本）
      move    —— 移动：把 agentId 移入 roomId，触发共处涌现
      catchup —— 惰性补演：一次批处理追溯离开的这段时间（F10）
      opening —— 首次进门预置（本轮由 seed 直接给，预留）
      time    —— 世界时间推进：phase 前进一档（确定性，不调 reactor）
      persona_revision —— 人设修订确认：driver 层结构写通，不走 step()
      floor_plan_update —— 自定义户型保存：driver 层结构写通，不走 step()
    """

    kind: Literal[
        "oracle", "move", "catchup", "opening", "time",
        "persona_revision", "floor_plan_update"
    ]
    text: Optional[str] = None       # oracle 用
    agentId: Optional[str] = None    # move 用
    roomId: Optional[str] = None     # move 用
    ts: Optional[int] = None         # 触发时刻；None 表示由引擎取 now
    # since：catchup 用——补演时间窗的**起点**（上次 last_seen 毫秒）。ts 则是窗口**终点**
    # （now）。WHY 加独立字段而非复用 text 塞毫秒串：since/ts 语义正交、低熵，事件溯源
    # 日志里可读；round-trip 零成本（asdict 自动含它，只需在 persistence._trigger_from_dict
    # 补一行 since=d.get("since")，schemas 不处理 Trigger 故无需改）。非 catchup 恒为 None。
    since: Optional[int] = None


# ---- WorldPatch：step() 的增量输出（决策 32：落日志的是 patch 非全量快照）----
@dataclass(frozen=True)
class WorldPatch:
    """一次 step() 对 world 的增量变更。

    只描述「变了什么」，不含 events（events 单独作为溯源日志的一列）。
    折叠语义见 apply_patch：把 patch 依次作用到 world_before 即得 world_after。
    """

    # agentId -> 字段增量（如 {"roomId": "...", "mood": "..."}）
    agent_updates: dict[str, dict] = field(default_factory=dict)
    # 新增角色（拼搭模式拖入）
    added_agents: tuple[Agent, ...] = ()
    # 替换完整角色列表（删除人物/家电模块时使用）
    agents: Optional[tuple[Agent, ...]] = None
    # 新增房间
    added_rooms: tuple[Room, ...] = ()
    # 替换完整房间列表（自定义户型删除空间时使用）
    rooms: Optional[tuple[Room, ...]] = None
    # 世界级字段增量（如 phase / clockLabel）
    world_updates: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return (
            not self.agent_updates
            and not self.added_agents
            and self.agents is None
            and not self.added_rooms
            and self.rooms is None
            and not self.world_updates
        )


def apply_patch(world: World, patch: WorldPatch, new_events: tuple[WorldEvent, ...] = ()) -> World:
    """把 patch + 新事件折叠进 world，返回**新** World（绝不就地改）。

    这是事件溯源「回溯=折叠 patch」的原子操作（决策 32）。因所有模型 frozen，
    这里必然产出全新对象，历史快照不会被污染——what-if 只读的安全性由此保证。
    """
    source_agents = patch.agents if patch.agents is not None else world.agents
    agents = tuple(
        replace(a, **patch.agent_updates[a.id]) if a.id in patch.agent_updates else a
        for a in source_agents
    )
    if patch.added_agents:
        agents = agents + patch.added_agents

    rooms = patch.rooms if patch.rooms is not None else world.rooms
    if patch.added_rooms:
        rooms = rooms + patch.added_rooms
    events = world.events + new_events if new_events else world.events

    world_updates = dict(patch.world_updates)
    return replace(world, agents=agents, rooms=rooms, events=events, **world_updates)
