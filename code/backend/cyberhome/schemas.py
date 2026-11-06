"""dataclass ↔ JSON dict 序列化 + HTTP 请求模型（F01）。

序列化方向是 F01 HTTP 闭环的真实所需：把 frozen dataclass `World` 转成
前端 `frontend/src/types.ts` 兼容的 camelCase JSON。关键事实——`models.py` 的字段名
本就是 camelCase（clockLabel/gridCols/roomId/agentId），`GridPos` 用 col/row，
所以 `dataclasses.asdict` 递归转换即达标（嵌套 dataclass→dict、tuple→list）。
无需任何重型序列化框架，这是「优雅极简」的落点。

反序列化方向（dict→dataclass）与序列化对称给出，供 F03 持久化恢复与 round-trip
字段校验复用；它守护「前后端字段严格对齐」这条红线不被未来改动悄悄破坏。
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Optional

from pydantic import BaseModel, Field

from .models import Agent, GridPos, Room, Trigger, World, WorldEvent
from .persona_revision import (
    PersonaRevisionAnswers,
    PersonaRevisionDraft,
    persona_revision_draft_to_dict,
)
from .personas import PersonaAnswers, PersonaDraft


# ---- HTTP 请求模型（Pydantic v2）----
class OracleRequest(BaseModel):
    """POST /oracle 请求体：一句投给世界的契机文本。"""

    text: str


class MoveAgentRequest(BaseModel):
    """POST /agents/{id}/move 请求体：把角色移入的目标房间 id。"""

    roomId: str


class AgentChatMessageRequest(BaseModel):
    """私聊上下文中的单条消息。仅用于当前请求，不落事件流。"""

    role: str = "user"
    text: str = ""

    def normalized_role(self) -> str:
        return self.role if self.role in ("user", "agent") else "user"


class AgentChatRequest(BaseModel):
    """POST /agents/{id}/chat 请求体：用户与单个 Agent 的私聊消息。"""

    text: str
    history: list[AgentChatMessageRequest] = Field(default_factory=list)


class PersonaDraftRequest(BaseModel):
    """POST /personas/draft 请求体：引导问答 + 重生成种子（F07b）。

    字段全可留空（除 kind），后端 stub 兜底；seed 供「重新生成」换措辞变体。
    与前端 CreationAnswers 对齐；freeform 是可选的自由补充。
    """

    kind: str = "human"
    name: str = ""
    avatar: str = ""
    catchphrase: str = ""
    caresAbout: str = ""
    angryReaction: str = ""
    happyReaction: str = ""
    clashWith: str = ""
    freeform: str = ""
    seed: int = 0

    def to_answers(self) -> PersonaAnswers:
        """转成合成层的 PersonaAnswers。kind 非法时归一到 human（不信任外部输入）。"""
        kind = self.kind if self.kind in ("human", "appliance") else "human"
        return PersonaAnswers(
            kind=kind, name=self.name, avatar=self.avatar, catchphrase=self.catchphrase,
            caresAbout=self.caresAbout, angryReaction=self.angryReaction,
            happyReaction=self.happyReaction, clashWith=self.clashWith, freeform=self.freeform,
        )


class CreateAgentRequest(BaseModel):
    """POST /agents 请求体：确认后的草稿字段 + 落入房间 id（F07b）。

    草稿字段由前端从 /personas/draft 返回（可经用户微调）带回；roomId 决定新成员先待哪个房间。
    mood 非法由 persona 合成层的 coerce_mood 归一到 calm（红线 6）。
    """

    kind: str = "human"
    name: str = ""
    avatar: str = ""
    persona: str = ""
    monologue: str = ""
    mood: str = "calm"
    catchphrase: str = ""
    roomId: str


class RoomPayload(BaseModel):
    """保存自定义户型时附带的新房间。字段对齐前端 Room。"""

    id: str
    kind: str
    name: str
    emoji: str = ""
    accent: str = "#c9743b"
    pos: dict = Field(default_factory=lambda: {"col": 1, "row": 1})

    def to_room(self) -> Room:
        kind = self.kind if self.kind in ("living", "kitchen", "bedroom", "study", "entry", "bath", "balcony") else "living"
        col = int(self.pos.get("col", 1)) if isinstance(self.pos, dict) else 1
        row = int(self.pos.get("row", 1)) if isinstance(self.pos, dict) else 1
        return Room(
            id=self.id.strip(),
            kind=kind,
            name=self.name.strip() or "新空间",
            emoji=self.emoji.strip(),
            pos=GridPos(col=col, row=row),
            accent=self.accent.strip() or "#c9743b",
        )


class FloorPlanUpdateRequest(BaseModel):
    """PUT /world/floor-plan：保存自定义户型草稿。

    floorPlan 保持 dict：几何结构由前端 FloorPlan 类型约束，后端只负责把它作为世界级
    事实持久化，并确保新增 roomId 先进入 World.rooms，避免移动端点找不到房间。
    """

    floorPlan: dict
    addedRooms: list[RoomPayload] = Field(default_factory=list)


class PersonaRevisionAnswersRequest(BaseModel):
    """P0-1 人设修订引导答案。字段全可留空，由合成层兜底。"""

    freeform: str = ""
    caresAbout: str = ""
    catchphrase: str = ""
    angryReaction: str = ""
    happyReaction: str = ""
    clashWith: str = ""

    def to_answers(self) -> PersonaRevisionAnswers:
        return PersonaRevisionAnswers(
            freeform=self.freeform,
            caresAbout=self.caresAbout,
            catchphrase=self.catchphrase,
            angryReaction=self.angryReaction,
            happyReaction=self.happyReaction,
            clashWith=self.clashWith,
        )


class PersonaRevisionDraftRequest(BaseModel):
    """POST /agents/{id}/persona-revisions/draft 请求体。"""

    mode: str = "adjust"
    answers: PersonaRevisionAnswersRequest = Field(default_factory=PersonaRevisionAnswersRequest)
    seed: int = 0


class PersonaRevisionDraftPayload(BaseModel):
    """确认端点带回的草稿。字段对齐前端 PersonaRevisionDraft。"""

    agentId: str
    mode: str = "adjust"
    name: str = ""
    avatar: str = ""
    persona: str = ""
    monologue: str = ""
    mood: str = "calm"
    catchphrase: str = ""
    summary: str = ""
    visibilitySuggestion: str = "silent"
    publicEventText: str = ""

    def to_draft(self) -> PersonaRevisionDraft:
        mode = "rewrite" if self.mode == "rewrite" else "adjust"
        visibility = self.visibilitySuggestion if self.visibilitySuggestion in ("silent", "public") else "silent"
        return PersonaRevisionDraft(
            agentId=self.agentId,
            mode=mode,
            name=self.name,
            avatar=self.avatar,
            persona=self.persona,
            monologue=self.monologue,
            mood=self.mood if self.mood in ("happy", "calm", "tired", "grumpy", "lonely") else "calm",
            catchphrase=self.catchphrase,
            summary=self.summary,
            visibilitySuggestion=visibility,
            publicEventText=self.publicEventText,
        )


class PersonaRevisionConfirmRequest(BaseModel):
    """POST /agents/{id}/persona-revisions/confirm 请求体。"""

    draft: PersonaRevisionDraftPayload
    visibility: str = "silent"
    roomId: Optional[str] = None
    moveWithRevision: bool = False

    def normalized_visibility(self) -> str:
        return self.visibility if self.visibility in ("silent", "public") else "silent"


class WhatIfRequest(BaseModel):
    """POST /world/what-if 请求体：对历史某时刻的「假如换一条触发」只读预演（F12 · 决策 33）。

    字段对齐 `Trigger`（kind/text/agentId/roomId），另加 `seq` 指定「从哪一时刻岔开」：
      seq=None → 从最新时刻预演；seq=-1 → 从创世时刻；否则从第 seq 步之后。
    kind 默认 oracle（最常见的「假如投一句神谕」）；非法 kind 归一到 oracle，不信任外部输入。
    ts 不在请求体里——what-if 是「假设」，时刻由端点用 _now_ms() 注入，保证台词时间戳有意义。
    """

    kind: str = "oracle"
    text: Optional[str] = None
    agentId: Optional[str] = None
    roomId: Optional[str] = None
    seq: Optional[int] = None

    def to_trigger(self, ts: int) -> Trigger:
        """转成引擎输入 Trigger。kind 非法时归一到 oracle（与前端动作子集对齐）。"""
        kind = self.kind if self.kind in ("oracle", "move", "catchup", "opening", "time") else "oracle"
        return Trigger(kind=kind, text=self.text, agentId=self.agentId, roomId=self.roomId, ts=ts)


class OpeningRequest(BaseModel):
    """POST /world/opening 请求体：重新生成开场的种子（F08 · Task 12）。

    无字段是必填——开场读的是当前 world（服务端权威），请求只带一个 seed 供「换一种讲法」
    换措辞变体（seed>0，与 personas 的重生成 seed 语义一致）。默认 0 = 首版讲法。
    """

    seed: int = 0


# ---- 序列化：dataclass -> 前端兼容 camelCase dict ----
def persona_draft_to_dict(draft: PersonaDraft) -> dict:
    """PersonaDraft -> 前端兼容 dict（字段与前端 PersonaDraft 对齐）。"""
    return asdict(draft)


def persona_revision_to_dict(draft: PersonaRevisionDraft) -> dict:
    """PersonaRevisionDraft -> 前端兼容 dict。"""
    return persona_revision_draft_to_dict(draft)


def world_to_dict(world: World) -> dict:
    """World -> 前端 `types.ts` 兼容 dict。asdict 递归 + tuple→list 即对齐。"""
    data = asdict(world)
    data["events"] = events_to_dict(world.events)
    return data


def event_to_dict(event: WorldEvent) -> dict:
    return asdict(event)


def events_to_dict(events: tuple[WorldEvent, ...]) -> list[dict]:
    """一串事件 -> 前端兼容 dict 列表。F12 历史时间线（replay_events / what-if）复用。"""
    return [asdict(e) for e in events_with_anchor_order(events)]


def dispatch_result(events: tuple[WorldEvent, ...], world: World) -> dict:
    """oracle / move 的统一响应：本次新增事件 + 更新后完整世界（前端两者都要）。"""
    return {
        "events": events_to_dict(events),
        "world": world_to_dict(world),
    }


def world_with_anchor_order(world: World) -> World:
    """Return a copy whose events are ordered for display without mutating history."""
    ordered = events_with_anchor_order(world.events)
    return world if ordered == world.events else replace(world, events=ordered)


def events_with_anchor_order(events: tuple[WorldEvent, ...]) -> tuple[WorldEvent, ...]:
    event_ids = {event.id for event in events}
    anchored_by_parent: dict[str, list[WorldEvent]] = {}
    anchored_ids: set[str] = set()
    for event in events:
        anchor_event_id = getattr(event, "anchorEventId", None)
        if not anchor_event_id or anchor_event_id not in event_ids:
            continue
        anchored_by_parent.setdefault(anchor_event_id, []).append(event)
        anchored_ids.add(event.id)

    if not anchored_ids:
        return events

    ordered: list[WorldEvent] = []
    emitted: set[str] = set()

    def emit(event: WorldEvent) -> None:
        if event.id in emitted:
            return
        emitted.add(event.id)
        ordered.append(event)
        for child in anchored_by_parent.get(event.id, ()):
            emit(child)

    for event in events:
        if event.id not in anchored_ids:
            emit(event)
    for event in events:
        emit(event)
    return tuple(ordered)


# ---- 反序列化：JSON dict -> dataclass（对称补全，F03 / round-trip 复用）----
def world_from_dict(data: dict) -> World:
    return World(
        id=data["id"],
        name=data["name"],
        phase=data["phase"],
        clockLabel=data["clockLabel"],
        gridCols=data["gridCols"],
        gridRows=data["gridRows"],
        rooms=tuple(_room_from_dict(r) for r in data["rooms"]),
        agents=tuple(_agent_from_dict(a) for a in data["agents"]),
        events=tuple(_event_from_dict(e) for e in data["events"]),
        floorPlan=data.get("floorPlan"),
        # .get 带默认——F03 前的旧世界快照 JSON 无此字段，用 data["ambientInventory"]
        # 会 KeyError 导致旧库无法加载（F11 向后兼容红线）。
        ambientInventory=tuple(data.get("ambientInventory", ())),
    )


def _room_from_dict(d: dict) -> Room:
    pos = d["pos"]
    return Room(
        id=d["id"], kind=d["kind"], name=d["name"], emoji=d["emoji"],
        pos=GridPos(col=pos["col"], row=pos["row"]), accent=d["accent"],
    )


def _agent_from_dict(d: dict) -> Agent:
    return Agent(
        id=d["id"], kind=d["kind"], name=d["name"], avatar=d["avatar"],
        persona=d["persona"], monologue=d["monologue"], mood=d["mood"],
        roomId=d["roomId"], catchphrase=d.get("catchphrase"),
    )


def _event_from_dict(d: dict) -> WorldEvent:
    return WorldEvent(
        id=d["id"], ts=d["ts"], text=d["text"], source=d["source"],
        clockLabel=d.get("clockLabel"), agentId=d.get("agentId"),
        anchorEventId=d.get("anchorEventId"),
    )
