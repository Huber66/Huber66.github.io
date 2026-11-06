"""Prompt 组装与「私有视角」上下文预算（F05 / 决策 34）。

本模块把「一个角色在某情境下该如何反应」组装成一份结构化 LLM prompt，
是 F06 LLMReactor 的输入构造层。三条设计红线钉在这里：

1. **私有视角**：每个角色只知道自己该知道的——自己的 persona/mood/独白、
   当前所在房间、房间里在场的其他角色（只呈现「别人眼中的样子」：名字+kind，
   绝不泄露别人的 persona/monologue 私有心理）、当前 trigger、世界时间；
   再加上从记忆里检索到的相关片段。不是全知视角。

2. **上下文预算分层裁剪顺序（决策 34，硬约束）**：
       persona 锚 > 当前情境 > 关系记忆 > 经历记忆
   预算收紧时**从后往前裁**：先砍经历，再砍关系；当前情境与 persona 锚**永不被裁**。
   实现上：persona 锚钉在 system 段、情境钉在 user 段，两者不受 token_cap 约束；
   token_cap 只作用于「记忆检索」这一段的预算分配（关系优先于经历）。

3. **输出固定可解析 JSON**：system 段写死 schema 说明，要求模型只返回一个 JSON 对象，
   字段 speak/reason/line/mood/memory_note，能被 llm.parse_json_response 消费。
   `parse_reaction()` 把 dict 映射成 frozen `ReactionDecision`，缺字段给安全默认。

不做 prompt 模板引擎（不引 Jinja）——f-string + 清晰分段函数即可（优雅原则：不过度设计）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, get_args

from .llm import parse_json_response
from .memory import MemoryStore, MemoryItem, _approx_tokens
from .models import Agent, Mood, Trigger, World

# 合法 mood 集合，从 Literal 单一事实源取（避免硬编码漂移）。
_VALID_MOODS = frozenset(get_args(Mood))

# phase 字面值 → 中文可读标签（world.phase 对齐 frontend/src/types.ts）。
_PHASE_LABEL = {
    "day": "白天",
    "dusk": "傍晚",
    "night": "夜晚",
    "lateNight": "深夜",
}

_KIND_LABEL = {"human": "人", "appliance": "家电/家具"}

# system 段固定文本：输出 schema 说明。含 "JSON" 字样，故 StubChatModel 会走 JSON 分支，
# 无 key 也能返回结构合法的占位输出，供 F06 在零 key 下解析跑通。
_JSON_SCHEMA_INSTRUCTIONS = (
    "【输出格式】你必须只返回一个 JSON 对象，不要包含任何多余的解释文字或代码围栏之外的内容。"
    "字段如下：\n"
    "- speak (bool): 你这次是否开口说话。允许沉默（false）。\n"
    "- reason (string): 你此刻的内心想法/理由（私有心理，别人听不到）。\n"
    "- line (string): 若开口，你说出口的那句话；不开口则留空字符串 \"\"。\n"
    "- mood (string, 可选): 若心情变化，给出新的心情，取值仅限 "
    "[happy, calm, tired, grumpy, lonely]；无变化可省略。\n"
    "- memory_note (string, 可选): 这次经历里值得你记住的一句话；没有则留空。\n"
    "示例：{\"speak\": true, \"reason\": \"...\", \"line\": \"...\", "
    "\"mood\": \"calm\", \"memory_note\": \"...\"}"
)

# oracle / move 两类情境的元规则（钉在 system 段，属 persona 锚同层，永不裁）。
_ORACLE_META = (
    "【情境类型·神谕】世界里会飘来一句「神谕」。它是一种影响，不是命令——"
    "它不能命令你做任何事，只是一个契机。你完全可以选择不回应（speak=false）。"
)
_MOVE_META = (
    "【情境类型·共处】此刻有角色与你身处同一个空间。请依据你和对方的关系、以及你的人设，"
    "做出自然的共处反应。你也可以只是心里活动而不开口（speak=false）。"
)

# ---------------------------------------------------------------------------
# 对外数据形状：frozen dataclass（调用方拿到即定，不被下游悄改）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BudgetReport:
    """一次组装的上下文预算账本（供测试断言与 F14 trace）。

    各层 token 用量 + 记忆条数 + 层级是否因预算被完全裁掉。
    `*_dropped` 表示「该记忆层因预算不足被完全跳过」；层内部分裁剪由 `*_items` 反映。
    """

    token_cap: int
    system_tokens: int       # persona 锚 + 元规则 + schema 说明（永不裁）
    situation_tokens: int    # 当前情境（永不裁）
    relation_tokens: int     # 关系记忆实际用量
    experience_tokens: int   # 经历记忆实际用量
    relation_items: int
    experience_items: int
    relation_dropped: bool
    experience_dropped: bool

    @property
    def total_tokens(self) -> int:
        return (
            self.system_tokens
            + self.situation_tokens
            + self.relation_tokens
            + self.experience_tokens
        )


@dataclass(frozen=True)
class AssembledPrompt:
    """组装产物：OpenAI 风格 messages + 预算账本。

    messages 直接喂给 `ChatModel.chat(messages, ...)`；budget_report 供观测与测试。
    """

    messages: list[dict]
    budget_report: BudgetReport


@dataclass(frozen=True)
class ReactionDecision:
    """LLM 反应的结构化结果。缺字段给安全默认，解析失败由 parse_reaction 返回 None。"""

    speak: bool
    reason: str
    line: str
    mood: Optional[Mood] = None
    memory_note: Optional[str] = None


# ---------------------------------------------------------------------------
# 主入口：组装 prompt
# ---------------------------------------------------------------------------
def assemble_prompt(
    agent: Agent,
    world: World,
    trigger: Trigger,
    memory: MemoryStore,
    *,
    token_cap: int = 1200,
    rel_top_k: int = 3,
    exp_top_k: int = 3,
) -> AssembledPrompt:
    """把「agent 在 trigger 情境下」组装成一份 prompt（私有视角 + 预算裁剪）。

    参数
      agent      : 视角主体（prompt 是「他/它」的第一人称）。
      world      : 取房间名 / 在场者 / 世界时间。
      trigger    : oracle（神谕文本）或 move（谁进了哪个房间）。
      memory     : 记忆检索来源（关系层 rel:{id}:{target}、经历层 agent:{id}）。
      token_cap  : 总上下文预算（中文≈1字1token 粗估）。
      rel_top_k  : 每个对端关系记忆检索条数上限。
      exp_top_k  : 经历记忆检索条数上限。

    裁剪顺序（决策 34）：先扣 persona 锚 + 情境的固定成本，剩余预算按
    「关系优先于经历」分配给记忆检索的 token_cap——经历最先被挤掉。
    """
    # ---- 固定层：persona 锚（system） + 元规则 + schema，永不裁 ----
    persona = _persona_anchor(agent)
    meta, situation, others, rel_query, exp_query = _build_context(agent, world, trigger)
    system_content = "\n\n".join([persona, meta, _JSON_SCHEMA_INSTRUCTIONS])

    system_tokens = _approx_tokens(system_content)
    situation_tokens = _approx_tokens(situation)
    remaining = token_cap - system_tokens - situation_tokens

    # ---- 关系记忆（优先于经历）----
    rel_budget = max(0, remaining)
    rel_items, rel_used = _retrieve_relations(
        memory, agent, others, rel_query, rel_budget, rel_top_k
    )
    relation_dropped = remaining <= 0

    # ---- 经历记忆（最先被裁）----
    exp_budget = remaining - rel_used
    exp_items, exp_used = _retrieve_experience(
        memory, agent, exp_query, max(0, exp_budget), exp_top_k
    )
    experience_dropped = exp_budget <= 0

    # ---- 组 user 段：情境 + （可选）记忆块 ----
    memory_block = _memory_block(rel_items, exp_items)
    user_content = situation if not memory_block else f"{situation}\n\n{memory_block}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    report = BudgetReport(
        token_cap=token_cap,
        system_tokens=system_tokens,
        situation_tokens=situation_tokens,
        relation_tokens=rel_used,
        experience_tokens=exp_used,
        relation_items=len(rel_items),
        experience_items=len(exp_items),
        relation_dropped=relation_dropped,
        experience_dropped=experience_dropped,
    )
    return AssembledPrompt(messages=messages, budget_report=report)


# ---------------------------------------------------------------------------
# 响应解析：dict -> ReactionDecision（缺字段给安全默认）
# ---------------------------------------------------------------------------
def parse_reaction(text: str | None) -> ReactionDecision | None:
    """把模型返回文本解析成 ReactionDecision；解析失败返回 None（F06 据此走降级）。

    复用 llm.parse_json_response（容忍 ```json``` 包裹与前后夹带文字）。字段容错：
      speak       —— 缺失时以「line 是否有内容」兜底（有台词=开口）。
      mood        —— 只认合法 Mood 字面值，非法一律置 None（不污染下游 patch）。
      memory_note —— 空串归一为 None。
    """
    data = parse_json_response(text)
    if data is None:
        return None

    line = str(data.get("line") or "").strip()
    speak = _coerce_bool(data.get("speak"), default=bool(line))
    reason = str(data.get("reason") or "").strip()
    mood = _coerce_mood(data.get("mood"))

    note_raw = data.get("memory_note")
    memory_note = str(note_raw).strip() if note_raw else None
    memory_note = memory_note or None  # 空串归一为 None

    return ReactionDecision(
        speak=speak, reason=reason, line=line, mood=mood, memory_note=memory_note
    )


# ---------------------------------------------------------------------------
# 内部纯函数：分层文本构造 + 记忆检索 + 类型强制
# ---------------------------------------------------------------------------
def _persona_anchor(agent: Agent) -> str:
    """persona 锚（system 段）：角色是谁。永远注入、永不裁剪——它是角色的灵魂。"""
    cp = f"\n- 口头禅：{agent.catchphrase}（可自然带出，不强求）" if agent.catchphrase else ""
    return (
        f"你现在就是「{agent.name}」，不是在扮演，你就是他/它本身。\n"
        f"- 身份：{_kind_label(agent.kind)}\n"
        f"- 人设：{agent.persona}"
        f"{cp}\n"
        f"- 此刻心情：{agent.mood}\n"
        f"- 内心独白基调：{agent.monologue}\n"
        f"请始终从「{agent.name}」的第一人称私有视角出发：你只知道自己知道的，"
        f"不是全知的旁观者，也读不到别人的内心。"
    )


def _build_context(agent: Agent, world: World, trigger: Trigger):
    """按 trigger.kind 分派情境。返回 (meta, situation, others, rel_query, exp_query)。"""
    when = f"{world.name}，{_phase_label(world.phase)}（{world.clockLabel}）"

    if trigger.kind == "oracle":
        room_name = _room_name(world, agent.roomId)
        others = _others_same_room(world, agent)
        text = (trigger.text or "").strip()
        situation = (
            f"此刻：{when}。\n"
            f"你在{room_name}。和你同处一室的还有：{_present_str(others)}。\n"
            f"世界里刚刚飘来一句神谕：「{text}」"
        )
        rel_query = text or "神谕"
        return _ORACLE_META, situation, others, rel_query, text or "神谕"

    if trigger.kind == "move":
        others, mover = _move_context(world, agent, trigger)
        target_room = _room_name(world, trigger.roomId)
        situation = _move_situation(when, agent, mover, target_room, others)
        if trigger.text:
            situation = f"{situation}\n刚才的神谕是：「{trigger.text.strip()}」。请把这句话当成本次移动的直接缘由来回应。"
        rel_query = f"{target_room} " + " ".join(o.name for o in others)
        return _MOVE_META, situation, others, rel_query, target_room

    # catchup / opening 兜底：中性情境，不崩（本任务不要求模板，F09/F08 再补）。
    room_name = _room_name(world, agent.roomId)
    others = _others_same_room(world, agent)
    situation = f"此刻：{when}。\n你在{room_name}。和你同处一室的还有：{_present_str(others)}。"
    return "", situation, others, room_name, room_name


def _move_situation(when: str, agent: Agent, mover: Optional[Agent], target_room: str, others: list[Agent]) -> str:
    """move 情境文案。区分「你走进去」与「别人走进你所在的房间」两种视角。"""
    if mover is not None and mover.id == agent.id:
        # 主体就是移动者：你走进目标房间，看到已在场的人
        return f"此刻：{when}。\n你走进了{target_room}，看到了：{_present_str(others)}。"
    # 主体是房间原住民：mover 走进来
    mover_name = mover.name if mover is not None else "有人"
    companions = [o for o in others if mover is None or o.id != mover.id]
    tail = f"（同室还有：{_present_str(companions)}）" if companions else ""
    return f"此刻：{when}。\n你在{target_room}。{mover_name}走了进来。{tail}"


def _move_context(world: World, agent: Agent, trigger: Trigger):
    """解析 move 情境下的在场者（含尚未 apply patch 的移动者）与移动者对象。

    注意时序：assemble 发生在 step 之前，move patch 尚未落地，移动者的 roomId 仍是
    旧房间。故对「原住民视角」须显式把移动者补进在场者，否则会漏掉共处对象。
    """
    room_id = trigger.roomId
    mover = _find_agent(world, trigger.agentId)
    present = [a for a in world.agents if a.roomId == room_id and a.id != agent.id]
    if mover is not None and mover.id != agent.id and all(mover.id != a.id for a in present):
        present = present + [mover]
    return present, mover


def _retrieve_relations(
    memory: MemoryStore,
    agent: Agent,
    others: list[Agent],
    query: str,
    budget: int,
    top_k: int,
) -> tuple[list[MemoryItem], int]:
    """对每个在场对端从 rel:{agent}:{target} 检索，共享 budget、前面的对端优先。"""
    if budget <= 0 or not others:
        return [], 0
    items: list[MemoryItem] = []
    used = 0
    remaining = budget
    for other in others:
        if remaining <= 0:
            break
        ns = f"rel:{agent.id}:{other.id}"
        found = memory.search(ns, query, top_k=top_k, token_cap=remaining)
        for it in found:
            cost = _approx_tokens(it.text)
            items.append(it)
            used += cost
            remaining -= cost
    return items, used


def _retrieve_experience(
    memory: MemoryStore, agent: Agent, query: str, budget: int, top_k: int
) -> tuple[list[MemoryItem], int]:
    """从 agent:{id} 检索通用经历。预算最紧时最先被裁（budget<=0 直接返回空）。"""
    if budget <= 0:
        return [], 0
    found = memory.search(f"agent:{agent.id}", query, top_k=top_k, token_cap=budget)
    used = sum(_approx_tokens(it.text) for it in found)
    return found, used


def _memory_block(rel_items: list[MemoryItem], exp_items: list[MemoryItem]) -> str:
    """把检索到的记忆拼成 user 段文本。两层都空时返回空串（不硬塞占位）。"""
    parts: list[str] = []
    if rel_items:
        lines = "\n".join(f"- {m.text}" for m in rel_items)
        parts.append(f"【你和相关的人/物之间隐约记得】\n{lines}")
    if exp_items:
        lines = "\n".join(f"- {m.text}" for m in exp_items)
        parts.append(f"【你自己的一些经历】\n{lines}")
    return "\n".join(parts)


def _others_same_room(world: World, agent: Agent) -> list[Agent]:
    """同房间的其他角色（私有视角：调用方只会取其外部可见信息）。"""
    return [a for a in world.agents if a.roomId == agent.roomId and a.id != agent.id]


def _present_str(others: list[Agent]) -> str:
    """在场者的「别人眼中的样子」：只呈现名字 + kind，绝不泄露 persona/monologue。"""
    if not others:
        return "只有你自己"
    return "、".join(f"{o.name}（{_kind_label(o.kind)}）" for o in others)


def _room_name(world: World, room_id: Optional[str]) -> str:
    for r in world.rooms:
        if r.id == room_id:
            return r.name
    return room_id or "某个房间"


def _find_agent(world: World, agent_id: Optional[str]) -> Optional[Agent]:
    if not agent_id:
        return None
    return next((a for a in world.agents if a.id == agent_id), None)


def _phase_label(phase: str) -> str:
    return _PHASE_LABEL.get(phase, phase)


def _kind_label(kind: str) -> str:
    return _KIND_LABEL.get(kind, kind)


def _coerce_bool(value, *, default: bool) -> bool:
    """把 speak 字段稳健转 bool：容忍 bool/数字/字符串（'true'/'是'/…），否则用 default。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是", "会", "开口"}
    return default


def _coerce_mood(value) -> Optional[Mood]:
    """只认合法 Mood 字面值；非法/缺失一律 None（不把脏 mood 传给下游 patch）。"""
    if isinstance(value, str) and value in _VALID_MOODS:
        return value  # type: ignore[return-value]
    return None
