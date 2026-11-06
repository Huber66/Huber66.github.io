"""开场故事合成 —— 读当前世界真实角色 + 布局，动态生成「活开场」（F08 / Task 12）。

把开场从「seed 写死的静态文案」升级为「读此刻家里有谁、都在哪个房间、心里想着什么」
动态生成的 3-5 条「正在发生的生活切片」。它是 POST /world/opening 的核心，**纯合成、
无副作用**：不写 world、不落库、不碰记忆（落溯源由 driver.open_world 负责，见其 WHY）。

结构严格仿 personas.py（F07b 的黄金模板），三条设计红线同构地钉在这里：

1. **无 key 也能跑**：合成走 ChatModel。无 key 时工厂给 StubChatModel，它对「含 JSON
   要求」的 prompt 返回结构合法但**不含 beats 字段**的占位 JSON——本模块据此判定「LLM
   未给出有效开场」，回退到读真实世界的确定性 stub（stub_opening）。故无论有无 key，
   端点都拿得到一份完整合法开场，绝不 500。这与 personas 判定 persona 缺失完全同构。
2. **确定性可复现**：stub 开场同一 (world, seed) 必得同一结果；「换一种讲法」= seed>0
   换一套措辞变体（_pick 公式与 personas 对齐，seed*131+salt*17+7 取模）。
3. **是「生活」不是「引导」**：开场是「家里正在发生的事」，绝无「欢迎使用 / 系统提示」
   式产品话术——文风参照 seed._OPENING_RAW（生活化、有温度、像正在发生）。

prompt 组装刻意不复用 personas / prompting 的组装——「据世界写开场」与「把问答合成人设」
「角色在情境下如何反应」是三件不同的事，各自内聚更优雅（与 personas 同一取舍）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .llm import ChatModel, parse_json_response
from .models import World

logger = logging.getLogger("cyberhome.opening")

# 开场条数区间：至少 3 条才成「一段开场」，至多 5 条免得絮叨（也与 seed 的 5 条对齐）。
_MIN_BEATS = 3
_MAX_BEATS = 5


# ---------------------------------------------------------------------------
# 对外数据形状：frozen dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpeningBeat:
    """一条开场叙事切片。

    agentId 关联某个家庭成员（该句是「谁此刻在做什么」）；世界级氛围句可为 None
    （不属于某一个角色，是整个家的动静）。driver 落事件时据此填 WorldEvent.agentId。
    """

    text: str
    agentId: str | None = None


# ---------------------------------------------------------------------------
# 主入口：合成开场（stub baseline → chat → parse → stub 兜底，绝不抛）
# ---------------------------------------------------------------------------
def synthesize_opening(
    chat_model: ChatModel,
    world: World,
    *,
    seed: int = 0,
    max_output_tokens: int = 520,
) -> tuple[OpeningBeat, ...]:
    """读当前 world 动态生成 3-5 条开场。**纯合成、无副作用。**

    流程与 synthesize_persona_draft 同构：先算确定性 stub 作 baseline（也是兜底来源）→
    调 LLM → 解析。LLM 给出有效开场（≥1 条）则用它，否则整体退回 stub。任何异常都吞掉
    后退回 stub——保证端点绝不 500（无 key 闭环不断的落点）。
    """
    fallback = stub_opening(world, seed)
    try:
        messages = assemble_opening_prompt(world, seed=seed)
        resp = chat_model.chat(messages, max_output_tokens=max_output_tokens)
        beats = parse_opening(resp.text, world)
    except Exception as exc:  # 合成失败绝不抛给端点：记脱敏摘要后退回 stub
        logger.warning("opening synth failed err=%s; falling back to stub", type(exc).__name__)
        return fallback
    return beats if beats else fallback


# ---------------------------------------------------------------------------
# prompt 组装：system 段写死含 "JSON" 的 schema（触发 StubChatModel 的 JSON 分支）
# ---------------------------------------------------------------------------
_SCHEMA_INSTRUCTIONS = (
    "【输出格式】你必须只返回一个 JSON 对象，不要包含任何多余的解释文字或代码围栏之外的内容。"
    "字段如下：\n"
    "- beats (array): 3 到 5 条开场叙事，按自然顺序排列，读起来像一段正在发生的家庭画面。"
    "数组的每一项是一个对象：\n"
    "  - text (string): 一句「此刻家里正在发生」的生活切片，温暖、具体、像正在发生的画面。\n"
    "  - agentId (string): 这句关联的家庭成员 id（见下方名单）；若是纯氛围叙述句则留空字符串。\n"
    "示例：{\"beats\": [{\"text\": \"...\", \"agentId\": \"agent_mom\"}, "
    "{\"text\": \"...\", \"agentId\": \"\"}]}"
)


def assemble_opening_prompt(world: World, *, seed: int = 0) -> list[dict]:
    """把当前世界组装成 OpenAI 风格 messages。system 写叙述者元规则 + JSON schema。

    system 段**必含 "JSON" 字样**——这是无 key 闭环的机关：StubChatModel 的 _wants_json
    据此走 JSON 分支返回默认占位模板（不含 beats），parse_opening 取不到 beats 即回退 stub。
    seed>0（重新生成）时在 user 段追加一句「换一种讲法」，让真 LLM 的重生成也有变化；
    StubChatModel 忽略语义，变化由 stub 的 seed 变体承载。
    """
    system_content = (
        f"你是这个叫「{world.name}」的家的叙述者。这里万物皆有主体性——人和家电一样，"
        "有脾气、有牵挂、有自己的小日子。请根据当前家里有谁、都在哪个房间、心里想着什么，"
        "写一段「此刻家里正在发生的生活切片」作为开场。要温暖、生活化、像正在发生的画面，"
        "落到具体的人和物件上；不要写产品引导，不要客套，不要出现「欢迎」「系统」这类话。\n\n"
        + _SCHEMA_INSTRUCTIONS
    )

    lines = [f"现在是{world.clockLabel}（时段：{world.phase}）。家里此刻的情况："]
    for room in world.rooms:
        residents = [a for a in world.agents if a.roomId == room.id]
        if not residents:
            lines.append(f"- {room.name}：暂时没人。")
            continue
        lines.append(f"- {room.name}：")
        for a in residents:
            kind_label = "家电/家具" if a.kind == "appliance" else "人"
            mono = _clean(a.monologue)
            catch = _clean(a.catchphrase or "")
            detail = f"  · {a.name}（{kind_label}，id={a.id}，此刻心情 {a.mood}）"
            if mono:
                detail += f"，正想着「{mono}」"
            if catch:
                detail += f"，口头禅「{catch}」"
            lines.append(detail)
    if seed > 0:
        lines.append("（我想换一种讲法，请重新写一版开场，换一套措辞和视角。）")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(lines)},
    ]


# ---------------------------------------------------------------------------
# 响应解析：dict -> tuple[OpeningBeat]（无有效 beats 返回 None → 上层走 stub）
# ---------------------------------------------------------------------------
def parse_opening(text: str | None, world: World) -> tuple[OpeningBeat, ...] | None:
    """把 LLM 返回解析成一串 OpeningBeat。

    返回 None 表示「LLM 未给出有效开场」（解析失败 / 无 beats / 无一条有效）——调用方据此
    整体退回 stub。这正是无 key 路径的判定点：StubChatModel 的默认占位 JSON 不含 beats
    字段，故 data.get("beats") 为 None → 返回 None → 走 stub_opening。

    逐条校验：text 非空才要；agentId 必须在 world.agents 里，否则置 None（不信 LLM 瞎编
    的 id——它可能编造不存在的角色）。至少 1 条有效即返回 tuple，超过 _MAX_BEATS 截断。
    """
    data = parse_json_response(text)
    if data is None:
        return None
    raw_beats = data.get("beats")
    if not isinstance(raw_beats, list) or not raw_beats:
        return None

    valid_ids = {a.id for a in world.agents}
    beats: list[OpeningBeat] = []
    for item in raw_beats:
        if not isinstance(item, dict):
            continue
        beat_text = _clean(str(item.get("text") or ""))
        if not beat_text:
            continue
        agent_id = str(item.get("agentId") or "").strip() or None
        if agent_id is not None and agent_id not in valid_ids:
            agent_id = None  # 不信 LLM 瞎编 id：非在册角色一律归为世界级叙述
        beats.append(OpeningBeat(text=beat_text, agentId=agent_id))
        if len(beats) >= _MAX_BEATS:
            break
    return tuple(beats) if beats else None


# ---------------------------------------------------------------------------
# 确定性 stub 开场（无 key 兜底基线）—— 读真实世界缝合，换角色/布局必变
# ---------------------------------------------------------------------------
# 「人」的开场句模板：吃 (名字, 房间名, 内心独白, 口头禅)。文风参照 seed._OPENING_RAW。
_HUMAN_BEATS: tuple = (
    lambda name, room, mono, catch: f"{name}在{room}里忙活着，嘴上念叨「{mono}」。",
    lambda name, room, mono, catch: f"{room}那头，{name}停下手里的活，轻声嘀咕「{mono}」。",
    lambda name, room, mono, catch: (
        f"{name}窝在{room}，{f'顺口来了句「{catch}」，' if catch else ''}心里想着「{mono}」。"
    ),
    lambda name, room, mono, catch: f"{name}从{room}探出头，像想起什么似的说「{mono}」。",
)

# 「家电 / 家具」的开场句模板：措辞更「物件」一点（嗡、运转、守着），但同样有主体性。
_APPLIANCE_BEATS: tuple = (
    lambda name, room, mono, catch: f"{room}的{name}低低响了一声，像在说「{mono}」。",
    lambda name, room, mono, catch: (
        f"{name}在{room}的角落运转着，{f'「{catch}」，' if catch else ''}心里嘀咕「{mono}」。"
    ),
    lambda name, room, mono, catch: f"{room}里，{name}又嗡了一下，仿佛念叨着「{mono}」。",
    lambda name, room, mono, catch: f"{name}在{room}守着，闷声念叨「{mono}」。",
)

# 世界级氛围句：不属于某个角色（agentId=None），用来在角色不足 3 个时补足开场。
# 吃整个 world，措辞落到 clockLabel / name 上，保证「换布局也有活开场」。
_WORLD_BEATS: tuple = (
    lambda world: f"{world.clockLabel}的光落进屋里，{world.name}的各个角落慢慢热络起来。",
    lambda world: f"此刻的{world.name}醒着，每个房间都有自己的动静。",
    lambda world: "风从窗缝钻进来，屋里的物件都轻轻应了一声。",
    lambda world: f"{world.clockLabel}，家里的声音一层层叠上来，像有人在低声说话。",
    lambda world: "灯影在墙上晃了晃，这个家又开始过它自己的日子。",
)

# 内心独白 / 人设都空时的兜底片段（保证句子不出现空「」）。按 kind 分。
_MONO_FALLBACK: dict[str, str] = {
    "human": "有点心事没说出口",
    "appliance": "又没人搭理我了",
}


def stub_opening(world: World, seed: int = 0) -> tuple[OpeningBeat, ...]:
    """无 key / 解析失败时的确定性兜底开场。同一 (world, seed) 必得同一结果。

    做法：取前 _MAX_BEATS 个在册角色，各缝一句「此刻 TA 在做什么」的开场（据其房间名 +
    内心独白 + 口头禅）；不足 _MIN_BEATS 条时用世界级氛围句补齐。用 _pick(seed, salt) 选
    模板变体，salt 取角色序号——保证：
      · **换角色**（改 name / monologue / catchphrase）→ 句子明显变；
      · **换布局**（改房间名 / 角色 roomId）→ 句中的房间名随之变。
    这正是 F08 验收线「换角色/布局后 opening 明显变化」的确定性落点。
    """
    agents = list(world.agents)[:_MAX_BEATS]
    beats: list[OpeningBeat] = [
        OpeningBeat(text=_agent_beat_text(world, agent, seed, salt=i), agentId=agent.id)
        for i, agent in enumerate(agents)
    ]

    # 角色不足 3 个：用世界级氛围句补到 _MIN_BEATS。轮询取不同模板避免补出重复句。
    base = abs(int(seed) * 131 + 40 * 17 + 7) % len(_WORLD_BEATS)
    fill = 0
    while len(beats) < _MIN_BEATS:
        tmpl = _WORLD_BEATS[(base + fill) % len(_WORLD_BEATS)]
        beats.append(OpeningBeat(text=tmpl(world), agentId=None))
        fill += 1

    return tuple(beats[:_MAX_BEATS])


def _agent_beat_text(world: World, agent, seed: int, salt: int) -> str:
    """把一个角色缝成一句开场。房间名取自当前世界（换布局即变），措辞由 _pick 变体决定。"""
    room = _room_name(world, agent.roomId)
    mono = _clean(agent.monologue) or _first_sentence(agent.persona) or _MONO_FALLBACK.get(
        agent.kind, _MONO_FALLBACK["human"]
    )
    catch = _clean(agent.catchphrase or "")
    templates = _APPLIANCE_BEATS if agent.kind == "appliance" else _HUMAN_BEATS
    return _pick(templates, seed, salt)(agent.name, room, mono, catch)


# ---------------------------------------------------------------------------
# 内部纯函数
# ---------------------------------------------------------------------------
_STRIP_CHARS = "，,。.、；;　 \t\r\n"


def _clean(s: str | None) -> str:
    """去首尾空白与标点，避免嵌入模板后出现「念叨「…。」。」这类双标点断句。"""
    return (s or "").strip().strip(_STRIP_CHARS).strip()


def _first_sentence(text: str | None) -> str:
    """取一句话人设的首个短句，作为内心独白缺失时的兜底嵌入片段。"""
    t = _clean(text)
    for sep in ("。", "，", "！", "？", "…", "!", "?", ","):
        i = t.find(sep)
        if i > 0:
            return _clean(t[:i])
    return t


def _room_name(world: World, room_id: str) -> str:
    """房间 id → 房间名；找不到就退回 id（降级不崩，与 engine._room_name 同口径）。"""
    for r in world.rooms:
        if r.id == room_id:
            return r.name
    return room_id


def _pick(seq: tuple, seed: int, salt: int):
    """确定性挑选：同一 (seed, salt) 稳定命中同一元素；换 seed 即换变体组合。

    公式与 personas._pick / 前端 personaDraft._pick 对齐（seed*131 + salt*17 + 7 取模），
    让开场 stub 与人设 stub 同构，可复现红线一致。
    """
    i = abs(int(seed) * 131 + salt * 17 + 7) % len(seq)
    return seq[i]
