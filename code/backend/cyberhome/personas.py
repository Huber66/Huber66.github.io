"""角色人设合成 —— 引导问答 → LLM 合成结构化草稿（F07b / 决策 5）。

「万物皆 Agent」的灵魂环节：用户不面对空白 prompt 框，而是回答几个具体小问题，
本模块把问答「捏」成一段读起来像人写的 persona + 一句内心独白 + 一个初始 mood。
是 POST /personas/draft 的核心，**纯合成、无副作用**：不写 world、不落库、不碰记忆。

三条设计红线钉在这里：
1. **无 key 也能跑**：合成走 ChatModel。无 key 时工厂给 StubChatModel，它对「含 JSON
   要求」的 prompt 返回结构合法但**不含 persona 字段**的占位 JSON——本模块据此判定
   「LLM 未给出有效草稿」，回退到基于问答的确定性 stub 草稿（stub_persona_draft）。
   故无论有无 key，端点都拿得到一份完整合法草稿，绝不 500。
2. **mood 必是 Mood 字面量**：LLM 返回非法 mood 一律归一到 calm（coerce_mood）。
3. **确定性可复现**：stub 草稿同一 (answers, seed) 必得同一结果；「重新生成」= seed+1
   换一套措辞组合（对齐前端 personaDraft.ts 的 stub，语义迁移到后端）。

prompt 组装刻意不复用 prompting.assemble_prompt——那是「角色在情境下如何反应」的私有
视角组装（吃 world/trigger/memory），与「把问答合成人设」是两件事。分开更内聚（优雅）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, get_args

from .llm import ChatModel, parse_json_response
from .models import AgentKind, Mood

logger = logging.getLogger("cyberhome.personas")

# 合法 mood 集合，从 Literal 单一事实源取（避免硬编码漂移）。
_VALID_MOODS = frozenset(get_args(Mood))

_KIND_LABEL = {"human": "人类角色", "appliance": "家电 / 家具"}


# ---------------------------------------------------------------------------
# 对外数据形状：frozen dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PersonaAnswers:
    """引导问答的原始答案（全部允许留空，stub 会兜底）。字段对齐前端 CreationAnswers。

    freeform 是「自由 prompt」补充（前端可选传入），非空时融入 persona 与 LLM prompt。
    """

    kind: AgentKind
    name: str = ""
    avatar: str = ""
    catchphrase: str = ""
    caresAbout: str = ""      # 最放不下、最爱操心的事
    angryReaction: str = ""   # 生气时会怎样（mood 推断主要看它）
    happyReaction: str = ""   # 开心时会怎样
    clashWith: str = ""       # 家里谁最容易和 TA 处不来
    freeform: str = ""        # 自由补充（可选）


@dataclass(frozen=True)
class PersonaDraft:
    """合成出的角色草稿。字段对齐前端 PersonaDraft（除 paletteId/type）。

    这是 /personas/draft 的响应内核，也是 /agents 创建请求的主体。mood 恒为合法 Mood。
    """

    kind: AgentKind
    name: str
    avatar: str
    persona: str      # 一句话人设（2-3 短句，读起来像人写的）
    monologue: str    # 初始内心独白（常驻低调气泡，保证「一进门就是活的」）
    mood: Mood
    catchphrase: str


# ---------------------------------------------------------------------------
# 主入口：合成草稿（chat → parse → 校验 → stub 兜底，绝不抛）
# ---------------------------------------------------------------------------
def synthesize_persona_draft(
    chat_model: ChatModel,
    answers: PersonaAnswers,
    *,
    seed: int = 0,
    max_output_tokens: int = 420,
) -> PersonaDraft:
    """把引导问答合成一份角色草稿。**纯合成、无副作用。**

    流程：先算出确定性 stub 草稿作 baseline（也是逐字段兜底来源）→ 调 LLM → 解析。
    LLM 给出有效草稿则用它（缺字段用 stub 兜底），否则整体退回 stub。任何异常都
    吞掉后退回 stub——保证端点绝不 500（无 key 闭环不断的落点）。
    """
    fallback = stub_persona_draft(answers, seed)
    try:
        messages = assemble_persona_prompt(answers, seed=seed)
        resp = chat_model.chat(messages, max_output_tokens=max_output_tokens)
        draft = parse_persona_draft(resp.text, answers, fallback)
    except Exception as exc:  # 合成失败绝不抛给端点：记脱敏摘要后退回 stub
        logger.warning("persona synth failed err=%s; falling back to stub", type(exc).__name__)
        return fallback
    return draft if draft is not None else fallback


# ---------------------------------------------------------------------------
# prompt 组装：system 段写死含 "JSON" 的 schema（触发 StubChatModel 的 JSON 分支）
# ---------------------------------------------------------------------------
_SCHEMA_INSTRUCTIONS = (
    "【输出格式】你必须只返回一个 JSON 对象，不要包含任何多余的解释文字或代码围栏之外的内容。"
    "字段如下：\n"
    "- name (string): TA 的名字。\n"
    "- avatar (string): 一个最能代表 TA 的 emoji。\n"
    "- persona (string): 一句话人设，2-3 个短句，读起来像人随口写下的，不要罗列属性。\n"
    "- monologue (string): TA 此刻心里正想着的一句内心独白（常驻低调，像自言自语）。\n"
    "- catchphrase (string): TA 挂在嘴边的口头禅；没有就留空字符串。\n"
    "- mood (string): TA 此刻的初始心情，取值仅限 [happy, calm, tired, grumpy, lonely]。\n"
    "示例：{\"name\": \"...\", \"avatar\": \"🙂\", \"persona\": \"...\", "
    "\"monologue\": \"...\", \"catchphrase\": \"...\", \"mood\": \"calm\"}"
)


def assemble_persona_prompt(answers: PersonaAnswers, *, seed: int = 0) -> list[dict]:
    """把引导问答组装成 OpenAI 风格 messages。system 写角色扮演元规则 + JSON schema。

    seed>0（重新生成）时在 user 段追加一句「换一种感觉」，让真 LLM 的重生成也有变化；
    StubChatModel 忽略语义、始终走 stub 兜底，变化由 stub 的 seed 变体承载。
    """
    kind_label = _KIND_LABEL.get(answers.kind, answers.kind)
    system_content = (
        f"你是一位擅长塑造鲜活角色的作者。用户想在一个「有温度的家」里创造一个「{kind_label}」，"
        "TA 和家里其他成员一样是有主体性、有脾气、有牵挂的存在（万物皆 Agent，人与家电平权）。"
        "请依据用户的零散回答，把 TA 捏成一个立体、可爱、可信的角色。"
        "语气温暖、生活化，不要冷冰冰地罗列设定。\n\n"
        + _SCHEMA_INSTRUCTIONS
    )

    lines = [f"请为我创造一个{kind_label}。以下是我对 TA 的零散想法（可能有留空，请你合理补全）："]
    _append_answer(lines, "名字", answers.name)
    _append_answer(lines, "emoji 头像", answers.avatar)
    _append_answer(lines, "口头禅", answers.catchphrase)
    _append_answer(lines, "TA 最放不下、最爱操心的事", answers.caresAbout)
    _append_answer(lines, "TA 生起气来的样子", answers.angryReaction)
    _append_answer(lines, "TA 一高兴的样子", answers.happyReaction)
    _append_answer(lines, "家里谁最容易和 TA 处不来", answers.clashWith)
    _append_answer(lines, "其他补充", answers.freeform)
    if seed > 0:
        lines.append("（我对上一版不太满意，请换一种感觉、换一套措辞重新捏一版。）")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _append_answer(lines: list[str], label: str, value: str) -> None:
    v = _clean(value)
    if v:
        lines.append(f"- {label}：{v}")


# ---------------------------------------------------------------------------
# 响应解析：dict -> PersonaDraft（缺字段用 fallback 逐字段兜底）
# ---------------------------------------------------------------------------
def parse_persona_draft(
    text: str | None, answers: PersonaAnswers, fallback: PersonaDraft
) -> PersonaDraft | None:
    """把 LLM 返回解析成 PersonaDraft。

    返回 None 表示「LLM 未给出有效草稿」（解析失败或核心字段 persona 空缺）——调用方据此
    整体退回 stub。这正是无 key 路径的判定点：StubChatModel 的默认占位 JSON 不含 persona
    字段，故 persona 空 → 返回 None → 走 stub。有 key 的真 LLM 给了 persona 才认为有效，
    其余缺字段（name/avatar/monologue/catchphrase/mood）逐一用 fallback 兜底。

    kind 永远以 answers 为准（由模板选定，不信 LLM）；mood 经 coerce_mood 归一化。
    """
    data = parse_json_response(text)
    if data is None:
        return None

    persona = _clean(str(data.get("persona") or ""))
    if not persona:  # 核心字段缺失：视为 LLM 无有效草稿，交由上层走 stub
        return None

    return PersonaDraft(
        kind=answers.kind,  # 模板选定，不信 LLM
        name=_clean(str(data.get("name") or "")) or fallback.name,
        avatar=(str(data.get("avatar") or "").strip()) or fallback.avatar,
        persona=persona,
        monologue=_clean(str(data.get("monologue") or "")) or fallback.monologue,
        mood=coerce_mood(data.get("mood"), default=fallback.mood),
        catchphrase=_clean(str(data.get("catchphrase") or "")) or fallback.catchphrase,
    )


def coerce_mood(value, *, default: Mood = "calm") -> Mood:
    """把任意值归一到合法 Mood。非法/缺失一律落到 default（默认 calm，温和默认）。

    与 prompting._coerce_mood 的差异：那里 reaction 的 mood 缺省是「无变化」故归 None；
    这里 persona 必须有初始 mood，故非法归 calm（红线 6）。
    """
    if isinstance(value, str) and value in _VALID_MOODS:
        return value  # type: ignore[return-value]
    return default


# ---------------------------------------------------------------------------
# 确定性 stub 草稿（无 key 兜底基线）—— 语义迁移自前端 personaDraft.draftPersonaLocally
# ---------------------------------------------------------------------------
_AVATAR_POOL: dict[str, tuple[str, ...]] = {
    "human": ("🙂", "🧑", "👦", "👧", "🧓", "👵", "🧔", "👩", "🧒", "👨", "🐱", "🐶"),
    "appliance": ("🔌", "💡", "📺", "📻", "🧊", "❄️", "🍳", "🧺", "🪑", "🕰️", "🖨️", "🚪"),
}

_CARES_TEMPLATES: dict[str, tuple] = {
    "human": (
        lambda x: f"最放不下的，是{x}",
        lambda x: f"心里头一直惦记着{x}",
        lambda x: f"嘴上不说，其实最在意{x}",
    ),
    "appliance": (
        lambda x: f"打心底操心着{x}",
        lambda x: f"成天惦记的，是{x}",
        lambda x: f"别看是个物件，最在意{x}",
    ),
}

_ANGRY_TEMPLATES = (
    lambda x: f"一上火就{x}",
    lambda x: f"惹急了会{x}",
    lambda x: f"生起气来{x}",
)

_HAPPY_TEMPLATES = (
    lambda x: f"一高兴就{x}",
    lambda x: f"碰上顺心事就{x}",
    lambda x: f"开心起来{x}",
)

_CLASH_TEMPLATES = (
    lambda x: f"偏偏和{x}处不到一块儿",
    lambda x: f"唯独跟{x}凑一起就来火",
    lambda x: f"跟{x}准要拌两句嘴",
)

_MONOLOGUE_TEMPLATES: dict[str, tuple] = {
    "human": (
        lambda x: f"{x}…也不知道现在怎么样了。",
        lambda x: f"唉，{x}，我这心里总放不下。",
        lambda x: f"别催我，{x}的事我记着呢。",
    ),
    "appliance": (
        lambda x: f"{x}…哼，总得有人操心吧。",
        lambda x: f"又没人理会我了…{x}。",
        lambda x: f"我可盯着呢——{x}。",
    ),
}

# kind 相关兜底碎片：用户全留空也要出一个鲜活草稿。
_FALLBACK: dict[str, dict] = {
    "human": {
        "cares": "这一家人的冷暖",
        "angry": "绷着脸不吭声",
        "happy": "嘴角悄悄往上扬",
        "catchphrase": ("我跟你说", "哎哟喂", "听我的", "说真的"),
        "name": "新朋友",
    },
    "appliance": {
        "cares": "这个家还记不记得它",
        "angry": "嗡嗡地闹点小脾气",
        "happy": "格外卖力地转起来",
        "catchphrase": ("我提醒你", "要我说啊", "滴——请注意", "想当年"),
        "name": "新伙计",
    },
}

# mood 推断信号：粗略但有来由，主要看「生气反应」的措辞（迁移自前端 MOOD_SIGNALS）。
_MOOD_SIGNALS: tuple = (
    ("grumpy", "angry", ("摔", "吼", "骂", "凶", "爆", "砸", "甩", "拍桌", "翻脸", "冷战", "不理", "发火", "炸", "瞪", "怼", "杠", "黑脸", "臭脸")),
    ("happy", "happy", ("笑", "跳", "唱", "蹦", "哼", "乐", "美滋滋", "雀跃", "手舞足蹈", "咧嘴", "眉开眼笑", "撒欢", "嗨")),
    ("tired", "all", ("累", "懒", "困", "叹气", "摆烂", "躺", "不想动", "没力气", "疲", "丧", "无所谓")),
    ("lonely", "all", ("一个人", "没人", "孤单", "冷清", "寂寞", "空荡", "独自", "落单", "冷落", "等人", "盼着")),
)


def stub_persona_draft(answers: PersonaAnswers, seed: int = 0) -> PersonaDraft:
    """无 key / 解析失败时的确定性兜底草稿。同一 (answers, seed) 必得同一结果。

    与前端 draftPersonaLocally 语义对齐：清洗每个碎片 + kind 兜底 → 缝合 persona
    （在意的事 → 性格两面 → 摩擦，可选）→ 拿在意的事作内心独白 → 推断 mood → 头像/名字兜底。
    """
    kind = answers.kind if answers.kind in _FALLBACK else "human"
    fb = _FALLBACK[kind]

    cares = _clean(answers.caresAbout) or fb["cares"]
    angry = _clean(answers.angryReaction) or fb["angry"]
    happy = _clean(answers.happyReaction) or fb["happy"]
    clash = _clean(answers.clashWith)
    freeform = _clean(answers.freeform)
    catchphrase = _clean(answers.catchphrase) or _pick(fb["catchphrase"], seed, 5)

    cares_clause = _pick(_CARES_TEMPLATES[kind], seed, 1)(cares)
    angry_clause = _pick(_ANGRY_TEMPLATES, seed, 2)(angry)
    happy_clause = _pick(_HAPPY_TEMPLATES, seed, 3)(happy)
    clash_clause = _pick(_CLASH_TEMPLATES, seed, 4)(clash) if clash else ""
    persona = f"{cares_clause}。{angry_clause}，{happy_clause}。"
    if clash_clause:
        persona += f"{clash_clause}。"
    if freeform:  # 自由补充融入末尾（不重复标点）
        persona += f"{freeform}。"

    monologue = _pick(_MONOLOGUE_TEMPLATES[kind], seed, 6)(cares)
    mood = _infer_mood(answers)
    avatar = answers.avatar.strip() or _pick(_AVATAR_POOL[kind], seed, 7)
    name = _clean(answers.name) or fb["name"]

    return PersonaDraft(
        kind=kind, name=name, avatar=avatar, persona=persona,
        monologue=monologue, mood=mood, catchphrase=catchphrase,
    )


def _infer_mood(answers: PersonaAnswers) -> Mood:
    """根据问答推断初始 mood；无明显信号时落到 calm（温和默认）。"""
    angry = answers.angryReaction or ""
    happy = answers.happyReaction or ""
    all_text = f"{answers.caresAbout} {angry} {happy} {answers.clashWith} {answers.freeform}"
    for mood, scope, keywords in _MOOD_SIGNALS:
        text = angry if scope == "angry" else happy if scope == "happy" else all_text
        if any(kw in text for kw in keywords):
            return mood  # type: ignore[return-value]
    return "calm"


# ---------------------------------------------------------------------------
# 内部纯函数
# ---------------------------------------------------------------------------
_STRIP_CHARS = "，,。.、；;　 \t\r\n"


def _clean(s: str | None) -> str:
    """去首尾空白与标点，避免嵌入模板后出现「…吃饭。也不知道」这类断句。"""
    return (s or "").strip().strip(_STRIP_CHARS).strip()


def _pick(seq, seed: int, salt: int):
    """确定性挑选：同一 (seed, salt) 稳定命中同一元素；换 seed 即换变体组合。

    公式与前端 personaDraft._pick 对齐（seed*131 + salt*17 + 7）取模，保证前后端 stub 同构。
    """
    i = abs(int(seed) * 131 + salt * 17 + 7) % len(seq)
    return seq[i]
