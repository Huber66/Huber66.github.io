"""Agent 人设修订草稿合成（P0-1）。

创建草稿（personas.py）回答的是“这个角色从无到有是谁”；本模块回答的是：
“这个已经活在世界里的角色，从此刻起应该如何被重新理解”。两者都产 persona 字段，
但修订必须保留旧历史，只更新当前 persona 锚，并生成面向用户的新旧对照摘要。

红线：
1. 无 key / LLM 失败也必须返回完整草稿。
2. 不修改 kind。
3. 输出文案生活化，不出现“人设已更新”这类软件感文本。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Literal

from .llm import ChatModel, parse_json_response
from .models import Agent, Mood
from .personas import coerce_mood

logger = logging.getLogger("cyberhome.persona_revision")

PersonaRevisionMode = Literal["adjust", "rewrite"]
PersonaRevisionVisibility = Literal["silent", "public"]


@dataclass(frozen=True)
class PersonaRevisionAnswers:
    """人设修订的引导答案。字段对齐前端 PersonaRevisionAnswers。"""

    freeform: str = ""
    caresAbout: str = ""
    catchphrase: str = ""
    angryReaction: str = ""
    happyReaction: str = ""
    clashWith: str = ""


@dataclass(frozen=True)
class PersonaRevisionDraft:
    """人设修订草稿。字段对齐前端 PersonaRevisionDraft。"""

    agentId: str
    mode: PersonaRevisionMode
    name: str
    avatar: str
    persona: str
    monologue: str
    mood: Mood
    catchphrase: str
    summary: str
    visibilitySuggestion: PersonaRevisionVisibility
    publicEventText: str


def persona_revision_draft_to_dict(draft: PersonaRevisionDraft) -> dict:
    return asdict(draft)


def synthesize_persona_revision_draft(
    chat_model: ChatModel,
    agent: Agent,
    answers: PersonaRevisionAnswers,
    *,
    mode: str = "adjust",
    seed: int = 0,
    max_output_tokens: int = 520,
) -> PersonaRevisionDraft:
    """生成修订草稿。纯合成、无副作用。

    先生成确定性 fallback，再尝试 LLM；任何异常或无效 JSON 都退回 fallback。
    """
    normalized_mode = _coerce_mode(mode)
    fallback = stub_persona_revision_draft(agent, answers, mode=normalized_mode, seed=seed)
    try:
        messages = assemble_persona_revision_prompt(agent, answers, mode=normalized_mode, seed=seed)
        resp = chat_model.chat(messages, max_output_tokens=max_output_tokens)
        parsed = parse_persona_revision_draft(resp.text, agent, normalized_mode, fallback)
    except Exception as exc:
        logger.warning("persona revision synth failed err=%s; falling back to stub", type(exc).__name__)
        return fallback
    return parsed if parsed is not None else fallback


_SCHEMA_INSTRUCTIONS = (
    "【输出格式】你必须只返回一个 JSON 对象，不要包含解释文字或代码围栏。字段如下：\n"
    "- name (string): 角色显示名；通常沿用原名，除非用户明确想改名。\n"
    "- avatar (string): 外观 seed；通常沿用原值，除非用户明确想改外观。\n"
    "- persona (string): 从此刻起的当前人设锚，2-3 个生活化短句，不要罗列属性。\n"
    "- monologue (string): TA 此刻新的内心独白。\n"
    "- catchphrase (string): 口头禅；没有就空字符串。\n"
    "- mood (string): 取值仅限 [happy, calm, tired, grumpy, lonely]。\n"
    "- summary (string): 面向用户的新旧变化摘要，生活化，不说“配置/人设更新”。\n"
    "- visibilitySuggestion (string): public 或 silent。大幅变化建议 public，轻微变化建议 silent。\n"
    "- publicEventText (string): 若用户选择公开，进入事件流的一句生活化旁白。"
)


def assemble_persona_revision_prompt(
    agent: Agent,
    answers: PersonaRevisionAnswers,
    *,
    mode: PersonaRevisionMode,
    seed: int = 0,
) -> list[dict]:
    """组装修订 prompt。system 必含 JSON，触发 StubChatModel 的 JSON 分支。"""
    mode_label = "在旧设定上调整" if mode == "adjust" else "完全重写当前设定"
    system = (
        "你是 CyberHome 的家庭叙事编辑。用户不是在删除过去，而是在让一个已经存在的角色"
        "从此刻起被重新理解。旧事件仍然发生过，新设定只影响未来。请保持温暖、克制、生活化，"
        "不要写软件操作感文案。\n\n"
        + _SCHEMA_INSTRUCTIONS
    )
    lines = [
        f"修订模式：{mode_label}",
        "当前角色：",
        f"- id：{agent.id}",
        f"- kind：{agent.kind}（不可修改）",
        f"- name：{agent.name}",
        f"- avatar seed：{agent.avatar}",
        f"- persona：{agent.persona}",
        f"- monologue：{agent.monologue}",
        f"- mood：{agent.mood}",
        f"- catchphrase：{agent.catchphrase or ''}",
        "",
        "用户这次的修订意图：",
    ]
    _append_answer(lines, "自由补充", answers.freeform)
    _append_answer(lines, "从此刻起最放不下的事", answers.caresAbout)
    _append_answer(lines, "新的口头禅", answers.catchphrase)
    _append_answer(lines, "生气时的新反应", answers.angryReaction)
    _append_answer(lines, "开心时的新反应", answers.happyReaction)
    _append_answer(lines, "更容易产生摩擦的人或物", answers.clashWith)
    if seed > 0:
        lines.append("（请换一种措辞重新生成，但不要改变用户的核心意图。）")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_persona_revision_draft(
    text: str | None,
    agent: Agent,
    mode: PersonaRevisionMode,
    fallback: PersonaRevisionDraft,
) -> PersonaRevisionDraft | None:
    data = parse_json_response(text)
    if data is None:
        return None
    persona = _clean(str(data.get("persona") or ""))
    if not persona:
        return None

    draft = PersonaRevisionDraft(
        agentId=agent.id,
        mode=mode,
        name=_clean(str(data.get("name") or "")) or fallback.name,
        avatar=str(data.get("avatar") or "").strip() or fallback.avatar,
        persona=persona,
        monologue=_clean(str(data.get("monologue") or "")) or fallback.monologue,
        mood=coerce_mood(data.get("mood"), default=fallback.mood),
        catchphrase=_clean(str(data.get("catchphrase") or "")) or fallback.catchphrase,
        summary=_clean(str(data.get("summary") or "")) or fallback.summary,
        visibilitySuggestion=_coerce_visibility(data.get("visibilitySuggestion"), fallback.visibilitySuggestion),
        publicEventText=_clean(str(data.get("publicEventText") or "")) or fallback.publicEventText,
    )
    return _ensure_public_event_text(agent, draft)


def stub_persona_revision_draft(
    agent: Agent,
    answers: PersonaRevisionAnswers,
    *,
    mode: PersonaRevisionMode = "adjust",
    seed: int = 0,
) -> PersonaRevisionDraft:
    core = _fallback_core(agent, answers)
    behavior = _behavior_sentence(agent, answers)
    extra = _extra_sentence(answers)
    persona = _persona_sentence(agent, core, behavior, extra, mode=mode, seed=seed)
    monologue = _pick(_MONOLOGUE_TEMPLATES, seed, 3)(core)
    mood = _infer_mood(agent, answers)
    catchphrase = _clean(answers.catchphrase) or (agent.catchphrase or "")
    summary = _summarize_change(agent, persona=persona, monologue=monologue, catchphrase=catchphrase)
    visibility = _suggest_visibility(agent, answers, mood=mood, persona=persona)
    draft = PersonaRevisionDraft(
        agentId=agent.id,
        mode=mode,
        name=agent.name,
        avatar=agent.avatar,
        persona=persona,
        monologue=monologue,
        mood=mood,
        catchphrase=catchphrase,
        summary=summary,
        visibilitySuggestion=visibility,
        publicEventText="",
    )
    return _ensure_public_event_text(agent, draft)


def _ensure_public_event_text(agent: Agent, draft: PersonaRevisionDraft) -> PersonaRevisionDraft:
    if draft.publicEventText:
        return draft
    return PersonaRevisionDraft(
        agentId=draft.agentId,
        mode=draft.mode,
        name=draft.name,
        avatar=draft.avatar,
        persona=draft.persona,
        monologue=draft.monologue,
        mood=draft.mood,
        catchphrase=draft.catchphrase,
        summary=draft.summary,
        visibilitySuggestion=draft.visibilitySuggestion,
        publicEventText=f"这个家像是忽然更懂{agent.name}了：{draft.summary}",
    )


def _coerce_mode(value: str) -> PersonaRevisionMode:
    return "rewrite" if value == "rewrite" else "adjust"


def _coerce_visibility(value, default: PersonaRevisionVisibility = "silent") -> PersonaRevisionVisibility:
    return value if value in ("silent", "public") else default


def _append_answer(lines: list[str], label: str, value: str) -> None:
    v = _clean(value)
    if v:
        lines.append(f"- {label}：{v}")


def _fallback_core(agent: Agent, answers: PersonaRevisionAnswers) -> str:
    return (
        _clean(answers.caresAbout)
        or _clean(answers.freeform)
        or agent.persona.replace("。", "\n").splitlines()[0][:28]
        or "这个家的冷暖"
    )


_PERSONA_TEMPLATES = (
    lambda name, core, behavior, extra: f"{name}从今天起更像一个把{core}放在心上的存在。{behavior}{extra}",
    lambda name, core, behavior, extra: f"现在的{name}，最藏不住的是对{core}的牵挂。{behavior}{extra}",
    lambda name, core, behavior, extra: f"{name}被重新理解成了这样：嘴上未必承认，心里却总惦记{core}。{behavior}{extra}",
)

_REWRITE_TEMPLATES = (
    lambda name, core, behavior, extra: f"从今天起，{name}像是换了一种生活的重心：{core}。{behavior}{extra}",
    lambda name, core, behavior, extra: f"{name}的新样子更鲜明了：TA 会把{core}看得很重。{behavior}{extra}",
    lambda name, core, behavior, extra: f"这个家重新认识了{name}：{core}成了 TA 心里最响的一件事。{behavior}{extra}",
)

_MONOLOGUE_TEMPLATES = (
    lambda core: f"{core}这件事，我还是得放在心上。",
    lambda core: f"别看我没说，{core}我一直记着。",
    lambda core: f"从今天起，{core}不能再随便糊弄过去了。",
)


def _persona_sentence(
    agent: Agent,
    core: str,
    behavior: str,
    extra: str,
    *,
    mode: PersonaRevisionMode,
    seed: int,
) -> str:
    templates = _REWRITE_TEMPLATES if mode == "rewrite" else _PERSONA_TEMPLATES
    return _pick(templates, seed, 1)(agent.name, core, behavior, extra)


def _behavior_sentence(agent: Agent, answers: PersonaRevisionAnswers) -> str:
    angry = _clean(answers.angryReaction)
    happy = _clean(answers.happyReaction)
    if angry and happy:
        return f"惹急了会{angry}，顺心时又会{happy}。"
    if angry:
        return f"惹急了会{angry}。"
    if happy:
        return f"顺心时会{happy}。"
    return (
        "平时有点固执，但真到了要出力的时候从不含糊。"
        if agent.kind == "appliance"
        else "平时不把话说满，但情绪都会藏在小动作里。"
    )


def _extra_sentence(answers: PersonaRevisionAnswers) -> str:
    clash = _clean(answers.clashWith)
    freeform = _clean(answers.freeform)
    parts: list[str] = []
    if clash:
        parts.append(f"和{clash}凑在一起时，关系会更有火花。")
    if freeform:
        parts.append(f"还有一点很重要：{freeform}。")
    return "".join(parts)


_MOOD_SIGNALS = (
    ("grumpy", "angry", ("摔", "吼", "骂", "凶", "爆", "砸", "甩", "拍桌", "翻脸", "冷战", "不理", "发火", "炸", "瞪", "怼", "杠", "黑脸", "臭脸")),
    ("happy", "happy", ("笑", "跳", "唱", "蹦", "哼", "乐", "美滋滋", "雀跃", "手舞足蹈", "咧嘴", "眉开眼笑", "撒欢", "嗨")),
    ("tired", "all", ("累", "懒", "困", "叹气", "摆烂", "躺", "不想动", "没力气", "疲", "丧", "无所谓")),
    ("lonely", "all", ("一个人", "没人", "孤单", "冷清", "寂寞", "空荡", "独自", "落单", "冷落", "等人", "盼着")),
)


def _infer_mood(agent: Agent, answers: PersonaRevisionAnswers) -> Mood:
    angry = answers.angryReaction or ""
    happy = answers.happyReaction or ""
    all_text = f"{answers.freeform} {answers.caresAbout} {angry} {happy} {answers.clashWith}"
    for mood, scope, keywords in _MOOD_SIGNALS:
        text = angry if scope == "angry" else happy if scope == "happy" else all_text
        if any(kw in text for kw in keywords):
            return mood  # type: ignore[return-value]
    return agent.mood


def _summarize_change(agent: Agent, *, persona: str, monologue: str, catchphrase: str) -> str:
    changes: list[str] = []
    if persona != agent.persona:
        changes.append("这个家的叙述更靠近 TA 的新牵挂")
    if monologue != agent.monologue:
        changes.append("TA 心里反复念着的事变了")
    if (catchphrase or "") != (agent.catchphrase or ""):
        changes.append("TA 说话时会带出新的口头禅")
    return (
        "，".join(changes) + "。"
        if changes
        else "这次更像是轻轻擦亮原本的设定，没有推翻 TA 过去的样子。"
    )


def _suggest_visibility(
    agent: Agent,
    answers: PersonaRevisionAnswers,
    *,
    mood: Mood,
    persona: str,
) -> PersonaRevisionVisibility:
    text = f"{answers.freeform} {answers.caresAbout} {answers.angryReaction} {answers.happyReaction} {answers.clashWith}"
    large_change = (
        mood != agent.mood
        or persona != agent.persona and len(_clean(answers.freeform)) >= 18
        or any(k in text for k in ("从今天", "以后", "再也", "秘密", "搬", "离开", "原来", "其实"))
    )
    return "public" if large_change else "silent"


_STRIP_CHARS = "，,。.、；;　 \t\r\n"


def _clean(s: str | None) -> str:
    return (s or "").strip().strip(_STRIP_CHARS).strip()


def _pick(seq, seed: int, salt: int):
    i = abs(int(seed) * 131 + salt * 17 + 11) % len(seq)
    return seq[i]
