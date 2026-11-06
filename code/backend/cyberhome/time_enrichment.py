"""Async enrichment for manual time advances.

The advance-time endpoint must return quickly, but the resulting life events
should still feel authored by the current world instead of fixed templates. This
module asks the configured chat model for a few extra life beats and falls back
to local prose if the model is unavailable, times out, or returns invalid JSON.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass

from .llm import ChatModel, parse_json_response
from .models import Agent, World, WorldEvent

logger = logging.getLogger("cyberhome.time_enrichment")


@dataclass(frozen=True)
class LifeBeat:
    text: str
    agentId: str | None = None


def synthesize_time_enrichment(
    chat_model: ChatModel,
    world: World,
    base_events: tuple[WorldEvent, ...],
    *,
    seed: int,
) -> tuple[LifeBeat, ...]:
    """Generate richer life beats for a manual time advance.

    The caller runs this off the request path. If the LLM path fails for any
    reason, return local fallback beats so the job still completes.
    """
    recent_texts = _recent_event_texts(world, base_events)
    group_scene = _group_move_scene(world, base_events)
    requires_llm_dialogue = group_scene.get("type") in {"group_move", "bedtime_copresent"}
    provider = str(getattr(chat_model, "provider", ""))
    allow_required_fallback = provider == "stub"
    fallback = (
        _required_dialogue_fallback(world, group_scene, avoid=recent_texts)
        if requires_llm_dialogue and allow_required_fallback
        else _fallback_beats(world, base_events, seed=seed, avoid=recent_texts)
    )
    logger.info(
        "time enrichment start group_move=%s base_events=%d",
        requires_llm_dialogue, len(base_events),
    )
    try:
        resp = _fast_chat(
            chat_model,
            _messages(world, base_events),
            max_output_tokens=360 if requires_llm_dialogue else 520,
            temperature=0.82,
            timeout=45.0 if requires_llm_dialogue else 10.0,
            max_retries=1 if requires_llm_dialogue else 0,
        )
        beats = _parse_beats(
            resp.text,
            world,
            avoid=recent_texts,
            group_scene=group_scene if requires_llm_dialogue else None,
        )
        if resp.provider == "stub" or not beats:
            logger.info(
                "time enrichment using fallback provider=%s parsed_beats=%d group_move=%s",
                resp.provider, len(beats), requires_llm_dialogue,
            )
            if requires_llm_dialogue and resp.provider != "stub":
                return ()
            return fallback
        logger.info("time enrichment llm beats=%d group_move=%s", len(beats), requires_llm_dialogue)
        return beats
    except Exception as exc:
        if requires_llm_dialogue and not allow_required_fallback:
            logger.warning("time enrichment required dialogue llm failed err=%s; no local fallback", type(exc).__name__)
            return ()
        logger.warning("time enrichment llm failed err=%s; using fallback", type(exc).__name__)
        return fallback


def _fast_chat(
    chat_model: ChatModel,
    messages: list[dict],
    *,
    max_output_tokens: int,
    temperature: float,
    timeout: float,
    max_retries: int,
):
    quick_chat = getattr(chat_model, "quick_chat", None)
    if callable(quick_chat):
        return quick_chat(
            messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
    return chat_model.chat(
        messages,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )


def _messages(world: World, base_events: tuple[WorldEvent, ...]) -> list[dict]:
    group_scene = _group_move_scene(world, base_events)
    rooms = []
    for room in world.rooms:
        present = [a for a in world.agents if a.roomId == room.id]
        rooms.append(
            {
                "id": room.id,
                "name": room.name,
                "kind": room.kind,
                "agents": [
                    {
                        "id": a.id,
                        "name": a.name,
                        "kind": a.kind,
                        "mood": a.mood,
                        "persona": a.persona,
                        "monologue": a.monologue,
                        "catchphrase": a.catchphrase or "",
                    }
                    for a in present
                ],
            }
        )
    base = [
        {
            "source": e.source,
            "agentId": e.agentId,
            "text": e.text,
        }
        for e in base_events[-8:]
    ]
    recent = [
        {
            "source": e.source,
            "agentId": e.agentId,
            "text": e.text,
        }
        for e in world.events[-18:]
    ]
    return [
        {
            "role": "system",
            "content": (
                "你是 CyberHome 的事件流编剧。请基于真实房间、真实 Agent 状态，"
                "补写推进时间后的自然生活片段。家电/家具也是和人平权的角色，不是功能入口。"
                "不要照抄示例，不要写系统说明，不要生成固定套话。只输出 JSON。"
            ),
        },
        {
            "role": "user",
            "content": (
                "当前时间："
                f"{world.clockLabel}\n"
                f"基础事件：{base}\n"
                f"最近已经写过的事件，禁止复述或换皮重复：{recent}\n"
                f"房间与在场 Agent：{rooms}\n\n"
                f"本轮关键情境：{group_scene}\n"
                "请输出 2 条事件，要求：\n"
                "1. 如果本轮关键情境里有两个或更多 Agent 一起到同一房间，或深夜已经同在卧室，必须写成他们之间的真实对话，"
                "每个参与者最多说一句，每句对话单独一条事件，并填写说话者 agentId；\n"
                "2. 如果是深夜一起去卧室，要写睡前小声说话、互相惦记、一天收尾的状态，"
                "不要只写移动旁白或房间氛围；\n"
                "3. 对话要贴合各自 persona、mood、catchphrase 和当前房间，不要模板化，不要复述最近事件；\n"
                "4. 不要补房间氛围，不要多轮来回；两个人就输出两条，一人一句。\n\n"
                "JSON 格式必须是："
                '{"events":[{"agentId":"agent_id 或 null","text":"一句进入事件流的中文，80字以内"}]}'
            ),
        },
    ]


def _group_move_scene(world: World, base_events: tuple[WorldEvent, ...]) -> dict:
    moved_ids = [event.agentId for event in base_events if event.source == "move" and event.agentId]
    moved = [agent for agent_id in moved_ids if (agent := _agent(world, agent_id))]
    by_room: dict[str, list[Agent]] = {}
    for agent in moved:
        by_room.setdefault(agent.roomId, []).append(agent)
    groups = []
    for room_id, agents in by_room.items():
        if len(agents) < 2:
            continue
        room = next((r for r in world.rooms if r.id == room_id), None)
        groups.append(
            {
                "roomId": room_id,
                "roomName": room.name if room else room_id,
                "roomKind": room.kind if room else "",
                "time": world.clockLabel,
                "agents": [
                    {
                        "id": a.id,
                        "name": a.name,
                        "kind": a.kind,
                        "mood": a.mood,
                        "persona": a.persona,
                        "monologue": a.monologue,
                        "catchphrase": a.catchphrase or "",
                    }
                    for a in agents
                ],
            }
        )
    if groups:
        return {
            "type": "group_move",
            "instruction": "这些 Agent 本轮一起到同一房间，必须优先生成他们之间的对话事件。",
            "groups": groups,
        }
    bedtime = _bedtime_copresent_group(world)
    if bedtime is not None:
        return {
            "type": "bedtime_copresent",
            "instruction": "深夜卧室里已有两位以上 human Agent，必须优先生成睡前对话事件。",
            "groups": [bedtime],
        }
    return {"type": "none", "instruction": "没有多人同房移动；可按普通生活片段处理。"}


def _bedtime_copresent_group(world: World) -> dict | None:
    if world.phase != "lateNight":
        return None
    bedroom = next((room for room in world.rooms if room.kind == "bedroom"), None)
    if bedroom is None:
        return None
    present = [
        agent for agent in world.agents
        if agent.kind == "human" and agent.roomId == bedroom.id
    ]
    if len(present) < 2:
        return None
    return {
        "roomId": bedroom.id,
        "roomName": bedroom.name,
        "roomKind": bedroom.kind,
        "time": world.clockLabel,
        "agents": [
            {
                "id": a.id,
                "name": a.name,
                "kind": a.kind,
                "mood": a.mood,
                "persona": a.persona,
                "monologue": a.monologue,
                "catchphrase": a.catchphrase or "",
            }
            for a in present[:3]
        ],
    }


def _parse_beats(
    text: str,
    world: World,
    *,
    avoid: set[str],
    group_scene: dict | None = None,
) -> tuple[LifeBeat, ...]:
    data = parse_json_response(text)
    if not isinstance(data, dict):
        return _parse_loose_lines(text, world, avoid=avoid)
    raw_events = data.get("events")
    if not isinstance(raw_events, list):
        return _parse_loose_lines(text, world, avoid=avoid)
    valid_ids = {a.id for a in world.agents}
    target_ids = _dialogue_agent_ids(group_scene)
    max_beats = len(target_ids) if target_ids else 4
    beats: list[LifeBeat] = []
    seen: set[str] = set()
    seen_agents: set[str] = set()
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        line = _clean_text(item.get("text"))
        if not line or line in seen or _too_similar(line, avoid | seen):
            continue
        agent_id = item.get("agentId")
        if agent_id not in valid_ids:
            agent_id = None
        if target_ids:
            if agent_id not in target_ids or agent_id in seen_agents:
                continue
        if _contradicts_agent_location(line, world, agent_id):
            continue
        seen.add(line)
        if agent_id:
            seen_agents.add(agent_id)
        beats.append(LifeBeat(text=line, agentId=agent_id))
        if len(beats) >= max_beats:
            break
    return tuple(beats)


def _parse_loose_lines(text: str, world: World, *, avoid: set[str]) -> tuple[LifeBeat, ...]:
    """Accept useful non-JSON model output instead of discarding it outright."""
    valid_names = {a.name: a.id for a in world.agents}
    beats: list[LifeBeat] = []
    seen: set[str] = set()
    for raw in str(text or "").splitlines():
        line = raw.strip()
        line = re.sub(r"^[\-\*\d\.\、\)\s]+", "", line).strip()
        if not line or line.startswith("{") or line.startswith("["):
            continue
        if len(line) < 6:
            continue
        agent_id = None
        for name, candidate_id in valid_names.items():
            if line.startswith(f"{name}:") or line.startswith(f"{name}：") or line.startswith(name):
                agent_id = candidate_id
                line = re.sub(rf"^{re.escape(name)}[:：\s]*", "", line).strip()
                break
        line = _clean_text(line)
        if not line or line in seen or _too_similar(line, avoid | seen):
            continue
        if _contradicts_agent_location(line, world, agent_id):
            continue
        seen.add(line)
        beats.append(LifeBeat(text=line, agentId=agent_id))
        if len(beats) >= 4:
            break
    return tuple(beats)


def _dialogue_agent_ids(group_scene: dict | None) -> tuple[str, ...]:
    if not group_scene or group_scene.get("type") not in {"group_move", "bedtime_copresent"}:
        return ()
    groups = group_scene.get("groups")
    if not isinstance(groups, list) or not groups:
        return ()
    group = groups[0]
    if not isinstance(group, dict):
        return ()
    agents = group.get("agents")
    if not isinstance(agents, list):
        return ()
    out: list[str] = []
    for item in agents:
        if not isinstance(item, dict):
            continue
        agent_id = item.get("id")
        if isinstance(agent_id, str) and agent_id and agent_id not in out:
            out.append(agent_id)
        if len(out) >= 2:
            break
    return tuple(out)


def _clean_text(value) -> str:
    text = str(value or "").strip()
    text = text.replace("\n", " ").replace("\r", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text[:90]


def _recent_event_texts(world: World, base_events: tuple[WorldEvent, ...]) -> set[str]:
    return {
        _normalize_text(event.text)
        for event in (*world.events[-24:], *base_events)
        if event.text
    }


def _normalize_text(text: str) -> str:
    return "".join(ch for ch in str(text).strip() if not ch.isspace())


def _too_similar(text: str, avoid: set[str]) -> bool:
    norm = _normalize_text(text)
    if not norm:
        return True
    for old in avoid:
        if not old:
            continue
        if norm == old or norm in old or old in norm:
            return True
        overlap = len(set(norm) & set(old))
        scale = max(1, min(len(set(norm)), len(set(old))))
        if scale >= 8 and overlap / scale > 0.82:
            return True
    return False


def _contradicts_agent_location(text: str, world: World, agent_id: str | None) -> bool:
    if not agent_id:
        return False
    agent = _agent(world, agent_id)
    if agent is None:
        return False
    current_room_id = agent.roomId
    for room in world.rooms:
        if room.id == current_room_id or not room.name:
            continue
        room_name = re.escape(room.name)
        if re.search(rf"(到了|到|去了|去|进了|进入|走到|挪到|挪去|靠近|在){room_name}", text):
            return True
        if re.search(rf"{room_name}(里|门口|旁边|边上|最亮的地方)", text):
            return True
    return False


def _required_dialogue_fallback(
    world: World,
    group_scene: dict,
    *,
    avoid: set[str],
) -> tuple[LifeBeat, ...]:
    """Never let required co-present dialogue disappear silently."""
    groups = group_scene.get("groups")
    group = groups[0] if isinstance(groups, list) and groups else {}
    raw_agents = group.get("agents") if isinstance(group, dict) else None
    agents: list[Agent] = []
    if isinstance(raw_agents, list):
        for item in raw_agents:
            if not isinstance(item, dict):
                continue
            agent = _agent(world, item.get("id"))
            if agent is not None:
                agents.append(agent)

    if len(agents) < 2:
        bedroom = next((room for room in world.rooms if room.kind == "bedroom"), None)
        if bedroom is not None:
            agents = [
                agent for agent in world.agents
                if agent.kind == "human" and agent.roomId == bedroom.id
            ]

    if len(agents) < 2:
        return ()

    first, second = agents[:2]
    room_name = _room_name(world, first.roomId)
    first_hint = f"{first.catchphrase}，" if first.catchphrase else ""
    second_hint = f"{second.catchphrase}，" if second.catchphrase else ""
    candidates = [
        LifeBeat(
            agentId=first.id,
            text=(
                f"{first.name}在{room_name}把声音放低：「{first_hint}"
                "今天先歇吧，明早再把没说完的事捋清楚。」"
            ),
        ),
        LifeBeat(
            agentId=second.id,
            text=(
                f"{second.name}轻轻应了一声：「{second_hint}"
                "别把一天的念头都带到枕边，灯暗了，人也该松下来。」"
            ),
        ),
    ]
    beats: list[LifeBeat] = []
    local_avoid = set(avoid)
    for beat in candidates:
        _append_unique(beats, beat, local_avoid)
    return tuple(beats)


def _fallback_beats(
    world: World,
    base_events: tuple[WorldEvent, ...],
    *,
    seed: int,
    avoid: set[str],
) -> tuple[LifeBeat, ...]:
    rng = random.Random(seed)
    beats: list[LifeBeat] = []

    moved = [e for e in base_events if e.source == "move" and e.agentId]
    if moved:
        event = moved[-1]
        agent = _agent(world, event.agentId)
        room_name = _room_name(world, agent.roomId if agent else None)
        if agent is not None:
            _append_unique(
                beats,
                LifeBeat(
                    agentId=agent.id,
                    text=_choose_unique(
                        rng,
                        [
                            f"{agent.name}在{room_name}停住脚，先摸了摸门框，像是在确认自己是不是真的该来这里。",
                            f"{agent.name}进了{room_name}，没有立刻开口，只把刚才的表情藏进了屋里的暗处。",
                            f"{agent.name}到了{room_name}，脚步声轻得很，却把旁边的人和物都惊动了一点。",
                            f"{agent.name}靠近{room_name}里最亮的地方，像是想借这点光把话想明白。",
                        ],
                        avoid,
                    ),
                ),
                avoid,
            )

    if moved and beats:
        event = moved[-1]
        agent = _agent(world, event.agentId)
        if world.phase == "night" and agent is not None and agent.id == "agent_dad" and agent.roomId == "room_study":
            return tuple(beats[:1])

    room_agents = _conversation_room(world)
    if room_agents:
        room_name, first, second = room_agents
        if first.kind == "appliance":
            line = f"{first.name}在{room_name}里短促地响了一声，像是把{second.name}刚才那句话记在了自己的噪音里。"
        elif second.kind == "appliance":
            line = f"{first.name}看了看{second.name}，忽然觉得这件东西沉默归沉默，其实一直在替屋子守着场。"
        else:
            line = f"{first.name}和{second.name}在{room_name}搭了句话，声音不高，却像家里自然冒出的烟火气。"
        _append_unique(beats, LifeBeat(agentId=first.id, text=line), avoid)

    if not moved and len(beats) < 2:
        room = rng.choice(list(world.rooms)) if world.rooms else None
        if room is not None:
            _append_unique(
                beats,
                LifeBeat(
                    text=_choose_unique(
                        rng,
                        [
                            f"{room.name}里的光换了个角度，连沉默的东西都像重新站了一次队。",
                            f"{room.name}短暂安静下来，远处一点细响反而被衬得很清楚。",
                        ],
                        avoid,
                    )
                ),
                avoid,
            )

    return tuple(beats[:3])


def _choose_unique(rng: random.Random, options: list[str], avoid: set[str]) -> str:
    shuffled = list(options)
    rng.shuffle(shuffled)
    for option in shuffled:
        if not _too_similar(option, avoid):
            return option
    return shuffled[0] if shuffled else ""


def _append_unique(beats: list[LifeBeat], beat: LifeBeat, avoid: set[str]) -> None:
    if not beat.text or _too_similar(beat.text, avoid):
        return
    beats.append(beat)
    avoid.add(_normalize_text(beat.text))


def _agent(world: World, agent_id: str | None) -> Agent | None:
    return next((a for a in world.agents if a.id == agent_id), None)


def _room_name(world: World, room_id: str | None) -> str:
    room = next((r for r in world.rooms if r.id == room_id), None)
    return room.name if room else "房间"


def _conversation_room(world: World) -> tuple[str, Agent, Agent] | None:
    for room in world.rooms:
        present = [a for a in world.agents if a.roomId == room.id]
        if len(present) >= 2:
            return room.name, present[0], present[1]
    return None
