"""共处涌现 reactor —— bidding 发言权 + 结构化反应生成（F06 / 决策 26、8.3）。

Reactor 职责单一：给定「世界 + 触发 + 记忆」，返回「谁开口、说什么、心情/独白/记忆
有无变化」的结构化反应列表（AgentReaction）。bidding（谁沉默、谁开口、最多几人）
内含在 react 里——engine 只把返回的反应落成事件与 patch，不掺和「谁该说话」。

两条实现红线：
1. **可复现**（事件溯源，决策 32）：StubReactor 随机源可注入（构造传 Random），
   同一 (world, trigger) 重放必得同一结果。真 LLM 阶段靠固定 seed / 缓存维持。
2. **降级不崩**（决策 8.3）：LLMReactor.react 永不把异常抛给引擎——单个 agent 的
   LLM/解析失败就跳过它，整体意外则委托 fallback，最坏返回 []（全体沉默，合法）。
   日志只记脱敏摘要（agent id + 异常类型名），绝不落 prompt 全文 / 密钥。
"""

from __future__ import annotations

import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .llm import ChatModel
from .memory import MemoryStore
from .models import Agent, Mood, Trigger, World
from .prompting import assemble_prompt, parse_reaction

logger = logging.getLogger("cyberhome.reactor")

_LOCAL_PROVIDERS = {"stub", "fake", "test"}


@dataclass(frozen=True)
class AgentReaction:
    """一个角色对某情境的结构化反应。frozen：调用方拿到即定，不被下游悄改。

    line 是**纯台词**（不含「名字：」前缀，沉默时为 ""）——归属由 engine 落事件时的
    agentId 承载；reason/monologue 是私有心理（不外显）；mood/monologue/memory_note
    无变化时为 None。
    """

    agent_id: str
    speak: bool
    line: str                          # 纯台词（不含「名字：」前缀），沉默时为 ""
    reason: str = ""                   # 私有心理，不外显
    mood: Optional[Mood] = None        # 心情变化，无变化为 None
    monologue: Optional[str] = None    # 内心独白更新，无则 None
    memory_note: Optional[str] = None  # 值得记住的一句（F09 之后落库），无则 None


class Reactor(ABC):
    """把「谁在场 + 什么情境 + 记得什么」变成「谁开口、说什么、状态有无变化」。"""

    @abstractmethod
    def react(self, world: World, trigger: Trigger, memory: MemoryStore) -> list[AgentReaction]:
        """给定世界 + 触发 + 记忆，返回「开口者 / 状态有变者」的结构化反应。

        bidding 内含：只返回真正 speak 的（capped 1-2 人）+ 沉默但心情/独白有变者；
        允许全体沉默返回 []。engine 据此落事件（speak 的产 reaction 事件）与 patch
        （mood/monologue 折进 agent_updates）。
        """


class StubReactor(Reactor):
    """半随机 stub。rng 可注入以保证 step() 可复现（事件溯源前提）。

    文案对齐前端 mock（reactToOracle/makeMoveText），去掉「名字：」前缀（归属由
    engine 的 event.agentId 承载，台词只留纯内容）。真 LLM 由 LLMReactor 接管。
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def react(self, world: World, trigger: Trigger, memory: MemoryStore) -> list[AgentReaction]:
        if trigger.kind == "oracle":
            return self._react_oracle(world, trigger)
        if trigger.kind == "move":
            return self._react_move(world, trigger)
        if trigger.kind == "catchup":
            return self._react_catchup(world, trigger)
        return []

    # ---- oracle：全体里 bidding 挑 1-2 人开口（有人沉默才真实，决策 26）----
    def _react_oracle(self, world: World, trigger: Trigger) -> list[AgentReaction]:
        agents = list(world.agents)
        if not agents:
            return []
        text = (trigger.text or "").strip()
        # rng 调用序刻意对齐旧 pick_reactors + react_to_oracle，保 step() 可复现
        count = min(len(agents), self._rng.randint(1, 2))
        chosen = self._rng.sample(agents, count)
        return [
            AgentReaction(agent_id=a.id, speak=True, line=self._oracle_line(a, text))
            for a in chosen
        ]

    def _oracle_line(self, agent: Agent, oracle: str) -> str:
        """旧 react_to_oracle 的模板，去掉「名字：」前缀后的纯台词。"""
        cp = f"{agent.catchphrase}，" if agent.catchphrase else ""
        gist = oracle.strip()[:18]
        if agent.kind == "appliance":
            replies = [
                f"{cp}“{gist}”？这跟我可有关系了。",
                f"{cp}要我说，{gist}这事儿，得看我心情。",
                f"（嗡地响了一下）{cp}听见了听见了。",
                f"{cp}想当年这种事，我见得多了。",
            ]
            return self._rng.choice(replies)
        by_mood = {
            "happy": [f"{cp}好呀好呀，“{gist}”，那我可得张罗起来！", f"{cp}听着就高兴，这事儿我喜欢。"],
            "calm": [f"{cp}嗯…“{gist}”，我记下了。", f"{cp}行，那就顺其自然吧。"],
            "tired": [f"{cp}啊……“{gist}”？等我缓缓啊。", f"{cp}唉，知道了知道了。"],
            "grumpy": [f"{cp}“{gist}”？哼，早说啊。", f"{cp}我就说吧，迟早的事。"],
            "lonely": [f"{cp}“{gist}”…那你会常回来看看吗？", f"{cp}嗯，有你这句话就够了。"],
        }
        return self._rng.choice(by_mood[agent.mood])

    # ---- move：移动者产一条共处叙事（line 即那句叙事）----
    def _react_move(self, world: World, trigger: Trigger) -> list[AgentReaction]:
        moved = next((a for a in world.agents if a.id == trigger.agentId), None)
        if moved is None:
            return []
        room_name = self._room_name(world, trigger.roomId)
        others = [a for a in world.agents if a.roomId == trigger.roomId and a.id != moved.id]
        return [
            AgentReaction(agent_id=moved.id, speak=True, line=self._move_text(moved, room_name, others))
        ]

    def _move_text(self, moved: Agent, room_name: str, others: list[Agent]) -> str:
        if not others:
            return f"{moved.name} 走进了{room_name}，四下看了看，暂时只有自己。"
        other = self._rng.choice(others)
        cp = f"{moved.catchphrase}，" if moved.catchphrase else ""
        templates = [
            f"{moved.name} 来到{room_name}，正好碰上 {other.name}。{cp}“你也在这儿呢。”",
            f"{moved.name} 挪进了{room_name}，和 {other.name} 对上了眼，空气里多了点故事。",
            f"{moved.name} 进{room_name}的时候，{other.name} 抬头看了一眼——这俩放一块儿，有点意思。",
        ]
        return self._rng.choice(templates)

    # ---- catchup：某房间在场者里挑 1 人，产一句「你不在时的生活切片」叙事（F10）----
    def _react_catchup(self, world: World, trigger: Trigger) -> list[AgentReaction]:
        """补演分片：trigger.roomId 定位这一片发生在哪个房间，取该房间在场者挑一人开口。

        rng 可注入 → 同一 (world, sub_trigger, 固定 rng) 必得同一句（可复现，事件溯源前提）。
        line 是**纯叙事句**（无「名字：」前缀，归属由 engine 落事件的 agentId 承载）；
        房间空则返回 []（engine 里该片不产事件，降级不崩）。
        """
        present = [a for a in world.agents if a.roomId == trigger.roomId]
        if not present:
            return []
        room_name = self._room_name(world, trigger.roomId)
        if len(present) >= 2:
            return self._catchup_dialogue(present, room_name)
        who = self._rng.choice(present)
        return [AgentReaction(agent_id=who.id, speak=True, line=self._catchup_line(who, room_name))]

    def _catchup_dialogue(self, present: list[Agent], room_name: str) -> list[AgentReaction]:
        """同屋 Agent 的小对话。家电/家具也平权参与，尽量显出物性。"""
        ordered = list(present)
        if len(ordered) > 3:
            ordered = self._rng.sample(ordered, 3)
        first = ordered[0]
        second = ordered[1]
        out = [
            AgentReaction(
                agent_id=first.id,
                speak=True,
                line=self._dialogue_opening(first, second, room_name),
                memory_note=f"{first.name}在{room_name}和{second.name}搭了句话。",
            ),
            AgentReaction(
                agent_id=second.id,
                speak=True,
                line=self._dialogue_reply(second, first),
                memory_note=f"{second.name}回应了{first.name}，两人在{room_name}有了点小动静。",
            ),
        ]
        if len(ordered) >= 3:
            third = ordered[2]
            out.append(
                AgentReaction(
                    agent_id=third.id,
                    speak=True,
                    line=self._dialogue_third(third, first, second),
                    memory_note=f"{third.name}旁听了{first.name}和{second.name}的对话。",
                )
            )
        return out

    def _dialogue_opening(self, agent: Agent, other: Agent, room_name: str) -> str:
        cp = f"{agent.catchphrase}，" if agent.catchphrase else ""
        if agent.kind == "appliance":
            return f"{agent.name}在{room_name}轻轻响了一下：“{cp}{other.name}，你听见我这动静了吗？”"
        return f"{agent.name}看向{other.name}：“{cp}你也在这儿啊，今天这屋里还挺热闹。”"

    def _dialogue_reply(self, agent: Agent, other: Agent) -> str:
        cp = f"{agent.catchphrase}，" if agent.catchphrase else ""
        if agent.kind == "appliance":
            return f"{agent.name}慢吞吞接话：“{cp}我一直在呢，别把我当摆设。”"
        if agent.mood == "grumpy":
            return f"{agent.name}哼了一声：“{cp}{other.name}你少来，我正烦着呢。”"
        if agent.mood == "happy":
            return f"{agent.name}笑起来：“{cp}这不正好，有人说话才像个家。”"
        return f"{agent.name}点点头：“{cp}嗯，我听着呢。”"

    def _dialogue_third(self, agent: Agent, first: Agent, second: Agent) -> str:
        cp = f"{agent.catchphrase}，" if agent.catchphrase else ""
        if agent.kind == "appliance":
            return f"{agent.name}在一旁插了一句：“{cp}你们聊归聊，可别忘了我也在场。”"
        return f"{agent.name}在旁边补了一句：“{cp}{first.name}和{second.name}这话题，听着还有点意思。”"

    def _catchup_line(self, agent: Agent, room_name: str) -> str:
        """一句凝练的生活切片文案。第三人称叙事（进事件流的「补记」），不带冒号前缀。"""
        cp = f"{agent.catchphrase}，" if agent.catchphrase else ""
        if agent.kind == "appliance":
            templates = [
                f"你不在时，{agent.name}在{room_name}自顾自嗡了几声，{cp}权当打发时间。",
                f"{room_name}里，{agent.name}安静地运转着，{cp}偶尔咕哝一句没人听。",
                f"{agent.name}在{room_name}守了这半天，{cp}把该操的心又操了一遍。",
            ]
            return self._rng.choice(templates)
        by_mood = {
            "happy": [f"{agent.name}在{room_name}里哼着歌忙活，{cp}这半天过得挺舒坦。"],
            "calm": [f"{agent.name}在{room_name}待着，{cp}翻翻这个看看那个，日子照旧。"],
            "tired": [f"{agent.name}在{room_name}打了个盹，{cp}醒来时天色都变了。"],
            "grumpy": [f"{agent.name}在{room_name}嘟囔了半天，{cp}也没个人搭腔。"],
            "lonely": [f"{agent.name}在{room_name}坐了好久，{cp}盼着有人回来。"],
        }
        return self._rng.choice(by_mood[agent.mood])

    @staticmethod
    def _room_name(world: World, room_id: Optional[str]) -> str:
        for r in world.rooms:
            if r.id == room_id:
                return r.name
        return room_id or "某个房间"


class LLMReactor(Reactor):
    """真 LLM 反应器：assemble_prompt → chat → parse_reaction → AgentReaction。

    降级红线（决策 8.3）：react 永不把异常抛给引擎。单个 agent 的 chat/parse 失败就
    跳过它（记脱敏 warning），整体意外委托 fallback（默认 StubReactor），最坏返回 []。
    无 key 时 chat_model 是 StubChatModel，其在「prompt 含 JSON 要求」时返回结构合法
    JSON（prompting 的 system 段写死 schema 含 "JSON" 字样），故无 key 也能跑通闭环。
    """

    def __init__(
        self,
        chat_model: ChatModel,
        memory: MemoryStore,
        *,
        fallback: Reactor | None = None,
        token_cap: int = 1200,
        max_speakers: int = 2,
    ) -> None:
        self._chat_model = chat_model
        self._memory = memory
        self._fallback = fallback if fallback is not None else StubReactor()
        self._token_cap = token_cap
        self._max_speakers = max_speakers

    def react(self, world: World, trigger: Trigger, memory: MemoryStore) -> list[AgentReaction]:
        try:
            return self._react_via_llm(world, trigger, memory)
        except Exception as exc:  # 兜底护栏：任何意外都不许抛给引擎，委托 fallback
            logger.warning(
                "llm reactor unexpected failure err=%s; delegating fallback", type(exc).__name__
            )
            return self._delegate_fallback(world, trigger, memory)

    def _react_via_llm(self, world: World, trigger: Trigger, memory: MemoryStore) -> list[AgentReaction]:
        candidates = self._bounded_candidates(world, trigger)
        reactions: list[AgentReaction] = []
        hard_failures = 0
        for agent in candidates:
            r = self._react_one(agent, world, trigger)
            if r is None:
                hard_failures += 1  # chat/parse 失败：跳过该 agent（不牵连其他人）
            else:
                reactions.append(r)
        # 有候选但全部硬失败（LLM/解析全崩）→ 委托 fallback，别让世界彻底哑火。
        # 注意：全体「合法沉默」(speak=False) 不算硬失败，此处仍正常返回 []。
        if candidates and hard_failures == len(candidates):
            logger.warning("llm reactor all %d candidate(s) failed; delegating fallback", len(candidates))
            return self._delegate_fallback(world, trigger, memory)
        # F09 记忆闭环：把本轮产出的非空 memory_note 写回记忆库（在 bidding 之前，以 agent
        # 产出为准，与「是否被选中开口」正交——沉默但有心理活动者的 note 也值得记）。写入落在
        # 这里而非 engine.step：step 是纯函数、不得有写库副作用（可复现红线），reactor 才是
        # 有状态 wiring 层。同一次 step 内写入发生在所有 _react_one 之后，不影响本轮任何 prompt。
        self._persist_memory_notes(world, reactions, trigger)
        # bidding：开口者（speak 且台词非空）按候选顺序 capped；沉默但状态有变者保留供 patch
        speakers = [r for r in reactions if r.speak and r.line]
        silent = [
            r for r in reactions
            if not (r.speak and r.line) and (r.mood is not None or r.monologue is not None)
        ]
        max_speakers = 3 if trigger.kind == "catchup" else self._max_speakers
        return speakers[:max_speakers] + silent

    def _react_one(self, agent: Agent, world: World, trigger: Trigger) -> Optional[AgentReaction]:
        try:
            prompt = assemble_prompt(agent, world, trigger, self._memory, token_cap=self._token_cap)
            resp = self._chat_model.chat(prompt.messages)
            decision = parse_reaction(resp.text)
        except Exception as exc:  # 单 agent 失败：跳过，不牵连其他人（脱敏日志）
            logger.warning("llm reactor react failed agent=%s err=%s", agent.id, type(exc).__name__)
            return None
        if decision is None:  # 解析失败：视为该 agent 本轮无有效反应
            logger.warning("llm reactor parse empty agent=%s", agent.id)
            return None
        # F14 成功路径 trace：把「触发类型 / 角色 / token 估算 / 耗时 / 是否开口」串成一条
        # 可定位日志（失败/超时/解析空/fallback 各路径已有各自 warning，此处补齐成功侧）。
        # 脱敏——只记数值与 id，绝不落 prompt/line 原文（prompt 摘要脱敏已在 llm._summarize）。
        # 用 debug 而非 info：正常涌现每步多 agent，info 会刷屏；排障时开 debug 即可定位。
        logger.debug(
            "llm reactor ok agent=%s kind=%s tokens=%d elapsed_ms=%s speak=%s",
            agent.id, trigger.kind, prompt.budget_report.total_tokens,
            (resp.usage or {}).get("elapsed_ms"), decision.speak,
        )
        return AgentReaction(
            agent_id=agent.id,
            speak=decision.speak,
            line=decision.line,
            reason=decision.reason,
            mood=decision.mood,
            memory_note=decision.memory_note,
        )

    def _persist_memory_notes(
        self, world: World, reactions: list[AgentReaction], trigger: Trigger
    ) -> None:
        """把本轮非空 memory_note 追加进记忆库（F09 记忆闭环，写经历层**与**关系层）。

        两层都写（对齐 spec F09「写入经历与关系摘要」）：
          · 经历层 `agent:{id}`  —— 一律写，是该 agent 自己的通用记忆。
          · 关系层 `rel:{id}:{other}` —— 对本轮**同处一室**的每个对端各写一份，
            让「A 记住的这件事」在日后 A 与 B 再同室时能被 `_retrieve_relations`
            按 `rel:{A}:{B}` 检索到（读路径见 prompting.py，此处是它的对称写路径）。
        同室对端取自 `_copresent_peers`，与读端的「同房间」口径一致（move 场景把移动者
        的有效房间视为目标房间），保证写进去的 key 日后读得回来（换后端零漂移的写侧对齐）。

        可复现红线：只在 `trigger.ts` 非 None 时写——ts 必须来自触发本身（事件溯源的
        确定性时刻），**绝不取 wall clock**（否则同一 (world, trigger) 重放会写入不同 ts，
        破坏可复现）。无 ts 的触发（少见）本轮不落库，宁可漏记不可污染重放。

        降级红线（对齐决策 8.3）：写入失败不抛给引擎——单条 add 异常只记脱敏 warning
        （namespace + 异常类型名，绝不落 note 原文/密钥）并跳过，不牵连其他写入与反应。
        """
        if trigger.ts is None:
            return
        for r in reactions:
            note = (r.memory_note or "").strip()
            if not note:
                continue
            # 经历层：该 agent 自己的记忆，一律写。
            self._safe_add(f"agent:{r.agent_id}", note, trigger.ts)
            # 关系层：对本轮同处一室的每个对端各写一份（key 与读端对称）。
            for other_id in self._copresent_peers(world, trigger, r.agent_id):
                self._safe_add(f"rel:{r.agent_id}:{other_id}", note, trigger.ts)

    def _safe_add(self, namespace: str, text: str, ts: int) -> None:
        """ADD-only 单条写入 + 降级护栏：写库失败只记脱敏 warning，绝不抛给引擎。"""
        try:
            self._memory.add(namespace=namespace, text=text, ts=ts)
        except Exception as exc:  # 写库失败：脱敏日志（ns + 异常类型名）+ 跳过，不牵连其他写入
            logger.warning(
                "llm reactor memory write failed ns=%s err=%s", namespace, type(exc).__name__
            )

    @staticmethod
    def _copresent_peers(world: World, trigger: Trigger, agent_id: str) -> list[str]:
        """本轮与 `agent_id` 同处一室的对端 id（关系层写入的目标集，与读端同室口径一致）。

        · oracle —— agent_id 所在房间里的其他人。
        · move   —— 目标房间的在场者 + 移动者本人；移动者的「有效房间」按目标房间算
          （move patch 尚未落地，但共处已发生，与 prompting._move_context 的时序处理对齐）。
        找不到该 agent（理论不会）时返回空，写入自然退化为只写经历层。
        """
        self_agent = next((a for a in world.agents if a.id == agent_id), None)
        if self_agent is None:
            return []

        def effective_room(a: Agent) -> Optional[str]:
            # move 的移动者视为已在目标房间；其余取当前 roomId。
            if trigger.kind == "move" and trigger.agentId == a.id:
                return trigger.roomId
            return a.roomId

        room_id = effective_room(self_agent)
        peers = {
            a.id for a in world.agents
            if a.id != agent_id and effective_room(a) == room_id
        }
        return sorted(peers)  # 排序保证可复现（写入顺序确定）

    def _candidates(self, world: World, trigger: Trigger) -> list[Agent]:
        """依 trigger.kind 定候选人。oracle=全体；move=移动者 + 目标房间在场者（移动者居首，
        便于 engine 用 agent_id 识别出「哪条是移动者的共处叙事」）；catchup=该片房间的在场者。"""
        if trigger.kind == "oracle":
            return list(world.agents)
        if trigger.kind == "move":
            mover = next((a for a in world.agents if a.id == trigger.agentId), None)
            if mover is None:
                return []
            present = [a for a in world.agents if a.roomId == trigger.roomId and a.id != mover.id]
            return [mover] + present
        if trigger.kind == "catchup":
            # 补演单片是「某房间某一刻的生活」：候选=该房间在场者。bidding（_react_via_llm）
            # 会按 max_speakers 截断，故一片不会齐声喧哗，与 StubReactor 挑 1 人的密度大体相当。
            return [a for a in world.agents if a.roomId == trigger.roomId]
        return []

    def _bounded_candidates(self, world: World, trigger: Trigger) -> list[Agent]:
        candidates = self._prioritized_candidates(world, trigger, self._candidates(world, trigger))
        if self._chat_model.provider in _LOCAL_PROVIDERS:
            return candidates
        limit = self._candidate_limit(trigger)
        if len(candidates) > limit:
            logger.info(
                "llm reactor candidate fanout reduced kind=%s provider=%s from=%d to=%d",
                trigger.kind,
                self._chat_model.provider,
                len(candidates),
                limit,
            )
        return candidates[:limit]

    def _candidate_limit(self, trigger: Trigger) -> int:
        default = 3 if trigger.kind == "catchup" else max(1, self._max_speakers)
        raw = os.getenv("CYBERHOME_LLM_REACTOR_CANDIDATE_LIMIT")
        if not raw:
            return default
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _prioritized_candidates(
        world: World, trigger: Trigger, candidates: list[Agent]
    ) -> list[Agent]:
        if not candidates:
            return []
        text = trigger.text or ""
        named = [agent for agent in candidates if agent.name and agent.name in text]
        named_ids = {agent.id for agent in named}
        rest = [agent for agent in candidates if agent.id not in named_ids]
        if trigger.kind == "move" and trigger.agentId:
            mover = [agent for agent in rest if agent.id == trigger.agentId]
            rest = [agent for agent in rest if agent.id != trigger.agentId]
            return named + mover + rest
        return named + rest

    def _delegate_fallback(
        self, world: World, trigger: Trigger, memory: MemoryStore
    ) -> list[AgentReaction]:
        try:
            return self._fallback.react(world, trigger, memory)
        except Exception as exc:  # fallback 也崩：最终兜底空反应（引擎对 move 另有中性叙事）
            logger.warning("fallback reactor also failed err=%s", type(exc).__name__)
            return []
