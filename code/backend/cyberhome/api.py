"""FastAPI 服务壳 —— 把 WorldDriver 包成 HTTP 服务（F01）。

这层刻意薄：真正的逻辑都在纯函数内核与 driver 里，端点只做「解析请求 →
构造 Trigger → dispatch → 序列化响应」。契约对齐前端 `frontend/src/mock/service.ts`
（loadWorld / castOracle / moveAgent），前端替换 mock 时改实现即可（F02）。

`create_app(driver=None)` 是工厂：默认新建独立 driver，允许测试注入独立实例
以隔离状态。模块级 `app = create_app()` 供 `uvicorn cyberhome.api:app` 启动。

持久化开关（F03）：默认 driver 是否落 SQLite 由环境变量 `CYBERHOME_PERSIST` 门控。
WHY 用 env 门控而非无条件持久化——注入 driver 的测试（test_api）用 `create_app()`
且**依赖每个用例状态隔离**；若默认无条件写固定 DB，测试会因跨用例污染 + 本地残留
而失败，且红线要求不改 test_api。env 门控让「生产启动开持久化、测试零配置走内存
隔离」两全，既忠于 F03「世界跨重启保留」的意图，又不破坏 F01 已定的端点合同与隔离。
生产启动：`CYBERHOME_PERSIST=1 uvicorn cyberhome.api:app`。
"""

from __future__ import annotations

import os
import time
import logging
import threading
import uuid
import json
from dataclasses import dataclass, replace
from typing import Iterator, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .driver import WorldDriver
from .llm import ChatModel, build_chat_model
from .memory import InMemoryStore, MemoryItem, build_memory_store
from .models import Agent, Trigger, World
from .opening import synthesize_opening
from .persistence import SQLitePersistenceStore
from .persona_revision import synthesize_persona_revision_draft
from .personas import coerce_mood, synthesize_persona_draft
from .reactor import LLMReactor, StubReactor
from .schemas import (
    AgentChatRequest,
    CreateAgentRequest,
    FloorPlanUpdateRequest,
    MoveAgentRequest,
    OpeningRequest,
    OracleRequest,
    PersonaDraftRequest,
    PersonaRevisionConfirmRequest,
    PersonaRevisionDraftRequest,
    WhatIfRequest,
    dispatch_result,
    events_to_dict,
    persona_draft_to_dict,
    persona_revision_to_dict,
    world_to_dict,
)
from .seed import make_initial_world, new_event_id
from .time_enrichment import LifeBeat, synthesize_time_enrichment

logger = logging.getLogger("cyberhome.api")


DEFAULT_AGENT_CHAT_MAX_OUTPUT_TOKENS = 900


@dataclass
class _TimeEnrichmentJob:
    id: str
    status: str = "pending"
    events: tuple = ()
    anchor_event_id: str | None = None
    error: str | None = None


def _now_ms() -> int:
    """当前毫秒时间戳，对齐前端 Date.now()。"""
    return int(time.time() * 1000)


def _default_driver(chat_model: ChatModel) -> WorldDriver:
    """构造生产默认 driver。CYBERHOME_PERSIST 为真则接 SQLite（bootstrap 恢复/创世），
    否则退回纯内存（重启回到 seed）——与 F03 前行为一致，测试默认走这条。

    生产默认 reactor 切为 LLMReactor（F06）：无 key 时 build_chat_model 返回
    StubChatModel，其对含 JSON 要求的 prompt 返回结构合法 JSON，故无 key 也能跑通
    闭环（确定性）；fallback=StubReactor 兜底。memory 用共享 InMemoryStore 占位
    （真正落库是 F09/F13），既让 LLMReactor 检索与 step 透传用同一实例，也保持接口就位。

    chat_model 由 create_app 传入并与 /personas/draft 端点共享同一实例（F07b）——
    合成 persona 与角色反应用同一个模型（同一 key 判定、同一 provider）。
    """
    persist = (os.environ.get("CYBERHOME_PERSIST") or "").strip().lower() in ("1", "true", "yes")
    memory = build_memory_store() if persist else InMemoryStore()
    reactor = LLMReactor(chat_model, memory, fallback=StubReactor())
    if persist:
        return WorldDriver.bootstrap(
            SQLitePersistenceStore(), reactor=reactor, memory=memory, now=_now_ms()
        )
    return WorldDriver(make_initial_world(now=_now_ms()), reactor=reactor, memory=memory)


def create_app(driver: WorldDriver | None = None, *, chat_model: ChatModel | None = None) -> FastAPI:
    """构造 FastAPI app。

    driver 为空则新建默认 driver（测试可注入以隔离状态）。
    chat_model 为空则由 build_chat_model 按 env 选择（无 key → StubChatModel）；
    测试可注入 StubChatModel 走确定性合成路径（F07b）。它同时喂给默认 driver 的
    LLMReactor 与 /personas/draft，保证「角色反应」与「人设合成」用同一个模型。
    """
    chat_model = chat_model or build_chat_model()
    driver = driver or _default_driver(chat_model)
    driver_lock = threading.RLock()
    time_jobs: dict[str, _TimeEnrichmentJob] = {}
    time_jobs_lock = threading.RLock()

    app = FastAPI(title="CyberHome", version="0.1.0")
    # 本地 Vite 联调：放开跨域。生产收敛白名单是部署化（非 MVP）的事。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _agent_or_404(agent_id: str) -> Agent:
        agent = next((a for a in driver.world.agents if a.id == agent_id), None)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return agent

    def _ensure_room_exists(room_id: str | None) -> None:
        if room_id is None:
            return
        if not any(r.id == room_id for r in driver.world.rooms):
            raise HTTPException(status_code=400, detail=f"Room not found: {room_id}")

    def _create_time_job() -> str:
        job_id = uuid.uuid4().hex
        with time_jobs_lock:
            time_jobs[job_id] = _TimeEnrichmentJob(id=job_id)
        return job_id

    def _set_time_job(job_id: str, **fields) -> None:
        with time_jobs_lock:
            job = time_jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def _get_time_job(job_id: str) -> _TimeEnrichmentJob:
        with time_jobs_lock:
            job = time_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Time enrichment job not found: {job_id}")
        return job

    def _run_time_enrichment_job(
        job_id: str,
        world_snapshot: World,
        base_events: tuple,
        seed: int,
    ) -> None:
        _set_time_job(job_id, status="running")
        try:
            anchor_event_id = _time_enrichment_anchor_event_id(base_events)
            beats = synthesize_time_enrichment(chat_model, world_snapshot, base_events, seed=seed)
            payload: tuple[tuple[str, str | None], ...] = tuple(
                (beat.text, beat.agentId) for beat in beats if beat.text.strip()
            )
            if payload:
                ts = max(_now_ms(), seed + 1_000)
                with driver_lock:
                    events = driver.append_life_beats(
                        payload,
                        trigger=Trigger(kind="catchup", since=seed, ts=ts),
                        now=ts,
                        anchor_event_id=anchor_event_id,
                    )
            else:
                events = ()
            logger.info("time enrichment job done id=%s beats=%d events=%d", job_id, len(beats), len(events))
            _set_time_job(job_id, status="done", events=events, anchor_event_id=anchor_event_id)
        except Exception as exc:
            logger.warning("time enrichment job failed id=%s err=%s", job_id, type(exc).__name__)
            _set_time_job(job_id, status="error", error=type(exc).__name__, events=())

    def _time_enrichment_world(world_before: World, world_after: World) -> World:
        """Generate enrichment for the phase the user just advanced into."""
        return world_after

    def _time_enrichment_anchor_event_id(base_events: tuple) -> str | None:
        """Place async LLM enrichment right after the most relevant fast event."""
        for event in reversed(base_events):
            if getattr(event, "source", None) == "move":
                return getattr(event, "id", None)
        if base_events:
            return getattr(base_events[-1], "id", None)
        return None

    def _world_with_events_after_anchor(
        world: World,
        events: tuple,
        anchor_event_id: str | None,
    ) -> World:
        if not anchor_event_id or not events:
            return world
        inserted_ids = {event.id for event in events}
        base_events = tuple(event for event in world.events if event.id not in inserted_ids)
        anchor_index = next(
            (i for i, event in enumerate(base_events) if event.id == anchor_event_id),
            -1,
        )
        if anchor_index < 0:
            return replace(world, events=(*base_events, *events))
        return replace(
            world,
            events=(
                *base_events[: anchor_index + 1],
                *events,
                *base_events[anchor_index + 1 :],
            ),
        )

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/world")
    def get_world() -> dict:
        """返回当前世界快照（前端兼容 camelCase dict）。"""
        return world_to_dict(driver.world)

    # === ANCHOR:F10_CATCHUP === (Task 10 惰性补演端点在此区插入，勿动此锚)

    @app.post("/world/catchup")
    def catchup() -> dict:
        """惰性补演（F10）：把「上次离开到现在」这段时间凝练成若干条 catchup 事件。

        前端在 GET /world 之后、进门时调用一次（非常驻——后端未运行期间无任何后台任务）。
        now 用 _now_ms() 注入 driver.catchup；driver 据 last_seen 算离开时长：不足阈值/未挂
        持久化/无 last_seen 则返回空 events（前端据此静默略过，不打断沉浸）。返回形状与
        oracle/move 一致（{events, world}），前端可直接把新事件并进事件流、用更新后 world 覆盖。
        """
        events = driver.catchup(now=_now_ms())
        return dispatch_result(events, driver.world)

    # === ANCHOR:F10_CATCHUP_END ===

    # === ANCHOR:F11_TIME === (Task 13 世界时间推进端点在此区插入，勿动此锚)

    @app.post("/world/advance-time")
    def advance_time(background_tasks: BackgroundTasks) -> dict:
        """世界时间推进（F11 demo 控件）：phase 前进一档，产一条 time 事件。
        走 driver.dispatch(Trigger(kind="time"))——与 oracle/move 同一条 commit 路径
        （落事件溯源日志 + 更新快照）。前端点信息条时调用，光影随后端 phase 漂移。"""
        ts = _now_ms()
        with driver_lock:
            world_before = driver.world
            events = driver.dispatch(Trigger(kind="time", ts=ts))
            world_after_fast_step = driver.world
            world_for_enrichment = _time_enrichment_world(world_before, world_after_fast_step)
        job_id = _create_time_job()
        background_tasks.add_task(
            _run_time_enrichment_job,
            job_id,
            world_for_enrichment,
            events,
            ts,
        )
        result = dispatch_result(events, world_after_fast_step)
        result["enrichmentJobId"] = job_id
        result["enrichmentStatus"] = "pending"
        return result

    @app.get("/world/time-enrichments/{job_id}")
    def get_time_enrichment(job_id: str) -> dict:
        """Return async LLM enrichment generated after a manual time advance."""
        job = _get_time_job(job_id)
        with driver_lock:
            world = _world_with_events_after_anchor(
                driver.world,
                job.events,
                job.anchor_event_id,
            )
        return {
            "id": job.id,
            "status": job.status,
            "events": events_to_dict(job.events),
            "world": world_to_dict(world),
            "anchorEventId": job.anchor_event_id,
            "error": job.error,
        }

    # === ANCHOR:F11_TIME_END ===

    # === ANCHOR:F08_OPENING === (Task 12 开场故事生成端点在此区插入，勿动此锚)

    @app.post("/world/opening")
    def generate_opening(body: OpeningRequest) -> dict:
        """开场故事生成（F08）：读当前 world 真实角色 + 布局动态生成 3-5 条 opening，
        追加进事件溯源。换角色/布局后开场明显变化，且无「欢迎使用」式系统话术。

        端点层调 chat_model（与 /personas/draft 对称，共享同一实例）：无 key 时是
        StubChatModel，其对含 JSON 要求的 prompt 返回不含 beats 的占位 JSON，合成层据此
        回退确定性 stub_opening——绝不 500。落溯源走 driver.open_world（source=opening 事件
        + 空 patch，与 add_agent 同一条 _commit 写通链路，不走 step）。返回形状与 oracle/move
        一致（{events, world}），前端可直接把新事件并进事件流、用更新后 world 覆盖。
        """
        beats = synthesize_opening(chat_model, driver.world, seed=body.seed)
        events = driver.open_world(beats, now=_now_ms())
        return dispatch_result(events, driver.world)

    # === ANCHOR:F08_OPENING_END ===

    @app.put("/world/floor-plan")
    def update_floor_plan(body: FloorPlanUpdateRequest) -> dict:
        """保存自定义户型图。

        自定义户型是世界级结构状态：floorPlan 写入 world 快照；新增空间先追加到
        World.rooms，后续角色 move 才能以 roomId 合法落位。几何仍不参与 step()。
        """
        existing = {room.id for room in driver.world.rooms}
        added_rooms = tuple(
            room.to_room()
            for room in body.addedRooms
            if room.id.strip() and room.id.strip() not in existing
        )
        valid_room_ids = existing | {room.id for room in added_rooms}
        for plan_room in body.floorPlan.get("rooms", []):
            room_id = plan_room.get("roomId") if isinstance(plan_room, dict) else None
            if room_id not in valid_room_ids:
                raise HTTPException(status_code=400, detail=f"FloorPlan roomId not found: {room_id}")
        events = driver.update_floor_plan(body.floorPlan, added_rooms=added_rooms, now=_now_ms())
        return dispatch_result(events, driver.world)

    @app.post("/oracle")
    def cast_oracle(body: OracleRequest) -> dict:
        """神谕：投放一句契机，在场角色各自反应（影响非命令，决策 7）。"""
        events = driver.dispatch(Trigger(kind="oracle", text=body.text))
        return dispatch_result(events, driver.world)

    @app.post("/agents/{agent_id}/move")
    def move_agent(agent_id: str, body: MoveAgentRequest) -> dict:
        """移动：把角色移入目标房间，触发共处涌现。"""
        events = driver.dispatch(
            Trigger(kind="move", agentId=agent_id, roomId=body.roomId)
        )
        return dispatch_result(events, driver.world)

    @app.delete("/agents/{agent_id}")
    def delete_agent(agent_id: str) -> dict:
        """删除一个人物/家电模块。"""
        _agent_or_404(agent_id)
        events = driver.delete_agent(agent_id, now=_now_ms())
        return dispatch_result(events, driver.world)

    @app.post("/agents/{agent_id}/chat")
    def chat_with_agent(agent_id: str, body: AgentChatRequest) -> dict:
        """与单个 Agent 私聊。

        这是状态弹窗内的私密会话，不写 world、不进事件流。每轮会把简短摘要追加到
        agent:{id} 私有记忆，后续私聊会检索注入，但不会污染公共事件流。
        """
        agent = _agent_or_404(agent_id)
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Message text is required")

        location_fact = _agent_location_fact(driver.world, agent)
        private_memory = _private_chat_memories(driver.memory, agent, text)
        messages = _assemble_agent_chat_messages(agent, body, private_memory, location_fact)
        try:
            response = chat_model.chat(
                messages,
                max_output_tokens=_agent_chat_max_output_tokens(),
                temperature=0.8,
            )
            reply = response.text.strip()
            if (
                not reply
                or response.provider == "stub"
                or _looks_like_incomplete_agent_reply(reply, agent)
            ):
                reply = _fallback_agent_chat_reply(agent, text, location_fact)
        except Exception as exc:
            logger.warning(
                "agent private chat llm failed agent=%s err=%s; using fallback reply",
                agent_id, type(exc).__name__,
            )
            reply = _fallback_agent_chat_reply(agent, text, location_fact)
        _remember_private_chat(
            driver.memory,
            agent,
            user_text=text,
            reply=reply,
            ts=_now_ms(),
            location_fact=location_fact,
        )
        return {"reply": reply}

    @app.post("/agents/{agent_id}/chat/stream")
    def stream_chat_with_agent(agent_id: str, body: AgentChatRequest) -> StreamingResponse:
        """Stream private Agent chat as newline-delimited JSON chunks.

        The final `done` packet carries the full reply after fallback filtering;
        private memory is written only once that final text is known.
        """
        agent = _agent_or_404(agent_id)
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Message text is required")

        location_fact = _agent_location_fact(driver.world, agent)
        private_memory = _private_chat_memories(driver.memory, agent, text)
        messages = _assemble_agent_chat_messages(agent, body, private_memory, location_fact)

        def generate() -> Iterator[str]:
            reply = ""
            emitted_any = False
            try:
                if getattr(chat_model, "provider", "") == "stub":
                    reply = _fallback_agent_chat_reply(agent, text, location_fact)
                    for chunk in _chunk_stream_text(reply):
                        emitted_any = True
                        yield _ndjson({"type": "delta", "text": chunk})
                else:
                    for chunk in _stream_model_chunks(
                        chat_model,
                        messages,
                        max_output_tokens=_agent_chat_max_output_tokens(),
                        temperature=0.8,
                    ):
                        if not chunk:
                            continue
                        emitted_any = True
                        reply += chunk
                        yield _ndjson({"type": "delta", "text": chunk})
                    reply = reply.strip()
                    if not reply or _looks_like_incomplete_agent_reply(reply, agent):
                        reply = _fallback_agent_chat_reply(agent, text, location_fact)
                        yield _ndjson({"type": "replace", "text": reply})
            except Exception as exc:
                logger.warning(
                    "agent private chat stream failed agent=%s err=%s; using fallback reply",
                    agent_id, type(exc).__name__,
                )
                reply = _fallback_agent_chat_reply(agent, text, location_fact)
                packet_type = "replace" if emitted_any else "delta"
                for chunk in _chunk_stream_text(reply):
                    yield _ndjson({"type": packet_type, "text": chunk})
                    packet_type = "delta"

            _remember_private_chat(
                driver.memory,
                agent,
                user_text=text,
                reply=reply,
                ts=_now_ms(),
                location_fact=location_fact,
            )
            yield _ndjson({"type": "done", "reply": reply})

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson; charset=utf-8",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/personas/draft")
    def draft_persona(body: PersonaDraftRequest) -> dict:
        """引导问答 → LLM 合成结构化人设草稿（F07b · 11.1）。

        **纯合成、无副作用**：不写 world、不落库、不碰记忆。无 key 时 chat_model 是
        StubChatModel，合成层据此回退确定性 stub 草稿；解析失败也有 stub 兜底，绝不 500。
        """
        draft = synthesize_persona_draft(chat_model, body.to_answers(), seed=body.seed)
        return persona_draft_to_dict(draft)

    @app.post("/agents/{agent_id}/persona-revisions/draft")
    def draft_persona_revision(agent_id: str, body: PersonaRevisionDraftRequest) -> dict:
        """已有 Agent 的人设修订草稿（P0-1）。

        纯合成、无副作用：读取当前 Agent 作为旧设定，返回新旧对照预览所需草稿。
        无 key / LLM 失败时合成层走确定性兜底，绝不因为模型不可用阻断编辑流。
        """
        agent = _agent_or_404(agent_id)
        draft = synthesize_persona_revision_draft(
            chat_model,
            agent,
            body.answers.to_answers(),
            mode=body.mode,
            seed=body.seed,
        )
        return persona_revision_to_dict(draft)

    @app.post("/agents/{agent_id}/persona-revisions/confirm")
    def confirm_persona_revision(agent_id: str, body: PersonaRevisionConfirmRequest) -> dict:
        """确认 Agent 人设修订（P0-1）。

        更新当前 persona 锚，但不回写旧历史。若用户选择公开，返回一条 source="system"
        的生活化旁白事件；若选择静默，events 可为空，但 driver 仍会追加溯源日志。
        若 moveWithRevision=true 且 roomId 变化，driver 会在修订后再 dispatch move，
        复用共处涌现，不静默改 roomId。
        """
        _agent_or_404(agent_id)
        draft = body.draft.to_draft()
        if draft.agentId != agent_id:
            raise HTTPException(
                status_code=400,
                detail=f"Draft agentId mismatch: {draft.agentId} != {agent_id}",
            )
        if body.moveWithRevision:
            _ensure_room_exists(body.roomId)

        events = driver.revise_agent_persona(
            agent_id,
            name=draft.name,
            avatar=draft.avatar,
            persona=draft.persona,
            monologue=draft.monologue,
            mood=draft.mood,
            catchphrase=draft.catchphrase,
            visibility=body.normalized_visibility(),
            public_event_text=draft.publicEventText,
            summary=draft.summary,
            room_id=body.roomId,
            move_with_revision=body.moveWithRevision,
            now=_now_ms(),
        )
        return dispatch_result(events, driver.world)

    @app.post("/agents")
    def create_agent(body: CreateAgentRequest) -> dict:
        """确认后的角色数据 → 写入 world + 追加「第一次来到这个家」事件（F07b · 11.2）。

        world 变更走 driver.add_agent（WorldPatch + apply_patch + 溯源写通），不绕过溯源。
        mood 经 coerce_mood 归一到合法 Mood（非法归 calm，红线 6）。返回更新后的 world + 新事件。
        """
        kind = body.kind if body.kind in ("human", "appliance") else "human"
        agent = Agent(
            id="",  # 由 driver.add_agent 用注入的 agent_id 生成器补齐
            kind=kind,
            name=body.name.strip() or ("新伙计" if kind == "appliance" else "新朋友"),
            avatar=body.avatar.strip() or ("🔌" if kind == "appliance" else "🙂"),
            persona=body.persona.strip(),
            monologue=body.monologue.strip(),
            mood=coerce_mood(body.mood),
            roomId=body.roomId,
            catchphrase=(body.catchphrase.strip() or None),
        )
        events = driver.add_agent(agent)
        return dispatch_result(events, driver.world)

    # === ANCHOR:F12_HISTORY === (Task 15 历史只读端点在此区插入，勿动此锚)
    # 三个**只读**端点：因 step() 是纯函数、溯源日志不可变（决策 32），回放/回溯/
    # what-if 是「免费午餐」——都只是对同一条日志的不同读法，绝不改主线世界/日志。
    # 全部委托 driver.store 的溯源原语（EventStore.replay_events/world_at/fork_readonly）。

    @app.get("/world/events")
    def get_events(until_seq: Optional[int] = None) -> dict:
        """回放：返回事件时间线（含创世 opening + 各步 events，按时间正序）。

        比 GET /world 的 world.events 更完整——后者前端会按 MAX_EVENTS 截断，这里给全量。
        until_seq 可选：只回放到第 until_seq 步（含），供「翻看到某一刻为止」。
        """
        events = driver.store.replay_events(until_seq=until_seq)
        return {"events": events_to_dict(events)}

    @app.get("/world/at")
    def get_world_at(seq: Optional[int] = None) -> dict:
        """回溯：返回第 seq 步之后的世界快照（把 patch 折叠到那一刻）。

        seq=None → 最新；seq=-1 → 创世。seq 越界不报错（world_at 语义是「折叠到不超过
        seq 的所有步」）；seq 非数字由 FastAPI query 类型校验挡在门外（422）。因所有模型
        frozen，折叠必产新对象，绝不污染历史——只读安全的根基。
        """
        return world_to_dict(driver.store.world_at(seq))

    @app.post("/world/what-if")
    def what_if(body: WhatIfRequest) -> dict:
        """what-if：对历史某时刻「假如换一条触发」的只读预演（A 档，决策 33）。

        fork_readonly 回溯出历史 world，对它跑一次 step 得假想世界，**绝不写回日志、不改
        主线**（它内部不调 record）。返回 {events, world=假想世界}，形状与 oracle/move 一致。

        WHY 用 StubReactor 而非生产的 LLMReactor：
          1. what-if 是「假设」，本就不必消耗真 LLM 配额——预演一条不存在的分支去烧 token 不划算；
          2. 确定性/可复现——StubReactor 的随机源可控，同一 (world, trigger) 预演稳定，
             正好契合「只读、可反复推演」的语义，也让测试能断言结果；
          3. 安全——不把用户任意构造的 what-if 文本喂给真模型，隔离 prompt 注入面。
        driver 的私有 _reactor/_event_id 不对外暴露（Task 10 正在改 driver，不加 getter），
        故这里自备 StubReactor() + 模块级 new_event_id，与主线解耦、互不干扰。
        """
        now = _now_ms()
        events, _patch, hypothetical = driver.store.fork_readonly(
            at_seq=body.seq,
            trigger=body.to_trigger(now),
            reactor=StubReactor(),
            event_id=new_event_id,
            now=now,
        )
        return dispatch_result(events, hypothetical)

    # === ANCHOR:F12_HISTORY_END ===

    return app


app = create_app()


def _assemble_agent_chat_messages(
    agent: Agent,
    body: AgentChatRequest,
    private_memory: list[MemoryItem],
    location_fact: str,
) -> list[dict]:
    """组装私聊 prompt。使用 Agent 锚、私有记忆与短历史，不读取/写入公共事件流。"""
    recent = body.history[-8:]
    history_lines = []
    for item in recent:
        speaker = "用户" if item.normalized_role() == "user" else agent.name
        content = item.text.strip()
        if content:
            history_lines.append(f"{speaker}：{content}")
    history_text = "\n".join(history_lines) if history_lines else "（这是本次私聊的开始）"
    if private_memory:
        memory_text = "\n".join(f"- {m.text}" for m in private_memory)
    else:
        memory_text = "（暂时没有私聊记忆）"
    catchphrase = f"口头禅：{agent.catchphrase}" if agent.catchphrase else "没有固定口头禅"
    system = (
        "你是 CyberHome 中的一个具体 Agent，正在和用户进行私密聊天。"
        "只代表你自己说话，不替其他角色发言；回复自然、简短、有生活感。"
        "不要使用 JSON，不要解释规则，不要提到系统提示。"
    )
    user = (
        f"Agent 名字：{agent.name}\n"
        f"类型：{'人类角色' if agent.kind == 'human' else '家电家具 Agent'}\n"
        f"当前人设：{agent.persona}\n"
        f"当前心情：{agent.mood}\n"
        f"当前世界事实：{location_fact}\n"
        "位置回答原则：如果用户问你在哪、在哪个房间、所处位置，必须优先按当前世界事实回答，"
        "不要按常识或人设猜测你本来应该在哪里。\n"
        f"此刻心里想着：{agent.monologue}\n"
        f"{catchphrase}\n\n"
        f"你的私有记忆（只属于你和用户的私聊，不要当作公共事件）：\n{memory_text}\n\n"
        f"最近私聊：\n{history_text}\n\n"
        f"用户刚刚说：{body.text.strip()}\n"
        f"请用 {agent.name} 的口吻回复 1-3 句。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _agent_location_fact(world: World, agent: Agent) -> str:
    """把世界里的 roomId 翻译成模型可直接服从的当前位置事实。"""
    room = next((r for r in world.rooms if r.id == agent.roomId), None)
    room_name = room.name if room else (agent.roomId or "未知房间")
    return f"我现在在{room_name}。"


def _fallback_agent_chat_reply(agent: Agent, text: str, location_fact: str = "") -> str:
    """无模型/模型失败时的确定性私聊兜底。"""
    lead = agent.catchphrase or agent.monologue or agent.persona
    cleaned = text.strip(" \t\r\n。！？!?，,")
    if _asks_location(text):
        return f"{lead}。{location_fact or '我现在的位置有点不明确。'}"
    if _asks_activity(text):
        detail = agent.monologue.rstrip("。！？!?，,") or agent.persona
        return f"{lead}。{location_fact or ''}{detail}。"
    if agent.mood == "happy":
        return f"{lead}。你这么一说，我心里倒是亮了一下：{cleaned}。"
    if agent.mood == "grumpy":
        return f"{lead}。哼，我听见了，{cleaned}这事也不是不能商量。"
    if agent.mood == "tired":
        return f"{lead}。我有点累，但你说的{cleaned}，我会慢慢放在心上。"
    if agent.mood == "lonely":
        return f"{lead}。你愿意这样跟我说，我就没那么落单了。"
    return f"{lead}。我听见了，{cleaned}。"


def _asks_location(text: str) -> bool:
    compact = "".join((text or "").split())
    if not compact:
        return False
    return any(marker in compact for marker in ("在哪", "哪里", "哪儿", "位置", "哪个房间", "什么房间"))


def _asks_activity(text: str) -> bool:
    compact = "".join((text or "").split())
    if not compact:
        return False
    return any(marker in compact for marker in ("在干嘛", "在做什么", "干什么", "做啥", "忙什么"))


def _looks_like_incomplete_agent_reply(reply: str, agent: Agent) -> bool:
    cleaned = (reply or "").strip()
    if not cleaned:
        return True
    if len(cleaned) <= 6 and cleaned[-1:] in "，,、：:；;":
        return True
    catchphrase = (agent.catchphrase or "").strip()
    if catchphrase:
        bare = cleaned.strip(" \t\r\n。！？!?，,、：:；;")
        if bare and (bare == catchphrase or catchphrase.startswith(bare)):
            return True
    return False


def _stream_model_chunks(
    chat_model,
    messages: list[dict],
    *,
    max_output_tokens: int | None,
    temperature: float | None,
) -> Iterator[str]:
    stream_chat = getattr(chat_model, "stream_chat", None)
    if callable(stream_chat):
        yield from stream_chat(
            messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        return
    response = chat_model.chat(
        messages,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    text = (response.text or "").strip()
    if text:
        yield text


def _ndjson(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _chunk_stream_text(text: str, size: int = 8) -> Iterator[str]:
    cleaned = text or ""
    for i in range(0, len(cleaned), size):
        yield cleaned[i:i + size]


def _agent_chat_max_output_tokens() -> int:
    raw = (os.environ.get("CYBERHOME_AGENT_CHAT_MAX_TOKENS") or "").strip()
    if not raw:
        return DEFAULT_AGENT_CHAT_MAX_OUTPUT_TOKENS
    try:
        return max(220, int(raw))
    except ValueError:
        return DEFAULT_AGENT_CHAT_MAX_OUTPUT_TOKENS


def _private_chat_memories(memory, agent: Agent, query: str) -> list[MemoryItem]:
    """读取单个 Agent 的私聊私有记忆。失败静默降级为空，不阻断聊天。"""
    if memory is None:
        return []
    try:
        return memory.search(f"agent:{agent.id}", query, top_k=5, token_cap=600)
    except Exception:
        return []


def _remember_private_chat(
    memory,
    agent: Agent,
    *,
    user_text: str,
    reply: str,
    ts: int,
    location_fact: str = "",
) -> None:
    """把一轮私聊压成一条 ADD-only 私有记忆，写入 agent:{id} 经历层。"""
    if memory is None:
        return
    summary = _chat_memory_summary(user_text, reply, location_fact=location_fact)
    if not summary:
        return
    try:
        memory.add(namespace=f"agent:{agent.id}", text=summary, ts=ts, importance=0.65)
    except Exception:
        # 记忆是增强能力，写入失败不应让私聊失败。
        return


def _chat_memory_summary(user_text: str, reply: str, *, location_fact: str = "") -> str:
    user = _clip_one_line(user_text, 90)
    agent_reply = _clip_one_line(reply, 120)
    location = _clip_one_line(location_fact, 40)
    if not user:
        return ""
    prefix = f"私聊记忆：当时{location}；" if location else "私聊记忆："
    return f"{prefix}用户曾对我说「{user}」；我当时回应「{agent_reply}」。"


def _clip_one_line(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "…"
