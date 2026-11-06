"""FastAPI 服务层测试（F01 验收）。

标准库 unittest + FastAPI TestClient，与现有 test_engine.py 同一套运行方式：
    cd backend && python3 -m unittest discover tests

每个 test 用 create_app() 造独立 app（独立 driver），避免测试间内存状态污染。
"""

import unittest
import time
import json
from dataclasses import replace

from fastapi.testclient import TestClient

from cyberhome.api import create_app
from cyberhome.driver import WorldDriver
from cyberhome.llm import LLMResponse, StubChatModel
from cyberhome.memory import InMemoryStore
from cyberhome.models import WorldEvent
from cyberhome.schemas import world_from_dict, world_to_dict
from cyberhome.seed import make_initial_world
from cyberhome.worldclock import phase_label

# persona 合成层的合法 mood 集合（与 models.Mood 单一事实源一致）。
_VALID_MOODS = {"happy", "calm", "tired", "grumpy", "lonely"}


class CapturingChatModel:
    provider = "test"

    def __init__(self):
        self.calls = []

    def chat(self, messages, *, max_output_tokens=None, temperature=None):
        self.calls.append(messages)
        return LLMResponse(text="我记下了。", provider=self.provider, model="test")


class ShortReplyChatModel(CapturingChatModel):
    def chat(self, messages, *, max_output_tokens=None, temperature=None):
        self.calls.append((messages, max_output_tokens, temperature))
        return LLMResponse(text="哎哟，", provider=self.provider, model="test")


class TimeEnrichmentChatModel(CapturingChatModel):
    def chat(self, messages, *, max_output_tokens=None, temperature=None):
        self.calls.append((messages, max_output_tokens, temperature))
        return LLMResponse(
            text='{"events":[{"agentId":"agent_dad","text":"老爸听见屋里的动静，顺口和身边的人搭了一句话。"}]}',
            provider=self.provider,
            model="test",
        )


class BedtimeContextChatModel(CapturingChatModel):
    def chat(self, messages, *, max_output_tokens=None, temperature=None):
        self.calls.append((messages, max_output_tokens, temperature))
        prompt = "\n".join(message["content"] for message in messages)
        self.last_prompt = prompt
        return LLMResponse(
            text=(
                '{"events":['
                '{"agentId":"agent_dad","text":"老爸压低声音说：我跟你讲，明早我去看看那扇门。"},'
                '{"agentId":"agent_mom","text":"老妈轻声回：哎哟，先睡，别把明天的事也搬到枕头边。"}'
                ']}'
            ),
            provider=self.provider,
            model="test",
        )


class StreamingChatModel(CapturingChatModel):
    def stream_chat(self, messages, *, max_output_tokens=None, temperature=None):
        self.calls.append((messages, max_output_tokens, temperature, "stream"))
        yield "我在"
        yield "厨房看着锅。"


class OracleMoveChatModel(CapturingChatModel):
    def chat(self, messages, *, max_output_tokens=None, temperature=None):
        self.calls.append(messages)
        return LLMResponse(
            text='{"speak":true,"reason":"神谕让自己拿菜去客厅","line":"哎哟，菜我拿去客厅，桌上腾个地方。","mood":"calm","memory_note":"老妈听见神谕后拿菜去了客厅。"}',
            provider=self.provider,
            model="test",
        )


class TestApi(unittest.TestCase):
    def _client(self) -> TestClient:
        """每个用例一个独立 app/driver，状态互不污染。"""
        return TestClient(create_app())

    def test_healthz(self):
        resp = self._client().get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_get_world_shape(self):
        resp = self._client().get("/world")
        self.assertEqual(resp.status_code, 200)
        world = resp.json()
        for key in ("rooms", "agents", "events", "phase", "clockLabel", "gridCols"):
            self.assertIn(key, world)
        self.assertIsInstance(world["agents"], list)
        self.assertIsInstance(world["rooms"], list)
        self.assertIsInstance(world["events"], list)

    def test_post_oracle_appends_events(self):
        resp = self._client().post("/oracle", json={"text": "要下雨了"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("events", data)
        self.assertIn("world", data)
        self.assertGreater(len(data["events"]), 0)
        # 首条应为「神谕已投放」，来源 oracle
        self.assertEqual(data["events"][0]["source"], "oracle")

    def test_post_oracle_can_move_named_agent_with_llm_reply(self):
        model = OracleMoveChatModel()
        client = TestClient(create_app(chat_model=model))

        resp = client.post("/oracle", json={"text": "老妈拿菜去客厅"})

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        mom = next(agent for agent in data["world"]["agents"] if agent["id"] == "agent_mom")
        self.assertEqual(mom["roomId"], "room_living")
        self.assertEqual(data["events"][0]["source"], "oracle")
        self.assertEqual(data["events"][1]["source"], "move")
        self.assertEqual(data["events"][1]["agentId"], "agent_mom")
        self.assertIn("菜我拿去客厅", data["events"][1]["text"])
        prompt = "\n".join(message["content"] for message in model.calls[0])
        self.assertIn("刚才的神谕是：「老妈拿菜去客厅」", prompt)
        self.assertIn("本次移动的直接缘由", prompt)

    def test_post_oracle_can_order_agent_to_come_to_room(self):
        model = OracleMoveChatModel()
        client = TestClient(create_app(chat_model=model))

        resp = client.post("/oracle", json={"text": "老爸来书房学习"})

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        dad = next(agent for agent in data["world"]["agents"] if agent["id"] == "agent_dad")
        self.assertEqual(dad["roomId"], "room_study")
        self.assertEqual(data["events"][0]["source"], "oracle")
        self.assertEqual(data["events"][1]["source"], "move")
        self.assertEqual(data["events"][1]["agentId"], "agent_dad")
        prompt = "\n".join(message["content"] for message in model.calls[0])
        self.assertIn("刚才的神谕是：「老爸来书房学习」", prompt)

    def test_post_move_changes_room(self):
        client = self._client()
        # 老爸初始在客厅，移入厨房后 world 中其 roomId 应已改变
        resp = client.post("/agents/agent_dad/move", json={"roomId": "room_kitchen"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        dad = next(a for a in data["world"]["agents"] if a["id"] == "agent_dad")
        self.assertEqual(dad["roomId"], "room_kitchen")
        # 本次事件为 move
        self.assertEqual(data["events"][0]["source"], "move")

    def test_advance_time_returns_and_persists_async_enrichment(self):
        model = TimeEnrichmentChatModel()
        client = TestClient(create_app(chat_model=model))

        resp = client.post("/world/advance-time")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("enrichmentJobId", data)
        self.assertEqual(data["events"][0]["source"], "time")

        job_id = data["enrichmentJobId"]
        job = None
        for _ in range(5):
            got = client.get(f"/world/time-enrichments/{job_id}")
            self.assertEqual(got.status_code, 200)
            job = got.json()
            if job["status"] == "done":
                break
            time.sleep(0.05)

        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "done")
        self.assertGreaterEqual(len(job["events"]), 1)
        self.assertEqual(job["events"][0]["source"], "catchup")
        self.assertIn("老爸听见屋里的动静", job["world"]["events"][-1]["text"])
        self.assertGreaterEqual(len(model.calls), 1)

    def test_advance_from_late_night_uses_new_day_context(self):
        base = make_initial_world(now=1_000_000)
        agents = tuple(
            replace(agent, roomId="room_bedroom")
            if agent.id in {"agent_dad", "agent_mom"}
            else agent
            for agent in base.agents
        )
        world = replace(
            base,
            phase="lateNight",
            clockLabel=phase_label("lateNight"),
            agents=agents,
        )
        model = BedtimeContextChatModel()
        client = TestClient(create_app(driver=WorldDriver(world), chat_model=model))

        resp = client.post("/world/advance-time")
        self.assertEqual(resp.status_code, 200)
        job_id = resp.json()["enrichmentJobId"]

        job = None
        for _ in range(5):
            got = client.get(f"/world/time-enrichments/{job_id}")
            self.assertEqual(got.status_code, 200)
            job = got.json()
            if job["status"] == "done":
                break
            time.sleep(0.05)

        self.assertEqual(job["status"], "done")
        self.assertNotIn("bedtime_copresent", model.last_prompt)
        self.assertIn("'type': 'none'", model.last_prompt)

    def test_time_enrichment_events_are_returned_after_move_anchor(self):
        base = make_initial_world(now=1_000_000)
        world = replace(base, phase="night", clockLabel=phase_label("night"))
        model = BedtimeContextChatModel()
        client = TestClient(create_app(driver=WorldDriver(world), chat_model=model))

        resp = client.post("/world/advance-time")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        move_event = next(event for event in data["events"] if event["source"] == "move")
        job_id = data["enrichmentJobId"]

        job = None
        for _ in range(5):
            got = client.get(f"/world/time-enrichments/{job_id}")
            self.assertEqual(got.status_code, 200)
            job = got.json()
            if job["status"] == "done":
                break
            time.sleep(0.05)

        self.assertEqual(job["status"], "done")
        self.assertEqual(job["anchorEventId"], move_event["id"])
        self.assertTrue(all(event["anchorEventId"] == move_event["id"] for event in job["events"]))
        self.assertTrue(all(event["clockLabel"] == move_event["clockLabel"] for event in job["events"]))
        event_ids = [event["id"] for event in job["world"]["events"]]
        anchor_index = event_ids.index(move_event["id"])
        self.assertEqual(
            event_ids[anchor_index + 1 : anchor_index + 1 + len(job["events"])],
            [event["id"] for event in job["events"]],
        )

        next_resp = client.post("/world/advance-time")
        self.assertEqual(next_resp.status_code, 200)
        next_event_ids = [event["id"] for event in next_resp.json()["world"]["events"]]
        next_anchor_index = next_event_ids.index(move_event["id"])
        self.assertEqual(
            next_event_ids[next_anchor_index + 1 : next_anchor_index + 1 + len(job["events"])],
            [event["id"] for event in job["events"]],
        )

    def test_post_agent_chat_replies_without_world_mutation(self):
        client = TestClient(create_app(chat_model=StubChatModel()))
        before = client.get("/world").json()
        resp = client.post("/agents/agent_dad/chat", json={"text": "今天过得怎么样？"})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["reply"])
        after = client.get("/world").json()
        self.assertEqual(after["events"], before["events"])
        self.assertEqual(after["agents"], before["agents"])

    def test_post_agent_chat_writes_private_memory_for_later_chat(self):
        memory = InMemoryStore()
        model = CapturingChatModel()
        driver = WorldDriver(make_initial_world(now=1_000_000), memory=memory)
        client = TestClient(create_app(driver=driver, chat_model=model))

        first = client.post("/agents/agent_dad/chat", json={"text": "我不吃香菜"})
        self.assertEqual(first.status_code, 200)
        got = memory.search("agent:agent_dad", "香菜", top_k=5, token_cap=100_000)
        self.assertTrue(any("我不吃香菜" in item.text for item in got))

        second = client.post("/agents/agent_dad/chat", json={"text": "你还记得我的口味吗？"})
        self.assertEqual(second.status_code, 200)
        second_prompt = "\n".join(m["content"] for m in model.calls[-1])
        self.assertIn("私聊记忆", second_prompt)
        self.assertIn("我不吃香菜", second_prompt)

    def test_post_agent_chat_streams_chunks_and_writes_private_memory(self):
        memory = InMemoryStore()
        model = StreamingChatModel()
        driver = WorldDriver(make_initial_world(now=1_000_000), memory=memory)
        client = TestClient(create_app(driver=driver, chat_model=model))

        with client.stream(
            "POST",
            "/agents/agent_mom/chat/stream",
            json={"text": "你在干嘛？"},
        ) as resp:
            self.assertEqual(resp.status_code, 200)
            packets = [json.loads(line) for line in resp.iter_lines() if line]

        self.assertEqual([p["type"] for p in packets], ["delta", "delta", "done"])
        self.assertEqual("".join(p["text"] for p in packets if p["type"] == "delta"), "我在厨房看着锅。")
        self.assertEqual(packets[-1]["reply"], "我在厨房看着锅。")
        got = memory.search("agent:agent_mom", "你在干嘛", top_k=5, token_cap=100_000)
        self.assertTrue(any("我在厨房看着锅" in item.text for item in got))
        _messages, max_output_tokens, temperature, mode = model.calls[-1]
        self.assertEqual(mode, "stream")
        self.assertGreaterEqual(max_output_tokens, 900)
        self.assertEqual(temperature, 0.8)

    def test_post_agent_chat_uses_current_room_location(self):
        client = TestClient(create_app(chat_model=StubChatModel()))
        moved = client.post("/agents/agent_fridge/move", json={"roomId": "room_bath"})
        self.assertEqual(moved.status_code, 200)

        resp = client.post("/agents/agent_fridge/chat", json={"text": "你现在在哪？"})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("卫生间", resp.json()["reply"])
        self.assertNotIn("厨房", resp.json()["reply"])

    def test_post_agent_chat_prompt_contains_live_location_fact(self):
        model = CapturingChatModel()
        driver = WorldDriver(make_initial_world(now=1_000_000))
        client = TestClient(create_app(driver=driver, chat_model=model))

        moved = client.post("/agents/agent_fridge/move", json={"roomId": "room_bath"})
        self.assertEqual(moved.status_code, 200)
        resp = client.post("/agents/agent_fridge/chat", json={"text": "你在哪个房间？"})
        self.assertEqual(resp.status_code, 200)

        prompt = "\n".join(m["content"] for m in model.calls[-1])
        self.assertIn("当前世界事实：我现在在卫生间。", prompt)
        self.assertIn("不要按常识或人设猜测", prompt)

    def test_post_agent_chat_filters_incomplete_model_reply(self):
        model = ShortReplyChatModel()
        client = TestClient(create_app(chat_model=model))

        resp = client.post("/agents/agent_mom/chat", json={"text": "老妈在干嘛？"})
        self.assertEqual(resp.status_code, 200)
        reply = resp.json()["reply"]
        self.assertNotEqual(reply, "哎哟，")
        self.assertIn("厨房", reply)

        _messages, max_output_tokens, _temperature = model.calls[-1]
        self.assertGreaterEqual(max_output_tokens, 900)

    def test_post_agent_chat_validates_input(self):
        client = TestClient(create_app(chat_model=StubChatModel()))
        self.assertEqual(client.post("/agents/missing/chat", json={"text": "你好"}).status_code, 404)
        self.assertEqual(client.post("/agents/agent_dad/chat", json={"text": "  "}).status_code, 400)

    def test_world_dict_roundtrip(self):
        """序列化↔反序列化对称：守护前后端字段严格对齐这条红线。"""
        world = make_initial_world(now=1_000_000)
        restored = world_from_dict(world_to_dict(world))
        self.assertEqual(restored, world)

    def test_world_dict_keeps_anchored_events_next_to_source_event(self):
        """Async enrichment may finish late, but display order must stay anchored."""
        anchor = WorldEvent(id="evt_anchor", ts=1, text="anchor", source="move")
        later = WorldEvent(id="evt_later", ts=2, text="later", source="time")
        child = WorldEvent(
            id="evt_child",
            ts=3,
            text="child",
            source="catchup",
            anchorEventId="evt_anchor",
        )
        world = replace(make_initial_world(now=1_000_000), events=(anchor, later, child))

        data = world_to_dict(world)

        self.assertEqual([event["id"] for event in data["events"]], ["evt_anchor", "evt_child", "evt_later"])
        self.assertEqual(data["events"][1]["anchorEventId"], "evt_anchor")
        self.assertEqual(world_from_dict(data).events[1].anchorEventId, "evt_anchor")

    def test_world_dict_keeps_orphan_anchored_events_in_place(self):
        """If an old anchor is missing, do not push the child event to the end."""
        first = WorldEvent(
            id="evt_first",
            ts=1,
            text="first",
            source="catchup",
            anchorEventId="evt_missing",
        )
        second = WorldEvent(id="evt_second", ts=2, text="second", source="time")
        world = replace(make_initial_world(now=1_000_000), events=(first, second))

        data = world_to_dict(world)

        self.assertEqual([event["id"] for event in data["events"]], ["evt_first", "evt_second"])

    # ---- F07b：创建漏斗接 LLM 合成 ----
    def test_personas_draft_structured(self):
        """/personas/draft 返回结构化草稿：字段齐全、kind 一致、mood 合法。"""
        resp = self._client().post(
            "/personas/draft",
            json={
                "kind": "appliance",
                "caresAbout": "家里的灯有没有关",
                "angryReaction": "嗡嗡地摔门",
                "happyReaction": "开心地转起来",
            },
        )
        self.assertEqual(resp.status_code, 200)
        draft = resp.json()
        for key in ("kind", "name", "avatar", "persona", "monologue", "mood", "catchphrase"):
            self.assertIn(key, draft)
        self.assertEqual(draft["kind"], "appliance")
        self.assertIn(draft["mood"], _VALID_MOODS)
        # 纯合成、无副作用：不写 world，agents 数量不变
        self.assertEqual(len(self._client().get("/world").json()["agents"]), 4)

    def test_personas_draft_stub_stable(self):
        """无 key（StubChatModel）路径：同一问答稳定不崩，mood 合法，非法 kind 归一。"""
        client = TestClient(create_app(chat_model=StubChatModel()))
        payload = {"kind": "bogus", "caresAbout": "孩子有没有吃饱", "seed": 0}
        first = client.post("/personas/draft", json=payload).json()
        second = client.post("/personas/draft", json=payload).json()
        self.assertEqual(first, second)  # 确定性可复现
        self.assertEqual(first["kind"], "human")  # 非法 kind 归一到 human
        self.assertIn(first["mood"], _VALID_MOODS)
        self.assertTrue(first["persona"])  # 有兜底草稿，非空

    def test_post_agents_adds_agent_and_event(self):
        """/agents：world.agents +1，追加一条 source=opening 的到来事件，新角色在指定房间。"""
        client = self._client()
        before = len(client.get("/world").json()["agents"])
        resp = client.post(
            "/agents",
            json={
                "kind": "human",
                "name": "小明",
                "avatar": "🧒",
                "persona": "爱问为什么的好奇小孩。",
                "monologue": "今天有什么新鲜事呢？",
                "mood": "happy",
                "catchphrase": "为什么呀",
                "roomId": "room_living",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["world"]["agents"]), before + 1)
        self.assertEqual(data["events"][0]["source"], "opening")
        new_agent = data["world"]["agents"][-1]
        self.assertEqual(new_agent["name"], "小明")
        self.assertEqual(new_agent["roomId"], "room_living")
        self.assertIn(new_agent["mood"], _VALID_MOODS)

    def test_post_agents_stub_stable(self):
        """无 key 路径下创建角色稳定不崩：非法 mood 归 calm，缺 name/avatar 有兜底。"""
        client = TestClient(create_app(chat_model=StubChatModel()))
        resp = client.post(
            "/agents",
            json={"kind": "appliance", "mood": "excited", "roomId": "room_kitchen"},
        )
        self.assertEqual(resp.status_code, 200)
        new_agent = resp.json()["world"]["agents"][-1]
        self.assertEqual(new_agent["mood"], "calm")  # 非法 mood 归一到 calm
        self.assertTrue(new_agent["name"])  # 缺 name 有兜底
        self.assertTrue(new_agent["avatar"])  # 缺 avatar 有兜底
        self.assertEqual(new_agent["roomId"], "room_kitchen")

    def test_put_world_floor_plan_persists_geometry_and_rooms(self):
        """保存自定义户型：floorPlan 进 world，新空间先进入 rooms，后续 move 才合法。"""
        client = self._client()
        floor_plan = {
            "id": "custom_plan",
            "name": "自己拼的小家",
            "width": 1280,
            "height": 760,
            "rooms": [
                {
                    "id": "plan_hobby",
                    "roomId": "room_hobby",
                    "kind": "study",
                    "name": "手作间",
                    "floor": "tatami",
                    "polygon": [{"x": 100, "y": 100}, {"x": 360, "y": 100}, {"x": 360, "y": 280}, {"x": 100, "y": 280}],
                    "labelAnchor": {"x": 140, "y": 160},
                    "agentAnchors": [{"x": 220, "y": 230}],
                }
            ],
            "walls": [],
            "doors": [],
            "windows": [],
        }
        resp = client.put(
            "/world/floor-plan",
            json={
                "floorPlan": floor_plan,
                "addedRooms": [
                    {
                        "id": "room_hobby",
                        "kind": "study",
                        "name": "手作间",
                        "emoji": "📚",
                        "accent": "#b8863b",
                        "pos": {"col": 1, "row": 1},
                    }
                ],
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["world"]["floorPlan"], floor_plan)
        self.assertTrue(any(room["id"] == "room_hobby" for room in data["world"]["rooms"]))
        self.assertEqual(data["events"][0]["source"], "system")

    def test_put_world_floor_plan_removes_deleted_rooms(self):
        client = self._client()
        current = client.get("/world").json()
        kept = current["floorPlan"]["rooms"][:1] if current.get("floorPlan") else [
            {
                "id": "plan_living",
                "roomId": "room_living",
                "kind": "living",
                "name": "客厅",
                "floor": "oak",
                "polygon": [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 100}, {"x": 0, "y": 100}],
                "labelAnchor": {"x": 20, "y": 20},
                "agentAnchors": [{"x": 50, "y": 80}],
            }
        ]
        floor_plan = {
            "id": "custom_plan",
            "name": "删掉多余空间",
            "width": 400,
            "height": 300,
            "rooms": kept,
            "walls": [],
            "doors": [],
            "windows": [],
        }
        resp = client.put("/world/floor-plan", json={"floorPlan": floor_plan, "addedRooms": []})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        kept_ids = {room["roomId"] for room in kept}
        self.assertEqual({room["id"] for room in data["world"]["rooms"]}, kept_ids)
        self.assertTrue(all(agent["roomId"] in kept_ids for agent in data["world"]["agents"]))

    def test_delete_agent_removes_module(self):
        client = self._client()
        before = client.get("/world").json()
        agent_id = before["agents"][0]["id"]
        resp = client.delete(f"/agents/{agent_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(any(agent["id"] == agent_id for agent in data["world"]["agents"]))
        self.assertEqual(data["events"][0]["source"], "system")


if __name__ == "__main__":
    unittest.main()
