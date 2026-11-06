"""Agent 人设修订后端闭环测试（P0-1）。

覆盖：
- 草稿端点纯合成、无副作用。
- 确认端点更新当前 persona 锚。
- public 产生可见 system 事件。
- silent 不产生可见事件，但仍追加事件溯源日志。
- moveWithRevision 复用 move 语义。
"""

import random
import unittest

from fastapi.testclient import TestClient

from cyberhome import StubChatModel, StubReactor, WorldDriver, make_initial_world
from cyberhome.api import create_app


def _fixed_reactor() -> StubReactor:
    return StubReactor(rng=random.Random(42))


def _counter_event_id():
    n = {"i": 0}

    def gen():
        n["i"] += 1
        return f"evt_revision_{n['i']}"

    return gen


def _draft(agent_id: str = "agent_dad", **updates) -> dict:
    draft = {
        "agentId": agent_id,
        "mode": "adjust",
        "name": "夜班老爸",
        "avatar": "老爸-工程师眼镜",
        "persona": "夜班老爸从今天起更惦记家里半夜有没有人饿着。",
        "monologue": "家里半夜有没有人饿着，我还是得看一眼。",
        "mood": "calm",
        "catchphrase": "我看一眼就知道",
        "summary": "这个家的叙述更靠近 TA 的新牵挂。",
        "visibilitySuggestion": "public",
        "publicEventText": "这个家像是忽然更懂老爸了：他开始惦记半夜有没有人饿着。",
    }
    draft.update(updates)
    return draft


class TestPersonaRevisionApi(unittest.TestCase):
    def test_draft_revision_is_structured_and_side_effect_free(self):
        client = TestClient(create_app(chat_model=StubChatModel()))
        before = client.get("/world").json()
        dad_before = next(a for a in before["agents"] if a["id"] == "agent_dad")

        resp = client.post(
            "/agents/agent_dad/persona-revisions/draft",
            json={
                "mode": "adjust",
                "answers": {
                    "caresAbout": "家里半夜有没有人饿着",
                    "catchphrase": "我看一眼就知道",
                    "happyReaction": "把热汤端到桌边",
                },
                "seed": 0,
            },
        )
        self.assertEqual(resp.status_code, 200)
        draft = resp.json()
        for key in (
            "agentId",
            "mode",
            "name",
            "avatar",
            "persona",
            "monologue",
            "mood",
            "catchphrase",
            "summary",
            "visibilitySuggestion",
            "publicEventText",
        ):
            self.assertIn(key, draft)
        self.assertEqual(draft["agentId"], "agent_dad")
        self.assertEqual(draft["mode"], "adjust")
        self.assertTrue(draft["persona"])
        self.assertIn(draft["visibilitySuggestion"], ("silent", "public"))

        after = client.get("/world").json()
        dad_after = next(a for a in after["agents"] if a["id"] == "agent_dad")
        self.assertEqual(dad_after, dad_before)
        self.assertEqual(len(after["events"]), len(before["events"]))

    def test_confirm_public_revision_updates_agent_and_returns_system_event(self):
        client = TestClient(create_app(chat_model=StubChatModel()))
        resp = client.post(
            "/agents/agent_dad/persona-revisions/confirm",
            json={"draft": _draft(), "visibility": "public"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["events"]), 1)
        self.assertEqual(data["events"][0]["source"], "system")
        self.assertEqual(data["events"][0]["agentId"], "agent_dad")
        dad = next(a for a in data["world"]["agents"] if a["id"] == "agent_dad")
        self.assertEqual(dad["name"], "夜班老爸")
        self.assertEqual(dad["avatar"], "老爸-工程师眼镜")
        self.assertEqual(dad["catchphrase"], "我看一眼就知道")
        self.assertIn("半夜", dad["persona"])

    def test_confirm_silent_revision_still_records_event_log(self):
        driver = WorldDriver(
            make_initial_world(now=1_000_000),
            reactor=_fixed_reactor(),
            event_id=_counter_event_id(),
        )
        client = TestClient(create_app(driver=driver, chat_model=StubChatModel()))
        before_events = len(driver.world.events)
        before_log = len(driver.store)

        resp = client.post(
            "/agents/agent_dad/persona-revisions/confirm",
            json={"draft": _draft(), "visibility": "silent"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["events"], [])
        self.assertEqual(len(driver.store), before_log + 1)
        self.assertEqual(len(driver.world.events), before_events)
        dad = next(a for a in driver.world.agents if a.id == "agent_dad")
        self.assertEqual(dad.name, "夜班老爸")
        self.assertIn("半夜", dad.persona)

    def test_confirm_revision_with_move_reuses_move_semantics(self):
        driver = WorldDriver(
            make_initial_world(now=1_000_000),
            reactor=_fixed_reactor(),
            event_id=_counter_event_id(),
        )
        client = TestClient(create_app(driver=driver, chat_model=StubChatModel()))

        resp = client.post(
            "/agents/agent_dad/persona-revisions/confirm",
            json={
                "draft": _draft(),
                "visibility": "silent",
                "roomId": "room_kitchen",
                "moveWithRevision": True,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(len(data["events"]), 1)
        self.assertEqual(data["events"][0]["source"], "move")
        dad = next(a for a in data["world"]["agents"] if a["id"] == "agent_dad")
        self.assertEqual(dad["roomId"], "room_kitchen")
        # 一条修订溯源 + 一条 move 溯源。
        self.assertEqual(len(driver.store), 2)

    def test_revision_errors_are_explicit(self):
        client = TestClient(create_app(chat_model=StubChatModel()))
        self.assertEqual(
            client.post("/agents/missing/persona-revisions/draft", json={}).status_code,
            404,
        )
        self.assertEqual(
            client.post(
                "/agents/agent_dad/persona-revisions/confirm",
                json={"draft": _draft(agent_id="agent_mom"), "visibility": "public"},
            ).status_code,
            400,
        )
        self.assertEqual(
            client.post(
                "/agents/agent_dad/persona-revisions/confirm",
                json={
                    "draft": _draft(),
                    "visibility": "silent",
                    "roomId": "room_nowhere",
                    "moveWithRevision": True,
                },
            ).status_code,
            400,
        )


if __name__ == "__main__":
    unittest.main()
