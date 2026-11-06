"""内核与事件溯源测试（步骤 5 验收，决策 32/33）。

重点证明「回放/回溯/what-if 是纯函数的免费午餐」，且 what-if A 档只读不污染主线。
零第三方依赖，标准库 unittest。运行：python -m unittest discover backend/tests
"""

import random
import unittest

from cyberhome import (
    StubReactor, Trigger, WorldDriver, agents_in_room, apply_patch,
    make_initial_world, step,
)
from cyberhome.models import WorldPatch


def _fixed_reactor() -> StubReactor:
    """固定 seed 的 reactor，保证 step() 可复现（事件溯源前提）。"""
    return StubReactor(rng=random.Random(42))


def _counter_event_id():
    n = {"i": 0}
    def gen():
        n["i"] += 1
        return f"evt_test_{n['i']}"
    return gen


class TestPurity(unittest.TestCase):
    def test_step_does_not_mutate_input(self):
        """step() 是纯函数：不得改动入参 world。"""
        world = make_initial_world(now=1_000_000)
        before_agents = world.agents
        before_events = world.events
        step(
            world,
            Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=2_000_000),
            reactor=_fixed_reactor(), event_id=_counter_event_id(),
        )
        # 入参对象引用未变（frozen + 返回新对象）
        self.assertIs(world.agents, before_agents)
        self.assertIs(world.events, before_events)
        self.assertEqual(world.agents[0].roomId, "room_living")  # 老爸仍在客厅

    def test_step_is_reproducible(self):
        """相同 (world, trigger, 固定 rng) → 相同输出。"""
        world = make_initial_world(now=1_000_000)
        trig = Trigger(kind="oracle", text="要下雨了", ts=2_000_000)
        e1, p1 = step(world, trig, reactor=_fixed_reactor(), event_id=_counter_event_id())
        e2, p2 = step(world, trig, reactor=_fixed_reactor(), event_id=_counter_event_id())
        self.assertEqual([e.text for e in e1], [e.text for e in e2])


class TestRoomPresence(unittest.TestCase):
    def test_agents_in_room(self):
        world = make_initial_world()
        living = agents_in_room(world, "room_living")
        self.assertEqual({a.id for a in living}, {"agent_dad", "agent_ac"})

    def test_move_changes_room_via_patch(self):
        world = make_initial_world(now=1_000_000)
        events, patch = step(
            world,
            Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=2_000_000),
            reactor=_fixed_reactor(), event_id=_counter_event_id(),
        )
        after = apply_patch(world, patch, events)
        moved = next(a for a in after.agents if a.id == "agent_dad")
        self.assertEqual(moved.roomId, "room_kitchen")
        # 移动文案应体现共处（厨房原有老妈、冰箱）
        self.assertEqual(events[0].source, "move")

    def test_oracle_can_order_agent_move(self):
        """神谕里明确“谁去哪个房间”时，应该触发真实 roomId 移动。"""
        from cyberhome.reactor import AgentReaction

        class OracleMoveReactor:
            def __init__(self):
                self.triggers = []

            def react(self, world, trigger, memory):
                self.triggers.append(trigger)
                if trigger.kind == "move":
                    return [
                        AgentReaction(
                            agent_id=trigger.agentId,
                            speak=True,
                            line="哎哟，菜我拿过去，客厅那边别乱放。",
                        )
                    ]
                return []

        reactor = OracleMoveReactor()
        world = make_initial_world(now=1_000_000)

        events, patch = step(
            world,
            Trigger(kind="oracle", text="老妈拿菜去客厅", ts=2_000_000),
            reactor=reactor,
            event_id=_counter_event_id(),
        )
        after = apply_patch(world, patch, events)
        mom = next(agent for agent in after.agents if agent.id == "agent_mom")

        self.assertEqual(mom.roomId, "room_living")
        self.assertEqual([trigger.kind for trigger in reactor.triggers], ["move"])
        self.assertEqual(reactor.triggers[0].text, "老妈拿菜去客厅")
        self.assertEqual([event.source for event in events], ["oracle", "move"])
        self.assertEqual(events[1].agentId, "agent_mom")
        self.assertIn("菜我拿过去", events[1].text)

    def test_oracle_can_order_agent_to_come_to_room(self):
        """“来某个房间”也应识别成移动命令，不能落回普通神谕广播。"""

        class RecordingReactor:
            def __init__(self):
                self.triggers = []

            def react(self, world, trigger, memory):
                self.triggers.append(trigger)
                return []

        reactor = RecordingReactor()
        world = make_initial_world(now=1_000_000)

        events, patch = step(
            world,
            Trigger(kind="oracle", text="老爸来书房学习", ts=2_000_000),
            reactor=reactor,
            event_id=_counter_event_id(),
        )
        after = apply_patch(world, patch, events)
        dad = next(agent for agent in after.agents if agent.id == "agent_dad")

        self.assertEqual(dad.roomId, "room_study")
        self.assertEqual([trigger.kind for trigger in reactor.triggers], ["move"])
        self.assertEqual([event.source for event in events], ["oracle", "move"])
        self.assertEqual(events[1].agentId, "agent_dad")
        self.assertIn("书房", events[1].text)

    def test_plain_oracle_does_not_move_agent(self):
        world = make_initial_world(now=1_000_000)

        events, patch = step(
            world,
            Trigger(kind="oracle", text="客厅今天好像更亮了", ts=2_000_000),
            reactor=_fixed_reactor(),
            event_id=_counter_event_id(),
        )

        self.assertNotIn("agent_mom", patch.agent_updates)
        self.assertEqual(events[0].source, "oracle")


class TestEventSourcing(unittest.TestCase):
    """决策 32/33 的核心验收。"""

    def _driver_with_history(self) -> WorldDriver:
        d = WorldDriver(make_initial_world(now=1_000_000), reactor=_fixed_reactor(), event_id=_counter_event_id())
        d.dispatch(Trigger(kind="oracle", text="要下雨了", ts=2_000_000))
        d.dispatch(Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=3_000_000))
        return d

    def test_replay_events_accumulates(self):
        """回放：initial events + 各步 events 依序汇总。"""
        d = self._driver_with_history()
        opening = len(make_initial_world(now=1_000_000).events)
        all_events = d.store.replay_events()
        # 应含开场 + 神谕相关 + 移动，且严格多于开场
        self.assertGreater(len(all_events), opening)
        # 回放到第 0 步（只含神谕那次）应少于回放到最新
        until_first = d.store.replay_events(until_seq=0)
        self.assertLess(len(until_first), len(all_events))

    def test_world_at_folds_patch(self):
        """回溯：折叠 patch 到某时刻得当时快照。"""
        d = self._driver_with_history()
        # 创世：老爸在客厅
        genesis = d.store.world_at(seq=-1)
        self.assertEqual(next(a for a in genesis.agents if a.id == "agent_dad").roomId, "room_living")
        # 第 0 步（神谕）后：老爸仍在客厅（神谕不改位置）
        after_oracle = d.store.world_at(seq=0)
        self.assertEqual(next(a for a in after_oracle.agents if a.id == "agent_dad").roomId, "room_living")
        # 第 1 步（移动）后：老爸到厨房
        after_move = d.store.world_at(seq=1)
        self.assertEqual(next(a for a in after_move.agents if a.id == "agent_dad").roomId, "room_kitchen")
        # 折叠出的最新快照应与 driver 当前世界一致
        self.assertEqual(
            {a.id: a.roomId for a in d.store.world_at().agents},
            {a.id: a.roomId for a in d.world.agents},
        )

    def test_whatif_is_readonly(self):
        """what-if A 档：预演不落库、不污染主线（决策 33）。"""
        d = self._driver_with_history()
        log_len_before = len(d.store)
        world_before = {a.id: a.roomId for a in d.world.agents}

        # 预演：若在创世时刻改让老妈进卧室会怎样
        events, patch, hypo = d.store.fork_readonly(
            at_seq=-1,
            trigger=Trigger(kind="move", agentId="agent_mom", roomId="room_bedroom", ts=9_000_000),
            reactor=_fixed_reactor(), event_id=_counter_event_id(),
        )
        # 假想世界里老妈在卧室
        self.assertEqual(next(a for a in hypo.agents if a.id == "agent_mom").roomId, "room_bedroom")
        # 但主线：日志长度不变、当前世界不变（推演完即弃）
        self.assertEqual(len(d.store), log_len_before)
        self.assertEqual({a.id: a.roomId for a in d.world.agents}, world_before)
        # 主线里老妈仍在厨房
        self.assertEqual(next(a for a in d.world.agents if a.id == "agent_mom").roomId, "room_kitchen")


class TestMemoryBudget(unittest.TestCase):
    """决策 34：search() 必带 top_k/token_cap，双重预算裁剪生效。"""

    def test_search_respects_budgets(self):
        from cyberhome.memory import InMemoryStore
        store = InMemoryStore()
        for i in range(10):
            store.add(namespace="agent_dad", text=f"老爸记得第{i}件事情很长很长很长的一句", ts=1000 + i)
        # top_k 截断
        got = store.search("agent_dad", "老爸", top_k=3, token_cap=10_000)
        self.assertLessEqual(len(got), 3)
        # token_cap 收紧到只够 1 条
        tight = store.search("agent_dad", "老爸", top_k=10, token_cap=15)
        self.assertLessEqual(len(tight), 1)


class _MoodReactor:
    """测试用 reactor：oracle 让指定 agent 带 mood/monologue，move 让移动者带 mood。

    验证 engine 把 AgentReaction 的 mood/monologue 折进 patch（8.2）。
    """

    def react(self, world, trigger, memory):
        from cyberhome.reactor import AgentReaction
        if trigger.kind == "oracle":
            return [
                AgentReaction(
                    agent_id="agent_dad", speak=True, line="我记下了",
                    mood="happy", monologue="其实还挺高兴的",
                ),
                # 沉默但心情有变：也应折进 patch，但不产 reaction 事件
                AgentReaction(agent_id="agent_mom", speak=False, line="", mood="tired"),
            ]
        if trigger.kind == "move":
            return [
                AgentReaction(
                    agent_id=trigger.agentId, speak=True, line="我挪过来了",
                    mood="calm",
                )
            ]
        return []


class TestReactionPatch(unittest.TestCase):
    """F06 8.2：mood/monologue 经 reaction 折进 patch 并落入 world。"""

    def test_oracle_folds_mood_and_monologue(self):
        world = make_initial_world(now=1_000_000)
        events, patch = step(
            world, Trigger(kind="oracle", text="要下雨了", ts=2_000_000),
            reactor=_MoodReactor(), event_id=_counter_event_id(),
        )
        # patch 携带 mood/monologue
        self.assertEqual(patch.agent_updates["agent_dad"]["mood"], "happy")
        self.assertEqual(patch.agent_updates["agent_dad"]["monologue"], "其实还挺高兴的")
        # 沉默者仅 mood 变化也被折入（不产事件但状态变）
        self.assertEqual(patch.agent_updates["agent_mom"]["mood"], "tired")
        # 折进 world 后生效
        after = apply_patch(world, patch, events)
        dad = next(a for a in after.agents if a.id == "agent_dad")
        self.assertEqual(dad.mood, "happy")
        self.assertEqual(dad.monologue, "其实还挺高兴的")
        # 只有开口者（老爸）产 reaction 事件，沉默的老妈不产
        reaction_ids = [e.agentId for e in events if e.source == "reaction"]
        self.assertIn("agent_dad", reaction_ids)
        self.assertNotIn("agent_mom", reaction_ids)

    def test_move_merges_roomid_and_mood(self):
        """move：roomId（engine）与 mood（reaction）合并进同一 agent_updates，不互相覆盖。"""
        world = make_initial_world(now=1_000_000)
        events, patch = step(
            world, Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=2_000_000),
            reactor=_MoodReactor(), event_id=_counter_event_id(),
        )
        upd = patch.agent_updates["agent_dad"]
        self.assertEqual(upd["roomId"], "room_kitchen")
        self.assertEqual(upd["mood"], "calm")
        after = apply_patch(world, patch, events)
        dad = next(a for a in after.agents if a.id == "agent_dad")
        self.assertEqual(dad.roomId, "room_kitchen")
        self.assertEqual(dad.mood, "calm")

    def test_move_always_emits_at_least_one_event(self):
        """move 至少 1 条事件（前端依赖 events.length>0），reactor 空返回时用中性兜底。"""
        class SilentReactor:
            def react(self, world, trigger, memory):
                return []  # 全体沉默
        world = make_initial_world(now=1_000_000)
        events, patch = step(
            world, Trigger(kind="move", agentId="agent_dad", roomId="room_kitchen", ts=2_000_000),
            reactor=SilentReactor(), event_id=_counter_event_id(),
        )
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0].source, "move")
        self.assertEqual(events[0].agentId, "agent_dad")
        # roomId 仍变更（engine 职责，不依赖 reactor）
        self.assertEqual(patch.agent_updates["agent_dad"]["roomId"], "room_kitchen")

    def test_step_pure_with_reaction_patch(self):
        """带 mood patch 的 step 仍纯：不改动入参 world。"""
        world = make_initial_world(now=1_000_000)
        before = world.agents
        step(
            world, Trigger(kind="oracle", text="要下雨了", ts=2_000_000),
            reactor=_MoodReactor(), event_id=_counter_event_id(),
        )
        self.assertIs(world.agents, before)
        self.assertEqual(world.agents[0].mood, "grumpy")  # 老爸原 mood 未被改


if __name__ == "__main__":
    unittest.main()
