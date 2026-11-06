import unittest
from dataclasses import replace

from cyberhome.llm import LLMResponse
from cyberhome.models import WorldEvent
from cyberhome.seed import make_initial_world
from cyberhome.time_enrichment import synthesize_time_enrichment


class LooseTextModel:
    provider = "test"

    def chat(self, messages, *, max_output_tokens=None, temperature=None):
        return LLMResponse(
            text=(
                "1. 老爸把脚步放轻，在客厅门口停了一下，像是怕打断屋里的声音。\n"
                "2. 冰箱隔着墙低低响了一声，倒像是在提醒大家晚饭还没真正结束。"
            ),
            provider=self.provider,
            model="test",
        )


class GroupDialogueModel:
    provider = "test"

    def __init__(self):
        self.messages = None

    def chat(self, messages, *, max_output_tokens=None, temperature=None):
        self.messages = messages
        return LLMResponse(
            text=(
                '{"events":['
                '{"agentId":"agent_dad","text":"老爸压低声音说：行了，今天先睡吧，明早我去看看厨房那扇门。"},'
                '{"agentId":"agent_mom","text":"老妈把被角抻平：你少逞能，明早我也起来，别又一个人瞎忙。"}'
                ']}'
            ),
            provider=self.provider,
            model="test",
        )


class EmptyDialogueModel:
    provider = "test"

    def chat(self, messages, *, max_output_tokens=None, temperature=None):
        return LLMResponse(text='{"events":[]}', provider=self.provider, model="test")


class StubEmptyDialogueModel(EmptyDialogueModel):
    provider = "stub"


class WrongLocationDialogueModel:
    provider = "test"

    def chat(self, messages, *, max_output_tokens=None, temperature=None):
        return LLMResponse(
            text='{"events":[{"agentId":"agent_dad","text":"老爸到了厨房，脚步声轻得很，却把旁边的人和物都惊动了一点。"}]}',
            provider=self.provider,
            model="test",
        )


class MultiTurnDialogueModel:
    provider = "test"

    def chat(self, messages, *, max_output_tokens=None, temperature=None):
        return LLMResponse(
            text=(
                '{"events":['
                '{"agentId":"agent_dad","text":"老爸压低声音说：我跟你讲，灯别全关。"},'
                '{"agentId":"agent_mom","text":"老妈轻声回：行，就留床头那盏。"},'
                '{"agentId":"agent_dad","text":"老爸又补了一句：明早我去看看窗。"},'
                '{"agentId":"agent_mom","text":"老妈笑了笑：明早再说。"}'
                ']}'
            ),
            provider=self.provider,
            model="test",
        )


def bedroom_world():
    base = make_initial_world(now=1_000_000)
    agents = tuple(
        replace(agent, roomId="room_bedroom")
        if agent.id in {"agent_dad", "agent_mom"}
        else agent
        for agent in base.agents
    )
    return replace(base, phase="lateNight", clockLabel="周四 深夜 01:10", agents=agents)


class TimeEnrichmentTests(unittest.TestCase):
    def test_loose_non_json_model_output_is_kept(self):
        world = make_initial_world(now=1_000_000)
        beats = synthesize_time_enrichment(LooseTextModel(), world, (), seed=2_000_000)

        self.assertGreaterEqual(len(beats), 2)
        self.assertIn("脚步放轻", beats[0].text)
        self.assertIn("晚饭", beats[1].text)

    def test_group_move_to_bedroom_requires_llm_dialogue(self):
        world = bedroom_world()
        base_events = (
            WorldEvent(
                id="evt_move_dad",
                ts=2_000_000,
                text="老爸看了看天色，挪去卧室歇一会儿。",
                source="move",
                clockLabel=world.clockLabel,
                agentId="agent_dad",
            ),
            WorldEvent(
                id="evt_move_mom",
                ts=2_000_001,
                text="老妈看了看天色，挪去卧室歇一会儿。",
                source="move",
                clockLabel=world.clockLabel,
                agentId="agent_mom",
            ),
        )
        model = GroupDialogueModel()

        beats = synthesize_time_enrichment(model, world, base_events, seed=2_000_000)

        self.assertEqual([beat.agentId for beat in beats], ["agent_dad", "agent_mom"])
        prompt = "\n".join(message["content"] for message in model.messages)
        self.assertIn("group_move", prompt)
        self.assertIn("必须优先生成他们之间的对话事件", prompt)

    def test_late_night_bedroom_copresence_requires_dialogue_without_new_move(self):
        world = bedroom_world()
        base_events = (
            WorldEvent(
                id="evt_time",
                ts=2_000_000,
                text="天色转向深夜，家里的光影悄悄变了。",
                source="time",
                clockLabel=world.clockLabel,
                agentId=None,
            ),
        )
        model = GroupDialogueModel()

        beats = synthesize_time_enrichment(model, world, base_events, seed=2_000_000)

        self.assertEqual([beat.agentId for beat in beats], ["agent_dad", "agent_mom"])
        prompt = "\n".join(message["content"] for message in model.messages)
        self.assertIn("bedtime_copresent", prompt)
        self.assertIn("深夜已经同在卧室", prompt)

    def test_required_dialogue_keeps_at_most_one_line_per_agent(self):
        world = bedroom_world()
        base_events = (
            WorldEvent(
                id="evt_time",
                ts=2_000_000,
                text="天色转向深夜，家里的光影悄悄变了。",
                source="time",
                clockLabel=world.clockLabel,
                agentId=None,
            ),
        )

        beats = synthesize_time_enrichment(MultiTurnDialogueModel(), world, base_events, seed=2_000_000)

        self.assertEqual([beat.agentId for beat in beats], ["agent_dad", "agent_mom"])
        self.assertEqual(len(beats), 2)
        self.assertIn("灯别全关", beats[0].text)
        self.assertIn("床头", beats[1].text)

    def test_required_bedtime_dialogue_does_not_fake_local_reply_for_real_provider(self):
        world = bedroom_world()
        base_events = (
            WorldEvent(
                id="evt_move_parents",
                ts=2_000_000,
                text="老爸和老妈看了看天色，一起挪去卧室歇一会儿。",
                source="move",
                clockLabel=world.clockLabel,
                agentId=None,
            ),
        )

        beats = synthesize_time_enrichment(EmptyDialogueModel(), world, base_events, seed=2_000_000)

        self.assertEqual(beats, ())

    def test_required_bedtime_dialogue_can_fallback_for_stub_provider(self):
        world = bedroom_world()
        base_events = (
            WorldEvent(
                id="evt_move_parents",
                ts=2_000_000,
                text="老爸和老妈看了看天色，一起挪去卧室歇一会儿。",
                source="move",
                clockLabel=world.clockLabel,
                agentId=None,
            ),
        )

        beats = synthesize_time_enrichment(StubEmptyDialogueModel(), world, base_events, seed=2_000_000)

        self.assertGreaterEqual(len(beats), 2)
        self.assertEqual([beat.agentId for beat in beats[:2]], ["agent_dad", "agent_mom"])
        self.assertIn("老爸", beats[0].text)
        self.assertIn("老妈", beats[1].text)

    def test_rejects_llm_event_that_moves_agent_to_wrong_room(self):
        world = bedroom_world()
        base_events = (
            WorldEvent(
                id="evt_time",
                ts=2_000_000,
                text="天色转向深夜，家里的光影悄悄变了。",
                source="time",
                clockLabel=world.clockLabel,
                agentId=None,
            ),
        )

        beats = synthesize_time_enrichment(
            WrongLocationDialogueModel(),
            world,
            base_events,
            seed=2_000_000,
        )

        self.assertFalse(any("厨房" in beat.text for beat in beats))

    def test_non_required_fallback_does_not_use_empty_room_filler_phrase(self):
        base = make_initial_world(now=1_000_000)
        room_by_agent = {
            "agent_dad": "room_living",
            "agent_mom": "room_kitchen",
            "agent_ac": "room_bedroom",
            "agent_fridge": "room_study",
        }
        agents = tuple(replace(agent, roomId=room_by_agent.get(agent.id, agent.roomId)) for agent in base.agents)
        world = replace(base, agents=agents, events=())

        beats = synthesize_time_enrichment(EmptyDialogueModel(), world, (), seed=2_000_000)

        self.assertFalse(any("话咽回去" in beat.text for beat in beats))

    def test_dad_night_study_move_fallback_adds_only_one_note(self):
        base = make_initial_world(now=1_000_000)
        agents = tuple(
            replace(agent, roomId="room_study") if agent.id == "agent_dad" else agent
            for agent in base.agents
        )
        world = replace(base, phase="night", clockLabel="周三 夜晚 21:30", agents=agents, events=())
        base_events = (
            WorldEvent(
                id="evt_move_dad",
                ts=2_000_000,
                text="老爸把手边的事收了收，去书房待一会儿。",
                source="move",
                clockLabel=world.clockLabel,
                agentId="agent_dad",
            ),
        )

        beats = synthesize_time_enrichment(EmptyDialogueModel(), world, base_events, seed=2_000_000)

        self.assertEqual(len(beats), 1)
        self.assertEqual(beats[0].agentId, "agent_dad")
        self.assertIn("书房", beats[0].text)
        self.assertNotIn("阳台", beats[0].text)

    def test_other_move_fallback_can_still_add_multiple_notes(self):
        base = make_initial_world(now=1_000_000)
        agents = tuple(
            replace(agent, roomId="room_kitchen") if agent.id in {"agent_mom", "agent_fridge"} else agent
            for agent in base.agents
        )
        world = replace(base, phase="dusk", clockLabel="周三 傍晚 18:40", agents=agents, events=())
        base_events = (
            WorldEvent(
                id="evt_move_mom",
                ts=2_000_000,
                text="老妈惦记着锅灶和晚饭，自己去了厨房。",
                source="move",
                clockLabel=world.clockLabel,
                agentId="agent_mom",
            ),
        )

        beats = synthesize_time_enrichment(EmptyDialogueModel(), world, base_events, seed=2_000_000)

        self.assertGreaterEqual(len(beats), 2)
        self.assertFalse(any("话咽回去" in beat.text for beat in beats))

    def test_move_fallback_does_not_add_generic_room_filler(self):
        base = make_initial_world(now=1_000_000)
        room_by_agent = {
            "agent_dad": "room_living",
            "agent_mom": "room_kitchen",
            "agent_ac": "room_bedroom",
            "agent_fridge": "room_study",
        }
        agents = tuple(replace(agent, roomId=room_by_agent.get(agent.id, agent.roomId)) for agent in base.agents)
        world = replace(base, phase="dusk", clockLabel="周三 傍晚 18:40", agents=agents, events=())
        base_events = (
            WorldEvent(
                id="evt_move_mom",
                ts=2_000_000,
                text="老妈惦记着锅灶和晚饭，自己去了厨房。",
                source="move",
                clockLabel=world.clockLabel,
                agentId="agent_mom",
            ),
        )

        beats = synthesize_time_enrichment(EmptyDialogueModel(), world, base_events, seed=2_000_000)

        self.assertEqual(len(beats), 1)
        self.assertEqual(beats[0].agentId, "agent_mom")
        self.assertFalse(any("家里的时间往前挪了一格" in beat.text for beat in beats))
        self.assertFalse(any("先替大家显出了变化" in beat.text for beat in beats))


if __name__ == "__main__":
    unittest.main()
