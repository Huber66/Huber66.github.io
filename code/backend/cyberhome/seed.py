"""种子世界 —— 对齐 `src/mock/data.ts` 的 makeInitialWorld()。

7 个语义空间（客厅/厨房/卧室/书房/玄关/卫生间/阳台）+ 4 角色（老爸/老妈/老空调/冰箱）。人设与措辞
逐字对齐前端，确保前后端是同一个「我们家」，便于替换 mock 后无缝衔接。
"""

from __future__ import annotations

import itertools

from .models import Agent, GridPos, Room, World, WorldEvent
from .worldclock import phase_label

_evt_counter = itertools.count(1)
_agent_counter = itertools.count(1)


def new_event_id() -> str:
    """事件 id 生成，对齐前端 newEventId() 的 'evt_xxx' 口径。"""
    return f"evt_{next(_evt_counter)}"


def new_agent_id() -> str:
    """新角色 id 生成，对齐前端 uid('agent') 的 'agent_xxx' 口径（F07b 创建漏斗用）。"""
    return f"agent_{next(_agent_counter)}"


ROOMS: tuple[Room, ...] = (
    Room(id="room_living", kind="living", name="客厅", emoji="🛋️", pos=GridPos(1, 1), accent="#c9743b"),
    Room(id="room_kitchen", kind="kitchen", name="厨房", emoji="🍳", pos=GridPos(2, 1), accent="#4e9d6b"),
    Room(id="room_bedroom", kind="bedroom", name="卧室", emoji="🛏️", pos=GridPos(1, 2), accent="#5b6bd6"),
    Room(id="room_study", kind="study", name="书房", emoji="📚", pos=GridPos(2, 2), accent="#b8863b"),
    Room(id="room_entry", kind="entry", name="玄关", emoji="🚪", pos=GridPos(1, 3), accent="#8a7650"),
    Room(id="room_bath", kind="bath", name="卫生间", emoji="🛁", pos=GridPos(2, 3), accent="#5f9fb8"),
    Room(id="room_balcony", kind="balcony", name="阳台", emoji="🪴", pos=GridPos(3, 1), accent="#6fa56d"),
)

AGENTS: tuple[Agent, ...] = (
    Agent(
        id="agent_dad", kind="human", name="老爸", avatar="👨",
        persona="嘴硬心软的退休工程师，爱看天气预报，嫌东嫌西但都记在心里。",
        monologue="又降温了…那小子加衣服了没。",
        mood="grumpy", catchphrase="我跟你讲", roomId="room_living",
    ),
    Agent(
        id="agent_mom", kind="human", name="老妈", avatar="👩",
        persona="爱唠叨的能量核心，记得每个人的口味，忙起来就哼歌。",
        monologue="冰箱里还有俩鸡蛋，晚上做个蛋炒饭吧。",
        mood="happy", catchphrase="哎哟", roomId="room_kitchen",
    ),
    Agent(
        id="agent_ac", kind="appliance", name="老空调", avatar="❄️",
        persona="服役十二年的老空调，关节不利索，爱抱怨但一到夏天最卖力。",
        monologue="哼，又没人给我擦灰…咳，滤网都堵了。",
        mood="grumpy", catchphrase="想当年", roomId="room_living",
    ),
    Agent(
        id="agent_fridge", kind="appliance", name="冰箱", avatar="🧊",
        persona="操心全家吃饭的话痨冰箱，惦记着有没有人忘了关门。",
        monologue="昨晚那扇门…是不是又没关严实。",
        mood="calm", catchphrase="我提醒你啊", roomId="room_kitchen",
    ),
)

# 开场故事文案，对齐前端 buildOpeningEvents()。
_OPENING_RAW: tuple[tuple[str, str], ...] = (
    ("老妈在厨房系上围裙，念叨着「今天做蛋炒饭」，油锅已经热上了。", "agent_mom"),
    ("老爸窝在客厅沙发看天气预报，嘟囔了一句「又降温」。", "agent_dad"),
    ("老空调被打开，嗡了一声，抱怨「关节又不利索了」。", "agent_ac"),
    ("冰箱小声提醒：「昨晚那扇门，好像没关严实哦。」", "agent_fridge"),
    ("老爸嫌空调太吵，老空调不服气：「想当年我可是家里最凉快的。」", "agent_ac"),
)


def _opening_events(now: int) -> tuple[WorldEvent, ...]:
    n = len(_OPENING_RAW)
    return tuple(
        WorldEvent(
            id=new_event_id(),
            ts=now - (n - i) * 40_000,
            text=text,
            source="opening",
            clockLabel="傍晚",
            agentId=agent_id,
        )
        for i, (text, agent_id) in enumerate(_OPENING_RAW)
    )


# ---- 氛围库存（打开时预生成，空闲心跳从这里取；不实时烧 token）----
# 逐字对齐前端 src/mock/data.ts 的 AMBIENT_LINES（10 条）。F11 起随 world 下发，
# 前端空闲心跳从 world.ambientInventory 取；前端保留同名常量作离线兜底。
AMBIENT_INVENTORY: tuple[str, ...] = (
    "空调轻轻嗡了一声，又安静下来。",
    "窗帘被穿堂风吹得动了动。",
    "厨房飘来一点葱花的香味。",
    "冰箱压缩机低低地响了一下。",
    "沙发上的抱枕被谁蹭歪了一点。",
    "墙上的挂钟滴答得格外清楚。",
    "客厅的灯忽明忽暗了一瞬，又稳住了。",
    "楼下有辆车驶过，灯光扫过墙面。",
    "老爸翻了个身，报纸滑到了地上。",
    "厨房水龙头没拧紧，滴了两滴水。",
)


def make_initial_world(now: int = 0) -> World:
    """组装初始世界。now 为基准毫秒时间戳（测试可传固定值保证可复现）。"""
    return World(
        id="world_home",
        name="我们家",
        phase="dusk",
        clockLabel=phase_label("dusk"),  # 从 worldclock 派生，杜绝与前端 PHASE_META 硬编码漂移
        gridCols=3,
        gridRows=3,
        rooms=ROOMS,
        agents=AGENTS,
        events=_opening_events(now),
        ambientInventory=AMBIENT_INVENTORY,
    )
