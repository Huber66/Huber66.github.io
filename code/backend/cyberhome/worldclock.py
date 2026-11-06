"""世界时钟 —— phase→文案的后端唯一事实源（F11，镜像前端 `PHASE_META`）。

WHY 独立模块：世界时钟后端化后，phase→(label/short) 的映射与 phase 轮转顺序
成了 engine（推进）与 seed（创世 clockLabel）共同依赖的知识。若各自硬编码，
两处极易漂移（改了 seed 忘了 engine）。抽成单一事实源，engine 和 seed 共用它，
从结构上杜绝漂移——这是「优雅=信息水平给定下熵最低」的落点。

它逐字镜像前端 `frontend/src/config/tuning.ts` 的 `PHASE_META`（label/short）与 `PHASE_ORDER`
（轮转顺序）。**只搬 label/short/next**——overlay/brightness 是纯前端 CSS 光照，
后端不需要（后端只管「现在是哪一档、叫什么」，光影渲染是前端的事）。
"""

from __future__ import annotations

from .models import WorldPhase

# phase 轮转顺序，逐字对齐前端 tuning.ts 的 PHASE_ORDER。环形前进：末档回到首档。
PHASE_ORDER: tuple[WorldPhase, ...] = ("day", "dusk", "night", "lateNight")
WEEKDAYS: tuple[str, ...] = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

# phase → (完整 label, 简短 short)，逐字镜像前端 PHASE_META 的 label/short。
# label 用于 world.clockLabel（信息条完整展示）；short 用于事件的 clockLabel（时间轴简短标记）。
_PHASE_TEXT: dict[WorldPhase, tuple[str, str]] = {
    "day": ("周三 午后 14:20", "午后"),
    "dusk": ("周三 傍晚 18:40", "傍晚"),
    "night": ("周三 夜晚 21:30", "夜晚"),
    "lateNight": ("周四 深夜 01:10", "深夜"),
}

_PHASE_TIME: dict[WorldPhase, str] = {
    "day": "午后 14:20",
    "dusk": "傍晚 18:40",
    "night": "夜晚 21:30",
    "lateNight": "深夜 01:10",
}


def next_phase(phase: WorldPhase) -> WorldPhase:
    """环形前进一档：day→dusk→night→lateNight→day。

    未知 phase（不在轮转顺序里）兜底回到首档 day——降级不崩，与前端
    nextPhase(indexOf 返回 -1 时 (-1+1)%4=0) 的口径一致。
    """
    idx = PHASE_ORDER.index(phase) if phase in PHASE_ORDER else -1
    return PHASE_ORDER[(idx + 1) % len(PHASE_ORDER)]


def phase_label(phase: WorldPhase) -> str:
    """phase → 完整时刻文案（如「周三 傍晚 18:40」）。供 world.clockLabel。"""
    return _PHASE_TEXT[phase][0]


def phase_short(phase: WorldPhase) -> str:
    """phase → 简短时刻标记（如「傍晚」）。供事件的 clockLabel。"""
    return _PHASE_TEXT[phase][1]


def advance_phase_label(current_label: str, current_phase: WorldPhase, next_phase_value: WorldPhase) -> str:
    """Return the next full clock label while preserving a real weekday cycle.

    The phase cycle stays four-step for light/life rhythm, but the weekday now
    moves like a real clock: day/dusk/night share the same date, night→lateNight
    crosses midnight, and lateNight→day stays on that new date.
    """
    weekday = _weekday_from_label(current_label) or _weekday_from_label(phase_label(current_phase))
    if next_phase_value == "lateNight" and current_phase != "lateNight":
        weekday = _next_weekday(weekday)
    return f"{weekday} {_PHASE_TIME[next_phase_value]}"


def _weekday_from_label(label: str) -> str | None:
    for weekday in WEEKDAYS:
        if label.startswith(weekday):
            return weekday
    return None


def _next_weekday(weekday: str) -> str:
    idx = WEEKDAYS.index(weekday) if weekday in WEEKDAYS else -1
    return WEEKDAYS[(idx + 1) % len(WEEKDAYS)]
