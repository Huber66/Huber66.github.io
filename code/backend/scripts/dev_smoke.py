"""开发用烟测脚本（F14 · 14.3）——一条命令跑通核心 API 闭环，确认 demo 前后端合同没坏。

WHY in-process TestClient 而非起真进程（uvicorn）：
  1. **确定性**：注入 StubChatModel 走无 key 降级路径，同输入同输出，不受真 LLM 抖动影响，
     烟测结果可复现（红线：无 key 也能跑通闭环）。
  2. **不占端口 / 不联网**：TestClient 直接在进程内驱动 ASGI app，不 bind 端口、不发网络请求，
     本地反复跑与 CI 并发跑都不会撞端口、不依赖外网。
  3. **CI 友好**：无需等 uvicorn 起服务再轮询健康检查，一次函数调用即得响应，快且无竞态。

覆盖：GET /world → POST /oracle → POST /agents/{id}/move → POST /world/catchup → POST /world/opening。
每步打印简明结果；任一步失败打印清晰错误并以非 0 退出码返回；全通过打印总结并返回 0。

用法（在 backend/ 下）：
    python3 scripts/dev_smoke.py       # 直接运行
    python3 -m scripts.dev_smoke       # 以模块运行亦可
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接运行（python3 scripts/dev_smoke.py）时，sys.path[0] 是 scripts/，cyberhome 不可见；
# 把 backend/（本文件父目录的父目录）补进去。以 -m 运行时 backend/ 已在 path，此句幂等。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from cyberhome.api import create_app  # noqa: E402
from cyberhome.llm import StubChatModel  # noqa: E402


def _gist(text: str, limit: int = 60) -> str:
    """事件文本摘要：strip 后截断，避免烟测输出被长文案刷屏。"""
    text = (text or "").strip()
    return text if len(text) <= limit else f"{text[:limit]}…"


def _expect_ok(resp, step: str) -> dict:
    """断言 HTTP 200 并返回 JSON；非 200 抛出带步骤名的错误，供 main 捕获打印。"""
    if resp.status_code != 200:
        raise RuntimeError(f"{step} 返回非 200：{resp.status_code} {resp.text[:200]}")
    return resp.json()


def run_smoke() -> None:
    """依次跑核心端点烟测；任一步不符合预期即抛异常（由 main 捕获转非 0 退出）。"""
    # 走确定性 stub 路径：无 key 降级，不联网、同输入同输出（见模块 docstring WHY）。
    client = TestClient(create_app(chat_model=StubChatModel()))

    # 1/5 GET /world —— 世界快照，取角色/房间用于后续步骤。
    world = _expect_ok(client.get("/world"), "GET /world")
    agents, rooms, events = world["agents"], world["rooms"], world["events"]
    if not agents or len(rooms) < 2:
        raise RuntimeError(f"世界不满足烟测前提：agents={len(agents)} rooms={len(rooms)}（需 agents≥1、rooms≥2）")
    print(f"[dev_smoke] 1/5 GET /world → 角色 {len(agents)}、房间 {len(rooms)}、初始事件 {len(events)} 条")

    # 2/5 POST /oracle —— 投一句契机，在场角色各自反应。
    data = _expect_ok(client.post("/oracle", json={"text": "今天降温了，记得加件衣服"}), "POST /oracle")
    oracle_events = data["events"]
    if not oracle_events:
        raise RuntimeError("POST /oracle 未产生任何事件")
    print(f"[dev_smoke] 2/5 POST /oracle → 新增 {len(oracle_events)} 事件；首条: {_gist(oracle_events[0]['text'])}")

    # 3/5 POST /agents/{id}/move —— 取现有角色 + 一个不同于其当前房间的房间，确保真的发生移动。
    mover = agents[0]
    target = next((r for r in rooms if r["id"] != mover["roomId"]), None)
    if target is None:
        raise RuntimeError("找不到与角色当前房间不同的目标房间，无法烟测移动")
    data = _expect_ok(
        client.post(f"/agents/{mover['id']}/move", json={"roomId": target["id"]}),
        "POST /agents/{id}/move",
    )
    move_events = data["events"]
    moved = next((a for a in data["world"]["agents"] if a["id"] == mover["id"]), None)
    if moved is None or moved["roomId"] != target["id"]:
        raise RuntimeError(f"移动后角色 {mover['id']} 房间未更新到 {target['id']}")
    print(
        f"[dev_smoke] 3/5 POST /move → {mover['name']} 移入「{target['name']}」，"
        f"新增 {len(move_events)} 事件；首条: {_gist(move_events[0]['text'])}"
    )

    # 4/5 POST /world/catchup —— 惰性补演。默认 driver 未挂持久化（无 last_seen），返回空 events
    # 属正常（见 driver.catchup 三种空返回情形），不算失败。
    data = _expect_ok(client.post("/world/catchup"), "POST /world/catchup")
    catchup_events = data["events"]
    if catchup_events:
        print(f"[dev_smoke] 4/5 POST /catchup → 补演 {len(catchup_events)} 事件；首条: {_gist(catchup_events[0]['text'])}")
    else:
        print("[dev_smoke] 4/5 POST /catchup → （未挂持久化，无补演）")

    # 5/5 POST /world/opening —— 开场故事生成，追加进事件溯源。
    data = _expect_ok(client.post("/world/opening", json={"seed": 0}), "POST /world/opening")
    opening_events = data["events"]
    if not opening_events:
        raise RuntimeError("POST /world/opening 未产生任何 opening 事件")
    print(f"[dev_smoke] 5/5 POST /opening → 生成 {len(opening_events)} 条开场；首条: {_gist(opening_events[0]['text'])}")


def main() -> int:
    try:
        run_smoke()
    except Exception as exc:  # 任一步失败：打印清晰错误，非 0 退出（CI 据此判红）。
        print(f"[dev_smoke] 烟测失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("[dev_smoke] 全部烟测通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
