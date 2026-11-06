"""开发用清库脚本（F03 · 4.3）——把持久化的世界快照/事件日志/last_seen 抹掉。

WHY 以脚本为主而非端点：清库是**开发者操作**，不该长期挂在 HTTP 面上（一个误触
的 POST /world/reset 能抹掉整个 demo 数据，是不该给生产留的口子）。命令式脚本要求
显式在终端敲，天然隔离了误触，也不污染 F01 已定的端点合同。

用法（在 backend/ 下）：
    python3 -m scripts.reset_db            # 清默认库全部世界
    python3 scripts/reset_db.py            # 直接运行亦可
    CYBERHOME_DB_PATH=/tmp/x.db python3 -m scripts.reset_db
    python3 -m scripts.reset_db world_home # 只清指定 world_id
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接运行（python3 scripts/reset_db.py）时，sys.path[0] 是 scripts/，cyberhome 不可见；
# 把 backend/（本文件父目录的父目录）补进去。以 -m 运行时 backend/ 已在 path，此句幂等。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cyberhome.persistence import SQLitePersistenceStore, resolve_db_path  # noqa: E402


def main(argv: list[str]) -> int:
    world_id = argv[0] if argv else None
    path = resolve_db_path()
    store = SQLitePersistenceStore(path)
    try:
        store.reset(world_id)
    finally:
        store.close()
    scope = f"world_id={world_id}" if world_id else "所有世界"
    print(f"[reset_db] 已清空 {scope} @ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
