"""服务器启动入口（方案 B）—— `python3 -m cyberhome` 起服务，并在此加载 .env。

WHY 加载点在这里、而**不**在 config/llm/api/__init__ 的顶层：
    测试（test_api / test_llm）会 import 那些模块。若在其 import 链上 `load_dotenv()`，
    本地 .env 里的真 key 会被灌进 os.environ，导致：
      · test_api 的 create_app() 无注入 chat_model → build_chat_model() 读到真 key →
        oracle/move 走真 LLMReactor **联网**；
      · test_llm 的 TestRealProvider（skipUnless os.getenv(...key)）被**真触发**联网。
    两者都破坏「测试不联网」红线。__main__ 是独立启动入口，测试从不 import 它，
    故这里加载 .env 既让 `python3 -m cyberhome` 拿到 key，又零污染测试。

用法（在 backend/ 下）：
    uv run --with-requirements requirements.txt python3 -m cyberhome
    # 或先装依赖后：python3 -m cyberhome
env 文件查找：
    backend/.env 优先；不存在时回退项目根目录 .env（服务启动入口专用，测试不加载）。
env 覆盖（可写在 .env 或命令行前缀）：
    CYBERHOME_LLM_PROVIDER=deepseek|mimo|stub   切换 provider（缺省按 key 自动降级）
    CYBERHOME_HOST / CYBERHOME_PORT             监听地址/端口（默认 127.0.0.1:8000，对齐前端）
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_env() -> Path | None:
    """加载 env 文件到 os.environ（不覆盖已存在的真实环境变量）。

    override=False：命令行显式 `CYBERHOME_LLM_PROVIDER=mimo python3 -m cyberhome` 优先于
    .env 文件值，符合「就近覆盖」直觉。dotenv 未安装时静默跳过（仍可靠真实环境变量启动）。
    """
    backend_env = Path(__file__).resolve().parent.parent / ".env"
    root_env = Path(__file__).resolve().parents[2] / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None
    for env_path in (backend_env, root_env):
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return env_path
    return None


def main() -> None:
    loaded = _load_env()

    # 加载后再 import：确保 load_llm_config() 读到的是已注入 .env 的 os.environ。
    from .config import load_llm_config
    from .llm import llm_runtime_status

    config = load_llm_config()
    host = (os.environ.get("CYBERHOME_HOST") or "127.0.0.1").strip()
    port = int((os.environ.get("CYBERHOME_PORT") or "8000").strip())
    persist = (os.environ.get("CYBERHOME_PERSIST") or "").strip().lower() in ("1", "true", "yes")

    # 脱敏启动横幅：只报 provider/model/是否有 key（布尔），绝不打印 key 本身（脱敏红线）。
    print("=" * 60, flush=True)
    print("CyberHome 后端启动", flush=True)
    print(f"  .env        : {loaded if loaded else '（未加载：无文件或未装 python-dotenv）'}", flush=True)
    print(f"  provider    : {config.provider}", flush=True)
    print(f"  model       : {config.model}", flush=True)
    print(f"  base_url    : {config.base_url or '（stub 无网络）'}", flush=True)
    print(
        f"  api_key     : {'已配置' if config.has_api_key else '缺省 → 降级 StubChatModel（非真实涌现）'}",
        flush=True,
    )
    print(f"  persist     : {'开（SQLite）' if persist else '关（内存，重启回 seed）'}", flush=True)
    print(f"  llm_runtime : {llm_runtime_status(config)}", flush=True)
    print(f"  listen      : http://{host}:{port}", flush=True)
    print("=" * 60, flush=True)

    import uvicorn

    # 传 app 实例而非 import 字符串：create_app() 此刻构造，chat_model 依当前（已注入 .env 的）
    # 环境经 build_chat_model 选定，与上面横幅所示 provider 一致。不开 reload（reload 需 import
    # 字符串且会另起子进程重走 import，横幅逻辑失真；demo 无需热重载）。
    from .api import create_app

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
