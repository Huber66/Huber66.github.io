#!/usr/bin/env python3
"""One-command local dev server for CyberHome.

Starts the FastAPI backend and Vite frontend together, clears stale listeners
on the configured ports, and shuts down the whole child process tree on exit.
"""

from __future__ import annotations

import argparse
import atexit
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = 8081
DEFAULT_FRONTEND_PORT = 3051

SHUTDOWN_TIMEOUT_SECONDS = 5.0
PORT_READY_TIMEOUT_SECONDS = 45.0
DOTENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class ChildProcess:
    label: str
    process: subprocess.Popen[bytes]
    port: int
    pgid: int


class DevServer:
    def __init__(self, host: str, backend_port: int, frontend_port: int) -> None:
        self.host = host
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        self.children: list[ChildProcess] = []
        self._cleaned = False

    def start(self) -> None:
        ensure_prerequisites()

        print("CyberHome dev server", flush=True)
        print(f"  backend : http://{self.host}:{self.backend_port}", flush=True)
        print(f"  frontend: http://{self.host}:{self.frontend_port}", flush=True)
        print("  stop    : Ctrl+C", flush=True)
        print("", flush=True)

        kill_port_processes(self.backend_port, "backend")
        kill_port_processes(self.frontend_port, "frontend")

        base_env = build_child_env()
        backend_env = {
            **base_env,
            "CYBERHOME_HOST": self.host,
            "CYBERHOME_PORT": str(self.backend_port),
        }
        frontend_env = {
            **base_env,
            "VITE_API_BASE": f"http://{self.host}:{self.backend_port}",
        }

        self.children.append(
            spawn_child(
                "backend",
                backend_command(),
                cwd=BACKEND_DIR,
                env=backend_env,
                port=self.backend_port,
            )
        )
        self.children.append(
            spawn_child(
                "frontend",
                frontend_command(self.host, self.frontend_port),
                cwd=FRONTEND_DIR,
                env=frontend_env,
                port=self.frontend_port,
            )
        )

        wait_for_port(self.host, self.backend_port, self.children[0], "backend")
        wait_for_port(self.host, self.frontend_port, self.children[1], "frontend")

    def run_forever(self) -> int:
        try:
            while True:
                for child in self.children:
                    code = child.process.poll()
                    if code is not None:
                        print(f"\n{child.label} exited with code {code}; stopping dev server.", flush=True)
                        self.cleanup()
                        return code if code != 0 else 0
                time.sleep(0.5)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True

        if self.children:
            print("\nStopping CyberHome dev server...", flush=True)

        for child in reversed(self.children):
            terminate_child_group(child)

        child_pgids = {child.pgid for child in self.children}
        kill_port_processes(self.backend_port, "backend", allowed_pgids=child_pgids)
        kill_port_processes(self.frontend_port, "frontend", allowed_pgids=child_pgids)


def ensure_prerequisites() -> None:
    if not BACKEND_DIR.is_dir():
        raise SystemExit(f"Missing backend directory: {BACKEND_DIR}")
    if not FRONTEND_DIR.is_dir():
        raise SystemExit(f"Missing frontend directory: {FRONTEND_DIR}")
    if shutil.which("npm") is None:
        raise SystemExit("Missing command: npm")


def backend_command() -> list[str]:
    if shutil.which("uv") is not None:
        # Use "python" not "python3" — on Windows, "python3" resolves to the
        # system Python via the launcher, bypassing uv's managed environment.
        return ["uv", "run", "--with-requirements", "requirements.txt", "python", "-m", "cyberhome"]
    return [sys.executable, "-m", "cyberhome"]


def frontend_command(host: str, port: int) -> list[str]:
    return ["npm", "run", "dev", "--", "--host", host, "--port", str(port), "--strictPort"]


def spawn_child(
    label: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    port: int,
) -> ChildProcess:
    print(f"Starting {label}: {' '.join(shlex.quote(part) for part in command)}", flush=True)
    # Resolve full executable path — Windows needs .cmd/.bat PATHEXT resolution
    # that CreateProcess (used by Popen with shell=False) does not perform.
    resolved = shutil.which(command[0], path=env.get("PATH"))
    if resolved:
        command = [resolved] + list(command[1:])
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
    return ChildProcess(label=label, process=process, port=port, pgid=process.pid)


def wait_for_port(host: str, port: int, child: ChildProcess, label: str) -> None:
    deadline = time.monotonic() + PORT_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        code = child.process.poll()
        if code is not None:
            print(f"{label} exited before port {port} became ready (code {code}).", flush=True)
            return
        if can_connect(host, port):
            print(f"{label} ready on http://{host}:{port}", flush=True)
            return
        time.sleep(0.25)
    print(f"{label} is still starting; continuing to stream logs.", flush=True)


def can_connect(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def build_child_env() -> dict[str, str]:
    env = os.environ.copy()
    locked_keys = set(env)

    # Precedence: real shell env > backend/.env > root .env > built-in defaults.
    load_env_file(ROOT / ".env", env, locked_keys)
    load_env_file(BACKEND_DIR / ".env", env, locked_keys)
    return env


def load_env_file(path: Path, env: dict[str, str], locked_keys: set[str]) -> None:
    if not path.exists():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"Warning: cannot read {path}: {exc}", flush=True)
        return

    for raw in lines:
        parsed = parse_dotenv_line(raw)
        if parsed is None:
            continue
        key, value = parsed
        if key in locked_keys:
            continue
        env[key] = value


def parse_dotenv_line(raw: str) -> tuple[str, str] | None:
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    key, sep, value = line.partition("=")
    if not sep:
        return None
    key = key.strip()
    if not DOTENV_KEY.match(key):
        return None
    value = strip_inline_comment(value.strip())
    return key, unquote_env_value(value)


def strip_inline_comment(value: str) -> str:
    if not value or value[0] in ("'", '"'):
        return value
    parts = shlex.split(value, comments=True, posix=True)
    return parts[0] if parts else ""


def unquote_env_value(value: str) -> str:
    if not value:
        return ""
    try:
        parts = shlex.split(value, comments=False, posix=True)
    except ValueError:
        return value.strip().strip("'\"")
    if not parts:
        return ""
    return parts[0]


def port_pids(port: int) -> list[int]:
    """Find PIDs listening on the given TCP port (cross-platform).

    Tries lsof first (macOS/Linux); falls back to netstat (Windows/Linux).
    """
    pids: set[int] = set()

    # Try lsof first (macOS/Linux)
    if shutil.which("lsof") is not None:
        try:
            result = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode in (0, 1):
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pids.add(int(line))
                    except ValueError:
                        pass
        except FileNotFoundError:
            pass

    # Fallback: netstat (Windows/Linux) — parse "TCP  host:port ... LISTENING  PID"
    if not pids and shutil.which("netstat") is not None:
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            for line in result.stdout.splitlines():
                if f":{port}" not in line or "LISTEN" not in line.upper():
                    continue
                parts = line.split()
                if parts:
                    try:
                        pids.add(int(parts[-1]))
                    except ValueError:
                        pass
        except FileNotFoundError:
            pass

    if not pids:
        return []
    return sorted(pid for pid in pids if pid != os.getpid())


def kill_port_processes(port: int, label: str, allowed_pgids: set[int] | None = None) -> None:
    pids = port_pids(port)
    if allowed_pgids is not None:
        pids = [pid for pid in pids if pid_pgid(pid) in allowed_pgids]
    if not pids:
        return

    print(f"Stopping stale {label} listener(s) on port {port}: {pids}", flush=True)
    terminate_pids(pids)


def terminate_pids(pids: Iterable[int]) -> None:
    unique_pids = sorted(set(pids))
    for pid in unique_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f"Warning: cannot terminate pid {pid}: {exc}", flush=True)

    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        alive = [pid for pid in unique_pids if pid_alive(pid)]
        if not alive:
            return
        time.sleep(0.2)

    for pid in unique_pids:
        if not pid_alive(pid):
            continue
        try:
            kill_signal = signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM
            os.kill(pid, kill_signal)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f"Warning: cannot kill pid {pid}: {exc}", flush=True)


def terminate_child_group(child: ChildProcess) -> None:
    process = child.process
    if process.poll() is not None:
        return

    print(f"Stopping {child.label}...", flush=True)
    if hasattr(os, "killpg"):
        try:
            os.killpg(child.pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            print(f"Warning: cannot terminate {child.label} process group: {exc}", flush=True)
            process.terminate()
    else:
        process.terminate()

    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass

    print(f"{child.label} did not stop in time; killing it.", flush=True)
    if hasattr(os, "killpg"):
        try:
            os.killpg(child.pgid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            print(f"Warning: cannot kill {child.label} process group: {exc}", flush=True)
            process.kill()
    else:
        process.kill()
    process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_pgid(pid: int) -> int | None:
    if not hasattr(os, "getpgid"):
        return None
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return None
    except PermissionError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start CyberHome backend and frontend together.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"listen host, default: {DEFAULT_HOST}")
    parser.add_argument(
        "--backend-port",
        type=int,
        default=DEFAULT_BACKEND_PORT,
        help=f"backend port, default: {DEFAULT_BACKEND_PORT}",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=DEFAULT_FRONTEND_PORT,
        help=f"frontend port, default: {DEFAULT_FRONTEND_PORT}",
    )
    return parser.parse_args()


def install_signal_handlers(server: DevServer) -> None:
    def handle(signum: int, _frame: object) -> None:
        server.cleanup()
        raise SystemExit(128 + signum)

    signums = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signums.append(signal.SIGHUP)
    for signum in signums:
        signal.signal(signum, handle)


def main() -> int:
    args = parse_args()
    server = DevServer(args.host, args.backend_port, args.frontend_port)
    install_signal_handlers(server)
    atexit.register(server.cleanup)
    server.start()
    return server.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
