"""
Container entrypoint that runs vLLM TranslateGemma and the embedded monitor.
"""

from __future__ import annotations

import os
import signal
import shlex
import subprocess
import sys
from pathlib import Path


SERVER_COMMAND = ["vllm", "serve"]
MONITOR_CLIENT = Path(__file__).with_name("client.py")


def _arg_value(args: list[str], flag: str, default: str | None = None) -> str | None:
    prefix = f"{flag}="
    for index, value in enumerate(args):
        if value == flag and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(prefix):
            return value[len(prefix):]
    return default


def _has_arg(args: list[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(value == flag or value.startswith(prefix) for value in args)


def _terminate(process: subprocess.Popen[bytes], timeout: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    server_args = sys.argv[1:]
    if len(server_args) == 1:
        server_args = shlex.split(server_args[0])
    server = subprocess.Popen([*SERVER_COMMAND, *server_args])
    monitor = None

    monitor_enabled = os.getenv("MONITOR_ENABLED", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if monitor_enabled:
        port = _arg_value(server_args, "--port", "8000")
        scheme = "https" if _has_arg(server_args, "--ssl-certfile") else "http"
        monitor = subprocess.Popen(
            [
                sys.executable,
                str(MONITOR_CLIENT),
                "--server-pid",
                str(server.pid),
                "--server-port",
                str(port),
                "--server-scheme",
                scheme,
            ],
        )

    stopping = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        for process in (server, monitor):
            if process and process.poll() is None:
                process.send_signal(signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        return_code = server.wait()
    finally:
        if monitor:
            _terminate(monitor)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
