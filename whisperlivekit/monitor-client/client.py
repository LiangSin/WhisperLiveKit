"""
In-container monitor client for WhisperLiveKit.

This monitor client runs inside the WhisperLiveKit container and reports 
the upstream payload based on the local server process and the /health endpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import ssl
import time
from threading import Event
from http.client import HTTPResponse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("wlk_monitor_client")

MONITOR_URL = os.getenv("MONITOR_URL") or "http://monitor_service:8080/update"
MONITOR_KEY = os.getenv("MONITOR_SECRET_KEY", "")
POLL_INTERVAL = 30
STARTUP_GRACE_SECONDS = 10

SERVICE_NAME = os.getenv("MONITOR_SERVICE_NAME", "WhisperLiveKit")
SOURCE = socket.gethostname()
HEALTH_TIMEOUT = 5.0
HEALTH_EXPECTED_STATUS = 200


def process_running(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedded WhisperLiveKit monitor client")
    parser.add_argument("--server-pid", type=int, default=0)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-scheme", choices=["http", "https"], default="http")
    return parser.parse_args()


def health_url(server_scheme: str, server_port: int) -> str:
    return f"{server_scheme}://127.0.0.1:{server_port}/health"


def _ssl_context(server_scheme: str) -> ssl.SSLContext | None:
    if server_scheme != "https":
        return None
    return ssl._create_unverified_context()


def http_probe(url: str, server_scheme: str) -> bool:
    """Return True when the fixed local /health endpoint responds correctly."""
    try:
        response: HTTPResponse
        with urlopen(url, timeout=HEALTH_TIMEOUT, context=_ssl_context(server_scheme)) as response:
            if response.status == HEALTH_EXPECTED_STATUS:
                return True
            logger.debug(
                "HTTP probe %s returned HTTP %d, expected %d",
                url,
                response.status,
                HEALTH_EXPECTED_STATUS,
            )
            return False
    except HTTPError as exc:
        logger.debug("HTTP probe %s returned HTTP %d", url, exc.code)
        return False
    except (OSError, URLError) as exc:
        logger.debug("HTTP probe %s failed: %s", url, exc)
        return False


def resolve_status(running: bool, probe_ok: bool) -> str:
    if not running:
        return "down"
    if not probe_ok:
        return "degraded"
    return "up"


def push(status: str) -> None:
    payload = json.dumps(
        {"name": SERVICE_NAME, "source": SOURCE, "status": status}
    ).encode("utf-8")
    request = Request(
        MONITOR_URL,
        data=payload,
        method="POST",
        headers={
            "X-Monitor-Key": MONITOR_KEY,
            "Content-Type": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=10) as response:
            body = response.read(120).decode("utf-8", errors="replace")
            if response.status == 200:
                logger.info("%-20s -> %s", SERVICE_NAME, status)
            else:
                logger.warning(
                    "Push rejected for %r: HTTP %d %s",
                    SERVICE_NAME,
                    response.status,
                    body.strip(),
                )
    except Exception as exc:
        logger.error("Failed to push %r: %s", SERVICE_NAME, exc)


def run_cycle(server_pid: int, url: str, server_scheme: str) -> None:
    running = process_running(server_pid)
    probe_ok = http_probe(url, server_scheme) if running else False
    push(resolve_status(running, probe_ok))


def main() -> None:
    args = parse_args()
    url = health_url(args.server_scheme, args.server_port)
    logger.info(
        "monitor_client starting - target=%s interval=%ds source=%s service=%s probe=%s",
        MONITOR_URL,
        POLL_INTERVAL,
        SOURCE,
        SERVICE_NAME,
        url,
    )

    logger.info("Waiting %d s before first check", STARTUP_GRACE_SECONDS)
    time.sleep(STARTUP_GRACE_SECONDS)

    stop_event = Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while not stop_event.is_set():
        try:
            run_cycle(args.server_pid, url, args.server_scheme)
        except Exception as exc:
            logger.error("Unexpected error in cycle: %s", exc)
        stop_event.wait(POLL_INTERVAL)


if __name__ == "__main__":
    main()
