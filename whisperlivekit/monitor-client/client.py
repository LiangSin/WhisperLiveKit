"""
In-container monitor client for WhisperLiveKit.

The old monitor container inspected Docker Compose state. This version runs
inside the WhisperLiveKit container and reports the same upstream payload based
on the local server process plus an optional HTTP probe.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import ssl
import time
from threading import Event
from http.client import HTTPResponse
from typing import Optional
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
SERVER_PID = int(os.getenv("MONITOR_TARGET_PID", "0") or "0")
HEALTH_URL = os.getenv("MONITOR_HEALTH_URL", "")
HEALTH_EXPECTED_STATUS = int(os.getenv("MONITOR_HEALTH_EXPECTED_STATUS", "200"))
HEALTH_TIMEOUT = float(os.getenv("MONITOR_HEALTH_TIMEOUT", "5"))
HEALTH_VERIFY_SSL = os.getenv("MONITOR_HEALTH_VERIFY_SSL", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)


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


def _ssl_context() -> Optional[ssl.SSLContext]:
    if HEALTH_VERIFY_SSL:
        return None
    return ssl._create_unverified_context()


def http_probe() -> Optional[bool]:
    """Return True/False for configured HTTP probe, or None when disabled."""
    if not HEALTH_URL:
        return None

    try:
        response: HTTPResponse
        with urlopen(HEALTH_URL, timeout=HEALTH_TIMEOUT, context=_ssl_context()) as response:
            if response.status == HEALTH_EXPECTED_STATUS:
                return True
            logger.debug(
                "HTTP probe %s returned HTTP %d, expected %d",
                HEALTH_URL,
                response.status,
                HEALTH_EXPECTED_STATUS,
            )
            return False
    except HTTPError as exc:
        logger.debug("HTTP probe %s returned HTTP %d", HEALTH_URL, exc.code)
        return False
    except (OSError, URLError) as exc:
        logger.debug("HTTP probe %s failed: %s", HEALTH_URL, exc)
        return False


def resolve_status(running: bool, probe_ok: Optional[bool]) -> str:
    if not running:
        return "down"
    if probe_ok is False:
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


def run_cycle() -> None:
    running = process_running(SERVER_PID)
    probe_ok = http_probe() if running else None
    push(resolve_status(running, probe_ok))


def main() -> None:
    logger.info(
        "monitor_client starting - target=%s interval=%ds source=%s service=%s",
        MONITOR_URL,
        POLL_INTERVAL,
        SOURCE,
        SERVICE_NAME,
    )
    if HEALTH_URL:
        logger.info("HTTP probe enabled - url=%s", HEALTH_URL)

    logger.info("Waiting %d s before first check", STARTUP_GRACE_SECONDS)
    time.sleep(STARTUP_GRACE_SECONDS)

    stop_event = Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while not stop_event.is_set():
        try:
            run_cycle()
        except Exception as exc:
            logger.error("Unexpected error in cycle: %s", exc)
        stop_event.wait(POLL_INTERVAL)


if __name__ == "__main__":
    main()
