"""
monitor_client — periodically checks every Docker Compose service and pushes
the results to monitor_service via PUSH /monitor/update.

Check strategy per service
──────────────────────────
1. Docker container state  (always)   - is the container running?
2. Docker health status    (if set)   - has the container's own HEALTHCHECK failed?
3. HTTP probe              (optional) - does the service actually respond?

Final status mapping
────────────────────
Container not running                     → "down"
Container running, Docker health=unhealthy → "degraded"
Container running, HTTP probe fails        → "degraded"
Container running, everything OK           → "up"
"""

import json
import logging
import os
import socket
import time

import docker
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("monitor_client")

MONITOR_URL   = os.getenv("MONITOR_URL",   "http://monitor_service:8080/update")
MONITOR_KEY   = os.getenv("MONITOR_SECRET_KEY", "")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
PROBES_FILE   = os.getenv("PROBES_FILE",   "/app/probes.json")
SOURCE        = socket.gethostname()


# ── Helpers ────────────────────────────────────────────────────────────────

def load_probes() -> dict:
    try:
        with open(PROBES_FILE) as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Cannot load probes file %s: %s", PROBES_FILE, exc)
        return {}


def container_state(dc: docker.DockerClient, name: str) -> tuple[bool, str | None]:
    """
    Returns (is_running, docker_health_status).
    docker_health_status is one of: 'healthy', 'unhealthy', 'starting', or None
    (None means either no HEALTHCHECK defined, or container not found).
    """
    try:
        c = dc.containers.get(name)
        state = c.attrs.get("State", {})
        if not state.get("Running", False):
            return False, None
        health = state.get("Health")
        return True, health.get("Status") if health else None
    except docker.errors.NotFound:
        logger.debug("Container %r not found", name)
        return False, None
    except Exception as exc:
        logger.warning("Docker check error for %r: %s", name, exc)
        return False, None


def http_probe(cfg: dict) -> bool:
    """Returns True when the HTTP probe receives the expected status code."""
    try:
        resp = requests.get(
            cfg["url"],
            verify=cfg.get("verify_ssl", True),
            timeout=cfg.get("timeout", 5),
            allow_redirects=True,
        )
        ok = resp.status_code == cfg.get("expected_status", 200)
        if not ok:
            logger.debug("HTTP probe %s → HTTP %d (expected %d)",
                         cfg["url"], resp.status_code, cfg.get("expected_status", 200))
        return ok
    except Exception as exc:
        logger.debug("HTTP probe %s failed: %s", cfg.get("url", "?"), exc)
        return False


def resolve_status(running: bool, docker_health: str | None, probe_ok: bool | None) -> str:
    if not running:
        return "down"
    if docker_health == "unhealthy":
        return "degraded"
    if probe_ok is False:
        return "degraded"
    return "up"


def push(name: str, status: str) -> None:
    try:
        resp = requests.post(
            MONITOR_URL,
            json={"name": name, "source": SOURCE, "status": status},
            headers={
                "X-Monitor-Key": MONITOR_KEY,
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("%-20s → %s", name, status)
        else:
            logger.warning(
                "Push rejected for %r: HTTP %d %s",
                name, resp.status_code, resp.text.strip()[:120],
            )
    except Exception as exc:
        logger.error("Failed to push %r: %s", name, exc)


# ── Main cycle ─────────────────────────────────────────────────────────────

def run_cycle(dc: docker.DockerClient, probes: dict) -> None:
    for service_name, cfg in probes.items():
        if not isinstance(cfg, dict):
            continue  # skip metadata keys like "_comment"
        container_name = cfg.get("container", service_name)

        running, docker_health = container_state(dc, container_name)

        probe_ok = None
        if running and "http_probe" in cfg:
            probe_ok = http_probe(cfg["http_probe"])

        status = resolve_status(running, docker_health, probe_ok)
        push(service_name, status)


def main() -> None:
    logger.info(
        "monitor_client starting — target=%s  interval=%ds  source=%s",
        MONITOR_URL, POLL_INTERVAL, SOURCE,
    )

    try:
        dc = docker.from_env()
        dc.ping()
        logger.info("Connected to Docker daemon")
    except Exception as exc:
        logger.critical("Cannot connect to Docker socket: %s", exc)
        raise SystemExit(1)

    # Brief grace period so dependent services can finish starting up.
    logger.info("Waiting 10 s before first check…")
    time.sleep(10)

    while True:
        probes = load_probes()
        if probes:
            try:
                run_cycle(dc, probes)
            except Exception as exc:
                logger.error("Unexpected error in cycle: %s", exc)
        else:
            logger.warning("No probes loaded — skipping cycle")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
