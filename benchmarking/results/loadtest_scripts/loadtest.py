#!/usr/bin/env python
"""Load test client for WhisperLiveKit.

usage: loadtest.py N DURATION_S

Opens N concurrent websocket connections to wss://localhost:8000/asr, each
streaming realtime-paced PCM (as a WAV stream: header + 0.5 s chunks) from a
rotating set of lecture recordings, offset so no two connections send the
same audio position. Samples vLLM queue metrics every 10 s.
"""
import asyncio, json, ssl, struct, sys, time, urllib.request
from pathlib import Path

import websockets

import os
URLS = os.environ.get(
    "LOADTEST_URLS",
    "wss://localhost:8000/asr,wss://localhost:8001/asr,wss://localhost:8002/asr",
).split(",")
METRICS_URL = "https://localhost:8765/metrics"
SR = 16000
CHUNK_S = 0.5
CHUNK_BYTES = int(SR * CHUNK_S) * 2
STAGGER_S = 0.4
OFFSET_S = 61  # per-connection start offset into the audio

PCM_DIR = Path(__file__).parent / "pcm"
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def wav_header():
    # 16 kHz mono s16le, bogus (huge) data size for streaming
    byte_rate = SR * 2
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFF0) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, SR, byte_rate, 2, 16) +
            b"data" + struct.pack("<I", 0xFFFFFFF0))


class ConnStats:
    def __init__(self, idx):
        self.idx = idx
        self.lines = 0
        self.messages = 0
        self.error = None
        self.max_send_late = 0.0


async def run_conn(idx, pcm_files, duration_s, stats):
    pcm = pcm_files[idx % len(pcm_files)].read_bytes()
    off = (idx * OFFSET_S * SR * 2) % max(len(pcm) - CHUNK_BYTES, 1)
    off -= off % 2
    try:
        async with websockets.connect(URLS[idx % len(URLS)], ssl=SSL_CTX, max_size=None) as ws:
            async def reader():
                async for msg in ws:
                    stats.messages += 1
                    try:
                        d = json.loads(msg)
                        stats.lines = max(stats.lines, len(d.get("lines") or []))
                    except Exception:
                        pass
            rtask = asyncio.create_task(reader())
            await ws.send(wav_header())
            t0 = time.monotonic()
            n_chunks = int(duration_s / CHUNK_S)
            for i in range(n_chunks):
                target = t0 + i * CHUNK_S
                late = time.monotonic() - target
                if late > stats.max_send_late:
                    stats.max_send_late = late
                if late < 0:
                    await asyncio.sleep(-late)
                start = off + i * CHUNK_BYTES
                chunk = pcm[start:start + CHUNK_BYTES]
                if len(chunk) < CHUNK_BYTES:  # wrap around
                    chunk = (chunk + pcm)[:CHUNK_BYTES]
                await ws.send(chunk)
            await asyncio.sleep(2)
            rtask.cancel()
    except Exception as e:
        stats.error = f"{type(e).__name__}: {e}"


def fetch_metrics():
    try:
        with urllib.request.urlopen(METRICS_URL, context=SSL_CTX, timeout=5) as r:
            text = r.read().decode()
        running = waiting = None
        for line in text.splitlines():
            if line.startswith("vllm:num_requests_running"):
                running = float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:num_requests_waiting"):
                waiting = float(line.rsplit(" ", 1)[1])
        return running, waiting
    except Exception:
        return None, None


async def sampler(samples, stop):
    while not stop.is_set():
        r, w = await asyncio.to_thread(fetch_metrics)
        if r is not None:
            samples.append((time.monotonic(), r, w))
            print(f"[metrics] vllm running={r:.0f} waiting={w:.0f}", flush=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=10)
        except asyncio.TimeoutError:
            pass


async def main():
    n = int(sys.argv[1])
    duration_s = float(sys.argv[2])
    pcm_files = sorted(PCM_DIR.glob("*.pcm"))
    assert pcm_files, "no pcm files"
    all_stats = [ConnStats(i) for i in range(n)]
    samples = []
    stop = asyncio.Event()
    stask = asyncio.create_task(sampler(samples, stop))

    async def delayed(i):
        await asyncio.sleep(i * STAGGER_S)
        await run_conn(i, pcm_files, duration_s, all_stats[i])

    await asyncio.gather(*(delayed(i) for i in range(n)))
    stop.set()
    await stask

    errs = [s for s in all_stats if s.error]
    no_lines = [s for s in all_stats if not s.error and s.lines == 0]
    max_late = max(s.max_send_late for s in all_stats)
    print("\n=== loadtest summary ===")
    print(f"connections: {n}, duration: {duration_s:.0f}s")
    print(f"conn errors: {len(errs)}" + (f" -> {[(s.idx, s.error) for s in errs]}" if errs else ""))
    print(f"conns with zero lines: {len(no_lines)}")
    print(f"max client send lateness: {max_late:.2f}s")
    if samples:
        waits = [w for _, _, w in samples]
        tail = waits[max(0, len(waits) - 6):]
        print(f"vllm waiting: max={max(waits):.0f} tail-avg={sum(tail)/len(tail):.1f} last={waits[-1]:.0f}")
        runs = [r for _, r, _ in samples]
        print(f"vllm running: max={max(runs):.0f} last={runs[-1]:.0f}")
    else:
        print("vllm metrics: NO SAMPLES")


asyncio.run(main())
