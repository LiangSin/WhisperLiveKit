#!/usr/bin/env python
"""Micro-benchmark for one SimulStreaming session's infer() hot path.

Loads the deployed model, streams a real WAV through insert_audio_chunk +
process_iter exactly like a live session, and reports:

  - per-round wall time (encoder + decode) and decoder forwards per round
  - GPU->CPU syncs per decode round (torch.cuda.set_sync_debug_mode)
  - _process_cross_attention cost vs. decode-step index (O(n) vs O(n^2))
  - a torch.profiler trace of one mid-stream round (sync/memcpy event counts)

Run on an idle GPU so production is untouched, e.g.:
  CUDA_VISIBLE_DEVICES=1 python benchmarking/microbench_infer.py \
      --audio dataset/PLCX-BLZ1hDpDOgZPSmdMcpgfO5uQ0i4XK/test1/-zD50IAAxfY.wav

Results are printed and appended as JSON to --json-out for cross-stage
comparison (Stage 0 baseline vs. later optimization stages).
"""
import argparse
import json
import statistics
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch

from whisperlivekit.simul_whisper import SimulStreamingASR
from whisperlivekit.simul_whisper.backend import SimulStreamingOnlineProcessor

REPO = Path(__file__).resolve().parent.parent

# ~200 tokens of realistic zh lecture context for the prefix pass.
DEFAULT_CONTEXT = (
    "今天我們要討論作業系統的記憶體管理機制，包括虛擬記憶體、分頁表、"
    "以及轉譯後備緩衝區的運作原理。上一堂課我們介紹了行程與執行緒的差別，"
    "行程擁有獨立的位址空間，而執行緒共享同一個位址空間。接下來請大家注意"
    "分頁錯誤的處理流程，當處理器存取的虛擬位址不在實體記憶體中時，"
    "會觸發缺頁中斷，作業系統必須從磁碟把對應的分頁載入。"
)


def build_session(args):
    asr = SimulStreamingASR(
        # transcription_common_params equivalents (deployed config)
        warmup_file=None,
        min_chunk_size=args.min_chunk_size,
        model_size=None,
        model_cache_dir=None,
        model_dir=None,
        model_path=str(REPO / args.model_path),
        lan="zh",
        direct_english_translation=False,
        # simulstreaming_params equivalents
        disable_fast_encoder=False,
        custom_alignment_heads=None,
        frame_threshold=25,
        beams=1,
        decoder_type=None,
        audio_max_len=20.0,
        audio_min_len=0.0,
        cif_ckpt_path=None,
        never_fire=False,
        init_prompt=args.context or None,
        static_init_prompt=None,
        max_context_tokens=None,
        backend="auto",
        whisper_device=None,
    )
    return SimulStreamingOnlineProcessor(asr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default="dataset/PLCX-BLZ1hDpDOgZPSmdMcpgfO5uQ0i4XK/test1/-zD50IAAxfY.wav")
    p.add_argument("--model-path", default="models/cool-whisper")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--chunk", type=float, default=1.0, help="seconds fed per process_iter round")
    p.add_argument("--min-chunk-size", type=float, default=1.0)
    p.add_argument("--context", default=DEFAULT_CONTEXT, help="init prompt text ('' to disable)")
    p.add_argument("--offset", type=float, default=60.0, help="start offset into the wav (skip intros)")
    p.add_argument("--json-out", default="benchmarking/results/microbench_infer.jsonl")
    p.add_argument("--label", default="", help="free-form label stored in the JSON record")
    p.add_argument("--profile-round", type=int, default=10, help="round index to torch-profile (-1 disables)")
    p.add_argument("--sync-debug", action="store_true", default=True)
    p.add_argument("--no-sync-debug", dest="sync_debug", action="store_false")
    args = p.parse_args()

    audio, _ = librosa.load(str(REPO / args.audio), sr=16000, mono=True,
                            offset=args.offset, duration=args.seconds)
    audio = audio.astype(np.float32)

    proc = build_session(args)
    alignatt = proc.model

    # --- instrumentation: count decoder forwards and time cross-attn ---
    forwards = {"n": 0}
    orig_logits = alignatt.logits

    def counted_logits(*a, **kw):
        forwards["n"] += 1
        return orig_logits(*a, **kw)

    alignatt.logits = counted_logits

    xattn_samples = []  # (step_index_within_round, seconds)
    orig_xattn = alignatt._process_cross_attention
    round_step = {"i": 0}

    def timed_xattn(*a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_xattn(*a, **kw)
        torch.cuda.synchronize()
        xattn_samples.append((round_step["i"], time.perf_counter() - t0))
        round_step["i"] += 1
        return out

    alignatt._process_cross_attention = timed_xattn

    chunk_samples = int(args.chunk * 16000)
    n_rounds = len(audio) // chunk_samples
    rounds = []
    total_syncs = 0

    for r in range(n_rounds):
        chunk = audio[r * chunk_samples:(r + 1) * chunk_samples]
        proc.insert_audio_chunk(chunk, (r + 1) * args.chunk)
        round_step["i"] = 0
        f0 = forwards["n"]

        profiling = (r == args.profile_round)
        sync_warns = 0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if profiling:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                tokens, _ = proc.process_iter()
        elif args.sync_debug:
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                torch.cuda.set_sync_debug_mode("warn")
                try:
                    tokens, _ = proc.process_iter()
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            sync_warns = len(wlist)
        else:
            tokens, _ = proc.process_iter()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        n_fwd = forwards["n"] - f0
        total_syncs += sync_warns
        rounds.append({
            "round": r, "wall_s": dt, "forwards": n_fwd,
            "tokens_out": len(tokens), "syncs": sync_warns,
        })
        print(f"round {r:3d}: {dt*1000:7.1f} ms  forwards={n_fwd:3d}  "
              f"tokens={len(tokens):3d}  syncs={sync_warns}", flush=True)
        if tokens:
            # Word/timestamp dump for cross-commit equivalence diffs.
            print("  words: " + " ".join(
                f"{t.text}@{t.start:.2f}-{t.end:.2f}" for t in tokens), flush=True)

        if profiling:
            evts = prof.key_averages()
            sync_evt = sum(e.count for e in evts if "Synchronize" in e.key)
            memcpy = sum(e.count for e in evts if "Memcpy DtoH" in e.key or "memcpy" in e.key.lower())
            print(f"\n[profiler] round {r}: cudaSynchronize-class events={sync_evt}, DtoH memcpy={memcpy}")
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12))

    # --- summary ---
    walls = [x["wall_s"] for x in rounds if x["forwards"] > 0]
    fwds = [x["forwards"] for x in rounds if x["forwards"] > 0]
    sync_rounds = [x for x in rounds if x["syncs"] > 0 and x["forwards"] > 0]
    syncs_per_token = (
        sum(x["syncs"] for x in sync_rounds) / max(sum(x["forwards"] for x in sync_rounds), 1)
    )
    # cross-attn scaling: mean cost of first step vs later steps
    by_step = {}
    for i, dt in xattn_samples:
        by_step.setdefault(i, []).append(dt)
    xattn_curve = {i: statistics.mean(v) * 1000 for i, v in sorted(by_step.items())}

    summary = {
        "label": args.label,
        "git": None,
        "rounds": len(walls),
        "wall_ms_p50": statistics.median(walls) * 1000 if walls else None,
        "wall_ms_max": max(walls) * 1000 if walls else None,
        "forwards_per_round_mean": statistics.mean(fwds) if fwds else None,
        "syncs_per_forward": round(syncs_per_token, 2),
        "xattn_ms_by_step": {k: round(v, 3) for k, v in xattn_curve.items()},
    }
    try:
        import subprocess
        summary["git"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        pass

    print("\n=== microbench summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    out = REPO / args.json_out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"\nappended to {out}")


if __name__ == "__main__":
    main()
