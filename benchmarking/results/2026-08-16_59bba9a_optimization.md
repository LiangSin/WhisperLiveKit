# 2026-08-16 — CPU/GIL optimization round (commits d899f58…59bba9a)

Goal (set after the 2026-08-15 load test): serve **15 concurrent
connections on one GPU with only 2 WLK processes** (previously needed 3).
The load test attributed the per-process ceiling to Python/GIL overhead
and per-token GPU→CPU syncs, not GPU compute — this round removes those
costs from the code.

## Changes shipped (one commit per stage)

| Commit | Stage | Change |
|---|---|---|
| 40defa1 | 1 | Greedy decoder actually used when beams=1 (was hard-coded BeamSearchDecoder); microbench harness `benchmarking/microbench_infer.py` |
| f508513 | 1 | Decode-loop de-sync: debug syncs gated on log level, single `.tolist()` for attended frames, prebuilt -inf suppress masks, `inference_mode`, TokenBuffer id/tensor caching, `need_logprobs` fast path. **17.7 → 2.9 GPU syncs per forward** |
| 51f3a28 | 2 | `_process_cross_attention` O(steps²) → incremental running-sum (O(steps)); only alignment heads' QK materialised. Verified word+timestamp-equivalent over a 120 s / 95-round fixture |
| d899f58 | 6 | Event loop unblocked: Silero VAD via `to_thread`, results formatter content-fingerprint gating, `end_silence`/`new_speaker` off-loop, SaT lock |
| 06d6812 | 5 | `--ct2-encoder-workers` (default 2): per-session encodes no longer serialize inside CT2 |
| 59bba9a | fix | WER-regression fixes: revert SaT single-forward rewrite (keep lock), restore original VAD chunk sizing (VAD events clamp to chunk boundaries — chunking is semantics), fp64 alignment running sums, force fp32-softmax path for cross-attention |

Also (deployment, not in this repo's history): vLLM `--enforce-eager`
removed (CUDA graphs back on; the old error did not reproduce) and
TranslateGemma requests wall-clock tick-aligned across all WLK processes
so vLLM sees one coalesced batch per interval (vLLM SM 22→12%, CPU
84→22% at the 2-conn baseline).

## Quality gates (final build = 59bba9a)

WER, youtube-debug preset, `--accumulate-mode lines`:

| Run | Baseline (da7cf1a) | Final build |
|---|---|---|
| max-speed ×3 | 0.0608–0.0612 | **0.0608 / 0.0592 / 0.0621** |
| realtime | 0.0625 (n=1) | **0.0604 / 0.0737** |

Translation, translate-debug preset, 12b q8 TranslateGemma (reference:
2026-08-13 quant sweep row chrF++ 56.11 / COMET 0.7270, single run,
`--enforce-eager` still on then):

| Run | chrF++ | COMET | ASR err |
|---|---|---|---|
| final build #1 | 53.26 | 0.7281 | 0.0721 |
| final build #2 | 51.92 | 0.7215 | 0.0721 |

COMET straddles the reference (Δ within the ±0.0066 same-build
run-to-run band) — semantic adequacy preserved. chrF++ reads 3–4 points
below the 8/13 reference, but is not cleanly comparable: all 73 source
windows differ from the reference run (the ASR text itself changed —
and improved, err 0.072 vs 0.079), same-build chrF spread is ±1.3, the
8/13 sweep's own chrF column moves ±4 between quant settings of the
same model (noise), and TG serving changed in between (enforce-eager
removed, tick-aligned batching). chrF measures character overlap with a
fixed reference wording, so equally-good-but-differently-worded ASR →
translation shifts it without a quality change; COMET is the gate that
controls for this, and it passes.

At/within baseline noise. The intermediate stage-6 commit (d899f58) had a
real WER regression (max-speed 0.069, realtime 0.093) from two causes,
both fixed in 59bba9a: the SaT single-forward rewrite changed split
behaviour, and 1 s PCM slicing moved VAD-event clamp boundaries.
Deploy d899f58…06d6812 only together with 59bba9a.

## Acceptance: 15 connections on 2×WLK + 1×TG, one GPU (H200)

`trial3.sh 15 600` — 8 conns → wlk1, 7 → wlk2, 600 s, real lecture audio:

- **catch-up triggers: 0 + 0** (pass criterion); conn errors 0; every
  connection produced output; client send lateness 0.00 s
- max-lag readings (10.49 s / 6.51 s) are the known silence-inflation
  artifact (occurs even at N=2), not backlog
- vLLM queue: waiting=0 throughout; tick-aligned bursts of 10–17
  requests handled in one continuous batch
- GPU0 util avg 71% (max 100%); SM% avg — wlk1 32, wlk2 15, vllm 24
- CPU — wlk1 avg 2.2 cores (max 3.9), wlk2 avg 1.3 (max 3.6): well below
  the pre-optimization 3–4-core average with spikes to 7–8 *while failing
  at similar per-process load*
- Peak VRAM: wlk1 9.2 GiB + wlk2 10.3 GiB + vllm 26.6 GiB ≈ 46 GiB

**Result: 15 connections per GPU with 2 WLK processes — target met.**
The third WLK container is retired. Headroom (CPU avg ~2 cores/process,
GPU 71%) suggests ~8–9 per process remains comfortable; the previous
3-process topology is no longer needed for 15.

## Post-optimization capacity ceilings (measured 2026-08-16 evening)

Ladders on the final build, 600 s per step, criterion 0 catch-up
triggers + 0 conn errors; top passes confirmed with a repeat run
(single passes near the knee are unreliable — see load-test report).

| Topology (all on GPU 0) | Pre-opt | Post-opt | Fails at | GPU avg at ceiling | Peak VRAM |
|---|---|---|---|---|---|
| 1×WLK + 1×TG | 8 | **12** (×2 runs) | 13 (3 steady-state triggers, t=227/272 s) | 55% | 10.2 + 26.6 ≈ 37 GiB |
| 2×WLK + 1×TG | 13 | **17** (×2 runs) | 18 (2+2 triggers) | 78–81% | 10.5 + 8.9 + 26.6 ≈ 46 GiB |

Per-process ceiling rose from ~7 (2-WLK share) / 8 (solo) to ~8.5–12:
the solo process now carries 12 alone. At the 2×WLK ceiling the GPU
timeline itself is filling (avg 81%, max 100%) — further per-process
CPU wins would buy little.

### Control: 3×WLK + 1×TG (measured 2026-08-18) — why we stop at 2

To confirm the ceiling is the GPU timeline and not process count or
VRAM, the same ladder was run with a third identical WLK (6+6+6 split):

| N | 3×WLK result | GPU avg | Peak VRAM (of 141 GiB) |
|---|---|---|---|
| 17 | pass, 0+0+0 triggers | 79% | 51.6 GiB |
| 18 | **fail, 1+2+1** (steady-state, t=27–162 s) | 78% (max 100%) | 57.5 GiB |

Identical ceiling to 2×WLK (17 pass / 18 fail). The third process costs
~10 GiB VRAM and buys zero connections: per-process CPU is already
comfortable (~1.6 cores each — the GIL is no longer the binding
constraint post-optimization), so a third Python process relieves
nothing, while the shared GPU timeline saturates in bursts (max 100%)
regardless of how the sessions are spread. The ~84 GiB of free VRAM is
therefore real but unusable for scaling this workload — capacity 18+
needs GPU-side work (Stage 3/4: fp16 decoder, CUDA graphs) or a second
GPU, not more WLK processes. Production stays at 2×WLK.

## Diagnosis of the current ceiling, and the next optimization

At the 17-connection ceiling the GPU reports ~80% avg / 100% burst
utilization while doing a small fraction of its arithmetic capacity —
nvidia-smi "utilization" counts wall-clock time with *any* kernel
resident, not compute saturation. The timeline is filled by launch
overhead and tiny kernels, not FLOPs: each decode forward issues ~450
kernel launches of µs-scale batch-1 matrix-vector ops (measured in the
microbench), so 17 sessions × ~8-10 forwards/s ≈ 50-70k launches/s
across four CUDA contexts (2-3 WLK + vLLM), each context time-sliced by
the scheduler. Failure at N=18 is wall-clock queueing during bursts,
not arithmetic exhaustion — which is also why free VRAM and extra
processes buy nothing.

**Future optimization — batch decode across connections inside one
WLK**: turn N sessions' concurrent decode steps into one batched
forward (450 kernels at batch=N instead of N×450 at batch=1; a batch-8
matrix-vector op costs nearly the same wall-clock as batch-1). This
attacks the launch-bound timeline directly and is the main lever for
18+. Prerequisites are substantial (why it was deferred): ragged
per-session KV caches + key-padding masks, divergent per-session
control flow (AlignAtt policy, rewinds, per-session prompts), and a
scheduler aligning asynchronous session rounds — essentially
continuous batching for Whisper decode. Cheaper steps that attack the
same overhead first: CUDA graphs / torch.compile on the decode step
(Stage 3), cross-session *encoder* batching (uniform [1,n_mels,3000]
shapes, no ragged problem — Stage 5 item 2), and NVIDIA MPS to remove
context time-slicing between the WLK processes and vLLM.

## Follow-up experiment: 27b Q8 TranslateGemma (measured 2026-08-19 night)

Question: can the translation model be upgraded 12b→27b (online FP8,
same `--quantization fp8`) under the 2×WLK + 1×TG topology? vLLM
`gpu-memory-utilization` raised 0.2→0.35 (27b weights don't fit in 28
GiB); everything else identical to the ceiling runs above.

| N | 27b result | GPU avg | vLLM SM avg | Peak VRAM |
|---|---|---|---|---|
| 15 | **fail, 7+0 triggers** (steady-state, t≈220–590 s, all on wlk1/8 conns) | 85% | 33% | 65.0 GiB |
| 14 | pass 0+0, ×2 clean runs | 79% / 84% | 32% | ~67 GiB |

**Ceiling with 27b = 14 (vs 17 with 12b); the production target of 15
does not fit.** Mechanism matches the timeline diagnosis above: vLLM's
SM share rises 24%→33% (≈2.2× FLOPs/token), its tick-aligned bursts
occupy the GPU timeline longer, and the WLK carrying 8 connections
falls behind during bursts. Translation throughput itself is not the
limit — vLLM `waiting=0` throughout, bursts of 13–16 requests handled
in one batch. VRAM is also not the limit (peak ~67 of 141 GiB).

Notes: the first N=14 attempt produced one contaminated pass and one
contaminated fail — another user's GPU job (avg 14% SM, max 60%)
started mid-experiment; both clean reruns pass. The N=15 fail predates
the foreign job and is clean. Config was reverted to 12b after the
experiment. If 27b quality is ever compelling enough to revisit:
per-GPU capacity must be planned at 14, or the TG instance moved to a
different GPU than the WLKs (the launch-bound WLK ceiling of 17 would
likely mostly return); judge 27b-vs-12b quality by COMET, not chrF++.

## Not pursued (goal met without them)

- Stage 3 (fp16 decoder, `.to()` churn removal, torch.compile/CUDA
  graphs) and Stage 4 (experimental KV retention) from the plan remain
  unimplemented — measure-gated options if a higher ceiling is ever
  needed.
- Cross-session batched decode and in-process worker pools: explicitly
  out of scope (weeks-scale rewrite).
