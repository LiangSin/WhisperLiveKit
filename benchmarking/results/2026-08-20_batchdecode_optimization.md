# 2026-08-20 — Cross-session batched decode + CUDA-graph step

Goal (follow-up to the 2026-08-16 round): the measured ceiling of 17
connections (2×WLK + 1×TG, one H200) was attributed to the GPU timeline
filling with batch-1 decode work — many tiny kernel launches and
repeated weight reads. This round merges concurrent sessions' decode
steps into one batched forward ("batch up requests across connections
within one WLK") and eliminates the per-step CPU dispatch cost with a
CUDA graph.

## What shipped (working tree, not yet committed)

New module `whisperlivekit/simul_whisper/batch_executor.py` plus small
hooks in `simul_whisper.py` (routing in `logits()`), `decoder_state.py`
(slot lifecycle), `backend.py` (executor creation), `parse_args.py` /
`core.py` (`--no-batch-decode`, `--batch-decode-slots`, default ON,
greedy+CUDA only).

Design (key discoveries that shaped it):

- **cool-whisper's decoder has only 2 layers** (distil architecture;
  encoder 32 layers lives in CT2). A batch-1 decode step is therefore
  ~1.3 ms of *CPU dispatch*, not GPU compute — the launch/dispatch
  overhead is the whole game.
- KV caches are round-scoped (cleared after every `infer()` round), so
  slot residency is per-round: the round-opening prefill runs
  concurrently in the session's own thread on the legacy dict-cache path
  (numerically identical), then its K/V is adopted into a slot buffer
  laid out `(layer, slot, head, pos, head_dim)` — per-layer slot views
  are regular strided batches, so the step path never gathers KV.
- The single-token step always runs **fixed-shape full width** (all 16
  slots, KV length fixed at 448, GPU-resident length vector + additive
  padding masks; inactive rows compute discarded garbage) and is
  **captured into a CUDA graph** at startup: per step the worker fills
  three small input buffers and replays the graph. Keys are stored
  pre-scaled so numerics match `(q*scale)@(k*scale)^T` exactly.
- Requests batch continuously with no gather window: whatever arrives
  while a step is on the GPU forms the next batch. Ticks across
  connections never need to align.
- Fallbacks: slot exhaustion → that round runs on the legacy dict cache
  (worker-side); graph capture failure → same fixed-shape step eagerly;
  `--no-batch-decode` → the original code path untouched.

## Verification

- Unit equivalence (random-weight model, CUDA, graph active): batched
  outputs vs legacy path max |Δlogits| = 2.8e-09, argmax identical;
  slot-exhaustion fallback, zero-step-round cancellation race, 448-cap
  long round, and cross-round slot reuse all pass, no slot leaks.
- Worker latency (N=8 probe, real model): step busy 6.5 ms (first
  eager variable-shape version) → **1.0–1.1 ms** (graph); queue wait
  3 ms/max 142 ms → 0.2 ms/max 8 ms.
- **WER gate (final build)**: youtube-debug, max-speed, lines,
  **0.0612 / 0.0612** (baseline band 0.0592–0.0621). An intermediate
  pre-graph build also measured 0.0608 / 0.0604.
- Translation not re-gated: the change affects decode execution only
  and WER is unchanged, so the translation input distribution is
  unchanged (COMET is the arbiter if ever in doubt — see 08-16 report).

## Capacity (GPU 1, clean H200 NVL, 600 s ladders, 0 triggers + 0 errors)

| N (2×WLK + 1×TG) | Result | GPU avg |
|---|---|---|
| 17 | pass 0+0 | 78% |
| 18 | pass 0+0 | 75% |
| 19 | **pass 0+0, ×2 runs** | 84% / 81% |
| 20 | fail 6+0 (wlk1 = 10 conns; repeat with GIL tweak: 8+1) | 87% |

**Ceiling: 17 → 19.** (GPU 0 runs were abandoned after repeated
contamination by another user's jobs; GPU 1 is the same H200 NVL model.)

Control experiments:

- **3×WLK at N=20 (7+7+6): fail 0+8+0** — the failing process carries
  only 7 connections. With decode load largely removed from the GPU
  timeline (SM sums ≈85% incl. vLLM, arithmetic mostly idle), the
  binding constraint is now the **per-process Python/GIL budget of the
  non-decode stages** (VAD dispatch, CT2 encode submissions, SaT,
  alignment post-processing, formatter, websocket/JSON): under load the
  executor worker's ~20 small host-side ops per step queue behind other
  threads' GIL slices (step busy 1 ms → 6–7 ms). More processes no
  longer buy headroom for a different reason than before.
- **`sys.setswitchinterval(0.002)`: rejected** — made it worse (step
  busy 6.3 → 13.0 ms, wait 3.2 → 7.6 ms; N=20 still fails 8+1). More
  GIL handoffs = more overhead. Reverted.
- Cold-start caveat: launching a 20-connection ladder seconds after a
  container restart (no warmup probe) produces keepalive-timeout
  connection errors and garbage results — warm the server with a small
  probe before any ladder.

## Where the next capacity is

Decode is no longer the bottleneck; the per-process Python ceiling
(~9–10 connections) is. Candidate next steps, in rough order of value:

1. Move the remaining per-step session-side Python (alignment softmax /
   running sums / argmax+tolist) into the batched executor so a step is
   one host round-trip with near-zero per-session dispatch.
2. Shrink the per-connection fixed Python costs (VAD cadence, SaT
   call frequency, formatter) — same lever as the 08-16 round, now the
   binding one again.
3. Free-threaded Python (3.13+ no-GIL) or a second process *pair* on
   the second GPU — the H200 NVL host has two GPUs; 2×(2×WLK+TG) should
   serve ~38 with the current build.

## Follow-up: 27b Q8 TranslateGemma revisited (2026-08-20 night)

The 08-19 experiment (pre-batching build) found 27b dropped the ceiling
to 14 because vLLM's longer bursts squeezed WLK's launch-bound decode
off the timeline. With batched/graph decode that mechanism is largely
gone, so 27b was re-laddered (fp8, gpu-mem-util 0.35, clean GPU 1,
warmup probe first):

| N (2×WLK + 27b TG) | old build | new build |
|---|---|---|
| 15 | fail 7+0 | pass 0+0 |
| 16 | — | **pass 0+0, ×2 runs** (GPU 89%, vLLM SM 36%, VRAM 66 GiB) |
| 17 | — | fail 3+0 (vLLM waiting=0 — still timeline squeeze, just later) |

**27b ceiling: 14 → 16.** The production target of 15 now fits with one
step of margin. Trade-off vs 12b (ceiling 19): 3 connections of
headroom for the larger translation model. Config was reverted to
12b/GPU0 after the experiment; switching is the same 3 env edits as
before plus `WLK_TRANSLATION_MODEL_SIZE=27b`.

## N=1 overhead (measured)

Full executor round-trip (queue → worker → graph replay → future) at
N=1: **1.36 ms/step vs 1.14 ms/step legacy** (+0.22 ms — the two thread
hops). ~2-4 steps per round, rounds every ~0.5-1 s → ~1 ms per round
against an end-to-end caption latency dominated by second-scale audio
buffering: imperceptible.

## Deployment notes

- Defaults: batched decode ON (`--no-batch-decode` to disable),
  `--batch-decode-slots 16` (~0.6 GiB VRAM for cool-whisper's 2-layer
  decoder; scales with layer count).
- Slot buffers add ~0.5 GiB per WLK process; peak VRAM at N=19 was
  ~45 GiB total (2×WLK ≈ 9-10 GiB each + vLLM 26.6 GiB).
- Not yet committed; not yet synced to cool-ai-assistant. Production
  (asr_backend) still runs the pre-batching image until rebuilt there.

## 2026-08-21 — 27b Q8 adopted; one-GPU 1×WLK and 3×WLK re-ladders

27b Q8 (fp8 online, gpu-mem-util 0.35) is now the configured
TranslateGemma in both the dev `.env`s and cool-ai-assistant's live
`asr_backend` `.env`s. Two more topologies were re-laddered on a clean
H200 NVL (GPU 1) with the batched build (HEAD 2fcfba4), 600 s per step,
pass = 0 lag catch-up triggers and 0 connection errors, warmup probe
before every ladder, original 4-file PCM fixture set (see methodology
note below).

| Topology (one GPU, 27b TG) | Result |
|---|---|
| 1×WLK | 10–13 pass; **14 pass 1/2** (the fail = 1 silence-edge trigger, "committed 0 tokens", GPU 75–78%); **15 fail 13 triggers** (steady-state, one session stuck lagging, GPU 76%); 16 fail 88 |
| 2×WLK (08-20 night, same build) | **16** pass ×2; 17 fail 3+0 |
| 3×WLK | 16, 17 pass; **18 pass ×2**; 19 pass 2/3 (one run: 1 borderline trigger on wlk2, "committed 0 tokens", t=163 s); **20 fail 0+14+0** (steady-state, GPU 94%) |

Read as: **1×WLK = 13 (14 borderline)**, 2×WLK = 16, **3×WLK = 18 (19 borderline)**.
Solo N=14–16 runs sit at GPU 75–78% with whisper CPU 230–270% avg: the
single process is Python-bound, not GPU-bound — same conclusion as the
12b round (pre-batching solo ceiling was 12 with 12b).

3×WLK detail (per-step GPU avg / peak VRAM total): 16: 87% / 73.6 GiB;
17: 89% / 74.2; 18: 89% / 74.3; 19: 90–92% / 75.3; 20: 94% / 75.7.
vLLM SM ≈33–37%, `waiting` always 0 — the limit is the shared GPU
timeline again, not translation queueing. So with 27b the third WLK
process now buys **+2 connections (16 → 18, arguably 19)** over 2×WLK —
unlike the 12b case where 3×WLK did not help: the 27b vLLM bursts are
what squeeze each WLK, and spreading the per-process Python/GIL work
over three processes leaves more slack per process to absorb them.

Methodology note (honesty item): after the 2026-08-21 host reboot
(/tmp wiped twice) the load-test fixture was first regenerated with only
ONE of the four original PCM files. With a single file, every
connection plays the same lecture and connection #3's start offset
(183 s) lands on a ~9.5 s non-speech span; the lag catch-up then
mis-fires exactly at the 10 s threshold ("lag=10.6s backlog=10.0s,
committed 0 tokens") depending on sub-second queue jitter — this
produced spurious 1-trigger "fails" (1×WLK N=14 ×2, 3×WLK N=17) that
were discarded. Results above are from the restored 4-file set
(`-5HVcv8NvFI`, `0biVT6Z32DQ`, `_xUGvhdZtmU`, `-zD50IAAxfY`, cycled per
connection, 61 s offsets). The scripts and fixture now live in
`benchmarking/results/loadtest_scripts/` (gitignored) so a reboot can't
lose them again. Side observation from the artifact: a VAD silence of
~10 s legitimately trips the catch-up with nothing to commit — harmless
for captions (it skips silence) but it inflates "trigger" counts; the
per-process max-lag column in the trial reports (constant 10.4 / 4.6 /
9.0 s across N) is dominated by such content silences, not by load.

## 2026-08-26 — NVFP4 TranslateGemma 27b on H200 (evaluation)

Prompted by the suggestion to use NVFP4 (as in `nvidia/Gemma-4-31B-IT-NVFP4`)
to speed up the 27b translation model.

**Feasibility.** vLLM 0.27.1 loads NVFP4 checkpoints on any SM≥75 GPU, but
native FP4 GEMM (W4A4 on FP4 tensor cores) exists only on Blackwell
(SM100/120). On our H200 (SM90) it logs
`Using MarlinNvFp4LinearKernel for NVFP4 GEMM` — i.e. **weight-only W4A16
via Marlin**: weights are 4-bit in memory and dequantized on the fly, all
arithmetic stays bf16. So on Hopper NVFP4 can only help through memory
bandwidth (weights 18.0 GiB vs 28.2 GiB for online FP8), not compute.

**Checkpoint.** `models/translategemma-27b-it-NVFP4` (17 GB, gitignored):
llm-compressor 0.13.0 `QuantizationModifier(scheme="NVFP4")` on
`Infomaniak-AI/vllm-translategemma-27b-it` (keeps the vLLM chat template
/ RoPE config), `lm_head`, vision tower and projector left in bf16, 128
calibration prompts built from our own lecture SRTs in the served
`<<<source>>>zh<<<target>>>en<<<text>>>` format (job script + calib set in
`benchmarking/results/nvfp4/`). Served via the new `MODEL_PATH` /
`--served-model-name` support in `translate-gemma/docker-compose.yml`
(WLK unchanged).

**Quality (`--translate-debug`, same build 2fcfba4, same day):**

| TG 27b variant | chrF++ | COMET |
|---|---|---|
| Q8 (online fp8), 08-13 reference | 58.00 | 0.7305 |
| Q8 (online fp8), paired run today | 56.96 | **0.7372** |
| NVFP4 run 1 | 50.69 | **0.7358** |
| NVFP4 run 2 | 51.86 | **0.7343** |

COMET Δ vs the paired fp8 run is −0.0014 / −0.0029 — inside the ±0.0066
same-build band: **no measurable quality loss** (chrF++ moves with ASR
segmentation noise, see 08-16 notes; COMET is the gate).

**Speed on H200 (direct vLLM probe, zh→en lecture sentences, ~48 output
tokens/request, ngram speculative decoding on):**

| conc | fp8 p50 / p90 / tok·s⁻¹ | NVFP4 p50 / p90 / tok·s⁻¹ |
|---|---|---|
| 1 | 756 / 1064 ms / 63 | 739 / 961 ms / 65 |
| 8 | 761 / 1082 ms / 284 | 767 / 1148 ms / 282 |
| 16 | 859 / 1221 ms / 465 | 878 / 1171 ms / 494 |

**No speedup on Hopper** (±3%, within noise). Per-token time here
(~15 ms) is not weight-bandwidth-bound — it is dominated by speculative
verification steps, small-batch launch overhead and the API path — so
halving the weight bytes does not show. The large speedup reported for
Gemma-4-31B NVFP4 is a Blackwell (native FP4 tensor core) effect; the
same checkpoint on H200 degenerates to a W4A16 path that is bandwidth-
equivalent to a good 4-bit AWQ/GPTQ model. Memory is the only win here:
−10 GiB weights (KV pool grows correspondingly under the same
`GPU_MEMORY_UTILIZATION`).

**Capacity (2×WLK + 1×TG NVFP4, GPU 1, 600 s, 4-file fixture):**
| N | fp8 (08-20) | NVFP4 (08-26) |
|---|---|---|
| 16 | pass ×2, GPU 89%, vLLM SM 36% | pass 0+0, GPU 92%, vLLM SM 42%, VRAM 65.9 GiB |
| 17 | fail 3+0 | **fail 12+0** (wlk1 steady-state lag from t≈246 s), GPU 93%, vLLM SM 42% |

**Ceiling unchanged at 16.** The Marlin W4A16 path actually occupies
*more* of the GPU timeline than fp8 (vLLM SM 42% vs 36%; on-the-fly
dequant + bf16 GEMM vs native fp8 tensor-core GEMM), so the WLK processes
get squeezed slightly harder — the N=17 failure is heavier (12 vs 3
triggers). (Top step not re-confirmed ×2: the 17 failure is unambiguous
and the conclusion does not depend on 16.)

**Decision: not adopted on this hardware.** Quality is equal, speed is
equal, GPU-timeline load is slightly worse, capacity is equal; the only
benefit is −10 GiB VRAM, which we do not need. Production stays on 27b
online-FP8. The NVFP4 checkpoint and the `MODEL_PATH` serving hook are
kept: on a Blackwell GPU (native FP4) the same checkpoint should show the
speedup the suggestion was based on, and switching is one `.env` line
(`MODEL_PATH=/models/translategemma-27b-it-NVFP4`, `QUANTIZATION_ARGS=`).
If more headroom is wanted on H200, the levers remain moving TG to the
second GPU or a lighter TG (12b fp8: ceiling 19).
