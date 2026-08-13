# TranslateGemma — standalone service

Runs an `Infomaniak-AI/vllm-translategemma-*-it` model behind vLLM's
OpenAI-compatible HTTPS API in its own container, decoupled from
WhisperLiveKit. vLLM's continuous-batching engine multiplexes requests
across all connected clients on the GPU.

WhisperLiveKit talks to it via HTTPS — pass
`--translategemma-url https://<host>:<port>/v1` (default
`https://localhost:8765/v1`) on the WhisperLiveKit side.

## Quick start (Docker Compose)

```bash
cp .env.example .env       # edit HF_TOKEN, MODEL_SIZE, GPU, port, ...
docker compose up -d
curl -k https://localhost:8765/v1/models
```

## Configuration

Configuration lives in `.env` (see `.env.example`):

| Variable | Default | Description |
| --- | --- | --- |
| `HF_TOKEN` | _(empty)_ | HuggingFace token used the first time the model is downloaded. The `Infomaniak-AI` checkpoints are public, so this can stay empty. |
| `VLLM_API_KEY` | _(empty)_ | API key required by the OpenAI-compatible endpoint. Must match `VLLM_API_KEY` in the repo-root `.env` so WhisperLiveKit can authenticate. |
| `MODEL_SIZE` | `4b` | One of `4b`, `12b`, `27b`. Resolves to `Infomaniak-AI/vllm-translategemma-${MODEL_SIZE}-it`. Must match `WLK_TRANSLATION_MODEL_SIZE` in the repo-root `.env` — the served model name contains the size, so a mismatch makes every translation request 404. |
| `DTYPE` | `bfloat16` | vLLM `--dtype`. Leave at `bfloat16` (also correct when FP8 quantization is enabled). |
| `QUANTIZATION_ARGS` | _(empty)_ | Extra vLLM flags controlling weight precision. Empty = bf16. See **Quantization** below. |
| `PORT` | `8765` | Host port that exposes the OpenAI-compatible API. |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU(s) the container may use. |
| `MAX_MODEL_LEN` | `512` | vLLM `--max-model-len`. |
| `GPU_MEMORY_UTILIZATION` | `0.85` | vLLM `--gpu-memory-utilization` — fraction of total GPU memory vLLM may claim. Weights + a ~3.3 GiB activation peak must fit inside it or startup fails; the rest becomes KV cache. |
| `MAX_NUM_SEQS` | `128` | vLLM `--max-num-seqs` (concurrency cap). |
| `HF_CACHE_DIR` | `~/.cache/huggingface` | Path on the host where model weights are cached. |
| `SSL_CONFIG_DIR` | `../ssl-config` | Host directory containing TLS certs mounted into the container. |
| `TG_MONITOR_ENABLED` | `true` (compose) | The container starts an embedded monitor client that pushes health status to `MONITOR_URL` (with `MONITOR_SECRET_KEY`); `.env.example` ships with `false`. |

## Quantization

`QUANTIZATION_ARGS` selects the weight precision, quantized online at load
time — same checkpoint, same served model name, no extra downloads:

| Precision | `QUANTIZATION_ARGS` | Notes |
| --- | --- | --- |
| bf16 | _(empty)_ | Stock configuration. |
| q8 | `--quantization fp8` | Online FP8 W8A8. **Measured quality-neutral** (COMET identical to bf16 at 4b/12b/27b) for ~35–42% less GPU RAM. Recommended. |
| q4 | `--quantization bitsandbytes` | In-flight bitsandbytes NF4, weight-only. Costs measurable quality (COMET −0.004…−0.017, worst at 27b); prefer a calibrated W4A16 checkpoint if 4-bit is really needed. |

Measured minimum GPU RAM to start (H200, this serving config) and benchmark
scores per size/precision are in
[`../benchmarking/results/2026-08-13_71f7ff7_translate-debug_quant-sweep.md`](../benchmarking/results/2026-08-13_71f7ff7_translate-debug_quant-sweep.md);
headline minimums: 4b 12.4 GiB (q8 9.8), 12b 27.6 GiB (q8 17.5), 27b 56.2 GiB (q8 32.4).

## Quick smoke test

```bash
curl -k https://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Infomaniak-AI/vllm-translategemma-4b-it",
    "messages": [{"role":"user","content":"<<<source>>>zh<<<target>>>en<<<text>>>你好，最近怎麼樣？"}],
    "temperature": 0,
    "max_tokens": 128,
    "stop": ["<eos>", "<end_of_turn>"]
  }'
```

## Notes

- The service speaks the standard OpenAI chat-completions protocol; vLLM
  applies the model's chat template automatically when given `messages`.
- The `<<<source>>>...<<<target>>>...<<<text>>>...` prompt format is part
  of the user message content and is what TranslateGemma expects.
- Multiple WhisperLiveKit instances can share one TranslateGemma server.
- The server runs with ngram speculative decoding and `--enforce-eager`
  (see `docker-compose.yml`), and exposes `/health` used by the compose
  healthcheck (`start_period` 120s — large models take a while to load).
- If `GPU_MEMORY_UTILIZATION` is set too low for the chosen model/precision,
  vLLM crashes during startup and the container restart-loops; raise the
  value and `docker compose up -d` again.
