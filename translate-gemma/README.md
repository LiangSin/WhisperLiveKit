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
| `HF_TOKEN` | _(empty)_ | HuggingFace token used the first time the model is downloaded. |
| `MODEL_SIZE` | `4b` | One of `4b`, `12b`, `27b`. Resolves to `Infomaniak-AI/vllm-translategemma-${MODEL_SIZE}-it`. |
| `PORT` | `8765` | Host port that exposes the OpenAI-compatible API. |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU(s) the container may use. |
| `MAX_MODEL_LEN` | `512` | vLLM `--max-model-len`. |
| `GPU_MEMORY_UTILIZATION` | `0.85` | vLLM `--gpu-memory-utilization`. |
| `MAX_NUM_SEQS` | `128` | vLLM `--max-num-seqs` (concurrency cap). |
| `HF_CACHE_DIR` | `~/.cache/huggingface` | Path on the host where model weights are cached. |
| `SSL_CONFIG_DIR` | `../ssl-config` | Host directory containing TLS certs mounted into the container. |

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
