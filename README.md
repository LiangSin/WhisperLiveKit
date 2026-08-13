<h1 align="center">WhisperLiveKit</h1>

<p align="center">
<img src="https://raw.githubusercontent.com/QuentinFuxa/WhisperLiveKit/refs/heads/main/demo.png" alt="WhisperLiveKit Demo" width="730">
</p>

<p align="center"><b>Real-time, Fully Local Speech-to-Text with Speaker Identification</b></p>

<p align="center">
<a href="https://pypi.org/project/whisperlivekit/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/whisperlivekit?color=g"></a>
<a href="https://pepy.tech/project/whisperlivekit"><img alt="PyPI Downloads" src="https://static.pepy.tech/personalized-badge/whisperlivekit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=installations"></a>
<a href="https://pypi.org/project/whisperlivekit/"><img alt="Python Versions" src="https://img.shields.io/badge/python-3.9--3.15-dark_green"></a>
<a href="https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache 2.0-dark_green"></a>
</p>


Real-time transcription directly to your browser, with a ready-to-use backend+server and a simple frontend.

> [!NOTE]
> **This is a fork** of [QuentinFuxa/WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) focused on **live Mandarin lecture captioning with English translation**. Main additions over upstream: [per-connection session config](docs/session_config.md) (custom prompts per WebSocket connection), SaT sentence-based output (stable completed-sentence lines instead of growing lines), a [TranslateGemma](translate-gemma/) sentence-level translation backend served by vLLM, hallucination filtering, per-connection archiving, and a [benchmarking suite](benchmarking/). See the **Fork features** section below. This fork is deployed with `docker compose` only (see **Installation & Quick Start**); local pip installs are untested here — refer to upstream for those.

#### Powered by Leading Research:

- Simul-[Whisper](https://github.com/backspacetg/simul_whisper)/[Streaming](https://github.com/ufal/SimulStreaming) (SOTA 2025) - Ultra-low latency transcription using [AlignAtt policy](https://arxiv.org/pdf/2305.11408)
- [NLLW](https://github.com/QuentinFuxa/NoLanguageLeftWaiting) (2025), based on [distilled](https://huggingface.co/entai2965/nllb-200-distilled-600M-ctranslate2) [NLLB](https://arxiv.org/abs/2207.04672) (2022, 2024) - Simulatenous translation from & to 200 languages.
- [WhisperStreaming](https://github.com/ufal/whisper_streaming) (SOTA 2023) - Low latency transcription using [LocalAgreement policy](https://www.isca-archive.org/interspeech_2020/liu20s_interspeech.pdf)
- [Streaming Sortformer](https://arxiv.org/abs/2507.18446) (SOTA 2025) - Advanced real-time speaker diarization
- [Diart](https://github.com/juanmc2005/diart) (SOTA 2021) - Real-time speaker diarization
- [Silero VAD](https://github.com/snakers4/silero-vad) (2024) - Enterprise-grade Voice Activity Detection
- [TranslateGemma](https://huggingface.co/google/translategemma-4b-it) (2025) - Gemma-3-based translation models (4b/12b/27b), served as a standalone [vLLM service](translate-gemma/)
- [SaT — Segment any Text](https://github.com/segment-any-text/wtpsplit) (2024) - Robust sentence boundary detection used for sentence-level output and translation


> **Why not just run a simple Whisper model on every audio batch?** Whisper is designed for complete utterances, not real-time chunks. Processing small segments loses context, cuts off words mid-syllable, and produces poor transcription. WhisperLiveKit uses state-of-the-art simultaneous speech research for intelligent buffering and incremental processing.


### Architecture

<img alt="Architecture" src="https://raw.githubusercontent.com/QuentinFuxa/WhisperLiveKit/refs/heads/main/architecture.png" />

*The backend supports multiple concurrent users. Voice Activity Detection reduces overhead when no voice is detected.*

## 🐋 Installation & Quick Start (Docker)

This fork is deployed with Docker Compose only. The compose setup orchestrates two GPU services: `whisperlivekit` (this server) and `translategemma` (the vLLM translation service, included from [`translate-gemma/docker-compose.yml`](translate-gemma/docker-compose.yml)).

### Prerequisites
- Docker with the NVIDIA container runtime
- A shared HuggingFace cache directory on the host (`HF_CACHE_DIR`, default `~/.cache/huggingface`)

### Quick Start

```bash
cp .env.example .env                                # WLK_* server flags, model paths, ports
cp translate-gemma/.env.example translate-gemma/.env  # MODEL_SIZE, GPU, quantization, API key
docker compose up --build                           # both services
docker compose up --build whisperlivekit            # transcription server only
```

Then open `https://localhost:8000` in your browser (the web UI is served by the container; the WebSocket endpoint for clients is `wss://<host>:8000/asr`).

Nearly every server flag is driven from `.env` (`WLK_*` variables); edit it and re-run `docker compose up -d` to apply. The full flag reference is in **Parameters & Configuration** below. The default model is a Chinese-tuned Whisper (`models/cool-whisper`, resolved inside the image via the `COOL_WHISPER_SNAPSHOT_DIR` build arg pointing into the HF cache); the list of available languages is in [tokenizer.py](whisperlivekit/simul_whisper/whisper/tokenizer.py).

### Notes

- The container entrypoint is `whisperlivekit/monitor-client/entrypoint.py`, which wraps the real server and can push health status to an external monitor (`MONITOR_*` env vars; `.env.example` ships with `WLK_MONITOR_ENABLED=false`).
- TranslateGemma model size must match on both sides: `MODEL_SIZE` in `translate-gemma/.env` and `WLK_TRANSLATION_MODEL_SIZE` in `.env`.
- There is no CPU image; both services expect a GPU.


## Parameters & Configuration


| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Whisper model size. List and recommandations [here](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/docs/available_models.md) | `small` |
| `--model-path` | Local .pt file/directory **or** Hugging Face repo ID containing the Whisper model. Overrides `--model`. Recommandations [here](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/docs/models_compatible_formats.md) | `None` |
| `--language` | List [here](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/whisperlivekit/simul_whisper/whisper/tokenizer.py). If you use `auto`, the model attempts to detect the language automatically, but it tends to bias towards English. | `auto` |
| `--target-language` | Enables translation to this language, using the backend chosen by `--translation-model` (NLLW: [200 languages](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/docs/supported_languages.md); TranslateGemma: sentence-level). If you want to translate to english, you can also use `--direct-english-translation`. The STT model will try to directly output the translation. | `None` |
| `--diarization` | Enable speaker identification | `False` |
| `--backend-policy` | Streaming strategy: `1`/`simulstreaming` uses AlignAtt SimulStreaming, `2`/`localagreement` uses the LocalAgreement policy | `simulstreaming` |
| `--backend` | Whisper implementation selector. `auto` picks MLX on macOS (if installed), otherwise Faster-Whisper, otherwise vanilla Whisper. You can also force `mlx-whisper`, `faster-whisper`, `whisper`, or `openai-api` (LocalAgreement only) | `auto` |
| `--no-vac` | Disable Voice Activity Controller | `False` |
| `--vac-backend` | VAC backend: `silero` or `ten-vad` (requires `pip install ten-vad`) | `silero` |
| `--no-vad` | Disable Voice Activity Detection | `False` |
| `--vad-threshold` | VAD speech probability threshold; probabilities above it count as speech | `0.5` |
| `--min-chunk-size` | Minimum audio chunk size in seconds (processing cadence) | `0.5` |
| `--warmup-file` | Audio file path for model warmup | `jfk.wav` |
| `--host` | Server host address | `localhost` |
| `--port` | Server port | `8000` |
| `--ssl-certfile` | Path to the SSL certificate file (for HTTPS support) | `None` |
| `--ssl-keyfile` | Path to the SSL private key file (for HTTPS support) | `None` |
| `--forwarded-allow-ips` | Ip or Ips allowed to reverse proxy the whisperlivekit-server. Supported types are  IP Addresses (e.g. 127.0.0.1), IP Networks (e.g. 10.100.0.0/16), or Literals (e.g. /path/to/socket.sock) | `None` |
| `--allowed-ips` | Comma-separated list of client IPs allowed to connect; unset allows all | `None` |
| `--allowed-networks` | Comma-separated list of allowed client networks in CIDR notation (e.g. `172.16.0.0/12,140.112.0.0/16`) | `None` |
| `--pcm-input` | raw PCM (s16le) data is expected as input and FFmpeg will be bypassed. Frontend will use AudioWorklet instead of MediaRecorder | `False` |

| Translation & Sentence Detection options | Description | Default |
|-----------|-------------|---------|
| `--target-language` | Target language code for translation. Required to enable translation. | `` |
| `--translation-model` | Translation backend: `nllw` (streaming, 200 languages) or `translategemma` (sentence-level, Google TranslateGemma). `translategemma` automatically enables sentence detection. | `nllw` |
| `--translation-model-size` | Model size: `600M`/`1.3B` for NLLW; `4b`/`12b`/`27b` for TranslateGemma. | `600M` |
| `--translategemma-url` | OpenAI-compatible endpoint of the standalone TranslateGemma vLLM service (see [`translate-gemma/`](./translate-gemma/)). Falls back to the `TRANSLATEGEMMA_URL` env var. Only used with `--translation-model translategemma`. | `https://localhost:8765/v1` |
| `--nllb-backend` | NLLW inference backend: `transformers` or `ctranslate2`. | `transformers` |
| `--sentence-detection` | Enable SaT-based sentence boundary detection on validated transcription: output `lines` become stable, completed sentences instead of growing lines (see [docs/API.md](docs/API.md)). Requires `pip install wtpsplit`. Automatically enabled with `--translation-model translategemma`. | `False` |
| `--translation-context-sentences` | Number of preceding validated sentences fed as context to TranslateGemma requests (0 disables). | `2` |
| `--translate-pending` | Also translate the still-open (pending) sentence periodically, so translated captions keep up with speech instead of waiting for a sentence boundary. Off by default to reduce caption flicker. | `False` |
| `--pending-translation-interval` | Seconds between provisional translations of the pending sentence (with `--translate-pending`). | `1.5` |
| `--no-send-pending` | Send only SaT-completed sentences to clients: pending sentence and unvalidated buffers are withheld, and each finalized sentence is held back until its translation arrived, so caption and translation appear together. | pending is sent |
| `--silence-commit-timeout` | Seconds of silence after which the pending sentence is force-committed (SaT is not called during silence). `0` disables. | `2.0` |
| `--sat-soft-max-tokens` | Soft sentence length limit: past this many tokens, split at SaT's best point if probable enough. | `40` |
| `--sat-max-tokens` | Hard sentence length limit: past this many tokens, split unconditionally. | `60` |

| Diarization options | Description | Default |
|-----------|-------------|---------|
| `--diarization-backend` |  `diart` or `sortformer` | `sortformer` |
| `--disable-punctuation-split` |  Disable punctuation based splits. See #214 | `False` |
| `--segmentation-model` | Hugging Face model ID for Diart segmentation model. [Available models](https://github.com/juanmc2005/diart/tree/main?tab=readme-ov-file#pre-trained-models) | `pyannote/segmentation-3.0` |
| `--embedding-model` | Hugging Face model ID for Diart embedding model. [Available models](https://github.com/juanmc2005/diart/tree/main?tab=readme-ov-file#pre-trained-models) | `pyannote/embedding` |

| Archiving options (fork feature, on by default) | Description | Default |
|-----------|-------------|---------|
| `--no-archive` | Disable per-connection archival outputs (audio segments + SRT transcript/translation under `archives/<timestamp>/`). | archiving on |
| `--archive-dir` | Base directory used to store connection archives. | `archives` |
| `--archive-segment-seconds` | Audio segment rotation interval in seconds. | `1800` |
| `--archive-subtitle-flush-seconds` | How often subtitle files are flushed to disk. | `5.0` |

| SimulStreaming backend options | Description | Default |
|-----------|-------------|---------|
| `--disable-fast-encoder` | Disable Faster Whisper or MLX Whisper backends for the encoder (if installed). Inference can be slower but helpful when GPU memory is limited | `False` |
| `--custom-alignment-heads` | Use your own alignment heads, useful when `--model-dir` is used. Use `scripts/determine_alignment_heads.py` to extract them. <img src="scripts/alignment_heads.png" alt="WhisperLiveKit Demo" width="300">
 | `None` |
| `--frame-threshold` | AlignAtt frame threshold (lower = faster, higher = more accurate) | `25` |
| `--beams` | Number of beams for beam search (1 = greedy decoding) | `1` |
| `--decoder` | Force decoder type (`beam` or `greedy`) | `auto` |
| `--audio-max-len` | Maximum audio buffer length (seconds) | `30.0` |
| `--audio-min-len` | Minimum audio length to process (seconds) | `0.0` |
| `--cif-ckpt-path` | Path to CIF model for word boundary detection | `None` |
| `--never-fire` | Never truncate incomplete words | `False` |
| `--init-prompt` | Initial prompt for the model | `None` |
| `--static-init-prompt` | Static prompt that doesn't scroll | `None` |
| `--max-context-tokens` | Maximum context tokens | `None` |



| WhisperStreaming backend options | Description | Default |
|-----------|-------------|---------|
| `--confidence-validation` | Use confidence scores for faster validation | `False` |
| `--buffer_trimming` | Buffer trimming strategy (`sentence` or `segment`) | `segment` |




> For diarization using Diart, you need to accept user conditions [here](https://huggingface.co/pyannote/segmentation) for the `pyannote/segmentation` model, [here](https://huggingface.co/pyannote/segmentation-3.0) for the `pyannote/segmentation-3.0` model and [here](https://huggingface.co/pyannote/embedding) for the `pyannote/embedding` model. **Then**, login to HuggingFace: `huggingface-cli login`

## Fork features

**Per-connection session config** — a client may send a JSON text frame `{"type":"config","init_prompt":...,"static_init_prompt":...}` *before its first audio chunk* to set per-session Whisper prompts (terminology lists, opening context). The server replies `config_ack`/`config_error`; old bytes-only clients are unaffected. SimulStreaming backend only. Full protocol spec: [docs/session_config.md](docs/session_config.md).

**Sentence mode** — with `--sentence-detection` (auto-on when TranslateGemma is used), output lines are stable, completed sentences detected by [SaT](https://github.com/segment-any-text/wtpsplit) instead of upstream's in-place growing lines. `--silence-commit-timeout`, `--sat-soft-max-tokens`/`--sat-max-tokens`, `--no-send-pending` and `--translate-pending` shape this behavior (see the parameter tables above).

**TranslateGemma translation** — sentence-level translation via a standalone vLLM service in [`translate-gemma/`](translate-gemma/) (one instance can serve several WhisperLiveKit servers). Model size is set with `MODEL_SIZE` (`4b`/`12b`/`27b`) in `translate-gemma/.env` and must match `--translation-model-size` on the server. Weight precision is configurable there too (`DTYPE`, `QUANTIZATION_ARGS`): online FP8 quantization (`--quantization fp8`) measured quality-neutral at every size for ~40% less GPU RAM, while 4-bit bitsandbytes costs measurable quality (see `benchmarking/results/2026-08-13_71f7ff7_translate-debug_quant-sweep.md`). Displayed translations use local-agreement prefix promotion to avoid flicker.

**Hallucination filtering** — `hallucination_filter.py` combines a blacklist of known hallucinated phrases (`boh.json`, "bag of hallucinations") with a rolling-hash looping detector; a hit triggers a decoder reset that salvages not-yet-transcribed audio.

**Per-connection archiving** — on by default; each connection writes `archives/<timestamp>/` with rotated webm audio segments plus transcript and translation SRT files (see Archiving options above).

**Custom / converted Whisper models** — `model_paths.py` sniffs a model directory: CTranslate2 `model.bin` → faster-whisper encoder + PyTorch decoder (fork optimization, disable with `--disable-fast-encoder`), MLX `weights.npz`, or plain `.pt`. Convert HF checkpoints with `scripts/convert_hf_whisper.py`; custom models need `--custom-alignment-heads` from `scripts/determine_alignment_heads.py` (the AlignAtt policy depends on them). See [docs/models_compatible_formats.md](docs/models_compatible_formats.md).

**Benchmarking suite** — `scripts/start_benchmark.sh --{libri,youtube,translate}[-debug]` runs a websocket client against a live server; score WER with `benchmarking/calculate_overall_metrics.py` and translation chrF++/COMET with `benchmarking/calculate_translate_metrics.py`. Dataset download scripts live in `scripts/download_dataset/`.

### 🚀 Deployment Guide

Production deployment is the Docker Compose setup above (this is how it runs in production behind an RTMP relay). A few general notes:

1. **Frontend**: Host your customized version of the bundled [web UI](whisperlivekit/web/live_transcription.html) & ensure WebSocket connection points correctly. Browser microphone capture requires HTTPS (`--ssl-certfile`/`--ssl-keyfile`; self-signed certs live in `ssl-config/`), and then `wss://` instead of `ws://` for the WebSocket URL.

2. **Reverse proxy** (optional):
    ```nginx    
   server {
       listen 80;
       server_name your-domain.com;
        location / {
            proxy_pass http://localhost:8000;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
    }}
    ```

3. **Access control**: `--allowed-ips` / `--allowed-networks` restrict which clients may connect.

## 🔮 Use Cases
Capture discussions in real-time for meeting transcription, help hearing-impaired users follow conversations through accessibility tools, transcribe podcasts or videos automatically for content creation, transcribe support calls with speaker identification for customer service...
