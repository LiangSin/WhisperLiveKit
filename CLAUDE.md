# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of QuentinFuxa/WhisperLiveKit focused on **live Mandarin lecture captioning with translation**, deployed at NTU (production consumer: the sibling `../LiveCaption` repo, which relays RTMP course streams to this server's `/asr` websocket). Key fork differences from upstream: per-connection session config (prompts), SaT sentence-based output instead of growing lines, a TranslateGemma sentence-level translation backend, per-connection archiving, hallucination filtering, and a Chinese-tuned default model (`models/cool-whisper`, a symlink into the HF cache).

## Commands

```bash
docker compose up --build whisperlivekit
```

**There are no unit tests.** Verification is empirical: run the server, then run the benchmark and compare WER / chrF++. Quick smoke tests: `whisperlivekit/smoke_translation_format_output.py`, and `sentence_detector.py` / `translategemma.py` have `__main__` blocks.

### Benchmarking (requires a running server)

```bash
scripts/start_benchmark.sh --youtube-debug          # presets: --{libri,youtube,translate}[-debug]
python benchmarking/calculate_overall_metrics.py benchmarking/results/<file>.json   # WER
python benchmarking/calculate_translate_metrics.py -hyp ... -ref ... --metrics chrf # chrF++/COMET
```

- `benchmarking/run_benchmark.py` is a websocket client; important flags: `--speed` (0 = max speed, 1.0 = realtime), `--accumulate-mode {lines,merge}` (`lines` = this fork's sentence output; `merge` = upstream-style growing lines), `--keywords-json` (sends per-connection prompts from `dataset/keywords*.json`).
- Results go in `benchmarking/results/` named `<date>_<git-sha>_<playlist>_<variant>.json`.
- Datasets: `scripts/download_dataset/{librispeech,youtube}.sh` → `dataset/<PLAYLIST>/<VIDEO_ID>.wav` + `.zh-TW.srt` references.
- `dataset/keyword_protocol.md` documents how the keyword-prompt JSON files were generated (per-lecture, syllabus-level terms only, frozen — do not tune them against benchmark scores).

### Custom models

`model_paths.py` sniffs a model dir: CT2 `model.bin` → faster-whisper encoder + PyTorch decoder (fork optimization; `--disable-fast-encoder` turns off), MLX `weights.npz`, or plain `.pt`. HF checkpoints convert via `scripts/convert_hf_whisper.py`; custom models need `--custom-alignment-heads` from `scripts/determine_alignment_heads.py` (AlignAtt depends on them).

## Architecture

**One shared engine, per-connection everything else.** `core.TranscriptionEngine` is a singleton created at server startup holding the loaded models (Whisper, VAD session, diarization, translation, SaT). Each `/asr` websocket gets its own `AudioProcessor` (`audio_processor.py`), which wires per-connection async stages through queues:

```
websocket bytes → FFmpegManager (webm/opus→PCM, skipped with --pcm-input)
  → VAD (silero_vad_iterator / ten_vad_iterator; emits Silence boundaries)
  → transcription_queue → SimulStreamingOnlineProcessor
  → sat_queue → StreamingSentenceDetector (SaT sentence boundaries)
  → translation_queue → GemmaTranslationProcessor or nllw
  → results_generator → websocket JSON
```

Shared data model for everything flowing through the pipeline: `timed_objects.py` (`ASRToken`, `Silence`, `Line`, `State`, …). Read it first when touching pipeline code.

**Session config protocol** (fork feature, spec in `docs/session_config.md`): a client may send a JSON text frame `{"type":"config","init_prompt":...,"static_init_prompt":...}` *before the first audio chunk*; `basic_server.py` defers `AudioProcessor` creation until the first audio bytes so the config can land first, replies `config_ack`/`config_error`. Old bytes-only clients are unaffected — preserve this backward compatibility in any protocol change.

**SimulStreaming backend** (`whisperlivekit/simul_whisper/`, default policy): AlignAtt attention-guided streaming decoding. Model weights are shared across sessions; each connection owns a lightweight `DecoderState` (KV cache, tokens, context) and a `dataclasses.replace()`-copied `AlignAttConfig` — this is what makes per-session prompts safe; never mutate `asr.cfg` directly. `static_init_prompt` is pinned in the decoder context (`trim_context` protects it); `init_prompt` scrolls away. The alternative `local_agreement/` policy (`--backend-policy localagreement`) is upstream's; per-session prompts don't apply to it.

**Sentence mode supersedes the legacy formatter.** With `--sentence-detection` (auto-on when TranslateGemma is used), `sentence_detector.format_sentence_lines` replaces `results_formater.format_output`: output lines are stable, completed sentences; `--silence-commit-timeout` force-commits an open sentence during silence; `--no-send-pending` withholds sentences until translated. Any change to output shape must consider both formatters plus the benchmark's two `--accumulate-mode`s.

**Hallucination handling**: `hallucination_filter.py` — blacklist phrases in `boh.json` ("bag of hallucinations") plus a rolling-hash `LoopingDetector`; a hit triggers a decoder reset that salvages not-yet-transcribed audio. VAD tuning matters here: silence boundaries force-finalize decoding, and decoding over silence is the main hallucination trigger (see comments in `audio_processor.py`).

**Translation backends**: `nllw` (token-level, external package) vs `translategemma.py` (sentence-level client to a separate vLLM container in `translate-gemma/`, OpenAI-compatible HTTPS API; one instance can serve multiple WLK servers). TranslateGemma uses local-agreement prefix promotion so displayed translations don't flicker.

**Archiving** is on by default: `archive_writer.py` writes `archives/<timestamp>/` (webm audio segments + transcript/translation SRT) per connection. `archives/` contains hundreds of real session dirs — exclude it when searching (`grep --exclude-dir=archives`).

## Gotchas

- Default `--language zh`, default model `models/cool-whisper` (a **symlink** — don't commit its target; `scripts/download_breeze.sh` shows the pattern).
- Browser mic capture requires HTTPS; `ssl-config/` holds self-signed certs and everything (server, benchmark, translate-gemma) speaks TLS with verification off internally.
- The Docker entrypoint is `whisperlivekit/monitor-client/entrypoint.py`, which wraps the real server and pushes health status to an external monitor (`MONITOR_*` env vars).
- Many fork-added CLI flags live in `parse_args.py`'s `simulstreaming_group` even when not SimulStreaming-specific.
- This host also runs the LiveCaption production stack (nginx, relay_service, ome, …) whose ASR backend is this repo's `whisperlivekit` container — restarting it briefly interrupts live captions (LiveCaption auto-reconnects).
