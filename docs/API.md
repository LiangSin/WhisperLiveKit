# WhisperLiveKit WebSocket API Documentation

This documentation is intended for devs who want to build custom frontends
against this fork's `/asr` WebSocket endpoint.

> Upstream WhisperLiveKit is migrating to an incremental "segments" API; this
> fork does **not** use it. The format below is what the server actually sends.

---

## Connection lifecycle

```
Client                                Server
  │  ── WebSocket connect ──────────→  │
  │  ←─ {"type":"config", ...} ──────  │   server hello (useAudioWorklet)
  │                                    │
  │  ── {"type":"config", ...} ──────→ │   optional session config
  │  ←─ {"type":"config_ack", ...} ──  │   (must precede the first audio frame)
  │                                    │
  │  ── audio bytes (binary frames) ─→ │   webm/opus (or raw PCM with --pcm-input)
  │  ←─ state snapshots (JSON) ──────  │   several per second
  │  ── empty binary frame (b"") ────→ │   end-of-stream signal
  │  ←─ {"type":"ready_to_stop"} ────  │
```

1. **Server hello** — immediately after accept:
   `{"type": "config", "useAudioWorklet": bool}` (true when the server runs
   with `--pcm-input` and expects raw PCM via AudioWorklet instead of
   MediaRecorder webm).
2. **Optional session config** — a JSON text frame sent *before the first
   audio chunk* can set per-connection Whisper prompts. Answered with
   `config_ack` / `config_error`. Full spec: [session_config.md](session_config.md).
3. **Audio** — binary frames. Sending an empty binary frame ends the session;
   the server flushes remaining audio and replies `ready_to_stop`.

---

## Transcript updates

Transcript messages are **complete state snapshots** (no `type` field):
re-render from each message, do not accumulate manually.

```typescript
{
  "status": "active_transcription" | "no_audio_detected" | "error",
  "lines": [
    {
      "speaker": int,                    // 1,2,3… ; -2 = silence line
      "text": str,                       // transcription text of this line
      "start": str,                      // "H:MM:SS" from session start
      "end": str,                        // "H:MM:SS"
      "translation": str,                // present only when translation is on
      "translation_provisional": bool,   // present only while translation is provisional
      "translation_stable_chars": int,   // with translation_provisional, see below
      "detected_language": str           // present once detected
    }
  ],
  "buffer_transcription": str,           // unvalidated transcription tail
  "buffer_diarization": str,             // text awaiting speaker assignment
  "buffer_translation": str,             // provisional translation tail (nllw backend)
  "remaining_time_transcription": float, // seconds of audio queued for ASR
  "remaining_time_diarization": float,
  "error": str                           // present only when status == "error"
}
```

Notes on fields:

- `lines` — with `--sentence-detection` (the production configuration, auto-on
  with TranslateGemma) each line is a **stable, completed sentence** detected
  by SaT: once emitted, a line's text no longer changes. Without sentence
  detection, lines follow upstream's formatter and may grow/merge in place.
- The **last line may be the still-open (pending) sentence** unless the server
  runs with `--no-send-pending`, in which case only completed sentences are
  sent and each one is withheld until its translation has arrived.
- `translation_provisional` / `translation_stable_chars` — with
  `--translate-pending`, the pending sentence's translation is re-generated
  periodically. `translation_stable_chars` is the number of leading characters
  frozen by local agreement: render `translation[:stable_chars]` normally and
  the tail in a lighter style to minimize visible flicker. A validated
  translation sends neither field.
- Lines with empty `text` are omitted, except silence markers (`speaker: -2`).
- `start`/`end` are formatted `"H:MM:SS"` strings, not floats.

### Status values

| Status | Description |
|--------|-------------|
| `active_transcription` | Normal operation. |
| `no_audio_detected` | No speech detected yet. |
| `error` | Something failed (e.g. FFmpeg); details in `error`. |

---

## Control messages

| Message | Direction | Meaning |
|---------|-----------|---------|
| `{"type":"config","useAudioWorklet":bool}` | server → client | Hello, sent once on connect. |
| `{"type":"config","init_prompt":...,"static_init_prompt":...}` | client → server | Optional per-session prompts, before first audio. See [session_config.md](session_config.md). |
| `{"type":"config_ack","applied":{...}}` | server → client | Config accepted; `applied` lists effective values. |
| `{"type":"config_error","message":...}` | server → client | Config rejected (sent too late / wrong types); not applied. |
| `{"type":"ready_to_stop"}` | server → client | All audio processed after the client sent an empty binary frame. |

---

## Minimal client example

```python
import asyncio, json, websockets

async def transcribe(chunks):
    async with websockets.connect("wss://host:8000/asr") as ws:
        await ws.recv()                       # server hello
        async def send():
            for chunk in chunks:              # webm/opus bytes
                await ws.send(chunk)
            await ws.send(b"")                # end of stream
        sender = asyncio.create_task(send())
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "ready_to_stop":
                break
            if "lines" in msg:
                for line in msg["lines"]:
                    print(line["text"], "→", line.get("translation", ""))
        await sender
```

A full-featured reference client is `benchmarking/run_benchmark.py`; the
bundled web UI (`whisperlivekit/web/live_transcription.html`) is the browser
reference implementation.
