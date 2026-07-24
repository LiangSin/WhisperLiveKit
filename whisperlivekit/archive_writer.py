from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Iterable


def _to_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _format_srt_block(index: int, start: float, end: float, text: str) -> str:
    return (
        f"{index}\n"
        f"{_to_srt_time(start)} --> {_to_srt_time(end)}\n"
        f"{text.strip()}\n\n"
    )


@dataclass(frozen=True)
class _CueKey:
    start_ms: int
    end_ms: int
    text: str


class ConnectionArchiveWriter:
    """Write per-connection audio and subtitle archives incrementally.

    Audio is segmented by ffmpeg's segment muxer so each part is playable.
    Subtitles are single-file append streams per connection.
    """

    def __init__(
        self,
        archive_dir: str,
        session_dir_name: str,
        segment_seconds: float = 1800.0,
        subtitle_flush_seconds: float = 5.0,
        audio_format: str = "webm",
    ) -> None:
        self.base_dir = Path(archive_dir)
        self.session_dir = self.base_dir / session_dir_name
        self.segment_seconds = max(1.0, float(segment_seconds))
        self.subtitle_flush_seconds = max(0.5, float(subtitle_flush_seconds))
        self.audio_format = audio_format

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._transcript_path = self.session_dir / "transcript.srt"
        self._translation_path = self.session_dir / "translation.srt"

        self._audio_proc: asyncio.subprocess.Process | None = None
        self._audio_stderr_task: asyncio.Task | None = None
        self._ffmpeg_failed = False

        self._transcript_fp = self._transcript_path.open("a", encoding="utf-8")
        self._translation_fp = self._translation_path.open("a", encoding="utf-8")
        self._transcript_index = self._compute_next_index(self._transcript_path)
        self._translation_index = self._compute_next_index(self._translation_path)
        self._last_subtitle_flush_at = monotonic()

        self._emitted_transcript: set[_CueKey] = set()
        self._emitted_translation: set[_CueKey] = set()

        self._pending_transcript = None
        self._pending_translation = None

    @staticmethod
    def _compute_next_index(path: Path) -> int:
        if not path.exists() or path.stat().st_size == 0:
            return 1
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            indexes = []
            i = 0
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.isdigit() and i + 1 < len(lines) and "-->" in lines[i + 1]:
                    indexes.append(int(stripped))
                    i += 2
                else:
                    i += 1
            return (max(indexes) + 1) if indexes else 1
        except Exception:
            return 1

    async def _start_audio_segmenter(self) -> None:
        if self._audio_proc is not None or self._ffmpeg_failed:
            return
        output_pattern = str(self.session_dir / "audio_part_%04d.webm")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "webm",
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(self.segment_seconds),
            "-reset_timestamps",
            "1",
            output_pattern,
        ]
        try:
            self._audio_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            self._audio_stderr_task = asyncio.create_task(self._drain_audio_stderr())
        except FileNotFoundError:
            self._ffmpeg_failed = True
        except Exception:
            self._ffmpeg_failed = True

    async def _drain_audio_stderr(self) -> None:
        if not self._audio_proc or not self._audio_proc.stderr:
            return
        try:
            while True:
                line = await self._audio_proc.stderr.readline()
                if not line:
                    break
        except asyncio.CancelledError:
            return

    @staticmethod
    def _cue_key(start: float, end: float, text: str) -> _CueKey:
        return _CueKey(int(round(start * 1000)), int(round(end * 1000)), text.strip())

    def _write_transcript_cue(self, start: float, end: float, text: str) -> None:
        clean_text = (text or "").strip()
        if not clean_text or end <= start:
            return
        key = self._cue_key(start, end, clean_text)
        if key in self._emitted_transcript:
            return
        self._transcript_fp.write(_format_srt_block(self._transcript_index, start, end, clean_text))
        self._transcript_index += 1
        self._emitted_transcript.add(key)

    def _write_translation_cue(self, start: float, end: float, text: str) -> None:
        clean_text = (text or "").strip()
        if not clean_text or end <= start:
            return
        key = self._cue_key(start, end, clean_text)
        if key in self._emitted_translation:
            return
        self._translation_fp.write(_format_srt_block(self._translation_index, start, end, clean_text))
        self._translation_index += 1
        self._emitted_translation.add(key)

    @staticmethod
    def _line_tuple(line):
        return (float(line.start or 0), float(line.end or 0), (line.text or "").strip())

    @staticmethod
    def _translation_tuple(line):
        return (float(line.start or 0), float(line.end or 0), (line.translation or "").strip())

    async def write_audio_chunk(self, chunk: bytes) -> None:
        if not chunk:
            return
        await self._start_audio_segmenter()
        if self._ffmpeg_failed or self._audio_proc is None or self._audio_proc.stdin is None:
            return
        try:
            self._audio_proc.stdin.write(chunk)
            await self._audio_proc.stdin.drain()
        except Exception:
            self._ffmpeg_failed = True

    def ingest_lines(self, lines: Iterable) -> None:
        materialized = [line for line in lines if getattr(line, "text", "").strip()]
        if not materialized:
            return

        stable_lines = materialized[:-1] if len(materialized) > 1 else []
        # The last line may still grow; keep it as pending so close() can
        # write the final version if no later ingest supersedes it.
        self._pending_transcript = self._line_tuple(materialized[-1])
        self._pending_translation = (
            None if getattr(materialized[-1], "translation_provisional", False)
            else self._translation_tuple(materialized[-1])
        )

        for line in stable_lines:
            start, end, text = self._line_tuple(line)
            self._write_transcript_cue(start, end, text)
            # Provisional (pending) translations are display-only; only the
            # validated translation belongs in the archive.
            if not getattr(line, "translation_provisional", False):
                tr_start, tr_end, tr_text = self._translation_tuple(line)
                self._write_translation_cue(tr_start, tr_end, tr_text)
        self.flush_subtitles_if_due()

    def flush_subtitles_if_due(self, force: bool = False) -> None:
        now = monotonic()
        if not force and (now - self._last_subtitle_flush_at) < self.subtitle_flush_seconds:
            return
        if self._transcript_fp:
            self._transcript_fp.flush()
        if self._translation_fp:
            self._translation_fp.flush()
        self._last_subtitle_flush_at = now

    async def close(self) -> None:
        # The last ingested line was held back as pending (it may still have
        # been growing); the stream is over now, so it is final — write it.
        if self._pending_transcript:
            self._write_transcript_cue(*self._pending_transcript)
            self._pending_transcript = None
        if self._pending_translation:
            self._write_translation_cue(*self._pending_translation)
            self._pending_translation = None
        self.flush_subtitles_if_due(force=True)
        if self._transcript_fp:
            self._transcript_fp.close()
            self._transcript_fp = None
        if self._translation_fp:
            self._translation_fp.close()
            self._translation_fp = None

        if self._audio_proc and self._audio_proc.stdin:
            try:
                self._audio_proc.stdin.close()
            except Exception:
                pass

        if self._audio_proc:
            try:
                await asyncio.wait_for(self._audio_proc.wait(), timeout=10.0)
            except Exception:
                try:
                    self._audio_proc.kill()
                except Exception:
                    pass
            finally:
                self._audio_proc = None

        if self._audio_stderr_task:
            self._audio_stderr_task.cancel()
            try:
                await self._audio_stderr_task
            except Exception:
                pass
            self._audio_stderr_task = None
