"""
Streaming sentence boundary detection using wtpsplit's SaT model.

Public surface:
  - StreamingSentenceDetector  — stateful per-connection sentence splitter
  - SentenceDetectionProcessor — async pipeline stage (used by AudioProcessor)
  - format_sentence_lines       — format_output replacement for sentence mode
"""

import asyncio
import bisect
import logging
import traceback
import torch
from collections import Counter
from time import monotonic
from typing import List, Optional

from whisperlivekit.timed_objects import ASRToken, Sentence, Silence, Line

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Core sentence splitter
# ---------------------------------------------------------------------------

class StreamingSentenceDetector:
    """
    Receives validated ASRTokens from the transcription pipeline and uses SaT
    to detect sentence boundaries, producing `Sentence` objects with
    accurate start/end timestamps.

    Parameters
    ----------
    sat_model:
        A pre-loaded ``wtpsplit.SaT`` instance shared across connections.
        If *None*, a new ``sat-3l-sm`` model is loaded locally.
    max_tokens:
        Hard limit: split unconditionally when pending tokens exceed this count.
    soft_max_tokens:
        Soft limit: split at the best SaT point (if probable enough) when
        pending tokens exceed this count.
    """

    def __init__(self, sat_model=None, max_tokens: int = 60,
                 soft_max_tokens: int = 40, soft_min_prob: float = 0.01):
        if sat_model is not None:
            self.sat = sat_model
        else:
            from wtpsplit import SaT
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.sat = SaT("sat-3l-sm")
            self.sat.half().to(device)

        self.max_tokens = max_tokens
        self.soft_max_tokens = soft_max_tokens
        self.soft_min_prob = soft_min_prob
        self.pending_tokens: List[ASRToken] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, new_tokens: List[ASRToken]) -> List[Sentence]:
        """
        Append newly validated tokens and return sentences.

        Split strategy (progressive):
        1. Default threshold — use whatever SaT finds naturally.
        2. Soft limit: use predict_proba to find the single best
           split point. Split only if its probability exceeds soft_min_prob.
        3. Hard limit: same predict_proba argmax, but split unconditionally.
           Falls back to character midpoint only if every position has
           probability 0.
        """
        self.pending_tokens.extend(t for t in new_tokens if t.text)
        pending_text = "".join(t.text for t in self.pending_tokens)
        n_tokens = len(self.pending_tokens)

        segments = self.sat.split(pending_text)

        if len(segments) < 2 and n_tokens >= self.soft_max_tokens:
            split = self._best_split_point(pending_text)
            if split is not None and split[1] >= self.soft_min_prob:
                segments = [pending_text[:split[0] + 1], pending_text[split[0] + 1:]]
                logger.debug("Soft limit split: %s", segments)

        if len(segments) < 2 and n_tokens >= self.max_tokens:
            split = self._best_split_point(pending_text)
            if split is not None:
                segments = [pending_text[:split[0] + 1], pending_text[split[0] + 1:]]
                logger.debug("Hard limit split: %s", segments)
            else:
                mid = len(pending_text) // 2
                segments = [pending_text[:mid], pending_text[mid:]]
                logger.debug("Midpoint split: %s", segments)

        return self._extract_sentences(segments)

    def _best_split_point(self, text: str):
        """Return (char_index, probability) of the best split in the first 80%,
        or None if every position has probability 0."""
        import numpy as np
        probs = self.sat.predict_proba(text)
        cutoff = max(1, int(len(text) * 0.8))
        best = int(np.argmax(probs[:cutoff]))
        if probs[best] < 1e-6:
            return None
        return (best, float(probs[best]))

    def flush(self) -> Optional[Sentence]:
        """
        Flush all remaining pending tokens as a final sentence.
        """
        if not self.pending_tokens:
            return None
        sentence = self._tokens_to_sentence(self.pending_tokens)
        self.pending_tokens = []
        return sentence

    def reset(self) -> None:
        """
        Clear pending tokens. Call when hallucination is detected so SaT
        does not build on invalid text.
        """
        logger.warning("[SaT] Resetting pending tokens: %s", "".join(t.text for t in self.pending_tokens))
        self.pending_tokens.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_sentences(self, segments: List[str]) -> List[Sentence]:
        """
        Map SaT segment character boundaries back to token indices and build
        Sentence objects for all segments.
        """
        # Cumulative character length of each token in the joined string
        cumlen: List[int] = []
        running = 0
        for t in self.pending_tokens:
            running += len(t.text)
            cumlen.append(running)

        sentences_out: List[Sentence] = []
        seg_char_end = 0
        token_start = 0

        for seg_idx, seg in enumerate(segments):
            seg_char_end += len(seg)
            is_last = seg_idx == len(segments) - 1
            token_end = token_start
            while (
                token_end < len(self.pending_tokens)
                and cumlen[token_end] <= seg_char_end
            ):
                token_end += 1
            if not is_last and token_end > token_start:
                sentence = self._tokens_to_sentence(
                    self.pending_tokens[token_start:token_end]
                )
                if sentence:
                    sentences_out.append(sentence)
                token_start = token_end
        # Tokens from token_start onward belong to the still-open segment
        self.pending_tokens = self.pending_tokens[token_start:]
        sentences_out.append(self._tokens_to_sentence(self.pending_tokens))

        return sentences_out

    @staticmethod
    def _tokens_to_sentence(tokens: List[ASRToken]) -> Optional[Sentence]:
        valid = [t for t in tokens if t.text.strip()]
        if not valid:
            return None
        text = "".join(t.text for t in tokens).strip()
        return Sentence(start=valid[0].start, end=valid[-1].end, text=text)


# ---------------------------------------------------------------------------
# Async pipeline stage
# ---------------------------------------------------------------------------

class SentenceDetectionProcessor:
    """
    Consumes items from ``sat_queue``. Completed sentences are stored in ``state.sentence_segments`` and
    optionally forwarded to ``translation_sentence_queue`` for offline translation.

    Parameters
    ----------
    detector:
        The per-connection `StreamingSentenceDetector`.
    sat_queue:
        Queue that receives ``list[ASRToken]`` batches and ``Silence`` events.
    state:
        Shared `~whisperlivekit.timed_objects.State` for this connection.
    lock:
        ``asyncio.Lock`` protecting ``state``.
    sentinel:
        The sentinel object that signals end-of-stream (same one used by the
        audio processor so identity comparison works correctly).
    translation_sentence_queue:
        Optional queue to forward sentences for offline translation.
    hallucination_reset:
        Object that signals hallucination detected upstream; clears state.
    translate_pending:
        Whether to translate the still-open (pending) sentence in real time.
    pending_translation_interval:
        Seconds between provisional translations of the pending sentence.
        Only effective when ``translate_pending`` is True.
    silence_commit_timeout:
        Seconds of silence after which the pending sentence is force-committed
        as a completed sentence (SaT is not called during silence, so without
        this the last pending caption would stay stuck until speech resumes).
        <= 0 disables the forced commit.
    """

    def __init__(
        self,
        detector: StreamingSentenceDetector,
        sat_queue: asyncio.Queue,
        state,
        lock: asyncio.Lock,
        sentinel: object,
        translation_sentence_queue: Optional[asyncio.Queue] = None,
        hallucination_reset: Optional[object] = None,
        translate_pending: bool = False,
        pending_translation_interval: float = 1.5,
        silence_commit_timeout: float = 2.0,
    ):
        self.detector = detector
        self.sat_queue = sat_queue
        self.state = state
        self.lock = lock
        self.sentinel = sentinel
        self.translation_sentence_queue = translation_sentence_queue
        self.hallucination_reset = hallucination_reset
        self.translate_pending = translate_pending
        self.pending_translation_interval = pending_translation_interval
        self.silence_commit_timeout = silence_commit_timeout
        self._silence_started_at: Optional[float] = None
        self._silence_flushed = False
        self._pending_translation_task: Optional[asyncio.Task] = None
        self._last_pending_key = None

    async def _flush_pending_sentence(self):
        sentence = await asyncio.to_thread(self.detector.flush)
        async with self.lock:
            if sentence:
                self.state.sentence_segments.append(sentence)
            self.state.sentence_pending = None
        self._last_pending_key = None
        if sentence and self.translation_sentence_queue:
            await self.translation_sentence_queue.put((sentence, False))
        self._silence_flushed = True

    async def _pending_translation_ticker(self):
        """Periodically enqueue a snapshot of the pending sentence for
        provisional translation, so translated captions keep up with speech
        instead of waiting for a SaT boundary."""
        while True:
            await asyncio.sleep(self.pending_translation_interval)
            async with self.lock:
                pending = self.state.sentence_pending
                snapshot = (
                    Sentence(start=pending.start, end=pending.end, text=pending.text)
                    if pending and (pending.text or "").strip()
                    else None
                )
            if snapshot is None:
                continue
            key = (snapshot.start, snapshot.text)
            if key == self._last_pending_key:
                continue
            self._last_pending_key = key
            await self.translation_sentence_queue.put((snapshot, True))

    async def _stop_pending_translation_ticker(self):
        if self._pending_translation_task is not None:
            self._pending_translation_task.cancel()
            await asyncio.gather(self._pending_translation_task, return_exceptions=True)
            self._pending_translation_task = None

    async def run(self):
        """Main processing loop — run as an ``asyncio.Task``."""
        if self.translation_sentence_queue and self.translate_pending and self.pending_translation_interval > 0:
            self._pending_translation_task = asyncio.create_task(
                self._pending_translation_ticker()
            )
        try:
            await self._run_loop()
        finally:
            await self._stop_pending_translation_ticker()
        logger.info("SentenceDetectionProcessor finished.")

    async def _run_loop(self):
        while True:
            try:
                # Once the silence outlasts silence_commit_timeout, the
                # pending sentence is flushed as completed.
                timeout = None
                if (
                    self._silence_started_at is not None
                    and not self._silence_flushed
                    and self.silence_commit_timeout > 0
                ):
                    timeout = max(
                        0.0,
                        self.silence_commit_timeout
                        - (monotonic() - self._silence_started_at),
                    )
                if timeout is not None:
                    try:
                        item = await asyncio.wait_for(self.sat_queue.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        logger.info(
                            "Silence exceeded %.1fs: force-committing pending sentence.",
                            self.silence_commit_timeout,
                        )
                        await self._flush_pending_sentence()
                        continue
                else:
                    item = await self.sat_queue.get()

                if item is self.sentinel:
                    logger.debug("SentenceDetectionProcessor: sentinel received, flushing.")
                    # Stop the ticker before the sentinel so no pending item
                    # lands in the translation queue after end-of-stream.
                    await self._stop_pending_translation_ticker()
                    await self._flush_pending_sentence()
                    if self.translation_sentence_queue:
                        await self.translation_sentence_queue.put(self.sentinel)
                    self.sat_queue.task_done()
                    break

                elif self.hallucination_reset is not None and item is self.hallucination_reset:
                    self.detector.reset()
                    async with self.lock:
                        self.state.sentence_pending = None
                        self.state.translation_pending = None
                    self._last_pending_key = None
                    self._silence_started_at = None
                    self._silence_flushed = False
                    self.sat_queue.task_done()

                elif isinstance(item, Silence):
                    if item.has_ended:
                        # Audio-time check: catches silences that outlast the
                        # threshold in stream time but not in wall-clock time
                        # (e.g. faster-than-realtime replay).
                        if (
                            not self._silence_flushed
                            and self.silence_commit_timeout > 0
                            and item.duration
                            and item.duration > self.silence_commit_timeout
                        ):
                            await self._flush_pending_sentence()
                        self._silence_started_at = None
                        self._silence_flushed = False
                    elif self._silence_started_at is None:
                        # Wall-clock flush during the silence is handled by
                        # the timeout on the queue wait above.
                        self._silence_started_at = monotonic()
                    self.sat_queue.task_done()

                elif isinstance(item, list):
                    if self._silence_started_at is not None:
                        # Tokens decoded from pre-silence audio arrive after
                        # the silence start event (the ASR force-finalize is
                        # asynchronous). The silence itself is still ongoing —
                        # a genuine speech resume is always preceded by the
                        # silence end event, which clears this tracking — so
                        # re-arm the commit timer instead of cancelling it.
                        self._silence_started_at = monotonic()
                        self._silence_flushed = False
                    sentences = await asyncio.to_thread(self.detector.push, item)
                    # Only store completed sentences; propagate pending for live display.
                    async with self.lock:
                        for sentence in sentences[:-1]:
                            if sentence:
                                self.state.sentence_segments.append(sentence)
                        self.state.sentence_pending = sentences[-1] if sentences and sentences[-1] else None
                    if self.translation_sentence_queue:
                        for sentence in sentences[:-1]:
                            if sentence:
                                await self.translation_sentence_queue.put((sentence, False))
                    self.sat_queue.task_done()

                else:
                    self.sat_queue.task_done()

            except Exception as e:
                logger.warning("Exception in SentenceDetectionProcessor: %s", e)
                logger.debug("Traceback: %s", traceback.format_exc())
                self.sat_queue.task_done()


# ---------------------------------------------------------------------------
# Output formatter for sentence-detection mode
# ---------------------------------------------------------------------------

def format_sentence_lines(state, args, tokens=None, withhold_untranslated=False):
    """
    Build `~whisperlivekit.timed_objects.Line` objects from SaT-detected
    sentences instead of raw ASR tokens.

    Parameters
    ----------
    state:
        The shared `~whisperlivekit.timed_objects.State`.
    args:
        The server argument namespace.  When ``args.diarization`` is falsy
        the expensive per-sentence speaker scan is skipped entirely.
    tokens:
        Optional pre-snapshotted token list. When *None*, ``state.tokens``
        is used directly.
    withhold_untranslated:
        Hold back the trailing lines whose translation has not arrived yet,
        so caption and translation reach the client together.
    """
    send_pending = getattr(args, "send_pending", True)
    sentences = list(state.sentence_segments)
    pending = getattr(state, "sentence_pending", None) if send_pending else None
    if pending:
        sentences = sentences + [pending]
    if not sentences:
        return [], []

    if tokens is None:
        tokens = state.tokens
    translation_segs = state.translation_validated_segments or []
    if not isinstance(translation_segs, list):
        translation_segs = [translation_segs] if translation_segs else []

    use_diarization = getattr(args, "diarization", False)

    lines: List[Line] = []
    for sentence in sentences:
        speaker = _dominant_speaker(tokens, sentence.start, sentence.end) if use_diarization else 1
        lines.append(Line(
            speaker=speaker,
            text=sentence.text,
            start=sentence.start,
            end=sentence.end,
        ))

    # Attach translations using the same overlap logic as format_output
    if lines and translation_segs:
        unassigned = []
        for ts in translation_segs:
            assigned = False
            for line in lines:
                if ts and ts.overlaps_with(line):
                    if ts.is_within(line):
                        line.translation += ts.text + " "
                        assigned = True
                        break
                    else:
                        ts0, ts1 = ts.approximate_cut_at(line.end)
                        if ts0 and line.overlaps_with(ts0):
                            line.translation += ts0.text + " "
                        if ts1:
                            unassigned.append(ts1)
                        assigned = True
                        break
            if not assigned:
                unassigned.append(ts)
        if unassigned:
            for line in lines:
                remaining = []
                for ts in unassigned:
                    if ts and ts.overlaps_with(line):
                        line.translation += ts.text + " "
                    else:
                        remaining.append(ts)
                unassigned = remaining

    # Provisional (pending) translation: attach to the line that shares its
    # exact start. Usually that is the pending (last) line; right after a SaT
    # cut it is the just-finalized sentence, which keeps the provisional text
    # on screen until the validated translation replaces it — instead of the
    # translation blinking out. A stale snapshot from an older generation
    # matches no line and is dropped.
    translation_pending = getattr(state, "translation_pending", None) if send_pending else None
    if translation_pending and (translation_pending.text or "").strip():
        for line in reversed(lines):
            if line.start == translation_pending.start:
                if not line.translation:
                    line.translation = translation_pending.text
                    line.translation_provisional = True
                    line.translation_stable_chars = getattr(
                        translation_pending, "stable_chars", 0
                    )
                break
            if line.start < translation_pending.start:
                break

    # Only the trailing untranslated run is withheld: an untranslated line
    # that is followed by a translated one lost its translation to an error
    # and will never get it, so holding it back would stall the stream.
    if withhold_untranslated:
        keep = len(lines)
        while keep and not (lines[keep - 1].translation or "").strip():
            keep -= 1
        lines = lines[:keep]

    # if lines and translation_segs:
    #     latest_transcription_end = lines[-1].end
    #     latest_translation_end = translation_segs[-1].end
    #     translation_delay = latest_transcription_end - latest_translation_end
    #     logger.debug(
    #         "Translation delay: %.2fs (transcription end=%.2f, translation end=%.2f)",
    #         translation_delay,
    #         latest_transcription_end,
    #         latest_translation_end,
    #     )

    return lines, []


class _TokenStartIndex:
    """Lazy-built index over token start times for bisect lookups.

    Rebuilt only when the token list identity or length changes.
    """
    __slots__ = ("_id", "_len", "_starts")

    def __init__(self):
        self._id = None
        self._len = 0
        self._starts: List[float] = []

    def get_starts(self, tokens: List[ASRToken]) -> List[float]:
        tid = id(tokens)
        tlen = len(tokens)
        if self._id != tid or self._len != tlen:
            self._starts = [t.start for t in tokens]
            self._id = tid
            self._len = tlen
        return self._starts

_token_start_idx = _TokenStartIndex()


def _dominant_speaker(tokens: List[ASRToken], start: float, end: float) -> int:
    """Return the most common speaker among tokens that overlap [start, end).

    Uses bisect on token start times to narrow the scan window instead of
    iterating over all tokens.
    """
    if not tokens:
        return 1
    starts = _token_start_idx.get_starts(tokens)
    # Tokens with t.start < end could overlap; bisect_left finds the
    # first token whose start >= end — everything before that is a
    # candidate.  Among those, we only want tokens with t.end > start.
    hi = bisect.bisect_left(starts, end)
    if hi == 0:
        return 1
    counts: dict = {}
    for i in range(hi):
        t = tokens[i]
        if t.end <= start:
            continue
        sp = getattr(t, "corrected_speaker", t.speaker)
        if sp is None or sp == -1:
            sp = 1
        else:
            sp = int(sp)
        counts[sp] = counts.get(sp, 0) + 1
    if not counts:
        return 1
    return max(counts, key=counts.__getitem__)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detector = StreamingSentenceDetector()
    while True:
        text = input("Enter text: ")
        fake_tokens = [
            ASRToken(start=float(i), end=float(i) + 0.5, text=w)
            for i, w in enumerate(text)
        ]
        sentences = detector.push(fake_tokens)
        for s in sentences:
            print(f"  [{s.start:.1f}-{s.end:.1f}] {s.text}")
