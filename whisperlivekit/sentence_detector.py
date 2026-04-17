"""
Streaming sentence boundary detection using wtpsplit's SaT model.

Public surface:
  - StreamingSentenceDetector  — stateful per-connection sentence splitter
  - SentenceDetectionProcessor — async pipeline stage (used by AudioProcessor)
  - format_sentence_lines       — format_output replacement for sentence mode
"""

import asyncio
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
        Force-flush when pending tokens exceed this count.
    """

    def __init__(self, sat_model=None, max_tokens: int = 30):
        if sat_model is not None:
            self.sat = sat_model
        else:
            from wtpsplit import SaT
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.sat = SaT("sat-3l-sm")
            self.sat.half().to(device)

        self.max_tokens = max_tokens
        self.pending_tokens: List[ASRToken] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, new_tokens: List[ASRToken]) -> List[Sentence]:
        """
        Append newly validated tokens and return sentences.
        """
        self.pending_tokens.extend(t for t in new_tokens if t.text)

        # Concatenate token texts.
        pending_text = "".join(t.text for t in self.pending_tokens)
        segments = self.sat.split(pending_text)

        if len(self.pending_tokens) >= self.max_tokens and len(segments) < 2:
            mid_char = len(pending_text) // 2
            segments = [pending_text[:mid_char], pending_text[mid_char:]]

        return self._extract_sentences(segments)

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
    ):
        self.detector = detector
        self.sat_queue = sat_queue
        self.state = state
        self.lock = lock
        self.sentinel = sentinel
        self.translation_sentence_queue = translation_sentence_queue
        self.hallucination_reset = hallucination_reset
        self._silence_started_at: Optional[float] = None
        self._silence_flushed = False

    async def _flush_pending_sentence(self):
        sentence = await asyncio.to_thread(self.detector.flush)
        async with self.lock:
            if sentence:
                self.state.sentence_segments.append(sentence)
            self.state.sentence_pending = None
        if sentence and self.translation_sentence_queue:
            await self.translation_sentence_queue.put((sentence, False))
        self._silence_flushed = True

    async def run(self):
        """Main processing loop — run as an ``asyncio.Task``."""
        while True:
            try:
                item = await self.sat_queue.get()

                if item is self.sentinel:
                    logger.debug("SentenceDetectionProcessor: sentinel received, flushing.")
                    await self._flush_pending_sentence()
                    if self.translation_sentence_queue:
                        await self.translation_sentence_queue.put(self.sentinel)
                    self.sat_queue.task_done()
                    break

                elif self.hallucination_reset is not None and item is self.hallucination_reset:
                    self.detector.reset()
                    async with self.lock:
                        self.state.sentence_pending = None
                    self._silence_started_at = None
                    self._silence_flushed = False
                    self.sat_queue.task_done()

                elif isinstance(item, Silence):
                    if item.has_ended:
                        if item.duration and item.duration > 5:
                            # Prolonged silence is a reliable sentence boundary.
                            await self._flush_pending_sentence()
                        self._silence_started_at = None
                        self._silence_flushed = False
                    else:
                        if self._silence_started_at is None:
                            self._silence_started_at = monotonic()
                        elif not self._silence_flushed and (monotonic() - self._silence_started_at) > 5:
                            await self._flush_pending_sentence()
                    self.sat_queue.task_done()

                elif isinstance(item, list):
                    # Any token batch means silence has ended/resumed speech.
                    self._silence_started_at = None
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

        logger.info("SentenceDetectionProcessor finished.")


# ---------------------------------------------------------------------------
# Output formatter for sentence-detection mode
# ---------------------------------------------------------------------------

def format_sentence_lines(state, args):
    """
    Build `~whisperlivekit.timed_objects.Line` objects from SaT-detected
    sentences instead of raw ASR tokens.

    Parameters
    ----------
    state:
        The shared `~whisperlivekit.timed_objects.State`.
    args:
        The server argument namespace (currently unused, reserved for future
        options such as per-speaker sentence splitting).
    """
    sentences = list(state.sentence_segments)
    pending = getattr(state, "sentence_pending", None)
    if pending:
        sentences = sentences + [pending]
    if not sentences:
        return [], []

    tokens = state.tokens
    translation_segs = state.translation_validated_segments or []
    if not isinstance(translation_segs, list):
        translation_segs = [translation_segs] if translation_segs else []

    lines: List[Line] = []
    for sentence in sentences:
        speaker = _dominant_speaker(tokens, sentence.start, sentence.end)
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


def _dominant_speaker(tokens: List[ASRToken], start: float, end: float) -> int:
    """Return the most common speaker among tokens that overlap [start, end)."""
    overlapping = [
        t for t in tokens
        if not getattr(t, "is_dummy", False) and t.start < end and t.end > start
    ]
    if not overlapping:
        return 1
    speakers = []
    for t in overlapping:
        sp = getattr(t, "corrected_speaker", t.speaker)
        if sp is None or sp == -1:
            sp = 1
        speakers.append(int(sp))
    return Counter(speakers).most_common(1)[0][0]


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
