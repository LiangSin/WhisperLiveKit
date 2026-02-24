import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_DEFAULT_BOH_PATH = Path(__file__).parent.parent / "boh.json"


# ---------------------------------------------------------------------------
# Rolling-hash triple-repetition detector
# ---------------------------------------------------------------------------

class TripleRepeatDetector:
    """Online detector for substrings repeated 3+ consecutive times.

    Algorithm
    ---------
    Core observation: if a triple repetition ends at position ``n-1``, its
    last third must end exactly at ``n``.  So after appending each character
    we only need to enumerate candidate period lengths ``L`` (1 … MAX_PERIOD)
    and check whether ``text[n-3L:n-2L] == text[n-2L:n-L] == text[n-L:n]``.

    Each segment comparison is O(1) via Rabin-Karp prefix hashes, giving
    O(MAX_PERIOD) per character = effectively O(1) amortised.

    Memory is bounded: when the internal buffer exceeds ``_TRIM_AT`` chars,
    it is truncated to the last ``MAX_PERIOD * 3`` chars and the hash tables
    are rebuilt in O(MAX_PERIOD) time.
    """

    BASE: int = 131
    MOD: int = (1 << 61) - 1   # 8th Mersenne prime — collision prob ≈ 1/2^61
    MAX_PERIOD: int = 50        # longest pattern period to detect
    _TRIM_AT: int = MAX_PERIOD * 8  # rebuild threshold

    def __init__(self) -> None:
        self._text: list[int] = []  # ordinal values of accumulated characters
        self._h: list[int] = [0]    # prefix hashes: _h[i] = hash(text[0:i])
        self._p: list[int] = [1]    # powers:        _p[i] = BASE^i % MOD

    def reset(self) -> None:
        """Discard all state (call after every context refresh)."""
        self._text = []
        self._h = [0]
        self._p = [1]

    def _hash(self, l: int, r: int) -> int:
        """O(1) Rabin-Karp hash of the slice text[l:r]."""
        return (self._h[r] - self._h[l] * self._p[r - l]) % self.MOD

    def _trim(self) -> None:
        """Retain only the last MAX_PERIOD*3 chars and rebuild prefix tables."""
        keep = self.MAX_PERIOD * 3
        self._text = self._text[-keep:]
        self._h = [0]
        self._p = [1]
        for c in self._text:
            self._h.append((self._h[-1] * self.BASE + c) % self.MOD)
            self._p.append(self._p[-1] * self.BASE % self.MOD)

    def feed(self, text: str) -> tuple[bool, Optional[str]]:
        """Append *text* one character at a time and report the first triple
        repetition found ending at any position in this batch.

        Returns ``(True, pattern)`` on detection, ``(False, None)`` otherwise.
        """
        for ch in text:
            c = ord(ch)
            self._text.append(c)
            self._h.append((self._h[-1] * self.BASE + c) % self.MOD)
            self._p.append(self._p[-1] * self.BASE % self.MOD)

            n = len(self._text)
            for L in range(1, min(n // 3, self.MAX_PERIOD) + 1):
                if (
                    self._hash(n - 3 * L, n - 2 * L)
                    == self._hash(n - 2 * L, n - L)
                    == self._hash(n - L, n)
                ):
                    pattern = "".join(chr(c) for c in self._text[n - L: n])
                    if len(self._text) > self._TRIM_AT:
                        self._trim()
                    return True, pattern

            if len(self._text) > self._TRIM_AT:
                self._trim()

        return False, None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_boh(path=None):
    resolved = Path(path) if path is not None else _DEFAULT_BOH_PATH
    if not resolved.exists():
        logger.warning(
            "[BoH] boh.json not found at '%s'. Hallucination filtering disabled.",
            resolved,
        )
        return []
    try:
        with open(resolved, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        phrases = [p for p in data.get("phrases", []) if p]
        logger.info("[BoH] Loaded %d hallucination phrase(s) from '%s'.", len(phrases), resolved)
        return phrases
    except Exception as exc:
        logger.error("[BoH] Failed to load boh.json '%s': %s", resolved, exc)
        return []


def contains_hallucination(
    text: str,
    phrases: list,
    *,
    detector: Optional[TripleRepeatDetector] = None,
    new_text: str = "",
) -> tuple[bool, Optional[str]]:
    """Return ``(True, matched)`` if a hallucination is detected.

    Two checks are performed in order:

    1. Incremental triple-repetition (only when `detector` and `new_text`
       are provided): feeds new_text into the rolling-hash detector and
       returns immediately if a substring repeated ≥ 3 consecutive times is
       found.

    2. BoH phrase substring match: checks whether text contains any of
       the known hallucination phrases as a substring.

    Pass `detector=None` (default) to skip check 1, e.g. for the buffer
    whose content is not incrementally accumulated.
    """
    # --- check 1: triple repetition (incremental) ---
    if detector is not None and new_text:
        is_repeat, pattern = detector.feed(new_text)
        if is_repeat:
            return True, f"{pattern!r} (3x repeat)"

    # --- check 2: BoH phrase substring match ---
    if not text or not phrases:
        return False, None
    for phrase in phrases:
        if phrase in text:
            return True, phrase

    return False, None
