"""Time-window alignment of translation output against reference subtitles.

Used by run_benchmark.py when --translate is set: the committed translation
lines (with their audio timestamps) and the video's reference subtitles are
bucketed into fixed windows, producing line-aligned hyp/ref/src files that
calculate_translate_metrics.py can score at segment level. Whole-lecture
lines overflow COMET's 512-token encoder window; ~60s windows do not.
"""

import os
import re

DEFAULT_WINDOW_SECONDS = 60.0

_TIME_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)


def parse_srt_timed(path):
    """Parse an SRT file into a list of (start_sec, text) entries.

    Applies the same bracket cleanup as YoutubeDataset._parse_srt so the
    windowed references match the whole-file references.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()
    entries = []
    for raw in re.split(r"\n\s*\n", content):
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        m = None
        text_start = None
        for i, line in enumerate(lines[:2]):
            m = _TIME_RE.match(line)
            if m:
                text_start = i + 1
                break
        if not m or text_start is None or text_start >= len(lines):
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        text = " ".join(lines[text_start:])
        for pattern in (r"\[.*?\]", r"\(.*?\)", r"【.*?】"):
            text = re.sub(pattern, "", text)
        text = " ".join(text.split())
        if text:
            entries.append((start, text))
    return entries


def find_reference_srts(audio_path):
    """Locate (src_srt, ref_srt) next to the audio file, or (None, None).

    Follows YoutubeDataset's convention: the 'zh' subtitle is the source
    transcript, the 'en' subtitle is the translation reference.
    """
    directory = os.path.dirname(audio_path)
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    src_path = ref_path = None
    try:
        candidates = [
            f for f in os.listdir(directory)
            if f.startswith(basename) and f.endswith(".srt")
        ]
    except OSError:
        return None, None
    for f in candidates:
        lower = f.lower()[len(basename):]
        if "zh" in lower and src_path is None:
            src_path = os.path.join(directory, f)
        elif "en" in lower and ref_path is None:
            ref_path = os.path.join(directory, f)
    return src_path, ref_path


def _bucket(entries, window):
    """Group (start_sec, text) entries into {window_index: joined_text}."""
    buckets = {}
    for start, text in entries:
        w = int(start // window)
        buckets[w] = (buckets[w] + " " + text).strip() if w in buckets else text
    return buckets


def aligned_windows(audio_path, hyp_segments, window=DEFAULT_WINDOW_SECONDS):
    """Build aligned (src, hyp, ref) window texts for one video.

    hyp_segments: list of (start_sec, translation_text) from committed lines.
    Returns a list of (src, hyp, ref) tuples covering every window where all
    three sides have text; returns [] when reference subtitles are missing.
    """
    src_path, ref_path = find_reference_srts(audio_path)
    if not src_path or not ref_path:
        return []
    src_b = _bucket(parse_srt_timed(src_path), window)
    ref_b = _bucket(parse_srt_timed(ref_path), window)
    hyp_b = _bucket([(s, " ".join(t.split())) for s, t in hyp_segments if t], window)
    triples = []
    for w in sorted(ref_b):
        if w in src_b and w in hyp_b:
            triples.append((src_b[w], hyp_b[w], ref_b[w]))
    return triples
