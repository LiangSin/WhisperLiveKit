"""
Lightweight smoke-check for the translation formatting path.

This avoids importing `whisperlivekit` package-level `__init__` (which pulls optional
runtime deps like librosa). It only loads the specific modules we need.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"Failed to load spec for {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def main() -> None:
    # Provide a minimal `whisperlivekit` package so submodules can be imported
    # without executing `whisperlivekit/__init__.py` (which imports optional heavy deps).
    pkg_name = "whisperlivekit"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(REPO_ROOT / "whisperlivekit")]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg

    timed_objects = _load_module("whisperlivekit.timed_objects", "whisperlivekit/timed_objects.py")
    results_formater = _load_module("whisperlivekit.results_formater", "whisperlivekit/results_formater.py")

    State = timed_objects.State
    ASRToken = timed_objects.ASRToken
    Translation = timed_objects.Translation

    s = State()
    s.tokens = [
        ASRToken(start=0.0, end=0.5, text="hello", speaker=1),
        ASRToken(start=0.5, end=1.0, text="world", speaker=1),
    ]

    # Critical regression: allow a single TimedText (not a list) without crashing.
    s.translation_validated_segments = Translation(text="hi world", start=0.0, end=1.0)

    class Args:
        diarization = False
        disable_punctuation_split = False

    lines, _ = results_formater.format_output(s, silence=False, args=Args(), sep=" ")
    assert lines, "Expected at least one output line"
    assert lines[0].translation.strip() == "hi world", f"Unexpected translation: {lines[0].translation!r}"
    print("OK: translation formatting smoke-check passed")


if __name__ == "__main__":
    main()

