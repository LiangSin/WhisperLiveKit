#!/usr/bin/env python3
"""
Record audio from BlackHole 2ch for verification.

Usage:
    python benchmarking/cuckoo/record_blackhole.py
    python benchmarking/cuckoo/record_blackhole.py --seconds 10 --output blackhole_check.wav

Tip:
    Start this recorder first, then run feed_audio.py in another terminal.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path


def get_blackhole_index() -> int:
    """Find the index of BlackHole 2ch in AVFoundation audio devices."""
    result = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    in_audio_section = False
    for line in result.stderr.splitlines():
        if "AVFoundation audio" in line:
            in_audio_section = True
            continue
        if in_audio_section and "BlackHole 2ch" in line:
            match = re.search(r"\[(\d+)\]", line)
            if match:
                return int(match.group(1))

    raise RuntimeError(
        "BlackHole 2ch not found.\n"
        "Please ensure it is installed: brew install blackhole-2ch\n"
        "After installation, please restart the machine."
    )


def record(seconds: int, output: Path) -> int:
    idx = get_blackhole_index()
    print(f"✔ BlackHole 2ch device index: {idx}")
    print(f"▶ Recording {seconds}s from BlackHole -> {output}")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "avfoundation",
        "-video_device_index",
        "-1",
        "-audio_device_index",
        str(idx),
        "-i",
        "",
        "-t",
        str(seconds),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output),
    ]
    def _emit(data: object) -> None:
        if not data:
            return
        if isinstance(data, bytes):
            print(data.decode("utf-8", errors="replace"), end="")
            return
        print(str(data), end="")

    proc = subprocess.run(cmd)

    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record from BlackHole 2ch.")
    parser.add_argument(
        "--seconds",
        type=int,
        default=10,
        help="Recording length in seconds (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarking/cuckoo/blackhole_check.wav"),
        help="Output WAV path (default: benchmarking/cuckoo/blackhole_check.wav).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seconds <= 0:
        print("❌ --seconds must be greater than 0")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    code = record(args.seconds, args.output)
    if code == 0:
        print(f"\n✅ Saved recording to: {args.output}")
    else:
        print(f"\n❌ ffmpeg failed (returncode={code})")
        print('   You can inspect devices with: ffmpeg -f avfoundation -list_devices true -i "" 2>&1')
    sys.exit(code)
