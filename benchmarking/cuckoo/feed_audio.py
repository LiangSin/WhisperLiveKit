#!/usr/bin/env python3
"""
Feed a WAV file into BlackHole 2ch virtual microphone.

Usage:
    python feed_audio.py /path/to/audio.wav

Prerequisites:
    brew install blackhole-2ch ffmpeg
"""
import re
import sys
import subprocess
from pathlib import Path


def get_blackhole_index() -> int:
    """Find the Core Audio index of BlackHole 2ch (matches AVFoundation audio list)."""
    result = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True
    )
    # Device list printed in stderr
    in_audio_section = False
    for line in result.stderr.splitlines():
        if "AVFoundation audio" in line:
            in_audio_section = True
        if in_audio_section and "BlackHole 2ch" in line:
            m = re.search(r'\[(\d+)\]', line)
            if m:
                return int(m.group(1))

    raise RuntimeError(
        "BlackHole 2ch not found.\n"
        "Please ensure it is installed: brew install blackhole-2ch\n"
        "After installation, please restart the machine."
    )


def feed(wav_path: Path):
    index = get_blackhole_index()
    print(f"✔ BlackHole 2ch device index: {index}")
    print(f"▶ Start playing: {wav_path.name}")
    print("  (After playback, the program will automatically end, please manually press End on the Cuckoo web page)\n")

    # macOS: AVFoundation is input-only in FFmpeg; playback to a device uses audiotoolbox.
    proc = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-ar", "16000",  # Resample to 16 kHz (common requirement for ASR)
        "-ac", "1",      # Single channel
        "-acodec", "pcm_s16le",
        "-f", "audiotoolbox",
        "-audio_device_index", str(index),
        "/dev/null",  # Output URL ignored; audio goes to the device above
    ])

    if proc.returncode == 0:
        print("\n✅ Playback completed")
    else:
        print(f"\n❌ ffmpeg error (returncode={proc.returncode})")
        print("    Please execute the following command to check the device list:")
        print('   ffmpeg -f avfoundation -list_devices true -i "" 2>&1')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python feed_audio.py /path/to/audio.wav")
        sys.exit(1)

    wav_path = Path(sys.argv[1])
    if not wav_path.exists():
        print(f"❌ File not found: {wav_path}")
        sys.exit(1)
    if wav_path.suffix.lower() != ".wav":
        print(f"⚠  File extension is not .wav, still trying to play...")

    feed(wav_path)