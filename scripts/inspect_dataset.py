#!/usr/bin/env python3
"""
Offline dataset health checker for WhisperLiveKit benchmarking datasets.

This script does NOT run automatically; run it manually in your env.
It will:
  - Load the specified DatasetClass (LibriSpeechDataset / YoutubeDataset / etc.)
  - For each sample, probe audio metadata (ffprobe)
  - Check transcript text length/emptiness
  - Emit a JSON report summarizing anomalies and basic stats.

Example:
  python benchmarking/check_dataset_health.py \
      --dataset_path dataset/youtube_data/debug \
      --dataset_class YoutubeDataset \
      --max_samples 200 \
      --output /tmp/youtube_health.json

  python benchmarking/check_dataset_health.py \
      --dataset_path /path/to/LibriSpeech/test-clean \
      --dataset_class LibriSpeechDataset \
      --output /tmp/libri_health.json
"""
import argparse
import json
import os
import subprocess
import sys
from typing import Dict, Any, List

# Ensure DatasetClass/ is importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, "benchmarking"))
import DatasetClass as ds_module  # type: ignore


def ffprobe_info(path: str) -> Dict[str, Any]:
    """Return ffprobe info (duration, sample_rate, channels)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_streams",
        "-of",
        "json",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", errors="ignore"))
    except Exception as e:
        return {"error": str(e)}

    fmt = data.get("format", {})
    streams = data.get("streams", [])
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {
        "duration": float(fmt.get("duration")) if fmt.get("duration") else None,
        "sample_rate": int(audio_stream.get("sample_rate"))
        if audio_stream and audio_stream.get("sample_rate")
        else None,
        "channels": int(audio_stream.get("channels"))
        if audio_stream and audio_stream.get("channels")
        else None,
    }


def analyze_dataset(dataset_path: str, dataset_class_name: str, max_samples: int = None):
    try:
        DatasetCls = getattr(ds_module, dataset_class_name)
    except AttributeError:
        raise SystemExit(f"Dataset class '{dataset_class_name}' not found in DatasetClass/")

    ds_instance = DatasetCls(dataset_path)
    results: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []

    for idx, chapter_samples in enumerate(ds_instance):
        if max_samples is not None and idx >= max_samples:
            break
        if not chapter_samples:
            anomalies.append({"type": "empty_chapter", "chapter_index": idx})
            continue

        group_id = ds_instance.get_group_id(chapter_samples)
        for sample in chapter_samples:
            audio_path = sample.get("audio_path")
            text = sample.get("text", "")
            entry: Dict[str, Any] = {
                "group_id": group_id,
                "sample_id": sample.get("id", ""),
                "audio_path": audio_path,
                "text_len": len(text.strip()) if isinstance(text, str) else 0,
            }

            # Transcript checks
            if not text or len(str(text).strip()) == 0:
                anomalies.append(
                    {"type": "empty_transcript", "group_id": group_id, "sample_id": sample.get("id", "")}
                )

            if not audio_path or not os.path.exists(audio_path):
                anomalies.append(
                    {"type": "missing_audio", "group_id": group_id, "sample_id": sample.get("id", ""), "path": audio_path}
                )
                entry["probe"] = {"error": "missing"}
                results.append(entry)
                continue

            probe = ffprobe_info(audio_path)
            entry["probe"] = probe

            results.append(entry)

    summary = {
        "dataset_class": dataset_class_name,
        "dataset_path": dataset_path,
        "total_chapters": len(ds_instance),
        "checked_chapters": min(len(ds_instance), max_samples) if max_samples else len(ds_instance),
        "total_samples": len(results),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "samples": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Check dataset health for WhisperLiveKit benchmarks.")
    parser.add_argument("--dataset_path", required=True, help="Root path of the dataset.")
    parser.add_argument("--dataset_class", required=True, help="Dataset class name (e.g., YoutubeDataset, LibriSpeechDataset).")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of chapters to check.")
    parser.add_argument("--output", required=True, help="Path to write JSON report.")
    args = parser.parse_args()

    summary = analyze_dataset(
        dataset_path=args.dataset_path,
        dataset_class_name=args.dataset_class,
        max_samples=args.max_samples,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Report written to {args.output}")
    print(f"Anomalies detected: {summary['anomaly_count']}")


if __name__ == "__main__":
    main()



