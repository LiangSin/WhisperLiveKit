#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/run_benchmark_preset.sh --<preset> [additional args...]

Presets:
  --libri, --libri-debug, --youtube, --youtube-debug, --translate, --translate-debug

Any extra arguments after the preset flag are forwarded to benchmarking/run_benchmark.py.
EOF
  exit 1
}

if [[ $# -eq 0 ]]; then
  usage
fi

preset=""
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --libri|--libri-debug|--youtube|--youtube-debug|--translate|--translate-debug)
      if [[ -n "$preset" ]]; then
        echo "Multiple presets specified. Choose only one." >&2
        usage
      fi
      preset="$1"
      ;;
    -h|--help)
      usage
      ;;
    *)
      extra_args+=("$1")
      ;;
  esac
  shift
done

if [[ -z "$preset" ]]; then
  echo "You must specify a preset (e.g., --libri)." >&2
  usage
fi

mkdir -p "$REPO_ROOT/benchmarking/results"

declare -a preset_args
case "$preset" in
  --libri)
    preset_args=(
      --dataset_path dataset/LibriSpeech/dev-clean
      --dataset_class LibriSpeechDataset
      --output benchmarking/results/libri-dev-clean.json
    )
    ;;
  --libri-debug)
    preset_args=(
      --dataset_path dataset/LibriSpeech/dev-clean-debug
      --dataset_class LibriSpeechDataset
      --output benchmarking/results/libri-dev-clean-debug.json
    )
    ;;
  --youtube)
    preset_args=(
      --dataset_path dataset/youtube_data/@NTUOCW
      --dataset_class YoutubeDataset
      --output benchmarking/results/youtube.json
    )
    ;;
  --youtube-debug)
    preset_args=(
      --dataset_path dataset/PLCX-BLZ1hDpDOgZPSmdMcpgfO5uQ0i4XK/test1
      --dataset_class YoutubeDataset
      --output benchmarking/results/youtube-debug.json
      --url wss://localhost:8000/asr
    )
    ;;
  --translate)
    preset_args=(
      --translate
      --dataset_path dataset/youtube_data/debug/@NTUOCW/PLCX-BLZ1hDpDOgZPSmdMcpgfO5uQ0i4XK
      --dataset_class YoutubeDataset
      --output benchmarking/results/translate-debug.json
    )
    ;;
  --translate-debug)
    preset_args=(
      --translate
      --dataset_path dataset/youtube_data/debug/@NTUOCW/PLCX-BLZ1hDpDOgZPSmdMcpgfO5uQ0i4XK/test1
      --dataset_class YoutubeDataset
      --output benchmarking/results/translate-debug.json
    )
    ;;
esac

echo "Running benchmarking/run_benchmark.py with preset $preset"
if [[ ${#extra_args[@]} -gt 0 ]]; then
  echo "Forwarding extra arguments: ${extra_args[*]}"
fi

python benchmarking/run_benchmark.py "${preset_args[@]}" "${extra_args[@]}"
