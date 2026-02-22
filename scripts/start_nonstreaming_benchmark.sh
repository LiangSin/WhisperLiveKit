#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/start_nonstreaming_benchmark.sh --<preset> --model <model_path> [additional args...]

Required arguments:
  --model     Path or Hugging Face model name for Whisper model

Presets:
  --libri, --libri-debug, --youtube, --youtube-debug

Optional arguments:
  --device    Device to run on (auto, cpu, cuda:0, etc.) [default: auto]
  --batch-size Batch size for processing [default: 1]

Any extra arguments after the preset flag are forwarded to benchmarking/run_nonstreaming_benchmark.py.
EOF
  exit 1
}

if [[ $# -lt 2 ]]; then
  usage
fi

preset=""
model_path=""
device="auto"
batch_size="1"
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --libri|--libri-debug|--youtube|--youtube-debug)
      if [[ -n "$preset" ]]; then
        echo "Multiple presets specified. Choose only one." >&2
        usage
      fi
      preset="$1"
      ;;
    --model)
      shift
      [[ $# -gt 0 ]] || usage
      model_path="$1"
      ;;
    --device)
      shift
      [[ $# -gt 0 ]] || usage
      device="$1"
      ;;
    --batch-size)
      shift
      [[ $# -gt 0 ]] || usage
      batch_size="$1"
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

if [[ -z "$model_path" ]]; then
  echo "You must specify a model path with --model." >&2
  usage
fi

mkdir -p "$REPO_ROOT/benchmarking/results"

declare -a preset_args
case "$preset" in
  --libri)
    preset_args=(
      --dataset_path dataset/LibriSpeech/dev-clean
      --dataset_class LibriSpeechDataset
      --output benchmarking/results/libri-dev-clean-nonstreaming.json
    )
    ;;
  --libri-debug)
    preset_args=(
      --dataset_path dataset/LibriSpeech/dev-clean-debug
      --dataset_class LibriSpeechDataset
      --output benchmarking/results/libri-dev-clean-debug-nonstreaming.json
    )
    ;;
  --youtube)
    preset_args=(
      --dataset_path dataset/youtube_data/@NTUOCW
      --dataset_class YoutubeDataset
      --output benchmarking/results/youtube-nonstreaming.json
    )
    ;;
  --youtube-debug)
    preset_args=(
      --dataset_path dataset/youtube_data/debug
      --dataset_class YoutubeDataset
      --output benchmarking/results/youtube-debug-nonstreaming.json
    )
    ;;
esac

echo "Running benchmarking/run_nonstreaming_benchmark.py with preset $preset"
echo "Model: $model_path"
echo "Device: $device"
if [[ ${#extra_args[@]} -gt 0 ]]; then
  echo "Forwarding extra arguments: ${extra_args[*]}"
fi

python benchmarking/run_nonstreaming_benchmark.py \
  "${preset_args[@]}" \
  --model_path "$model_path" \
  --device "$device" \
  --batch_size "$batch_size" \
  "${extra_args[@]}"


