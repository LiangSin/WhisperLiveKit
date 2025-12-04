#!/usr/bin/env bash

# Download audio and CC subtitles from a YouTube channel/playlist
# using yt-dlp + FFmpeg, and organize them in a structured format.
#
# Dependencies:
#   - yt-dlp  (https://github.com/yt-dlp/yt-dlp)
#   - ffmpeg  (for audio conversion and subtitle processing)
#
# Usage:
#   Edit the CHANNEL_URL variable below, then run:
#     ./download_youtube_channel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_DIR="${REPO_ROOT}/dataset"
mkdir -p "${DATASET_DIR}"
cd "${DATASET_DIR}"

# ============================================
# Configuration: Edit the channel URL here
# ============================================
CHANNEL_URLS=(
  "https://www.youtube.com/watch?v=P0QGcyEhFsw&list=PLCX-BLZ1hDpBjBFXoUat1lakxrPYp2bRd"
  "https://www.youtube.com/watch?v=TfkgUaXgMLo&list=PLCX-BLZ1hDpAparCQIfqgmjVkLnX9pPZ4"
  "https://www.youtube.com/watch?v=fzS3jBov86w&list=PLCX-BLZ1hDpDojkD7e92LfwTT_x6ierVN"
  "https://www.youtube.com/watch?v=imK5_767Z_4&list=PLCX-BLZ1hDpDyJhcvZNaGcomSTAgfPfKC"
  "https://www.youtube.com/watch?v=Wiv1dUsPihA&list=PLCX-BLZ1hDpCxlZxPO6RRTcwauTl683Vt"
  "https://www.youtube.com/watch?v=gp8_fqK0FGE&list=PLCX-BLZ1hDpA0mvnbfYLBvbKJYYtWgks-"
  "https://www.youtube.com/watch?v=y-IKSJqiBlo&list=PLCX-BLZ1hDpCFVMNsDEymt5gxVjhXq2XY"
  "https://www.youtube.com/watch?v=HTK8_8mEZWM&list=PLCX-BLZ1hDpA7xbmlovu7zzdM8oefFm1q"
  "https://www.youtube.com/watch?v=yOoru8zZZIs&list=PLCX-BLZ1hDpDOgZPSmdMcpgfO5uQ0i4XK"
  "https://www.youtube.com/watch?v=BI_ywtB3wLY&list=PLCX-BLZ1hDpCWfv4UrMiAX4qGg9rC0W6m"
  "https://www.youtube.com/watch?v=_oD4AKCq2Tw&list=PLfjnaD_kBFeSkLN7mpE3Ua5-3Kt2PJuOt"
  "https://www.youtube.com/watch?v=QLiKmca4kzI&list=PLJV_el3uVTsNZEFAdQsDeOdzAaHTca2Gi"
  "https://www.youtube.com/watch?v=AVIKFXLCPY8&list=PLJV_el3uVTsPz6CTopeRp2L2t4aL_KgiI"
  "https://www.youtube.com/watch?v=7wMYhPUGy40&list=PLXVfgk9fNX2LdYV_KJH9LtbqTzg7r_CN9"
  # "https://www.youtube.com/@NTUOCW"
  # "https://www.youtube.com/@NTUSpeech"
)

# Output directory (default: dataset/youtube_data)
OUTPUT_DIR="${DATASET_DIR}/youtube_data"

# ============================================
# Check dependencies
# ============================================
if ! command -v yt-dlp >/dev/null 2>&1; then
  echo "Error: yt-dlp not found. Please install it first."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found. Please install it first."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  Downloading YouTube Channel/Playlist"
echo "----------------------------------------"
echo "  Output Dir  : $OUTPUT_DIR"
echo "========================================"

# Notes:
#  - --yes-playlist: Download the playlist, if the URL refers to a video and a playlist
#  - --extract-audio: Extract audio only
#  - --audio-format wav: Convert to WAV format (suitable for ASR/training)
#  - --audio-quality 0: Best quality
#  - --write-subs --write-auto-subs: Download both manual and auto-generated subtitles
#  - --sub-langs: Specify language (e.g., zh-Hant, en,*)
#  - --sub-format vtt: Convert to VTT format
#  - --output: Use uploader_id + video_id + title for organized filenames
#
# Output structure:
#   ${OUTPUT_DIR}/<uploader_id>/<video_id>_<title>.wav
#   ${OUTPUT_DIR}/<uploader_id>/<video_id>_<title>.<lang>.vtt
#   ${OUTPUT_DIR}/<uploader_id>/<video_id>_<title>.info.json

for CHANNEL_URL in "${CHANNEL_URLS[@]}"; do
  echo
  echo "========================================"
  echo "  Source URL  : $CHANNEL_URL"
  echo "========================================"
  yt-dlp \
    --yes-playlist \
    --extract-audio \
    --audio-format wav \
    --audio-quality 0 \
    --write-subs \
    --write-auto-subs \
    --sub-langs zh-TW \
    --sub-format vtt \
    --paths "$OUTPUT_DIR" \
    --output "%(uploader_id)s/%(playlist_id)s/%(id)s.%(ext)s" \
    -- "$CHANNEL_URL"
done


echo "\033[1m\033[32mDownload completed.\033[0m Check output directory: $OUTPUT_DIR"

