#!/usr/bin/env bash
set -euo pipefail

# Simple color helpers so our messages stand out from verbose command output
RESET='\033[0m'
BOLD='\033[1m'
BLUE='\033[34m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'

info() {
  # General informational message
  echo -e "${BOLD}${BLUE}[SETUP]${RESET} $*"
}

success() {
  echo -e "${BOLD}${GREEN}[SETUP] $*${RESET}"
}

warn() {
  echo -e "${BOLD}${YELLOW}[SETUP] $*${RESET}"
}

error() {
  echo -e "${BOLD}${RED}[SETUP] $*${RESET}" 1>&2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

info "This script should be run inside your target environment or virtual environment."
printf "${BOLD}${BLUE} Continue? [Y|n] ${RESET}"
response=""
read -r response
response="${response:-Y}"
if [[ "$response" =~ ^([Nn]|[Nn][Oo])$ ]]; then
  warn "Aborting setup."
  exit 1
fi

info "Installing WhisperLiveKit in editable mode..."
pip install -e .

info "Installing benchmarking dependencies..."
pip install -r benchmarking/requirements.txt

info "Installing additional dependencies..."
pip install safetensors faster_whisper huggingface_hub yt-dlp

info "Checking for ffmpeg..."
if ! command -v ffmpeg >/dev/null 2>&1; then
  info "ffmpeg not found in PATH. Attempting to install via conda..."
  if ! command -v conda >/dev/null 2>&1; then
    error "conda is not available; cannot install ffmpeg automatically. Please install ffmpeg manually and re-run this script."
    exit 1
  fi
  conda install -y ffmpeg
  success "ffmpeg installed via conda."
else
  info "ffmpeg found in PATH. Skipping conda installation."
fi

TARGET_LINK="$REPO_ROOT/models/cool-whisper"
if [[ ! -e "$TARGET_LINK" ]]; then
  info "models/cool-whisper not found. Downloading model artifacts..."
  if ! command -v hf >/dev/null 2>&1; then
    error "The 'hf' CLI was not found in your PATH. Install huggingface_hub to obtain it."
    exit 1
  fi

  info "Logging into Hugging Face. You will be prompted for your access token."
  hf auth login

  DOWNLOAD_DIR="$(python - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo = "andybi7676/cool-whisper-hf"
filenames = ["model.safetensors", "config.json"]
paths = [hf_hub_download(repo_id=repo, filename=name) for name in filenames]
parent = os.path.dirname(paths[0])
if any(os.path.dirname(path) != parent for path in paths[1:]):
    raise SystemExit("Downloaded files ended up in different directories.")
print(parent, end="")
PY
)"

  mkdir -p "$REPO_ROOT/models"
  ln -s "$DOWNLOAD_DIR" "$TARGET_LINK"
  success "Created symlink models/cool-whisper -> $DOWNLOAD_DIR"
else
  info "models/cool-whisper already exists. Skipping download."
fi

success "Environment setup complete."
