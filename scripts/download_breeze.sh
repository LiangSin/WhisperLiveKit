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

error() {
  echo -e "${BOLD}${RED}[SETUP] $*${RESET}" 1>&2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TARGET_LINK="$REPO_ROOT/models/breeze"
if [[ ! -e "$TARGET_LINK" ]]; then
  info "models/breeze not found. Downloading model artifacts..."
  if ! command -v hf >/dev/null 2>&1; then
    error "The 'hf' CLI was not found in your PATH. Install huggingface_hub to obtain it."
    exit 1
  fi

  DOWNLOAD_DIR="$(python - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo = "MediaTek-Research/Breeze-ASR-25"
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
  success "Created symlink models/breeze -> $DOWNLOAD_DIR"
else
  info "models/breeze already exists. Skipping download."
fi