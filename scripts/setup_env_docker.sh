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

info "Installing hugginface hub..."
pip install huggingface_hub


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

# CTranslate2 encoder weights: with model.bin + vocabulary.json present next to
# the PyTorch weights (2-4x faster than the PyTorch encoder).
MODEL_DIR="$(readlink -f "$TARGET_LINK")"
if [[ -f "$MODEL_DIR/model.bin" ]]; then
  info "CTranslate2 weights already exist in $MODEL_DIR. Skipping conversion."
else
  info "Converting cool-whisper to CTranslate2 format (float16)..."
  if ! command -v ct2-transformers-converter >/dev/null 2>&1; then
    info "Installing ctranslate2 + transformers for the conversion..."
    pip install ctranslate2 transformers
  fi
  CT2_TMP_DIR="$(mktemp -d)"
  # The converter refuses to write into a non-empty directory, so convert to a
  # temp dir and move only the CT2 artifacts next to the PyTorch weights.
  if ct2-transformers-converter --model "$MODEL_DIR" \
      --output_dir "$CT2_TMP_DIR/cool-whisper-ct2" --quantization float16; then
    mv "$CT2_TMP_DIR/cool-whisper-ct2/model.bin" \
       "$CT2_TMP_DIR/cool-whisper-ct2/vocabulary.json" "$MODEL_DIR/"
    rm -rf "$CT2_TMP_DIR"
    success "CTranslate2 weights installed in $MODEL_DIR"
  else
    rm -rf "$CT2_TMP_DIR"
    warn "CTranslate2 conversion failed; the server will fall back to the PyTorch encoder."
  fi
fi

info "Checking for SSL certificates..."
if [[ ! -f "ssl-config/key.pem" ]] || [[ ! -f "ssl-config/cert.pem" ]]; then
  info "SSL certificates not found. Generating self-signed certificates..."
  mkdir -p ssl-config
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ssl-config/key.pem -out ssl-config/cert.pem -subj "/CN=whisperlivekit"
  success "SSL certificates generated in ssl-config/"
else
  info "SSL certificates already exist. Skipping generation."
fi

success "Environment setup complete."
