#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/start_server.sh --cuda <id>

Required arguments:
  --cuda      GPU id for CUDA_VISIBLE_DEVICES

Optional arguments:
  --model     Model path to pass to whisperlivekit-server (default: models/cool-whisper)
  --vac       VAC backend to use: 'silero' (default, requires torch + model files) or 'ten-vad' (default: ten-vad).
  --lang      Source language code, e.g. en,de,cs, or 'auto' for language detection (default: zh).
  --target-lang     Target language for translation.
  --translation-model  Translation model: 'nllw' (default) or 'translategemma'.
  --translation-model-size  Model size string, e.g. '600M'/'1.3B' for nllw, '4b'/'12b'/'27b' for translategemma.
  --translategemma-url  Base URL of the standalone TranslateGemma vLLM service (default: http://localhost:8765/v1).
                        Only used when --translation-model translategemma is selected.
                        See translate-gemma/README.md for how to launch the service.
  --sentence-detection  Enable SaT sentence boundary detection (requires pip install wtpsplit).
                        Automatically enabled when --translation-model translategemma is used.
  --bind      Bind host address
  --allowed-ips <ips>           Comma-separated list of allowed IP addresses
                                Example: "192.168.1.100,10.0.0.50"
  --allowed-networks <networks> Comma-separated list of allowed IP networks in CIDR notation
                                Example: "192.168.1.0/24,140.112.91.0/24"
EOF
  exit 1
}

cuda_id=""
model_path=""
allowed_ips=""
allowed_networks=""
bind_host=""
vac_backend=""
language=""
target_language=""
translation_model=""
translation_model_size=""
translategemma_url=""
sentence_detection=0
passthrough_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      shift
      [[ $# -gt 0 ]] || usage
      cuda_id="$1"
      ;;
    --model)
      shift
      [[ $# -gt 0 ]] || usage
      model_path="$1"
      ;;
    --allowed-ips)
      shift
      [[ $# -gt 0 ]] || usage
      allowed_ips="$1"
      ;;
    --allowed-networks)
      shift
      [[ $# -gt 0 ]] || usage
      allowed_networks="$1"
      ;;
    --bind)
      shift
      [[ $# -gt 0 ]] || usage
      bind_host="$1"
      ;;
    --vac)
      shift
      [[ $# -gt 0 ]] || usage
      vac_backend="$1"
      ;;
    --lang)
      shift
      [[ $# -gt 0 ]] || usage
      language="$1"
      ;;
    --target-lang)
      shift
      [[ $# -gt 0 ]] || usage
      target_language="$1"
      ;;
    --translation-model)
      shift
      [[ $# -gt 0 ]] || usage
      translation_model="$1"
      ;;
    --translation-model-size)
      shift
      [[ $# -gt 0 ]] || usage
      translation_model_size="$1"
      ;;
    --translategemma-url)
      shift
      [[ $# -gt 0 ]] || usage
      translategemma_url="$1"
      ;;
    --sentence-detection)
      sentence_detection=1
      ;;
    *)
      # Unknown args should be passed through to whisperlivekit-server
      passthrough_args+=("$1")
      ;;
  esac
  shift
done

[[ -n "$cuda_id" ]] || usage

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$cuda_id"
echo "CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES"

cmd_args=()

if [[ -n "$model_path" ]]; then
  cmd_args+=(--model-path $model_path)
else
  cmd_args+=(--model-path models/cool-whisper)
fi

if [[ -n "$vac_backend" ]]; then
  cmd_args+=(--vac-backend $vac_backend)
else
  cmd_args+=(--vac-backend ten-vad)
fi

if [[ -n "$language" ]]; then
  cmd_args+=(--language $language)
else
  cmd_args+=(--language zh)
fi

if [[ -n "$target_language" ]]; then
  cmd_args+=(--target-language $target_language)
fi

if [[ -n "$translation_model" ]]; then
  cmd_args+=(--translation-model $translation_model)
fi

if [[ -n "$translation_model_size" ]]; then
  cmd_args+=(--translation-model-size $translation_model_size)
fi

if [[ -n "$translategemma_url" ]]; then
  cmd_args+=(--translategemma-url $translategemma_url)
fi

if [[ "$sentence_detection" -eq 1 ]]; then
  cmd_args+=(--sentence-detection)
fi

if [[ -n "$allowed_ips" ]]; then
  cmd_args+=(--allowed-ips $allowed_ips)
fi
if [[ -n "$allowed_networks" ]]; then
  cmd_args+=(--allowed-networks $allowed_networks)
fi
if [[ -n "$bind_host" ]]; then
  cmd_args+=(--host $bind_host --ssl-certfile ssl-config/cert.pem --ssl-keyfile ssl-config/key.pem)
fi
if [[ ${#passthrough_args[@]} -gt 0 ]]; then
  cmd_args+=("${passthrough_args[@]}")
fi

# Suppress noisy Caffe2/NNPACK warnings (AMD CPUs often trigger these)
export GLOG_minloglevel=${GLOG_minloglevel:-2}
export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-ERROR}

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%m%d-%H%M")"
LOG_FILE="$LOG_DIR/$TIMESTAMP.log"

echo "Starting whisperlivekit-server with model: $model_path"
echo "Logs will be written to $LOG_FILE"
whisperlivekit-server "${cmd_args[@]}" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "Press Ctrl+C to stop the server."

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

trap cleanup EXIT INT TERM

wait "$SERVER_PID"