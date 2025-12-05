#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/start_server.sh --cuda <id> --model <model_path>

Required arguments:
  --cuda      GPU id for CUDA_VISIBLE_DEVICES
  --model     Model path to pass to whisperlivekit-server
EOF
  exit 1
}

cuda_id=""
model_path=""

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
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
  shift
done

[[ -n "$cuda_id" ]] || usage
[[ -n "$model_path" ]] || usage

export CUDA_VISIBLE_DEVICES="$cuda_id"
echo "CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%m%d-%H%M")"
LOG_FILE="$LOG_DIR/$TIMESTAMP.log"

echo "Starting whisperlivekit-server with model: $model_path"
echo "Logs will be written to $LOG_FILE"
whisperlivekit-server --model-path "$model_path" > "$LOG_FILE" 2>&1 &
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
