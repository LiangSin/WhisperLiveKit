#!/bin/bash

# Download the LibriSpeech dataset - dev-clean
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "${REPO_ROOT}/dataset"
cd "${REPO_ROOT}/dataset"

echo "Downloading LibriSpeech dataset - dev-clean..."
wget https://openslr.trmal.net/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz

echo "LibriSpeech dataset - dev-clean downloaded successfully."