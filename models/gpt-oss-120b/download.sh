#!/usr/bin/env bash
# Download GPT-OSS 120B weights from HuggingFace.
# Model: openai/gpt-oss-120b (~240 GB, MXFP4 quantized by OpenAI for GB10)
#
# Usage:
#   ./download.sh                          # downloads to default MODELS_DIR
#   MODELS_DIR=/data/models ./download.sh  # custom path

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/home/alejandroacho/Models}"
DEST="$MODELS_DIR/gpt-oss-120b-q8"

echo "Downloading openai/gpt-oss-120b → $DEST"
echo "Size: ~240 GB. Use Ctrl+C to pause — re-running will resume."
echo ""

hf download openai/gpt-oss-120b \
    --local-dir "$DEST" \
    --max-workers 8

echo ""
echo "Done. Weights saved to: $DEST"
