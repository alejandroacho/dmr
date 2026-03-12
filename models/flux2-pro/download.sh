#!/usr/bin/env bash
# Download FLUX.1-dev weights from HuggingFace in Diffusers format.
# Model: black-forest-labs/FLUX.1-dev (~34 GB, FP16)
#
# Requirements:
#   - HuggingFace account with access to FLUX.1-dev (accept terms at:
#     https://huggingface.co/black-forest-labs/FLUX.1-dev)
#   - huggingface-cli logged in: hf login
#
# Usage:
#   ./download.sh                          # downloads to default MODELS_DIR
#   MODELS_DIR=/data/models ./download.sh  # custom path

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/home/alejandroacho/Models}"
DEST="$MODELS_DIR/flux2-pro-fp16"

echo "Downloading black-forest-labs/FLUX.1-dev → $DEST"
echo "Size: ~34 GB. Use Ctrl+C to pause — re-running will resume."
echo ""

hf download black-forest-labs/FLUX.1-dev \
    --local-dir "$DEST" \
    --max-workers 8

echo ""
echo "Done. Weights saved to: $DEST"
