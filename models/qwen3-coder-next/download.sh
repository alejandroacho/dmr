#!/usr/bin/env bash
# Download Qwen3-Coder-Next FP8 weights from HuggingFace.
# Model: Qwen/Qwen3-Coder-Next-FP8 (~95 GB, FP8 native quantization)
#
# The gateway serves this model directly via the HuggingFace Hub cache,
# so --local-dir is optional. Run this to pre-download and avoid the
# ~15 min wait on first container start.
#
# Usage:
#   ./download.sh                          # downloads to HF cache (default)
#   MODELS_DIR=/data/models ./download.sh  # downloads to local dir instead

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-}"

echo "Downloading Qwen/Qwen3-Coder-Next-FP8 (~95 GB)"
echo "Use Ctrl+C to pause — re-running will resume."
echo ""

if [[ -n "$MODELS_DIR" ]]; then
    DEST="$MODELS_DIR/qwen3-coder-next-fp8"
    echo "Destination: $DEST"
    hf download Qwen/Qwen3-Coder-Next-FP8 \
        --local-dir "$DEST" \
        --max-workers 8
else
    echo "Destination: HuggingFace cache (~/.cache/huggingface/hub)"
    hf download Qwen/Qwen3-Coder-Next-FP8 \
        --max-workers 8
fi

echo ""
echo "Done."
