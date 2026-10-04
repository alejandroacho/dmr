#!/usr/bin/env bash
# Download HiDream-O1-Image weights (ComfyUI repackaged) for the media node.
# Repo: Comfy-Org/HiDream-O1-Image (public, no gating)
#
# Text-to-image, native canvas 2048x2048 (see EmptyHiDreamO1LatentImage).
# These are all-in-one checkpoints — model, text encoder and VAE in one file —
# so they live under checkpoints/ and load with CheckpointLoaderSimple.
#
#   hidream_o1_image_fp8_scaled       7.51 GiB   base
#   hidream_o1_image_dev_fp8_scaled   7.51 GiB   dev
#                                    ─────────
#                                    15.02 GiB  (~16.1 GB)
#
# Both variants are fetched so they can be compared; drop one from the list to
# halve the download. Higher-quality options in the same repo, if VRAM allows:
# *_bf16 (15.24 GiB each) or *_mxfp8 (8.31 GiB, native FP8 on Blackwell).
#
# Usage:
#   ./download.sh                          # downloads to default MODELS_DIR
#   MODELS_DIR=/data/models ./download.sh  # custom path

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/home/alejandroacho/Models}"
DEST="$MODELS_DIR/hidream-o1"

export HF_XET_HIGH_PERFORMANCE=1

echo "Downloading Comfy-Org/HiDream-O1-Image → $DEST"
echo "Size: ~16 GB. Use Ctrl+C to pause — re-running will resume."
echo ""

hf download Comfy-Org/HiDream-O1-Image \
    checkpoints/hidream_o1_image_fp8_scaled.safetensors \
    checkpoints/hidream_o1_image_dev_fp8_scaled.safetensors \
    --local-dir "$DEST" \
    --max-workers 8

echo ""
echo "Done. Weights saved to: $DEST"
du -sh "$DEST"
