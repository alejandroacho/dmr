#!/usr/bin/env bash
# Download MiniMax-H3 weights (ComfyUI repackaged format) for the media node.
# Repo: Comfy-Org/MiniMax-H3 (public, no gating)
#
# Downloads only the pruned INT8 ConvRot variants — the quality/VRAM sweet spot
# for a single GB10 (~63 GB total, both generation modes resident at once):
#
#   FL2VA set (42.47 GB)                                  ref2va (20.97 GB)
#   ├── minimax_h3_fl2va_pruned_int8_convrot   19.53 GiB  └── minimax_h3_ref2va_pruned_int8_convrot
#   ├── qwen3vl_32b_minimax_h3_nvfp4_awq       14.61 GiB      (shares the encoder + VAEs above)
#   ├── minimax_h3_video_vae_fp16               4.85 GiB
#   └── minimax_h3_audio_vae_fp32               0.56 GiB
#
# The repo layout (diffusion_models/, text_encoders/, vae/) is already ComfyUI's
# models/ layout, so DEST can be mounted straight into the container.
#
# Usage:
#   ./download.sh                          # downloads to default MODELS_DIR
#   MODELS_DIR=/data/models ./download.sh  # custom path

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/home/alejandroacho/Models}"
DEST="$MODELS_DIR/minimax-h3"

# Parallel Xet chunk transfer — saturates the link on big shards
export HF_XET_HIGH_PERFORMANCE=1

echo "Downloading Comfy-Org/MiniMax-H3 → $DEST"
echo "Size: ~63 GB. Use Ctrl+C to pause — re-running will resume."
echo ""

hf download Comfy-Org/MiniMax-H3 \
    diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
    diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
    text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
    vae/minimax_h3_video_vae_fp16.safetensors \
    vae/minimax_h3_audio_vae_fp32.safetensors \
    --local-dir "$DEST" \
    --max-workers 8

echo ""
echo "Done. Weights saved to: $DEST"
du -sh "$DEST"
