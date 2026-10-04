#!/usr/bin/env bash
# Download ACE-Step 1.5 XL Turbo weights (ComfyUI repackaged) for the media node.
# Repo: Comfy-Org/ace_step_1.5_ComfyUI_files (public, no gating)
#
# Text-to-music. The XL Turbo variant samples in 8 steps at cfg 1.0 — the file
# set and settings come from ComfyUI's own blueprint,
# "blueprints/Text to Audio (ACE-Step 1.5).json":
#
#   acestep_v1.5_xl_turbo_bf16    9.29 GiB   DiT (XL turbo)
#   qwen_0.6b_ace15               1.11 GiB   ┐ DualCLIPLoader, type "ace"
#   qwen_4b_ace15                 7.80 GiB   ┘ (both are needed)
#   ace_1.5_vae                   0.31 GiB   audio VAE (DCAE + vocoder)
#                                ─────────
#                                18.51 GiB  (~19.9 GB)
#
# Usage:
#   ./download.sh                          # downloads to default MODELS_DIR
#   MODELS_DIR=/data/models ./download.sh  # custom path

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/home/alejandroacho/Models}"
DEST="$MODELS_DIR/ace-step-1.5"

export HF_XET_HIGH_PERFORMANCE=1

echo "Downloading Comfy-Org/ace_step_1.5_ComfyUI_files → $DEST"
echo "Size: ~20 GB. Use Ctrl+C to pause — re-running will resume."
echo ""

# The repo nests everything under split_files/, which is already a ComfyUI models
# tree (diffusion_models/, text_encoders/, vae/). Left in place rather than moved
# or copied: extra_model_paths.yaml points at $DEST/split_files, so there is no
# duplicate 19 GB and `hf download` can still resume against its own cache.
hf download Comfy-Org/ace_step_1.5_ComfyUI_files \
    split_files/diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors \
    split_files/text_encoders/qwen_0.6b_ace15.safetensors \
    split_files/text_encoders/qwen_4b_ace15.safetensors \
    split_files/vae/ace_1.5_vae.safetensors \
    --local-dir "$DEST" \
    --max-workers 8

echo ""
echo "Done. Weights saved to: $DEST/split_files"
du -sh "$DEST"
