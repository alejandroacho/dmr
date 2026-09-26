#!/usr/bin/env bash
# Official ComfyUI repack: https://huggingface.co/Comfy-Org/Qwen-Image-2.1
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-/home/alejandroacho/Models}"
DEST="$MODELS_DIR/qwen-image-2.1"
export HF_XET_HIGH_PERFORMANCE=1
hf download Comfy-Org/Qwen-Image-2.1 \
    diffusion_models/qwen_image_2.1_int8_convrot.safetensors \
    text_encoders/qwen3vl_8b_int8_convrot.safetensors \
    vae/qwen_image_2.1_vae_bf16.safetensors \
    --local-dir "$DEST" \
    --max-workers 8
echo "Done. Weights saved to: $DEST"
du -sh "$DEST"
