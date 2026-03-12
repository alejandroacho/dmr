# FLUX.1-dev inference server
# Builds the 'comfyui-flux:latest' image used by the creative_image profile.
# Compatible with NVIDIA GB10 Blackwell (sm_121).
#
# This is a reference copy — the canonical Dockerfile is at the repo root:
#   Dockerfile.comfyui
#
# Build (from repo root):
#   docker build -f Dockerfile.comfyui -t comfyui-flux:latest .
#
# Run standalone (for testing):
#   docker run --rm --gpus all -p 8004:8004 \
#     -v /home/alejandroacho/Models/flux2-pro-fp16:/models:ro \
#     comfyui-flux:latest
#
# Download weights first:
#   ./download.sh
