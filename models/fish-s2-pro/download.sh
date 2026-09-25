#!/usr/bin/env bash
set -euo pipefail
# Run with a Python environment containing huggingface_hub.
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="fishaudio/s2-pro",
    revision="1de9996b6be38b745688de084d87a5633f714e4e",
    local_dir="/home/alejandroacho/Models/fish-s2-pro",
    allow_patterns=["*.json", "*.safetensors", "*.pth", "*.jinja", "LICENSE.md", "README.md"],
)
PY
