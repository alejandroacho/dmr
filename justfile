# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Blackwell Smart Gateway — Justfile
#  Install just: cargo install just  |  brew install just
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Path to your fork of gb10-vllm-mxfp4-docker
# https://github.com/alejandroacho/gb10-vllm-mxfp4-docker
spark_repo := env_var_or_default("SPARK_REPO", "$HOME/gb10-vllm-mxfp4-docker")

# Path where model weights are stored
models_dir := env_var_or_default("MODELS_DIR", "/home/alejandroacho/Models")

# Gateway URL
gateway := "http://localhost:8000"

# ── Default: list all recipes ──────────────────────────
[private]
default:
    @just --list

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODEL DOWNLOADS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Download GPT-OSS 120B weights (~240 GB)
download-gpt-oss:
    MODELS_DIR={{models_dir}} models/gpt-oss-120b/download.sh

# Download Qwen3-Coder-Next FP8 weights (~95 GB) to HF cache
download-qwen3:
    models/qwen3-coder-next/download.sh

# Download Qwen3-Coder-Next FP8 to a local directory
download-qwen3-local:
    MODELS_DIR={{models_dir}} models/qwen3-coder-next/download.sh

# Download MiniMax-H3 weights — video + native audio (~63 GB)
download-minimax:
    MODELS_DIR={{models_dir}} models/minimax-h3/download.sh

# Download ACE-Step 1.5 XL Turbo weights — music (~20 GB)
download-ace:
    MODELS_DIR={{models_dir}} models/ace-step-1.5/download.sh

# Download HiDream-O1-Image weights — images, both variants (~16 GB)
download-hidream:
    MODELS_DIR={{models_dir}} models/hidream-o1/download.sh

# Download everything the media node serves (~99 GB)
download-media: download-minimax download-ace download-hidream

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DOCKER IMAGE BUILDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Build the MXFP4 spark image for GPT-OSS 120B from your fork (~30 min first time)
build-spark:
    docker build -t vllm-mxfp4-spark {{spark_repo}}

# Build the Qwen3-Coder-Next vLLM image (applies GB10 patches at build time)
build-qwen3:
    docker build -f models/qwen3-coder-next/Dockerfile -t blackwell-vllm:latest models/qwen3-coder-next/

# Build the gateway container
build-gateway:
    docker compose build gateway

# Build the media node images (ComfyUI + CUDA 13, ~15 min first time)
build-media:
    docker compose --profile media build media-node media-gateway

# Build everything (spark first, then qwen3, then gateway)
build: build-spark build-qwen3 build-gateway

# Pull the latest cu130-nightly and rebuild qwen3 image
update-qwen3:
    docker pull vllm/vllm-openai:cu130-nightly
    just build-qwen3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GATEWAY — START / STOP / LOGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Start the gateway
up:
    docker compose up -d gateway

# Stop the gateway (and all inference containers)
down:
    docker compose down

# Restart the gateway
restart:
    docker compose restart gateway

# Follow gateway logs
logs:
    docker compose logs -f gateway

# Follow logs for a specific inference container (usage: just container-logs qwen3)
container-logs name:
    docker logs -f vllm-{{name}}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MEDIA NODE (node 3) — MiniMax-H3, standalone
#  Runs media only: no text models, no profiles, no orchestration.
#  Two containers: media-gateway (:8000) in front of media-node (:8010).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Start the whole media node (adapter + its Gateway)
media-up:
    docker compose --profile media up -d media-node media-gateway

# Stop the media node
media-down:
    docker compose --profile media stop media-gateway media-node

# Follow the inference logs (model loading, sampling)
media-logs:
    docker logs -f media-node

# Follow the Gateway logs (requests in, timings)
media-gateway-logs:
    docker logs -f media-gateway

# Adapter health, bypassing the Gateway
media-health:
    curl -s http://localhost:8010/health | python3 -m json.tool

# Gateway health — includes the backend's state and VRAM
media-status:
    curl -s {{gateway}}/health | python3 -m json.tool

# Models exposed by the media node
media-models:
    curl -s {{gateway}}/v1/models | python3 -m json.tool

# Smoke test — video with native audio, saved to h3-test.mp4
# 768x448 is the practical operating point; the 1344x768 default is ~5x slower.
media-test prompt="a lighthouse in a storm, waves crashing, thunder":
    #!/usr/bin/env bash
    set -euo pipefail
    curl -s -X POST {{gateway}}/v1/av/generate \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"{{prompt}}\", \"width\": 768, \"height\": 448, \"num_frames\": 124}" \
    | python3 -c "
    import sys, json, base64
    r = json.load(sys.stdin)
    if not r.get('success'):
        print(json.dumps(r, indent=2)); sys.exit(1)
    d = r['data']
    open('h3-test.mp4','wb').write(base64.b64decode(d['video_base64']))
    print(f\"Saved h3-test.mp4 — mode={d['mode']} seed={d['seed']} {d['processing_time_ms']/1000:.1f}s\")
    "

# Smoke test — music, saved to music-test.mp3 (prompt is style tags, not prose)
media-test-music prompt="lofi hip hop, mellow piano, vinyl crackle" seconds="20":
    #!/usr/bin/env bash
    set -euo pipefail
    curl -s -X POST {{gateway}}/v1/audio/music \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"{{prompt}}\", \"duration\": {{seconds}}}" \
    | python3 -c "
    import sys, json, base64
    r = json.load(sys.stdin)
    if not r.get('success'):
        print(json.dumps(r, indent=2)); sys.exit(1)
    d = r['data']
    open('music-test.mp3','wb').write(base64.b64decode(d['audio_base64']))
    print(f\"Saved music-test.mp3 — {d['duration']}s {d['bpm']}bpm {d['key_scale']} seed={d['seed']} {d['processing_time_ms']/1000:.1f}s\")
    "

# Smoke test — image, saved to image-test.png
media-test-image prompt="a noir portrait, dramatic rim light, 35mm film" size="1024":
    #!/usr/bin/env bash
    set -euo pipefail
    curl -s -X POST {{gateway}}/v1/images/generate \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"{{prompt}}\", \"width\": {{size}}, \"height\": {{size}}}" \
    | python3 -c "
    import sys, json, base64
    r = json.load(sys.stdin)
    if not r.get('success'):
        print(json.dumps(r, indent=2)); sys.exit(1)
    d = r['data']
    open('image-test.png','wb').write(base64.b64decode(d['images'][0]))
    print(f\"Saved image-test.png — {d['variant']} {d['width']}x{d['height']} steps={d['steps']} seed={d['seed']} {d['processing_time_ms']/1000:.1f}s\")
    "

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STATUS & MONITORING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Check gateway health
health:
    curl -s {{gateway}}/health | python3 -m json.tool

# Show current VRAM usage
vram:
    curl -s {{gateway}}/status/vram | python3 -m json.tool

# Show active profile and loaded models
profile:
    curl -s {{gateway}}/status/profile | python3 -m json.tool

# Show swap status
swap:
    curl -s {{gateway}}/status/swap | python3 -m json.tool

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PROFILE SWITCHING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Switch to Focus Code mode (Qwen3-Coder-Next 80B — default)
focus-code:
    curl -s -X POST {{gateway}}/admin/profile/focus_code | python3 -m json.tool

# Switch to Focus mode (GPT-OSS 120B)
focus:
    curl -s -X POST {{gateway}}/admin/profile/focus | python3 -m json.tool

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  QUICK SETUP (new machine from scratch)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Full setup: build all images + download both models
setup: build download-gpt-oss download-qwen3
    @echo ""
    @echo "Setup complete. Run 'just up' to start the gateway."

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DEVELOPMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Run tests
test:
    pytest tests/ -v

# Run the gateway locally (without Docker)
dev:
    python -m uvicorn gateway.app:app --host 0.0.0.0 --port 8000 --reload
