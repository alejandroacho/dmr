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
