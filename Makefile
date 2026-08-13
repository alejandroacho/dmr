# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Blackwell Smart Gateway — Makefile
#  Requires: GNU make, docker, docker compose, python3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Path to your fork of gb10-vllm-mxfp4-docker
# https://github.com/alejandroacho/gb10-vllm-mxfp4-docker
SPARK_REPO ?= $(HOME)/gb10-vllm-mxfp4-docker

# Path where model weights are stored
MODELS_DIR ?= /home/alejandroacho/Models

# Gateway URL
GATEWAY ?= http://localhost:8000

# These are task-runner targets, not build artifacts: never run them in parallel
# and never look for files with matching names.
.NOTPARALLEL:
MAKEFLAGS += --no-print-directory
.DEFAULT_GOAL := help

# ── Default: list all targets ──────────────────────────

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; \
		printf "\nBlackwell Smart Gateway\n\nUsage:\n  make \033[36m<target>\033[0m\n"} \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' \
		$(MAKEFILE_LIST)
	@echo ""

##@ Model downloads

.PHONY: download-gpt-oss
download-gpt-oss: ## Download GPT-OSS 120B weights (~240 GB)
	MODELS_DIR=$(MODELS_DIR) models/gpt-oss-120b/download.sh

.PHONY: download-qwen3
download-qwen3: ## Download Qwen3-Coder-Next FP8 weights (~95 GB) to HF cache
	models/qwen3-coder-next/download.sh

.PHONY: download-qwen3-local
download-qwen3-local: ## Download Qwen3-Coder-Next FP8 to a local directory
	MODELS_DIR=$(MODELS_DIR) models/qwen3-coder-next/download.sh

##@ Docker image builds

.PHONY: build-spark
build-spark: ## Build the MXFP4 spark image for GPT-OSS 120B from your fork (~30 min first time)
	docker build -t vllm-mxfp4-spark $(SPARK_REPO)

.PHONY: build-qwen3
build-qwen3: ## Build the Qwen3-Coder-Next vLLM image (applies GB10 patches at build time)
	docker build -f models/qwen3-coder-next/Dockerfile -t blackwell-vllm:latest models/qwen3-coder-next/

.PHONY: build-blackwell
build-blackwell: ## Build the blackwell-vllm image with Gemma 4 support (for Ray cluster nodes)
	docker build -f ray-cluster/Dockerfile.blackwell-vllm -t blackwell-vllm:latest ray-cluster/

.PHONY: build-gateway
build-gateway: ## Build the gateway container
	docker compose build gateway

.PHONY: build
build: build-spark build-qwen3 build-gateway ## Build everything (spark, then qwen3, then gateway)

.PHONY: update-qwen3
update-qwen3: ## Pull the latest cu130-nightly and rebuild the qwen3 image
	docker pull vllm/vllm-openai:cu130-nightly
	$(MAKE) build-qwen3

##@ Gateway — start / stop / logs

.PHONY: up
up: ## Start the gateway
	docker compose up -d gateway

.PHONY: down
down: ## Stop the gateway (and all inference containers)
	docker compose down

.PHONY: restart
restart: ## Restart the gateway
	docker compose restart gateway

.PHONY: logs
logs: ## Follow gateway logs
	docker compose logs -f gateway

.PHONY: container-logs
container-logs: ## Follow logs for an inference container (usage: make container-logs NAME=qwen3)
	@test -n "$(NAME)" || { echo "error: NAME is required — usage: make container-logs NAME=qwen3"; exit 1; }
	docker logs -f vllm-$(NAME)

##@ Status & monitoring

.PHONY: health
health: ## Check gateway health
	curl -s $(GATEWAY)/health | python3 -m json.tool

.PHONY: vram
vram: ## Show current VRAM usage
	curl -s $(GATEWAY)/status/vram | python3 -m json.tool

.PHONY: profile
profile: ## Show active profile and loaded models
	curl -s $(GATEWAY)/status/profile | python3 -m json.tool

.PHONY: swap
swap: ## Show swap status
	curl -s $(GATEWAY)/status/swap | python3 -m json.tool

##@ Profile switching

.PHONY: focus-code
focus-code: ## Switch to Focus Code mode (Qwen3-Coder-Next 80B — default)
	curl -s -X POST $(GATEWAY)/admin/profile/focus_code | python3 -m json.tool

.PHONY: focus
focus: ## Switch to Focus mode (GPT-OSS 120B)
	curl -s -X POST $(GATEWAY)/admin/profile/focus | python3 -m json.tool

##@ Quick setup (new machine from scratch)

.PHONY: setup
setup: build download-gpt-oss download-qwen3 ## Full setup: build all images + download both models
	@echo ""
	@echo "Setup complete. Run 'make up' to start the gateway."

##@ Development

.PHONY: test
test: ## Run tests
	pytest tests/ -v

.PHONY: dev
dev: ## Run the gateway locally (without Docker)
	python -m uvicorn gateway.app:app --host 0.0.0.0 --port 8000 --reload
