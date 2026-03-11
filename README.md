# Blackwell Orchestrator & Smart Gateway

Intelligent middleware layer for autonomous VRAM management (~120 GB), dynamic model swapping, and smart routing on an ASUS GX10 with a single NVIDIA GB10 Blackwell GPU.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    9 External Agents                     │
│         (text, image, video — unified endpoint)          │
└──────────────────────┬───────────────────────────────────┘
                       │  HTTP / JSON
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Smart Gateway  (FastAPI :8000)              │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │ Smart Router│  │  VRAM Monitor│  │ Request Buffer │   │
│  │  & Trigger  │  │  (nvidia-smi)│  │ (Long Polling) │   │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘   │
│         │                │                   │           │
│  ┌──────▼────────────────▼───────────────────▼────────┐  │
│  │         Container Orchestrator (Docker SDK)        │  │
│  │         Mutex-protected profile swapping           │  │
│  └──────────────────────┬─────────────────────────────┘  │
└─────────────────────────┼────────────────────────────────┘
                          │  Docker Socket
         ┌────────────────┼──────────────────────┐
         ▼                ▼                      ▼
┌──────────────┐ ┌───────────────────┐ ┌──────────────────┐
│ vLLM :8001   │ │ vLLM :8002        │ │ ComfyUI / LTX    │
│ GPT-OSS 120B │ │ Qwen3 Coder Next  │ │ :8004 / :8005    │
│  (mxfp4)     │ │ 80B MoE (auto)    │ │ FLUX.2 / LTX-V2  │
└──────────────┘ └───────────────────┘ └──────────────────┘
  FOCUS PROFILE    FOCUS_CODE PROFILE    CREATIVE PROFILE
                   (default at startup)
```

## VRAM Profiles

| Profile key | Models | VRAM Used | Use Case |
|-------------|--------|-----------|----------|
| **`focus_code`** ⭐ | Qwen3 Coder Next 80B MoE (auto) | ~95 GB | Code, engineering — **default at startup** |
| **`focus`** | GPT-OSS 120B (mxfp4 CUTLASS sm_121) | ~84 GB | Reasoning, general tasks |
| **`creative_image`** | Qwen3 Coder 30B + FLUX.2 Pro (FP16) | ~77 GB | Text + image generation |
| **`creative_video`** | Qwen3 Coder 30B + LTX-Video 2 (Q8) | ~77 GB | Text + video generation |

> **Note:** `focus` (GPT-OSS 120B) requires a custom vLLM image with CUTLASS MXFP4 kernels compiled for sm_121. Build it first: `just build-spark` from [github.com/alejandroacho/gb10-vllm-mxfp4-docker](https://github.com/alejandroacho/gb10-vllm-mxfp4-docker) (~30 min). Expected throughput: 57–60 tok/s on single GB10.

> **Note:** `focus_code` (Qwen3-Coder-Next FP8) uses `blackwell-vllm:latest` (same as `vllm/vllm-openai:cu130-nightly`) with two runtime patches applied at container startup. Expected throughput: ~43–48 tok/s on single GB10. The model is downloaded automatically from HuggingFace on first run (~95 GB).

Profile transitions are **automatic** — the Gateway detects visual keywords in requests and swaps models transparently.

---

## Docker Images

Two custom Docker images are required. Here is what each one is and why.

### `vllm-mxfp4-spark:latest` — GPT-OSS 120B only

Built from [github.com/alejandroacho/gb10-vllm-mxfp4-docker](https://github.com/alejandroacho/gb10-vllm-mxfp4-docker). This is a **fork of vLLM** that NVIDIA published as the reference implementation for DGX Spark / GB10. It contains:

- **CUTLASS MXFP4 MoE kernels** compiled for SM121 — the first open implementation of block-scaled MXFP4 GEMM on GB10
- Automatic tile selection: 64×128 PingPong for decode (small batches), 128×128 Cooperative for prefill (large batches)
- FP8 E4M3 KV cache with GPT-OSS attention sink support
- PyTorch and Triton compiled natively for SM121 (GB10 compute capability)
- `fastsafetensors` for fast NVMe-to-GPU weight loading

Without this image, GPT-OSS runs 40–50% slower (SGLang and llama.cpp both beat a stock vLLM install). With it: **57–60 tok/s single node, 72 tok/s TP=2 with RDMA**.

```bash
# Clone your fork and build (~30 min, uses BuildKit cache for fast rebuilds)
git clone https://github.com/alejandroacho/gb10-vllm-mxfp4-docker ~/gb10-vllm-mxfp4-docker
just build-spark
```

This image is **not compatible with Qwen3-Coder-Next**. It has a custom `vllm.envs` that is missing `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER`, which Qwen3-Next's initialization code requires. Do not attempt to serve Qwen3-Next on this image.

---

### `blackwell-vllm:latest` — Qwen3-Coder-Next FP8 and other models

This is `vllm/vllm-openai:cu130-nightly` re-tagged locally:

```bash
docker pull vllm/vllm-openai:cu130-nightly
docker tag vllm/vllm-openai:cu130-nightly blackwell-vllm:latest
```

It is a stock vLLM nightly build against CUDA 13.0. It runs on GB10 in SM120 fallback mode (PyTorch warns about SM121 not being supported up to 12.0, but inference still works correctly).

**Two patches** are baked into the image at build time (see `models/qwen3-coder-next/`):

| Patch | What it fixes |
|---|---|
| Revert PR #34279 | That vLLM PR added `tl.int64` type annotations to Triton MoE kernel strides that cause severe slowness on GB10. The revert is a no-op if the nightly already has it reverted. |
| `_triton_alloc_setup.py` | Patches `triton.runtime._allocation.NullAllocator.__call__` to use `torch.cuda.caching_allocator_alloc`. Without this, Triton crashes with "Kernel requires a runtime memory allocation, but no allocator was set" on the first inference call. |

The patch files live in `models/qwen3-coder-next/` and are baked into the image at build time via `models/qwen3-coder-next/Dockerfile`. No runtime mounts needed — just `just build-qwen3`.

Expected throughput for Qwen3-Coder-Next FP8 on GB10: **43–48 tok/s** (decode), **~3000 tok/s** (prefill), up to 262K token context with FlashInfer backend.

---

## Prerequisites

| Requirement | Minimum |
|---|---|
| **OS** | Linux (Ubuntu 22.04+ recommended) |
| **Docker** | Docker Engine 24+ with Docker Compose V2 |
| **NVIDIA Driver** | 535+ (Blackwell-compatible) |
| **NVIDIA Container Toolkit** | `nvidia-container-toolkit` installed and configured |
| **GPU** | 1× NVIDIA GB10 Blackwell with ~120 GB unified VRAM |
| **System RAM** | 218 GB+ (512 GB+ enables fast pause/unpause swap strategy) |
| **Storage** | NVMe SSD with models stored at `/mnt/nvme_data/models/` |
| **Python** | 3.10+ (only needed for local development without Docker) |

---

## Quick Start (fresh machine)

### 1. Prerequisites

```bash
# Install Docker Engine + NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Install just
cargo install just   # or: brew install just / apt install just

# Install the hf CLI and log in (needed to download gated models like gpt-oss-120b)
pip install -U huggingface_hub
hf login
```

### 2. Clone the repos

```bash
git clone https://github.com/alejandroacho/Server ~/Server
git clone https://github.com/alejandroacho/gb10-vllm-mxfp4-docker ~/gb10-vllm-mxfp4-docker
cd ~/Server
```

### 3. Build Docker images

```bash
# Build everything: spark (~30 min), qwen3 (~2 min), gateway (~1 min)
just build
```

### 4. Download model weights

```bash
just download-gpt-oss    # openai/gpt-oss-120b  → ~/Models/gpt-oss-120b-q8  (~240 GB)
just download-qwen3      # Qwen/Qwen3-Coder-Next-FP8 → HF cache             (~95 GB)
```

> Ctrl+C pauses any download — re-running resumes it.

### 5. Launch

```bash
just up
```

### 7. Verify it's running

```bash
just health    # gateway health check
just vram      # VRAM usage
just profile   # active profile and loaded models
```

---

## Local Development (Without Docker)

```bash
cd ~/Server

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Gateway
python -m uvicorn gateway.app:app --host 0.0.0.0 --port 8000 --reload
```

> **Note:** Local development requires Docker Engine running (the Gateway manages inference containers via the Docker socket).

---

## API Endpoints

### Inference (for the 9 agents)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | **Unified endpoint** — auto-detects text/image/video and routes accordingly. OpenAI-compatible. |
| `POST` | `/v1/images/generate` | Direct image generation (FLUX.2 Pro) |
| `POST` | `/v1/videos/generate` | Direct video generation (LTX-Video 2) |
| `GET`  | `/v1/models` | List available models (OpenAI format) |

### System Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Gateway health check + cluster status |
| `GET` | `/status/vram` | Detailed VRAM report from nvidia-smi |
| `GET` | `/status/swap` | Current swap status (in progress, elapsed, queue) |
| `GET` | `/status/profile` | Active profile and loaded models |
| `GET` | `/status/cache` | Radix Prefix Cache statistics |

### Administration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/admin/profile/{name}` | Manually switch profile (`focus`, `focus_code`, `creative_image`, `creative_video`) |
| `POST` | `/admin/container/{name}/stop` | Stop a specific inference container |
| `POST` | `/admin/container/{name}/remove` | Remove a specific inference container |

### Interactive API Docs

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Usage Examples

### Text completion (auto-routes to Focus profile)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a senior software engineer."},
      {"role": "user", "content": "Explain the observer pattern in Go."}
    ],
    "temperature": 0.7,
    "max_tokens": 2048,
    "agent_id": "agent-1"
  }'
```

### Image generation (auto-triggers Creative profile swap)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "generate_image: A futuristic cityscape at sunset"}
    ],
    "tool_choice": "generate_image",
    "media_type": "image",
    "media_params": {
      "width": 1024,
      "height": 1024,
      "steps": 30
    },
    "agent_id": "agent-3"
  }'
```

### Direct image endpoint

```bash
curl -X POST http://localhost:8000/v1/images/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A photorealistic mountain landscape with northern lights",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "agent_id": "agent-5"
  }'
```

### Video generation

```bash
curl -X POST http://localhost:8000/v1/videos/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A drone flyover of a coral reef in crystal clear water",
    "width": 768,
    "height": 512,
    "num_frames": 81,
    "fps": 24,
    "agent_id": "agent-7"
  }'
```

### Manual profile switch

```bash
# Switch to Focus Code mode (Qwen3 Coder Next 80B — default)
curl -X POST http://localhost:8000/admin/profile/focus_code

# Switch to Focus mode (GPT-OSS 120B — verify GB10 compatibility first)
curl -X POST http://localhost:8000/admin/profile/focus

# Switch to Creative Image mode
curl -X POST http://localhost:8000/admin/profile/creative_image

# Switch to Creative Video mode
curl -X POST http://localhost:8000/admin/profile/creative_video
```

---

## Swap Behavior

When a request requires a different profile than the currently active one:

| Config | Behavior |
|--------|----------|
| `LONG_POLLING_ENABLED=true` | Connection stays open; response sent once the swap completes (~2-5s) |
| `LONG_POLLING_ENABLED=false` | Immediate `HTTP 503` with `Retry-After` header |

### Swap strategies

| System RAM | Strategy | Speed | Detail |
|---|---|---|---|
| ≥ 512 GB | `pause/unpause` | Fast (~1-2s) | Containers stay in memory, VRAM freed |
| < 512 GB | `stop/start` | Slower (~3-5s) | Full container restart, deep VRAM cleanup |

---

## Project Structure

```
Server/
├── .env                        # Environment variables
├── .gitignore
├── Dockerfile                  # Gateway container image
├── docker-compose.yml          # Full stack definition
├── justfile                    # Task runner (install: cargo install just)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/
│   ├── gpt-oss-120b/
│   │   ├── Dockerfile          # FROM vllm-mxfp4-spark:latest (built from fork)
│   │   └── download.sh         # hf download openai/gpt-oss-120b
│   └── qwen3-coder-next/
│       ├── Dockerfile          # FROM cu130-nightly + GB10 patches applied at build time
│       ├── fix_slowness.diff   # Reverts vLLM PR #34279 (Triton MoE slowness on GB10)
│       ├── _triton_alloc_setup.py  # Fixes Triton NullAllocator crash
│       ├── _triton_alloc_setup.pth # Auto-loaded by Python on startup
│       └── download.sh         # hf download Qwen/Qwen3-Coder-Next-FP8
└── gateway/
    ├── __init__.py             # Package init + version
    ├── app.py                  # FastAPI application (main entry point)
    ├── config.py               # Central configuration + VRAM profiles
    ├── orchestrator.py         # Docker container lifecycle manager
    ├── proxy.py                # HTTP proxy to inference backends
    ├── request_buffer.py       # Request queue + Radix Prefix Cache
    ├── router.py               # Smart routing + trigger detection
    ├── schemas.py              # Pydantic data models
    └── vram_monitor.py         # nvidia-smi VRAM monitoring
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_HOST` | `0.0.0.0` | Host to bind |
| `GATEWAY_PORT` | `8000` | Port to listen on |
| `MODELS_PATH` | `/mnt/nvme_data/models` | Path to model weights |
| `SYSTEM_RAM_GB` | `218` | System RAM (determines swap strategy) |
| `SWAP_TIMEOUT_S` | `600` | Max seconds to wait for a swap |
| `VRAM_POLL_INTERVAL_S` | `2.0` | nvidia-smi polling interval |
| `VRAM_SAFETY_MARGIN_MB` | `4096` | VRAM to keep free as safety buffer |
| `DOCKER_SOCKET` | `unix:///var/run/docker.sock` | Docker daemon socket |
| `DOCKER_NETWORK` | `blackwell-gateway_blackwell_net` | Docker network name |
| `LONG_POLLING_ENABLED` | `true` | Hold connections during swaps |
| `LONG_POLLING_TIMEOUT_S` | `600` | Max long polling wait |
| `MAX_QUEUE_SIZE` | `200` | Max requests queued during swap |
| `RETRY_AFTER_SECONDS` | `5` | Retry-After header value for 503s |

---

## Troubleshooting

### Gateway won't start
```bash
# Check logs
docker compose logs -f gateway

# Verify Docker socket is accessible
ls -la /var/run/docker.sock

# Verify NVIDIA runtime is available
docker run --rm --runtime=nvidia nvidia/cuda:12.0-base nvidia-smi
```

### Swap is too slow
- Increase `SYSTEM_RAM_GB` in `.env` if you have ≥512 GB RAM (enables fast pause/unpause)
- Check NVMe speed: `fio --name=test --rw=read --bs=1M --size=1G --numjobs=1`
- Ensure models are on NVMe, not HDD

### VRAM errors during swap
```bash
# Check current VRAM usage
curl http://localhost:8000/status/vram

# Force profile switch (cleans up stale containers)
curl -X POST "http://localhost:8000/admin/profile/focus?force=true"
```

### Container stuck in STARTING state
```bash
# Check container logs
docker logs vllm-gpt-oss-120b

# Force remove and let the Gateway recreate it
curl -X POST http://localhost:8000/admin/container/vllm-gpt-oss-120b/remove
```

### Qwen3-Coder-Next: "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER" AttributeError

This means Qwen3-Next is being served on the `vllm-mxfp4-spark` image instead of `blackwell-vllm`. The spark image has a custom `vllm.envs` that is missing this variable. Verify `config.py` has `QWEN3_CODER_NEXT_80B` using `container_image="blackwell-vllm:latest"`.

### Qwen3-Coder-Next: "No module named 'fastsafetensors'"

Remove `--load-format fastsafetensors` from the model args. The `blackwell-vllm` (cu130-nightly) image does not include the `fastsafetensors` package. Use the default safetensors loader.

### Qwen3-Coder-Next: "Kernel requires a runtime memory allocation, but no allocator was set"

The Triton allocator patch is missing. Verify `/tmp/qwen3-patches/_triton_alloc_setup.py` exists on the host and that the Gateway mounts it into the container at startup.

### Qwen3-Coder-Next: 2–3 tok/s instead of ~43 tok/s

vLLM PR #34279 introduced `tl.int64` annotations in Triton MoE strides that cause severe slowness on GB10. Check that the `fix_slowness.diff` patch is being applied. To verify which MoE backend is active, look for this line in the container logs:
```
Using TRITON Fp8 MoE backend out of potential backends: [...]
```
TRITON is expected (and patched to be fast). If you see it and speed is still low, the `fix_slowness.diff` patch did not apply — check whether the current nightly already has it reverted.

### Free VRAM less than desired on Qwen3 startup

GPT-OSS is still running. The Gateway should stop it automatically before starting Qwen3. If it does not, swap manually:
```bash
curl -X POST http://localhost:8000/admin/profile/focus_code
```

---

## License

Private — Alejandro Acho. All rights reserved.
