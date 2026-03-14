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
│ vLLM :8001   │ │ vLLM :8002        │ │ Diffusers :8004  │
│ GPT-OSS 120B │ │ Qwen3 Coder Next  │ │ FLUX.1-dev BF16  │
│  (mxfp4)     │ │ 80B MoE (fp8)     │ │ (pytorch:25.01)  │
└──────────────┘ └───────────────────┘ └──────────────────┘
  FOCUS PROFILE    FOCUS_CODE PROFILE    CREATIVE_IMAGE
                   (default at startup)
```

## VRAM Profiles

| Profile key | Models | VRAM Used | Use Case |
|-------------|--------|-----------|----------|
| **`focus_code`** ⭐ | Qwen3 Coder Next 80B MoE (fp8) | ~95 GB | Code, engineering — **default at startup** |
| **`focus`** | GPT-OSS 120B (mxfp4 CUTLASS sm_121) | ~84 GB | Reasoning, general tasks |
| **`creative_image`** | FLUX.1-dev (BF16) | ~24 GB | Image generation |
| **`creative_video`** | Qwen3 Coder 30B + LTX-Video 2 (Q8) | ~77 GB | Text + video generation |

> **Note:** `creative_image` runs FLUX.1-dev solo (no text model). Image generation via `/v1/images/generate`. Current speed: ~12s/step on pytorch:25.01 (no sm_121 kernels). Pending migration to a Blackwell-native image for <1s/step.

> **Note:** `focus` (GPT-OSS 120B) requires a custom vLLM image with CUTLASS MXFP4 kernels compiled for sm_121. Build it first: `just build-spark` (~30 min). Runs with `--enforce-eager` (CUDA graphs crash on SM121 with MXFP4 batching). Expected throughput: ~57 tok/s single request, ~5-6 tok/s per request with 10 concurrent agents. Supports up to 10 simultaneous requests (`--max-num-seqs 10`).

> **Note:** `focus_code` (Qwen3-Coder-Next FP8) uses `blackwell-vllm:latest` with two runtime patches applied at build time. Expected throughput: ~43–48 tok/s on single GB10.

Profile transitions are **automatic** — the Gateway detects visual keywords in requests and swaps models transparently.

---

## Docker Images

### `vllm-mxfp4-spark:latest` — GPT-OSS 120B only

Built from [github.com/alejandroacho/gb10-vllm-mxfp4-docker](https://github.com/alejandroacho/gb10-vllm-mxfp4-docker). Contains:

- **CUTLASS MXFP4 MoE kernels** compiled for SM121
- FP8 E4M3 KV cache with GPT-OSS attention sink support
- PyTorch and Triton compiled natively for SM121
- `fastsafetensors` for fast NVMe-to-GPU weight loading

```bash
git clone https://github.com/alejandroacho/gb10-vllm-mxfp4-docker ~/gb10-vllm-mxfp4-docker
just build-spark
```

This image is **not compatible with Qwen3-Coder-Next**. It has a custom `vllm.envs` missing `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER`.

---

### `blackwell-vllm:latest` — Qwen3-Coder-Next FP8 and other vLLM models

`vllm/vllm-openai:cu130-nightly` re-tagged locally with two patches baked in (see `models/qwen3-coder-next/`):

| Patch | What it fixes |
|---|---|
| Revert PR #34279 | Removes `tl.int64` Triton MoE stride annotations that cause severe slowness on GB10 |
| `_triton_alloc_setup.py` | Patches `triton.runtime._allocation.NullAllocator` to use CUDA caching allocator |

Expected throughput: **43–48 tok/s** decode, **~3000 tok/s** prefill, up to 262K token context.

---

### `comfyui-flux:latest` — FLUX.1-dev image generation

Built from `Dockerfile.comfyui`. Runs `inference/flux_server.py` — a FastAPI server that loads FLUX.1-dev via Diffusers and exposes `POST /generate`.

- Base image: `nvcr.io/nvidia/pytorch:25.01-py3` (CUDA 12.8, compatible with driver 525+)
- Model loaded in BF16 to avoid APEX fused layer norm issues with FP16
- Health endpoint returns 503 while model loads, 200 when ready
- Current speed: ~12s/step (~6 min for 30 steps) — pytorch:25.01 has no sm_121 kernels

```bash
# Build
docker build -f Dockerfile.comfyui -t comfyui-flux:latest .

# Download weights (~34 GB, requires HuggingFace login with FLUX.1-dev access)
cd models/flux2-pro && ./download.sh
```

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
| **Storage** | NVMe SSD with models at `/home/alejandroacho/Models/` |
| **Python** | 3.10+ (only needed for local development without Docker) |

---

## Quick Start (fresh machine)

### 1. Prerequisites

```bash
# Install Docker Engine + NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Install just
cargo install just   # or: brew install just / apt install just

# Install the hf CLI and log in
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
# Build everything: spark (~30 min), qwen3 (~2 min), flux (~5 min), gateway (~1 min)
just build

# Or individually:
just build-spark       # vllm-mxfp4-spark:latest  (GPT-OSS 120B)
just build-qwen3       # blackwell-vllm:latest     (Qwen3-Coder-Next)
docker build -f Dockerfile.comfyui -t comfyui-flux:latest .   # FLUX.1-dev
docker compose build gateway
```

### 4. Download model weights

```bash
just download-gpt-oss    # openai/gpt-oss-120b      → ~/Models/gpt-oss-120b-q8  (~240 GB)
just download-qwen3      # Qwen/Qwen3-Coder-Next-FP8 → HF cache                 (~95 GB)
cd models/flux2-pro && ./download.sh   # FLUX.1-dev  → ~/Models/flux2-pro-fp16  (~34 GB)
```

> Ctrl+C pauses any download — re-running resumes it.

### 5. Launch

```bash
just up
```

### 6. Verify it's running

```bash
just health    # gateway health check
just vram      # VRAM usage
just profile   # active profile and loaded models
```

---

## Local Development (Without Docker)

```bash
cd ~/Server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn gateway.app:app --host 0.0.0.0 --port 8000 --reload
```

> **Note:** Local development requires Docker Engine running (the Gateway manages inference containers via the Docker socket).

---

## API Endpoints

### Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | **Unified endpoint** — auto-detects text/image/video and routes accordingly. OpenAI-compatible. |
| `POST` | `/v1/images/generate` | Direct image generation (FLUX.1-dev). Returns base64 PNG. |
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

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Usage Examples

### Text completion

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

### Image generation (direct endpoint)

```bash
curl -X POST http://localhost:8000/v1/images/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A photorealistic mountain landscape with northern lights",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "seed": 42
  }'
```

Response: `{"success": true, "data": {"images": ["<base64 PNG>"], "seed": 42, ...}}`

To save to disk:
```bash
curl -s -X POST http://localhost:8000/v1/images/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red cat"}' \
  | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
img = base64.b64decode(data['data']['images'][0])
open('output.png', 'wb').write(img)
print('Saved output.png')
"
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
    "fps": 24
  }'
```

### Manual profile switch

```bash
curl -X POST http://localhost:8000/admin/profile/focus_code      # Qwen3 Coder Next 80B (default)
curl -X POST http://localhost:8000/admin/profile/focus            # GPT-OSS 120B
curl -X POST http://localhost:8000/admin/profile/creative_image   # FLUX.1-dev
curl -X POST http://localhost:8000/admin/profile/creative_video   # LTX-Video 2
```

---

## Swap Behavior

| Config | Behavior |
|--------|----------|
| `LONG_POLLING_ENABLED=true` | Connection stays open; response sent once the swap completes |
| `LONG_POLLING_ENABLED=false` | Immediate `HTTP 503` with `Retry-After` header |

| System RAM | Strategy | Speed |
|---|---|---|
| ≥ 512 GB | `pause/unpause` | ~1-2s |
| < 512 GB | `stop/start` | ~3-5s |

---

## Project Structure

```
Server/
├── Dockerfile                  # Gateway container image
├── Dockerfile.comfyui          # FLUX.1-dev inference server image
├── Dockerfile.ltx              # LTX-Video 2 inference server image
├── docker-compose.yml          # Full stack definition
├── justfile                    # Task runner
├── requirements.txt            # Python dependencies
├── inference/
│   ├── flux_server.py          # FLUX.1-dev FastAPI server (port 8004)
│   └── ltx_server.py           # LTX-Video 2 FastAPI server (port 8005)
├── models/
│   ├── gpt-oss-120b/
│   │   ├── Dockerfile          # FROM vllm-mxfp4-spark:latest
│   │   └── download.sh         # hf download openai/gpt-oss-120b (~240 GB)
│   ├── qwen3-coder-next/
│   │   ├── Dockerfile          # FROM cu130-nightly + GB10 patches
│   │   ├── fix_slowness.diff   # Reverts vLLM PR #34279
│   │   ├── _triton_alloc_setup.py
│   │   ├── _triton_alloc_setup.pth
│   │   └── download.sh         # hf download Qwen/Qwen3-Coder-Next-FP8 (~95 GB)
│   └── flux2-pro/
│       ├── Dockerfile          # Reference → Dockerfile.comfyui at root
│       └── download.sh         # hf download black-forest-labs/FLUX.1-dev (~34 GB)
└── gateway/
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
| `MODELS_PATH` | `/home/alejandroacho/Models` | Path to model weights on host |
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
docker compose logs -f gateway
ls -la /var/run/docker.sock
docker run --rm --runtime=nvidia nvidia/cuda:12.0-base nvidia-smi
```

### FLUX container fails with "model_index.json not found"
The model directory is empty. Download weights:
```bash
cd models/flux2-pro && ./download.sh
```

### FLUX container: "expected scalar type Float but found Half"
Model is loaded in FP16. Switch to BF16 in `inference/flux_server.py`:
```python
torch_dtype=torch.bfloat16
```

### FLUX generation is slow (~12s/step)
pytorch:25.01 does not have native SM121 (GB10) kernels. The container warns "GB10 GPU may not yet be supported". Performance will improve when migrating to a Blackwell-native PyTorch image. Current workaround: proxy timeout set to 900s.

### Swap is too slow
- Increase `SYSTEM_RAM_GB` if you have ≥512 GB RAM (enables fast pause/unpause)
- Check NVMe speed: `fio --name=test --rw=read --bs=1M --size=1G --numjobs=1`

### VRAM errors during swap
```bash
curl http://localhost:8000/status/vram
curl -X POST "http://localhost:8000/admin/profile/focus?force=true"
```

### GPT-OSS 120B: `cudaErrorIllegalAddress` crash with concurrent requests

CUDA graphs are incompatible with MXFP4 CUTLASS kernels on SM121 (GB10) when batching multiple requests. The vLLM EngineCore crashes with `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`.

**Fix:** `--enforce-eager` is enabled in `config.py` to disable CUDA graphs. This adds ~5-10% latency per token but eliminates the crash entirely, allowing multi-agent concurrency.

The `--max-num-seqs` parameter controls how many requests vLLM batches simultaneously. Default: `10`. All requests run in parallel sharing GPU throughput (e.g. 10 concurrent requests ≈ 5-6 tok/s each instead of ~57 tok/s for a single one).

### Config changes don't take effect after editing `config.py`

The Gateway builds container args at creation time. If a container is already running (or gets restarted by Docker's `unless-stopped` policy), it keeps its original args. To apply new config:

```bash
# 1. Remove the container (stops and deletes it)
curl -X POST http://localhost:8000/admin/container/<container-name>/remove

# 2. Force-recreate with new args
curl -X POST "http://localhost:8000/admin/profile/<profile>?force=true"
```

This causes ~60-90s downtime while vLLM reloads the model.

### Container stuck in STARTING state
```bash
docker logs <container-name>
curl -X POST http://localhost:8000/admin/container/<container-name>/remove
```

### Qwen3-Coder-Next: "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER" AttributeError
Qwen3-Next is being served on `vllm-mxfp4-spark` instead of `blackwell-vllm`. Verify `config.py` has `QWEN3_CODER_NEXT_80B` using `container_image="blackwell-vllm:latest"`.

### Qwen3-Coder-Next: 2–3 tok/s instead of ~43 tok/s
vLLM PR #34279 causes slowness on GB10. Check that `fix_slowness.diff` is applied. Look for `Using TRITON Fp8 MoE backend` in container logs.

---

## License

Private — Alejandro Acho. All rights reserved.
