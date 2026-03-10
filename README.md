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

> **Note:** `focus` (GPT-OSS 120B) requires a custom vLLM image with CUTLASS MXFP4 kernels compiled for sm_121. Build it first: `docker build -t vllm-mxfp4-spark .` from [github.com/christopherowen/spark-vllm-mxfp4-docker](https://github.com/christopherowen/spark-vllm-mxfp4-docker) (~30 min). Expected throughput: 57–60 tok/s on single GB10.

Profile transitions are **automatic** — the Gateway detects visual keywords in requests and swaps models transparently.

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

## Quick Start (Docker — Recommended)

### 1. Clone and navigate

```bash
cd ~/Server
```

### 2. Download model weights

Install the Hugging Face CLI if you don't have it:

```bash
pip install -U huggingface_hub
huggingface-cli login   # You need a Hugging Face account (https://huggingface.co)
```

Create the model directories and download each model:

```bash
# Create all model directories
sudo mkdir -p /mnt/nvme_data/models/{gpt-oss-120b-q8,qwen3-coder-next-80b-q8,qwen3-coder-q8,flux2-pro-fp16,ltx-video-2-q8}
```

#### Focus Profile models

```bash
# GPT-OSS 120B (primary reasoning model)
# Repo: openai/gpt-oss-120b
sudo huggingface-cli download openai/gpt-oss-120b \
  --local-dir /mnt/nvme_data/models/gpt-oss-120b-q8

# Qwen3 Coder Next 80B MoE (primary coding model)
# Repo: Qwen/Qwen3-Coder-Next  (80B total, 3B activated — MoE)
# Requires vllm >= 0.15.0
sudo huggingface-cli download Qwen/Qwen3-Coder-Next \
  --local-dir /mnt/nvme_data/models/qwen3-coder-next-80b-q8
```

#### Creative Profile models

```bash
# Qwen3 Coder 30B (lighter coder for creative mode)
# Repo: Qwen/Qwen3-Coder-30B-A3B-Instruct  (30B total, 3B activated — MoE)
sudo huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --local-dir /mnt/nvme_data/models/qwen3-coder-q8

# FLUX.1 Dev (image generation — open-weight FLUX model)
# Repo: black-forest-labs/FLUX.1-dev
sudo huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir /mnt/nvme_data/models/flux2-pro-fp16

# LTX-Video (video generation)
# Repo: Lightricks/LTX-Video
sudo huggingface-cli download Lightricks/LTX-Video \
  --local-dir /mnt/nvme_data/models/ltx-video-2-q8
```

> **Note:** These downloads are large (50-150 GB each). Ensure your NVMe has enough space. Use `--resume-download` to continue interrupted downloads.

#### Model Reference Table

| Model | Hugging Face Repo | VRAM | Engine | Profile |
|-------|-------------------|------|--------|---------|
| GPT-OSS 120B | `openai/gpt-oss-120b` | ~84 GB (mxfp4 CUTLASS) | vLLM (custom) | `focus` |
| Qwen3 Coder Next 80B | `Qwen/Qwen3-Coder-Next` | ~95 GB (auto/gptq) | vLLM | `focus_code` |
| Qwen3 Coder 30B | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | ~35 GB (auto) | vLLM | Creative |
| FLUX.1 Dev | `black-forest-labs/FLUX.1-dev` | ~42 GB (FP16) | ComfyUI | `creative_image` |
| LTX-Video | `Lightricks/LTX-Video` | ~42 GB (Q8) | Diffusers | `creative_video` |

Expected directory layout after downloading:

```
/mnt/nvme_data/models/
├── gpt-oss-120b-q8/           # Must contain config.json + *.safetensors
├── qwen3-coder-next-80b-q8/   # Must contain config.json + *.safetensors
├── qwen3-coder-q8/            # Must contain config.json + *.safetensors
├── flux2-pro-fp16/            # Must contain config files + model weights
└── ltx-video-2-q8/            # Must contain config files + model weights
```

### 3. Configure environment

Edit `.env` to match your hardware:

```bash
# Key settings to review:
SYSTEM_RAM_GB=218        # Your actual system RAM (affects swap strategy)
MODELS_PATH=/mnt/nvme_data/models   # Path to your model weights
```

### 4. Build and launch

```bash
# Build the Gateway container
docker compose build gateway

# Start the Gateway (it manages inference containers dynamically)
docker compose up -d gateway
```

### 5. Verify it's running

```bash
# Check health
curl http://localhost:8000/health

# Check VRAM status
curl http://localhost:8000/status/vram

# Check active profile
curl http://localhost:8000/status/profile
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
├── .env                     # Environment variables
├── .gitignore
├── Dockerfile               # Gateway container image
├── docker-compose.yml       # Full stack definition
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── gateway/
    ├── __init__.py           # Package init + version
    ├── app.py                # FastAPI application (main entry point)
    ├── config.py             # Central configuration + VRAM profiles
    ├── orchestrator.py       # Docker container lifecycle manager
    ├── proxy.py              # HTTP proxy to inference backends
    ├── request_buffer.py     # Request queue + Radix Prefix Cache
    ├── router.py             # Smart routing + trigger detection
    ├── schemas.py            # Pydantic data models
    └── vram_monitor.py       # nvidia-smi VRAM monitoring
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

---

## License

Private — Alejandro Acho. All rights reserved.
