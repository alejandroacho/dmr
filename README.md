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

## Media Node (node 3) — video+audio, music, images

A third GB10 that serves **media only** and runs **standalone** — it is not part
of the text cluster. No text models, no VRAM profiles, no container swapping, no
Docker socket. Its own Gateway sits in front, so agents talk the same API they
use elsewhere.

```
                    clients / agents
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  node 3 — standalone media node                            │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  media-gateway  :8000   (gateway.media_app)          │  │
│  │    POST /v1/av/generate      → video + audio         │  │
│  │    POST /v1/audio/music      → music                 │  │
│  │    POST /v1/images/generate  → images                │  │
│  │    GET  /v1/models  /health  /status/vram  /docs     │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │ http://media-node:8010          │
│  ┌───────────────────────▼──────────────────────────────┐  │
│  │  media-node     :8010   (media_server.py)            │  │
│  │    builds a ComfyUI graph per request                 │  │
│  │                    │                                  │  │
│  │  ComfyUI  :8188 ◀──┘  (web UI)                       │  │
│  │    MiniMax-H3 · ACE-Step 1.5 · Qwen-Image-2.1            │  │
│  │    INT8-convrot, NVFP4 and FP8 ops, all native       │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

One ComfyUI serves all three families and evicts between them as needed, so
switching modality costs a reload but never a redeploy.

| Modality | Model | Weights | Endpoint |
|---|---|---|---|
| video + audio | MiniMax-H3 (t2va/fl2va/ref2va) | ~63 GB | `/v1/av/generate` |
| music | ACE-Step 1.5 XL Turbo | ~20 GB | `/v1/audio/music` |
| image | Qwen-Image-2.1 (INT8) | DiT + encoder + VAE | `/v1/images/generate` |

`gateway/media_app.py` deliberately does **not** import `ContainerOrchestrator`.
The main app's startup adopts or recreates VRAM profiles and force-removes
"orphaned" containers — on this node that would delete the container serving the
models. No Docker socket is mounted either, so it could not do so if asked.

### Weights

**MiniMax-H3** — `Comfy-Org/MiniMax-H3`, `just download-minimax`:

| File | Size | Role |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot` | 19.53 GiB | t2va + first/last-frame DiT |
| `minimax_h3_ref2va_pruned_int8_convrot` | 19.53 GiB | omni-reference DiT |
| `qwen3vl_32b_minimax_h3_nvfp4_awq` | 14.61 GiB | conditioning encoder (Qwen3-VL-32B) |
| `minimax_h3_video_vae_fp16` | 4.85 GiB | video VAE |
| `minimax_h3_audio_vae_fp32` | 0.56 GiB | audio VAE |

The FL2VA set (DiT + encoder + both VAEs) is 42.47 GB; ref2va adds 20.97 GB and
reuses the encoder and VAEs.

**ACE-Step 1.5 XL Turbo** — `Comfy-Org/ace_step_1.5_ComfyUI_files`, `just download-ace`:

| File | Size | Role |
|---|---|---|
| `acestep_v1.5_xl_turbo_bf16` | 9.29 GiB | DiT (8-step turbo) |
| `qwen_0.6b_ace15` + `qwen_4b_ace15` | 8.91 GiB | both are required (DualCLIPLoader, type `ace`) |
| `ace_1.5_vae` | 0.31 GiB | audio VAE (DCAE + vocoder) |

**Qwen-Image-2.1** — [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), `just download-qwen-image`:

- `diffusion_models/qwen_image_2.1_int8_convrot.safetensors`
- `text_encoders/qwen3vl_8b_int8_convrot.safetensors`
- `vae/qwen_image_2.1_vae_bf16.safetensors`

All three files are required. Rebuild with `just build-media` using a current
ComfyUI revision with `TextEncodeQwenImage21`, then run `just media-up` on the media node.
The download retains the old HiDream weights; they can be removed separately if unused.

ComfyUI web UI: `http://192.168.1.86:8188/` (published by `media-node`).
Import `models/qwen-image-2.1/workflow.json` for the official Qwen image workflow.
Saved workflows and UI settings persist in the `media_user` volume.
The adapter continues to use `http://127.0.0.1:8188` inside the container.

### Task modes (video + audio)

The node picks the mode — and therefore the checkpoint — from the request:

| Mode | Trigger | Inputs |
|---|---|---|
| `t2va` | prompt only | — |
| `fl2va` | `first_frame` and/or `last_frame` | 1–2 keyframes |
| `ref2va` | any `ref_*` field | ≤9 images, ≤3 videos, ≤3 video soundtracks, ≤3 audios; adapter also caps the total at 12 |

### Setup (on node 3)

```bash
just download-media      # all three families, ~99 GB → ~/Models/
just build-media         # ComfyUI + CUDA 13 image (~15 min)
just media-up            # adapter :8010 + its Gateway :8000
just media-status        # 503 until ComfyUI is up (~15s), then 200
just media-test          "a lighthouse in a storm"      # → h3-test.mp4
just media-test-music    "lofi hip hop, mellow piano"   # → music-test.mp3
just media-test-image    "a noir portrait, 35mm film"   # → image-test.png
```

The build needs `gcc` and `python3-dev` in the image: Triton JIT-compiles kernels
at request time for the NVFP4 encoder and the INT8/convrot ops, shelling out to
`cc` and linking against `Python.h`. Without them the graph fails **mid-request**
with `Failed to find C compiler` — the container still starts and reports healthy.

| Variable | Default | Description |
|---|---|---|
| `MEDIA_NODE_URL` | `http://192.168.1.86:8010` | Adapter address (`http://media-node:8010` in compose) |
| `MEDIA_NODE_TIMEOUT_S` | `0` | Per-request ceiling; `0` = none (see below) |
| `GENERATE_TIMEOUT_S` | `0` | *(adapter)* Whole-generation ceiling; `0` = none |
| `COMFY_HTTP_TIMEOUT_S` | `600` | *(adapter)* Ceiling on one call to ComfyUI |
| `MEDIA_ASSET_DIR` | *(unset)* | Enables `response_format: "url"` |
| `MEDIA_PUBLIC_URL` | `GATEWAY_PUBLIC_URL` | Base URL for saved assets |
| `MEDIA_NODE_ALIAS_VIDEOS` | `false` | Also answer `POST /v1/videos/generate` (forced on in `media_app`) |
| `NODE_NAME` | `media-node` | Label reported by `/health` |

**Both generation ceilings default to none, on purpose.** A 15s clip at the full
1344x768 canvas runs ~55 min in a single pass, so the previous 1800s value cut off
exactly the requests it existed to protect — and cut them off *after* the GPU had
done the work. It also failed confusingly: the Gateway's ceiling raised
`asyncio.TimeoutError`, which is not an `aiohttp.ClientError`, so it escaped as a
bare 500. (A ceiling set explicitly now returns 504 and says which knob fired.)

Nothing waits forever as a result, because neither hop relies on a clock:

- the caller's disconnect is polled on both hops, and dropping the client cancels
  the ComfyUI job in ~1.5s (see *Request limits and cancellation*);
- if ComfyUI dies, the entrypoint takes the container down with it, so the
  Gateway's connection drops and the request fails as a 503;
- `COMFY_HTTP_TIMEOUT_S` still bounds every *individual* call to ComfyUI — a
  `/history` poll, a queue edit, reading a finished file off loopback. That knob
  is separate from the generation deadline precisely because sharing one value
  between a 55-minute job and a 20 ms poll is what made "no ceiling"
  inexpressible.

For reference: ~12.6 min for a 5s clip, ~55 min for 15s.

### File handling and retention

Four separate paths, three of which are swept:

| What | Where | Swept? |
|---|---|---|
| Model weights | `~/Models/*` → `/models:ro` | no — read-only input |
| Uploaded references | ComfyUI `input/in_*` (volume `media_input`) | **yes** |
| Generations | ComfyUI `output/media/` (volume `media_output`) | **yes** |
| Saved assets (`url` mode) | Gateway `/assets` (volume `media_assets`) | **yes** |

The adapter never reads ComfyUI's filesystem: it takes `{filename, subfolder,
type}` from `/history/{id}` and fetches the bytes over `GET /view`, so the Gateway
needs no access to ComfyUI's volume.

Both processes sweep at startup and then hourly, deleting anything older than
`MEDIA_RETENTION_HOURS` (default 24, `0` disables). The sweep is deliberately
narrow — only `output/media/` and `input/in_*`, never ComfyUI's own bundled inputs
or its output-dir marker file.

> **`url` mode has a lifetime.** A URL handed to a client stops resolving once its
> file is swept, so a transcript keeps a dead link rather than the content. With
> `b64_json` the bytes travelled in the response, and whether they are kept is the
> caller's business. Raise `MEDIA_RETENTION_HOURS` on the Gateway if callers are
> expected to come back for old results.

> **Readiness vs. loaded:** `/health` turns 200 once ComfyUI is up with the H3
> nodes registered — the ~63 GB of weights load lazily on the **first generation
> request**, which therefore takes ~40s longer than later ones. The models then
> stay resident.

> **VRAM reporting caveat:** inside the container NVML reports a 512 GB total and
> `used: 0` on GB10 — its unified memory isn't visible that way from a container.
> Temperature and utilization are correct. For real memory figures run
> `nvidia-smi` on the host.

### Optional: attaching it to another Gateway instead

If you later want the media node reachable *through* the main Gateway rather than
directly, `gateway/media_node.py` is a self-contained `APIRouter` — it imports
nothing from the rest of the package and touches no shared state. Two lines in
that Gateway's `app.py`, **after** its own routes are defined:

```python
from gateway import media_node
media_node.attach(app)          # adds /v1/av/generate + /status/media-node
```

then start it with `MEDIA_NODE_URL=http://<node-3>:8010`.

> Note that a Gateway on a different subnet may not be able to reach the media
> node: if the path crosses a NAT'ing router, traffic only flows outward from the
> media node. Check with `curl http://<node-3>:8010/health` from that Gateway's
> host before wiring it up.

### Request limits

| Parameter | Cap | Why |
|---|---|---|
| `num_frames` | 362 (`AV_MAX_FRAMES`) | H3's trained range is ~124-362; beyond it the output degrades and the cost explodes |
| `width x height` | 1,032,192 px (`AV_MAX_PIXELS`) | H3's native canvas, e.g. 1344x768 |
| `duration` (music) | 600s | keeps one request from owning the GPU indefinitely |
| `width`/`height` (image) | 4096 | the node's own limit |

Cost is driven by video-latent tokens (`T x H/16 x W/16`), not pixels, and
attention is superlinear in that. 999 frames at full canvas is ~1.19M tokens
against the default's 149k — a 12-18 hour job. Such requests are now rejected
with a 422 in milliseconds; anything past 2x the default token count is logged
as expensive.

Abandoning a request cancels the work: the Gateway watches for the caller
disconnecting, drops its upstream call, and the node cancels the ComfyUI job
(~1.5s end to end). Note that ComfyUI checks its interrupt flag only between
sampling steps — if a step itself takes tens of minutes, restart the container.

### Generation defaults

Native canvas is a 768px short edge capped at 768×1344, each axis a multiple of
32. Frame counts snap up to the model's `17k+5` grid at 24 fps (124 ≈ 5.2s;
trained range ≈ 124–362).

| Parameter | Default | Note |
|---|---|---|
| `steps` | `20` | H3's documented default |
| `sampler` / `scheduler` | `res_multistep` / `simple` | |
| `cfg_scale` | `1.0` | No CFG — one forward pass per step. Raise to 3–6 only if prompt adherence is weak |
| `shift_video` / `shift_audio` | `12.0` / `3.0` | ComfyUI `MiniMaxH3SigmaShift` node defaults |

### Measured throughput (single GB10, 20 steps, no Sage Attention)

| Resolution | Frames | Video latent tokens | Time |
|---|---|---|---|
| 1344×768 (default) | 124 (5.2s) | ~149k | **17m 33s** |
| 768×448 | 124 (5.2s) | ~50k | **3m 42s** |
| 512×320 | 5 | ~24k | ~20s (2 steps) |

Cost is dominated by attention over the video latent (`T×H/16×W/16` tokens, where
124 frames → T=37), so it grows far faster than pixel count: 3× the tokens cost
4.7× the time. **768×448 is the practical operating point**; the default canvas is
for final renders. Generation is compute-bound — no offloading occurs, the models
stay resident, and requests are serialized by a lock in the adapter.

Untried lever: Sage Attention, which H3's docs say roughly doubles speed. It is
deliberately not in the image — installing it requires matching the exact
PyTorch/CUDA build, and it's the wrong variable to add while bringing a pipeline up.

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
| `POST` | `/v1/av/generate` | Video **with native stereo audio** (MiniMax-H3, media node) |
| `POST` | `/v1/audio/music` | Music (ACE-Step 1.5 XL Turbo, media node) |
| `GET`  | `/v1/models` | List available models (OpenAI format) |

### System Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Gateway health check + cluster status |
| `GET` | `/status/vram` | Detailed VRAM report from nvidia-smi |
| `GET` | `/status/swap` | Current swap status (in progress, elapsed, queue) |
| `GET` | `/status/profile` | Active profile and loaded models |
| `GET` | `/status/cache` | Radix Prefix Cache statistics |
| `GET` | `/status/media-node` | Reachability and readiness of the media node (node 3) |

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

### Video + native audio (MiniMax-H3, media node)

> These run against the **media node's own Gateway** (node 3, `:8000`). Substitute
> its address — e.g. `http://192.168.1.86:8000` — from another machine.

```bash
curl -X POST http://localhost:8000/v1/av/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A lighthouse in a storm, waves crashing, thunder rolling",
    "num_frames": 124,
    "steps": 20
  }'
```

Response: `{"success": true, "data": {"video_base64": "<mp4 with stereo audio>", "mode": "t2va", "seed": ..., ...}}`

Or `just media-test "your prompt"` to write `h3-test.mp4` directly.

**First/last frame (fl2va)** — send one or both keyframes as base64:

```bash
curl -X POST http://localhost:8000/v1/av/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"the camera pulls back slowly\",
       \"first_frame\": \"$(base64 -w0 start.png)\",
       \"last_frame\": \"$(base64 -w0 end.png)\"}"
```

**Omni-reference (ref2va)** — reference images/videos/audio, addressed in the prompt
as `<Picture i>` / `<Video k>` / `<Audio j>` (1-based, per type):

```bash
curl -X POST http://localhost:8000/v1/av/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"<Picture 1> walks through the market in <Picture 2>\",
       \"ref_images\": [\"$(base64 -w0 person.png)\", \"$(base64 -w0 market.png)\"]}"
```

Sending any `ref_*` field switches node 3 to the ref2va checkpoint automatically.
Limits, transcribed from the Autogrow templates in
`comfy_extras/nodes_minimax_h3.py`: **9** `ref_images`, **3** `ref_videos`, **3**
`ref_video_audios`, **3** `ref_audios`. `ref_video_audio_N` is the soundtrack *of*
`ref_video_N` — the node pairs them by index and silently ignores any with no
matching video, so the adapter rejects that case rather than letting you pay for
an upload that gets dropped. When a video's soundtrack is not supplied, its own
audio track is reused.

The adapter additionally caps the **total** at 12 files. Note that ceiling is not
in the node schema, whose per-container maxima add up to 18 — so unless it comes
from H3's model card it is stricter than ComfyUI. It is left in place rather than
loosened on a guess; raise it if you have the reference that says 18 is fine.

> For long clips prefer `"response_format": "url"` over base64 — set `MEDIA_ASSET_DIR`
> on the Gateway and it returns a `video_url` instead of ~50 MB of JSON. Available
> on all three endpoints.

### Music (ACE-Step 1.5 XL Turbo)

`prompt` is a list of style tags, not prose — ACE-Step is not a natural-language
model. Leave `lyrics` empty for an instrumental.

```bash
curl -X POST http://localhost:8000/v1/audio/music \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "lofi hip hop, mellow piano, vinyl crackle, relaxed",
    "duration": 20,
    "bpm": 85,
    "key_scale": "F major"
  }'
```

Response: `{"success": true, "data": {"audio_base64": "<mp3>", "duration": 20.0, ...}}`

XL Turbo samples in 8 steps, so a 20s clip takes ~16s including the cold model
load. Note the two separate CFG knobs: `cfg_scale` (diffusion, default 1.0) and
`lm_cfg_scale` (the audio-code LM, default 2.0) — both come from the blueprint.

### Images (Qwen-Image-2.1)

```bash
curl -X POST http://localhost:8000/v1/images/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a noir portrait of a lighthouse keeper, 35mm film"}'
```

Response: `{"success": true, "data": {"images": ["<base64 PNG>"], "model": "qwen-image-2.1", ...}}`

Defaults: 25 steps, CFG 1, Euler sampler, `simple` scheduler, 2048×2048 canvas.
Override `steps`, `cfg_scale`, `scheduler`, `width`, `height` and `batch_size` as needed.
The old HiDream `variant` and `noise_scale` fields are no longer accepted.
The `image` alias now resolves to `qwen-image-2.1`.
Reference images enable editing (up to 10), with reference encoding at 1024px
and output dimensions controlled by `width`/`height`; matching the reference's
aspect ratio helps preserve its layout. PNG output preserves generated transparency.

```bash
curl -X POST http://localhost:8000/v1/images/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"make the background a neon city at night\",
       \"ref_images\": [\"$(base64 -w0 portrait.png)\"]}"
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
├── Dockerfile.media            # Media node image (ComfyUI + CUDA 13)
├── docker-compose.yml          # Full stack definition
├── justfile                    # Task runner
├── requirements.txt            # Python dependencies
├── inference/
│   ├── flux_server.py          # FLUX.1-dev FastAPI server (port 8004)
│   ├── ltx_server.py           # LTX-Video 2 FastAPI server (port 8005)
│   └── media_server.py         # Media node ComfyUI adapter (port 8010)
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
│   ├── flux2-pro/
│   │   ├── Dockerfile          # Reference → Dockerfile.comfyui at root
│   │   └── download.sh         # hf download black-forest-labs/FLUX.1-dev (~34 GB)
│   ├── minimax-h3/
│   │   └── download.sh         # Comfy-Org/MiniMax-H3 (~63 GB)
│   ├── ace-step-1.5/
│   │   └── download.sh         # ACE-Step 1.5 XL Turbo (~20 GB)
│   └── qwen-image-2.1/
│       └── download.sh         # Qwen-Image-2.1 (DiT + encoder + VAE)
└── gateway/
    ├── app.py                  # FastAPI application (main entry point)
    ├── config.py               # Central configuration + VRAM profiles
    ├── media_app.py            # Media-only Gateway — entry point on node 3
    ├── media_node.py           # MiniMax-H3 routes — standalone, drop-in APIRouter
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
| `MEDIA_NODE_HOST` | `192.168.1.86` | Media node (node 3) address |
| `MEDIA_NODE_PORT` | `8010` | Media node adapter port |
| `MEDIA_NODE_URL` | `http://192.168.1.86:8010` | Full media node URL used by `media_node.py` |

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
