# Blackwell Orchestrator & Smart Gateway

FastAPI middleware that fronts a two-node NVIDIA DGX Spark (GB10 Blackwell) cluster, serving two large text models and swapping between them on demand.

The catalog is deliberately small: **DeepSeek-V4-Flash**, **Qwen3.5-122B** and **Qwen3.8-Flash-Next**. All three are too large for one node, so all three shard across the pair with tensor parallelism.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    External Agents                       │
│              (OpenAI-compatible, text only)              │
└──────────────────────┬───────────────────────────────────┘
                       │  HTTP / JSON
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Smart Gateway  (FastAPI :8000)              │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │ Smart Router│  │  VRAM Monitor│  │ Request Buffer │   │
│  │  (by model) │  │  (nvidia-smi)│  │ (Long Polling) │   │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘   │
│         │                │                   │           │
│  ┌──────▼────────────────▼───────────────────▼────────┐  │
│  │              Container Orchestrator                │  │
│  │        Mutex-protected profile swapping            │  │
│  └───────┬──────────────────────────────┬─────────────┘  │
└──────────┼──────────────────────────────┼────────────────┘
           │ Docker socket                │ SSH
           ▼                              ▼
┌────────────────────────┐   ┌────────────────────────────┐
│  HEAD  192.168.200.12  │   │  WORKER  192.168.200.13    │
│  container: vllm_node  │◄─►│  container: vllm_node      │
│  (host net, sleep ∞)   │IB │  (host net, sleep ∞)       │
│                        │   │                            │
│  vllm serve rank 0     │   │  vllm serve rank 1         │
│  :8020 DeepSeek        │   │  --headless                │
│  :8021 Qwen3.5         │   │                            │
└────────────────────────┘   └────────────────────────────┘
      GB10 · 128 GB                 GB10 · 128 GB
```

### The key idea: the Gateway owns processes, not containers

The two `vllm_node` containers are created and owned by **[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)**'s `launch-cluster.sh`, not by the Gateway. Their foreground process is `sleep infinity`, and each model runs as an exec'd child process inside them.

That split is deliberate. Those containers need patches (mods) applied at launch time and host networking across both nodes — things a generic Docker orchestrator cannot reproduce. So the Gateway starts and stops the `vllm serve` processes and **never creates, removes, pauses, or recreates the containers**. This is the `spark_cluster` engine in `gateway/config.py`.

Consequences worth knowing:

- Rank 0 runs on the head and is reached through the local Docker socket. Ranks ≥ 1 live on another machine's Docker daemon and are reached over **SSH**.
- A Gateway restart re-adopts a healthy model without reloading weights (~0s instead of minutes).
- If the cluster containers are down, a profile swap fails fast with the exact command to bring them back.

---

## Models & Profiles

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
│  │  ComfyUI  :8188 ◀──┘  (loopback only)                │  │
│  │    MiniMax-H3 · ACE-Step 1.5 · HiDream-O1            │  │
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
| image | HiDream-O1-Image (dev + base) | ~16 GB | `/v1/images/generate` |

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

**HiDream-O1-Image** — `Comfy-Org/HiDream-O1-Image`, `just download-hidream`:

| File | Size | Role |
|---|---|---|
| `hidream_o1_image_dev_fp8_scaled` | 7.51 GiB | dev — all-in-one, 28 steps, no CFG |
| `hidream_o1_image_fp8_scaled` | 7.51 GiB | base — all-in-one, 40 steps, CFG 5 |

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
| `MEDIA_NODE_URL` | `http://192.168.8.147:8010` | Adapter address (`http://media-node:8010` in compose) |
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

| Profile key | Model | Sharding | Port | Context |
|---|---|---|---|---|
| **`deepseek`** ⭐ | DeepSeek-V4-Flash-0731 — 284B total / 13B active MoE, FP4 experts + FP8 dense | Native multi-node, TP=2 (2 processes) | 8020 | 1,048,576 |
| **`qwen35`** | Qwen3.5-122B-A10B-FP8 — native FP8 | Ray, TP=2 (1 process) | 8021 | 262,144 |
| **`qwen38`** | Qwen3.8-Flash-Next-NVFP4 — NVFP4 + MXFP8 mixed, hybrid attention/SSM | Native multi-node, TP=2 (2 processes) | 8022 | 262,144 |

`deepseek` is the default at startup.

### Why these three

DeepSeek-V4-Flash is the reason the cluster exists: 167 GB of weights, ~220 GB of footprint across both nodes, and dspark speculative decoding. Qwen3.5-122B-A10B-FP8 is the second-opinion model — Qwen's own FP8 quantization (not a third-party int4 requant), ~127 GB, comfortable at ~64 GB per node with room for a 256K KV cache.

Qwen3.8-Flash-Next-NVFP4 is the newest of the three: ~106 GB in NVIDIA's mixed NVFP4/MXFP8 quantization, a hybrid attention/SSM architecture with MTP speculative decoding, and — unlike Qwen3.5 — no Ray and no mods at all. It is multimodal upstream; the Gateway only ever sends it text.

Two Qwen3.5 recipes were rejected: `qwen3.5-397b-int4-autoround` is labeled EXPERIMENTAL upstream and its 226 GB leave almost nothing for KV cache within the 256 GB the pair has; `qwen3.5-122b-int4-autoround` is a lossy requant whose only advantage — fitting on one node — is irrelevant here.

### The models need different cluster modes

This is the sharpest operational edge in the whole setup:

| | DeepSeek | Qwen3.5 | Qwen3.8-Flash-Next |
|---|---|---|---|
| Distribution | vLLM native (`--nnodes/--node-rank`, one `--headless` rank per worker) | Ray (`--distributed-executor-backend ray`, single process) | vLLM native, same as DeepSeek |
| Requires Ray inside the containers | No | **Yes** | No |
| Mod | none since 2026-09 (upstream moved it to the built-in `b12x` loader) | `fix-qwen3.5-chat-template` (drops a jinja file in `/workspace`) | none (`mods: []`) |

Only Qwen3.5 still asks for a mod, and it merely copies a file, so a single container launch can serve all three models. Ray mode does not: DeepSeek and Qwen3.8 ignore a running Ray cluster, but Qwen3.5 hard-fails without one. Launch the containers in Ray mode if you want to swap freely to Qwen3.5 as well.

The Gateway checks this before every Qwen3.5 swap and refuses with a clear message rather than starting a process that would die. Qwen3.8 needs no such check.

---

## Prerequisites

| Requirement | Minimum |
|---|---|
| **Nodes** | 2× NVIDIA DGX Spark (GB10 Blackwell), 128 GB unified memory each |
| **Interconnect** | ConnectX-7 direct link between the nodes (see [docs/NETWORKING.md](https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md) in the spark repo) |
| **SSH** | Passwordless from head to worker |
| **OS** | Linux aarch64, NVIDIA driver 580+ / CUDA 13 |
| **Docker** | Engine 24+ with Compose V2 and NVIDIA Container Toolkit, on **both** nodes |
| **Storage** | ~170 GB per model, per node |
| **Python** | 3.10+ (local development only) |

---

## Cluster Setup

Everything below runs on the **head node**. For the other end — what must exist
on the worker, what is deployed there from this repo, and the two things that
are *not* in git and would be lost on a reinstall — see [NODE2.md](NODE2.md).

### 1. Clone the launcher and install `uv`

```bash
git clone https://github.com/eugr/spark-vllm-docker.git ~/spark-vllm-docker
curl -LsSf https://astral.sh/uv/install.sh | sh   # hf-download.sh calls `uvx hf download`
```

### 2. Pin the cluster configuration

Autodiscovery is bypassed on purpose here. This host has **four** CX7 links with two interfaces per subnet (`192.168.100.0/24` and `192.168.200.0/24`), and `autodiscover.sh` rejects duplicate subnets — correctly, since it makes routing ambiguous. Instead, `~/spark-vllm-docker/.env` pins the nodes and interfaces:

```bash
CLUSTER_NODES="192.168.200.12,192.168.200.13"
ETH_IF="enp1s0f1np1"                    # holds .12 here and .13 there
IB_IF="rocep1s0f1,roceP2p1s0f1"         # port 1 of both CX7 cards, 2 RDMA rails
LOCAL_IP="192.168.200.12"
MASTER_PORT="29501"
COPY_HOSTS="192.168.200.13"
```

> Re-addressing the four links so each gets its own `/24` (per the spark repo's networking guide) would fix routing properly and re-enable autodiscovery. It needs root and would break the current `192.168.200.x` references.

### 3. Pull the B12X runner image onto both nodes

```bash
cd ~/spark-vllm-docker
./build-and-copy.sh --exp-b12x -c
```

`--exp-b12x` pulls the prebuilt, upstream-tested `eugr/spark-vllm-b12x:latest` (~23 GB) and tags it `vllm-node-b12x`. Nothing is compiled — older instructions that build vLLM and a FlashInfer PR from source are obsolete.

### 4. Download weights to `~/hf-cache` on both nodes

**Use `HF_HOME=~/hf-cache`, not the default cache.** On this cluster the worker's `~/.cache/huggingface/hub` is owned by root (a container created it) and holds another user's models, so rsync cannot write there. `~/hf-cache` is user-owned on both nodes and needs no root.

```bash
mkdir -p ~/hf-cache/hub
ssh 192.168.200.13 'mkdir -p ~/hf-cache/hub'

cd ~/spark-vllm-docker
HF_HOME=~/hf-cache ./hf-download.sh deepseek-ai/DeepSeek-V4-Flash-0731 -c   # ~167 GB
HF_HOME=~/hf-cache ./hf-download.sh Qwen/Qwen3.5-122B-A10B-FP8 -c           # ~127 GB
```

`-c` copies to `COPY_HOSTS` over the fast link (~575 MB/s in practice, ~5 min for DeepSeek).

> If a model is already in the default cache, hardlink it instead of downloading twice — same filesystem, zero extra space:
> `cp -al ~/.cache/huggingface/hub/models--org--name ~/hf-cache/hub/`

### 5. Launch the cluster containers

```bash
cd ~/spark-vllm-docker
HF_HOME=~/hf-cache ./run-recipe.sh deepseek-v4-flash-0731 --port 8020 -d
```

This applies the mods, brings up `vllm_node` on both nodes, and starts the model. From here the Gateway takes over the process lifecycle.

### 6. Give the Gateway SSH access to the worker

The Gateway needs to start the headless rank on the worker, which lives on another Docker daemon. Use a **dedicated** key, not your personal one:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_spark_gateway -C "blackwell-gateway->spark-worker"
ssh-copy-id -f -i ~/.ssh/id_spark_gateway.pub 192.168.200.13
```

`docker-compose.yml` mounts it read-only at `/ssh/id_spark`.

> **Security:** this key lets the Gateway container run `docker` on the worker — root-equivalent access there. It is the unavoidable cost of managing a remote rank. To narrow it, restrict the key with a `command=` prefix in the worker's `authorized_keys`.

### 7. Start the Gateway

```bash
cd ~/Server
docker compose build gateway && docker compose up -d gateway
make health
```

### 8. Make the cluster survive reboots

**Without this step DeepSeek does not come back after a power cut or a reboot.** `launch-cluster.sh` creates the containers with `docker run --rm`, which Docker refuses to combine with a restart policy — so `vllm_node` runs with `restart=no` and a container that dies is removed and never recreated. Nothing on the worker recreates its own container either: the head owns the launch, so the head has to notice.

`ray-cluster/ensure_vllm_cluster.sh` closes that gap. It defaults to exactly the DeepSeek launch of step 5 (`RECIPE=deepseek-v4-flash-0731`, `PORT=8020`) and is idempotent — a complete cluster is left strictly alone, so it is safe from both a boot unit and a timer:

```bash
sudo cp ~/Server/ray-cluster/vllm-cluster.service /etc/systemd/system/
sudo cp ~/Server/ray-cluster/vllm-cluster.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-cluster.timer
```

Verify:

```bash
systemctl list-timers vllm-cluster.timer
journalctl -u vllm-cluster.service -n 20
```

The timer fires 90 s after boot and re-checks every 2 minutes. Two behaviours are worth knowing before you rely on it:

- **A partial cluster is torn down and relaunched whole.** The launcher only checks whether *some* container is running and then skips the launch, which would leave the node that lost its container without a rank. A TP=2 model cannot serve on one rank, so the script stops **both** first.
- **An unreachable worker is left alone.** A network blip must never be read as "the worker is gone" — tearing down a healthy cluster costs a full 167 GB weight reload for nothing.

Run it by hand at any time (it exits in ~1 s when all is well):

```bash
bash ~/Server/ray-cluster/ensure_vllm_cluster.sh
```

> The service runs as `alejandroacho`, not root: it needs that user's SSH key to reach the worker and its `docker` group membership locally. It only ensures the **containers** are up — the model itself then loads in the background (~3–5 min), and the Gateway re-adopts it without reloading weights.

---

## Local Development (Without Docker)

```bash
cd ~/Server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn gateway.app:app --host 0.0.0.0 --port 8000 --reload
```

> Requires a reachable Docker socket (the Gateway drives containers through it) and, for Spark models, an `ssh` binary plus a readable `SPARK_SSH_KEY`.

Tests mock Docker, NVML and SSH entirely — no GPU or cluster needed:

```bash
pytest tests/ -v
```

Because the shipped catalog holds only `spark_cluster` models, tests that exercise the generic Docker container lifecycle use a **synthetic catalog** registered by `tests/conftest.py` (`TEST_MODEL_A/B/C`, `TEST_PROFILE_A/B`). Add new Docker-managed fixtures there rather than coupling tests to the real catalog.

---

## API Endpoints

### Inference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible. Streaming supported. Swaps profile if the requested model belongs to another one. |
| `POST` | `/v1/av/generate` | Video **with native stereo audio** (MiniMax-H3, media node) |
| `POST` | `/v1/audio/music` | Music (ACE-Step 1.5 XL Turbo, media node) |
| `POST` | `/v1/images/generate` | Images (HiDream-O1, media node) |
| `GET` | `/v1/models` | Available models plus the active profile's label aliases (`chat`, `code`) and the media-node models |

### System status

| Method | Endpoint | Description |
|---|---|---|
|| `GET` | `/health` | Health, active profile mode, per-container state, and Ray cluster state (`ray`) when — and only when — a loaded model needs Ray. `status` is `degraded` if that cluster is short on nodes |
|| `GET` | `/status/vram` | Memory report for the **local node only** — cluster models are sharded, so the other node's half is not counted. On unified-memory hosts the figures come from the host's `MemAvailable` (see below), not from NVML |
|| `GET` | `/status/swap` | Swap in progress, elapsed, queue depth |
|| `GET` | `/status/profile` | Active profile and its models |
|| `GET` | `/v1/profiles` | All profiles |
|| `GET` | `/v1/profiles/active` | Active profile with its registry key — the reliable source for "what is loaded" |
|| `GET` | `/status/cache` | Radix prefix cache statistics |
| `GET` | `/status/media-node` | Reachability and readiness of the media node (node 3) |

### Administration

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/admin/profile/{name}` | Switch profile (`deepseek`, `qwen35`, `qwen38`). Add `?force=true` to restart the active one. |
| `POST` | `/admin/container/{name}/stop` | Stop a Docker-managed workload |
| `POST` | `/admin/container/{name}/remove` | Remove a Docker-managed workload |

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Usage Examples

### Text completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Explain MoE routing in one paragraph"}],
    "max_tokens": 4000
  }'
```

Or use a label that follows the active profile: `"model": "chat"`.

### Video + native audio (MiniMax-H3, media node)

> These run against the **media node's own Gateway** (node 3, `:8000`). Substitute
> its address — e.g. `http://192.168.8.147:8000` — from another machine.

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

### Images (HiDream-O1)

```bash
curl -X POST http://localhost:8000/v1/images/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a noir portrait of a lighthouse keeper, 35mm film"}'
```

Response: `{"success": true, "data": {"images": ["<base64 PNG>"], "variant": "dev", ...}}`

Two variants, whose sampling settings come from the official templates — you
normally only pick the variant and leave the rest alone:

| `variant` | Steps | CFG | Sampler | 1024² |
|---|---|---|---|---|
| `dev` (default) | 28 | 1.0 | `SamplerLCM` | ~12s |
| `base` | 40 | 5.0 | `dpmpp_2m_sde_gpu` + seam smoothing | ~30s |

Native canvas is 2048×2048. Reference images enable HiDream-O1's editing mode —
1 image is an instruction edit, 2–10 is multi-reference:

```bash
curl -X POST http://localhost:8000/v1/images/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"make the background a neon city at night\",
       \"ref_images\": [\"$(base64 -w0 portrait.png)\"]}"
```

### Manual profile switch

```bash
curl -X POST http://localhost:8000/admin/profile/deepseek
curl -X POST http://localhost:8000/admin/profile/qwen35
```

A swap tears down the current model's processes and starts the new ones. Expect **2-5 minutes**, not seconds — `LONG_POLLING_ENABLED=true` holds client connections meanwhile.

### ⚠️ Give DeepSeek room to think

The recipe sets `reasoning_effort=high`, so the model reasons at length before answering, and the reasoning is billed against `max_tokens`. With a tight budget you get an **empty `content`** and `finish_reason: "length"` — all tokens went to reasoning. Use generous `max_tokens` (4000+), or lower the effort per request.

The chain of thought arrives in the **`reasoning`** field of the message, not `reasoning_content`.

---

## Swap Behavior

| Config | Behavior |
|---|---|
| `LONG_POLLING_ENABLED=true` | Connection held; response sent once the swap completes |
| `LONG_POLLING_ENABLED=false` | Immediate `HTTP 503` with `Retry-After` |

Docker-managed models still honor the pause/unpause vs stop/start strategy (`SYSTEM_RAM_GB ≥ 512` picks pause/unpause), but the two catalog models are `spark_cluster`: swapping them means killing and relaunching `vllm serve`, dominated by weight loading.

Measured on this cluster: DeepSeek reaches `READY` in **~165 s** from a warm container, **~285 s** cold. InstantTensor loads its 72,317 tensors at ~3,500/s, and CUDA graph capture adds ~25 s.

---

## Memory Accounting on Unified-Memory Hosts

The GB10 has no dedicated VRAM: the GPU allocates from the same LPDDR the OS
uses. Two NVML paths that work on discrete GPUs both fail here:

- `nvmlDeviceGetMemoryInfo` returns **`Not Supported`** (so does
  `nvidia-smi --query-gpu=memory.used`, which prints `N/A`).
- Per-process accounting only sees processes in the caller's PID namespace.
  The Gateway runs in its own container while `vllm serve` runs in
  `vllm_node`, so it reads back an **empty** process list.

Together those made the Gateway report 0 MB used and a fully free GPU while
vLLM held ~101 GiB. The memory figures therefore come from the host's
`/proc/meminfo`, bind-mounted at `HOST_MEMINFO_PATH`, using **`MemAvailable`**
— the kernel's own estimate of what a new allocation can get, which counts
reclaimable page cache and every other container. Swap is deliberately
excluded: serving a model from swap is worse than refusing to load it. NVML is
still used for temperature and utilization, which it reports correctly.

Note that vLLM's `--gpu-memory-utilization` (0.85 here) reserves its pool
**once at startup** and never returns it, so this figure does not drop between
requests. It only drops when `vllm serve` exits.

The catalog's profiles keep `skip_vram_check=True`, and should: their
`vram_required_mb` is a **cluster-wide** total (DeepSeek declares 220 GB
across two nodes), so comparing it against one node's memory would reject
loads that fit fine. Enabling the check would first require a per-node budget.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GATEWAY_HOST` | `0.0.0.0` | Host to bind |
| `GATEWAY_PORT` | `8000` | Port to listen on |
| `RAY_HEAD_HOST` | `192.168.200.12` | Head node IP — where `spark_cluster` models expose their HTTP port |
| `SPARK_CLUSTER_CONTAINER` | `vllm_node` | Container the serve processes are exec'd into |
| `SPARK_WORKER_HOSTS` | `192.168.200.13` | Comma-separated worker IPs (ranks ≥ 1) |
| `SPARK_SSH_USER` | `alejandroacho` | SSH user for the workers |
| `SPARK_SSH_KEY` | `/ssh/id_spark` | SSH identity inside the container |
| `SPARK_MASTER_PORT` | `29501` | Multi-node coordination port (match the launcher's `.env`) |
| `MODELS_PATH` | `/home/alejandroacho/Models` | Local weights path (Docker-managed models only) |
| `SYSTEM_RAM_GB` | auto | Total RAM in GiB, read from `MemTotal`. Determines swap strategy and caps the VRAM budget. Set only to override |
| `HOST_MEMINFO_PATH` | `/host/meminfo` | Host `/proc/meminfo`, bind-mounted read-only. The memory source of truth on unified-memory hosts |
| `SWAP_TIMEOUT_S` | `600` | Max seconds to wait for a swap |
| `VRAM_POLL_INTERVAL_S` | `2.0` | nvidia-smi polling interval |
| `VRAM_SAFETY_MARGIN_MB` | `4096` | VRAM kept free as a buffer |
| `DOCKER_SOCKET` | `unix:///var/run/docker.sock` | Docker daemon socket |
| `DOCKER_NETWORK` | `blackwell-gateway_blackwell_net` | Docker network name |
| `LONG_POLLING_ENABLED` | `true` | Hold connections during swaps |
| `LONG_POLLING_TIMEOUT_S` | `600` | Max long-polling wait |
| `MAX_QUEUE_SIZE` | `200` | Max requests queued during a swap |
| `RETRY_AFTER_SECONDS` | `5` | `Retry-After` value for 503s |
| `RAY_WATCHDOG_INTERVAL_S` | `30` | Seconds between backend health checks (0 disables) |
| `MEDIA_NODE_IP` | `192.168.8.147` | Media node (node 3) IP — address of the adapter as seen from outside |
| `MEDIA_NODE_HOST` | `192.168.8.147` | Media node host (overrides the IP for the rare split-brain case) |
| `MEDIA_NODE_PORT` | `8010` | Media node adapter port |
| `MEDIA_NODE_URL` | `http://192.168.8.147:8010` | Full media node URL used by `media_node.py` |
| `GATEWAY_PUBLIC_URL` | `http://192.168.1.125:8000` | Public base URL of the gateway (media asset links) |

---

## Project Structure

```
Server/
```
Server/
├── NODE2.md                    # Worker inventory: what must be on it, what is not in git
├── Dockerfile                  # Gateway image (python:3.12-slim + openssh-client)
├── Dockerfile.comfyui          # FLUX.1-dev inference server image (legacy)
├── Dockerfile.ltx              # LTX-Video 2 inference server image (legacy)
├── Dockerfile.media            # Media node image — ComfyUI + CUDA 13 (node 3)
├── docker-compose.yml          # Gateway (default) + media profile (media-node, media-gateway)
├── Makefile                    # Task runner
├── requirements.txt
├── gateway/
│   ├── app.py                  # FastAPI application and endpoints (LLM gateway)
│   ├── config.py               # Model catalog, profiles, environment, media-node models
│   ├── media_app.py            # Media-only Gateway — entry point on node 3
│   ├── media_node.py           # Media-node routes — drop-in APIRouter (attached in app.py)
│   ├── orchestrator.py         # Lifecycle: Docker workloads + spark_cluster processes
│   ├── proxy.py                # HTTP proxy to the backends
│   ├── request_buffer.py       # Request queue + radix prefix cache
│   ├── router.py               # Model/profile routing (text only)
│   ├── schemas.py              # Pydantic models
│   ├── vram_monitor.py         # nvidia-smi monitoring
│   └── backends/               # Docker and Kubernetes orchestration backends
├── inference/
│   ├── flux_server.py          # FLUX.1-dev FastAPI server (legacy)
│   ├── ltx_server.py           # LTX-Video 2 FastAPI server (legacy)
│   └── media_server.py         # Media node ComfyUI adapter (port 8010)
├── models/
│   ├── gpt-oss-120b/           # download.sh (~240 GB)
│   ├── qwen3-coder-next/       # download.sh + Dockerfile (~95 GB)
│   ├── flux2-pro/              # download.sh (legacy)
│   ├── minimax-h3/             # download.sh — video + native audio (~63 GB)
│   ├── ace-step-1.5/           # download.sh — music (~20 GB)
│   └── hidream-o1/             # download.sh — images (~16 GB)
├── tests/                      # Fully mocked; no GPU or cluster required
├── ray-cluster/                # Cluster keep-alive (active) + legacy Ray tooling
│   ├── ensure_vllm_cluster.sh  # ACTIVE — idempotent vllm_node keep-alive (step 8)
│   ├── vllm-cluster.service    # ACTIVE — boot unit for the above
│   ├── vllm-cluster.timer      # ACTIVE — re-checks every 2 min
│   └── ...                     # Legacy single-node Ray tooling (see note below)
└── k8s/                        # Kubernetes manifests
```

> **Not all of `ray-cluster/` is legacy.** The three files marked ACTIVE above keep the *current* Spark cluster alive and are what step 8 installs; `ensure_vllm_cluster.sh` drives `spark-vllm-docker`'s launcher, not Ray. They live here for historical reasons — the directory predates them.
>
> The rest of `ray-cluster/` (`reset_ray_node.sh`, `run_cluster.sh`, `ray-node-{head,worker}.service`, `discover-sparks.sh`, `Dockerfile.blackwell-vllm`, `patch_gemma4.py`), plus `inference/`, `models/`, `Dockerfile.comfyui` and `Dockerfile.ltx`, are leftovers from the previous single-node, multimedia-capable setup. They drive a separate `ray-node-head` container on the `blackwell-vllm:latest` image via the `ray_vllm` engine, which **no model in the current catalog uses**. Do not confuse that container with `vllm_node`. The media-node files (`Dockerfile.media`, `gateway/media_app.py`, `gateway/media_node.py`, `inference/media_server.py`, the `models/{minimax-h3,ace-step-1.5,hidream-o1}/download.sh`) are **not** legacy — they are the active media profile (node 3).

---

## Troubleshooting

### Swap refused: "cluster container 'vllm_node' is not running"

The Gateway never creates that container. Bring it up on both nodes:

```bash
cd ~/spark-vllm-docker && HF_HOME=~/hf-cache ./run-recipe.sh deepseek-v4-flash-0731 --port 8020 -d
```

Or, equivalently and safe to repeat, the keep-alive script — which also handles the case where only *one* of the two ranks died:

```bash
bash ~/Server/ray-cluster/ensure_vllm_cluster.sh
```

**If this happened after a reboot or a power cut, the fix is step 8**, not this command — `vllm_node` runs with `restart=no` and does not come back on its own. Check whether the keep-alive is actually installed:

```bash
systemctl list-timers vllm-cluster.timer   # empty output = never installed
```

### Swap refused: "it shards through Ray but 'vllm_node' sees 0 Ray node(s)"

Qwen3.5 needs a Ray cluster spanning the containers, and they were launched in native multi-node mode (what DeepSeek uses). Relaunch in Ray mode:

```bash
cd ~/spark-vllm-docker && HF_HOME=~/hf-cache ./run-recipe.sh qwen3.5-122b-fp8 --port 8021 -d
```

### Empty `content` in the response

Not a bug — see the warning under Usage Examples. Raise `max_tokens`.

### SSH failures in the Gateway logs

```bash
docker exec blackwell-gateway ssh -i /ssh/id_spark -o BatchMode=yes \
  alejandroacho@192.168.200.13 hostname
```

`Permission denied (publickey)` means the key never landed on the worker. Note that `ssh-copy-id` may report "all keys were skipped because they already exist" while installing nothing — use `-f` to force, and verify from inside the container as above rather than from the host, where your personal key or `~/.ssh/config` can mask the failure.

### Model loading hangs for 20+ minutes

`--load-format instanttensor` without the `instanttensor-hybrid-draft-loader` mod. DeepSeek stopped using either as of 2026-09-20 — it now loads with `--load-format b12x`, which is built into the image — so this only applies if you pin the old loader back.

### A relaunch silently keeps the old image

`run-recipe.sh` prints `Cluster containers are already running. Skipping launch.` and **still exits 0**, having exec'd the model into whatever image the running containers already had. A new image only takes effect after `./launch-cluster.sh -t vllm-node-b12x --name vllm_node stop` (and `docker rm -f vllm_node` on both nodes for good measure). Verify with `docker ps --format '{{.Image}}'` rather than trusting the exit code.

### A swap brings back the wrong model

`vllm-cluster.timer` fires every 2 minutes and relaunches the cluster whenever it finds it incomplete — which is exactly what a swap looks like mid-flight. It used to hardcode DeepSeek and would exec it into the container where the incoming model was still loading, leaving the two fighting over the node's memory. It now asks the Gateway which profile is active (`/v1/profiles/active`) and relaunches that one, falling back to DeepSeek only when the Gateway cannot be reached.

### `vllm serve: error: argument --reasoning-config: Invalid JSON`

The JSON reached vLLM mangled by a shell layer. The Gateway writes the serve command to a file through a quoted heredoc precisely to avoid this — if you see it, something re-wrapped the command in `bash -c '...'`, where `shlex`'s single quotes collide with the wrapper's and the shell eats the braces.

### Reading a model's own log

```bash
docker exec vllm_node tail -f /tmp/vllm_deepseek-v4-flash_r0.log      # rank 0, head
ssh 192.168.200.13 'docker exec vllm_node tail -f /tmp/vllm_deepseek-v4-flash_r1.log'
```

### Inspecting what actually runs

```bash
docker exec vllm_node bash -c "cat /proc/\$(pgrep -f 'vllm serve' | head -1)/cmdline | tr '\0' '\n'"
```

---

## Current State

- **`qwen38`** is operational and currently active: 106 GB of NVFP4 weights on both nodes, serving on 8022 at ~10 tok/s single-stream with 262,144 tokens of context (as of 2026-09-20).
- **`deepseek`** has its weights on both nodes and served on 8020 at ~49 tok/s with 1,146,734 tokens of KV cache, but it was migrated to `--load-format b12x` / `--attention-backend B12X` on 2026-09-20 to match the new upstream recipe and **has not been started once under that configuration**. The image it last ran under is tagged `vllm-node-b12x:pre-qwen38` on both nodes.
- **`qwen35`** is configured but **not yet operational**: its 127 GB of weights are not downloaded on either node, and the containers currently run in native (non-Ray) mode, so its preflight check will refuse the swap. Space is no longer the blocker it once was — as of 2026-09-10 the head has 545 GB free and the worker 417 GB.
- **The cluster keep-alive of step 8 is installed and running** (as of 2026-09-10): `vllm-cluster.timer` is enabled on the head and fires every 2 minutes, so a model comes back on its own after a reboot. Since 2026-09-20 it relaunches whichever profile the Gateway reports active, not always DeepSeek.
- **`hf-download.sh -c` does not copy the weights** under HuggingFace's current cache layout: the big blobs live in a shared `~/hf-cache/hub/blobs/` store and the per-model directory holds only symlinks into it, so the copy moves a few MB, reports `Copy complete`, and leaves the worker with dangling links. Copy that store across by hand (`rsync -a ~/hf-cache/hub/blobs/ <worker>:~/hf-cache/hub/blobs/`) and verify the symlinks resolve on the far side.
- **The worker still carries legacy Ray debt.** `ray-node-worker.service` is enabled there and fails on every boot, and 157 GB of orphaned HuggingFace cache sit in its `~/.cache/huggingface`. See [NODE2.md](NODE2.md) §5.
- **The legacy `ray-node-head` container is still running on the head** (`blackwell-vllm:latest`, `ray-node-head.service` enabled), even though no model in the catalog uses the `ray_vllm` engine.

---

## License

Private — Alejandro Acho. All rights reserved.
