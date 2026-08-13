# Blackwell Orchestrator & Smart Gateway

FastAPI middleware that fronts a two-node NVIDIA DGX Spark (GB10 Blackwell) cluster, serving two large text models and swapping between them on demand.

The catalog is deliberately small: **DeepSeek-V4-Flash** and **Qwen3.5-122B**. Both are too large for one node, so both shard across the pair with tensor parallelism.

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

| Profile key | Model | Sharding | Port | Context |
|---|---|---|---|---|
| **`deepseek`** ⭐ | DeepSeek-V4-Flash-0731 — 284B total / 13B active MoE, FP4 experts + FP8 dense | Native multi-node, TP=2 (2 processes) | 8020 | 1,048,576 |
| **`qwen35`** | Qwen3.5-122B-A10B-FP8 — native FP8 | Ray, TP=2 (1 process) | 8021 | 262,144 |

`deepseek` is the default at startup.

### Why these two

DeepSeek-V4-Flash is the reason the cluster exists: 167 GB of weights, ~220 GB of footprint across both nodes, and dspark speculative decoding. Qwen3.5-122B-A10B-FP8 is the second-opinion model — Qwen's own FP8 quantization (not a third-party int4 requant), ~127 GB, comfortable at ~64 GB per node with room for a 256K KV cache.

Two Qwen3.5 recipes were rejected: `qwen3.5-397b-int4-autoround` is labeled EXPERIMENTAL upstream and its 226 GB leave almost nothing for KV cache within the 256 GB the pair has; `qwen3.5-122b-int4-autoround` is a lossy requant whose only advantage — fitting on one node — is irrelevant here.

### The two models need different cluster modes

This is the sharpest operational edge in the whole setup:

| | DeepSeek | Qwen3.5 |
|---|---|---|
| Distribution | vLLM native (`--nnodes/--node-rank`, one `--headless` rank per worker) | Ray (`--distributed-executor-backend ray`, single process) |
| Requires Ray inside the containers | No | **Yes** |
| Mod | `instanttensor-hybrid-draft-loader` (patches vLLM's model loader) | `fix-qwen3.5-chat-template` (drops a jinja file in `/workspace`) |

The two **mods coexist** — one patches Python, the other only copies a file — so a single container launch can serve both models. Ray mode does not, though: DeepSeek ignores a running Ray cluster, but Qwen3.5 hard-fails without one. Launch the containers in Ray mode if you want to swap freely between them.

The Gateway checks this before every Qwen3.5 swap and refuses with a clear message rather than starting a process that would die.

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

Everything below runs on the **head node**.

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
| `GET` | `/v1/models` | Available models plus the active profile's label aliases (`chat`, `code`) |

### System status

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health, active profile mode, per-container state, and Ray cluster state (`ray`) when — and only when — a loaded model needs Ray. `status` is `degraded` if that cluster is short on nodes |
| `GET` | `/status/vram` | Memory report for the **local node only** — cluster models are sharded, so the other node's half is not counted. On unified-memory hosts the figures come from the host's `MemAvailable` (see below), not from NVML |
| `GET` | `/status/swap` | Swap in progress, elapsed, queue depth |
| `GET` | `/status/profile` | Active profile and its models |
| `GET` | `/v1/profiles` | All profiles |
| `GET` | `/v1/profiles/active` | Active profile with its registry key — the reliable source for "what is loaded" |
| `GET` | `/status/cache` | Radix prefix cache statistics |

### Administration

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/admin/profile/{name}` | Switch profile (`deepseek`, `qwen35`). Add `?force=true` to restart the active one. |
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

---

## Project Structure

```
Server/
├── Dockerfile                  # Gateway image (python:3.12-slim + openssh-client)
├── docker-compose.yml          # Gateway only — models live on the Spark cluster
├── Makefile                    # Task runner
├── requirements.txt
├── gateway/
│   ├── app.py                  # FastAPI application and endpoints
│   ├── config.py               # Model catalog, profiles, environment
│   ├── orchestrator.py         # Lifecycle: Docker workloads + spark_cluster processes
│   ├── proxy.py                # HTTP proxy to the backends
│   ├── request_buffer.py       # Request queue + radix prefix cache
│   ├── router.py               # Model/profile routing (text only)
│   ├── schemas.py              # Pydantic models
│   ├── vram_monitor.py         # nvidia-smi monitoring
│   └── backends/               # Docker and Kubernetes orchestration backends
├── tests/                      # Fully mocked; no GPU or cluster required
├── ray-cluster/                # Legacy Ray cluster tooling (see note below)
└── k8s/                        # Kubernetes manifests
```

> `ray-cluster/`, `inference/`, `models/`, `Dockerfile.comfyui` and `Dockerfile.ltx` are leftovers from the previous single-node, multimedia-capable setup. Nothing in the current catalog references them.

---

## Troubleshooting

### Swap refused: "cluster container 'vllm_node' is not running"

The Gateway never creates that container. Bring it up on both nodes:

```bash
cd ~/spark-vllm-docker && HF_HOME=~/hf-cache ./run-recipe.sh deepseek-v4-flash-0731 --port 8020 -d
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

`--load-format instanttensor` without the `instanttensor-hybrid-draft-loader` mod. `run-recipe.sh` applies it from the recipe; a hand-rolled `launch-cluster.sh` invocation must pass `--apply-mod mods/instanttensor-hybrid-draft-loader`.

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

- **`deepseek`** is operational: weights on both nodes, serving on 8020, ~49 tok/s single-stream, 1,146,734 tokens of KV cache.
- **`qwen35`** is configured but **not yet operational**: its 127 GB of weights are not downloaded, and the containers currently run in native (non-Ray) mode, so its preflight check will refuse the swap. Downloading it leaves only ~24 GB free on the head node — worth freeing space first.

---

## License

Private — Alejandro Acho. All rights reserved.
