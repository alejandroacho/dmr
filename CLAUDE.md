# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Local development

```bash
# Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Gateway (requires Docker Engine running for container management)
python -m uvicorn gateway.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker (recommended)

```bash
# Build and start the Gateway
docker compose build gateway
docker compose up -d gateway

# View logs
docker compose logs -f gateway
```

### Tests

```bash
# Run all tests (Docker and nvidia-smi are fully mocked)
pytest tests/

# Run a single test file
pytest tests/test_app.py

# Run a single test
pytest tests/test_app.py::test_health_endpoint -v
```

### Media node (node 3 — standalone, runs there)

```bash
just download-media      # all three families (~99 GB)
just build-media         # ComfyUI + CUDA 13 image
just media-up            # adapter :8010 + its own Gateway :8000
just media-status        # Gateway health — includes per-modality availability
just media-health        # adapter health, bypassing the Gateway
just media-models        # models exposed
just media-test "..."        # video+audio → h3-test.mp4
just media-test-music "..."  # music      → music-test.mp3
just media-test-image "..."  # image      → image-test.png
```

## Architecture

The Gateway is a FastAPI middleware layer that routes requests from 9 external agents to locally-running inference containers, managing VRAM budgets automatically.

### Request flow

1. **`gateway/app.py`** — Receives all requests; holds singleton instances of all subsystems. The app runs with a single uvicorn worker (shared in-memory state).
2. **`gateway/router.py`** (`SmartRouter`) — Detects media type (text/image/video) from explicit fields, `tool_choice`, or prompt keywords. Selects the VRAM profile and target model.
3. **`gateway/orchestrator.py`** (`ContainerOrchestrator`) — If the required profile differs from the active one, performs a container swap via the Docker SDK. Swap strategy is `pause/unpause` (fast, ~1-2s) for ≥512 GB RAM, otherwise `stop/start` (~3-5s).
4. **`gateway/proxy.py`** (`InferenceProxy`) — Proxies the request to the backend container via `aiohttp`. Handles text (OpenAI-compatible), image (ComfyUI), and video (Diffusers) backends.
5. **`gateway/request_buffer.py`** (`RequestBuffer`, `RadixPrefixCache`) — **Currently unwired.** The queue was superseded by the shielded-swap-task path in `chat_completions`, which holds the connection open and dispatches once the swap resolves. Nothing calls `enqueue()` in production (its only caller, `_handle_during_swap`, is itself unreferenced), so `drain_all()`/`reject_all()` always operate on an empty queue and `/status/swap` reports `queued_requests: 0` unconditionally. `RadixPrefixCache` is likewise decorative: `chat_completions` computes the prefix hash and discards it — the hint is never sent to vLLM, which does its own prefix caching. Wire them or delete them, but do not trust them as-is.

### Media node (node 3) — standalone, not part of the cluster

A third GB10 serves **media only**, three modalities from one ComfyUI. It runs on
its own with its own Gateway; it is *not* wired into the text cluster.

| Modality | Model | Adapter route | Gateway route |
|---|---|---|---|
| `av` | MiniMax-H3 | `POST /generate` | `/v1/av/generate` |
| `music` | ACE-Step 1.5 XL Turbo | `POST /generate/music` | `/v1/audio/music` |
| `image` | HiDream-O1-Image | `POST /generate/image` | `/v1/images/generate` |

- **`gateway/media_app.py`** — the entry point on node 3 (`uvicorn
  gateway.media_app:app`). It deliberately does **not** import
  `ContainerOrchestrator`: the main app's startup adopts/recreates profiles and
  force-removes "orphaned" containers, which here would delete the container
  serving the models. No Docker socket is mounted either. Tests assert the
  orchestration surface is absent — don't add it.
- **`gateway/media_node.py`** — standalone `APIRouter` with the actual media
  routes. Imports nothing from the rest of `gateway/` and touches no shared state,
  so it also drops into the main Gateway via `media_node.attach(app)` if the node
  is ever put behind it. In that case attach it **last**, after the app's own
  routes exist: `attach()` warns about paths the host already serves (on the main
  Gateway `/v1/images/generate` is FLUX's, so H3's image route would be shadowed).
- **`inference/media_server.py`** — the inference adapter. Builds a ComfyUI API
  graph per request and proxies to a loopback ComfyUI (`:8188`), exposing the same
  `POST /generate` + `GET /health` contract as the other inference servers. Its
  `/health` reports per-modality availability, checking both that the nodes are
  registered *and* that the weight files are visible to ComfyUI's loaders — so a
  missing download surfaces there instead of as a failed generation.
- The media models are in `config.py` as `REMOTE_MEDIA_MODELS`, deliberately
  **not** in `ALL_MODELS` or `PROFILES`: `ALL_MODELS` drives orphan-container
  *force-removal*. Off-node models are proxied to, never orchestrated.
  `MEDIA_MODEL_ENDPOINTS` maps each to the endpoint that drives it.
- `ModelDefinition.host` marks a model as off-node; `ModelDefinition.base_url`
  resolves to the host when set, else the local container's DNS name.
- Nothing is profile-swapped. Task mode (`t2va`/`fl2va`/`ref2va`) and image
  variant (`dev`/`base`) come from the request payload; ComfyUI evicts between
  model families on its own.

Where the graphs came from — all transcribed, none guessed:

| Modality | Source |
|---|---|
| `av` | `comfy_extras/nodes_minimax_h3.py` node schemas |
| `music` | ComfyUI `blueprints/Text to Audio (ACE-Step 1.5).json` |
| `image` | `Comfy-Org/workflow_templates` `image_hidream_o1{,_dev}.json` |

HiDream-O1 is worth a warning: it does **not** use `KSampler`. It needs
`ModelNoiseScale` plus a `SamplerCustom` + `BasicScheduler` pair, and the sampler
differs per variant (`SamplerLCM` for dev at 28 steps/cfg 1, `dpmpp_2m_sde_gpu`
for base at 40 steps/cfg 5, which also adds `HiDreamO1PatchSeamSmoothing`).

Three things that only fail at request time, all already handled — don't regress them:

1. **`gcc` + `python3-dev` in `Dockerfile.media`.** Triton JIT-compiles kernels
   for the NVFP4 encoder and INT8/convrot ops, shelling out to `cc` and linking
   against `Python.h`. Missing them → `Failed to find C compiler` mid-graph, with
   the container reporting healthy.
2. **`SaveVideo`'s `codec` is a `DYNAMICCOMBO`.** Its API value is the bare option
   key (`"auto"`), not `{"codec": "auto"}` — a dict is silently dropped and the
   node fails with a missing positional argument.
3. **Autogrow inputs use dotted paths**, and the index base differs per node.
   H3's ref2va uses `TemplatePrefix`, so **zero-based**:
   `ref_images.ref_image_0`, `ref_videos.ref_video_0`,
   `ref_video_audios.ref_video_audio_0`, `ref_audios.ref_audio_0`.
   HiDream-O1 uses `TemplateNames`, so **one-based**: `images.image_1`…`image_10`.
   Either way it is the container id, then the template's name. ComfyUI re-nests
   those into the dict `execute()` receives (`build_nested_inputs` in
   `comfy_api/latest/_io.py`); the bare `ref_image_0` form arrives as an
   unexpected keyword argument.

Note what `/object_info` does and does not tell you: it reports Autogrow and
DynamicCombo inputs **unexpanded** (`ref_images` as one `COMFY_AUTOGROW_V3` entry),
so it confirms which inputs exist but not the key encoding. For that, read
`_expand_schema_for_dynamic` in `comfy_api/latest/_io.py`. Validating input *names*
against `/object_info` is not enough — both bugs 2 and 3 passed a name check and
failed on value shape.

### Profile and model configuration

All model definitions and VRAM profiles live in **`gateway/config.py`**. Profiles:

| Key | Description |
|---|---|
| `deepseek` | DeepSeek-V4-Flash-0731, native multi-node TP=2, 1M context — **default at startup** |
| `qwen35` | Qwen3.5-122B-A10B-FP8, Ray TP=2, 256K context (needs the containers launched in Ray mode) |
| `qwen38` | Qwen3.8-Flash-Next-NVFP4, native multi-node TP=2, 256K context, no Ray and no mods |

MiniMax-H3 has no profile — it lives on node 3, standalone, always loaded.

Each `ModelDefinition` carries the Docker image, container name, port, quantization, and vLLM args. `VRAMProfile` groups models and computes total VRAM.

### Swap deduplication

`_get_or_create_swap_task()` in `app.py` ensures only one swap task runs at a time. Concurrent requests that trigger the same swap reuse the existing `asyncio.Task`. The task is wrapped with `asyncio.shield` so a client timeout does not cancel an in-flight swap.

### Key design constraints

- **Single worker only** — all state (active profile, swap task, prefix cache) is in-process. Never run with multiple uvicorn workers.
- **Docker socket required** — the Gateway spawns/stops inference containers at runtime; it must have access to `/var/run/docker.sock`.
- **Schemas in `gateway/schemas.py`** — all Pydantic models and enums. The `AgentRequest` model is the unified OpenAI-compatible request used by all 9 agents.

### Test structure

`tests/conftest.py` patches `docker.DockerClient` globally (via `autouse=True`) and provides a `mock_vram_monitor` fixture that bypasses `nvidia-smi`. All tests run without any GPU or Docker daemon.

`tests/test_media_server.py` loads `inference/media_server.py` by path (the
`inference/` directory is not a package) and asserts the ComfyUI graph wiring —
socket indices, autogrow key names, checkpoint selection — against the node
schemas in ComfyUI's `comfy_extras/nodes_minimax_h3.py`. A wrong socket index
there is a request-time 400 from ComfyUI, not an import error, so these tests are
the only cheap guard.

### File retention

Three locations accumulate files and all three are swept: ComfyUI's
`output/media/` and `input/in_*` (by `media_server.cleanup_once()`), and the
Gateway's `MEDIA_ASSET_DIR` (by `media_app._cleanup_once()`). Both processes sweep
at startup and hourly; `MEDIA_RETENTION_HOURS` defaults to 24, `0` disables.

Two constraints the sweeps must keep:

- **Stay narrow.** Only `output/media/` (this server's `filename_prefix`) and
  `input/in_*` (its own upload naming). ComfyUI ships bundled inputs that its
  templates reference and a marker file in `output/`; a broader glob deletes them.
  Tests pin this.
- **Never abort.** Per-file `OSError` is logged and skipped — a file mid-write or
  already gone must not stop the pass — and the loop catches everything so a bad
  sweep cannot kill the task.

`response_format: "url"` responses outlive their file by design: the URL 404s once
swept. That is a deliberate trade the operator sets, not a bug.

### Request limits and cancellation

Two failure modes that took the node down for nine hours, both now handled:

**Unbounded requests.** `num_frames=999` at full canvas is ~1.19M video-latent
tokens against the default's 149k. Attention is superlinear in that, so it was a
12-18 hour job — and 999 is outside H3's trained range (~124-362), so the output
would have been junk. `AVRequest` now caps frames at `AV_MAX_FRAMES` (362) and
pixels at `AV_MAX_PIXELS` (768*1344), with bounds on the other modalities too.
`av_latent_tokens()` computes the estimate; requests past `AV_COST_WARN_RATIO`x
the default are logged as expensive. The Gateway deliberately does *not* mirror
these bounds — the node is the single authority and its 422 propagates.

**Abandoned jobs.** Submitting to ComfyUI is fire-and-forget: releasing the GPU
lock does not stop the job. Worse, the next request then submits another, and
they queue up *inside ComfyUI* where the adapter's lock cannot see them — which
is how seven jobs accumulated. Two things were needed:

1. **uvicorn does not cancel a handler when its client disconnects** — it only
   discards the response — so `CancelledError` never fires. Both the adapter and
   the Gateway poll `request.is_disconnected()` while waiting.
2. **There are two hops.** A client dropping off does not reach the node, whose
   client (the Gateway) is still connected and correctly reports no disconnect.
   The Gateway must cancel its own upstream task, which closes the connection and
   is how the node learns to call ComfyUI's `/queue {"delete": [...]}` and
   `/interrupt`.

Verified end to end: killing the client cancels the ComfyUI job in ~1.5s.

Note `/interrupt` is only checked *between sampling steps*. With a step that
takes tens of minutes it will appear to do nothing; restarting the container is
the reliable escape hatch.

**Neither hop caps a generation by time, and that is deliberate — don't re-add
it.** `MEDIA_NODE_TIMEOUT_S` (Gateway→adapter) and `GENERATE_TIMEOUT_S`
(adapter→ComfyUI poll loop) both default to `0` = no ceiling. A 15s clip at full
canvas runs ~55 min in one pass (~12.6 min for 5s), so the previous shared 1800s
value killed exactly the jobs it was meant to protect, after the GPU had already
done the work. What replaces the clock is the cancellation machinery above: the
caller's disconnect is polled on both hops, and a dead ComfyUI takes its container
down, dropping the connection. A ceiling set explicitly still works and now
returns 504 naming the knob — `asyncio.TimeoutError` is not an
`aiohttp.ClientError`, so before it escaped as a bare 500.

`COMFY_HTTP_TIMEOUT_S` (default 600) is a *separate* knob bounding one HTTP call
to ComfyUI. Keep it separate: one value cannot sanely bound both a 55-minute job
and a 20 ms `/history` poll, which is why "no generation ceiling" was previously
inexpressible.

**ref2va limits come from the node's Autogrow templates**, not from guesswork:
9 `ref_images`, 3 `ref_videos`, 3 `ref_video_audios`, 3 `ref_audios`.
`ref_video_audio_N` is the soundtrack *of* `ref_video_N` — `execute()` pairs them
by index and drops any with no matching video, so the adapter rejects that case
instead of letting the caller pay for an upload that never enters the graph
(unsupplied soundtracks fall back to the video's own audio via
`GetVideoComponents`). The extra total-of-12 cap in `generate_av` is *not* in the
schema, whose maxima sum to 18; it is flagged in place rather than loosened,
since it may come from H3's model card.

**Both HiDream-O1 checkpoints are separate downloads and either may be absent.**
`_probe_comfy` records per-variant availability in `_image_variants` and the
modality counts as available if *either* is present; `generate_image` then gates
on the variant actually requested. Checking only the dev checkpoint let `/health`
advertise `image` while a `variant="base"` request died inside ComfyUI as an
opaque 502 — the late failure the probe exists to prevent.

### Test dependencies

`pytest` plus `pytest-asyncio` (`asyncio_mode = "auto"` in `pyproject.toml`).
Neither is in `requirements.txt`, which is runtime-only.
