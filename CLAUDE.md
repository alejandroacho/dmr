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

## Architecture

The Gateway is a FastAPI middleware layer that routes requests from 9 external agents to locally-running inference containers, managing VRAM budgets automatically.

### Request flow

1. **`gateway/app.py`** — Receives all requests; holds singleton instances of all subsystems. The app runs with a single uvicorn worker (shared in-memory state).
2. **`gateway/router.py`** (`SmartRouter`) — Detects media type (text/image/video) from explicit fields, `tool_choice`, or prompt keywords. Selects the VRAM profile and target model.
3. **`gateway/orchestrator.py`** (`ContainerOrchestrator`) — If the required profile differs from the active one, performs a container swap via the Docker SDK. Swap strategy is `pause/unpause` (fast, ~1-2s) for ≥512 GB RAM, otherwise `stop/start` (~3-5s).
4. **`gateway/proxy.py`** (`InferenceProxy`) — Proxies the request to the backend container via `aiohttp`. Handles text (OpenAI-compatible), image (ComfyUI), and video (Diffusers) backends.
5. **`gateway/request_buffer.py`** (`RequestBuffer`, `RadixPrefixCache`) — Holds incoming requests in an asyncio queue during swaps (long polling). `RadixPrefixCache` hashes system prompt prefixes to hint vLLM's prefix caching.

### Profile and model configuration

All model definitions and VRAM profiles live in **`gateway/config.py`**. Profiles:

| Key | Description |
|---|---|
| `focus` | GPT-OSS 120B only (single GPU, ~84 GB, MXFP4 CUTLASS sm_121 — requires `vllm-mxfp4-spark:latest` image) |
| `focus_code` | Qwen3 Coder Next 80B MoE — **default at startup** |
| `creative_image` | Qwen3 Coder 30B + FLUX.2 Pro (~77 GB) |
| `creative_video` | Qwen3 Coder 30B + LTX-Video 2 (~77 GB) |

Each `ModelDefinition` carries the Docker image, container name, port, quantization, and vLLM args. `VRAMProfile` groups models and computes total VRAM.

### Swap deduplication

`_get_or_create_swap_task()` in `app.py` ensures only one swap task runs at a time. Concurrent requests that trigger the same swap reuse the existing `asyncio.Task`. The task is wrapped with `asyncio.shield` so a client timeout does not cancel an in-flight swap.

### Key design constraints

- **Single worker only** — all state (active profile, swap task, prefix cache) is in-process. Never run with multiple uvicorn workers.
- **Docker socket required** — the Gateway spawns/stops inference containers at runtime; it must have access to `/var/run/docker.sock`.
- **Schemas in `gateway/schemas.py`** — all Pydantic models and enums. The `AgentRequest` model is the unified OpenAI-compatible request used by all 9 agents.

### Test structure

`tests/conftest.py` patches `docker.DockerClient` globally (via `autouse=True`) and provides a `mock_vram_monitor` fixture that bypasses `nvidia-smi`. All tests run without any GPU or Docker daemon.
