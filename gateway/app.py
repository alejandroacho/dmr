"""
Blackwell Orchestrator & Smart Gateway
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Main FastAPI application.
Manages intelligent routing, Docker container lifecycle,
and dynamic model swapping on an ASUS GX10 with ~120 GB unified VRAM.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from gateway import __version__
from gateway.config import (
    GATEWAY_HOST,
    GATEWAY_PORT,
    GATEWAY_PUBLIC_URL,
    LONG_POLLING_ENABLED,
    LONG_POLLING_TIMEOUT_S,
    PROFILES,
    PROFILE_FOCUS,
    PROFILE_FOCUS_CODE,
    RETRY_AFTER_SECONDS,
)
from gateway.orchestrator import ContainerOrchestrator, STATE_FILE
from gateway.proxy import InferenceProxy
from gateway.request_buffer import (
    BufferOverflowError,
    RadixPrefixCache,
    RequestBuffer,
)
from gateway.router import SmartRouter
from gateway.schemas import (
    AgentRequest,
    ContainerState,
    GatewayResponse,
    HealthResponse,
    ImageGenerationRequest,
    MediaType,
    ProfileMode,
    ProfileDetail,
    ProfilesOverviewResponse,
    ProfileStatusResponse,
    SwapStatusResponse,
    VideoGenerationRequest,
    VRAMReport,
)
from gateway.vram_monitor import VRAMMonitor

# ──────────────────── Swap Task Tracking ───────────────
# Prevents duplicate swaps and shields from cancellation.

_active_swap_task: asyncio.Task | None = None
_active_swap_target: str | None = None

# ──────────────────── Logging ──────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gateway.app")

# ──────────────────── Singletons ───────────────────────

vram_monitor = VRAMMonitor()
orchestrator = ContainerOrchestrator(vram_monitor)
router_engine = SmartRouter()
request_buffer = RequestBuffer()
prefix_cache = RadixPrefixCache()
inference_proxy = InferenceProxy()

_start_time: float = 0.0


# ──────────────────── Lifecycle ────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    _start_time = time.time()

    logger.info("━━━ Blackwell Smart Gateway v%s starting ━━━", __version__)

    # Start subsystems
    await vram_monitor.start()
    await inference_proxy.startup()

    # Try to reuse whatever profile was already running (avoids a 7-min model reload)
    adopted = await orchestrator.detect_and_adopt_running_profile()
    if adopted:
        logger.info("Autodetect: resumed profile '%s' without restarting containers.", adopted)
    else:
        # No usable containers found — restore last known profile or fall back to default
        await orchestrator.cleanup_orphaned_containers()
        last_profile_key = ContainerOrchestrator.load_persisted_profile()
        startup_profile = PROFILES.get(last_profile_key, PROFILE_FOCUS_CODE) if last_profile_key else PROFILE_FOCUS_CODE
        if last_profile_key and last_profile_key in PROFILES:
            logger.info("Restoring last active profile: '%s'", last_profile_key)
        else:
            logger.info("Loading default profile: FOCUS_CODE (Qwen3 Coder Next 80B)")
        success = await orchestrator.switch_profile(startup_profile)
        if not success:
            logger.warning("Could not load startup profile '%s'.", startup_profile.name)

    logger.info("━━━ Gateway READY en %s:%d ━━━", GATEWAY_HOST, GATEWAY_PORT)

    yield  # App running

    # Shutdown
    logger.info("━━━ Shutting down Smart Gateway ━━━")
    await vram_monitor.stop()
    await inference_proxy.shutdown()
    logger.info("━━━ Gateway shut down successfully ━━━")


# ──────────────────── App ──────────────────────────────

app = FastAPI(
    title="Blackwell Orchestrator & Smart Gateway",
    description=(
        "Middleware layer for autonomous VRAM management (~120 GB), "
        "dynamic model swapping, and intelligent routing on an "
        "ASUS GX10 with a single NVIDIA GB10 Blackwell GPU."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────── Static file serving for generated images ────
GENERATED_IMAGES_DIR = Path(os.getenv("GENERATED_IMAGES_DIR", "/tmp/gateway_images"))
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/generated", StaticFiles(directory=str(GENERATED_IMAGES_DIR)), name="generated")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ──────────── Health & Status ─────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Gateway health check and overall cluster status."""
    report = vram_monitor.latest
    profile_name = orchestrator.active_profile
    mode = None
    if profile_name:
        mode = ProfileMode.FOCUS if "focus" in profile_name else ProfileMode.CREATIVE

    return HealthResponse(
        status="ok" if report.healthy else "degraded",
        version=__version__,
        active_profile=mode,
        vram=report,
        containers=orchestrator.container_states,
        uptime_seconds=time.time() - _start_time,
    )


@app.get("/status/vram", response_model=VRAMReport, tags=["System"])
async def vram_status():
    """Detailed VRAM report for the cluster."""
    return await vram_monitor.query_gpus()


@app.get("/status/swap", response_model=SwapStatusResponse, tags=["System"])
async def swap_status():
    """Current model swap status (if in progress)."""
    return SwapStatusResponse(
        swapping=orchestrator.is_swapping,
        elapsed_seconds=orchestrator.swap_elapsed,
        estimated_remaining_seconds=max(0, 5.0 - orchestrator.swap_elapsed),
        queued_requests=request_buffer.queue_size,
    )


@app.get("/status/profile", response_model=ProfileStatusResponse, tags=["System"])
async def profile_status():
    """Detailed status of the active profile and loaded models."""
    profile_key = orchestrator.active_profile or "focus"
    mode = ProfileMode.FOCUS if "focus" in profile_key else ProfileMode.CREATIVE

    # Get slots for the active profile
    from gateway.config import ALL_MODELS
    model_map = {m.container_name: m for m in ALL_MODELS}

    # Build reverse label map for the active profile
    active = orchestrator.active_vram_profile
    label_by_name: dict[str, str] = {}
    if active:
        for lbl, m in active.labels.items():
            label_by_name[m.name] = lbl

    models = []
    for name, state in orchestrator.container_states.items():
        if name in model_map:
            slot = model_map[name].to_slot(state)
            slot.label = label_by_name.get(model_map[name].name, "")
            models.append(slot)

    return ProfileStatusResponse(
        active_profile=mode,
        models=models,
        vram=vram_monitor.latest,
    )


@app.get("/v1/profiles", response_model=ProfilesOverviewResponse, tags=["System"])
async def list_profiles():
    """Lists all available VRAM profiles and their models."""
    from gateway.config import PROFILES, ALL_MODELS

    active_key = orchestrator.active_profile

    details: list[ProfileDetail] = []
    for key, profile in PROFILES.items():
        is_active = (orchestrator._profile_key(profile) == active_key)
        # Build label map for this profile
        label_by_container: dict[str, str] = {}
        for lbl, m in profile.labels.items():
            label_by_container[m.container_name] = lbl

        slots: list[ModelSlot] = []
        for m in profile.primary_models + profile.secondary_models:
            state = orchestrator.container_states.get(m.container_name, ContainerState.STOPPED)
            slot = m.to_slot(state if is_active else ContainerState.STOPPED)
            slot.label = label_by_container.get(m.container_name, "")
            slots.append(slot)

        details.append(ProfileDetail(
            key=key,
            mode=profile.mode,
            description=profile.description,
            total_vram_required_mb=profile.total_vram_required_mb,
            is_active=is_active,
            models=slots,
        ))

    return ProfilesOverviewResponse(
        active_profile=active_key,
        profiles=details,
    )


@app.get("/v1/profiles/active", response_model=ProfileDetail, tags=["System"])
async def active_profile():
    """Returns the currently active VRAM profile and its models."""
    from gateway.config import PROFILES

    active_key = orchestrator.active_profile
    if not active_key:
        raise HTTPException(status_code=404, detail="No active profile found")

    # Match by _profile_key (e.g. "creative:flux2-pro|qwen3.5-4b") → dict key (e.g. "creative_image")
    profile = None
    dict_key = active_key
    for k, p in PROFILES.items():
        if orchestrator._profile_key(p) == active_key:
            profile = p
            dict_key = k
            break

    if not profile:
        raise HTTPException(status_code=404, detail=f"Active profile '{active_key}' not in registry")

    label_by_container: dict[str, str] = {}
    for lbl, m in profile.labels.items():
        label_by_container[m.container_name] = lbl

    slots: list[ModelSlot] = []
    for m in profile.primary_models + profile.secondary_models:
        state = orchestrator.container_states.get(m.container_name, ContainerState.STOPPED)
        slot = m.to_slot(state)
        slot.label = label_by_container.get(m.container_name, "")
        slots.append(slot)

    return ProfileDetail(
        key=dict_key,
        mode=profile.mode,
        description=profile.description,
        total_vram_required_mb=profile.total_vram_required_mb,
        is_active=True,
        models=slots,
    )


@app.get("/status/cache", tags=["System"])
async def cache_status():
    """Radix Prefix Cache statistics."""
    return prefix_cache.get_stats()


# ──────────── Unified Inference Endpoint ────────

@app.post("/v1/chat/completions", tags=["Inference"])
async def chat_completions(request: AgentRequest):
    """
    Unified endpoint compatible with OpenAI Chat Completions.
    The Gateway automatically detects whether the request is for text,
    image, or video and routes to the corresponding backend.
    """
    start_time = time.time()

    logger.info("Request model='%s' agent=%s stream=%s", request.model, request.agent_id, request.stream)

    # 1. Routing (pass active profile so labels resolve without swap)
    decision = router_engine.route(request, active_profile=orchestrator.active_vram_profile)

    # 2. Compute prefix if there are system messages
    if request.messages:
        prefix_cache.get_prefix_hash(request.messages)

    # 3. Check if swap is needed
    current_profile = orchestrator.active_profile or ""
    target_key = orchestrator._profile_key(decision.profile)
    model_ready = orchestrator.is_model_ready(decision.target_model.container_name)

    if current_profile != target_key or not model_ready:
        decision.requires_swap = True

        if current_profile == target_key and not model_ready:
            # Profile matches but container is not READY (crashed, restarting, etc.)
            # Force the swap to bring it back up.
            logger.warning(
                "Model '%s' is not ready (state=%s). Forcing restart.",
                decision.target_model.container_name,
                orchestrator.container_states.get(decision.target_model.container_name),
            )

        # Obtain or create a swap task (never duplicate)
        force_swap = current_profile == target_key and not model_ready
        swap_task = _get_or_create_swap_task(decision.profile, target_key, force=force_swap)

        if LONG_POLLING_ENABLED:
            # Wait for swap to complete (Long Polling).
            # Use asyncio.shield so the swap continues even if this
            # request times out — prevents the "death loop".
            try:
                success = await asyncio.wait_for(
                    asyncio.shield(swap_task), timeout=LONG_POLLING_TIMEOUT_S
                )
                if not success:
                    raise HTTPException(
                        status_code=503,
                        detail="Model swap failed",
                        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
                    )
            except asyncio.TimeoutError:
                # The swap is still running — tell the client to retry.
                # Do NOT reject buffered requests (the swap will finish).
                raise HTTPException(
                    status_code=503,
                    detail="Swap still in progress, please retry",
                    headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
                )
        else:
            # Immediate 503
            raise HTTPException(
                status_code=503,
                detail="Profile switch in progress",
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )

    # 4. Proxy to inference backend
    try:
        result = await _dispatch_to_backend(decision, request)
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))

    # Check if the proxy returned an error dict
    if isinstance(result, dict) and "error" in result:
        err = result["error"]
        code = err.get("code", 502)
        raise HTTPException(status_code=code, detail=err.get("message", "Backend error"))

    elapsed = (time.time() - start_time) * 1000

    # 5. If streaming, return SSE
    if request.stream:
        if decision.media_type == MediaType.TEXT:
            return StreamingResponse(
                result,
                media_type="text/event-stream",
                headers={"X-Processing-Time-Ms": f"{elapsed:.1f}"},
            )
        # Non-text results (image/video): the client expects SSE but the
        # backend returned a single JSON dict.  Wrap it as one SSE chunk
        # so streaming parsers (OpenWebUI, etc.) can consume it.
        if isinstance(result, dict) and result.get("object", "").startswith("chat.completion"):
            import json as _json

            async def _single_chunk_sse():
                yield f"data: {_json.dumps(result)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _single_chunk_sse(),
                media_type="text/event-stream",
                headers={"X-Processing-Time-Ms": f"{elapsed:.1f}"},
            )

    # Log VRAM only for non-streaming requests (generator is complete at this point)
    vram = vram_monitor.latest
    if vram.healthy and vram.gpus:
        gpu = vram.gpus[0]
        logger.info(
            "VRAM — used=%d MB / %d MB (free=%d MB) | GPU util=%d%% | temp=%d°C",
            vram.total_used_mb, vram.total_vram_mb, vram.total_free_mb,
            gpu.utilization_pct, gpu.temperature_c,
        )

    # 6. Normal response
    # If the backend already returned an OpenAI-compatible dict (e.g. image
    # generation wrapped as chat.completion), return it directly so clients
    # like OpenWebUI can find choices[0].message.content at the top level.
    if isinstance(result, dict) and result.get("object") == "chat.completion":
        result.setdefault("processing_time_ms", elapsed)
        return JSONResponse(content=result)

    return GatewayResponse(
        success=True,
        data=result,
        profile=decision.profile.mode,
        processing_time_ms=elapsed,
        model_used=decision.target_model.name,
    )


# ──────────── Specific Multimedia Endpoints ────────

@app.post("/v1/images/generate", tags=["Multimedia"])
async def generate_image(request: ImageGenerationRequest):
    """
    Generates an image with FLUX.2 Pro.
    Automatically triggers the switch to Creative Mode if needed.
    """
    start_time = time.time()

    # Force creative image profile
    from gateway.config import PROFILE_CREATIVE_IMAGE, FLUX2_PRO

    current = orchestrator.active_profile or ""
    target_key = orchestrator._profile_key(PROFILE_CREATIVE_IMAGE)
    model_ready = orchestrator.is_model_ready(FLUX2_PRO.container_name)

    if current != target_key or not model_ready:
        if not model_ready and current == target_key:
            logger.warning(
                "Model '%s' is not ready (state=%s). Forcing restart.",
                FLUX2_PRO.container_name,
                orchestrator.container_states.get(FLUX2_PRO.container_name),
            )
        if orchestrator.is_swapping:
            raise HTTPException(
                status_code=503,
                detail="Swap in progress",
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )
        force_swap = current == target_key and not model_ready
        success = await orchestrator.switch_profile(PROFILE_CREATIVE_IMAGE, force=force_swap)
        if not success:
            raise HTTPException(status_code=503, detail="Could not activate creative mode")

    payload = request.model_dump()
    result = await inference_proxy.generate_image(FLUX2_PRO, payload)
    elapsed = (time.time() - start_time) * 1000

    return GatewayResponse(
        success="error" not in result,
        data=result,
        profile=ProfileMode.CREATIVE,
        processing_time_ms=elapsed,
        model_used="flux2-pro",
    )


@app.post("/v1/images/generations", tags=["Multimedia"])
async def dalle_generate_image(request: Request):
    """
    DALL-E compatible image generation endpoint.
    Open WebUI and other clients use this format.
    """
    import time as _time
    body = await request.json()
    start_time = _time.time()

    from gateway.config import PROFILE_CREATIVE_IMAGE, FLUX2_PRO
    current = orchestrator.active_profile or ""
    target_key = orchestrator._profile_key(PROFILE_CREATIVE_IMAGE)
    model_ready = orchestrator.is_model_ready(FLUX2_PRO.container_name)
    if current != target_key or not model_ready:
        force_swap = current == target_key and not model_ready
        success = await orchestrator.switch_profile(PROFILE_CREATIVE_IMAGE, force=force_swap)
        if not success:
            raise HTTPException(status_code=503, detail="Could not activate image generation mode")

    payload = {
        "prompt": body.get("prompt", ""),
        "width": body.get("size", "1024x1024").split("x")[0] if "size" in body else body.get("width", 1024),
        "height": body.get("size", "1024x1024").split("x")[1] if "size" in body else body.get("height", 1024),
        "steps": body.get("steps", 30),
        "seed": body.get("seed"),
    }
    result = await inference_proxy.generate_image(FLUX2_PRO, payload)
    created = int(_time.time())

    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])

    import base64 as _b64


    data_items = []
    for img_b64 in result.get("images", []):
        filename = f"{uuid.uuid4().hex}.png"
        filepath = GENERATED_IMAGES_DIR / filename
        filepath.write_bytes(_b64.b64decode(img_b64))
        image_url = f"{GATEWAY_PUBLIC_URL}/generated/{filename}"
        data_items.append({"url": image_url, "b64_json": img_b64})

    return {
        "created": created,
        "data": data_items,
    }


@app.post("/v1/videos/generate", tags=["Multimedia"])
async def generate_video(request: VideoGenerationRequest):
    """
    Generates a video with LTX-Video 2.
    Automatically triggers the switch to Creative Mode if needed.
    """
    start_time = time.time()

    from gateway.config import PROFILE_CREATIVE_VIDEO, LTX_VIDEO_2

    current = orchestrator.active_profile or ""
    target_key = orchestrator._profile_key(PROFILE_CREATIVE_VIDEO)
    model_ready = orchestrator.is_model_ready(LTX_VIDEO_2.container_name)

    if current != target_key or not model_ready:
        if not model_ready and current == target_key:
            logger.warning(
                "Model '%s' is not ready (state=%s). Forcing restart.",
                LTX_VIDEO_2.container_name,
                orchestrator.container_states.get(LTX_VIDEO_2.container_name),
            )
        if orchestrator.is_swapping:
            raise HTTPException(
                status_code=503,
                detail="Swap in progress",
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )
        force_swap = current == target_key and not model_ready
        success = await orchestrator.switch_profile(PROFILE_CREATIVE_VIDEO, force=force_swap)
        if not success:
            raise HTTPException(status_code=503, detail="Could not activate creative video mode")

    payload = request.model_dump()
    result = await inference_proxy.generate_video(LTX_VIDEO_2, payload)
    elapsed = (time.time() - start_time) * 1000

    return GatewayResponse(
        success="error" not in result,
        data=result,
        profile=ProfileMode.CREATIVE,
        processing_time_ms=elapsed,
        model_used="ltx-video-2",
    )


# ──────────── Manual Profile Management ────────────

@app.post("/admin/profile/{profile_name}", tags=["Administration"])
async def set_profile(profile_name: str, force: bool = False):
    """
    Manually switches the VRAM profile.
    Valid values: focus, creative_image, creative_video
    """
    if profile_name not in PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Profile '{profile_name}' not valid. "
                   f"Options: {list(PROFILES.keys())}",
        )

    if orchestrator.is_swapping:
        raise HTTPException(
            status_code=409,
            detail="A swap is already in progress",
        )

    profile = PROFILES[profile_name]
    success = await orchestrator.switch_profile(profile, force=force)

    if not success:
        raise HTTPException(status_code=500, detail="Swap failed")

    return GatewayResponse(
        success=True,
        data={"message": f"Profile switched to '{profile_name}'"},
        profile=profile.mode,
    )


@app.post("/admin/container/{container_name}/stop", tags=["Administration"])
async def stop_container(container_name: str):
    """Manually stops an inference container."""
    await orchestrator.stop_container(container_name)
    return {"status": "stopped", "container": container_name}


@app.post("/admin/container/{container_name}/remove", tags=["Administration"])
async def remove_container(container_name: str):
    """Manually removes an inference container."""
    await orchestrator.remove_container(container_name)
    return {"status": "removed", "container": container_name}


# ──────────── Model Listing (OpenAI compat) ──────

@app.get("/v1/models", tags=["Inference"])
async def list_models():
    """Lists available models (OpenAI format).

    Each model appears with its real name.  Models in the active profile
    also appear under their label alias (e.g. "chat", "code", "image")
    so agents can use stable identifiers across profile switches.
    """
    from gateway.config import ALL_MODELS

    data = [
        {
            "id": m.name,
            "object": "model",
            "owned_by": "blackwell-gateway",
            "engine": m.engine,
            "vram_mb": m.vram_required_mb,
            "quantization": m.quantization,
        }
        for m in ALL_MODELS
    ]

    # Add label aliases from the active profile
    active = orchestrator.active_vram_profile
    if active:
        for label, model in active.labels.items():
            data.append({
                "id": label,
                "object": "model",
                "owned_by": "blackwell-gateway",
                "engine": model.engine,
                "vram_mb": model.vram_required_mb,
                "quantization": model.quantization,
                "alias_for": model.name,
            })

    return {"object": "list", "data": data}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INTERNAL HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get_or_create_swap_task(profile, target_key: str, force: bool = False) -> asyncio.Task:
    """
    Returns the active swap task if it targets the same profile,
    otherwise creates a new one.  Prevents duplicate swaps.

    The task is wrapped in `_execute_swap_and_drain` so it automatically
    drains buffered requests on success or rejects them on failure.

    Pass force=True to re-run the swap even if the profile key matches
    (e.g. when a container crashed and needs to be restarted).
    """
    global _active_swap_task, _active_swap_target

    # Reuse an existing swap to the SAME target (only when not forced)
    if (
        not force
        and _active_swap_task is not None
        and not _active_swap_task.done()
        and _active_swap_target == target_key
    ):
        logger.debug("Reusing existing swap task → '%s'", target_key)
        return _active_swap_task

    logger.info("Swap required: '%s' → '%s'", orchestrator.active_profile or "", target_key)

    _active_swap_target = target_key
    _active_swap_task = asyncio.create_task(
        _execute_swap_and_drain(profile, force=force)
    )
    return _active_swap_task


async def _execute_swap_and_drain(profile, force: bool = False) -> bool:
    """
    Executes the swap and handles buffer drain/reject.
    This runs as a long-lived task that is never cancelled.
    """
    global _active_swap_task, _active_swap_target

    try:
        success = await orchestrator.switch_profile(profile, force=force)
        if success:
            asyncio.create_task(_drain_buffered_requests())
        else:
            await request_buffer.reject_all("Swap failed")
        return success
    except Exception as exc:
        logger.error("Swap task error: %s", exc)
        await request_buffer.reject_all(f"Swap error: {exc}")
        return False
    finally:
        _active_swap_task = None
        _active_swap_target = None


async def _handle_during_swap(request: AgentRequest):
    """Handles a request when a swap is in progress."""
    if LONG_POLLING_ENABLED:
        try:
            future = await request_buffer.enqueue(request)
            result = await asyncio.wait_for(future, timeout=LONG_POLLING_TIMEOUT_S)
            if isinstance(result, dict) and "error" in result:
                err = result["error"]
                code = err.get("code", 502)
                raise HTTPException(status_code=code, detail=err.get("message", "Backend error"))
            return GatewayResponse(success=True, data=result)
        except BufferOverflowError:
            raise HTTPException(
                status_code=503,
                detail="Queue full — swap in progress",
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="Long polling timeout",
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )
    else:
        raise HTTPException(
            status_code=503,
            detail="Model swap in progress",
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )


async def _drain_buffered_requests() -> None:
    """Drains and dispatches all buffered requests after a successful swap."""
    try:
        drained = await request_buffer.drain_all()
        for buffered in drained:
            try:
                decision = router_engine.route(
                    buffered.request,
                    active_profile=orchestrator.active_vram_profile,
                )
                result = await _dispatch_to_backend(decision, buffered.request)
                await request_buffer.resolve(buffered, result)
            except Exception as exc:
                logger.error("Error dispatching buffered request: %s", exc)
                if not buffered.future.done():
                    buffered.future.set_exception(exc)
    except Exception as exc:
        logger.error("Error draining request buffer: %s", exc)


async def _dispatch_to_backend(decision, request: AgentRequest) -> Any:
    """Dispatches the request to the correct inference backend."""
    model = decision.target_model

    if decision.media_type == MediaType.TEXT:
        payload = {
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        return await inference_proxy.chat_completion(
            model, payload, stream=request.stream
        )

    elif decision.media_type == MediaType.IMAGE:
        payload = request.media_params or {}
        # Extract prompt from last message if not in media_params
        if "prompt" not in payload and request.messages:
            last_msg = request.messages[-1]
            payload["prompt"] = last_msg.get("content", "")
        result = await inference_proxy.generate_image(model, payload)
        # Wrap as OpenAI chat completion so any client can render it
        if "images" in result and result["images"]:
            import base64 as _b64

            b64 = result["images"][0]
            img_id = f"img-{result.get('seed', 0)}"

            # Save image to disk and serve via URL (avoids huge SSE chunks)
            filename = f"{uuid.uuid4().hex}.png"
            filepath = GENERATED_IMAGES_DIR / filename
            filepath.write_bytes(_b64.b64decode(b64))

        
            image_url = f"{GATEWAY_PUBLIC_URL}/generated/{filename}"
            content = f"![generated image]({image_url})"

            if request.stream:
                return {
                    "id": img_id,
                    "object": "chat.completion.chunk",
                    "model": model.name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }],
                }
            return {
                "id": img_id,
                "object": "chat.completion",
                "model": model.name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        return result

    elif decision.media_type == MediaType.VIDEO:
        payload = request.media_params or {}
        if "prompt" not in payload and request.messages:
            last_msg = request.messages[-1]
            payload["prompt"] = last_msg.get("content", "")
        return await inference_proxy.generate_video(model, payload)

    raise ValueError(f"Unknown MediaType: {decision.media_type}")


# ──────────── Direct entry point ────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gateway.app:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        log_level="info",
        workers=1,  # Single worker for shared state
        access_log=True,
    )
