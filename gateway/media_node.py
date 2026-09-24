"""
Media node (node 3) integration for the Smart Gateway.

Exposes MiniMax-H3 joint video+audio generation through the Gateway's public
API, proxying to the dedicated media node instead of a local container.

This module is intentionally self-contained — it imports nothing from the rest
of the `gateway` package and touches no shared state (no profile, no swap, no
VRAM accounting). The media node is a separate machine with its own 124 GB
unified pool, so nothing here can collide with the text cluster. That makes it
safe to drop into any version of the Gateway:

    from gateway.media_node import attach
    attach(app)                     # after `app = FastAPI(...)`

Configuration (environment):

    MEDIA_NODE_IP           node address, the one place it is configured (see .env)
    MEDIA_NODE_URL          full base URL; overrides MEDIA_NODE_IP/PORT if set
    MEDIA_NODE_TIMEOUT_S    per-request ceiling, 0 = none (default 0)
    MEDIA_ASSET_DIR         directory for saved mp4s, enables response_format="url"
    MEDIA_PUBLIC_URL        public base URL for those files (default GATEWAY_PUBLIC_URL)
    MEDIA_NODE_ALIAS_VIDEOS also answer POST /v1/videos/generate (default false)
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("gateway.media_node")

# ─────────────────── Configuration ───────────────────

# Node 3. Currently on WiFi at 192.168.8.147; set MEDIA_NODE_IP to its wired
# 192.168.1.x address once the cable is in — nothing else needs to change. Under
# compose the media profile sets MEDIA_NODE_URL to the container name instead, so
# these defaults only apply when running the Gateway outside it.
MEDIA_NODE_IP: str = os.getenv("MEDIA_NODE_IP", "192.168.8.147")
MEDIA_NODE_PORT: int = int(os.getenv("MEDIA_NODE_PORT", "8010"))
MEDIA_NODE_URL: str = os.getenv(
    "MEDIA_NODE_URL", f"http://{MEDIA_NODE_IP}:{MEDIA_NODE_PORT}"
).rstrip("/")

# A 15s 2K generation with audio runs ~55 min, so a ceiling here aborts the very
# requests it was added for, and does so after the GPU has already done the work.
# 0 = no total timeout (the default). The connect timeout below still catches an
# unreachable node in seconds, and the disconnect watch in _dispatch still frees
# the node's GPU the moment a caller gives up — neither depends on this cap.
MEDIA_NODE_TIMEOUT_S: int = int(os.getenv("MEDIA_NODE_TIMEOUT_S", "0"))
MEDIA_NODE_CONNECT_TIMEOUT_S: int = int(os.getenv("MEDIA_NODE_CONNECT_TIMEOUT_S", "10"))

# Optional: write finished mp4s here and hand clients a URL instead of ~50 MB
# of base64. Leave unset to keep responses base64-only.
MEDIA_ASSET_DIR: str = os.getenv("MEDIA_ASSET_DIR", "")
MEDIA_PUBLIC_URL: str = os.getenv(
    "MEDIA_PUBLIC_URL",
    os.getenv("GATEWAY_PUBLIC_URL", "http://192.168.1.125:8000"),
).rstrip("/")

# How often to check whether the caller is still there, while waiting upstream.
DISCONNECT_POLL_S: float = float(os.getenv("MEDIA_DISCONNECT_POLL_S", "2.0"))

ASSET_ROUTE = "/assets/media"

# Also answer the older /v1/videos/generate path (see attach()).
ALIAS_LEGACY_VIDEO_PATH: bool = os.getenv("MEDIA_NODE_ALIAS_VIDEOS", "false").lower() == "true"

router = APIRouter(tags=["media"])


# ─────────────────── Schemas ─────────────────────────

class MusicGenerationRequest(BaseModel):
    """ACE-Step 1.5 XL Turbo. `prompt` is a style/genre tag list, not prose."""

    prompt: str = ""
    lyrics: str = ""
    duration: float = 120.0

    bpm: int = 120
    time_signature: Literal["2", "3", "4", "6"] = "4"
    language: str = "en"
    key_scale: str = "C major"

    steps: int = 8                  # XL Turbo samples in 8
    cfg_scale: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    shift: float = 3.0

    lm_cfg_scale: float = 2.0
    temperature: float = 0.85
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    generate_audio_codes: bool = True

    mp3_quality: Literal["V0", "128k", "320k"] = "320k"
    seed: int | None = None

    response_format: Literal["b64_json", "url"] = "b64_json"
    agent_id: str | None = None


class ImageGenerationRequest(BaseModel):
    """HiDream-O1-Image. Sampling defaults come from the variant, not from here."""

    prompt: str
    negative_prompt: str = ""
    variant: Literal["dev", "base"] = "dev"

    width: int = 2048
    height: int = 2048
    batch_size: int = 1

    steps: int | None = None
    cfg_scale: float | None = None
    noise_scale: float | None = None
    scheduler: str = "normal"
    seed: int | None = None

    # 1 image = instruction edit; 2-10 = multi-reference.
    ref_images: list[str] = Field(default_factory=list)

    response_format: Literal["b64_json", "url"] = "b64_json"
    agent_id: str | None = None


class VideoGenerationRequest(BaseModel):
    """MiniMax-H3 generation request.

    The task mode is inferred from what you send:
      - prompt only .................... t2va
      - prompt + first/last_frame ...... fl2va
      - prompt + any ref_* field ....... ref2va
    """

    prompt: str
    negative_prompt: str = ""

    # H3's native canvas: 768px short edge, capped at 768*1344, axes multiple of 32.
    width: int = 1344
    height: int = 768
    # Snapped up to the model's 17k+5 grid at 24 fps; 124 frames ~ 5.2s.
    num_frames: int = 124
    fps: int = 24

    steps: int = 20
    cfg_scale: float = 1.0
    sampler: str = "res_multistep"
    scheduler: str = "simple"
    shift_video: float = 12.0
    shift_audio: float = 3.0
    seed: int | None = None

    # base64 (bare or data: URI)
    first_frame: str | None = None
    last_frame: str | None = None

    # ref2va per-container maxima, from the node's Autogrow templates: 9 images,
    # 3 videos, 3 video soundtracks, 3 standalone audios. ref_video_audio_N is
    # the soundtrack *of* ref_video_N — paired by index, extras are ignored.
    # The node enforces these; the adapter also caps the total at 12.
    ref_images: list[str] = Field(default_factory=list)
    ref_videos: list[str] = Field(default_factory=list)
    ref_video_audios: list[str] = Field(default_factory=list)
    ref_audios: list[str] = Field(default_factory=list)
    ref_image_size: Literal["match", "max"] = "match"

    response_format: Literal["b64_json", "url"] = "b64_json"
    agent_id: str | None = None


# ─────────────────── HTTP session ────────────────────

_session: aiohttp.ClientSession | None = None


def _get_session() -> aiohttp.ClientSession:
    """Lazily created session — avoids depending on the app's startup hooks."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                # None disables the ceiling entirely — see MEDIA_NODE_TIMEOUT_S.
                total=MEDIA_NODE_TIMEOUT_S or None,
                connect=MEDIA_NODE_CONNECT_TIMEOUT_S,
            ),
        )
    return _session


async def shutdown() -> None:
    """Closes the session. Call from the host app's shutdown path.

    Optional — the session is lazy, so skipping this only costs an "Unclosed
    client session" warning at exit, never a failed request.
    """
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


# ─────────────────── Endpoints ───────────────────────

async def probe() -> tuple[bool, dict]:
    """Asks the media node how it's doing. Returns (ready, payload).

    Reachable and ready are different states and the payload reports both: the
    adapter answers on its port long before the weights finish loading, and
    replies 503 until then. Treating any HTTP response as healthy would have
    callers send generations into a backend that rejects them.
    """
    try:
        async with _get_session().get(
            f"{MEDIA_NODE_URL}/health",
            timeout=aiohttp.ClientTimeout(total=15, connect=5),
        ) as resp:
            try:
                body = await resp.json()
            except Exception:
                body = {"raw": (await resp.text())[:200]}
            ready = resp.status == 200 and body.get("ready", True) is True
            return ready, {"url": MEDIA_NODE_URL, "reachable": True, "ready": ready,
                           "http_status": resp.status, "node": body}
    except Exception as exc:
        # Expected while the node boots, or if the network path is one-way.
        return False, {"url": MEDIA_NODE_URL, "reachable": False, "ready": False,
                       "error": f"{type(exc).__name__}: {exc}"}


@router.get("/status/media-node")
async def media_node_status() -> JSONResponse:
    """Health of the media node, without touching the local cluster."""
    ready, payload = await probe()
    return JSONResponse(payload, status_code=200 if ready else 503)


async def _dispatch(
    path: str,
    payload: dict[str, Any],
    label: str,
    request: Request | None = None,
) -> tuple[dict, float]:
    """Forwards a generation to the media node and translates its errors.

    When *request* is given, the caller's disconnect is watched for and the
    upstream call is aborted with it. That matters because there are two hops:
    dropping the client here does not reach the node on its own, and the node
    would keep a GPU busy for a request nobody is waiting for.
    """
    if request is not None:
        upstream = asyncio.create_task(_dispatch(path, payload, label))
        while True:
            done, _ = await asyncio.wait({upstream}, timeout=DISCONNECT_POLL_S)
            if upstream in done:
                return upstream.result()
            if await request.is_disconnected():
                # Cancelling closes the connection to the node, which is how the
                # node learns to cancel its ComfyUI job.
                upstream.cancel()
                logger.info("Caller disconnected — abandoning %s upstream", label)
                raise HTTPException(status_code=499, detail="client disconnected")

    start = time.time()
    try:
        async with _get_session().post(f"{MEDIA_NODE_URL}{path}", json=payload) as resp:
            if resp.status != 200:
                body = (await resp.text())[:2000]
                logger.error("Media node %s error (status=%d): %s", label, resp.status, body)
                # Every 4xx is the caller's fault and must reach them as such —
                # a validation error surfacing as 502 tells them to blame the
                # gateway. 503 also passes through: it means "loading, retry".
                # Only genuine backend failures become 502.
                passthrough = resp.status < 500 or resp.status == 503
                # Unwrap the node's own {"detail": ...} so the message isn't
                # double-encoded JSON by the time a client reads it.
                try:
                    parsed = json.loads(body)
                    detail = parsed.get("detail", parsed) if isinstance(parsed, dict) else parsed
                except ValueError:
                    detail = body
                raise HTTPException(status_code=resp.status if passthrough else 502,
                                    detail=detail)
            result = await resp.json()
    except aiohttp.ClientError as exc:
        logger.error("Media node unreachable at %s: %s", MEDIA_NODE_URL, exc)
        raise HTTPException(status_code=503,
                            detail=f"media node unreachable at {MEDIA_NODE_URL}: {exc}") from exc
    except TimeoutError as exc:
        # Only reachable when MEDIA_NODE_TIMEOUT_S is set to a positive value.
        # aiohttp raises this — not ClientError — on the total timeout, so
        # without its own clause it escaped as a bare 500 that told the caller
        # nothing, which is how a cut-off generation used to look.
        logger.error("Media node %s exceeded MEDIA_NODE_TIMEOUT_S=%ds — the node may still be running it",
                     label, MEDIA_NODE_TIMEOUT_S)
        raise HTTPException(
            status_code=504,
            detail=f"media node did not answer within MEDIA_NODE_TIMEOUT_S={MEDIA_NODE_TIMEOUT_S}s; "
                   f"the generation may still be running on the node",
        ) from exc

    elapsed = (time.time() - start) * 1000
    logger.info("Media node %s done %.1fms — seed=%s", label, elapsed, result.get("seed", "?"))
    return result, elapsed


@router.post("/v1/av/generate")
async def generate_av(req: VideoGenerationRequest, request: Request) -> JSONResponse:
    """Generates video with native stereo audio on the media node (MiniMax-H3)."""
    payload = req.model_dump(exclude={"response_format"})

    logger.info("av request — %dx%d | %d frames @ %dfps | steps=%d | agent=%s | node=%s",
                req.width, req.height, req.num_frames, req.fps, req.steps,
                req.agent_id, MEDIA_NODE_URL)

    result, elapsed = await _dispatch("/generate", payload, "av", request)
    if req.response_format == "url":
        result = _to_url_response(result, "video_base64", "mp4")

    result["processing_time_ms"] = elapsed
    return JSONResponse({"success": True, "data": result})


@router.post("/v1/audio/music")
async def generate_music(req: MusicGenerationRequest, request: Request) -> JSONResponse:
    """Generates music on the media node (ACE-Step 1.5 XL Turbo)."""
    payload = req.model_dump(exclude={"response_format"})

    logger.info("music request — %.0fs | %d bpm %s | steps=%d | agent=%s",
                req.duration, req.bpm, req.key_scale, req.steps, req.agent_id)

    result, elapsed = await _dispatch("/generate/music", payload, "music", request)
    if req.response_format == "url":
        result = _to_url_response(result, "audio_base64", "mp3")

    result["processing_time_ms"] = elapsed
    return JSONResponse({"success": True, "data": result})


@router.post("/v1/images/generate")
async def generate_image(req: ImageGenerationRequest, request: Request) -> JSONResponse:
    """Generates images on the media node (HiDream-O1-Image)."""
    payload = req.model_dump(exclude={"response_format"})

    logger.info("image request — %s | %dx%d x%d | refs=%d | agent=%s",
                req.variant, req.width, req.height, req.batch_size,
                len(req.ref_images), req.agent_id)

    result, elapsed = await _dispatch("/generate/image", payload, "image", request)
    if req.response_format == "url":
        result = _to_urls_response(result, "images", "png")

    result["processing_time_ms"] = elapsed
    return JSONResponse({"success": True, "data": result})


# ─────────────────── Asset handling ──────────────────

def _save_asset(b64: str, ext: str) -> tuple[str, int]:
    """Writes one base64 payload to MEDIA_ASSET_DIR. Returns (url, size)."""
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"media node returned invalid base64: {exc}") from exc

    directory = Path(MEDIA_ASSET_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}.{ext}"
    (directory / name).write_bytes(raw)
    return f"{MEDIA_PUBLIC_URL}{ASSET_ROUTE}/{name}", len(raw)


def _require_asset_dir() -> None:
    if not MEDIA_ASSET_DIR:
        raise HTTPException(
            status_code=400,
            detail='response_format="url" requires MEDIA_ASSET_DIR to be set on the Gateway',
        )


def _to_url_response(result: dict[str, Any], b64_key: str, ext: str) -> dict[str, Any]:
    """Swaps a single base64 payload for a URL — video or audio."""
    _require_asset_dir()

    b64 = result.pop(b64_key, None)
    if not b64:
        raise HTTPException(status_code=502, detail=f"media node returned no {b64_key} payload")

    url, size = _save_asset(b64, ext)
    result["url"] = url
    # Keep the modality-specific key too, so existing callers don't break.
    result[f"{ext_to_kind(ext)}_url"] = url
    result["size_bytes"] = size
    return result


def _to_urls_response(result: dict[str, Any], b64_list_key: str, ext: str) -> dict[str, Any]:
    """Same, for a batch — image generation can return several files."""
    _require_asset_dir()

    payloads = result.pop(b64_list_key, None)
    if not payloads:
        raise HTTPException(status_code=502, detail=f"media node returned no {b64_list_key} payload")

    saved = [_save_asset(b64, ext) for b64 in payloads]
    result["urls"] = [url for url, _ in saved]
    result["size_bytes"] = sum(size for _, size in saved)
    return result


def ext_to_kind(ext: str) -> str:
    """mp4 → video, mp3 → audio: the legacy per-modality URL key."""
    return {"mp4": "video", "mp3": "audio", "png": "image"}.get(ext, "asset")


def attach(app, alias_legacy: bool | None = None) -> None:
    """Wires the media node into a Gateway app: routes plus asset serving.

    Safe to call on any Gateway version — it only adds routes, and never
    replaces one the app already defines.

    *alias_legacy* overrides MEDIA_NODE_ALIAS_VIDEOS. Pass True from an app that
    is known to have no local video backend (see media_app.py).
    """
    # Warn about paths the host app already serves. Starlette resolves in
    # registration order, so attaching last means the host's own handler wins —
    # the safe default, but silent. On the main Gateway /v1/images/generate is
    # FLUX's, so HiDream-O1 would be unreachable there without this notice.
    own = {getattr(route, "path", None) for route in app.routes}
    clashes = sorted(p for p in (getattr(r, "path", None) for r in router.routes) if p in own)
    if clashes:
        logger.warning(
            "These media routes are already served by this app and will NOT reach the media node: %s. "
            "Reach them directly on the node's adapter instead.", ", ".join(clashes),
        )

    app.include_router(router)
    paths = sorted(p for p in (getattr(r, "path", None) for r in router.routes) if p and p not in own)

    # H3 generates video *with* audio, so /v1/av/generate is its own path rather
    # than a reuse of /v1/videos/generate (which in this project means LTX-Video,
    # silent). Set MEDIA_NODE_ALIAS_VIDEOS=true on a Gateway that has no local
    # video backend to also answer the older path. Opt-in on purpose: auto-
    # detection can't see routes added via include_router, so it could shadow a
    # working endpoint without telling anyone.
    if ALIAS_LEGACY_VIDEO_PATH if alias_legacy is None else alias_legacy:
        legacy = "/v1/videos/generate"
        if any(getattr(route, "path", None) == legacy for route in app.routes):
            logger.warning(
                "MEDIA_NODE_ALIAS_VIDEOS is set but %s is already registered — skipping alias", legacy
            )
        else:
            app.add_api_route(legacy, generate_av, methods=["POST"], tags=["media"])
            paths.append(legacy)

    if MEDIA_ASSET_DIR:
        from fastapi.staticfiles import StaticFiles

        directory = Path(MEDIA_ASSET_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        app.mount(ASSET_ROUTE, StaticFiles(directory=str(directory)), name="media-assets")
        logger.info("Media assets served from %s at %s", directory, ASSET_ROUTE)

    logger.info("Media node attached at %s — routes: %s", MEDIA_NODE_URL, ", ".join(paths))
