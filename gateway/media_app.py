"""
Media-only Gateway — for a media node that runs standalone.

Serves three modalities from one node:

    /v1/av/generate      MiniMax-H3             video with native stereo audio
    /v1/audio/music      ACE-Step 1.5 XL Turbo  text-to-music
    /v1/images/generate  HiDream-O1-Image       text-to-image at 2048px

Same public API as the main Gateway for media work, minus everything that
assumes a local text cluster. Deliberately does **not** import
`ContainerOrchestrator`: the main app's startup adopts or force-recreates
profiles and force-removes "orphaned" containers, which on a media node would
delete the very container serving the models. Nor is there a VRAM profile to
swap — ComfyUI on the node evicts between model families by itself.

So: no Docker socket, no VRAM profiles, no swapping, no request buffer. Just the
media routes, health, and model discovery.

Run:
    uvicorn gateway.media_app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gateway import media_node
from gateway.config import MEDIA_MODEL_ENDPOINTS, REMOTE_MEDIA_MODELS
from gateway.vram_monitor import VRAMMonitor

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gateway.media_app")

GATEWAY_HOST: str = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT: int = int(os.getenv("GATEWAY_PORT", "8000"))
NODE_NAME: str = os.getenv("NODE_NAME", "media-node")

# Assets saved for response_format="url" are deleted once they age past this.
# Note the trade-off: a URL handed to a client stops resolving when its file is
# swept, so a chat transcript keeps a dead link, not the content. Raise this (or
# set 0 to keep forever) if callers are expected to come back for old results.
RETENTION_HOURS: float = float(os.getenv("MEDIA_RETENTION_HOURS", "24"))
CLEANUP_INTERVAL_S: int = int(os.getenv("MEDIA_CLEANUP_INTERVAL_S", "3600"))

# NVML only — no Docker involved. Reports the unified pool on a GB10.
vram_monitor = VRAMMonitor()

_start_time: float = 0.0


def _cleanup_once() -> tuple[int, int]:
    """Deletes saved assets older than the retention window. Returns (files, bytes)."""
    if RETENTION_HOURS <= 0 or not media_node.MEDIA_ASSET_DIR:
        return 0, 0

    directory = Path(media_node.MEDIA_ASSET_DIR)
    if not directory.is_dir():
        return 0, 0

    cutoff = time.time() - RETENTION_HOURS * 3600
    files = freed = 0
    for path in directory.iterdir():
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            if stat.st_mtime >= cutoff:
                continue
            size = stat.st_size
            path.unlink()
        except OSError as exc:
            # Being written, or already gone — never abort the sweep.
            logger.debug("could not remove %s: %s", path, exc)
            continue
        files += 1
        freed += size
    return files, freed


async def _cleanup_loop() -> None:
    if RETENTION_HOURS <= 0:
        logger.info("Asset retention disabled — files kept indefinitely")
        return
    if not media_node.MEDIA_ASSET_DIR:
        return          # nothing is persisted in b64-only mode

    logger.info("Asset retention: %.1fh, sweeping every %ds", RETENTION_HOURS, CLEANUP_INTERVAL_S)
    while True:
        try:
            files, freed = await asyncio.to_thread(_cleanup_once)
            if files:
                logger.info("Asset sweep removed %d file(s), %.1f MB", files, freed / 2**20)
        except Exception:                       # a broken sweep must not kill the loop
            logger.exception("Asset sweep failed")
        await asyncio.sleep(CLEANUP_INTERVAL_S)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _start_time
    _start_time = time.time()

    logger.info("━━━ Media Gateway v%s starting (%s) ━━━", __version__, NODE_NAME)
    await vram_monitor.start()
    cleanup = asyncio.create_task(_cleanup_loop())
    logger.info("━━━ Media Gateway READY on %s:%d → %s ━━━",
                GATEWAY_HOST, GATEWAY_PORT, media_node.MEDIA_NODE_URL)

    yield

    logger.info("━━━ Shutting down Media Gateway ━━━")
    cleanup.cancel()
    await vram_monitor.stop()
    await media_node.shutdown()


app = FastAPI(
    title="Blackwell Media Gateway",
    description=(
        "Media-only Gateway for a standalone node: video with native stereo audio "
        "(MiniMax-H3), music (ACE-Step 1.5 XL Turbo) and images (HiDream-O1). "
        "No text models, no VRAM profiles, no container orchestration."
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


# ──────────── Health & Status ─────────────────────────

@app.get("/health", tags=["System"])
async def health() -> JSONResponse:
    """Health of this Gateway and of the inference backend behind it."""
    ready, backend = await media_node.probe()
    report = vram_monitor.latest

    # "loading" is the common case for the first seconds after a restart: the
    # adapter answers on its port before ComfyUI has registered its nodes.
    status = "ok" if ready else ("loading" if backend.get("reachable") else "degraded")

    # Per-modality availability comes from the node itself, which checks both that
    # the nodes are registered and that the weights are on disk — so a missing
    # download shows up here rather than as a failed generation later.
    modalities = (backend.get("node") or {}).get("modalities")

    return JSONResponse(
        {
            "status": status,
            "version": __version__,
            "role": "media-node",
            "node": NODE_NAME,
            "models": [m.name for m in REMOTE_MEDIA_MODELS],
            "modalities": modalities,
            "retention_hours": RETENTION_HOURS,
            "backend": backend,
            "vram": report.model_dump() if report else None,
            "uptime_seconds": time.time() - _start_time,
        },
        # 503 until generations would actually succeed, so healthchecks and
        # clients wait instead of firing requests the backend will reject.
        status_code=200 if ready else 503,
    )


@app.get("/status/vram", tags=["System"])
async def vram_status() -> JSONResponse:
    """VRAM report from NVML."""
    report = await vram_monitor.query_gpus()
    return JSONResponse(report.model_dump())


# ──────────── Model discovery ─────────────────────────

@app.get("/v1/models", tags=["Inference"])
async def list_models() -> JSONResponse:
    """Lists the media models (OpenAI format).

    Both H3 checkpoints stay resident, so both are always listed — there is no
    profile to swap and nothing to wait for beyond the initial load.
    """
    data = [
        {
            "id": m.name,
            "object": "model",
            "owned_by": "blackwell-media-gateway",
            "engine": m.engine,
            "vram_mb": m.vram_required_mb,
            "quantization": m.quantization,
            "endpoint": MEDIA_MODEL_ENDPOINTS.get(m.name),
        }
        for m in REMOTE_MEDIA_MODELS
    ]
    # Stable aliases so callers need not know checkpoint names.
    aliases = {
        "av": "minimax-h3-fl2va",
        "video": "minimax-h3-fl2va",
        "music": "ace-step-1.5-xl-turbo",
        "image": "hidream-o1-image",
    }
    data += [
        {"id": alias, "object": "model", "owned_by": "blackwell-media-gateway",
         "engine": "comfyui", "alias_for": target,
         "endpoint": MEDIA_MODEL_ENDPOINTS.get(target)}
        for alias, target in aliases.items()
    ]
    return JSONResponse({"object": "list", "data": data})


# ──────────── Media routes ────────────────────────────
# alias_legacy=True: this app has no local video backend of its own, so
# /v1/videos/generate is free and pointing it at H3 keeps existing clients working.
media_node.attach(app, alias_legacy=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gateway.media_app:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        log_level="info",
        workers=1,
        access_log=True,
    )
