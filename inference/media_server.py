"""
Media inference server for the Blackwell media node.

Wraps one local ComfyUI behind the same `POST /generate` + `GET /health` contract
the Gateway speaks to the other inference servers, and serves three modalities
from it:

    av      MiniMax-H3            video with native stereo audio (t2va/fl2va/ref2va)
    music   ACE-Step 1.5 XL Turbo text-to-music, 8 steps
    image   Qwen-Image-2.1      text-to-image at 2048px, optional reference images

One ComfyUI process, one GPU, one lock: requests are serialised, and ComfyUI
evicts whichever model it needs to. All the graphs below are transcribed from
authoritative sources rather than guessed —

    av     comfy_extras/nodes_minimax_h3.py (node schemas)
    music  ComfyUI blueprints/"Text to Audio (ACE-Step 1.5).json"
    image  Comfy-Org/workflow_templates image_qwen_image_2_1_{t2i,image_edit}.json

— because several of the encodings are not discoverable from `/object_info`
alone (see the DYNAMICCOMBO and Autogrow notes further down).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("media_server")

# ─────────────────── Configuration ───────────────────

COMFY_URL: str = os.getenv("COMFY_URL", "http://127.0.0.1:8188")
PORT: int = int(os.getenv("PORT", "8010"))

# ── av: MiniMax-H3 ──
CKPT_FL2VA: str = os.getenv("CKPT_FL2VA", "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
CKPT_REF2VA: str = os.getenv("CKPT_REF2VA", "minimax_h3_ref2va_pruned_int8_convrot.safetensors")
H3_TEXT_ENCODER: str = os.getenv("H3_TEXT_ENCODER", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
H3_VIDEO_VAE: str = os.getenv("H3_VIDEO_VAE", "minimax_h3_video_vae_fp16.safetensors")
H3_AUDIO_VAE: str = os.getenv("H3_AUDIO_VAE", "minimax_h3_audio_vae_fp32.safetensors")

# ── music: ACE-Step 1.5 XL Turbo ──
ACE_DIT: str = os.getenv("ACE_DIT", "acestep_v1.5_xl_turbo_bf16.safetensors")
# Two encoders, in this order — DualCLIPLoader with type "ace".
ACE_CLIP_1: str = os.getenv("ACE_CLIP_1", "qwen_0.6b_ace15.safetensors")
ACE_CLIP_2: str = os.getenv("ACE_CLIP_2", "qwen_4b_ace15.safetensors")
ACE_VAE: str = os.getenv("ACE_VAE", "ace_1.5_vae.safetensors")

# ── image: Qwen-Image-2.1 ── separate diffusion model, encoder and VAE
IMAGE_DIT: str = os.getenv("IMAGE_DIT", "qwen_image_2.1_int8_convrot.safetensors")
IMAGE_TEXT_ENCODER: str = os.getenv("IMAGE_TEXT_ENCODER", "qwen3vl_8b_int8_convrot.safetensors")
IMAGE_VAE: str = os.getenv("IMAGE_VAE", "qwen_image_2.1_vae_bf16.safetensors")
IMAGE_STEPS = 25
IMAGE_CFG = 1.0

# Ceiling on a whole generation. 0 = none, which is the default: a 15s clip at
# full canvas runs ~55 min in one pass, so the old 1800s cap killed exactly the
# requests it existed to protect — and killed them after paying the full GPU
# cost. Nothing hangs forever regardless: the caller's disconnect is watched for
# (see _await_result) and if ComfyUI dies the entrypoint takes the container with
# it, so every wait has an end that is not a clock.
GENERATE_TIMEOUT_S: int = int(os.getenv("GENERATE_TIMEOUT_S", "0"))
# Ceiling on one HTTP call to ComfyUI — deliberately separate from the deadline
# above. Every call this session makes is short (a /history poll, a queue edit,
# reading one finished file off loopback), so a low value here still catches a
# wedged ComfyUI without bounding the generation itself. Conflating the two is
# what made "no generation timeout" impossible to express before.
COMFY_HTTP_TIMEOUT_S: int = int(os.getenv("COMFY_HTTP_TIMEOUT_S", "600"))
POLL_INTERVAL_S: float = float(os.getenv("POLL_INTERVAL_S", "1.5"))
COMFY_BOOT_TIMEOUT_S: int = int(os.getenv("COMFY_BOOT_TIMEOUT_S", "300"))

# ── Retention ──
# Generated files and uploaded references are deleted once they age past this.
# Nothing here is a system of record: every response already carried the bytes
# (or a URL the Gateway serves from its own copy). 0 disables the sweep.
RETENTION_HOURS: float = float(os.getenv("MEDIA_RETENTION_HOURS", "24"))
CLEANUP_INTERVAL_S: int = int(os.getenv("MEDIA_CLEANUP_INTERVAL_S", "3600"))
COMFY_ROOT: str = os.getenv("COMFY_ROOT", "/opt/ComfyUI")

# H3 native canvas: 768px short edge, capped at 768*1344, axes multiple of 32.
AV_WIDTH, AV_HEIGHT = 1344, 768
# Frame counts snap to the model's 17k+5 grid at 24 fps; 124 frames ~ 5.2s.
AV_LENGTH, AV_FPS = 124, 24

# Request ceilings. Cost is dominated by attention over the video latent
# (T x H/16 x W/16 tokens) and grows far faster than pixel count, so an
# unbounded request is not merely slow — 999 frames at full canvas is ~1.19M
# tokens, tens of hours, and well outside the model's trained range, so the
# result would be junk anyway. The node itself allows up to 3600 frames; these
# caps keep requests inside what H3 was trained for (~124-362 frames).
AV_MAX_FRAMES: int = int(os.getenv("AV_MAX_FRAMES", "362"))
AV_MAX_PIXELS: int = int(os.getenv("AV_MAX_PIXELS", str(768 * 1344)))
# Warn (don't reject) past this multiple of the default request's token count.
AV_COST_WARN_RATIO: float = float(os.getenv("AV_COST_WARN_RATIO", "2.0"))

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".m4v")
AUDIO_EXTENSIONS = (".mp3", ".flac", ".opus", ".wav")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# Nodes and weight files each modality needs. /health reports a modality as
# available only when both are present, so a half-finished download shows up as
# an unavailable modality instead of a failed request later.
MODALITY_NODES: dict[str, tuple[str, ...]] = {
    "av": ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo", "MiniMaxH3SigmaShift"),
    "music": ("TextEncodeAceStepAudio1.5", "EmptyAceStep1.5LatentAudio", "ModelSamplingAuraFlow"),
    "image": ("UNETLoader", "CLIPLoader", "VAELoader", "TextEncodeQwenImage21",
              "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage", "LoadImage"),
}

_session: aiohttp.ClientSession | None = None
_boot_error: str | None = None
_capabilities: dict[str, bool] = {"av": False, "music": False, "image": False}
_files: dict[str, set[str]] = {}
# The models are large and the GPU is single; overlapping generations would only
# thrash unified memory, so one runs at a time.
_gpu_lock = asyncio.Lock()


# ─────────────────── Request models ──────────────────

def av_latent_tokens(width: int, height: int, num_frames: int) -> int:
    """Video-latent token count — the thing that actually drives cost.

    Mirrors ComfyUI's grid: frames snap up to 17k+5, and the latent is
    T x H/16 x W/16 where T = ((frames - 5) / 17) * 5 + 2.
    """
    frames = max(5, num_frames)
    while frames % 17 != 5:
        frames += 1
    latent_t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
    return latent_t * (height // 16) * (width // 16)


AV_DEFAULT_TOKENS = av_latent_tokens(AV_WIDTH, AV_HEIGHT, AV_LENGTH)


class AVRequest(BaseModel):
    """MiniMax-H3. Mode follows the assets sent: none → t2va, keyframes → fl2va,
    any ref_* → ref2va (which also switches checkpoint)."""

    prompt: str
    negative_prompt: str = ""

    width: int = Field(default=AV_WIDTH, ge=32, le=2048)
    height: int = Field(default=AV_HEIGHT, ge=32, le=2048)
    # Capped at H3's trained range; see AV_MAX_FRAMES.
    num_frames: int = Field(default=AV_LENGTH, ge=5, le=AV_MAX_FRAMES)
    fps: int = Field(default=AV_FPS, ge=1, le=60)

    steps: int = Field(default=20, ge=1, le=100)
    # 1.0 disables CFG. Raise to ~3-6 only if prompt adherence is weak.
    cfg_scale: float = 1.0
    sampler: str = "res_multistep"
    scheduler: str = "simple"
    shift_video: float = 12.0
    shift_audio: float = 3.0
    seed: int | None = None

    first_frame: str | None = None
    last_frame: str | None = None

    ref_images: list[str] = Field(default_factory=list)
    ref_videos: list[str] = Field(default_factory=list)
    ref_video_audios: list[str] = Field(default_factory=list)
    ref_audios: list[str] = Field(default_factory=list)
    ref_image_size: Literal["match", "max"] = "match"

    agent_id: str | None = None

    @model_validator(mode="after")
    def _within_canvas(self):
        pixels = self.width * self.height
        if pixels > AV_MAX_PIXELS:
            raise ValueError(
                f"{self.width}x{self.height} is {pixels:,} pixels; H3's canvas caps at "
                f"{AV_MAX_PIXELS:,} (e.g. 1344x768). Raise AV_MAX_PIXELS to override."
            )
        return self


class MusicRequest(BaseModel):
    """ACE-Step 1.5 XL Turbo. Defaults are the blueprint's."""

    # "tags" is the style/genre prompt; ACE-Step is not a natural-language model.
    prompt: str = ""
    lyrics: str = ""
    # ACE-Step's latent is linear in duration, so this is a soft cost, but an
    # unbounded value still lets one request occupy the GPU indefinitely.
    duration: float = Field(default=120.0, gt=0, le=600)

    bpm: int = Field(default=120, ge=20, le=300)
    time_signature: Literal["2", "3", "4", "6"] = "4"
    language: str = "en"
    key_scale: str = "C major"

    steps: int = Field(default=8, ge=1, le=100)   # XL Turbo samples in 8
    cfg_scale: float = 1.0          # KSampler CFG
    sampler: str = "euler"
    scheduler: str = "simple"
    shift: float = 3.0              # ModelSamplingAuraFlow

    # These steer the audio-code LM inside the conditioning node — distinct from
    # cfg_scale above, which is the diffusion sampler's.
    lm_cfg_scale: float = 2.0
    temperature: float = 0.85
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    generate_audio_codes: bool = True

    mp3_quality: Literal["V0", "128k", "320k"] = "320k"
    seed: int | None = None
    agent_id: str | None = None


class ImageRequest(BaseModel):
    """Qwen-Image-2.1 generation and reference-image editing."""

    model_config = {"extra": "forbid"}

    prompt: str
    negative_prompt: str = ""

    # Qwen-Image-2.1's native canvas.
    width: int = Field(default=2048, ge=64, le=4096)
    height: int = Field(default=2048, ge=64, le=4096)
    batch_size: int = Field(default=1, ge=1, le=8)

    steps: int | None = Field(default=None, ge=1, le=150)   # None → 25
    cfg_scale: float | None = None  # None → 1.0
    scheduler: str = "simple"
    seed: int | None = None

    # 1 image = instruction edit; 2-10 = multi-reference.
    ref_images: list[str] = Field(default_factory=list)

    agent_id: str | None = None


# ─────────────────── ComfyUI plumbing ────────────────

async def _comfy_get(path: str, **kwargs) -> Any:
    async with _session.get(f"{COMFY_URL}{path}", **kwargs) as resp:
        resp.raise_for_status()
        return await resp.json()


def _combo_options(info: dict, node: str, field: str) -> set[str]:
    """Files a loader node can see, straight from its combo options."""
    try:
        spec = info[node]["input"]["required"][field]
        return set(spec[0]) if isinstance(spec[0], list) else set()
    except (KeyError, IndexError, TypeError):
        return set()


async def _probe_comfy() -> None:
    """Waits for ComfyUI, then works out which modalities are actually usable."""
    global _boot_error, _capabilities, _files

    deadline = time.time() + COMFY_BOOT_TIMEOUT_S
    last_exc: Exception | None = None

    while time.time() < deadline:
        try:
            info = await _comfy_get("/object_info", timeout=aiohttp.ClientTimeout(total=60))
        except Exception as exc:            # still importing torch and nodes
            last_exc = exc
            await asyncio.sleep(2)
            continue

        _files = {
            "diffusion_models": _combo_options(info, "UNETLoader", "unet_name"),
            "text_encoders": _combo_options(info, "CLIPLoader", "clip_name"),
            "vae": _combo_options(info, "VAELoader", "vae_name"),
            "checkpoints": _combo_options(info, "CheckpointLoaderSimple", "ckpt_name"),
        }

        required_files = {
            "av": [("diffusion_models", CKPT_FL2VA), ("text_encoders", H3_TEXT_ENCODER),
                   ("vae", H3_VIDEO_VAE), ("vae", H3_AUDIO_VAE)],
            "music": [("diffusion_models", ACE_DIT), ("text_encoders", ACE_CLIP_1),
                      ("text_encoders", ACE_CLIP_2), ("vae", ACE_VAE)],
            "image": [("diffusion_models", IMAGE_DIT), ("text_encoders", IMAGE_TEXT_ENCODER),
                      ("vae", IMAGE_VAE)],
        }

        caps, missing = {}, {}
        for modality, nodes in MODALITY_NODES.items():
            absent_nodes = [n for n in nodes if n not in info]
            absent_files = [f for kind, f in required_files[modality] if f not in _files[kind]]
            caps[modality] = not absent_nodes and not absent_files
            if absent_nodes or absent_files:
                missing[modality] = {"nodes": absent_nodes, "files": absent_files}

        _capabilities = caps
        _boot_error = None if any(caps.values()) else f"no modality available: {json.dumps(missing)}"
        logger.info("ComfyUI ready — %d node types | available: %s",
                    len(info), ", ".join(k for k, v in caps.items() if v) or "none")
        if missing:
            logger.warning("Unavailable modalities: %s", json.dumps(missing))
        return

    _boot_error = f"ComfyUI did not become ready in {COMFY_BOOT_TIMEOUT_S}s: {last_exc}"
    logger.error(_boot_error)


async def _upload_asset(payload_b64: str, kind: str) -> str:
    """Uploads a base64 asset to ComfyUI's input dir; returns its filename."""
    if "," in payload_b64 and payload_b64.lstrip().startswith("data:"):
        payload_b64 = payload_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64 in {kind}: {exc}") from exc

    suffix = {"image": "png", "audio": "wav", "video": "mp4"}[kind]
    content_type = {"image": "image/png", "audio": "audio/wav", "video": "video/mp4"}[kind]
    filename = f"in_{kind}_{uuid.uuid4().hex[:12]}.{suffix}"

    form = aiohttp.FormData()
    form.add_field("image", raw, filename=filename, content_type=content_type)
    form.add_field("overwrite", "true")

    # /upload/image is ComfyUI's generic file upload — the form field is "image"
    # whatever the actual type, and the bytes are written to the input dir as-is.
    async with _session.post(f"{COMFY_URL}/upload/image", data=form) as resp:
        if resp.status != 200:
            raise HTTPException(status_code=502, detail=f"ComfyUI rejected {kind} upload: {await resp.text()}")
        return (await resp.json()).get("name", filename)


# ─────────────────── av: MiniMax-H3 ──────────────────

def _av_mode(uploads: dict[str, list[str]]) -> str:
    if uploads["ref_images"] or uploads["ref_videos"] or uploads["ref_audios"]:
        return "ref2va"
    if uploads["first_frame"] or uploads["last_frame"]:
        return "fl2va"
    return "t2va"


def _build_av_workflow(req: AVRequest, seed: int, uploads: dict[str, list[str]]) -> tuple[dict, str]:
    """
        UNETLoader ─▶ SigmaShift ──────────────────────────┐
        CLIPLoader ─┐                                      ▼
        VAE(video) ─┼─▶ ImageToVideo | ReferenceToVideo ─▶ KSampler ─┬─▶ VAEDecode ──────┐
        VAE(audio) ─┘         (positive, AV latent)                 └─▶ VAEDecodeAudio ─┤
                                                                                        ▼
                                                                    CreateVideo ─▶ SaveVideo
    """
    is_ref = _av_mode(uploads) == "ref2va"

    g: dict[str, Any] = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": CKPT_REF2VA if is_ref else CKPT_FL2VA,
                            "weight_dtype": "default"}},
        "shift": {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"model": ["unet", 0],
                             "shift_video": req.shift_video,
                             "shift_audio": req.shift_audio}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": H3_TEXT_ENCODER, "type": "minimax"}},
        "vae_video": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
        "vae_audio": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
    }

    if is_ref:
        cond_inputs: dict[str, Any] = {
            "clip": ["clip", 0],
            "vae": ["vae_video", 0],
            "audio_vae": ["vae_audio", 0],
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "length": req.num_frames,
            "ref_image_size": req.ref_image_size,
        }
        # Autogrow keys are dotted "<container>.<prefix><index>" paths, which
        # ComfyUI re-nests into the dict execute() receives. The bare
        # "ref_image_0" form arrives as an unexpected keyword argument.
        for i, name in enumerate(uploads["ref_images"]):
            g[f"load_ref_img_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            cond_inputs[f"ref_images.ref_image_{i}"] = [f"load_ref_img_{i}", 0]

        for i, name in enumerate(uploads["ref_videos"]):
            # The node wants frames (IMAGE), so the container gets split.
            g[f"load_ref_vid_{i}"] = {"class_type": "LoadVideo", "inputs": {"file": name}}
            g[f"split_ref_vid_{i}"] = {"class_type": "GetVideoComponents",
                                       "inputs": {"video": [f"load_ref_vid_{i}", 0]}}
            cond_inputs[f"ref_videos.ref_video_{i}"] = [f"split_ref_vid_{i}", 0]

            explicit = uploads["ref_video_audios"]
            audio_key = f"ref_video_audios.ref_video_audio_{i}"
            if i < len(explicit) and explicit[i]:
                g[f"load_ref_vid_audio_{i}"] = {"class_type": "LoadAudio",
                                                "inputs": {"audio": explicit[i]}}
                cond_inputs[audio_key] = [f"load_ref_vid_audio_{i}", 0]
            else:
                # GetVideoComponents returns (images, audio, fps, bit_depth) —
                # reuse the container's own soundtrack when none was supplied.
                cond_inputs[audio_key] = [f"split_ref_vid_{i}", 1]

        for i, name in enumerate(uploads["ref_audios"]):
            g[f"load_ref_audio_{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
            cond_inputs[f"ref_audios.ref_audio_{i}"] = [f"load_ref_audio_{i}", 0]

        g["cond"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond_inputs}
    else:
        cond_inputs = {
            "clip": ["clip", 0],
            "vae": ["vae_video", 0],
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "length": req.num_frames,
        }
        for slot, name in (("first_frame", uploads["first_frame"]),
                           ("last_frame", uploads["last_frame"])):
            if name:
                node_id = f"load_{slot}"
                g[node_id] = {"class_type": "LoadImage", "inputs": {"image": name[0]}}
                cond_inputs[slot] = [node_id, 0]

        g["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond_inputs}

    # H3 emits positive only; CFG needs a negative branch. At cfg 1.0 the sampler
    # ignores it, so it costs nothing to always wire up.
    g["negative"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["cond", 0]}}
    g["sampler"] = {
        "class_type": "KSampler",
        "inputs": {"model": ["shift", 0], "positive": ["cond", 0], "negative": ["negative", 0],
                   "latent_image": ["cond", 1], "seed": seed, "steps": req.steps,
                   "cfg": req.cfg_scale, "sampler_name": req.sampler,
                   "scheduler": req.scheduler, "denoise": 1.0},
    }
    # One nested AV latent in, two streams out: VAEDecode takes the video half,
    # VAEDecodeAudio the audio half.
    g["decode_video"] = {"class_type": "VAEDecode",
                         "inputs": {"samples": ["sampler", 0], "vae": ["vae_video", 0]}}
    g["decode_audio"] = {"class_type": "VAEDecodeAudio",
                         "inputs": {"samples": ["sampler", 0], "vae": ["vae_audio", 0]}}
    g["mux"] = {"class_type": "CreateVideo",
                "inputs": {"images": ["decode_video", 0], "audio": ["decode_audio", 0],
                           "fps": float(req.fps)}}
    # `codec` is a DYNAMICCOMBO: the API value is the bare option key. A dict
    # here is silently dropped and SaveVideo fails on a missing argument.
    g["save"] = {"class_type": "SaveVideo",
                 "inputs": {"video": ["mux", 0], "filename_prefix": "media/av",
                            "format": "auto", "codec": "auto"}}
    return g, "save"


# ─────────────────── music: ACE-Step 1.5 ────────────

def _build_music_workflow(req: MusicRequest, seed: int) -> tuple[dict, str]:
    """Transcribed from ComfyUI's "Text to Audio (ACE-Step 1.5)" blueprint.

        UNETLoader ─▶ ModelSamplingAuraFlow ─┐
        DualCLIPLoader ─▶ TextEncodeAceStep ─┼─▶ KSampler ─▶ VAEDecodeAudio ─▶ SaveAudioMP3
        EmptyAceStepLatentAudio ─────────────┘
    """
    g: dict[str, Any] = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": ACE_DIT, "weight_dtype": "default"}},
        "shift": {"class_type": "ModelSamplingAuraFlow",
                  "inputs": {"model": ["unet", 0], "shift": req.shift}},
        # Both encoders are required, in this order.
        "clip": {"class_type": "DualCLIPLoader",
                 "inputs": {"clip_name1": ACE_CLIP_1, "clip_name2": ACE_CLIP_2,
                            "type": "ace", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": ACE_VAE}},
        "latent": {"class_type": "EmptyAceStep1.5LatentAudio",
                   "inputs": {"seconds": req.duration, "batch_size": 1}},
        "cond": {"class_type": "TextEncodeAceStepAudio1.5",
                 "inputs": {
                     "clip": ["clip", 0],
                     "tags": req.prompt,
                     "lyrics": req.lyrics,
                     "seed": seed,
                     "bpm": req.bpm,
                     "duration": req.duration,
                     "timesignature": req.time_signature,
                     "language": req.language,
                     "keyscale": req.key_scale,
                     "generate_audio_codes": req.generate_audio_codes,
                     # The conditioning node's own CFG, for the audio-code LM.
                     "cfg_scale": req.lm_cfg_scale,
                     "temperature": req.temperature,
                     "top_p": req.top_p,
                     "top_k": req.top_k,
                     "min_p": req.min_p,
                 }},
    }
    g["negative"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["cond", 0]}}
    g["sampler"] = {
        "class_type": "KSampler",
        "inputs": {"model": ["shift", 0], "positive": ["cond", 0], "negative": ["negative", 0],
                   "latent_image": ["latent", 0], "seed": seed, "steps": req.steps,
                   "cfg": req.cfg_scale, "sampler_name": req.sampler,
                   "scheduler": req.scheduler, "denoise": 1.0},
    }
    g["decode"] = {"class_type": "VAEDecodeAudio",
                   "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    # MP3 over SaveAudio's FLAC: an API ships these over HTTP. SaveAudioAdvanced
    # would mean another DYNAMICCOMBO for no gain.
    g["save"] = {"class_type": "SaveAudioMP3",
                 "inputs": {"audio": ["decode", 0], "filename_prefix": "media/music",
                            "quality": req.mp3_quality}}
    return g, "save"


# ─────────────────── image: Qwen-Image-2.1 ──────────────

def _build_image_workflow(req: ImageRequest, seed: int, ref_images: list[str]) -> tuple[dict, str]:
    """Official Qwen 2.1 workflow, with explicit output size and batch size.

    Sources: Comfy-Org/workflow_templates image_qwen_image_2_1_{t2i,image_edit}.json.
    Reference images enter both the vision encoder and VAE through TextEncodeQwenImage21.
    """
    g: dict[str, Any] = {
        "model": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": IMAGE_DIT, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": IMAGE_TEXT_ENCODER, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": IMAGE_VAE}},
        "encode": {"class_type": "TextEncodeQwenImage21",
                   "inputs": {"clip": ["clip", 0], "prompt": req.prompt,
                              "negative_prompt": req.negative_prompt, "resolution": 1024}},
        "latent": {"class_type": "EmptyLatentImage",
                   "inputs": {"width": req.width, "height": req.height, "batch_size": req.batch_size}},
        "sample": {"class_type": "KSampler",
                   "inputs": {"model": ["model", 0], "seed": seed,
                              "steps": req.steps if req.steps is not None else IMAGE_STEPS,
                              "cfg": req.cfg_scale if req.cfg_scale is not None else IMAGE_CFG,
                              "sampler_name": "euler", "scheduler": req.scheduler, "denoise": 1.0,
                              "positive": ["encode", 0], "negative": ["encode", 1],
                              "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": "media/image"}},
    }
    if ref_images:
        g["encode"]["inputs"]["vae"] = ["vae", 0]
        for i, name in enumerate(ref_images, start=1):
            g[f"load_ref_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            g["encode"]["inputs"][f"images.image_{i}"] = [f"load_ref_{i}", 0]
    return g, "save"


# ─────────────────── Execution ───────────────────────

async def _submit(graph: dict) -> str:
    async with _session.post(f"{COMFY_URL}/prompt",
                             json={"prompt": graph, "client_id": f"gw-{uuid.uuid4().hex[:8]}"}) as resp:
        body = await resp.text()
        if resp.status != 200:
            # ComfyUI's validation error names the offending node and input —
            # pass it through verbatim, it is the only useful diagnostic.
            raise HTTPException(status_code=502, detail=f"ComfyUI rejected the workflow: {body[:2000]}")
        return json.loads(body)["prompt_id"]


class ClientGone(Exception):
    """The caller went away while we were waiting for ComfyUI."""


async def _await_result(prompt_id: str, is_disconnected: Any = None) -> dict:
    # None when GENERATE_TIMEOUT_S is 0 — poll until the job ends or the caller does.
    deadline = time.time() + GENERATE_TIMEOUT_S if GENERATE_TIMEOUT_S > 0 else None

    while deadline is None or time.time() < deadline:
        # uvicorn does not cancel the handler when a client disconnects — it
        # just discards the response — so the only way to notice is to ask.
        # Without this the job runs on for nobody, and later requests pile up
        # inside ComfyUI's queue where this server's lock cannot see them.
        if is_disconnected is not None and await is_disconnected():
            raise ClientGone(prompt_id)

        history = await _comfy_get(f"/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                messages = status.get("messages") or []
                raise HTTPException(status_code=500,
                                    detail=f"ComfyUI execution failed: {json.dumps(messages)[:2000]}")
            if entry.get("outputs"):
                return entry["outputs"]
        await asyncio.sleep(POLL_INTERVAL_S)

    # Unreachable unless GENERATE_TIMEOUT_S was set to a positive value.
    raise HTTPException(status_code=504,
                        detail=f"generation exceeded GENERATE_TIMEOUT_S={GENERATE_TIMEOUT_S}s "
                               f"(prompt_id={prompt_id})")


def _find_outputs(outputs: dict, extensions: tuple[str, ...]) -> list[dict]:
    """Pulls saved-file references out of a history entry, preferring *extensions*."""
    candidates: list[dict] = []
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for value in node_output.values():
            if isinstance(value, list):
                candidates += [v for v in value if isinstance(v, dict) and "filename" in v]

    matching = [c for c in candidates if str(c["filename"]).lower().endswith(extensions)]
    if matching:
        return matching
    if candidates:
        return candidates
    raise HTTPException(status_code=500,
                        detail=f"no output file in ComfyUI history: {json.dumps(outputs)[:1000]}")


async def _fetch_output(ref: dict) -> bytes:
    params = {"filename": ref["filename"], "subfolder": ref.get("subfolder", ""),
              "type": ref.get("type", "output")}
    async with _session.get(f"{COMFY_URL}/view", params=params) as resp:
        if resp.status != 200:
            raise HTTPException(status_code=502, detail=f"could not read {ref['filename']} from ComfyUI")
        return await resp.read()


async def _cancel_prompt(prompt_id: str) -> None:
    """Drops a prompt from ComfyUI once nobody is waiting for it.

    Releasing the GPU lock is not enough: the prompt has already been submitted,
    so without this it keeps the GPU busy for no one and the next request queues
    up behind it inside ComfyUI, invisible to this server's serialisation.
    """
    try:
        # Harmless if it already started — it just won't be in the pending list.
        async with _session.post(f"{COMFY_URL}/queue", json={"delete": [prompt_id]}):
            pass
        queue = await _comfy_get("/queue", timeout=aiohttp.ClientTimeout(total=15))
        # Queue entries are [number, prompt_id, prompt, extra, outputs].
        running = [entry[1] for entry in queue.get("queue_running", []) if len(entry) > 1]
        if prompt_id in running:
            async with _session.post(f"{COMFY_URL}/interrupt"):
                pass
            logger.info("Interrupted abandoned prompt %s", prompt_id)
        else:
            logger.info("Removed abandoned prompt %s from the queue", prompt_id)
    except Exception as exc:
        logger.warning("Could not cancel prompt %s: %s", prompt_id, exc)


async def _run(
    graph: dict,
    extensions: tuple[str, ...],
    request: Request | None = None,
) -> tuple[list[dict], list[bytes], float]:
    start = time.time()
    is_disconnected = request.is_disconnected if request is not None else None

    async with _gpu_lock:
        prompt_id = await _submit(graph)
        try:
            outputs = await _await_result(prompt_id, is_disconnected)
        except ClientGone:
            logger.info("Caller disconnected — cancelling prompt %s", prompt_id)
            await _cancel_prompt(prompt_id)
            raise HTTPException(status_code=499, detail="client disconnected") from None
        except asyncio.CancelledError:
            # Detached on purpose: awaiting inside a cancelled task aborts at the
            # first await, leaving the job running with nobody to collect it.
            asyncio.create_task(_cancel_prompt(prompt_id))
            raise
        except Exception:
            # Timeouts and backend errors leave the same orphan behind.
            await _cancel_prompt(prompt_id)
            raise
    refs = _find_outputs(outputs, extensions)
    payloads = [await _fetch_output(r) for r in refs]
    return refs, payloads, (time.time() - start) * 1000


# ─────────────────── Retention ───────────────────────

def sweep_dir(directory: Path, max_age_s: float, pattern: str = "*") -> tuple[int, int]:
    """Deletes files under *directory* older than *max_age_s*. Returns (files, bytes).

    Recursive, and only ever touches files — empty directories are left alone
    because ComfyUI creates and expects some of them.
    """
    if max_age_s <= 0 or not directory.is_dir():
        return 0, 0

    cutoff = time.time() - max_age_s
    files = freed = 0

    for path in directory.rglob(pattern):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            if stat.st_mtime >= cutoff:
                continue
            size = stat.st_size
            path.unlink()
        except OSError as exc:
            # A file being written, or already gone — never abort the sweep.
            logger.debug("could not remove %s: %s", path, exc)
            continue
        files += 1
        freed += size

    return files, freed


def cleanup_once() -> tuple[int, int]:
    """One retention pass over everything this process owns."""
    max_age_s = RETENTION_HOURS * 3600
    files = freed = 0

    # Generations. Confined to the media/ subfolder this server writes to, so
    # ComfyUI's own output-dir marker file is never touched.
    f, b = sweep_dir(Path(COMFY_ROOT) / "output" / "media", max_age_s)
    files, freed = files + f, freed + b

    # Uploaded references. Only our own "in_*" names: ComfyUI ships bundled
    # example inputs that some of its templates expect to find.
    f, b = sweep_dir(Path(COMFY_ROOT) / "input", max_age_s, pattern="in_*")
    files, freed = files + f, freed + b

    return files, freed


async def _cleanup_loop() -> None:
    """Sweeps at startup and then every CLEANUP_INTERVAL_S."""
    if RETENTION_HOURS <= 0:
        logger.info("Retention disabled (MEDIA_RETENTION_HOURS=0) — files kept indefinitely")
        return

    logger.info("Retention: %.1fh, sweeping every %ds", RETENTION_HOURS, CLEANUP_INTERVAL_S)
    while True:
        try:
            files, freed = await asyncio.to_thread(cleanup_once)
            if files:
                logger.info("Retention sweep removed %d file(s), %.1f MB", files, freed / 2**20)
        except Exception:                       # a broken sweep must not kill the loop
            logger.exception("Retention sweep failed")
        await asyncio.sleep(CLEANUP_INTERVAL_S)


def _require(modality: str) -> None:
    if not _capabilities.get(modality):
        raise HTTPException(
            status_code=503,
            detail=_boot_error or f"modality '{modality}' unavailable — check its weights are downloaded",
        )


# ─────────────────── Lifecycle ───────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _session
    _session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=COMFY_HTTP_TIMEOUT_S, connect=10),
    )
    # Probe in the background so the container is up immediately; /health
    # reports 503 until ComfyUI answers.
    probe = asyncio.create_task(_probe_comfy())
    cleanup = asyncio.create_task(_cleanup_loop())

    yield

    probe.cancel()
    cleanup.cancel()
    await _session.close()


app = FastAPI(title="Blackwell media inference server", lifespan=lifespan)


# ─────────────────── Endpoints ───────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    ready = any(_capabilities.values())
    body: dict[str, Any] = {
        "status": "ok" if ready else "loading",
        "ready": ready,
        "modalities": _capabilities,
        "retention_hours": RETENTION_HOURS,
        "models": {
            "av": {"model": "minimax-h3", "modes": ["t2va", "fl2va", "ref2va"],
                   "checkpoints": {"fl2va": CKPT_FL2VA, "ref2va": CKPT_REF2VA}},
            "music": {"model": "ace-step-1.5-xl-turbo", "checkpoint": ACE_DIT},
            "image": {"model": "qwen-image-2.1",
                      "diffusion_model": IMAGE_DIT,
                      "text_encoder": IMAGE_TEXT_ENCODER, "vae": IMAGE_VAE},
        },
    }
    if _boot_error:
        body["error"] = _boot_error
    return JSONResponse(body, status_code=200 if ready else 503)


@app.post("/generate")
async def generate_av(req: AVRequest, request: Request) -> JSONResponse:
    """MiniMax-H3 — video with native stereo audio."""
    _require("av")

    # Per-container maxima, transcribed from the Autogrow templates in
    # comfy_extras/nodes_minimax_h3.py: ref_images max 9, and 3 each for
    # ref_videos, ref_video_audios and ref_audios. Beyond these ComfyUI rejects
    # the graph, so catching them here only makes the error legible.
    if len(req.ref_images) > 9:
        raise HTTPException(status_code=400, detail="ref2va accepts at most 9 reference images")
    if len(req.ref_videos) > 3 or len(req.ref_audios) > 3:
        raise HTTPException(status_code=400, detail="ref2va accepts at most 3 reference videos and 3 audios")
    if len(req.ref_video_audios) > 3:
        raise HTTPException(status_code=400,
                            detail="ref2va accepts at most 3 reference video soundtracks")
    # ref_video_audio_N is the soundtrack *of* ref_video_N — the node pairs them
    # by index and ignores any with no matching video. Without this the extras are
    # base64-decoded, uploaded and then silently dropped from the graph.
    if len(req.ref_video_audios) > len(req.ref_videos):
        raise HTTPException(
            status_code=400,
            detail=f"got {len(req.ref_video_audios)} ref_video_audios for {len(req.ref_videos)} "
                   f"ref_videos — each is the soundtrack of the same-numbered video, so the "
                   f"extras would be ignored. Pass standalone audio as ref_audios instead.",
        )
    # An explicitly supplied soundtrack is a file the caller sent, so it counts.
    # NB: the 12 is *not* in the node schema, whose per-container maxima allow
    # 9+3+3+3=18. It is a stricter ceiling this adapter imposes; if it came from
    # H3's model card it should cite it, and if not it is rejecting requests
    # ComfyUI would accept. Left as-is rather than loosened on a guess.
    total_refs = (len(req.ref_images) + len(req.ref_videos)
                  + len(req.ref_video_audios) + len(req.ref_audios))
    if total_refs > 12:
        raise HTTPException(status_code=400, detail=f"ref2va accepts at most 12 reference files, got {total_refs}")

    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
    uploads: dict[str, list[str]] = {
        "first_frame": [await _upload_asset(req.first_frame, "image")] if req.first_frame else [],
        "last_frame": [await _upload_asset(req.last_frame, "image")] if req.last_frame else [],
        "ref_images": [await _upload_asset(a, "image") for a in req.ref_images],
        "ref_videos": [await _upload_asset(a, "video") for a in req.ref_videos],
        "ref_video_audios": [await _upload_asset(a, "audio") for a in req.ref_video_audios],
        "ref_audios": [await _upload_asset(a, "audio") for a in req.ref_audios],
    }
    mode = _av_mode(uploads)
    graph, _ = _build_av_workflow(req, seed, uploads)

    tokens = av_latent_tokens(req.width, req.height, req.num_frames)
    ratio = tokens / AV_DEFAULT_TOKENS
    logger.info("av/%s | %dx%d | %d frames @ %dfps | steps=%d cfg=%.1f | %s tokens (%.1fx default) "
                "| seed=%d | agent=%s",
                mode, req.width, req.height, req.num_frames, req.fps,
                req.steps, req.cfg_scale, f"{tokens:,}", ratio, seed, req.agent_id)
    if ratio > AV_COST_WARN_RATIO:
        # Attention is superlinear in this, so the wall clock grows faster still.
        logger.warning("Expensive request: %.1fx the default token count — expect a long run",
                       ratio)

    refs, payloads, elapsed = await _run(graph, VIDEO_EXTENSIONS, request)
    logger.info("av done in %.1fms — %s (%.1f MB)",
                elapsed, refs[0]["filename"], len(payloads[0]) / 2**20)

    return JSONResponse({
        "video_base64": base64.b64encode(payloads[0]).decode(),
        "filename": refs[0]["filename"],
        "modality": "av", "mode": mode, "has_audio": True,
        "fps": req.fps, "num_frames": req.num_frames,
        "width": req.width, "height": req.height,
        "seed": seed, "steps": req.steps,
        "processing_time_ms": elapsed,
    })


@app.post("/generate/music")
async def generate_music(req: MusicRequest, request: Request) -> JSONResponse:
    """ACE-Step 1.5 XL Turbo — text to music."""
    _require("music")

    if req.duration <= 0:
        raise HTTPException(status_code=400, detail="duration must be positive")

    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
    graph, _ = _build_music_workflow(req, seed)

    logger.info("music | %.0fs | %d bpm %s %s | steps=%d cfg=%.1f | seed=%d | agent=%s",
                req.duration, req.bpm, req.key_scale, req.language,
                req.steps, req.cfg_scale, seed, req.agent_id)

    refs, payloads, elapsed = await _run(graph, AUDIO_EXTENSIONS, request)
    logger.info("music done in %.1fms — %s (%.1f MB)",
                elapsed, refs[0]["filename"], len(payloads[0]) / 2**20)

    return JSONResponse({
        "audio_base64": base64.b64encode(payloads[0]).decode(),
        "filename": refs[0]["filename"],
        "modality": "music", "format": "mp3", "mp3_quality": req.mp3_quality,
        "duration": req.duration, "bpm": req.bpm, "key_scale": req.key_scale,
        "has_lyrics": bool(req.lyrics),
        "seed": seed, "steps": req.steps,
        "processing_time_ms": elapsed,
    })


@app.post("/generate/image")
async def generate_image(req: ImageRequest, request: Request) -> JSONResponse:
    """Qwen-Image-2.1 — text to image, optionally with reference images."""
    _require("image")

    if len(req.ref_images) > 10:
        raise HTTPException(status_code=400, detail="Qwen-Image-2.1 accepts at most 10 reference images")

    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
    uploaded = [await _upload_asset(a, "image") for a in req.ref_images]
    graph, _ = _build_image_workflow(req, seed, uploaded)

    steps = req.steps if req.steps is not None else IMAGE_STEPS
    logger.info("image | %dx%d x%d | steps=%d | refs=%d | seed=%d | agent=%s",
                req.width, req.height, req.batch_size,
                steps, len(uploaded), seed, req.agent_id)

    refs, payloads, elapsed = await _run(graph, IMAGE_EXTENSIONS, request)
    logger.info("image done in %.1fms — %d file(s), first %s",
                elapsed, len(payloads), refs[0]["filename"])

    return JSONResponse({
        "images": [base64.b64encode(p).decode() for p in payloads],
        "filenames": [r["filename"] for r in refs],
        "modality": "image", "model": "qwen-image-2.1",
        "width": req.width, "height": req.height, "batch_size": req.batch_size,
        "reference_images": len(uploaded),
        "seed": seed, "steps": steps,
        "processing_time_ms": elapsed,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
