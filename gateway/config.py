"""
Central configuration for the Smart Gateway.
Defines VRAM profiles, available models, and system parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from gateway.schemas import (
    ContainerState,
    ModelSlot,
    ProfileMode,
    SwapStrategy,
)


# ─────────────────── Environment Variables ──────────────────────

GATEWAY_HOST: str = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT: int = int(os.getenv("GATEWAY_PORT", "8000"))

# Public base URL for generated assets (images, videos).
# Must be reachable from clients like OpenWebUI.
GATEWAY_PUBLIC_URL: str = os.getenv("GATEWAY_PUBLIC_URL", "http://192.168.1.125:8000")

# Local path where model weights reside
MODELS_PATH: str = os.getenv("MODELS_PATH", "/home/alejandroacho/Models")

# Total system RAM; determines swap strategy
SYSTEM_RAM_GB: int = int(os.getenv("SYSTEM_RAM_GB", "512"))

# Maximum timeout for swaps before aborting (seconds)
# GB10 can take ~4 min to load large models (weights + KV cache + warmup)
SWAP_TIMEOUT_S: int = int(os.getenv("SWAP_TIMEOUT_S", "600"))

# nvidia-smi polling interval (seconds)
VRAM_POLL_INTERVAL_S: float = float(os.getenv("VRAM_POLL_INTERVAL_S", "2.0"))

# Safety VRAM margin to always keep free (MB)
VRAM_SAFETY_MARGIN_MB: int = int(os.getenv("VRAM_SAFETY_MARGIN_MB", "4096"))

# ── Media node (node 3) ──
# Dedicated GB10 that serves MiniMax-H3 video+audio and nothing else. It is a
# separate machine with its own 124 GB unified pool, so it never participates in
# this Gateway's VRAM profiles or container swapping.
# MEDIA_NODE_IP is the single place the node's address is configured (see .env);
# MEDIA_NODE_HOST still overrides it for the rare case of addressing the adapter
# and the asset URLs differently. The default is the wired address of the media node.
MEDIA_NODE_IP: str = os.getenv("MEDIA_NODE_IP", "192.168.1.86")
MEDIA_NODE_HOST: str = os.getenv("MEDIA_NODE_HOST", MEDIA_NODE_IP)
MEDIA_NODE_PORT: int = int(os.getenv("MEDIA_NODE_PORT", "8010"))

# Docker socket
DOCKER_SOCKET: str = os.getenv("DOCKER_SOCKET", "unix:///var/run/docker.sock")

# Docker network for inference containers
DOCKER_NETWORK: str = os.getenv("DOCKER_NETWORK", "blackwell_net")

# Retry-After header for 503 during swaps (seconds)
RETRY_AFTER_SECONDS: int = int(os.getenv("RETRY_AFTER_SECONDS", "5"))

# How often to emit retry/wait log lines (seconds); avoids log spam during long swaps
RETRY_LOG_INTERVAL_S: int = int(os.getenv("RETRY_LOG_INTERVAL_S", "10"))

# Maximum queued requests during a swap
MAX_QUEUE_SIZE: int = int(os.getenv("MAX_QUEUE_SIZE", "200"))

# Long Polling mode vs immediate 503
LONG_POLLING_ENABLED: bool = os.getenv("LONG_POLLING_ENABLED", "true").lower() == "true"

# Maximum Long Polling timeout (seconds)
LONG_POLLING_TIMEOUT_S: int = int(os.getenv("LONG_POLLING_TIMEOUT_S", "600"))


# ─────────────────── Trigger Keywords ─────────────────────

# Keywords that trigger a switch to CREATIVE mode
VISUAL_TRIGGER_KEYWORDS: list[str] = [
    "gen_video",
    "render_frame",
    "generate_image",
    "create_image",
    "create_video",
    "render_video",
    "flux",
    "ltx_video",
    "image_generation",
    "video_generation",
    "gen_image",
    "draw",
    "illustrate",
    "render",
    "animate",
    "storyboard",
]

VISUAL_TOOL_NAMES: list[str] = [
    "gen_video",
    "render_frame",
    "generate_image",
    "create_image",
    "create_video",
    "flux_generate",
    "ltx_generate",
    "comfyui_render",
]


# ─────────────────── Model Definitions ───────────────────────

@dataclass
class ModelDefinition:
    """Metadata for a model available in the cluster."""
    name: str
    container_image: str           # Docker image
    container_name: str            # Container name
    vram_required_mb: int
    port: int
    quantization: str = "Q8_0"
    tensor_parallel_size: int = 2
    max_model_len: int = 128000
    kv_cache_dtype: str = "fp8"
    model_path: str = ""           # Path within MODELS_PATH (local)
    hf_model_id: str = ""          # HuggingFace model ID (overrides model_path)
    engine: str = "vllm"           # vllm | comfyui | diffusers
    extra_args: dict[str, Any] = field(default_factory=dict)
    # Prefix injected before model path in the container command.
    # Needed when the image ENTRYPOINT is a generic shell (e.g. entrypoint.sh
    # that does exec "$@") instead of the vllm binary itself.
    cmd_prefix: list[str] = field(default_factory=list)
    # Extra volumes to mount in addition to the model volume.
    # Format: {host_path: {"bind": container_path, "mode": "ro"|"rw"}}
    extra_volumes: dict[str, Any] = field(default_factory=dict)
    # Extra environment variables to inject into the container.
    extra_env: dict[str, str] = field(default_factory=dict)
    # Remote node hostname/IP. Empty means the model runs as a local container
    # reachable by name on DOCKER_NETWORK. When set, the model lives on another
    # machine: the Gateway proxies to it but never manages its lifecycle.
    host: str = ""

    @property
    def is_remote(self) -> bool:
        return bool(self.host)

    @property
    def base_url(self) -> str:
        """Backend base URL — remote host if set, else the container's DNS name."""
        return f"http://{self.host or self.container_name}:{self.port}"

    def to_slot(self, state: ContainerState = ContainerState.STOPPED) -> ModelSlot:
        return ModelSlot(
            name=self.name,
            container_name=self.container_name,
            vram_allocated_mb=self.vram_required_mb,
            state=state,
            port=self.port,
            quantization=self.quantization,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            kv_cache_dtype=self.kv_cache_dtype,
        )


# ────── Model Catalog ──────

GPT_OSS_120B = ModelDefinition(
    name="gpt-oss-120b",
    # Custom image built from github.com/christopherowen/spark-vllm-mxfp4-docker
    # Includes CUTLASS MXFP4 kernels compiled for sm_121 (GB10 Blackwell).
    # Build: docker build -t vllm-mxfp4-spark .
    container_image="vllm-mxfp4-spark:latest",
    container_name="vllm-gpt-oss-120b",
    vram_required_mb=84_000,        # 70% of ~120 GB = ~84 GB (mxfp4 weights + fp8 KV)
    port=8001,
    quantization="mxfp4",
    tensor_parallel_size=1,
    max_model_len=131072,           # Full 128K context — fits at 70% utilization
    kv_cache_dtype="fp8",
    model_path="gpt-oss-120b-q8",
    engine="vllm",
    # The custom image ENTRYPOINT is entrypoint.sh (exec "$@"), not the vllm binary,
    # so we must prefix the command with "vllm serve".
    cmd_prefix=["vllm", "serve"],
    extra_args={
        "--mxfp4-backend": "CUTLASS",
        "--mxfp4-layers": "moe,qkv,o,lm_head",
        "--attention-backend": "FLASHINFER",
        "--gpu-memory-utilization": "0.70",
        "--max-num-seqs": "10",
        "--max-num-batched-tokens": "8192",
        "--load-format": "fastsafetensors",
        "--enforce-eager": True,          # Disable CUDA graphs — avoids cudaErrorIllegalAddress with MXFP4 CUTLASS on SM121
    },
    # Provide o200k_harmony.tiktoken (copy of o200k_base) so the openai_harmony
    # Rust binary can load the harmony encoding without internet access.
    # The Rust code extends o200k_base with GPT-OSS harmony special tokens internally.
    extra_volumes={
        "/home/alejandroacho/Server/tiktoken_encodings": {
            "bind": "/tiktoken_enc",
            "mode": "ro",
        }
    },
    extra_env={
        "TIKTOKEN_ENCODINGS_BASE": "/tiktoken_enc",
    },
)

QWEN3_CODER_NEXT_80B = ModelDefinition(
    name="qwen3-coder-next-80b",
    container_image="blackwell-vllm:latest",
    container_name="vllm-qwen3-coder-next-80b",
    vram_required_mb=95_000,        # FP8 ~90 GB weights + KV cache
    port=8002,
    quantization="auto",            # FP8 auto-detected from model config
    tensor_parallel_size=1,
    max_model_len=131072,           # 128K context
    kv_cache_dtype="fp8",
    hf_model_id="Qwen/Qwen3-Coder-Next-FP8",
    engine="vllm",
    extra_args={
        "--gpu-memory-utilization": "0.85",
        "--attention-backend": "flashinfer",
        "--enable-auto-tool-choice": True,
        "--tool-call-parser": "qwen3_coder",
        "--enforce-eager": True,
    },
)

QWEN3_CODER_BASE = ModelDefinition(
    name="qwen3-coder",
    container_image="blackwell-vllm:latest",
    container_name="vllm-qwen3-coder",
    vram_required_mb=35_000,        # ~35 GB: 30B-A3B MoE at Q8 (3B active params)
    port=8003,
    quantization="auto",            # Auto-detect from config.json
    tensor_parallel_size=1,
    max_model_len=32768,
    kv_cache_dtype="fp8",
    model_path="qwen3-coder-q8",
    engine="vllm",
    extra_args={"--gpu-memory-utilization": "0.92"},
)

FLUX2_PRO = ModelDefinition(
    name="flux2-pro",
    container_image="comfyui-flux:latest",
    container_name="comfyui-flux2-pro",
    vram_required_mb=42_000,        # FP16 ~ 42 GB
    port=8004,
    quantization="FP16",
    tensor_parallel_size=1,
    max_model_len=0,
    kv_cache_dtype="none",
    model_path="flux2-pro-fp16",
    engine="comfyui",
)

LTX_VIDEO_2 = ModelDefinition(
    name="ltx-video-2",
    container_image="ltx-video:latest",
    container_name="diffusers-ltx-video-2",
    vram_required_mb=42_000,        # Q8 ~ 42 GB
    port=8005,
    quantization="Q8",
    tensor_parallel_size=1,
    max_model_len=0,
    kv_cache_dtype="none",
    model_path="ltx-video-2-q8",
    engine="diffusers",
)

# ────── Media node models (node 3, remote) ──────
# MiniMax-H3: omni-modal DiT generating video with native stereo audio, up to 2K
# and ~15s. Both task checkpoints share the Qwen3-VL-32B conditioning encoder
# and both VAEs, so all of it (~63 GB) stays resident on node 3's GB10 — no
# swapping, and no interaction with this node's text profiles.
#
# fl2va  — t2va and first/last-frame-to-video+audio
# ref2va — omni-reference (9 images, 3 videos, 3 video soundtracks, 3 audios)
#
# Both entries point at the same adapter port: node 3 picks the checkpoint from
# the request. They are deliberately absent from ALL_MODELS and PROFILES —
# those drive local container lifecycle (including force-removal of "orphans").

MINIMAX_H3_FL2VA = ModelDefinition(
    name="minimax-h3-fl2va",
    # One image and one container serve all three families (see Dockerfile.media),
    # so these match ACE-Step's and Qwen Image's. Purely informational — `host` is
    # set, so this model is proxied to and never orchestrated from here.
    container_image="media-node:latest",
    container_name="media-node",
    vram_required_mb=42_470,        # fl2va INT8 convrot + NVFP4 encoder + both VAEs
    port=MEDIA_NODE_PORT,
    quantization="int8_convrot",
    tensor_parallel_size=1,
    max_model_len=0,
    kv_cache_dtype="none",
    model_path="minimax-h3",
    engine="comfyui",
    host=MEDIA_NODE_HOST,
)

MINIMAX_H3_REF2VA = ModelDefinition(
    name="minimax-h3-ref2va",
    container_image="media-node:latest",
    container_name="media-node",
    vram_required_mb=20_970,        # ref2va INT8 convrot (encoder + VAEs already loaded)
    port=MEDIA_NODE_PORT,
    quantization="int8_convrot",
    tensor_parallel_size=1,
    max_model_len=0,
    kv_cache_dtype="none",
    model_path="minimax-h3",
    engine="comfyui",
    host=MEDIA_NODE_HOST,
)

ACE_STEP_15_XL_TURBO = ModelDefinition(
    name="ace-step-1.5-xl-turbo",
    container_image="media-node:latest",
    container_name="media-node",
    vram_required_mb=19_900,        # DiT bf16 + both Qwen encoders + VAE
    port=MEDIA_NODE_PORT,
    quantization="bf16",
    tensor_parallel_size=1,
    max_model_len=0,
    kv_cache_dtype="none",
    model_path="ace-step-1.5",
    engine="comfyui",
    host=MEDIA_NODE_HOST,
)

QWEN_IMAGE_21 = ModelDefinition(
    name="qwen-image-2.1",
    container_image="media-node:latest",
    container_name="media-node",
    vram_required_mb=20_000,        # estimated INT8 model + encoder + VAE; runtime varies
    port=MEDIA_NODE_PORT,
    quantization="int8_convrot",
    tensor_parallel_size=1,
    max_model_len=0,
    kv_cache_dtype="none",
    model_path="qwen-image-2.1",
    engine="comfyui",
    host=MEDIA_NODE_HOST,
)

# Models served by the media node — exposed via /v1/models and proxied to,
# never orchestrated. One ComfyUI holds all of them and evicts as needed.
REMOTE_MEDIA_MODELS: list[ModelDefinition] = [
    MINIMAX_H3_FL2VA,
    MINIMAX_H3_REF2VA,
    ACE_STEP_15_XL_TURBO,
    QWEN_IMAGE_21,
]

# Which Gateway endpoint drives each model.
MEDIA_MODEL_ENDPOINTS: dict[str, str] = {
    MINIMAX_H3_FL2VA.name: "/v1/av/generate",
    MINIMAX_H3_REF2VA.name: "/v1/av/generate",
    ACE_STEP_15_XL_TURBO.name: "/v1/audio/music",
    QWEN_IMAGE_21.name: "/v1/images/generate",
}


QWEN25_CODER_7B = ModelDefinition(
    name="qwen2.5-coder-7b",
    container_image="blackwell-vllm:latest",
    container_name="vllm-qwen25-coder-7b",
    vram_required_mb=8_000,             # ~8 GB: 7B params at FP8 + KV cache
    port=8007,
    quantization="fp8",                 # On-the-fly FP8 quantization (model is BF16 natively)
    tensor_parallel_size=1,
    max_model_len=32768,
    kv_cache_dtype="fp8",
    hf_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    engine="vllm",
    extra_args={
        "--gpu-memory-utilization": "0.30",
        "--enforce-eager": True,            # GB10 CUDA graph compat
    },
)

QWEN3_5_4B = ModelDefinition(
    name="qwen3.5-4b",
    container_image="blackwell-vllm:latest",
    container_name="vllm-qwen3-5-4b",
    vram_required_mb=4_000,         # ~4 GB: 4B params at FP8
    port=8006,
    quantization="auto",
    tensor_parallel_size=1,
    max_model_len=32768,
    kv_cache_dtype="fp8",
    hf_model_id="Qwen/Qwen3.5-4B",
    engine="vllm",
    extra_args={
        "--gpu-memory-utilization": "0.15",
        "--enforce-eager": True,            # Disable CUDA graphs — avoids cudagraph mode mismatch with Mamba hybrid arch on GB10
        "--reasoning-parser": "qwen3",      # Parse <think>...</think> into reasoning_content field
    },
)


# ────── Load Profiles ──────

@dataclass
class VRAMProfile:
    """VRAM distribution profile across models."""
    mode: ProfileMode
    description: str
    primary_models: list[ModelDefinition]
    secondary_models: list[ModelDefinition] = field(default_factory=list)
    labels: dict[str, ModelDefinition] = field(default_factory=dict)
    total_vram_required_mb: int = 0

    def __post_init__(self):
        all_models = self.primary_models + self.secondary_models
        self.total_vram_required_mb = sum(m.vram_required_mb for m in all_models)


PROFILE_FOCUS = VRAMProfile(
    mode=ProfileMode.FOCUS,
    description="Reasoning Mode: GPT-OSS 120B + Qwen2.5-Coder-7B (~92 GB)",
    primary_models=[GPT_OSS_120B],
    secondary_models=[QWEN25_CODER_7B],
    labels={"chat": GPT_OSS_120B, "code": QWEN25_CODER_7B},
)

PROFILE_FOCUS_CODE = VRAMProfile(
    mode=ProfileMode.FOCUS,
    description="Code Mode: Qwen3-Coder-Next 80B MoE FP8 + Qwen3.5-4B chat (~99 GB)",
    primary_models=[QWEN3_CODER_NEXT_80B],
    secondary_models=[QWEN3_5_4B],
    labels={"code": QWEN3_CODER_NEXT_80B, "chat": QWEN3_5_4B},
)

PROFILE_CREATIVE_IMAGE = VRAMProfile(
    mode=ProfileMode.CREATIVE,
    description="Creative Image Mode: FLUX.2 Pro + Qwen3.5-4B chat (~46 GB)",
    primary_models=[FLUX2_PRO],
    secondary_models=[QWEN3_5_4B],
    labels={"image": FLUX2_PRO, "chat": QWEN3_5_4B},
)

PROFILE_CREATIVE_VIDEO = VRAMProfile(
    mode=ProfileMode.CREATIVE,
    description="Creative Video Mode: Qwen3 Coder 30B + LTX-Video 2 (~77 GB)",
    primary_models=[QWEN3_CODER_BASE],
    secondary_models=[LTX_VIDEO_2],
    labels={"video": LTX_VIDEO_2, "chat": QWEN3_CODER_BASE},
)

PROFILES: dict[str, VRAMProfile] = {
    "focus": PROFILE_FOCUS,
    "focus_code": PROFILE_FOCUS_CODE,
    "creative_image": PROFILE_CREATIVE_IMAGE,
    "creative_video": PROFILE_CREATIVE_VIDEO,
}

# Flat list of every model in the catalog (used for orphan cleanup)
ALL_MODELS: list[ModelDefinition] = [
    GPT_OSS_120B,
    QWEN3_CODER_NEXT_80B,
    QWEN3_CODER_BASE,
    QWEN25_CODER_7B,
    FLUX2_PRO,
    LTX_VIDEO_2,
    QWEN3_5_4B,
]


def get_swap_strategy() -> SwapStrategy:
    """Determines swap strategy based on available RAM."""
    if SYSTEM_RAM_GB >= 512:
        return SwapStrategy.PAUSE_UNPAUSE
    return SwapStrategy.STOP_START
