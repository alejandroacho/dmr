"""
Pydantic data models for the Smart Gateway.
Defines request/response structures for the 9 external agents.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ──────────────────────────── Enums ────────────────────────────

class ProfileMode(str, Enum):
    """Memory load profiles."""
    FOCUS = "focus"          # Engineering Mode (GPT-OSS 120B + Qwen3 Coder Next 80B)
    CREATIVE = "creative"    # Creative Mode   (Qwen3 Coder + FLUX.2/LTX-Video)


class ContainerState(str, Enum):
    """Container lifecycle states."""
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    PAUSED = "paused"
    ERROR = "error"


class MediaType(str, Enum):
    """Multimedia generation types."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class SwapStrategy(str, Enum):
    """Model swap strategy."""
    PAUSE_UNPAUSE = "pause_unpause"   # Fast, requires +512GB RAM
    STOP_START = "stop_start"         # Deep VRAM cleanup


# ──────────────────────────── VRAM ─────────────────────────────

class GPUInfo(BaseModel):
    """Individual GPU information."""
    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    temperature_c: int
    utilization_pct: int


class VRAMReport(BaseModel):
    """Complete VRAM status report for the cluster."""
    timestamp: float = Field(default_factory=time.time)
    gpus: list[GPUInfo] = []
    total_vram_mb: int = 0
    total_used_mb: int = 0
    total_free_mb: int = 0
    healthy: bool = True


# ──────────────────────────── Modelos ──────────────────────────

class ModelSlot(BaseModel):
    """Describes a model slot loaded into VRAM."""
    name: str
    container_name: str
    vram_allocated_mb: int
    state: ContainerState = ContainerState.STOPPED
    port: int
    quantization: str = "Q8_0"
    tensor_parallel_size: int = 2
    max_model_len: int = 128000
    kv_cache_dtype: str = "fp8"


# ──────────────────────── Requests ─────────────────────────────

class AgentRequest(BaseModel):
    """
    Unified request structure for the 9 agents.
    Compatible with the OpenAI Chat Completions API.
    """
    model: str = "auto"
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Messages in OpenAI format [{role, content}]",
    )
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    # Extended fields for multimedia
    media_type: MediaType = MediaType.TEXT
    media_params: dict[str, Any] | None = Field(
        default=None,
        description="Extra parameters for visual generation (resolution, fps, etc.)",
    )
    # Agent metadata
    agent_id: str | None = Field(
        default=None,
        description="Unique agent identifier (1-9)",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Request priority (1=low, 10=critical)",
    )


class ImageGenerationRequest(BaseModel):
    """Specific image generation request (FLUX.2 Pro)."""
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg_scale: float = 7.5
    seed: int | None = None
    agent_id: str | None = None


class VideoGenerationRequest(BaseModel):
    """Specific video generation request (LTX-Video 2)."""
    prompt: str
    negative_prompt: str = ""
    width: int = 768
    height: int = 512
    num_frames: int = 81
    fps: int = 24
    steps: int = 50
    cfg_scale: float = 7.0
    seed: int | None = None
    agent_id: str | None = None


# ──────────────────────── Responses ────────────────────────────

class GatewayResponse(BaseModel):
    """Standard Gateway response."""
    success: bool
    data: Any | None = None
    error: str | None = None
    profile: ProfileMode | None = None
    processing_time_ms: float | None = None
    model_used: str | None = None


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str = "ok"
    version: str = "1.0.0"
    active_profile: ProfileMode | None = None
    vram: VRAMReport | None = None
    containers: dict[str, ContainerState] = {}
    uptime_seconds: float = 0.0


class SwapStatusResponse(BaseModel):
    """Current model swap status."""
    swapping: bool = False
    from_profile: ProfileMode | None = None
    to_profile: ProfileMode | None = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    queued_requests: int = 0


class ProfileStatusResponse(BaseModel):
    """Detailed status of models loaded in the active profile."""
    active_profile: ProfileMode
    models: list[ModelSlot] = []
    vram: VRAMReport | None = None
