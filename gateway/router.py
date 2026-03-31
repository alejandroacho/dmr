"""
Smart Router & Trigger Detection module.
Intercepts agent requests and determines which VRAM profile
and which backend model should serve them.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from gateway.config import (
    VISUAL_TRIGGER_KEYWORDS,
    VISUAL_TOOL_NAMES,
    PROFILES,
    PROFILE_FOCUS,
    PROFILE_FOCUS_CODE,
    PROFILE_CREATIVE_IMAGE,
    PROFILE_CREATIVE_VIDEO,
    VRAMProfile,
    ModelDefinition,
    GPT_OSS_120B,
    QWEN3_CODER_NEXT_80B,
    QWEN3_CODER_BASE,
    QWEN25_CODER_7B,
    FLUX2_PRO,
    LTX_VIDEO_2,
    QWEN3_5_4B,
)
from gateway.schemas import AgentRequest, MediaType, ProfileMode

logger = logging.getLogger("gateway.router")

# Compiled regex for fast visual keyword detection
_VISUAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in VISUAL_TRIGGER_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


class SmartRouter:
    """
    Analyzes incoming requests and determines:
    1. Whether a profile switch is required (Focus ↔ Creative).
    2. Which specific backend model should serve the request.
    3. What type of media is being requested (TEXT / IMAGE / VIDEO).
    """

    def route(
        self,
        request: AgentRequest,
        active_profile: VRAMProfile | None = None,
    ) -> RoutingDecision:
        """
        Main entry point.
        Returns a routing decision with profile, model, and media type.

        If *active_profile* is provided, label aliases (e.g. "chat", "code")
        are resolved against it first so that requests using a label stay
        in the current profile whenever possible.
        """
        requested = (request.model or "").strip()

        # Try label resolution before anything else
        if requested and requested not in ("auto", ""):
            resolved = self._resolve_label(requested, active_profile)
            if resolved:
                profile, model = resolved
                # Infer media type from the resolved model's engine
                if model.engine == "comfyui":
                    media_type = MediaType.IMAGE
                elif model.engine == "diffusers":
                    media_type = MediaType.VIDEO
                else:
                    media_type = self._detect_media_type(request)

                logger.debug(
                    "Routing (label '%s'): agent=%s media=%s profile=%s model=%s",
                    requested, request.agent_id, media_type.value,
                    profile.mode.value, model.name,
                )
                return RoutingDecision(
                    media_type=media_type,
                    profile=profile,
                    target_model=model,
                )

        # Standard routing (no label match)
        media_type = self._detect_media_type(request)
        profile = self._select_profile(request, media_type, active_profile)
        model = self._select_model(request, media_type, profile)

        decision = RoutingDecision(
            media_type=media_type,
            profile=profile,
            target_model=model,
            requires_swap=False,  # Determined by the Gateway comparing with active profile
        )

        logger.debug(
            "Routing: agent=%s media=%s profile=%s model=%s",
            request.agent_id,
            media_type.value,
            profile.mode.value,
            model.name,
        )

        return decision

    # ──────────── Media Type Detection ──────────

    def _detect_media_type(self, request: AgentRequest) -> MediaType:
        """
        Determines the requested media type by inspecting:
        1. Explicit media_type field.
        2. tool_choice field.
        3. Prompt content (keywords).
        """
        # 1. If the agent declares it explicitly
        if request.media_type != MediaType.TEXT:
            return request.media_type

        # 2. Model name — image/video models identified by name
        requested_model = (request.model or "").lower()
        if any(k in requested_model for k in ("flux", "stable-diffusion", "sdxl")):
            return MediaType.IMAGE
        if any(k in requested_model for k in ("ltx", "video-gen", "animate")):
            return MediaType.VIDEO

        # 3. Inspect tool_choice
        tool_name = self._extract_tool_name(request.tool_choice)
        if tool_name:
            if any(vt in tool_name.lower() for vt in ("video", "animate", "ltx")):
                return MediaType.VIDEO
            if any(vt in tool_name.lower() for vt in ("image", "render", "flux", "draw")):
                return MediaType.IMAGE
            if tool_name.lower() in VISUAL_TOOL_NAMES:
                # Distinguish video vs image by name
                if "video" in tool_name.lower():
                    return MediaType.VIDEO
                return MediaType.IMAGE

        # 3. Inspect prompt content
        prompt_text = self._extract_prompt_text(request)
        if prompt_text:
            if self._contains_video_keywords(prompt_text):
                return MediaType.VIDEO
            if _VISUAL_PATTERN.search(prompt_text):
                return MediaType.IMAGE

        return MediaType.TEXT

    # ──────────── Profile Selection ──────────────────

    def _select_profile(
        self,
        request: AgentRequest,
        media_type: MediaType,
        active_profile: VRAMProfile | None = None,
    ) -> VRAMProfile:
        """Selects the VRAM profile based on the detected media type and model hint.

        If the active profile already has a model that can serve the request
        (i.e. a "chat" label for text), stay on it to avoid unnecessary swaps.
        """
        if media_type == MediaType.VIDEO:
            return PROFILE_CREATIVE_VIDEO
        if media_type == MediaType.IMAGE:
            return PROFILE_CREATIVE_IMAGE

        # Text: check if user explicitly requests a specific model
        requested = (request.model or "").lower()

        # Explicit GPT-OSS request → reasoning profile
        if "gpt" in requested or "120b" in requested:
            return PROFILE_FOCUS

        # Explicit coding model request → code profile
        if any(k in requested for k in ("qwen", "coder", "next", "80b")):
            return PROFILE_FOCUS_CODE

        # If the active profile has a chat-capable model, stay on it
        if active_profile and "chat" in active_profile.labels:
            return active_profile

        # Default text → GPT-OSS reasoning mode
        return PROFILE_FOCUS

    # ──────────── Model Selection ──────────────────

    def _select_model(
        self,
        request: AgentRequest,
        media_type: MediaType,
        profile: VRAMProfile,
    ) -> ModelDefinition:
        """Selects the specific backend model within the profile.

        Always returns a model that belongs to *profile* so that the
        orchestrator never tries to proxy to a container that isn't part
        of the active profile.
        """
        # Multimedia
        if media_type == MediaType.IMAGE:
            return FLUX2_PRO
        if media_type == MediaType.VIDEO:
            return LTX_VIDEO_2

        # Text: decide based on the requested model or profile
        requested = request.model.lower() if request.model else "auto"

        if requested in ("auto", ""):
            # Use the profile's "chat" label if available, else primary model
            if "chat" in profile.labels:
                return profile.labels["chat"]
            return profile.primary_models[0]

        # Try to match against models *in the selected profile* first.
        all_profile_models = profile.primary_models + profile.secondary_models
        for model in all_profile_models:
            if model.name.lower() in requested or requested in model.name.lower():
                return model

        # Fallback: best-effort match across the global catalog, but only
        # if the result is in the profile.  Otherwise stay in-profile.
        global_match = self._match_global(requested)
        if global_match and global_match in all_profile_models:
            return global_match

        # Last resort: primary model of the profile (never return an
        # out-of-profile model — that causes a phantom swap).
        if "code" in profile.labels:
            return profile.labels["code"]
        if "chat" in profile.labels:
            return profile.labels["chat"]
        return profile.primary_models[0]

    @staticmethod
    def _match_global(requested: str) -> ModelDefinition | None:
        """Best-effort keyword match against the full model catalog."""
        if "120b" in requested or "gpt" in requested:
            return GPT_OSS_120B
        if "next" in requested or "80b" in requested:
            return QWEN3_CODER_NEXT_80B
        if "2.5" in requested and "coder" in requested:
            return QWEN25_CODER_7B
        if "7b" in requested and "coder" in requested:
            return QWEN25_CODER_7B
        if "coder" in requested or "qwen" in requested:
            return QWEN3_CODER_BASE
        return None

    # ──────────── Helpers ──────────────────────────────

    @staticmethod
    def _extract_tool_name(tool_choice: str | dict | None) -> Optional[str]:
        """Extracts the tool name from the tool_choice field."""
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return tool_choice if tool_choice not in ("auto", "none", "required") else None
        if isinstance(tool_choice, dict):
            # OpenAI format: {"type": "function", "function": {"name": "..."}}
            func = tool_choice.get("function", {})
            return func.get("name")
        return None

    @staticmethod
    def _extract_prompt_text(request: AgentRequest) -> str:
        """Extracts the combined text from the agent's messages."""
        parts: list[str] = []
        for msg in request.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        return " ".join(parts)

    @staticmethod
    def _resolve_label(
        name: str,
        active_profile: VRAMProfile | None,
    ) -> tuple[VRAMProfile, ModelDefinition] | None:
        """Resolve a label alias to a (profile, model) pair.

        Priority: active profile first (avoids unnecessary swap),
        then any profile that defines the label.
        """
        if active_profile and name in active_profile.labels:
            return (active_profile, active_profile.labels[name])
        for profile in PROFILES.values():
            if name in profile.labels:
                return (profile, profile.labels[name])
        return None

    @staticmethod
    def _contains_video_keywords(text: str) -> bool:
        """Detects video-specific keywords in the text."""
        video_patterns = [
            r"\bgen_video\b",
            r"\brender_video\b",
            r"\bcreate_video\b",
            r"\bltx[_\-]?video\b",
            r"\banimate\b",
            r"\bvideo[_\-]?generation\b",
        ]
        combined = re.compile("|".join(video_patterns), re.IGNORECASE)
        return bool(combined.search(text))


class RoutingDecision:
    """Routing analysis result."""

    def __init__(
        self,
        media_type: MediaType,
        profile: VRAMProfile,
        target_model: ModelDefinition,
        requires_swap: bool = False,
    ):
        self.media_type = media_type
        self.profile = profile
        self.target_model = target_model
        self.requires_swap = requires_swap

    def __repr__(self) -> str:
        return (
            f"RoutingDecision(media={self.media_type.value}, "
            f"profile={self.profile.mode.value}, "
            f"model={self.target_model.name}, "
            f"swap={self.requires_swap})"
        )
