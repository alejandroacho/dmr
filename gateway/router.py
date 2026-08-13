"""
Smart Router & Trigger Detection module.
Intercepts agent requests and determines which VRAM profile
and which backend model should serve them.
"""

from __future__ import annotations

import logging

from gateway.config import (
    PROFILES,
    PROFILE_DEEPSEEK,
    VRAMProfile,
    ModelDefinition,
)
from gateway.schemas import AgentRequest, MediaType

logger = logging.getLogger("gateway.router")


class SmartRouter:
    """
    Analyzes incoming requests and determines:
    1. Whether a profile switch is required.
    2. Which specific backend model should serve the request.

    The catalog is text-only, so every decision carries MediaType.TEXT.
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
        # Lowercased: labels in the catalog are lowercase, so "Chat" used to
        # miss and fall through to a full profile lookup.
        requested = (request.model or "").strip().lower()

        # Try label resolution before anything else
        if requested and requested != "auto":
            resolved = self._resolve_label(requested, active_profile)
            if resolved:
                profile, model = resolved
                logger.debug(
                    "Routing (label '%s'): agent=%s profile=%s model=%s",
                    requested, request.agent_id, profile.mode.value, model.name,
                )
                return RoutingDecision(
                    media_type=MediaType.TEXT,
                    profile=profile,
                    target_model=model,
                )

        # Standard routing (no label match)
        media_type = MediaType.TEXT
        profile = self._select_profile(request, active_profile)
        model = self._select_model(request, profile)

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

    # ──────────── Profile Selection ──────────────────

    def _select_profile(
        self,
        request: AgentRequest,
        active_profile: VRAMProfile | None = None,
    ) -> VRAMProfile:
        """Selects the VRAM profile from the requested model hint.

        Profile is derived from the model catalog: whichever profile owns the
        requested model wins. This avoids hardcoded string matching — adding a
        model to a profile in config.py is sufficient.
        """
        requested = (request.model or "").strip().lower()

        if requested and requested not in ("auto", ""):
            profile = self._find_profile_for_model(requested)
            if profile:
                return profile

        # Staying put beats a multi-minute swap when nothing specific was asked.
        if active_profile:
            return active_profile

        return PROFILE_DEEPSEEK

    @staticmethod
    def _find_profile_for_model(requested: str) -> VRAMProfile | None:
        """Return the profile that owns the requested model.

        Tries exact name match first, then checks if the requested string
        is contained in (or contains) a model name — longest model-name
        match wins to avoid 'qwen' matching the wrong profile.
        """
        best_profile: VRAMProfile | None = None
        best_match_len = 0
        for profile in PROFILES.values():
            for model in profile.primary_models + profile.secondary_models:
                name = model.name.lower()
                if name == requested:
                    return profile  # exact match, done
                if requested in name or name in requested:
                    if len(name) > best_match_len:
                        best_match_len = len(name)
                        best_profile = profile
        return best_profile

    # ──────────── Model Selection ──────────────────

    def _select_model(
        self,
        request: AgentRequest,
        profile: VRAMProfile,
    ) -> ModelDefinition:
        """Selects the specific backend model within the profile.

        Always returns a model that belongs to *profile* so that the
        orchestrator never tries to proxy to a container that isn't part
        of the active profile.
        """
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
        """Best-effort match against the full model catalog by name."""
        from gateway.config import ALL_MODELS
        best: ModelDefinition | None = None
        best_len = 0
        for model in ALL_MODELS:
            name = model.name.lower()
            if name == requested:
                return model
            if requested in name or name in requested:
                if len(name) > best_len:
                    best_len = len(name)
                    best = model
        return best

    # ──────────── Helpers ──────────────────────────────

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
