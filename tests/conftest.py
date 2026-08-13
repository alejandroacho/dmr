"""
conftest.py — Shared fixtures for gateway tests.
All Docker / NVML / aiohttp dependencies are mocked.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.backends.base import ExecResult, WorkloadInfo


# ─── Synthetic Docker-managed catalog for tests ──────────────────
#
# The shipped catalog holds only spark_cluster models, whose backend is a
# process exec'd inside a container the Gateway must never create or destroy.
# Tests that exercise the generic container lifecycle (create, pause, unpause,
# remove) need Docker-managed models, so they are defined here and registered
# into the catalog for the duration of the test session.

def _register_test_catalog():
    from gateway.config import ALL_MODELS, PROFILES, ModelDefinition, VRAMProfile
    from gateway.schemas import ProfileMode

    model_a = ModelDefinition(
        name="test-model-a",
        container_image="test-image:latest",
        container_name="vllm-test-a",
        vram_required_mb=40_000,
        port=9101,
        tensor_parallel_size=1,
        hf_model_id="test-org/model-a",
        engine="vllm",
    )
    model_b = ModelDefinition(
        name="test-model-b",
        container_image="test-image:latest",
        container_name="vllm-test-b",
        vram_required_mb=50_000,
        port=9102,
        tensor_parallel_size=1,
        hf_model_id="test-org/model-b",
        engine="vllm",
    )
    model_c = ModelDefinition(
        name="test-model-c",
        container_image="test-image:latest",
        container_name="vllm-test-c",
        vram_required_mb=8_000,
        port=9103,
        tensor_parallel_size=1,
        hf_model_id="test-org/model-c",
        engine="vllm",
    )
    # Nothing in the shipped catalog uses the ray_vllm engine (its models are
    # spark_cluster), so the Ray-head code path needs a model of its own.
    model_ray = ModelDefinition(
        name="test-model-ray",
        container_image="test-image:latest",
        container_name="vllm-test-ray",
        vram_required_mb=60_000,
        port=9104,
        tensor_parallel_size=2,
        hf_model_id="test-org/model-ray",
        engine="ray_vllm",
    )

    profile_a = VRAMProfile(
        mode=ProfileMode.FOCUS,
        description="Test profile A (single Docker-managed model)",
        primary_models=[model_a],
        labels={"chat": model_a},
    )
    profile_b = VRAMProfile(
        mode=ProfileMode.FOCUS,
        description="Test profile B (primary + secondary)",
        primary_models=[model_b],
        secondary_models=[model_c],
        labels={"code": model_b, "chat": model_c},
    )

    profile_ray = VRAMProfile(
        mode=ProfileMode.FOCUS,
        description="Test profile Ray (model served inside the Ray head)",
        primary_models=[model_ray],
        labels={"chat": model_ray},
        skip_vram_check=True,
    )

    ALL_MODELS.extend([model_a, model_b, model_c, model_ray])
    PROFILES["test_a"] = profile_a
    PROFILES["test_b"] = profile_b
    PROFILES["test_ray"] = profile_ray
    return model_a, model_b, model_c, model_ray, profile_a, profile_b, profile_ray


(
    TEST_MODEL_A,
    TEST_MODEL_B,
    TEST_MODEL_C,
    TEST_MODEL_RAY,
    TEST_PROFILE_A,
    TEST_PROFILE_B,
    TEST_PROFILE_RAY,
) = _register_test_catalog()


# ─── Patch docker.DockerClient BEFORE importing gateway modules ──

@pytest.fixture(autouse=True)
def _mock_docker(monkeypatch):
    """Prevents DockerBackend from connecting to a real Docker daemon."""
    mock_client = MagicMock()
    mock_client.containers = MagicMock()
    monkeypatch.setattr(
        "docker.DockerClient",
        lambda *a, **kw: mock_client,
    )
    return mock_client


# ─── Mock OrchestrationBackend ────

@pytest.fixture()
def mock_backend():
    """A MagicMock that satisfies the OrchestrationBackend protocol."""
    backend = MagicMock()
    backend.supports_pause = True

    # Default: get_workload returns None (container not found)
    backend.get_workload = AsyncMock(return_value=None)
    backend.create_and_start = AsyncMock()
    backend.stop_workload = AsyncMock()
    backend.remove_workload = AsyncMock()
    backend.pause_workload = AsyncMock()
    backend.unpause_workload = AsyncMock()
    backend.exec_in_workload = AsyncMock(
        return_value=ExecResult(exit_code=0, output="")
    )
    backend.list_workloads = AsyncMock(return_value=[])
    backend.ensure_network = AsyncMock()
    backend.resolve_hostname = MagicMock(
        side_effect=lambda name, engine: "192.168.200.12" if engine == "ray_vllm" else name
    )

    return backend


# ─── VRAMMonitor that never calls NVML ───

@pytest.fixture()
def mock_vram_monitor():
    from gateway.schemas import VRAMReport, GPUInfo
    from gateway.vram_monitor import VRAMMonitor

    monitor = VRAMMonitor.__new__(VRAMMonitor)
    monitor._poll_interval = 999
    monitor._running = False
    monitor._task = None

    # Default: plenty of VRAM
    monitor._latest_report = VRAMReport(
        gpus=[
            GPUInfo(
                index=0,
                name="Mock GPU",
                vram_total_mb=131072,
                vram_used_mb=0,
                vram_free_mb=131072,
                temperature_c=30,
                utilization_pct=0,
            )
        ],
        total_vram_mb=131072,
        total_used_mb=0,
        total_free_mb=131072,
        healthy=True,
    )

    # query_gpus returns mock data (no subprocess)
    async def _fake_query():
        return monitor._latest_report

    monitor.query_gpus = _fake_query  # type: ignore
    monitor.start = AsyncMock()
    monitor.stop = AsyncMock()
    return monitor


# ─── Tiny helper to set VRAM free amount ──────

def set_vram_free(monitor, free_mb: int):
    """Mutate the mock monitor so it reports `free_mb` of free VRAM."""
    from gateway.schemas import VRAMReport, GPUInfo

    monitor._latest_report = VRAMReport(
        gpus=[
            GPUInfo(
                index=0,
                name="Mock GPU",
                vram_total_mb=131072,
                vram_used_mb=131072 - free_mb,
                vram_free_mb=free_mb,
                temperature_c=30,
                utilization_pct=0,
            )
        ],
        total_vram_mb=131072,
        total_used_mb=131072 - free_mb,
        total_free_mb=free_mb,
        healthy=True,
    )
