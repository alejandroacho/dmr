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
