"""
Tests for gateway.vram_monitor — NVML-based GPU monitoring.
Validates:
  - Standard GPU with dedicated VRAM via NVML
  - Unified memory fallback (process-based counting)
  - Mock report when NVML is unavailable
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from gateway.schemas import GPUInfo, VRAMReport


# ──────────────────────────────────────────────────────
#  NVML mock helpers
# ──────────────────────────────────────────────────────

@dataclass
class FakeMemInfo:
    total: int
    used: int
    free: int


@dataclass
class FakeUtilRates:
    gpu: int
    memory: int


@dataclass
class FakeProcess:
    pid: int
    usedGpuMemory: int


@pytest.fixture(autouse=True)
def _no_host_meminfo(monkeypatch, tmp_path):
    """Default every test to "the host mount is absent".

    Without this the suite would behave differently inside the gateway
    container, where /host/meminfo really exists. Tests that exercise the
    host path point HOST_MEMINFO_PATH at a fixture file of their own.
    """
    monkeypatch.setattr(
        "gateway.vram_monitor.HOST_MEMINFO_PATH", str(tmp_path / "absent")
    )


# Trimmed to the fields the parser reads, with the real box's figures:
# 121 GiB total, ~7 GiB available while vLLM holds its preallocated pool.
MEMINFO_SAMPLE = """\
MemTotal:       127535272 kB
MemFree:         1118800 kB
MemAvailable:    7479680 kB
Buffers:           59176 kB
Cached:         11188896 kB
SwapTotal:      16777212 kB
SwapFree:       12681216 kB
"""


def _write_meminfo(monkeypatch, tmp_path, content: str = MEMINFO_SAMPLE):
    path = tmp_path / "meminfo"
    path.write_text(content)
    monkeypatch.setattr("gateway.vram_monitor.HOST_MEMINFO_PATH", str(path))
    return path


def _make_nvml_mocks(
    *,
    name: str = "NVIDIA RTX 4090",
    total_mb: int = 24564,
    used_mb: int = 8000,
    temp_c: int = 42,
    gpu_util: int = 35,
    mem_raises: bool = False,
    processes: list[FakeProcess] | None = None,
):
    """Build a pynvml mock for a single GPU."""
    import pynvml

    handle = MagicMock()

    mock_pynvml = MagicMock()
    mock_pynvml.NVMLError = pynvml.NVMLError
    mock_pynvml.NVML_TEMPERATURE_GPU = 0
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = handle
    mock_pynvml.nvmlDeviceGetName.return_value = name

    if mem_raises:
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = pynvml.NVMLError(
            pynvml.NVML_ERROR_NOT_SUPPORTED
        )
    else:
        free_mb = total_mb - used_mb
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = FakeMemInfo(
            total=total_mb * 1024 * 1024,
            used=used_mb * 1024 * 1024,
            free=free_mb * 1024 * 1024,
        )

    mock_pynvml.nvmlDeviceGetTemperature.return_value = temp_c
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = FakeUtilRates(
        gpu=gpu_util, memory=0,
    )
    mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = processes or []

    return mock_pynvml


# ──────────────────────────────────────────────────────
#  Dedicated VRAM (normal GPUs)
# ──────────────────────────────────────────────────────

class TestDedicatedVRAM:
    def test_parses_memory_correctly(self):
        mock = _make_nvml_mocks(
            name="NVIDIA RTX 4090",
            total_mb=24564,
            used_mb=8000,
            temp_c=42,
            gpu_util=35,
        )
        with patch.dict("gateway.vram_monitor.__dict__", {"_nvml_ok": True}), \
             patch("gateway.vram_monitor.pynvml", mock):
            from gateway.vram_monitor import VRAMMonitor
            report = VRAMMonitor._query_gpus_sync()

        assert report.healthy is True
        assert len(report.gpus) == 1
        assert report.gpus[0].name == "NVIDIA RTX 4090"
        assert report.total_vram_mb == 24564
        assert report.total_used_mb == 8000
        assert report.total_free_mb == 24564 - 8000

    def test_temp_and_util(self):
        mock = _make_nvml_mocks(temp_c=42, gpu_util=35)
        with patch.dict("gateway.vram_monitor.__dict__", {"_nvml_ok": True}), \
             patch("gateway.vram_monitor.pynvml", mock):
            from gateway.vram_monitor import VRAMMonitor
            report = VRAMMonitor._query_gpus_sync()

        assert report.gpus[0].temperature_c == 42
        assert report.gpus[0].utilization_pct == 35


# ──────────────────────────────────────────────────────
#  Unified memory (GB10 / Grace-Blackwell)
# ──────────────────────────────────────────────────────

class TestUnifiedMemory:
    """When nvmlDeviceGetMemoryInfo raises, fall back to process accounting."""

    def test_detects_unified_memory(self):
        procs = [
            FakeProcess(pid=2553, usedGpuMemory=18 * 1024 * 1024),
            FakeProcess(pid=2651, usedGpuMemory=6 * 1024 * 1024),
            FakeProcess(pid=415592, usedGpuMemory=69564 * 1024 * 1024),
        ]
        mock = _make_nvml_mocks(
            name="NVIDIA GB10", mem_raises=True, processes=procs,
        )
        with patch.dict("gateway.vram_monitor.__dict__", {"_nvml_ok": True}), \
             patch("gateway.vram_monitor.pynvml", mock):
            from gateway.vram_monitor import VRAMMonitor
            report = VRAMMonitor._query_gpus_sync()

        assert report.healthy is True
        assert report.gpus[0].name == "NVIDIA GB10"

    def test_total_equals_system_ram(self):
        from gateway.config import SYSTEM_RAM_GB

        mock = _make_nvml_mocks(name="NVIDIA GB10", mem_raises=True)
        with patch.dict("gateway.vram_monitor.__dict__", {"_nvml_ok": True}), \
             patch("gateway.vram_monitor.pynvml", mock):
            from gateway.vram_monitor import VRAMMonitor
            report = VRAMMonitor._query_gpus_sync()

        assert report.total_vram_mb == SYSTEM_RAM_GB * 1024

    def test_used_equals_sum_of_processes(self):
        procs = [
            FakeProcess(pid=2553, usedGpuMemory=18 * 1024 * 1024),
            FakeProcess(pid=2651, usedGpuMemory=6 * 1024 * 1024),
            FakeProcess(pid=415592, usedGpuMemory=69564 * 1024 * 1024),
        ]
        mock = _make_nvml_mocks(
            name="NVIDIA GB10", mem_raises=True, processes=procs,
        )
        with patch.dict("gateway.vram_monitor.__dict__", {"_nvml_ok": True}), \
             patch("gateway.vram_monitor.pynvml", mock):
            from gateway.vram_monitor import VRAMMonitor
            report = VRAMMonitor._query_gpus_sync()

        assert report.total_used_mb == 69588  # 18 + 6 + 69564

    def test_free_equals_total_minus_used(self):
        from gateway.config import SYSTEM_RAM_GB

        procs = [
            FakeProcess(pid=2553, usedGpuMemory=18 * 1024 * 1024),
            FakeProcess(pid=2651, usedGpuMemory=6 * 1024 * 1024),
            FakeProcess(pid=415592, usedGpuMemory=69564 * 1024 * 1024),
        ]
        mock = _make_nvml_mocks(
            name="NVIDIA GB10", mem_raises=True, processes=procs,
        )
        with patch.dict("gateway.vram_monitor.__dict__", {"_nvml_ok": True}), \
             patch("gateway.vram_monitor.pynvml", mock):
            from gateway.vram_monitor import VRAMMonitor
            report = VRAMMonitor._query_gpus_sync()

        expected_free = (SYSTEM_RAM_GB * 1024) - 69588
        assert report.total_free_mb == expected_free

    def test_empty_processes_means_all_free(self):
        from gateway.config import SYSTEM_RAM_GB

        mock = _make_nvml_mocks(
            name="NVIDIA GB10", mem_raises=True, processes=[],
        )
        with patch.dict("gateway.vram_monitor.__dict__", {"_nvml_ok": True}), \
             patch("gateway.vram_monitor.pynvml", mock):
            from gateway.vram_monitor import VRAMMonitor
            report = VRAMMonitor._query_gpus_sync()

        assert report.total_used_mb == 0
        assert report.total_free_mb == SYSTEM_RAM_GB * 1024


# ──────────────────────────────────────────────────────
#  NVML unavailable → mock report
# ──────────────────────────────────────────────────────

class TestNVMLUnavailable:
    def test_returns_mock_report_when_nvml_not_ok(self):
        with patch.dict("gateway.vram_monitor.__dict__", {"_nvml_ok": False}):
            from gateway.vram_monitor import VRAMMonitor
            report = VRAMMonitor._query_gpus_sync()

        assert report.healthy is True
        assert report.total_free_mb > 0


# ──────────────────────────────────────────────────────
#  Host meminfo (unified memory, the honest source)
# ──────────────────────────────────────────────────────

class TestHostMeminfo:

    def test_parses_total_and_available(self, monkeypatch, tmp_path):
        from gateway.vram_monitor import VRAMMonitor

        _write_meminfo(monkeypatch, tmp_path)
        total_mb, avail_mb = VRAMMonitor._read_host_memory()

        assert total_mb == 127535272 // 1024
        assert avail_mb == 7479680 // 1024

    def test_missing_mount_returns_none(self):
        from gateway.vram_monitor import VRAMMonitor

        # The autouse fixture already points at a nonexistent path.
        assert VRAMMonitor._read_host_memory() is None

    def test_reports_real_usage_where_nvml_reports_nothing(
        self, monkeypatch, tmp_path
    ):
        """The GB10 bug: NVML says "Not Supported" and process accounting is
        blind across PID namespaces, so the gateway used to report 0 MB used
        and a fully free GPU while vLLM held ~101 GiB."""
        from gateway.vram_monitor import VRAMMonitor

        _write_meminfo(monkeypatch, tmp_path)
        mock = _make_nvml_mocks(name="NVIDIA GB10", mem_raises=True, processes=[])
        with patch.dict("gateway.vram_monitor.__dict__", {"_nvml_ok": True}), \
             patch("gateway.vram_monitor.pynvml", mock):
            report = VRAMMonitor._query_gpus_sync()

        assert report.total_used_mb == 124546 - 7304   # ~114 GiB held
        assert report.total_free_mb == 7304            # ~7 GiB really available
        assert report.gpus[0].temperature_c == 42      # NVML still used for these

    def test_host_meminfo_outranks_nvml_numbers(self, monkeypatch, tmp_path):
        """Even when NVML answers, the shared pool is what limits what fits."""
        from gateway.vram_monitor import VRAMMonitor

        _write_meminfo(monkeypatch, tmp_path)
        mock = _make_nvml_mocks(name="NVIDIA GB10", total_mb=131072, used_mb=0)
        with patch.dict("gateway.vram_monitor.__dict__", {"_nvml_ok": True}), \
             patch("gateway.vram_monitor.pynvml", mock):
            report = VRAMMonitor._query_gpus_sync()

        assert report.total_free_mb == 7304

    def test_swap_is_not_counted_as_headroom(self, monkeypatch, tmp_path):
        """12 GiB of free swap must not inflate the budget: serving from swap
        is worse than refusing to load."""
        from gateway.vram_monitor import VRAMMonitor

        _write_meminfo(monkeypatch, tmp_path)
        mock = _make_nvml_mocks(name="NVIDIA GB10", mem_raises=True)
        with patch.dict("gateway.vram_monitor.__dict__", {"_nvml_ok": True}), \
             patch("gateway.vram_monitor.pynvml", mock):
            report = VRAMMonitor._query_gpus_sync()

        assert report.total_free_mb == 7304  # not 7304 + 12681216//1024

    def test_truncated_meminfo_falls_back(self, monkeypatch, tmp_path):
        """A file without MemAvailable must not be trusted half-parsed."""
        from gateway.vram_monitor import VRAMMonitor

        _write_meminfo(monkeypatch, tmp_path, "MemTotal:       127535272 kB\n")
        assert VRAMMonitor._read_host_memory() is None


class TestSystemRamDetection:

    def test_detects_total_ram_from_meminfo(self, tmp_path):
        from gateway.config import _detect_system_ram_gb

        path = tmp_path / "meminfo"
        path.write_text(MEMINFO_SAMPLE)
        with patch("gateway.config.HOST_MEMINFO_PATH", str(path)):
            assert _detect_system_ram_gb() == 121  # not the nominal 128

    def test_falls_back_to_default_when_unreadable(self, tmp_path):
        from gateway.config import _detect_system_ram_gb

        missing = str(tmp_path / "absent")
        with patch("gateway.config.HOST_MEMINFO_PATH", missing), \
             patch("builtins.open", side_effect=OSError("no meminfo")):
            assert _detect_system_ram_gb(default=512) == 512
