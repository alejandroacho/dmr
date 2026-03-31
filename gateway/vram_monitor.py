"""
VRAM monitoring module.
Queries GPU state via NVML (pynvml) and generates GPU status reports.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import pynvml

from gateway.schemas import GPUInfo, VRAMReport
from gateway.config import VRAM_POLL_INTERVAL_S, VRAM_SAFETY_MARGIN_MB, SYSTEM_RAM_GB

logger = logging.getLogger("gateway.vram_monitor")

# NVML is initialised once at module level; the library is safe to call
# from any thread after init.
_nvml_ok = False
try:
    pynvml.nvmlInit()
    _nvml_ok = True
    logger.info("NVML initialised — driver %s", pynvml.nvmlSystemGetDriverVersion())
except pynvml.NVMLError as exc:
    logger.warning("NVML init failed (%s). GPU monitoring will use simulated data.", exc)


class VRAMMonitor:
    """
    Async VRAM monitor via nvidia-smi.
    Maintains a periodically updated snapshot.
    """

    def __init__(self, poll_interval: float = VRAM_POLL_INTERVAL_S):
        self._poll_interval = poll_interval
        self._latest_report: VRAMReport = VRAMReport()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def latest(self) -> VRAMReport:
        """Latest available VRAM report."""
        return self._latest_report

    # ─────────────────── Lifecycle ─────────────────────

    async def start(self) -> None:
        """Starts the nvidia-smi polling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("VRAMMonitor started (interval=%.1fs)", self._poll_interval)

    async def stop(self) -> None:
        """Stops the polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("VRAMMonitor stopped.")

    # ─────────────────── Polling ───────────────────────

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                report = await self.query_gpus()
                self._latest_report = report
            except Exception as exc:
                logger.error("Error querying nvidia-smi: %s", exc)
                self._latest_report.healthy = False
            await asyncio.sleep(self._poll_interval)

    async def query_gpus(self) -> VRAMReport:
        """Queries nvidia-smi and returns a VRAMReport."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._query_gpus_sync
        )

    @staticmethod
    def _query_gpus_sync() -> VRAMReport:
        """Synchronous GPU query via NVML (executed in thread pool)."""
        if not _nvml_ok:
            return VRAMMonitor._mock_report()

        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as exc:
            logger.error("NVML device count failed: %s", exc)
            return VRAMReport(healthy=False)

        gpus: list[GPUInfo] = []

        for idx in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name = pynvml.nvmlDeviceGetName(handle)

                # --- Memory ---
                cap_mb = SYSTEM_RAM_GB * 1024  # Configured addressable GPU memory
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_mb = mem_info.total // (1024 * 1024)
                    used_mb = mem_info.used // (1024 * 1024)
                    free_mb = mem_info.free // (1024 * 1024)

                    # On unified memory systems (GB10/ATS) NVML reports
                    # the full system RAM as GPU memory.  Cap to the
                    # configured addressable pool so VRAM budgets are
                    # meaningful.
                    if total_mb > cap_mb:
                        total_mb = cap_mb
                        free_mb = max(total_mb - used_mb, 0)
                except pynvml.NVMLError:
                    # NVML cannot report memory at all — fall back to
                    # process accounting.
                    total_mb = cap_mb
                    used_mb = VRAMMonitor._sum_process_memory_nvml(handle)
                    free_mb = max(total_mb - used_mb, 0)
                    logger.debug(
                        "Unified memory (%s): total=%d MiB, "
                        "process_used=%d MiB, free=%d MiB",
                        name, total_mb, used_mb, free_mb,
                    )

                # --- Temperature ---
                try:
                    temp_c = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU,
                    )
                except pynvml.NVMLError:
                    temp_c = 0

                # --- Utilization ---
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except pynvml.NVMLError:
                    gpu_util = 0

                gpus.append(GPUInfo(
                    index=idx,
                    name=name,
                    vram_total_mb=total_mb,
                    vram_used_mb=used_mb,
                    vram_free_mb=free_mb,
                    temperature_c=temp_c,
                    utilization_pct=gpu_util,
                ))
            except pynvml.NVMLError as exc:
                logger.error("NVML error for GPU %d: %s", idx, exc)

        total_vram = sum(g.vram_total_mb for g in gpus)
        total_used = sum(g.vram_used_mb for g in gpus)
        total_free = sum(g.vram_free_mb for g in gpus)

        return VRAMReport(
            gpus=gpus,
            total_vram_mb=total_vram,
            total_used_mb=total_used,
            total_free_mb=total_free,
            healthy=True,
        )

    @staticmethod
    def _sum_process_memory_nvml(handle) -> int:
        """Sum memory used by all processes on a device (unified memory fallback)."""
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            return sum(p.usedGpuMemory // (1024 * 1024) for p in procs if p.usedGpuMemory)
        except pynvml.NVMLError:
            return 0

    @staticmethod
    def _mock_report() -> VRAMReport:
        """Simulated report for development without GPU."""
        mock_gpus = []
        for i in range(2):
            mock_gpus.append(GPUInfo(
                index=i,
                name=f"NVIDIA Blackwell B200 (simulated) #{i}",
                vram_total_mb=131072,    # 128 GB
                vram_used_mb=0,
                vram_free_mb=131072,
                temperature_c=35,
                utilization_pct=0,
            ))
        return VRAMReport(
            gpus=mock_gpus,
            total_vram_mb=262144,
            total_used_mb=0,
            total_free_mb=262144,
            healthy=True,
        )

    # ─────────────── Utilities ────────────────────────

    def has_enough_vram(
        self,
        required_mb: int,
        report: VRAMReport | None = None,
    ) -> bool:
        """Checks if there is enough free VRAM (with safety margin).
        
        Args:
            required_mb: Minimum VRAM needed.
            report: Optional fresh VRAMReport. Uses cached if not provided.
        """
        source = report or self._latest_report
        return source.total_free_mb >= (required_mb + VRAM_SAFETY_MARGIN_MB)

    def get_free_vram_mb(self) -> int:
        return self._latest_report.total_free_mb


