"""
VRAM monitoring module.
Queries nvidia-smi and generates GPU status reports.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import xml.etree.ElementTree as ET
from typing import Optional

from gateway.schemas import GPUInfo, VRAMReport
from gateway.config import VRAM_POLL_INTERVAL_S, VRAM_SAFETY_MARGIN_MB, SYSTEM_RAM_GB

logger = logging.getLogger("gateway.vram_monitor")


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
        """Synchronous version of the query (executed in thread pool)."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-q", "-x"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.error("nvidia-smi returned code %d", result.returncode)
                return VRAMReport(healthy=False)

            return VRAMMonitor._parse_nvidia_xml(result.stdout)
        except FileNotFoundError:
            logger.warning("nvidia-smi not found, returning simulated report.")
            return VRAMMonitor._mock_report()
        except subprocess.TimeoutExpired:
            logger.error("nvidia-smi timeout.")
            return VRAMReport(healthy=False)

    @staticmethod
    def _parse_nvidia_xml(xml_str: str) -> VRAMReport:
        """Parses the XML output from nvidia-smi.

        Handles two memory architectures:
        - **Dedicated VRAM** (normal GPUs): reads ``fb_memory_usage``.
        - **Unified memory** (GB10 / Grace-Blackwell ATS mode):
          ``fb_memory_usage`` values are "N/A".  Falls back to summing
          per-process ``used_memory`` from the ``<processes>`` section
          and using ``SYSTEM_RAM_GB`` as the total addressable pool.
        """
        root = ET.fromstring(xml_str)
        gpus: list[GPUInfo] = []

        for idx, gpu_elem in enumerate(root.findall("gpu")):
            name = gpu_elem.findtext("product_name", "Unknown GPU")

            # --- Memory: try fb_memory_usage first ----
            fb_mem = gpu_elem.find("fb_memory_usage")
            total_str = fb_mem.findtext("total", "N/A") if fb_mem is not None else "N/A"

            unified = (total_str.strip() == "N/A")

            if not unified:
                # Normal GPU with dedicated VRAM
                used_str = fb_mem.findtext("used", "0 MiB") if fb_mem is not None else "0 MiB"
                free_str = fb_mem.findtext("free", "0 MiB") if fb_mem is not None else "0 MiB"
                total_mb = _parse_mib(total_str)
                used_mb = _parse_mib(used_str)
                free_mb = _parse_mib(free_str)
            else:
                # Unified memory (GB10/ATS): sum process memory
                total_mb = SYSTEM_RAM_GB * 1024  # Config-driven
                used_mb = VRAMMonitor._sum_process_memory(gpu_elem)
                free_mb = max(total_mb - used_mb, 0)
                logger.debug(
                    "Unified memory detected (%s). "
                    "total=%d MiB, process_used=%d MiB, free=%d MiB",
                    name, total_mb, used_mb, free_mb,
                )

            # --- Temperature ---
            temp_elem = gpu_elem.find("temperature")
            temp_str = temp_elem.findtext("gpu_temp", "0 C") if temp_elem is not None else "0 C"
            temp_c = int(temp_str.replace(" C", "").strip()) if temp_str else 0

            # --- Utilization ---
            util_elem = gpu_elem.find("utilization")
            gpu_util_str = util_elem.findtext("gpu_util", "0 %") if util_elem is not None else "0 %"
            gpu_util = int(gpu_util_str.replace(" %", "").strip()) if gpu_util_str else 0

            gpus.append(GPUInfo(
                index=idx,
                name=name,
                vram_total_mb=total_mb,
                vram_used_mb=used_mb,
                vram_free_mb=free_mb,
                temperature_c=temp_c,
                utilization_pct=gpu_util,
            ))

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
    def _sum_process_memory(gpu_elem) -> int:
        """Sum ``used_memory`` of all GPU processes (unified memory fallback)."""
        total = 0
        procs = gpu_elem.find("processes")
        if procs is None:
            return 0
        for proc in procs.findall("process_info"):
            mem_str = proc.findtext("used_memory", "0 MiB")
            total += _parse_mib(mem_str)
        return total

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


def _parse_mib(value: str) -> int:
    """Converts '12345 MiB' → 12345."""
    try:
        return int(value.replace("MiB", "").replace("mib", "").strip())
    except (ValueError, AttributeError):
        return 0
