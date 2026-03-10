"""
Tests for gateway.vram_monitor — VRAM parsing and unified memory support.
Validates:
  - Standard GPU with fb_memory_usage (dedicated VRAM)
  - GB10 unified memory fallback (process-based counting)
"""

from __future__ import annotations

import pytest

from gateway.vram_monitor import VRAMMonitor, _parse_mib


# ──────────────────────────────────────────────────────
#  nvidia-smi XML stubs
# ──────────────────────────────────────────────────────

DEDICATED_VRAM_XML = """\
<?xml version="1.0" ?>
<nvidia_smi_log>
  <attached_gpus>1</attached_gpus>
  <gpu id="0000:01:00.0">
    <product_name>NVIDIA RTX 4090</product_name>
    <fb_memory_usage>
      <total>24564 MiB</total>
      <reserved>200 MiB</reserved>
      <used>8000 MiB</used>
      <free>16564 MiB</free>
    </fb_memory_usage>
    <bar1_memory_usage>
      <total>256 MiB</total>
      <used>10 MiB</used>
      <free>246 MiB</free>
    </bar1_memory_usage>
    <temperature>
      <gpu_temp>42 C</gpu_temp>
    </temperature>
    <utilization>
      <gpu_util>35 %</gpu_util>
    </utilization>
    <processes>
      <process_info>
        <pid>1234</pid>
        <type>C</type>
        <process_name>python</process_name>
        <used_memory>7500 MiB</used_memory>
      </process_info>
    </processes>
  </gpu>
</nvidia_smi_log>
"""

UNIFIED_MEMORY_XML = """\
<?xml version="1.0" ?>
<nvidia_smi_log>
  <attached_gpus>1</attached_gpus>
  <gpu id="0000:01:00.0">
    <product_name>NVIDIA GB10</product_name>
    <fb_memory_usage>
      <total>N/A</total>
      <reserved>N/A</reserved>
      <used>N/A</used>
      <free>N/A</free>
    </fb_memory_usage>
    <bar1_memory_usage>
      <total>N/A</total>
      <used>N/A</used>
      <free>N/A</free>
    </bar1_memory_usage>
    <temperature>
      <gpu_temp>38 C</gpu_temp>
    </temperature>
    <utilization>
      <gpu_util>2 %</gpu_util>
    </utilization>
    <processes>
      <process_info>
        <pid>2553</pid>
        <type>G</type>
        <process_name>/usr/lib/xorg/Xorg</process_name>
        <used_memory>18 MiB</used_memory>
      </process_info>
      <process_info>
        <pid>2651</pid>
        <type>G</type>
        <process_name>/usr/bin/gnome-shell</process_name>
        <used_memory>6 MiB</used_memory>
      </process_info>
      <process_info>
        <pid>415592</pid>
        <type>C</type>
        <process_name>VLLM::EngineCore</process_name>
        <used_memory>69564 MiB</used_memory>
      </process_info>
    </processes>
  </gpu>
</nvidia_smi_log>
"""

UNIFIED_MEMORY_EMPTY_PROCESSES_XML = """\
<?xml version="1.0" ?>
<nvidia_smi_log>
  <attached_gpus>1</attached_gpus>
  <gpu id="0000:01:00.0">
    <product_name>NVIDIA GB10</product_name>
    <fb_memory_usage>
      <total>N/A</total>
      <reserved>N/A</reserved>
      <used>N/A</used>
      <free>N/A</free>
    </fb_memory_usage>
    <temperature>
      <gpu_temp>30 C</gpu_temp>
    </temperature>
    <utilization>
      <gpu_util>0 %</gpu_util>
    </utilization>
    <processes>
    </processes>
  </gpu>
</nvidia_smi_log>
"""


# ──────────────────────────────────────────────────────
#  _parse_mib helper
# ──────────────────────────────────────────────────────

class TestParseMib:
    def test_normal_value(self):
        assert _parse_mib("12345 MiB") == 12345

    def test_zero(self):
        assert _parse_mib("0 MiB") == 0

    def test_na_returns_zero(self):
        assert _parse_mib("N/A") == 0

    def test_empty_string(self):
        assert _parse_mib("") == 0


# ──────────────────────────────────────────────────────
#  Dedicated VRAM parsing (normal GPUs)
# ──────────────────────────────────────────────────────

class TestDedicatedVRAMParsing:
    def test_parses_fb_memory_correctly(self):
        report = VRAMMonitor._parse_nvidia_xml(DEDICATED_VRAM_XML)

        assert report.healthy is True
        assert len(report.gpus) == 1
        assert report.gpus[0].name == "NVIDIA RTX 4090"
        assert report.total_vram_mb == 24564
        assert report.total_used_mb == 8000
        assert report.total_free_mb == 16564

    def test_temp_and_util(self):
        report = VRAMMonitor._parse_nvidia_xml(DEDICATED_VRAM_XML)

        assert report.gpus[0].temperature_c == 42
        assert report.gpus[0].utilization_pct == 35


# ──────────────────────────────────────────────────────
#  Unified memory parsing (GB10 / Grace-Blackwell)
# ──────────────────────────────────────────────────────

class TestUnifiedMemoryParsing:
    """
    When fb_memory_usage.total is N/A, the parser should fall back
    to summing process memory and using SYSTEM_RAM_GB for total.
    """

    def test_detects_unified_memory(self):
        report = VRAMMonitor._parse_nvidia_xml(UNIFIED_MEMORY_XML)

        assert report.healthy is True
        assert report.gpus[0].name == "NVIDIA GB10"

    def test_total_equals_system_ram(self):
        from gateway.config import SYSTEM_RAM_GB

        report = VRAMMonitor._parse_nvidia_xml(UNIFIED_MEMORY_XML)

        expected_total = SYSTEM_RAM_GB * 1024
        assert report.total_vram_mb == expected_total

    def test_used_equals_sum_of_processes(self):
        report = VRAMMonitor._parse_nvidia_xml(UNIFIED_MEMORY_XML)

        # 18 (Xorg) + 6 (gnome-shell) + 69564 (vLLM) = 69588
        assert report.total_used_mb == 69588

    def test_free_equals_total_minus_used(self):
        from gateway.config import SYSTEM_RAM_GB

        report = VRAMMonitor._parse_nvidia_xml(UNIFIED_MEMORY_XML)

        expected_free = (SYSTEM_RAM_GB * 1024) - 69588
        assert report.total_free_mb == expected_free

    def test_empty_processes_means_all_free(self):
        from gateway.config import SYSTEM_RAM_GB

        report = VRAMMonitor._parse_nvidia_xml(UNIFIED_MEMORY_EMPTY_PROCESSES_XML)

        expected_total = SYSTEM_RAM_GB * 1024
        assert report.total_used_mb == 0
        assert report.total_free_mb == expected_total

    def test_temp_and_util_still_parsed(self):
        report = VRAMMonitor._parse_nvidia_xml(UNIFIED_MEMORY_XML)

        assert report.gpus[0].temperature_c == 38
        assert report.gpus[0].utilization_pct == 2


# ──────────────────────────────────────────────────────
#  _sum_process_memory helper
# ──────────────────────────────────────────────────────

class TestSumProcessMemory:
    def test_sums_all_processes(self):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(UNIFIED_MEMORY_XML)
        gpu_elem = root.find("gpu")
        total = VRAMMonitor._sum_process_memory(gpu_elem)
        assert total == 69588  # 18 + 6 + 69564

    def test_empty_processes(self):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(UNIFIED_MEMORY_EMPTY_PROCESSES_XML)
        gpu_elem = root.find("gpu")
        total = VRAMMonitor._sum_process_memory(gpu_elem)
        assert total == 0
