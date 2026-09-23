"""
Tests for gateway.orchestrator — Container Lifecycle & Swap Logic.
Validates the fixes for:
  - Bug 1: Profile claimed AFTER healthcheck (not before)
  - Bug 2: Container state stays STARTING until healthcheck passes
  - Bug 3: VRAM check uses fresh query + wait loop
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.backends.base import ExecResult, WorkloadInfo
from gateway.config import (
    VRAMProfile,
    VRAM_SAFETY_MARGIN_MB,
)
from gateway.schemas import ContainerState, ProfileMode, SwapStrategy
from gateway.orchestrator import ContainerOrchestrator
from tests.conftest import (
    TEST_MODEL_A,
    TEST_MODEL_B,
    TEST_PROFILE_A,
    TEST_PROFILE_B,
    set_vram_free,
)


# ──────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────

def _make_orchestrator(vram_monitor, backend=None) -> ContainerOrchestrator:
    """Creates an orchestrator with a mocked backend."""
    if backend is None:
        backend = MagicMock()
        backend.supports_pause = True
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

    orch = ContainerOrchestrator.__new__(ContainerOrchestrator)
    orch._backend = backend
    orch._vram = vram_monitor
    orch._swap_lock = asyncio.Lock()
    orch._container_states = {}
    orch._active_profile = None
    orch._swap_in_progress = False
    orch._swap_start_time = 0.0
    return orch


# ──────────────────────────────────────────────────────
#  Bug 1: Profile set AFTER healthcheck
# ──────────────────────────────────────────────────────

class TestProfileClaimedAfterHealthcheck:
    """
    Verifies that `_active_profile` is NOT set until `_wait_all_ready`
    has completed (or timed out).
    """

    @pytest.mark.asyncio
    async def test_profile_not_set_during_healthcheck(self, mock_vram_monitor):
        """
        The profile key must only appear AFTER _wait_all_ready finishes.
        We intercept _wait_all_ready to check the ordering.
        """
        orch = _make_orchestrator(mock_vram_monitor)
        profile_during_wait = None

        async def _spy_wait(models, *args, **kwargs):
            nonlocal profile_during_wait
            profile_during_wait = orch._active_profile
            await asyncio.sleep(0.01)

        orch._wait_all_ready = _spy_wait
        orch._teardown_current = AsyncMock()
        orch._ensure_container_running = AsyncMock()

        set_vram_free(mock_vram_monitor, 131072)

        result = await orch.switch_profile(TEST_PROFILE_A)

        assert result is True
        assert profile_during_wait is None
        assert orch._active_profile is not None
        assert orch._active_profile == orch._registry_key(TEST_PROFILE_A)

    @pytest.mark.asyncio
    async def test_profile_set_even_on_healthcheck_timeout(self, mock_vram_monitor):
        """
        Even if healthcheck times out, the profile should still be claimed
        (graceful degradation).
        """
        orch = _make_orchestrator(mock_vram_monitor)

        async def _timeout_wait(models, *args, **kwargs):
            raise asyncio.TimeoutError("Simulated timeout")

        orch._wait_all_ready = _timeout_wait
        orch._teardown_current = AsyncMock()
        orch._ensure_container_running = AsyncMock()
        set_vram_free(mock_vram_monitor, 131072)

        result = await orch.switch_profile(TEST_PROFILE_A)

        assert result is True
        assert orch._active_profile is not None


# ──────────────────────────────────────────────────────
#  Bug 2: Container state stays STARTING
# ──────────────────────────────────────────────────────

class TestContainerStateStarting:
    """
    Verifies that _ensure_container_running leaves the container
    in STARTING state (not READY), so _wait_all_ready controls
    the transition to READY after healthcheck.
    """

    @pytest.mark.asyncio
    async def test_container_state_is_starting_after_ensure(self, mock_vram_monitor):
        """Container not found → triggers create_and_start → state=STARTING."""
        orch = _make_orchestrator(mock_vram_monitor)

        # Backend returns None → workload not found → create_and_start called
        await orch._ensure_container_running(TEST_MODEL_A, SwapStrategy.STOP_START)

        state = orch._container_states[TEST_MODEL_A.container_name]
        assert state == ContainerState.STARTING, (
            f"Expected STARTING, got {state}. "
            "Container should not be READY until healthcheck passes."
        )

    @pytest.mark.asyncio
    async def test_container_state_is_starting_for_existing_exited(self, mock_vram_monitor):
        """An existing workload in 'exited' state should be removed and
        recreated fresh."""
        orch = _make_orchestrator(mock_vram_monitor)

        # Backend finds an exited workload
        orch._backend.get_workload = AsyncMock(
            return_value=WorkloadInfo(name="vllm-test-a", status="exited")
        )

        await orch._ensure_container_running(TEST_MODEL_A, SwapStrategy.STOP_START)

        # Old workload should be removed
        orch._backend.remove_workload.assert_called_once_with("vllm-test-a")
        # A new workload should be created
        orch._backend.create_and_start.assert_called_once()
        state = orch._container_states[TEST_MODEL_A.container_name]
        assert state == ContainerState.STARTING

    @pytest.mark.asyncio
    async def test_container_state_is_starting_for_running(self, mock_vram_monitor):
        """Even if workload is already running, state should be STARTING
        (healthcheck hasn't confirmed yet)."""
        orch = _make_orchestrator(mock_vram_monitor)

        orch._backend.get_workload = AsyncMock(
            return_value=WorkloadInfo(name="vllm-test-a", status="running")
        )

        await orch._ensure_container_running(TEST_MODEL_A, SwapStrategy.STOP_START)

        state = orch._container_states[TEST_MODEL_A.container_name]
        assert state == ContainerState.STARTING


# ──────────────────────────────────────────────────────
#  Bug 3: VRAM check uses fresh report + wait loop
# ──────────────────────────────────────────────────────

class TestVRAMFreshCheck:
    """
    Verifies that has_enough_vram uses the fresh VRAMReport passed
    as argument, not the stale cached one.
    """

    def test_has_enough_vram_uses_report_arg(self, mock_vram_monitor):
        from gateway.schemas import VRAMReport

        set_vram_free(mock_vram_monitor, 0)

        fresh = VRAMReport(
            total_vram_mb=131072,
            total_used_mb=0,
            total_free_mb=131072,
            healthy=True,
        )

        assert mock_vram_monitor.has_enough_vram(90000) is False
        assert mock_vram_monitor.has_enough_vram(90000, fresh) is True

    def test_has_enough_vram_includes_safety_margin(self, mock_vram_monitor):
        from gateway.schemas import VRAMReport

        needed = 90000
        barely_enough = needed + VRAM_SAFETY_MARGIN_MB
        fresh = VRAMReport(
            total_vram_mb=131072,
            total_used_mb=131072 - barely_enough,
            total_free_mb=barely_enough,
            healthy=True,
        )
        assert mock_vram_monitor.has_enough_vram(needed, fresh) is True

        not_enough = barely_enough - 1
        fresh2 = VRAMReport(
            total_vram_mb=131072,
            total_used_mb=131072 - not_enough,
            total_free_mb=not_enough,
            healthy=True,
        )
        assert mock_vram_monitor.has_enough_vram(needed, fresh2) is False

    @pytest.mark.asyncio
    async def test_vram_wait_loop_retries_until_freed(self, mock_vram_monitor):
        orch = _make_orchestrator(mock_vram_monitor)
        orch._teardown_current = AsyncMock()
        orch._ensure_container_running = AsyncMock()

        query_count = 0

        async def _gradual_free():
            nonlocal query_count
            query_count += 1
            if query_count >= 3:
                set_vram_free(mock_vram_monitor, 131072)
            else:
                set_vram_free(mock_vram_monitor, 0)
            return mock_vram_monitor._latest_report

        mock_vram_monitor.query_gpus = _gradual_free

        async def _noop_wait(models, *a, **kw):
            pass

        orch._wait_all_ready = _noop_wait

        result = await orch.switch_profile(TEST_PROFILE_A)

        assert result is True
        assert query_count >= 3


# ──────────────────────────────────────────────────────
#  Swap mutex / concurrency
# ──────────────────────────────────────────────────────

class TestSwapMutex:

    @pytest.mark.asyncio
    async def test_is_swapping_flag(self, mock_vram_monitor):
        orch = _make_orchestrator(mock_vram_monitor)
        was_swapping = False

        async def _check_swapping(models, *a, **kw):
            nonlocal was_swapping
            was_swapping = orch.is_swapping
            await asyncio.sleep(0.01)

        orch._wait_all_ready = _check_swapping
        orch._teardown_current = AsyncMock()
        orch._ensure_container_running = AsyncMock()
        set_vram_free(mock_vram_monitor, 131072)

        await orch.switch_profile(TEST_PROFILE_A)

        assert was_swapping is True
        assert orch.is_swapping is False

    @pytest.mark.asyncio
    async def test_skip_if_same_profile(self, mock_vram_monitor):
        orch = _make_orchestrator(mock_vram_monitor)
        orch._active_profile = orch._registry_key(TEST_PROFILE_A)
        orch._teardown_current = AsyncMock()

        result = await orch.switch_profile(TEST_PROFILE_A)

        assert result is True
        orch._teardown_current.assert_not_called()

    @pytest.mark.asyncio
    async def test_swap_failure_returns_false(self, mock_vram_monitor):
        orch = _make_orchestrator(mock_vram_monitor)

        async def _explode(strategy, preserve=None):
            raise RuntimeError("Backend crashed")

        orch._teardown_current = _explode
        set_vram_free(mock_vram_monitor, 131072)

        result = await orch.switch_profile(TEST_PROFILE_A)

        assert result is False
        assert orch.is_swapping is False


# ──────────────────────────────────────────────────────
#  Teardown preserves target containers
# ──────────────────────────────────────────────────────

class TestTeardownPreserve:

    @pytest.mark.asyncio
    async def test_teardown_skips_preserved_containers(self, mock_vram_monitor):
        orch = _make_orchestrator(mock_vram_monitor)

        orch._container_states = {
            "vllm-test-b": ContainerState.READY,
            "vllm-test-a": ContainerState.STARTING,
        }

        await orch._teardown_current(
            SwapStrategy.STOP_START,
            preserve={"vllm-test-a"},
        )

        assert orch._container_states["vllm-test-b"] == ContainerState.STOPPED
        assert orch._container_states["vllm-test-a"] == ContainerState.STARTING

    @pytest.mark.asyncio
    async def test_teardown_without_preserve_stops_all(self, mock_vram_monitor):
        orch = _make_orchestrator(mock_vram_monitor)

        orch._container_states = {
            "container-a": ContainerState.READY,
            "container-b": ContainerState.STARTING,
        }

        await orch._teardown_current(SwapStrategy.STOP_START)

        assert orch._container_states["container-a"] == ContainerState.STOPPED
        assert orch._container_states["container-b"] == ContainerState.STOPPED

    @pytest.mark.asyncio
    async def test_switch_profile_passes_target_containers_to_teardown(self, mock_vram_monitor):
        orch = _make_orchestrator(mock_vram_monitor)
        teardown_calls = []

        async def _spy_teardown(strategy, preserve=None, preserve_models=None):
            teardown_calls.append({
                "strategy": strategy,
                "preserve": preserve,
                "preserve_models": preserve_models,
            })

        orch._teardown_current = _spy_teardown
        orch._ensure_container_running = AsyncMock()

        async def _noop_wait(models, *a, **kw):
            pass
        orch._wait_all_ready = _noop_wait

        set_vram_free(mock_vram_monitor, 131072)

        await orch.switch_profile(TEST_PROFILE_A)

        assert len(teardown_calls) >= 1
        preserve_set = teardown_calls[0]["preserve"]
        assert TEST_MODEL_A.container_name in preserve_set
        # Container names alone cannot tell two spark_cluster models apart —
        # they share one container — so the model identities must come too.
        assert TEST_MODEL_A.name in teardown_calls[0]["preserve_models"]


# ──────────────────────────────────────────────────────
#  Profile key generation
# ──────────────────────────────────────────────────────

class TestProfileKey:
    def test_profile_key_deterministic(self):
        key = ContainerOrchestrator._profile_key(TEST_PROFILE_A)
        assert "focus:" in key
        assert "test-model-a" in key

    def test_different_profiles_different_keys(self):
        key1 = ContainerOrchestrator._profile_key(TEST_PROFILE_A)
        key2 = ContainerOrchestrator._profile_key(TEST_PROFILE_B)
        assert key1 != key2


# ──────────────────────────────────────────────────────
#  Orphan cleanup on startup
# ──────────────────────────────────────────────────────

class TestCleanupOrphanedContainers:

    @pytest.mark.asyncio
    async def test_removes_found_workloads(self, mock_vram_monitor):
        """Found workloads from ALL_MODELS should be removed."""
        orch = _make_orchestrator(mock_vram_monitor)

        # Backend finds workloads for non-ray models
        orch._backend.get_workload = AsyncMock(
            return_value=WorkloadInfo(name="test", status="running")
        )

        await orch.cleanup_orphaned_containers()

        assert orch._backend.remove_workload.call_count >= 1

    @pytest.mark.asyncio
    async def test_ignores_not_found(self, mock_vram_monitor):
        """Workloads that don't exist should be silently skipped."""
        orch = _make_orchestrator(mock_vram_monitor)

        # Backend returns None for all workloads
        orch._backend.get_workload = AsyncMock(return_value=None)

        # Should not raise
        await orch.cleanup_orphaned_containers()

    @pytest.mark.asyncio
    async def test_removes_paused_workloads(self, mock_vram_monitor):
        orch = _make_orchestrator(mock_vram_monitor)
        orch._backend.get_workload = AsyncMock(
            return_value=WorkloadInfo(name="test", status="paused")
        )

        await orch.cleanup_orphaned_containers()

        assert orch._backend.remove_workload.call_count >= 1

    @pytest.mark.asyncio
    async def test_removes_exited_workloads(self, mock_vram_monitor):
        orch = _make_orchestrator(mock_vram_monitor)
        orch._backend.get_workload = AsyncMock(
            return_value=WorkloadInfo(name="test", status="exited")
        )

        await orch.cleanup_orphaned_containers()

        assert orch._backend.remove_workload.call_count >= 1


# ──────────────────────────────────────────────────────
#  Network check delegates to backend
# ──────────────────────────────────────────────────────

class TestEnsureNetwork:

    @pytest.mark.asyncio
    async def test_ensure_running_calls_network_check(self, mock_vram_monitor):
        """_ensure_container_running should call backend.ensure_network for running workloads."""
        orch = _make_orchestrator(mock_vram_monitor)

        orch._backend.get_workload = AsyncMock(
            return_value=WorkloadInfo(name="vllm-test-a", status="running")
        )

        await orch._ensure_container_running(TEST_MODEL_A, SwapStrategy.STOP_START)

        orch._backend.ensure_network.assert_called_once_with("vllm-test-a")
        assert orch._container_states[TEST_MODEL_A.container_name] == ContainerState.STARTING


# ──────────────────────────────────────────────────────
#  Swap deduplication under the mutex
# ──────────────────────────────────────────────────────

class TestSwapRecheckUnderLock:

    @staticmethod
    def _stub_lifecycle(orch):
        """Stub the slow parts so only the swap control flow is exercised."""
        orch._teardown_current = AsyncMock()
        orch._ensure_container_running = AsyncMock()
        orch._wait_all_ready = AsyncMock()

    @pytest.mark.asyncio
    async def test_second_waiter_skips_a_swap_already_done_for_it(
        self, mock_vram_monitor
    ):
        """Regression: the "already active" check only ran before acquiring the
        mutex. A swap queued behind another swap to the SAME profile went ahead
        and tore the model down to reload identical weights — minutes of work
        for no change."""
        orch = _make_orchestrator(mock_vram_monitor)
        self._stub_lifecycle(orch)
        set_vram_free(mock_vram_monitor, 131072)

        first = asyncio.create_task(orch.switch_profile(TEST_PROFILE_A))
        # Let the first swap take the lock before the second queues behind it.
        await asyncio.sleep(0)
        second = asyncio.create_task(orch.switch_profile(TEST_PROFILE_A))

        assert await first is True
        assert await second is True

        assert orch._ensure_container_running.await_count == len(
            TEST_PROFILE_A.primary_models + TEST_PROFILE_A.secondary_models
        ), "the queued swap reloaded the model again"
        assert orch._active_profile == orch._registry_key(TEST_PROFILE_A)

    @pytest.mark.asyncio
    async def test_force_still_reruns_the_swap(self, mock_vram_monitor):
        """force=True must survive the new re-check: it exists precisely to
        restart a profile that is already marked active but is unhealthy."""
        orch = _make_orchestrator(mock_vram_monitor)
        self._stub_lifecycle(orch)
        set_vram_free(mock_vram_monitor, 131072)

        assert await orch.switch_profile(TEST_PROFILE_A) is True
        orch._ensure_container_running.reset_mock()

        assert await orch.switch_profile(TEST_PROFILE_A, force=True) is True
        assert orch._ensure_container_running.await_count >= 1
