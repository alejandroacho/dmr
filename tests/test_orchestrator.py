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

from gateway.config import (
    PROFILE_FOCUS,
    PROFILE_FOCUS_CODE,
    GPT_OSS_120B,
    QWEN3_CODER_NEXT_80B,
    VRAMProfile,
    VRAM_SAFETY_MARGIN_MB,
)
from gateway.schemas import ContainerState, ProfileMode, SwapStrategy
from gateway.orchestrator import ContainerOrchestrator
from tests.conftest import set_vram_free


# ──────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────

def _make_orchestrator(vram_monitor) -> ContainerOrchestrator:
    """Creates an orchestrator with mocked Docker client."""
    orch = ContainerOrchestrator.__new__(ContainerOrchestrator)
    orch._client = MagicMock()
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

        original_wait = orch._wait_all_ready

        async def _spy_wait(models, *args, **kwargs):
            nonlocal profile_during_wait
            # Capture active_profile WHILE waiting
            profile_during_wait = orch._active_profile
            # Simulate a brief wait
            await asyncio.sleep(0.01)

        orch._wait_all_ready = _spy_wait
        orch._teardown_current = AsyncMock()
        orch._ensure_container_running = AsyncMock()

        set_vram_free(mock_vram_monitor, 131072)  # plenty of VRAM

        result = await orch.switch_profile(PROFILE_FOCUS)

        assert result is True
        # During _wait_all_ready, the profile should NOT have been claimed yet
        assert profile_during_wait is None
        # After completion, it SHOULD be set
        assert orch._active_profile is not None
        assert "gpt-oss-120b" in orch._active_profile

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

        result = await orch.switch_profile(PROFILE_FOCUS)

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
        orch = _make_orchestrator(mock_vram_monitor)

        # Simulate container.get() raises NotFound → triggers _create_and_start
        from docker.errors import NotFound
        orch._client.containers.get.side_effect = NotFound("not found")
        orch._client.containers.run.return_value = MagicMock()

        await orch._ensure_container_running(GPT_OSS_120B, SwapStrategy.STOP_START)

        state = orch._container_states[GPT_OSS_120B.container_name]
        assert state == ContainerState.STARTING, (
            f"Expected STARTING, got {state}. "
            "Container should not be READY until healthcheck passes."
        )

    @pytest.mark.asyncio
    async def test_container_state_is_starting_for_existing_exited(self, mock_vram_monitor):
        """An existing container in 'exited' state should be removed and
        recreated fresh (to avoid stale network bindings)."""
        orch = _make_orchestrator(mock_vram_monitor)

        mock_container = MagicMock()
        mock_container.status = "exited"
        orch._client.containers.get.return_value = mock_container
        orch._client.containers.run.return_value = MagicMock()

        await orch._ensure_container_running(GPT_OSS_120B, SwapStrategy.STOP_START)

        # Old container should be removed
        mock_container.remove.assert_called_once_with(force=True)
        # A new container should be created
        orch._client.containers.run.assert_called_once()
        state = orch._container_states[GPT_OSS_120B.container_name]
        assert state == ContainerState.STARTING

    @pytest.mark.asyncio
    async def test_container_state_is_starting_for_running(self, mock_vram_monitor):
        """Even if container is already running, state should be STARTING
        (healthcheck hasn't confirmed yet)."""
        orch = _make_orchestrator(mock_vram_monitor)
        from gateway.config import DOCKER_NETWORK

        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.attrs = {
            "NetworkSettings": {
                "Networks": {DOCKER_NETWORK: {"NetworkID": "abc"}}
            }
        }
        orch._client.containers.get.return_value = mock_container

        await orch._ensure_container_running(GPT_OSS_120B, SwapStrategy.STOP_START)

        state = orch._container_states[GPT_OSS_120B.container_name]
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
        """When a fresh report is passed, it should be used instead of cache."""
        from gateway.schemas import VRAMReport

        # Cache says 0 free
        set_vram_free(mock_vram_monitor, 0)

        # Fresh report says plenty free
        fresh = VRAMReport(
            total_vram_mb=131072,
            total_used_mb=0,
            total_free_mb=131072,
            healthy=True,
        )

        # Without report arg → should fail (uses stale cache)
        assert mock_vram_monitor.has_enough_vram(90000) is False

        # With report arg → should succeed (uses fresh data)
        assert mock_vram_monitor.has_enough_vram(90000, fresh) is True

    def test_has_enough_vram_includes_safety_margin(self, mock_vram_monitor):
        """VRAM check includes the safety margin."""
        from gateway.schemas import VRAMReport

        # Exactly at the limit (required + safety margin)
        needed = 90000
        barely_enough = needed + VRAM_SAFETY_MARGIN_MB
        fresh = VRAMReport(
            total_vram_mb=131072,
            total_used_mb=131072 - barely_enough,
            total_free_mb=barely_enough,
            healthy=True,
        )
        assert mock_vram_monitor.has_enough_vram(needed, fresh) is True

        # 1 MB below → should fail
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
        """
        When VRAM is insufficient after teardown, the orchestrator should
        poll up to 15 times (not just sleep 2s).
        """
        orch = _make_orchestrator(mock_vram_monitor)
        orch._teardown_current = AsyncMock()
        orch._ensure_container_running = AsyncMock()

        query_count = 0
        original_query = mock_vram_monitor.query_gpus

        async def _gradual_free():
            nonlocal query_count
            query_count += 1
            # Simulate VRAM freeing on the 3rd query
            if query_count >= 3:
                set_vram_free(mock_vram_monitor, 131072)
            else:
                set_vram_free(mock_vram_monitor, 0)
            return mock_vram_monitor._latest_report

        mock_vram_monitor.query_gpus = _gradual_free

        async def _noop_wait(models, *a, **kw):
            pass

        orch._wait_all_ready = _noop_wait

        result = await orch.switch_profile(PROFILE_FOCUS)

        assert result is True
        # Should have queried at least 3 times (initial + retries)
        assert query_count >= 3


# ──────────────────────────────────────────────────────
#  Swap mutex / concurrency
# ──────────────────────────────────────────────────────

class TestSwapMutex:
    """Verifies swap operations are serialized."""

    @pytest.mark.asyncio
    async def test_is_swapping_flag(self, mock_vram_monitor):
        """is_swapping should be True during swap and False after."""
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

        await orch.switch_profile(PROFILE_FOCUS)

        assert was_swapping is True
        assert orch.is_swapping is False

    @pytest.mark.asyncio
    async def test_skip_if_same_profile(self, mock_vram_monitor):
        """Switching to the already-active profile should skip."""
        orch = _make_orchestrator(mock_vram_monitor)
        orch._active_profile = orch._profile_key(PROFILE_FOCUS)
        orch._teardown_current = AsyncMock()

        result = await orch.switch_profile(PROFILE_FOCUS)

        assert result is True
        orch._teardown_current.assert_not_called()

    @pytest.mark.asyncio
    async def test_swap_failure_returns_false(self, mock_vram_monitor):
        """If an exception occurs during swap, return False."""
        orch = _make_orchestrator(mock_vram_monitor)

        async def _explode(strategy, preserve=None):
            raise RuntimeError("Docker daemon crashed")

        orch._teardown_current = _explode
        set_vram_free(mock_vram_monitor, 131072)

        result = await orch.switch_profile(PROFILE_FOCUS)

        assert result is False
        assert orch.is_swapping is False  # Must reset even on failure


# ──────────────────────────────────────────────────────
#  Teardown preserves target containers
# ──────────────────────────────────────────────────────

class TestTeardownPreserve:
    """
    Validates that _teardown_current does NOT stop containers
    that belong to the target profile (the 'preserve' set).
    This prevents the death loop where the orchestrator stops
    the container it just started.
    """

    @pytest.mark.asyncio
    async def test_teardown_skips_preserved_containers(self, mock_vram_monitor):
        """Containers in the preserve set should NOT be stopped."""
        orch = _make_orchestrator(mock_vram_monitor)

        # Simulate: qwen3 is READY, gpt-oss is STARTING (just created)
        orch._container_states = {
            "vllm-qwen3-coder-next-80b": ContainerState.READY,
            "vllm-gpt-oss-120b": ContainerState.STARTING,
        }

        mock_container = MagicMock()
        mock_container.status = "running"
        orch._client.containers.get.return_value = mock_container

        # Preserve gpt-oss (target), only tear down qwen3
        await orch._teardown_current(
            SwapStrategy.STOP_START,
            preserve={"vllm-gpt-oss-120b"},
        )

        # qwen3 should be stopped
        assert orch._container_states["vllm-qwen3-coder-next-80b"] == ContainerState.STOPPED
        # gpt-oss should remain STARTING (untouched)
        assert orch._container_states["vllm-gpt-oss-120b"] == ContainerState.STARTING

    @pytest.mark.asyncio
    async def test_teardown_without_preserve_stops_all(self, mock_vram_monitor):
        """Without preserve, all READY/STARTING containers are stopped."""
        orch = _make_orchestrator(mock_vram_monitor)

        orch._container_states = {
            "container-a": ContainerState.READY,
            "container-b": ContainerState.STARTING,
        }

        mock_container = MagicMock()
        mock_container.status = "running"
        orch._client.containers.get.return_value = mock_container

        await orch._teardown_current(SwapStrategy.STOP_START)

        assert orch._container_states["container-a"] == ContainerState.STOPPED
        assert orch._container_states["container-b"] == ContainerState.STOPPED

    @pytest.mark.asyncio
    async def test_switch_profile_passes_target_containers_to_teardown(self, mock_vram_monitor):
        """switch_profile should send target container names to teardown."""
        orch = _make_orchestrator(mock_vram_monitor)
        teardown_calls = []

        async def _spy_teardown(strategy, preserve=None):
            teardown_calls.append({"strategy": strategy, "preserve": preserve})

        orch._teardown_current = _spy_teardown
        orch._ensure_container_running = AsyncMock()

        async def _noop_wait(models, *a, **kw):
            pass
        orch._wait_all_ready = _noop_wait

        set_vram_free(mock_vram_monitor, 131072)

        await orch.switch_profile(PROFILE_FOCUS)

        # The first teardown call should preserve the target model containers
        assert len(teardown_calls) >= 1
        preserve_set = teardown_calls[0]["preserve"]
        assert GPT_OSS_120B.container_name in preserve_set


# ──────────────────────────────────────────────────────
#  Profile key generation
# ──────────────────────────────────────────────────────

class TestProfileKey:
    def test_profile_key_deterministic(self):
        key = ContainerOrchestrator._profile_key(PROFILE_FOCUS)
        assert "focus:" in key
        assert "gpt-oss-120b" in key

    def test_different_profiles_different_keys(self):
        key1 = ContainerOrchestrator._profile_key(PROFILE_FOCUS)
        key2 = ContainerOrchestrator._profile_key(PROFILE_FOCUS_CODE)
        assert key1 != key2


# ──────────────────────────────────────────────────────
#  Orphan cleanup on startup
# ──────────────────────────────────────────────────────

class TestCleanupOrphanedContainers:
    """
    Validates that cleanup_orphaned_containers stops all known
    inference containers regardless of which profile is active.
    """

    @pytest.mark.asyncio
    async def test_stops_running_containers(self, mock_vram_monitor):
        """Running containers from ALL_MODELS should be removed."""
        orch = _make_orchestrator(mock_vram_monitor)

        mock_container = MagicMock()
        mock_container.status = "running"
        orch._client.containers.get.return_value = mock_container

        await orch.cleanup_orphaned_containers()

        # remove(force=True) should have been called for every container found
        assert mock_container.remove.call_count >= 1

    @pytest.mark.asyncio
    async def test_ignores_not_found(self, mock_vram_monitor):
        """Containers that don't exist should be silently skipped."""
        from docker.errors import NotFound

        orch = _make_orchestrator(mock_vram_monitor)
        orch._client.containers.get.side_effect = NotFound("gone")

        # Should not raise
        await orch.cleanup_orphaned_containers()

    @pytest.mark.asyncio
    async def test_removes_paused_containers(self, mock_vram_monitor):
        """Paused containers should also be removed."""
        orch = _make_orchestrator(mock_vram_monitor)

        mock_container = MagicMock()
        mock_container.status = "paused"
        orch._client.containers.get.return_value = mock_container

        await orch.cleanup_orphaned_containers()

        assert mock_container.remove.call_count >= 1

    @pytest.mark.asyncio
    async def test_removes_exited_containers(self, mock_vram_monitor):
        """Exited containers should also be removed (stale network config)."""
        orch = _make_orchestrator(mock_vram_monitor)

        mock_container = MagicMock()
        mock_container.status = "exited"
        orch._client.containers.get.return_value = mock_container

        await orch.cleanup_orphaned_containers()

        assert mock_container.remove.call_count >= 1


# ──────────────────────────────────────────────────────
#  Network reconnection for stale containers
# ──────────────────────────────────────────────────────

class TestEnsureCorrectNetwork:
    """
    Validates that _ensure_correct_network reconnects a container
    to the inference Docker network when it's on a stale network.
    """

    @pytest.mark.asyncio
    async def test_no_action_if_already_on_network(self, mock_vram_monitor):
        """Container already on the correct network → no reconnect."""
        import asyncio
        from gateway.config import DOCKER_NETWORK

        orch = _make_orchestrator(mock_vram_monitor)

        mock_container = MagicMock()
        mock_container.attrs = {
            "NetworkSettings": {
                "Networks": {DOCKER_NETWORK: {"NetworkID": "abc123"}}
            }
        }

        await orch._ensure_correct_network(mock_container, asyncio.get_event_loop())

        # networks.get should NOT have been called (no reconnect needed)
        orch._client.networks.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconnects_if_on_wrong_network(self, mock_vram_monitor):
        """Container on a stale network → should be reconnected."""
        import asyncio
        from gateway.config import DOCKER_NETWORK

        orch = _make_orchestrator(mock_vram_monitor)

        mock_container = MagicMock()
        mock_container.name = "vllm-gpt-oss-120b"
        mock_container.attrs = {
            "NetworkSettings": {
                "Networks": {"old_stale_network": {"NetworkID": "old123"}}
            }
        }

        mock_network = MagicMock()
        orch._client.networks.get.return_value = mock_network

        await orch._ensure_correct_network(mock_container, asyncio.get_event_loop())

        orch._client.networks.get.assert_called_once_with(DOCKER_NETWORK)
        mock_network.connect.assert_called_once_with(mock_container)

    @pytest.mark.asyncio
    async def test_ensure_running_calls_network_check(self, mock_vram_monitor):
        """_ensure_container_running should verify network for running containers."""
        orch = _make_orchestrator(mock_vram_monitor)

        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.attrs = {
            "NetworkSettings": {
                "Networks": {"stale_net": {"NetworkID": "old"}}
            }
        }
        orch._client.containers.get.return_value = mock_container

        mock_network = MagicMock()
        orch._client.networks.get.return_value = mock_network

        await orch._ensure_container_running(GPT_OSS_120B, SwapStrategy.STOP_START)

        # Network should have been reconnected
        mock_network.connect.assert_called_once_with(mock_container)
        # State should still be STARTING
        assert orch._container_states[GPT_OSS_120B.container_name] == ContainerState.STARTING
