"""
Tests for gateway.app — FastAPI Endpoint Integration.
Validates swap triggering, long-polling, and error handling
using httpx + FastAPI TestClient.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from gateway.schemas import AgentRequest, ContainerState, MediaType, ProfileMode, VRAMReport


# ──────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────

@pytest.fixture()
def patched_app(mock_vram_monitor):
    """
    Import app with mocked subsystems to avoid Docker/GPU deps.
    """
    import gateway.app as app_module

    # Replace singletons
    app_module.vram_monitor = mock_vram_monitor
    app_module.orchestrator._vram = mock_vram_monitor
    app_module.orchestrator._client = MagicMock()
    app_module.orchestrator._container_states = {}
    app_module.orchestrator._swap_in_progress = False
    app_module.orchestrator._active_profile = None

    # Reset swap task tracking
    app_module._active_swap_task = None
    app_module._active_swap_target = None

    # Mock inference proxy
    app_module.inference_proxy._session = MagicMock()

    return app_module.app


@pytest_asyncio.fixture()
async def client(patched_app):
    async with AsyncClient(
        transport=ASGITransport(app=patched_app), base_url="http://test"
    ) as c:
        yield c


# ──────────────────────────────────────────────────────
#  Health endpoint
# ──────────────────────────────────────────────────────

class TestHealthEndpoint:

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "version" in data

    @pytest.mark.asyncio
    async def test_health_includes_vram(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert "vram" in data


# ──────────────────────────────────────────────────────
#  Swap status endpoint
# ──────────────────────────────────────────────────────

class TestSwapStatus:

    @pytest.mark.asyncio
    async def test_swap_status_idle(self, client):
        resp = await client.get("/status/swap")
        assert resp.status_code == 200
        data = resp.json()
        assert data["swapping"] is False

    @pytest.mark.asyncio
    async def test_swap_status_during_swap(self, client, patched_app):
        import gateway.app as app_module
        app_module.orchestrator._swap_in_progress = True
        app_module.orchestrator._swap_start_time = __import__("time").time()

        resp = await client.get("/status/swap")
        data = resp.json()
        assert data["swapping"] is True

        # Cleanup
        app_module.orchestrator._swap_in_progress = False


# ──────────────────────────────────────────────────────
#  Chat completions — swap trigger
# ──────────────────────────────────────────────────────

class TestChatCompletionsSwap:

    @pytest.mark.asyncio
    async def test_503_when_swap_in_progress_no_long_polling(self, client):
        """When swap is running and long polling is off, should return 503."""
        import gateway.app as app_module

        app_module.orchestrator._swap_in_progress = True
        original = app_module.LONG_POLLING_ENABLED

        try:
            app_module.LONG_POLLING_ENABLED = False

            resp = await client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hello"}],
            })

            assert resp.status_code == 503
        finally:
            app_module.LONG_POLLING_ENABLED = original
            app_module.orchestrator._swap_in_progress = False

    @pytest.mark.asyncio
    async def test_successful_completion_when_profile_active(self, client):
        """When the correct profile is already active, should proxy through."""
        import gateway.app as app_module
        from gateway.config import PROFILE_FOCUS

        # Set active profile to focus
        key = app_module.orchestrator._profile_key(PROFILE_FOCUS)
        app_module.orchestrator._active_profile = key

        # Mock inference proxy
        expected_response = {
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}}]
        }
        app_module.inference_proxy.chat_completion = AsyncMock(
            return_value=expected_response
        )

        resp = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True


# ──────────────────────────────────────────────────────
#  Model listing
# ──────────────────────────────────────────────────────

class TestModelListing:

    @pytest.mark.asyncio
    async def test_list_models(self, client):
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        names = [m["id"] for m in data["data"]]
        assert "gpt-oss-120b" in names
        assert "qwen3-coder-next-80b" in names


# ──────────────────────────────────────────────────────
#  Swap task reuse & shield
# ──────────────────────────────────────────────────────

class TestSwapTaskManagement:
    """
    Validates the death-loop prevention fixes:
    - Swap tasks are shielded from cancellation
    - Duplicate swap tasks to the same target are reused
    - _execute_swap_and_drain cleans up tracking state
    """

    @pytest.mark.asyncio
    async def test_get_or_create_returns_same_task_for_same_target(self):
        """Two calls to _get_or_create_swap_task with the same target
           should return the exact same Task object."""
        import gateway.app as app_module
        from gateway.config import PROFILE_FOCUS

        # Reset state
        app_module._active_swap_task = None
        app_module._active_swap_target = None

        # Mock orchestrator to never finish (long-running swap)
        original_switch = app_module.orchestrator.switch_profile
        never_done = asyncio.Future()

        app_module.orchestrator.switch_profile = AsyncMock(
            return_value=never_done
        )

        try:
            target_key = app_module.orchestrator._profile_key(PROFILE_FOCUS)

            task1 = app_module._get_or_create_swap_task(PROFILE_FOCUS, target_key)
            task2 = app_module._get_or_create_swap_task(PROFILE_FOCUS, target_key)

            assert task1 is task2, "Should reuse the same task for the same target"
        finally:
            # Cleanup
            if app_module._active_swap_task and not app_module._active_swap_task.done():
                app_module._active_swap_task.cancel()
                try:
                    await app_module._active_swap_task
                except (asyncio.CancelledError, Exception):
                    pass
            app_module._active_swap_task = None
            app_module._active_swap_target = None
            app_module.orchestrator.switch_profile = original_switch

    @pytest.mark.asyncio
    async def test_execute_swap_and_drain_clears_state(self):
        """After _execute_swap_and_drain completes, tracking vars should be None."""
        import gateway.app as app_module
        from gateway.config import PROFILE_FOCUS

        app_module._active_swap_task = MagicMock()
        app_module._active_swap_target = "some_target"

        original_switch = app_module.orchestrator.switch_profile
        app_module.orchestrator.switch_profile = AsyncMock(return_value=True)
        original_drain = app_module._drain_buffered_requests
        app_module._drain_buffered_requests = AsyncMock()

        try:
            result = await app_module._execute_swap_and_drain(PROFILE_FOCUS)

            assert result is True
            assert app_module._active_swap_task is None
            assert app_module._active_swap_target is None
        finally:
            app_module.orchestrator.switch_profile = original_switch
            app_module._drain_buffered_requests = original_drain

    @pytest.mark.asyncio
    async def test_execute_swap_rejects_buffer_on_failure(self):
        """If the swap fails, buffered requests should be rejected."""
        import gateway.app as app_module
        from gateway.config import PROFILE_FOCUS

        app_module._active_swap_task = MagicMock()
        app_module._active_swap_target = "some_target"

        original_switch = app_module.orchestrator.switch_profile
        app_module.orchestrator.switch_profile = AsyncMock(return_value=False)

        original_reject = app_module.request_buffer.reject_all
        app_module.request_buffer.reject_all = AsyncMock()

        try:
            result = await app_module._execute_swap_and_drain(PROFILE_FOCUS)

            assert result is False
            app_module.request_buffer.reject_all.assert_called_once_with("Swap failed")
        finally:
            app_module.orchestrator.switch_profile = original_switch
            app_module.request_buffer.reject_all = original_reject

    @pytest.mark.asyncio
    async def test_shield_prevents_swap_cancellation(self):
        """asyncio.shield should prevent the swap task from being cancelled
        when wait_for times out."""
        swap_started = asyncio.Event()
        swap_completed = asyncio.Event()
        was_cancelled = False

        async def _slow_swap():
            nonlocal was_cancelled
            swap_started.set()
            try:
                await asyncio.sleep(0.5)
                swap_completed.set()
                return True
            except asyncio.CancelledError:
                was_cancelled = True
                raise

        task = asyncio.create_task(_slow_swap())
        await swap_started.wait()

        # wait_for with shield — timeout should NOT cancel the task
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(task), timeout=0.05
            )

        # The task should still be running, NOT cancelled
        assert not task.cancelled()
        assert not task.done()

        # Let it finish
        await task
        assert swap_completed.is_set()
        assert was_cancelled is False
