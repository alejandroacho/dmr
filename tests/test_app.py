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
    # Mock the backend instead of Docker client directly
    mock_be = MagicMock()
    mock_be.supports_pause = True
    mock_be.get_workload = AsyncMock(return_value=None)
    mock_be.create_and_start = AsyncMock()
    mock_be.stop_workload = AsyncMock()
    mock_be.remove_workload = AsyncMock()
    mock_be.list_workloads = AsyncMock(return_value=[])
    mock_be.ensure_network = AsyncMock()
    mock_be.resolve_hostname = MagicMock(
        side_effect=lambda name, engine: "192.168.200.12" if engine == "ray_vllm" else name
    )
    app_module.orchestrator._backend = mock_be
    app_module.orchestrator._container_states = {}
    app_module.orchestrator._swap_in_progress = False
    app_module.orchestrator._active_profile = None
    app_module.orchestrator._ray_status = None
    app_module.orchestrator._ray_status_profile = None
    app_module.orchestrator._ray_status_at = 0.0

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

    @pytest.mark.asyncio
    async def test_health_reports_declared_mode_not_key_substring(
        self, client, patched_app
    ):
        """Regression: the mode used to be guessed with `"focus" in key`, so
        FOCUS profiles without "focus" in their name reported as CREATIVE."""
        import gateway.app as app_module

        app_module.orchestrator._active_profile = "deepseek"
        resp = await client.get("/health")
        assert resp.json()["active_profile"] == "focus"


# ──────────────────────────────────────────────────────
#  Ray reporting in /health
# ──────────────────────────────────────────────────────

class TestHealthRay:
    """/health reports Ray only when the loaded models depend on it."""

    @pytest.mark.asyncio
    async def test_no_ray_block_when_profile_does_not_need_ray(
        self, client, patched_app
    ):
        import gateway.app as app_module

        app_module.orchestrator._active_profile = "test_a"
        await app_module.orchestrator.refresh_ray_status()

        resp = await client.get("/health")
        data = resp.json()
        assert data["ray"] is None
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_degraded_when_required_ray_cluster_is_short_on_nodes(
        self, client, patched_app
    ):
        import gateway.app as app_module
        from gateway.backends.base import ExecResult

        # qwen35 shards through Ray with TP=2 but only 1 node answers.
        app_module.orchestrator._backend.exec_in_workload = AsyncMock(
            return_value=ExecResult(exit_code=0, output="1\n")
        )
        app_module.orchestrator._active_profile = "qwen35"
        await app_module.orchestrator.refresh_ray_status()

        resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["ray"]["healthy"] is False
        cluster = data["ray"]["clusters"][0]
        assert cluster["nodes_active"] == 1
        assert cluster["nodes_required"] == 2
        assert "qwen3.5-122b" in cluster["models"]
        assert "1 of 2 nodes" in cluster["detail"]

    @pytest.mark.asyncio
    async def test_ok_when_required_ray_cluster_is_complete(
        self, client, patched_app
    ):
        import gateway.app as app_module
        from gateway.backends.base import ExecResult

        app_module.orchestrator._backend.exec_in_workload = AsyncMock(
            return_value=ExecResult(exit_code=0, output="2\n")
        )
        app_module.orchestrator._active_profile = "qwen35"
        await app_module.orchestrator.refresh_ray_status()

        resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ray"]["healthy"] is True
        assert data["ray"]["checked_seconds_ago"] is not None

    @pytest.mark.asyncio
    async def test_degraded_when_ray_head_is_missing(self, client, patched_app):
        import gateway.app as app_module

        # list_workloads returns [] -> no Ray head container running.
        app_module.orchestrator._active_profile = "test_ray"
        await app_module.orchestrator.refresh_ray_status()

        resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"
        cluster = data["ray"]["clusters"][0]
        assert cluster["nodes_active"] == 0
        assert "ray-node-head" in cluster["detail"]

    @pytest.mark.asyncio
    async def test_stale_reading_from_another_profile_is_discarded(
        self, client, patched_app
    ):
        """A verdict polled before a swap must not be reported afterwards."""
        import gateway.app as app_module
        from gateway.backends.base import ExecResult

        app_module.orchestrator._backend.exec_in_workload = AsyncMock(
            return_value=ExecResult(exit_code=0, output="1\n")
        )
        app_module.orchestrator._active_profile = "qwen35"
        await app_module.orchestrator.refresh_ray_status()
        assert app_module.orchestrator.ray_status is not None

        app_module.orchestrator._active_profile = "test_a"
        resp = await client.get("/health")
        data = resp.json()
        assert data["ray"] is None
        assert data["status"] == "ok"


# ──────────────────────────────────────────────────────
#  Ray watchdog
# ──────────────────────────────────────────────────────

class TestRayWatchdog:

    @pytest.mark.asyncio
    async def test_tick_marks_ready_model_error_when_ray_loses_a_node(
        self, patched_app
    ):
        """A worker that drops out after startup must not leave the model READY."""
        import gateway.app as app_module
        from gateway.backends.base import ExecResult
        from gateway.config import QWEN35_122B_FP8

        orch = app_module.orchestrator
        orch._active_profile = "qwen35"
        orch._container_states[QWEN35_122B_FP8.container_name] = ContainerState.READY
        # TP=2 but `ray status` only sees one node.
        orch._backend.exec_in_workload = AsyncMock(
            return_value=ExecResult(exit_code=0, output="1\n")
        )

        await app_module._ray_watchdog_tick()

        assert (
            orch._container_states[QWEN35_122B_FP8.container_name]
            == ContainerState.ERROR
        )
        assert orch.ray_status is not None and orch.ray_status.healthy is False

    @pytest.mark.asyncio
    async def test_tick_keeps_model_ready_when_ray_cluster_is_complete(
        self, patched_app, monkeypatch
    ):
        import gateway.app as app_module
        from gateway.backends.base import ExecResult
        from gateway.config import QWEN35_122B_FP8

        orch = app_module.orchestrator
        orch._active_profile = "qwen35"
        orch._container_states[QWEN35_122B_FP8.container_name] = ContainerState.READY
        orch._backend.exec_in_workload = AsyncMock(
            return_value=ExecResult(exit_code=0, output="2\n")
        )
        # monkeypatch, not plain assignment: `orchestrator` is a module
        # singleton, so an unrestored method would leak into later tests.
        monkeypatch.setattr(orch, "_check_vllm_health", AsyncMock(return_value=True))

        await app_module._ray_watchdog_tick()

        assert (
            orch._container_states[QWEN35_122B_FP8.container_name]
            == ContainerState.READY
        )
        assert orch.ray_status.healthy is True


# ──────────────────────────────────────────────────────
#  Sampling parameter passthrough
# ──────────────────────────────────────────────────────

class TestForwardedParams:
    """The gateway used to forward only messages/temperature/max_tokens/
    stream/tools, silently dropping everything else a client sent."""

    def test_forwards_declared_sampling_params(self):
        from gateway.schemas import AgentRequest

        req = AgentRequest(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "hi"}],
            top_p=0.9,
            stop=["\n\n"],
            seed=42,
            presence_penalty=0.5,
        )
        params = req.forwarded_params()

        assert params["top_p"] == 0.9
        assert params["stop"] == ["\n\n"]
        assert params["seed"] == 42
        assert params["presence_penalty"] == 0.5

    def test_forwards_unknown_openai_params(self):
        from gateway.schemas import AgentRequest

        req = AgentRequest(
            messages=[{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
            logit_bias={"123": -100},
        )
        params = req.forwarded_params()

        assert params["response_format"] == {"type": "json_object"}
        assert params["logit_bias"] == {"123": -100}

    def test_omits_gateway_only_fields(self):
        from gateway.schemas import AgentRequest

        req = AgentRequest(
            model="auto",
            messages=[{"role": "user", "content": "hi"}],
            agent_id="agent-3",
            priority=9,
        )
        params = req.forwarded_params()

        # The proxy sets "model" itself to the served-model-name.
        for field in ("model", "agent_id", "priority", "media_type", "media_params"):
            assert field not in params

    def test_omits_unset_optionals(self):
        """vLLM should apply its own defaults, not receive explicit nulls."""
        from gateway.schemas import AgentRequest

        params = AgentRequest(messages=[{"role": "user", "content": "hi"}]).forwarded_params()

        assert "top_p" not in params
        assert "seed" not in params
        assert params["temperature"] == 0.7  # declared defaults still travel


# ──────────────────────────────────────────────────────
#  Profile status endpoint
# ──────────────────────────────────────────────────────

class TestProfileStatus:

    @pytest.mark.asyncio
    async def test_profile_status_reports_the_loaded_model_not_the_last_defined(
        self, client, patched_app
    ):
        """Regression: cluster models share the 'vllm_node' container, so
        mapping container states back to models kept only the last definition
        and /status/profile reported qwen3.5 while deepseek was loaded."""
        import gateway.app as app_module
        from gateway.config import DEEPSEEK_V4_FLASH

        app_module.orchestrator._active_profile = "deepseek"
        app_module.orchestrator._container_states[
            DEEPSEEK_V4_FLASH.container_name
        ] = ContainerState.READY

        resp = await client.get("/status/profile")
        data = resp.json()
        assert [m["name"] for m in data["models"]] == ["deepseek-v4-flash"]

    @pytest.mark.asyncio
    async def test_profile_status_reports_declared_mode_not_key_substring(
        self, client, patched_app
    ):
        """Regression: same guess-from-key bug /health had — FOCUS profiles
        without "focus" in the key were reported as CREATIVE."""
        import gateway.app as app_module

        app_module.orchestrator._active_profile = "deepseek"
        resp = await client.get("/status/profile")
        assert resp.json()["active_profile"] == "focus"


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
        from tests.conftest import TEST_PROFILE_A, TEST_MODEL_A

        # Set active profile to focus with model READY
        key = app_module.orchestrator._registry_key(TEST_PROFILE_A)
        app_module.orchestrator._active_profile = key
        app_module.orchestrator._container_states[TEST_MODEL_A.container_name] = (
            ContainerState.READY
        )

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
        assert "test-model-a" in names
        assert "test-model-b" in names
        assert "test-model-c" in names

    @pytest.mark.asyncio
    async def test_list_models_includes_labels_for_active_profile(self, client):
        """When a profile is active, its label aliases should appear."""
        import gateway.app as app_module
        from tests.conftest import TEST_PROFILE_A

        key = app_module.orchestrator._registry_key(TEST_PROFILE_A)
        app_module.orchestrator._active_profile = key

        resp = await client.get("/v1/models")
        data = resp.json()
        names = [m["id"] for m in data["data"]]
        assert "chat" in names

        # The "chat" entry should reference test-model-a
        chat_entry = next(m for m in data["data"] if m["id"] == "chat")
        assert chat_entry["alias_for"] == "test-model-a"

    @pytest.mark.asyncio
    async def test_list_models_labels_change_with_profile(self, client):
        """Labels should reflect the active profile's model mapping."""
        import gateway.app as app_module
        from tests.conftest import TEST_PROFILE_B

        key = app_module.orchestrator._registry_key(TEST_PROFILE_B)
        app_module.orchestrator._active_profile = key

        resp = await client.get("/v1/models")
        data = resp.json()
        names = [m["id"] for m in data["data"]]
        assert "chat" in names
        assert "code" in names

        chat_entry = next(m for m in data["data"] if m["id"] == "chat")
        assert chat_entry["alias_for"] == "test-model-c"

        code_entry = next(m for m in data["data"] if m["id"] == "code")
        assert code_entry["alias_for"] == "test-model-b"


# ──────────────────────────────────────────────────────
#  Label-based routing
# ──────────────────────────────────────────────────────

class TestLabelRouting:

    def test_label_lookup_is_case_insensitive(self):
        """Regression: labels in the catalog are lowercase, so a client sending
        model="Chat" missed the alias and fell through to a profile lookup."""
        from gateway.router import SmartRouter
        from gateway.schemas import AgentRequest
        from tests.conftest import TEST_PROFILE_B, TEST_MODEL_C

        router = SmartRouter()
        decision = router.route(
            AgentRequest(model="Chat", messages=[{"role": "user", "content": "hi"}]),
            active_profile=TEST_PROFILE_B,
        )

        assert decision.target_model is TEST_MODEL_C
        assert decision.profile is TEST_PROFILE_B

    @pytest.mark.asyncio
    async def test_chat_label_stays_in_current_profile(self, client):
        """model='chat' should NOT trigger a swap — it resolves within active profile."""
        import gateway.app as app_module
        from tests.conftest import TEST_PROFILE_B, TEST_MODEL_C

        key = app_module.orchestrator._registry_key(TEST_PROFILE_B)
        app_module.orchestrator._active_profile = key
        app_module.orchestrator._container_states[TEST_MODEL_C.container_name] = (
            ContainerState.READY
        )

        expected = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        app_module.inference_proxy.chat_completion = AsyncMock(return_value=expected)

        resp = await client.post("/v1/chat/completions", json={
            "model": "chat",
            "messages": [{"role": "user", "content": "hello"}],
        })

        assert resp.status_code == 200
        # The proxy should have been called with the Qwen3.5-4B model
        call_args = app_module.inference_proxy.chat_completion.call_args
        model_used = call_args[0][0]  # first positional arg = ModelDefinition
        assert model_used.name == "test-model-c"

    @pytest.mark.asyncio
    async def test_code_label_triggers_swap_from_focus(self, client):
        """model='code' from focus profile should trigger swap to focus_code."""
        import gateway.app as app_module
        from tests.conftest import TEST_PROFILE_A, TEST_MODEL_A

        key = app_module.orchestrator._registry_key(TEST_PROFILE_A)
        app_module.orchestrator._active_profile = key
        app_module.orchestrator._container_states[TEST_MODEL_A.container_name] = (
            ContainerState.READY
        )

        # Disable long polling so we get an immediate 503 on swap
        original_lp = app_module.LONG_POLLING_ENABLED
        app_module.LONG_POLLING_ENABLED = False

        try:
            resp = await client.post("/v1/chat/completions", json={
                "model": "code",
                "messages": [{"role": "user", "content": "write code"}],
            })
            # Should trigger a swap → 503 (long polling disabled)
            assert resp.status_code == 503
        finally:
            app_module.LONG_POLLING_ENABLED = original_lp


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
        from tests.conftest import TEST_PROFILE_A

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
            target_key = app_module.orchestrator._profile_key(TEST_PROFILE_A)

            task1 = app_module._get_or_create_swap_task(TEST_PROFILE_A, target_key)
            task2 = app_module._get_or_create_swap_task(TEST_PROFILE_A, target_key)

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
    async def test_execute_swap_and_drain_clears_its_own_state(self):
        """A swap task clears the tracking vars it owns when it finishes."""
        import gateway.app as app_module
        from tests.conftest import TEST_PROFILE_A

        original_switch = app_module.orchestrator.switch_profile
        app_module.orchestrator.switch_profile = AsyncMock(return_value=True)
        original_drain = app_module._drain_buffered_requests
        app_module._drain_buffered_requests = AsyncMock()

        try:
            # Registered the way production does it, so the task can recognise
            # itself as the owner of the tracking vars.
            task = asyncio.create_task(
                app_module._execute_swap_and_drain(TEST_PROFILE_A)
            )
            app_module._active_swap_task = task
            app_module._active_swap_target = "test_a"

            assert await task is True
            assert app_module._active_swap_task is None
            assert app_module._active_swap_target is None
        finally:
            app_module.orchestrator.switch_profile = original_switch
            app_module._drain_buffered_requests = original_drain

    @pytest.mark.asyncio
    async def test_finishing_swap_does_not_clear_a_newer_swaps_state(self):
        """Regression: the finally block used to blank the tracking vars
        unconditionally. When a swap to another profile had replaced them, the
        newer swap was left untracked and a later request started a second,
        redundant swap to the same target."""
        import gateway.app as app_module
        from tests.conftest import TEST_PROFILE_A

        original_switch = app_module.orchestrator.switch_profile
        app_module.orchestrator.switch_profile = AsyncMock(return_value=True)
        original_drain = app_module._drain_buffered_requests
        app_module._drain_buffered_requests = AsyncMock()

        try:
            old_task = asyncio.create_task(
                app_module._execute_swap_and_drain(TEST_PROFILE_A)
            )
            # A swap to a different profile takes over the tracking slot while
            # the first one is still finishing.
            newer_task = MagicMock()
            app_module._active_swap_task = newer_task
            app_module._active_swap_target = "test_b"

            await old_task

            assert app_module._active_swap_task is newer_task
            assert app_module._active_swap_target == "test_b"
        finally:
            app_module._active_swap_task = None
            app_module._active_swap_target = None
            app_module.orchestrator.switch_profile = original_switch
            app_module._drain_buffered_requests = original_drain

    @pytest.mark.asyncio
    async def test_execute_swap_rejects_buffer_on_failure(self):
        """If the swap fails, buffered requests should be rejected."""
        import gateway.app as app_module
        from tests.conftest import TEST_PROFILE_A

        app_module._active_swap_task = MagicMock()
        app_module._active_swap_target = "some_target"

        original_switch = app_module.orchestrator.switch_profile
        app_module.orchestrator.switch_profile = AsyncMock(return_value=False)

        original_reject = app_module.request_buffer.reject_all
        app_module.request_buffer.reject_all = AsyncMock()

        try:
            result = await app_module._execute_swap_and_drain(TEST_PROFILE_A)

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


def test_profile_detail_reports_every_label_per_model():
    """All three spark_cluster models share the 'vllm_node' container, so a
    container-keyed label map collapsed them onto whichever label came last
    and every model was reported as "code". A model holding both roles must
    say so."""
    from gateway.app import _build_profile_detail
    from gateway.config import PROFILE_QWEN38

    detail = _build_profile_detail("qwen38", PROFILE_QWEN38, is_active=True)

    assert len(detail.models) == 1
    assert detail.models[0].label == "chat, code"
