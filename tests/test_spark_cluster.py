"""
Tests for the `spark_cluster` engine — multi-node vLLM on the DGX Spark pair.

Unlike Docker-managed models, these run as one `vllm serve` process per node
inside the long-lived `vllm_node` containers that launch-cluster.sh owns:
rank 0 on the head node (reached via the Docker socket) and rank >= 1 headless
on the workers (reached over SSH). The Gateway manages the processes; it must
never create or remove the containers.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.backends.base import WorkloadInfo
from gateway.config import (
    DEEPSEEK_V4_FLASH,
    EXEC_ENGINES,
    PROFILE_DEEPSEEK,
    PROFILE_QWEN38,
    PROFILES,
    QWEN38_FLASH_NEXT,
    SPARK_CLUSTER_CONTAINER,
)
from gateway.orchestrator import ContainerOrchestrator
from gateway.schemas import ContainerState, SwapStrategy


# ──────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────

def _orchestrator(vram_monitor, backend) -> ContainerOrchestrator:
    return ContainerOrchestrator(vram_monitor=vram_monitor, backend=backend)


def _decode_script(call_args) -> str:
    """Recovers the bash script from a `echo <b64> | base64 -d | bash` command."""
    text = call_args if isinstance(call_args, str) else call_args[-1]
    payload = text.split("echo ", 1)[1].split(" |", 1)[0]
    return base64.b64decode(payload).decode()


def _running(name: str) -> WorkloadInfo:
    return WorkloadInfo(name=name, status="running")


# ──────────────────────────────────────────────────────
#  Catalog wiring
# ──────────────────────────────────────────────────────

def test_deepseek_is_registered_as_exec_engine():
    assert DEEPSEEK_V4_FLASH.engine == "spark_cluster"
    assert DEEPSEEK_V4_FLASH.engine in EXEC_ENGINES
    assert PROFILES["deepseek"] is PROFILE_DEEPSEEK
    assert DEEPSEEK_V4_FLASH.cluster_nodes == 2


def test_deepseek_port_does_not_collide_with_other_models():
    """8010 belongs to GEMMA4_31B_FP8_VLLM; DeepSeek must sit outside that range."""
    from gateway.config import ALL_MODELS

    others = [m.port for m in ALL_MODELS if m is not DEEPSEEK_V4_FLASH]
    assert DEEPSEEK_V4_FLASH.port not in others


def test_deepseek_profile_skips_local_vram_check():
    """Weights are sharded across two nodes, so local NVML sees only half."""
    assert PROFILE_DEEPSEEK.skip_vram_check is True


def test_qwen38_shards_natively_and_needs_no_ray():
    """Unlike Qwen3.5, this recipe passes no --distributed-executor-backend,
    so it uses vLLM's native multi-node mode and imposes no Ray precondition
    on the cluster containers."""
    assert QWEN38_FLASH_NEXT.engine == "spark_cluster"
    assert QWEN38_FLASH_NEXT.cluster_nodes == 2
    assert QWEN38_FLASH_NEXT.requires_ray_cluster is False
    assert QWEN38_FLASH_NEXT.required_ray_nodes == 0
    assert PROFILES["qwen38"] is PROFILE_QWEN38
    assert PROFILE_QWEN38.skip_vram_check is True
    assert PROFILE_QWEN38.labels["chat"] is QWEN38_FLASH_NEXT


def test_qwen38_port_does_not_collide_with_other_models():
    from gateway.config import ALL_MODELS

    others = [m.port for m in ALL_MODELS if m is not QWEN38_FLASH_NEXT]
    assert QWEN38_FLASH_NEXT.port not in others

# ──────────────────────────────────────────────────────
#  Command construction
# ──────────────────────────────────────────────────────

def test_build_cmd_keeps_auto_context_and_omits_quantization():
    cmd = [str(p) for p in ContainerOrchestrator._build_cmd(DEEPSEEK_V4_FLASH)]

    assert cmd[cmd.index("--max-model-len") + 1] == "auto"
    # Native FP4/FP8 checkpoint: passing --quantization would override it.
    assert "--quantization" not in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == "deepseek-v4-flash"
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"


def test_build_cmd_preserves_json_and_dotted_flags():
    cmd = [str(p) for p in ContainerOrchestrator._build_cmd(DEEPSEEK_V4_FLASH)]

    spec = cmd[cmd.index("--speculative-config") + 1]
    assert '"method":"dspark"' in spec

    # Nested kwargs must stay single tokens with "=", not be split in two.
    assert "--default-chat-template-kwargs.thinking=true" in cmd
    assert "--default-chat-template-kwargs.reasoning_effort=high" in cmd
    assert cmd[cmd.index("--load-format") + 1] == "b12x"
    assert cmd[cmd.index("--attention-backend") + 1] == "B12X"


def test_qwen38_cmd_declares_its_mixed_quantization():
    """NVFP4 + MXFP8 checkpoint: unlike the native FP8/FP4 models, vLLM needs
    --quantization spelled out, and the hybrid SSM layers need an aligned
    mamba cache."""
    cmd = [str(p) for p in ContainerOrchestrator._build_cmd(QWEN38_FLASH_NEXT)]

    assert cmd[cmd.index("--quantization") + 1] == "modelopt_mixed"
    assert cmd[cmd.index("--mamba-cache-mode") + 1] == "align"
    assert cmd[cmd.index("--load-format") + 1] == "b12x"
    assert cmd[cmd.index("--gdn-decode-kernel") + 1] == "b12x"
    assert cmd[cmd.index("--tool-call-parser") + 1] == "qwen3_xml"
    assert cmd[cmd.index("--max-model-len") + 1] == "262144"

    spec = cmd[cmd.index("--speculative-config") + 1]
    assert '"method":"mtp"' in spec


# ──────────────────────────────────────────────────────
#  Start / stop across both nodes
# ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_launches_worker_before_head(mock_vram_monitor, mock_backend):
    orch = _orchestrator(mock_vram_monitor, mock_backend)
    order: list[str] = []

    async def _worker(host, command):
        order.append(f"worker:{host}")
        return "true"

    async def _head(name, cmd):
        order.append("head")
        return MagicMock(exit_code=0, output="")

    with patch.object(orch, "_run_on_worker", side_effect=_worker):
        mock_backend.exec_in_workload = AsyncMock(side_effect=_head)
        await orch._start_spark_serve(DEEPSEEK_V4_FLASH)

    # Teardown of a stale process runs first (worker then head), then the
    # launch itself must reach the worker before the rank-0 master.
    launch_order = order[-2:]
    assert launch_order[0].startswith("worker:")
    assert launch_order[1] == "head"


@pytest.mark.asyncio
async def test_headless_only_on_worker_rank(mock_vram_monitor, mock_backend):
    orch = _orchestrator(mock_vram_monitor, mock_backend)
    worker_calls: list[str] = []

    async def _worker(host, command):
        worker_calls.append(command)
        return "true"

    with patch.object(orch, "_run_on_worker", side_effect=_worker):
        await orch._start_spark_serve(DEEPSEEK_V4_FLASH)

    worker_script = _decode_script(worker_calls[-1])
    head_script = _decode_script(mock_backend.exec_in_workload.call_args[0][1])

    assert "--headless" in worker_script
    assert "--node-rank 1" in worker_script
    assert "--headless" not in head_script
    assert "--node-rank 0" in head_script
    # Only rank 0 owns the HTTP port.
    assert f"--port {DEEPSEEK_V4_FLASH.port}" in head_script


@pytest.mark.asyncio
async def test_json_flags_survive_the_shell_layers(mock_vram_monitor, mock_backend):
    """Regression: nesting the command in `bash -c '...'` let the wrapper's
    quotes collide with shlex's, and the shell ate the JSON braces — vLLM saw
    --reasoning-config as the bare string 'reasoning_parser:deepseek_v4'."""
    orch = _orchestrator(mock_vram_monitor, mock_backend)

    with patch.object(orch, "_run_on_worker", AsyncMock(return_value="")):
        await orch._start_spark_serve(DEEPSEEK_V4_FLASH)

    script = _decode_script(mock_backend.exec_in_workload.call_args[0][1])
    assert (
        '{"reasoning_parser":"deepseek_v4",'
        '"reasoning_start_str":"","reasoning_end_str":""}'
    ) in script
    assert '{"method":"dspark","num_speculative_tokens":5,' in script
    assert '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' in script


@pytest.mark.asyncio
async def test_b12x_env_is_exported_on_every_rank(mock_vram_monitor, mock_backend):
    """launch-cluster.sh exports these in its launch script, not via docker run,
    so the container env does not carry them and each rank must set them."""
    orch = _orchestrator(mock_vram_monitor, mock_backend)
    worker_calls: list[str] = []

    async def _worker(host, command):
        worker_calls.append(command)
        return "true"

    with patch.object(orch, "_run_on_worker", side_effect=_worker):
        await orch._start_spark_serve(DEEPSEEK_V4_FLASH)

    for script in (
        _decode_script(worker_calls[-1]),
        _decode_script(mock_backend.exec_in_workload.call_args[0][1]),
    ):
        assert "export VLLM_USE_B12X_MOE=1" in script
        assert "export CUTE_DSL_ARCH=sm_121a" in script
        assert "export B12X_MLA_SM120_UNIFIED=1" in script


@pytest.mark.asyncio
async def test_swap_stops_the_outgoing_model_in_the_shared_container(
    mock_vram_monitor, mock_backend
):
    """Swapping between two spark_cluster models must stop the outgoing serve
    process. Both execs live in the same 'vllm_node', so the outgoing model's
    container is always among the preserved ones — teardown used to skip it,
    and the survivor held the distributed master port until the incoming model
    died with EADDRINUSE."""
    orch = _orchestrator(mock_vram_monitor, mock_backend)
    orch._active_profile = "qwen38"
    orch._container_states[SPARK_CLUSTER_CONTAINER] = ContainerState.READY

    with patch.object(orch, "_stop_exec_model", AsyncMock()) as stop:
        await orch._teardown_current(
            SwapStrategy.STOP_START,
            preserve={SPARK_CLUSTER_CONTAINER},
            preserve_models={DEEPSEEK_V4_FLASH.name},
        )

    stopped = [c.args[0].name for c in stop.await_args_list]
    assert QWEN38_FLASH_NEXT.name in stopped
    assert DEEPSEEK_V4_FLASH.name not in stopped


@pytest.mark.asyncio
async def test_stop_kills_both_ranks(mock_vram_monitor, mock_backend):
    orch = _orchestrator(mock_vram_monitor, mock_backend)
    worker_calls: list[str] = []

    async def _worker(host, command):
        worker_calls.append(command)
        return ""

    with patch.object(orch, "_run_on_worker", side_effect=_worker):
        await orch._stop_spark_serve(DEEPSEEK_V4_FLASH)

    # The launch tag is unmatchable (vllm is a shebang script, so the kernel
    # rewrites argv and `exec -a` is lost), so teardown keys off the model id.
    worker_script = _decode_script(worker_calls[-1])
    head_script = _decode_script(mock_backend.exec_in_workload.call_args[0][1])
    for script in (head_script, worker_script):
        assert f"vllm serve {DEEPSEEK_V4_FLASH.hf_model_id}" in script
        assert "spark-serve-" not in script
        # The weights are held by the renamed TP workers, not by the process
        # that owns the command line — kill the group, then sweep orphans.
        assert "kill -TERM -" in script
        assert "VLLM::" in script


@pytest.mark.asyncio
async def test_unreachable_worker_does_not_abort_teardown(
    mock_vram_monitor, mock_backend
):
    """A dead worker must not block releasing the head node."""
    orch = _orchestrator(mock_vram_monitor, mock_backend)

    with patch.object(
        orch, "_run_on_worker", side_effect=RuntimeError("host unreachable")
    ):
        await orch._stop_spark_serve(DEEPSEEK_V4_FLASH)

    assert mock_backend.exec_in_workload.await_count >= 1


# ──────────────────────────────────────────────────────
#  Preflight
# ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_refused_when_local_container_is_missing(
    mock_vram_monitor, mock_backend
):
    orch = _orchestrator(mock_vram_monitor, mock_backend)
    mock_backend.get_workload = AsyncMock(return_value=None)

    with pytest.raises(RuntimeError, match="run-recipe.sh"):
        await orch._verify_spark_cluster_ready(DEEPSEEK_V4_FLASH)


@pytest.mark.asyncio
async def test_start_refused_when_worker_container_is_down(
    mock_vram_monitor, mock_backend
):
    orch = _orchestrator(mock_vram_monitor, mock_backend)
    mock_backend.get_workload = AsyncMock(
        return_value=_running(SPARK_CLUSTER_CONTAINER)
    )

    with patch.object(orch, "_run_on_worker", AsyncMock(return_value="false")):
        with pytest.raises(RuntimeError, match="not running on worker"):
            await orch._verify_spark_cluster_ready(DEEPSEEK_V4_FLASH)


@pytest.mark.asyncio
async def test_preflight_passes_when_both_nodes_are_up(
    mock_vram_monitor, mock_backend
):
    orch = _orchestrator(mock_vram_monitor, mock_backend)
    mock_backend.get_workload = AsyncMock(
        return_value=_running(SPARK_CLUSTER_CONTAINER)
    )

    with patch.object(orch, "_run_on_worker", AsyncMock(return_value="true")):
        await orch._verify_spark_cluster_ready(DEEPSEEK_V4_FLASH)


# ──────────────────────────────────────────────────────
#  The cluster container is never a disposable workload
# ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_orphan_cleanup_never_removes_the_cluster_container(
    mock_vram_monitor, mock_backend
):
    orch = _orchestrator(mock_vram_monitor, mock_backend)
    mock_backend.get_workload = AsyncMock(return_value=None)

    with patch.object(orch, "_stop_spark_serve", AsyncMock()) as stop:
        await orch.cleanup_orphaned_containers()

    removed = [c.args[0] for c in mock_backend.remove_workload.await_args_list]
    assert SPARK_CLUSTER_CONTAINER not in removed
    stop.assert_awaited()


@pytest.mark.asyncio
async def test_adoption_identifies_the_model_not_just_the_container(
    mock_vram_monitor, mock_backend
):
    """All three spark_cluster models exec into the same 'vllm_node', so a
    running container says nothing about which model is serving. When only
    Qwen3.8's port answers, adoption must pick qwen38 — it used to key off the
    container name alone and always returned the first profile, 'deepseek',
    leaving the Gateway proxying to a dead port."""
    orch = _orchestrator(mock_vram_monitor, mock_backend)

    async def _health(name, port, engine="vllm"):
        return engine == "spark_cluster" and port == QWEN38_FLASH_NEXT.port

    with patch.object(orch, "_check_vllm_health", side_effect=_health):
        with patch.object(orch, "_stop_exec_model", AsyncMock()):
            mock_backend.get_workload = AsyncMock(return_value=None)
            adopted = await orch.detect_and_adopt_running_profile()

    assert adopted == "qwen38"


@pytest.mark.asyncio
async def test_adoption_stops_process_instead_of_removing_container(
    mock_vram_monitor, mock_backend
):
    """A healthy DeepSeek that is not part of the adopted profile must lose its
    serve process, not its container — removing it would kill the cluster."""
    orch = _orchestrator(mock_vram_monitor, mock_backend)

    # Only the Spark model answers /health, and no profile is a full match
    # unless it is DeepSeek's, so force a mismatch by reporting an extra
    # Docker-managed container as running too.
    async def _health(name, port, engine="vllm"):
        return engine == "spark_cluster"

    with patch.object(orch, "_check_vllm_health", side_effect=_health):
        with patch.object(orch, "_stop_exec_model", AsyncMock()) as stop:
            mock_backend.get_workload = AsyncMock(return_value=None)
            adopted = await orch.detect_and_adopt_running_profile()

    assert adopted == "deepseek"
    removed = [c.args[0] for c in mock_backend.remove_workload.await_args_list]
    assert SPARK_CLUSTER_CONTAINER not in removed
    assert orch._container_states[SPARK_CLUSTER_CONTAINER] == ContainerState.READY
    stop.assert_not_awaited()


# ──────────────────────────────────────────────────────
#  Addressing
# ──────────────────────────────────────────────────────

def test_spark_cluster_resolves_to_head_node_ip():
    """The serve process uses host networking on the head node, so the bridge
    DNS name of the container does not reach it."""
    from gateway.config import RAY_HEAD_HOST
    from gateway.proxy import InferenceProxy

    assert (
        InferenceProxy._default_resolve(SPARK_CLUSTER_CONTAINER, "spark_cluster")
        == RAY_HEAD_HOST
    )
