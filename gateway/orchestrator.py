"""
Container / Pod Lifecycle Orchestrator.
Manages START / STOP / PAUSE / UNPAUSE of inference workloads.
Implements mutex (semaphore) to prevent VRAM collisions.

Backend-agnostic: delegates all runtime operations to an
OrchestrationBackend (Docker or Kubernetes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import time
from typing import Optional

from gateway.backends.base import OrchestrationBackend
from gateway.config import (
    MODELS_PATH,
    RAY_HEAD_HOST,
    SWAP_TIMEOUT_S,
    RETRY_LOG_INTERVAL_S,
    ALL_MODELS,
    PROFILES,
    ModelDefinition,
    VRAMProfile,
    get_swap_strategy,
)
from gateway.schemas import ContainerState, ProfileMode, SwapStrategy
from gateway.vram_monitor import VRAMMonitor

logger = logging.getLogger("gateway.orchestrator")

STATE_FILE = os.environ.get("GATEWAY_STATE_FILE", "/data/gateway_state.json")

# Name pattern for the Ray head workload
RAY_HEAD_NAME = "ray-node-head"


class ContainerOrchestrator:
    """
    Manages the lifecycle of inference workloads.
    Uses an asyncio.Lock (mutex) to serialize all swap operations
    and prevent memory collisions.
    """

    def __init__(self, vram_monitor: VRAMMonitor, backend: OrchestrationBackend):
        self._backend = backend
        self._vram = vram_monitor
        self._swap_lock = asyncio.Lock()  # Global swap mutex
        self._container_states: dict[str, ContainerState] = {}
        self._active_profile: Optional[str] = None
        self._swap_in_progress = False
        self._swap_start_time: float = 0.0

    # ──────────────── State persistence ────────────────

    def _persist_state(self) -> None:
        """Write active profile to disk so restarts can resume it."""
        if not self._active_profile:
            return
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump({"active_profile": self._active_profile}, f)
        except Exception as exc:
            logger.warning("Could not persist gateway state: %s", exc)

    @staticmethod
    def load_persisted_profile() -> Optional[str]:
        """Return the last persisted profile key, or None if unavailable."""
        try:
            with open(STATE_FILE) as f:
                return json.load(f).get("active_profile")
        except Exception:
            return None

    # ──────────────── Propiedades ──────────────────────

    @property
    def is_swapping(self) -> bool:
        return self._swap_in_progress

    @property
    def active_profile(self) -> Optional[str]:
        return self._active_profile

    @property
    def swap_elapsed(self) -> float:
        if not self._swap_in_progress:
            return 0.0
        return time.time() - self._swap_start_time

    @property
    def active_vram_profile(self) -> Optional[VRAMProfile]:
        """Return the active VRAMProfile object, or None."""
        if not self._active_profile:
            return None
        for profile in PROFILES.values():
            if self._profile_key(profile) == self._active_profile:
                return profile
        return None

    @property
    def container_states(self) -> dict[str, ContainerState]:
        return dict(self._container_states)

    def is_model_ready(self, container_name: str) -> bool:
        """Returns True only if the container is in READY state."""
        return self._container_states.get(container_name) == ContainerState.READY

    # ────────────── Startup Cleanup ────────────────

    async def cleanup_orphaned_containers(self) -> None:
        """Remove all known inference workloads on startup.

        For Docker-managed models: force-removes containers.
        For Ray vllm models: kills any lingering vllm serve processes inside
        the Ray head node.
        """
        removed = 0
        ray_cleaned = False

        for model in ALL_MODELS:
            if model.engine == "ray_vllm":
                if not ray_cleaned:
                    await self._kill_all_ray_vllm()
                    ray_cleaned = True
                continue

            name = model.container_name
            workload = await self._backend.get_workload(name)
            if workload is not None:
                await self._backend.remove_workload(name)
                removed += 1
                logger.info("Removed orphaned workload '%s'.", name)

        if removed:
            logger.info("Orphan cleanup: removed %d workload(s).", removed)
        else:
            logger.info("Orphan cleanup: no leftover workloads found.")

    async def detect_and_adopt_running_profile(self) -> Optional[str]:
        """Detects which VRAM profile is already running and adopts it.

        Iterates over all known profiles and checks whether every model
        in the profile is currently running. If a complete match is found,
        the profile is adopted in-memory without restarting anything.

        Returns the adopted profile key, or None if no match was found.
        """
        # Build a map of container_name → status for all known models
        running: dict[str, str] = {}
        for model in ALL_MODELS:
            name = model.container_name
            if model.engine == "ray_vllm":
                healthy = await self._check_vllm_health(name, model.port, model.engine)
                if healthy:
                    running[name] = "running"
                continue
            workload = await self._backend.get_workload(name)
            if workload is not None:
                running[name] = workload.status

        if not running:
            logger.info("Autodetect: no inference workloads found.")
            return None

        # Find a profile whose workloads are all present (running or starting)
        valid_statuses = {"running", "created", "restarting", "pending"}
        for profile_key, profile in PROFILES.items():
            all_models = profile.primary_models + profile.secondary_models
            required = {m.container_name for m in all_models}
            if required and required.issubset(running) and all(
                running[n] in valid_statuses for n in required
            ):
                self._active_profile = self._profile_key(profile)
                for model in all_models:
                    name = model.container_name
                    status = running[name]
                    self._container_states[name] = (
                        ContainerState.READY
                        if status == "running"
                        else ContainerState.STARTING
                    )
                    source = "Ray cluster" if model.engine == "ray_vllm" else "backend"
                    logger.info(
                        "Autodetect: adopted '%s' (%s, status: %s).",
                        name, source, status,
                    )

                # Remove workloads that don't belong to this profile
                orphans = set(running) - required
                for name in orphans:
                    await self._backend.remove_workload(name)
                    logger.info("Autodetect: removed orphan '%s'.", name)

                logger.info("Autodetect: adopted profile '%s'.", profile_key)
                self._persist_state()
                return profile_key

        logger.info(
            "Autodetect: found workloads %s but no complete profile match.",
            list(running),
        )
        return None

    # ────────────── Core: Profile Swap ──────────

    async def switch_profile(
        self,
        target_profile: VRAMProfile,
        force: bool = False,
    ) -> bool:
        """
        Switches to the specified VRAM profile.
        Serialized with mutex to prevent collisions.
        Returns True if the switch was successful.
        """
        profile_key = self._profile_key(target_profile)

        if self._active_profile == profile_key and not force:
            logger.info("Profile '%s' already active, skipping.", profile_key)
            return True

        async with self._swap_lock:
            self._swap_in_progress = True
            self._swap_start_time = time.time()
            strategy = get_swap_strategy()

            # If the backend doesn't support pause, force STOP_START
            if not self._backend.supports_pause and strategy == SwapStrategy.PAUSE_UNPAUSE:
                strategy = SwapStrategy.STOP_START

            try:
                logger.info(
                    "Starting swap → '%s' (strategy=%s)",
                    profile_key, strategy.value,
                )

                all_models = target_profile.primary_models + target_profile.secondary_models
                target_names = {m.container_name for m in all_models}

                # 1. Stop/pause workloads from current profile (preserve targets)
                await self._teardown_current(strategy, preserve=target_names)

                # 2. Verify available VRAM
                vram_report = await self._vram.query_gpus()
                needed = target_profile.total_vram_required_mb
                if target_profile.skip_vram_check:
                    logger.info("Skipping VRAM check for Ray cluster profile '%s'.", profile_key)
                elif not self._vram.has_enough_vram(needed, vram_report):
                    logger.warning(
                        "Insufficient VRAM (%d MB free, %d MB required). "
                        "Forcing deep cleanup...",
                        vram_report.total_free_mb, needed,
                    )
                    await self._teardown_current(SwapStrategy.STOP_START, preserve=target_names)

                    for _attempt in range(15):
                        await asyncio.sleep(1)
                        vram_report = await self._vram.query_gpus()
                        if self._vram.has_enough_vram(needed, vram_report):
                            logger.info("VRAM freed: %d MB available.", vram_report.total_free_mb)
                            break
                    else:
                        logger.warning(
                            "VRAM still not fully freed after 15s (%d MB free). Proceeding anyway...",
                            vram_report.total_free_mb,
                        )

                # 3. Start workloads for the target profile
                for model_def in all_models:
                    await self._ensure_container_running(model_def, strategy)

                # 4. Wait for all to report healthy
                try:
                    await self._wait_all_ready(all_models)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Health-check timeout for profile '%s'. "
                        "Workloads are running but models may still be loading.",
                        profile_key,
                    )
                    for m in all_models:
                        if self._container_states.get(m.container_name) == ContainerState.STARTING:
                            self._container_states[m.container_name] = ContainerState.READY
                            logger.info(
                                "Promoted '%s' to READY after timeout (still loading).",
                                m.container_name,
                            )

                # 5. Claim the profile
                self._active_profile = profile_key
                self._persist_state()

                elapsed = time.time() - self._swap_start_time
                logger.info("Swap completed → '%s' in %.2fs", profile_key, elapsed)
                return True

            except Exception as exc:
                logger.error("Error in swap to '%s': %s", profile_key, exc, exc_info=True)
                return False
            finally:
                self._swap_in_progress = False

    # ────────────── Individual Workloads ───────

    async def _ensure_container_running(
        self,
        model: ModelDefinition,
        strategy: SwapStrategy,
    ) -> None:
        """Ensures the model is running — either as a workload or
        as a vllm serve process inside the Ray cluster."""
        name = model.container_name
        self._container_states[name] = ContainerState.STARTING

        if model.engine == "ray_vllm":
            logger.info("Starting Ray vllm model '%s' on port %d...", model.name, model.port)
            if model.tensor_parallel_size >= 2:
                cluster_ready = await self._verify_ray_cluster_ready(required_nodes=2)
                if not cluster_ready:
                    self._container_states[name] = ContainerState.ERROR
                    raise RuntimeError(
                        f"Ray cluster needs 2 nodes for TP=2 model '{model.name}' "
                        f"but worker is disconnected. Run ray-cluster/reset_ray_node.sh --worker on Node 2."
                    )
            await self._start_ray_vllm(model)
            self._container_states[name] = ContainerState.STARTING
            return

        logger.info("Starting workload '%s'...", name)

        workload = await self._backend.get_workload(name)
        if workload is not None:
            status = workload.status
            if status == "paused" and strategy == SwapStrategy.PAUSE_UNPAUSE:
                await self._backend.unpause_workload(name)
                logger.info("Workload '%s' resumed (unpause).", name)
            elif status in ("exited", "created", "dead"):
                logger.info(
                    "Workload '%s' in state '%s', removing to recreate fresh.",
                    name, status,
                )
                await self._backend.remove_workload(name)
                await self._create_and_start(model)
                logger.info("Workload '%s' recreated and started.", name)
            elif status == "running":
                await self._backend.ensure_network(name)
                logger.info("Workload '%s' already running.", name)
            else:
                await self._backend.remove_workload(name)
                await self._create_and_start(model)
        else:
            await self._create_and_start(model)

        self._container_states[name] = ContainerState.STARTING

    async def _create_and_start(self, model: ModelDefinition) -> None:
        """Creates and starts a new workload for the model."""
        env_vars = self._build_env(model)
        volumes = {**model.extra_volumes}
        if not model.hf_model_id:
            volumes[f"{MODELS_PATH}/{model.model_path}"] = {
                "bind": "/models",
                "mode": "ro",
            }

        cmd = self._build_cmd(model)

        try:
            await self._backend.create_and_start(
                name=model.container_name,
                image=model.container_image,
                command=cmd,
                environment=env_vars,
                volumes=volumes,
                port=model.port,
                shm_size="16g",
            )
            logger.info("Workload '%s' created and started.", model.container_name)
        except Exception as exc:
            logger.error("Error creating '%s': %s", model.container_name, exc)
            self._container_states[model.container_name] = ContainerState.ERROR
            raise

    async def _teardown_current(
        self,
        strategy: SwapStrategy,
        preserve: set[str] | None = None,
    ) -> None:
        """Stops or pauses all workloads from the current profile.

        Args:
            strategy: How to tear down (stop or pause).
            preserve: Workload names to SKIP (target profile workloads).
        """
        preserve = preserve or set()

        for name, state in list(self._container_states.items()):
            if name in preserve:
                continue
            if state not in (ContainerState.READY, ContainerState.STARTING):
                continue

            self._container_states[name] = ContainerState.STOPPING

            # Check if this is a Ray vllm model
            model_def = next((m for m in ALL_MODELS if m.container_name == name), None)
            if model_def and model_def.engine == "ray_vllm":
                await self._stop_ray_vllm(model_def)
                self._container_states[name] = ContainerState.STOPPED
                continue

            try:
                if strategy == SwapStrategy.PAUSE_UNPAUSE and self._backend.supports_pause:
                    await self._backend.pause_workload(name)
                    self._container_states[name] = ContainerState.PAUSED
                    logger.info("Workload '%s' paused.", name)
                else:
                    await self._backend.stop_workload(name, timeout=10)
                    self._container_states[name] = ContainerState.STOPPED
                    logger.info("Workload '%s' stopped.", name)
            except Exception as exc:
                logger.error("Error stopping '%s': %s", name, exc)

    async def _wait_all_ready(
        self,
        models: list[ModelDefinition],
        timeout: int = SWAP_TIMEOUT_S,
    ) -> None:
        """Waits for all workloads to report health OK."""
        start = time.time()

        for model in models:
            name = model.container_name
            last_log: float = 0.0
            while True:
                elapsed = time.time() - start
                if elapsed > timeout:
                    raise asyncio.TimeoutError(
                        f"Timeout waiting for '{name}' (>{timeout}s)"
                    )

                now = time.time()
                should_log = (now - last_log) >= RETRY_LOG_INTERVAL_S

                if model.engine == "ray_vllm":
                    healthy = await self._check_vllm_health(name, model.port, model.engine)
                    if healthy:
                        self._container_states[name] = ContainerState.READY
                        logger.info("Ray vllm '%s' READY (%.0fs).", name, elapsed)
                        break
                    elif should_log:
                        logger.info(
                            "Waiting for Ray vllm '%s' healthcheck... (%.0fs elapsed)",
                            name, elapsed,
                        )
                else:
                    workload = await self._backend.get_workload(name)
                    if workload and workload.status == "running":
                        healthy = await self._check_vllm_health(
                            model.container_name, model.port, model.engine
                        )
                        if healthy:
                            self._container_states[name] = ContainerState.READY
                            logger.info("Workload '%s' READY (%.0fs).", name, elapsed)
                            break
                        elif should_log:
                            logger.info(
                                "Waiting for '%s' healthcheck... (%.0fs elapsed)",
                                name, elapsed,
                            )
                    elif should_log:
                        status = workload.status if workload else "not found"
                        logger.info(
                            "Waiting for '%s' to start (status=%s, %.0fs elapsed)",
                            name, status, elapsed,
                        )

                if should_log:
                    last_log = now
                await asyncio.sleep(1)

    async def _check_vllm_health(
        self,
        container_name: str,
        port: int,
        engine: str = "vllm",
    ) -> bool:
        """HTTP healthcheck for an inference backend."""
        import aiohttp

        host = self._backend.resolve_hostname(container_name, engine)
        url = f"http://{host}:{port}/health"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3)
            ) as session:
                async with session.get(url) as resp:
                    return resp.status == 200
        except Exception:
            return False

    # ──────────── Workload Stop/Remove ────────────────

    async def stop_container(self, container_name: str) -> None:
        """Stops a specific workload."""
        await self._backend.stop_workload(container_name, timeout=10)
        self._container_states[container_name] = ContainerState.STOPPED

    async def remove_container(self, container_name: str) -> None:
        """Removes a specific workload."""
        await self._backend.remove_workload(container_name)
        self._container_states.pop(container_name, None)

    # ──────────── Helpers ──────────────────────────────

    @staticmethod
    def _build_env(model: ModelDefinition) -> dict[str, str]:
        """Builds environment variables for the workload."""
        env = {
            "MODEL_NAME": model.name,
            "NVIDIA_VISIBLE_DEVICES": "all",
            **model.extra_env,
        }
        if model.engine == "vllm":
            env["VLLM_PORT"] = str(model.port)
        return env

    @staticmethod
    def _build_cmd(model: ModelDefinition) -> str | list[str] | None:
        """Builds the workload startup command."""
        if model.engine in ("vllm", "ray_vllm"):
            model_arg = model.hf_model_id if model.hf_model_id else "/models"
            cmd_parts = [
                *model.cmd_prefix,
                model_arg,
                "--port", str(model.port),
                "--served-model-name", model.name,
                "--tensor-parallel-size", str(model.tensor_parallel_size),
                "--max-model-len", str(model.max_model_len),
                "--kv-cache-dtype", model.kv_cache_dtype,
                "--enable-prefix-caching",
                "--trust-remote-code",
            ]
            if model.quantization.lower() != "auto":
                cmd_parts.extend(["--quantization", model.quantization.lower()])
            for key, val in model.extra_args.items():
                if isinstance(val, bool):
                    if val:
                        cmd_parts.append(key)
                else:
                    cmd_parts.extend([key, str(val)])
            return cmd_parts

        # ComfyUI / Diffusers: use the image's entrypoint
        return None

    @staticmethod
    def _profile_key(profile: VRAMProfile) -> str:
        """Generates a unique key for a profile."""
        model_names = sorted(
            m.name for m in profile.primary_models + profile.secondary_models
        )
        return f"{profile.mode.value}:{'|'.join(model_names)}"

    # ──────────── Ray Cluster Management ──────────────

    async def _verify_ray_cluster_ready(self, required_nodes: int = 2) -> bool:
        """Returns True if the Ray cluster has at least `required_nodes` active nodes."""
        ray_workload = await self._find_ray_head()
        if not ray_workload:
            return False
        try:
            result = await self._backend.exec_in_workload(
                ray_workload,
                ["bash", "-c", "ray status 2>&1 | grep -c ' node_'"],
            )
            count_str = result.output.strip()
            return int(count_str) >= required_nodes
        except Exception as exc:
            logger.warning("Could not verify Ray cluster node count: %s", exc)
            return False

    async def _find_ray_head(self) -> Optional[str]:
        """Find the running Ray head workload name."""
        workloads = await self._backend.list_workloads(label_filter=RAY_HEAD_NAME)
        for w in workloads:
            if w.name == RAY_HEAD_NAME or RAY_HEAD_NAME in w.name:
                return w.name
        logger.warning("No Ray head workload found. Run ray-cluster/reset_ray_node.sh --head first.")
        return None

    async def _start_ray_vllm(self, model: ModelDefinition) -> None:
        """Starts vllm serve inside the Ray head as a background process."""
        ray_name = await self._find_ray_head()
        if not ray_name:
            raise RuntimeError(
                f"Cannot start '{model.name}': Ray head workload not found."
            )

        # Stop any existing vllm process occupying this port
        await self._stop_ray_vllm(model)

        cmd_list = self._build_cmd(model)
        if not model.cmd_prefix:
            cmd_list = ["vllm", "serve"] + list(cmd_list)
        cmd_str = " ".join(shlex.quote(str(p)) for p in cmd_list)

        log_file = f"/tmp/vllm_{model.container_name}.log"
        # The cmd_str is embedded inside bash -c '...', so any single quotes
        # produced by shlex.quote must be escaped for the outer shell layer.
        # Replace ' with '\'' (end quote, escaped quote, start quote).
        inner_cmd = cmd_str.replace("'", "'\\''")
        bg_cmd = (
            f"nohup bash -c 'exec -a vllm-serve-{model.container_name} {inner_cmd}' "
            f"> {log_file} 2>&1 &"
        )

        await self._backend.exec_in_workload(
            ray_name,
            ["bash", "-c", bg_cmd],
        )
        logger.info(
            "Started vllm serve for '%s' in Ray workload '%s' (port %d, TP=%d).",
            model.name, ray_name, model.port, model.tensor_parallel_size,
        )

    async def _stop_ray_vllm(self, model: ModelDefinition) -> None:
        """Kills the vllm serve process for this model inside the Ray head."""
        ray_name = await self._find_ray_head()
        if not ray_name:
            logger.warning("Cannot stop '%s': Ray head not found.", model.name)
            return

        kill_cmd = (
            f"pkill -TERM -f 'vllm-serve-{model.container_name}' 2>/dev/null; "
            f"sleep 2; "
            f"pkill -KILL -f 'vllm-serve-{model.container_name}' 2>/dev/null; "
            f"pkill -f 'vllm.*--port {model.port}' 2>/dev/null; "
            f"true"
        )

        await self._backend.exec_in_workload(
            ray_name,
            ["bash", "-c", kill_cmd],
        )
        logger.info("Stopped vllm serve for '%s' in Ray workload.", model.name)

        # Clean up Ray placement groups that held GPUs for this model.
        # Without this, TP=2 models leave placement groups alive even after
        # the vllm process is killed, blocking GPU allocation for the next model.
        await self._cleanup_ray_placement_groups(ray_name)

    async def _kill_all_ray_vllm(self) -> None:
        """Kills all vllm serve processes inside the Ray head (used at startup cleanup)."""
        ray_name = await self._find_ray_head()
        if not ray_name:
            return
        await self._backend.exec_in_workload(
            ray_name,
            ["bash", "-c", "pkill -f 'vllm-serve-vllm-' 2>/dev/null; pkill -f 'vllm serve' 2>/dev/null; true"],
        )
        logger.info("Killed all vllm serve processes in Ray workload.")

        # Also clean up any stale placement groups
        await self._cleanup_ray_placement_groups(ray_name)

    async def _cleanup_ray_placement_groups(self, ray_name: str) -> None:
        """Remove all Ray placement groups to free reserved GPU resources.

        When a TP=2 vllm serve process is killed, Ray keeps its placement
        group alive (state=CREATED) with GPUs reserved. This prevents the
        next model from allocating GPUs. We must explicitly remove them.
        """
        cleanup_cmd = [
            "python3", "-c",
            # Use RAY_ADDRESS so ray.init() connects as a client driver
            # instead of registering a new worker node.  The driver node
            # still appears briefly in `ray status` but deregisters on
            # ray.shutdown() without leaving a phantom.
            "import os, ray\n"
            "os.environ.setdefault('RAY_ADDRESS', 'auto')\n"
            "ray.init()\n"
            "pgs = ray.util.placement_group_table()\n"
            "for pg_id in pgs:\n"
            "    try:\n"
            "        from ray.util.placement_group import PlacementGroup\n"
            "        pg = PlacementGroup(ray.PlacementGroupID(bytes.fromhex(pg_id)))\n"
            "        ray.util.remove_placement_group(pg)\n"
            "    except Exception:\n"
            "        pass\n"
            "print(f'Removed {len(pgs)} placement group(s)')\n"
            "ray.shutdown()\n"
        ]
        try:
            result = await self._backend.exec_in_workload(ray_name, cleanup_cmd)
            if result.output.strip():
                logger.info("Ray placement group cleanup: %s", result.output.strip())
        except Exception as exc:
            logger.warning("Could not clean up Ray placement groups: %s", exc)
