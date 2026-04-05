"""
Docker Container Lifecycle Orchestrator.
Manages START / STOP / PAUSE / UNPAUSE of inference containers.
Implements mutex (semaphore) to prevent VRAM collisions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import time
from typing import Optional

import docker
from docker.errors import NotFound, APIError
from docker.models.containers import Container

from gateway.config import (
    DOCKER_SOCKET,
    DOCKER_NETWORK,
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


class ContainerOrchestrator:
    """
    Manages the lifecycle of inference containers.
    Uses an asyncio.Lock (mutex) to serialize all swap operations
    and prevent memory collisions.
    """

    def __init__(self, vram_monitor: VRAMMonitor):
        self._client: docker.DockerClient = docker.DockerClient(
            base_url=DOCKER_SOCKET
        )
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
        """Remove all known inference containers on startup.

        For Docker-managed models: force-removes containers.
        For Ray vllm models: kills any lingering vllm serve processes inside
        the Ray head node container.
        """
        loop = asyncio.get_event_loop()
        removed = 0
        ray_cleaned = False

        for model in ALL_MODELS:
            if model.engine == "ray_vllm":
                if not ray_cleaned:
                    await self._kill_all_ray_vllm()
                    ray_cleaned = True
                continue

            name = model.container_name
            try:
                container = await loop.run_in_executor(
                    None, lambda n=name: self._client.containers.get(n)
                )
                await loop.run_in_executor(
                    None, lambda c=container: c.remove(force=True)
                )
                removed += 1
                logger.info("Removed orphaned container '%s'.", name)
            except NotFound:
                pass
            except APIError as exc:
                logger.warning("Could not remove orphaned '%s': %s", name, exc)

        if removed:
            logger.info("Orphan cleanup: removed %d container(s).", removed)
        else:
            logger.info("Orphan cleanup: no leftover containers found.")

    async def detect_and_adopt_running_profile(self) -> Optional[str]:
        """Detects which VRAM profile is already running and adopts it.

        Iterates over all known profiles and checks whether every model
        container in the profile exists in Docker (running or starting).
        If a complete match is found, the profile is adopted in-memory
        without restarting any containers; containers that don't belong
        to the matched profile are force-removed.

        Returns the adopted profile key, or None if no match was found.
        """
        loop = asyncio.get_event_loop()

        # Build a map of container_name → docker status for all known models
        running: dict[str, str] = {}  # name → status
        for model in ALL_MODELS:
            name = model.container_name
            if model.engine == "ray_vllm":
                # Check via HTTP health instead of Docker container
                healthy = await self._check_vllm_health(name, model.port, model.engine)
                if healthy:
                    running[name] = "running"
                continue
            try:
                container = await loop.run_in_executor(
                    None, lambda n=name: self._client.containers.get(n)
                )
                running[name] = container.status
            except NotFound:
                pass

        if not running:
            logger.info("Autodetect: no inference containers found.")
            return None

        # Find a profile whose containers are all present (running or starting)
        valid_statuses = {"running", "created", "restarting"}
        for profile_key, profile in PROFILES.items():
            all_models = profile.primary_models + profile.secondary_models
            required = {m.container_name for m in all_models}
            if required and required.issubset(running) and all(
                running[n] in valid_statuses for n in required
            ):
                # Adopt this profile — use the same key format as _profile_key()
                # so that subsequent switch_profile() calls match correctly
                self._active_profile = self._profile_key(profile)
                for model in all_models:
                    name = model.container_name
                    status = running[name]
                    self._container_states[name] = (
                        ContainerState.READY
                        if status == "running"
                        else ContainerState.STARTING
                    )
                    source = "Ray cluster" if model.engine == "ray_vllm" else "docker"
                    logger.info(
                        "Autodetect: adopted '%s' (%s, status: %s).",
                        name, source, status,
                    )

                # Remove containers that don't belong to this profile
                orphans = set(running) - required
                for name in orphans:
                    try:
                        container = await loop.run_in_executor(
                            None, lambda n=name: self._client.containers.get(n)
                        )
                        await loop.run_in_executor(
                            None, lambda c=container: c.remove(force=True)
                        )
                        logger.info("Autodetect: removed orphan '%s'.", name)
                    except (NotFound, APIError) as exc:
                        logger.warning("Autodetect: could not remove '%s': %s", name, exc)

                logger.info("Autodetect: adopted profile '%s'.", profile_key)
                self._persist_state()
                return profile_key

        # Containers exist but don't form a complete profile — treat as orphans
        logger.info(
            "Autodetect: found containers %s but no complete profile match.",
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

            try:
                logger.info(
                    "Starting swap → '%s' (strategy=%s)",
                    profile_key, strategy.value,
                )

                # Compute target container names to protect from teardown
                all_models = target_profile.primary_models + target_profile.secondary_models
                target_names = {m.container_name for m in all_models}

                # 1. Stop/pause containers from current profile (preserve targets)
                await self._teardown_current(strategy, preserve=target_names)

                # 2. Verify available VRAM (skip for Ray profiles — memory is
                #    distributed across nodes and not visible to local NVML)
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

                    # Wait for VRAM to actually be freed (up to 15s)
                    for _attempt in range(15):
                        await asyncio.sleep(1)
                        vram_report = await self._vram.query_gpus()
                        if self._vram.has_enough_vram(needed, vram_report):
                            logger.info(
                                "VRAM freed: %d MB available.",
                                vram_report.total_free_mb,
                            )
                            break
                    else:
                        logger.warning(
                            "VRAM still not fully freed after 15s "
                            "(%d MB free). Proceeding anyway...",
                            vram_report.total_free_mb,
                        )

                # 3. Start containers for the target profile
                for model_def in all_models:
                    await self._ensure_container_running(model_def, strategy)

                # 4. Wait for all to report healthy BEFORE claiming profile
                try:
                    await self._wait_all_ready(all_models)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Health-check timeout for profile '%s'. "
                        "Containers are running but models may still be loading.",
                        profile_key,
                    )
                    # Promote any still-STARTING containers to READY so that
                    # incoming requests don't trigger a redundant swap to the
                    # same profile.  The models are loading; re-swapping would
                    # only restart them and make things worse.
                    for m in all_models:
                        if self._container_states.get(m.container_name) == ContainerState.STARTING:
                            self._container_states[m.container_name] = ContainerState.READY
                            logger.info(
                                "Promoted '%s' to READY after timeout (still loading).",
                                m.container_name,
                            )

                # 5. Claim the profile only after health checks pass (or timeout)
                self._active_profile = profile_key
                self._persist_state()

                elapsed = time.time() - self._swap_start_time
                logger.info(
                    "Swap completed → '%s' in %.2fs", profile_key, elapsed
                )
                return True

            except Exception as exc:
                logger.error("Error in swap to '%s': %s", profile_key, exc, exc_info=True)
                return False
            finally:
                self._swap_in_progress = False

    # ────────────── Individual Containers ───────

    async def _ensure_container_running(
        self,
        model: ModelDefinition,
        strategy: SwapStrategy,
    ) -> None:
        """Ensures the model is running — either as a Docker container or
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

        logger.info("Starting container '%s'...", name)
        loop = asyncio.get_event_loop()

        try:
            container: Container = await loop.run_in_executor(
                None, lambda: self._client.containers.get(name)
            )

            status = container.status
            if status == "paused" and strategy == SwapStrategy.PAUSE_UNPAUSE:
                await loop.run_in_executor(None, container.unpause)
                logger.info("Container '%s' resumed (unpause).", name)
            elif status in ("exited", "created", "dead"):
                logger.info(
                    "Container '%s' in state '%s', removing to recreate fresh.",
                    name, status,
                )
                await loop.run_in_executor(
                    None, lambda c=container: c.remove(force=True)
                )
                await self._create_and_start(model)
                logger.info("Container '%s' recreated and started.", name)
            elif status == "running":
                await self._ensure_correct_network(container, loop)
                logger.info("Container '%s' already running.", name)
            else:
                await loop.run_in_executor(None, lambda: container.remove(force=True))
                await self._create_and_start(model)

        except NotFound:
            await self._create_and_start(model)

        # Leave as STARTING — _wait_all_ready will set READY after healthcheck
        self._container_states[name] = ContainerState.STARTING

    async def _create_and_start(self, model: ModelDefinition) -> None:
        """Creates and starts a new container for the model."""
        loop = asyncio.get_event_loop()

        env_vars = self._build_env(model)
        volumes = {**model.extra_volumes}
        if not model.hf_model_id:
            volumes[f"{MODELS_PATH}/{model.model_path}"] = {
                "bind": "/models",
                "mode": "ro",
            }

        cmd = self._build_cmd(model)

        # Override the image's built-in HEALTHCHECK to use the correct port.
        # vLLM images default to localhost:8000 but each model uses its own port.
        healthcheck = docker.types.Healthcheck(
            test=["CMD-SHELL", f"curl -f http://localhost:{model.port}/health || exit 1"],
            interval=30_000_000_000,    # 30s in nanoseconds
            timeout=10_000_000_000,     # 10s
            start_period=300_000_000_000,  # 5min — model loading takes time
            retries=3,
        )

        try:
            container = await loop.run_in_executor(
                None,
                lambda: self._client.containers.run(
                    image=model.container_image,
                    name=model.container_name,
                    command=cmd,
                    environment=env_vars,
                    volumes=volumes,
                    ports={f"{model.port}/tcp": model.port},
                    network=DOCKER_NETWORK,
                    detach=True,
                    dns=["8.8.8.8", "8.8.4.4"],
                    device_requests=[
                        docker.types.DeviceRequest(
                            count=-1,  # All GPUs
                            capabilities=[["gpu"]],
                        )
                    ],
                    restart_policy={"Name": "unless-stopped"},
                    shm_size="16g",
                    healthcheck=healthcheck,
                ),
            )
            logger.info("Container '%s' created and started.", model.container_name)
        except APIError as exc:
            logger.error("Docker error creating '%s': %s", model.container_name, exc)
            self._container_states[model.container_name] = ContainerState.ERROR
            raise

    async def _teardown_current(
        self,
        strategy: SwapStrategy,
        preserve: set[str] | None = None,
    ) -> None:
        """Stops or pauses all containers from the current profile.

        Args:
            strategy: How to tear down (stop or pause).
            preserve: Container names to SKIP (target profile containers).
        """
        loop = asyncio.get_event_loop()
        preserve = preserve or set()

        for name, state in list(self._container_states.items()):
            if name in preserve:
                continue
            if state not in (ContainerState.READY, ContainerState.STARTING):
                continue

            self._container_states[name] = ContainerState.STOPPING

            # Find the model definition to check engine type
            model_def = next((m for m in ALL_MODELS if m.container_name == name), None)
            if model_def and model_def.engine == "ray_vllm":
                await self._stop_ray_vllm(model_def)
                self._container_states[name] = ContainerState.STOPPED
                continue

            try:
                container = await loop.run_in_executor(
                    None, lambda n=name: self._client.containers.get(n)
                )

                if strategy == SwapStrategy.PAUSE_UNPAUSE:
                    if container.status == "running":
                        await loop.run_in_executor(None, container.pause)
                        self._container_states[name] = ContainerState.PAUSED
                        logger.info("Container '%s' paused.", name)
                else:
                    await loop.run_in_executor(
                        None, lambda c=container: c.stop(timeout=10)
                    )
                    self._container_states[name] = ContainerState.STOPPED
                    logger.info("Container '%s' stopped.", name)

            except NotFound:
                self._container_states[name] = ContainerState.STOPPED
            except APIError as exc:
                logger.error("Error stopping '%s': %s", name, exc)

    async def _wait_all_ready(
        self,
        models: list[ModelDefinition],
        timeout: int = SWAP_TIMEOUT_S,
    ) -> None:
        """Waits for all containers to report health OK."""
        start = time.time()
        loop = asyncio.get_event_loop()

        for model in models:
            name = model.container_name
            attempt = 0
            last_log: float = 0.0
            while True:
                elapsed = time.time() - start
                if elapsed > timeout:
                    raise asyncio.TimeoutError(
                        f"Timeout esperando a '{name}' (>{timeout}s)"
                    )

                attempt += 1
                now = time.time()
                should_log = (now - last_log) >= RETRY_LOG_INTERVAL_S

                if model.engine == "ray_vllm":
                    # No Docker container to check — poll HTTP health directly.
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
                    try:
                        container = await loop.run_in_executor(
                            None, lambda n=name: self._client.containers.get(n)
                        )
                        if container.status == "running":
                            if model.engine in ("vllm", "comfyui", "diffusers"):
                                healthy = await self._check_vllm_health(
                                    model.container_name, model.port, model.engine
                                )
                            else:
                                healthy = True

                            if healthy:
                                self._container_states[name] = ContainerState.READY
                                logger.info("Container '%s' READY (%.0fs).", name, elapsed)
                                break
                            elif should_log:
                                logger.info(
                                    "Waiting for '%s' healthcheck... (%.0fs elapsed)",
                                    name, elapsed,
                                )
                        elif should_log:
                            logger.info(
                                "Waiting for '%s' to start (status=%s, %.0fs elapsed)",
                                name, container.status, elapsed,
                            )
                    except (NotFound, APIError):
                        if should_log:
                            logger.info(
                                "Waiting for '%s' container to appear... (%.0fs elapsed)",
                                name, elapsed,
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
        """HTTP healthcheck for a vLLM backend.

        Ray vllm models are reached via the Ray head node IP (host network).
        Docker-managed models are reached via container hostname (bridge network).
        """
        import aiohttp

        host = RAY_HEAD_HOST if engine == "ray_vllm" else container_name
        url = f"http://{host}:{port}/health"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3)
            ) as session:
                async with session.get(url) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def _ensure_correct_network(
        self,
        container: Container,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Ensure a running container is connected to DOCKER_NETWORK.

        After a `docker compose up --build`, the compose-managed network
        may have been recreated.  Orphaned containers from a previous
        session can remain attached to the *old* network, making them
        unreachable by hostname from the gateway.
        """
        networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
        if DOCKER_NETWORK in networks:
            return  # Already on the correct network

        logger.warning(
            "Container '%s' is NOT on network '%s' (current: %s). Reconnecting...",
            container.name,
            DOCKER_NETWORK,
            list(networks.keys()),
        )
        try:
            net = await loop.run_in_executor(
                None, lambda: self._client.networks.get(DOCKER_NETWORK)
            )
            await loop.run_in_executor(
                None, lambda: net.connect(container)
            )
            logger.info(
                "Container '%s' reconnected to '%s'.",
                container.name,
                DOCKER_NETWORK,
            )
        except (NotFound, APIError) as exc:
            logger.error(
                "Could not reconnect '%s' to '%s': %s",
                container.name,
                DOCKER_NETWORK,
                exc,
            )

    # ──────────── Container Stop/Remove ────────────────

    async def stop_container(self, container_name: str) -> None:
        """Stops a specific container."""
        loop = asyncio.get_event_loop()
        try:
            container = await loop.run_in_executor(
                None, lambda: self._client.containers.get(container_name)
            )
            await loop.run_in_executor(None, lambda: container.stop(timeout=10))
            self._container_states[container_name] = ContainerState.STOPPED
        except (NotFound, APIError) as exc:
            logger.warning("Could not stop '%s': %s", container_name, exc)

    async def remove_container(self, container_name: str) -> None:
        """Removes a specific container."""
        loop = asyncio.get_event_loop()
        try:
            container = await loop.run_in_executor(
                None, lambda: self._client.containers.get(container_name)
            )
            await loop.run_in_executor(
                None, lambda: container.remove(force=True)
            )
            self._container_states.pop(container_name, None)
        except (NotFound, APIError) as exc:
            logger.warning("Could not remove '%s': %s", container_name, exc)

    # ──────────── Helpers ──────────────────────────────

    @staticmethod
    def _build_env(model: ModelDefinition) -> dict[str, str]:
        """Builds environment variables for the container."""
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
        """Builds the container startup command."""
        if model.engine in ("vllm", "ray_vllm"):
            # For ray_vllm, model weights are loaded from HuggingFace cache
            # (already present in the Ray container from previous runs).
            # Local model_path is not used — always use hf_model_id or /models.
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
        ray_container = await self._get_ray_container()
        if not ray_container:
            return False
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: ray_container.exec_run(
                    ["bash", "-c", "ray status 2>&1 | grep -c ' node_'"],
                    detach=False,
                ),
            )
            count_str = result.output.decode("utf-8", errors="ignore").strip()
            return int(count_str) >= required_nodes
        except Exception as exc:
            logger.warning("Could not verify Ray cluster node count: %s", exc)
            return False

    async def _get_ray_container(self) -> Optional[Container]:
        """Find the running Ray head node container (node-<digits>)."""
        loop = asyncio.get_event_loop()
        try:
            containers = await loop.run_in_executor(
                None,
                lambda: self._client.containers.list(filters={"status": "running"}),
            )
            for c in containers:
                if re.match(r"^ray-node-head$", c.name):
                    return c
            logger.warning("No Ray head container found. Run ray-cluster/reset_ray_node.sh --head first.")
            return None
        except Exception as exc:
            logger.error("Error finding Ray container: %s", exc)
            return None

    async def _start_ray_vllm(self, model: ModelDefinition) -> None:
        """Starts vllm serve inside the Ray head node container as a background process."""
        ray_container = await self._get_ray_container()
        if not ray_container:
            raise RuntimeError(
                f"Cannot start '{model.name}': Ray head container not found."
            )

        # Stop any existing vllm process occupying this port
        await self._stop_ray_vllm(model)

        cmd_list = self._build_cmd(model)
        # For ray_vllm the Docker entrypoint is not involved — prepend 'vllm serve'
        # unless cmd_prefix already contains it (e.g. custom images).
        if not model.cmd_prefix:
            cmd_list = ["vllm", "serve"] + list(cmd_list)
        cmd_str = " ".join(shlex.quote(str(p)) for p in cmd_list)

        log_file = f"/tmp/vllm_{model.container_name}.log"
        # Use exec -a to set a unique process title (argv[0]) so pkill can
        # reliably identify and kill this specific model's process without
        # relying on PID files (which are lost if the container restarts).
        bg_cmd = (
            f"nohup bash -c 'exec -a vllm-serve-{model.container_name} {cmd_str}' "
            f"> {log_file} 2>&1 &"
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: ray_container.exec_run(["bash", "-c", bg_cmd], detach=False),
        )
        logger.info(
            "Started vllm serve for '%s' in Ray container '%s' (port %d, TP=%d).",
            model.name, ray_container.name, model.port, model.tensor_parallel_size,
        )

    async def _stop_ray_vllm(self, model: ModelDefinition) -> None:
        """Kills the vllm serve process for this model inside the Ray container."""
        ray_container = await self._get_ray_container()
        if not ray_container:
            logger.warning("Cannot stop '%s': Ray container not found.", model.name)
            return

        # Kill by process title set via exec -a in _start_ray_vllm.
        # Fallback to port-based pkill covers processes started before this change.
        kill_cmd = (
            f"pkill -TERM -f 'vllm-serve-{model.container_name}' 2>/dev/null; "
            f"sleep 2; "
            f"pkill -KILL -f 'vllm-serve-{model.container_name}' 2>/dev/null; "
            f"pkill -f 'vllm.*--port {model.port}' 2>/dev/null; "
            f"true"
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: ray_container.exec_run(["bash", "-c", kill_cmd], detach=False),
        )
        logger.info("Stopped vllm serve for '%s' in Ray container.", model.name)

    async def _kill_all_ray_vllm(self) -> None:
        """Kills all vllm serve processes inside the Ray container (used at startup cleanup)."""
        ray_container = await self._get_ray_container()
        if not ray_container:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: ray_container.exec_run(
                ["bash", "-c", "pkill -f 'vllm-serve-vllm-' 2>/dev/null; pkill -f 'vllm serve' 2>/dev/null; true"],
                detach=False,
            ),
        )
        logger.info("Killed all vllm serve processes in Ray container.")
