"""
Docker Container Lifecycle Orchestrator.
Manages START / STOP / PAUSE / UNPAUSE of inference containers.
Implements mutex (semaphore) to prevent VRAM collisions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import docker
from docker.errors import NotFound, APIError
from docker.models.containers import Container

from gateway.config import (
    DOCKER_SOCKET,
    DOCKER_NETWORK,
    MODELS_PATH,
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
    def container_states(self) -> dict[str, ContainerState]:
        return dict(self._container_states)

    def is_model_ready(self, container_name: str) -> bool:
        """Returns True only if the container is in READY state."""
        return self._container_states.get(container_name) == ContainerState.READY

    # ────────────── Startup Cleanup ────────────────

    async def cleanup_orphaned_containers(self) -> None:
        """Remove all known inference containers on startup.

        Prevents VRAM contention from containers left running by a
        previous gateway session (Docker restart-policy keeps them alive).
        Containers are **removed** (not just stopped) so that stale
        Docker-network bindings are discarded.  ``_ensure_container_running``
        will create fresh containers on the current network.
        """
        loop = asyncio.get_event_loop()
        removed = 0

        for model in ALL_MODELS:
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
                    logger.info(
                        "Autodetect: adopted '%s' (docker status: %s).",
                        name, status,
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

                # 2. Verify available VRAM (use fresh query, not stale cache)
                vram_report = await self._vram.query_gpus()
                needed = target_profile.total_vram_required_mb
                if not self._vram.has_enough_vram(needed, vram_report):
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

                # 5. Claim the profile only after health checks pass (or timeout)
                self._active_profile = profile_key

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
        """Ensures the model container is running."""
        name = model.container_name
        self._container_states[name] = ContainerState.STARTING
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
                # Remove and recreate so the container gets the current
                # Docker network.  Restarting a stopped container would
                # keep its old (possibly stale) network binding.
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
                # Ensure container is on the correct Docker network
                await self._ensure_correct_network(container, loop)
                logger.info("Container '%s' already running.", name)
            else:
                # Unexpected state, recreate
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
        volumes = {
            f"{MODELS_PATH}/{model.model_path}": {
                "bind": "/models",
                "mode": "ro",
            },
            **model.extra_volumes,
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
                try:
                    container = await loop.run_in_executor(
                        None, lambda n=name: self._client.containers.get(n)
                    )
                    if container.status == "running":
                        # Attempt HTTP healthcheck if vLLM
                        if model.engine == "vllm":
                            healthy = await self._check_vllm_health(
                                model.container_name, model.port
                            )
                        else:
                            healthy = True  # ComfyUI/Diffusers: running = ready

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

    async def _check_vllm_health(self, container_name: str, port: int) -> bool:
        """HTTP healthcheck for a vLLM container via Docker network."""
        import aiohttp

        # Use container name as hostname (same Docker network)
        url = f"http://{container_name}:{port}/health"
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
        if model.engine == "vllm":
            # Standard vllm/vllm-openai images have 'python -m vllm...' as ENTRYPOINT,
            # so the model path is the first positional arg.
            # Custom images whose ENTRYPOINT is a generic shell script (exec "$@")
            # must set cmd_prefix=["vllm", "serve"] in their ModelDefinition.
            cmd_parts = [
                *model.cmd_prefix,
                "/models",
                "--port", str(model.port),
                "--served-model-name", model.name,
                "--tensor-parallel-size", str(model.tensor_parallel_size),
                "--max-model-len", str(model.max_model_len),
                "--kv-cache-dtype", model.kv_cache_dtype,
                "--enable-prefix-caching",
                "--trust-remote-code",
            ]
            # Only pass --quantization if not 'auto' (let vLLM detect from config.json)
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
