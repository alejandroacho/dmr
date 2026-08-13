"""
Docker backend for container orchestration.

Wraps the Docker SDK (docker-py) to implement the OrchestrationBackend
protocol. This is the original backend — extracted from orchestrator.py
to allow swapping with a Kubernetes backend.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import docker
from docker.errors import APIError, NotFound

from gateway.backends.base import ExecResult, WorkloadInfo
from gateway.config import DOCKER_NETWORK, DOCKER_SOCKET, EXEC_ENGINES, RAY_HEAD_HOST

logger = logging.getLogger("gateway.backends.docker")


class DockerBackend:
    """OrchestrationBackend implementation using the Docker SDK."""

    def __init__(self) -> None:
        self._client: docker.DockerClient = docker.DockerClient(
            base_url=DOCKER_SOCKET
        )

    @property
    def supports_pause(self) -> bool:
        return True

    async def get_workload(self, name: str) -> WorkloadInfo | None:
        loop = asyncio.get_event_loop()
        try:
            container = await loop.run_in_executor(
                None, lambda: self._client.containers.get(name)
            )
            return WorkloadInfo(
                name=container.name,
                status=container.status,
                attrs=container.attrs or {},
            )
        except NotFound:
            return None
        except APIError as exc:
            logger.warning("Docker error getting '%s': %s", name, exc)
            return None

    async def create_and_start(
        self,
        name: str,
        image: str,
        command: str | list[str] | None,
        environment: dict[str, str],
        volumes: dict[str, Any],
        port: int,
        shm_size: str = "16g",
        entrypoint: list[str] | None = None,
    ) -> None:
        loop = asyncio.get_event_loop()

        healthcheck = docker.types.Healthcheck(
            test=["CMD-SHELL", f"curl -f http://localhost:{port}/health || exit 1"],
            interval=30_000_000_000,       # 30s
            timeout=10_000_000_000,        # 10s
            start_period=300_000_000_000,  # 5min
            retries=3,
        )

        run_kwargs: dict[str, Any] = dict(
            image=image,
            name=name,
            command=command,
            environment=environment,
            volumes=volumes,
            ports={f"{port}/tcp": port},
            network=DOCKER_NETWORK,
            detach=True,
            dns=["8.8.8.8", "8.8.4.4"],
            device_requests=[
                docker.types.DeviceRequest(
                    count=-1,
                    capabilities=[["gpu"]],
                )
            ],
            restart_policy={"Name": "unless-stopped"},
            shm_size=shm_size,
            healthcheck=healthcheck,
        )
        if entrypoint is not None:
            run_kwargs["entrypoint"] = entrypoint

        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.containers.run(**run_kwargs),
            )
            logger.info("Container '%s' created and started.", name)
        except APIError as exc:
            logger.error("Docker error creating '%s': %s", name, exc)
            raise

    async def stop_workload(self, name: str, timeout: int = 10) -> None:
        loop = asyncio.get_event_loop()
        try:
            container = await loop.run_in_executor(
                None, lambda: self._client.containers.get(name)
            )
            await loop.run_in_executor(
                None, lambda: container.stop(timeout=timeout)
            )
            logger.info("Container '%s' stopped.", name)
        except NotFound:
            pass
        except APIError as exc:
            logger.warning("Could not stop '%s': %s", name, exc)

    async def remove_workload(self, name: str) -> None:
        loop = asyncio.get_event_loop()
        try:
            container = await loop.run_in_executor(
                None, lambda: self._client.containers.get(name)
            )
            await loop.run_in_executor(
                None, lambda: container.remove(force=True)
            )
            logger.info("Container '%s' removed.", name)
        except NotFound:
            pass
        except APIError as exc:
            logger.warning("Could not remove '%s': %s", name, exc)

    async def pause_workload(self, name: str) -> None:
        loop = asyncio.get_event_loop()
        try:
            container = await loop.run_in_executor(
                None, lambda: self._client.containers.get(name)
            )
            if container.status == "running":
                await loop.run_in_executor(None, container.pause)
                logger.info("Container '%s' paused.", name)
        except (NotFound, APIError) as exc:
            logger.warning("Could not pause '%s': %s", name, exc)

    async def unpause_workload(self, name: str) -> None:
        loop = asyncio.get_event_loop()
        try:
            container = await loop.run_in_executor(
                None, lambda: self._client.containers.get(name)
            )
            await loop.run_in_executor(None, container.unpause)
            logger.info("Container '%s' resumed (unpause).", name)
        except (NotFound, APIError) as exc:
            logger.warning("Could not unpause '%s': %s", name, exc)

    async def exec_in_workload(
        self, name: str, command: list[str]
    ) -> ExecResult:
        loop = asyncio.get_event_loop()
        try:
            container = await loop.run_in_executor(
                None, lambda: self._client.containers.get(name)
            )
            result = await loop.run_in_executor(
                None,
                lambda: container.exec_run(command, detach=False),
            )
            output = result.output.decode("utf-8", errors="ignore") if result.output else ""
            return ExecResult(exit_code=result.exit_code or 0, output=output)
        except NotFound:
            return ExecResult(exit_code=1, output=f"Container '{name}' not found")
        except APIError as exc:
            return ExecResult(exit_code=1, output=str(exc))

    async def list_workloads(
        self, label_filter: str | None = None
    ) -> list[WorkloadInfo]:
        """List running containers.

        Args:
            label_filter: For Docker, this is used as a name pattern match
                          (e.g. "ray-node-head") since Docker doesn't have
                          K8s-style labels. If None, lists all running containers.
        """
        loop = asyncio.get_event_loop()
        try:
            containers = await loop.run_in_executor(
                None,
                lambda: self._client.containers.list(filters={"status": "running"}),
            )
            results = []
            for c in containers:
                if label_filter and label_filter not in c.name:
                    continue
                results.append(WorkloadInfo(
                    name=c.name,
                    status=c.status,
                    attrs=c.attrs or {},
                ))
            return results
        except Exception as exc:
            logger.error("Error listing containers: %s", exc)
            return []

    async def ensure_network(self, workload_name: str) -> None:
        """Ensure a container is connected to DOCKER_NETWORK."""
        loop = asyncio.get_event_loop()
        try:
            container = await loop.run_in_executor(
                None, lambda: self._client.containers.get(workload_name)
            )
            networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
            if DOCKER_NETWORK in networks:
                return

            logger.warning(
                "Container '%s' not on network '%s' (current: %s). Reconnecting...",
                workload_name, DOCKER_NETWORK, list(networks.keys()),
            )
            net = await loop.run_in_executor(
                None, lambda: self._client.networks.get(DOCKER_NETWORK)
            )
            await loop.run_in_executor(None, lambda: net.connect(container))
            logger.info("Container '%s' reconnected to '%s'.", workload_name, DOCKER_NETWORK)
        except (NotFound, APIError) as exc:
            logger.error("Could not ensure network for '%s': %s", workload_name, exc)

    def resolve_hostname(self, name: str, engine: str) -> str:
        """Docker bridge DNS for local containers, head-node IP for exec engines.

        ray_vllm and spark_cluster backends listen on the head node's host
        network, so they are not reachable through the bridge DNS name.
        """
        if engine in EXEC_ENGINES:
            return RAY_HEAD_HOST
        return name
