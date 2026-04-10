"""
Kubernetes backend for workload orchestration.

Implements the OrchestrationBackend protocol using the official
Kubernetes Python client. Inference models are pre-deployed as
Deployments with replicas=0 (dormant); the gateway scales them
to 1 (active) or 0 (stopped) as needed.

Ray vllm models are managed via exec into the Ray head pod,
identical to the Docker backend but using the K8s exec API.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from gateway.backends.base import ExecResult, WorkloadInfo
from gateway.config import RAY_HEAD_HOST

logger = logging.getLogger("gateway.backends.kubernetes")

# Lazy-imported at first use to avoid ImportError when kubernetes
# package is not installed (Docker-only deployments).
_k8s_client = None
_k8s_config = None
_k8s_stream = None


def _ensure_k8s_imports():
    """Lazily import kubernetes SDK."""
    global _k8s_client, _k8s_config, _k8s_stream
    if _k8s_client is None:
        from kubernetes import client as c, config as cfg
        from kubernetes.stream import stream as s
        _k8s_client = c
        _k8s_config = cfg
        _k8s_stream = s


class KubernetesBackend:
    """OrchestrationBackend implementation using the Kubernetes API.

    Design decisions:
    - Inference models are pre-deployed as Deployments with replicas=0.
      The gateway scales them to 1 (start) or 0 (stop).
    - Pause is not supported; scale to 0 is used instead.
    - Ray head pod is discovered via label selector and exec'd into.
    - Service DNS provides hostname resolution (same namespace).
    """

    def __init__(self) -> None:
        _ensure_k8s_imports()

        # Load config: in-cluster when running as a pod, kubeconfig otherwise
        try:
            _k8s_config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config.")
        except _k8s_config.ConfigException:
            _k8s_config.load_kube_config()
            logger.info("Loaded kubeconfig from default location.")

        self._core_v1 = _k8s_client.CoreV1Api()
        self._apps_v1 = _k8s_client.AppsV1Api()
        self._namespace = os.getenv("K8S_NAMESPACE", "blackwell")

    @property
    def supports_pause(self) -> bool:
        return False

    async def get_workload(self, name: str) -> WorkloadInfo | None:
        """Get pod info by deployment name (app label)."""
        loop = asyncio.get_event_loop()
        try:
            pods = await loop.run_in_executor(
                None,
                lambda: self._core_v1.list_namespaced_pod(
                    namespace=self._namespace,
                    label_selector=f"app={name}",
                ),
            )
            for pod in pods.items:
                return WorkloadInfo(
                    name=pod.metadata.name,
                    status=self._pod_status(pod),
                    attrs={"phase": pod.status.phase if pod.status else "Unknown"},
                )
            return None
        except Exception as exc:
            logger.warning("K8s error getting workload '%s': %s", name, exc)
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
    ) -> None:
        """Scale the pre-existing Deployment to 1 replica.

        The Deployment and its pod spec (image, command, volumes, GPU,
        probes) are defined in K8s YAML manifests, not here. This method
        only toggles the replica count.

        If the Deployment doesn't exist yet (first run or manual cleanup),
        it creates one dynamically as a fallback.
        """
        loop = asyncio.get_event_loop()
        try:
            # Try scaling existing Deployment
            await loop.run_in_executor(
                None,
                lambda: self._apps_v1.patch_namespaced_deployment_scale(
                    name=name,
                    namespace=self._namespace,
                    body={"spec": {"replicas": 1}},
                ),
            )
            logger.info("Scaled Deployment '%s' to 1 replica.", name)
        except Exception as exc:
            # Deployment doesn't exist — create it dynamically
            logger.info(
                "Deployment '%s' not found (%s), creating dynamically...", name, exc
            )
            await self._create_deployment(
                name, image, command, environment, volumes, port, shm_size
            )

    async def _create_deployment(
        self,
        name: str,
        image: str,
        command: str | list[str] | None,
        environment: dict[str, str],
        volumes: dict[str, Any],
        port: int,
        shm_size: str,
    ) -> None:
        """Dynamically create a Deployment + Service for an inference model."""
        loop = asyncio.get_event_loop()

        # Build env vars
        env_list = [
            _k8s_client.V1EnvVar(name=k, value=v)
            for k, v in environment.items()
        ]

        # Build volume mounts and volumes
        k8s_volumes = []
        k8s_volume_mounts = []

        # Shared memory (replaces --shm-size)
        k8s_volumes.append(_k8s_client.V1Volume(
            name="dshm",
            empty_dir=_k8s_client.V1EmptyDirVolumeSource(
                medium="Memory",
                size_limit=shm_size.replace("g", "Gi"),
            ),
        ))
        k8s_volume_mounts.append(_k8s_client.V1VolumeMount(
            name="dshm", mount_path="/dev/shm",
        ))

        # Model volumes from host paths
        for idx, (host_path, mount_info) in enumerate(volumes.items()):
            vol_name = f"vol-{idx}"
            bind_path = mount_info.get("bind", host_path) if isinstance(mount_info, dict) else host_path
            k8s_volumes.append(_k8s_client.V1Volume(
                name=vol_name,
                host_path=_k8s_client.V1HostPathVolumeSource(
                    path=host_path, type="Directory",
                ),
            ))
            k8s_volume_mounts.append(_k8s_client.V1VolumeMount(
                name=vol_name,
                mount_path=bind_path,
                read_only=mount_info.get("mode", "rw") == "ro" if isinstance(mount_info, dict) else False,
            ))

        # Build command
        cmd = None
        args = None
        if isinstance(command, list):
            cmd = [command[0]] if command else None
            args = command[1:] if len(command) > 1 else None
        elif isinstance(command, str):
            cmd = ["/bin/bash", "-c"]
            args = [command]

        container = _k8s_client.V1Container(
            name=name,
            image=image,
            command=cmd,
            args=args,
            env=env_list,
            ports=[_k8s_client.V1ContainerPort(container_port=port)],
            volume_mounts=k8s_volume_mounts,
            resources=_k8s_client.V1ResourceRequirements(
                limits={"nvidia.com/gpu": "1"},
            ),
            startup_probe=_k8s_client.V1Probe(
                http_get=_k8s_client.V1HTTPGetAction(path="/health", port=port),
                initial_delay_seconds=30,
                period_seconds=10,
                failure_threshold=30,  # 5 min
                timeout_seconds=5,
            ),
            readiness_probe=_k8s_client.V1Probe(
                http_get=_k8s_client.V1HTTPGetAction(path="/health", port=port),
                period_seconds=30,
                timeout_seconds=10,
                failure_threshold=3,
            ),
            liveness_probe=_k8s_client.V1Probe(
                http_get=_k8s_client.V1HTTPGetAction(path="/health", port=port),
                initial_delay_seconds=300,
                period_seconds=30,
                timeout_seconds=10,
                failure_threshold=3,
            ),
        )

        deployment = _k8s_client.V1Deployment(
            metadata=_k8s_client.V1ObjectMeta(
                name=name,
                namespace=self._namespace,
                labels={"app": name},
            ),
            spec=_k8s_client.V1DeploymentSpec(
                replicas=1,
                selector=_k8s_client.V1LabelSelector(
                    match_labels={"app": name},
                ),
                template=_k8s_client.V1PodTemplateSpec(
                    metadata=_k8s_client.V1ObjectMeta(
                        labels={"app": name},
                    ),
                    spec=_k8s_client.V1PodSpec(
                        containers=[container],
                        volumes=k8s_volumes,
                    ),
                ),
            ),
        )

        await loop.run_in_executor(
            None,
            lambda: self._apps_v1.create_namespaced_deployment(
                namespace=self._namespace, body=deployment,
            ),
        )
        logger.info("Created Deployment '%s' in namespace '%s'.", name, self._namespace)

        # Create a ClusterIP Service for DNS resolution
        service = _k8s_client.V1Service(
            metadata=_k8s_client.V1ObjectMeta(
                name=name,
                namespace=self._namespace,
            ),
            spec=_k8s_client.V1ServiceSpec(
                selector={"app": name},
                ports=[_k8s_client.V1ServicePort(port=port, target_port=port)],
                type="ClusterIP",
            ),
        )
        try:
            await loop.run_in_executor(
                None,
                lambda: self._core_v1.create_namespaced_service(
                    namespace=self._namespace, body=service,
                ),
            )
            logger.info("Created Service '%s'.", name)
        except Exception:
            # Service may already exist
            pass

    async def stop_workload(self, name: str, timeout: int = 10) -> None:
        """Scale Deployment to 0 replicas."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._apps_v1.patch_namespaced_deployment_scale(
                    name=name,
                    namespace=self._namespace,
                    body={"spec": {"replicas": 0}},
                ),
            )
            logger.info("Scaled Deployment '%s' to 0 replicas.", name)
        except Exception as exc:
            logger.warning("Could not scale down '%s': %s", name, exc)

    async def remove_workload(self, name: str) -> None:
        """Scale to 0. We keep the Deployment object for future use."""
        await self.stop_workload(name)

    async def pause_workload(self, name: str) -> None:
        """K8s does not support pause. Scale to 0 instead."""
        logger.warning("Pause not supported in K8s. Scaling '%s' to 0.", name)
        await self.stop_workload(name)

    async def unpause_workload(self, name: str) -> None:
        """Scale back to 1."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._apps_v1.patch_namespaced_deployment_scale(
                    name=name,
                    namespace=self._namespace,
                    body={"spec": {"replicas": 1}},
                ),
            )
            logger.info("Scaled Deployment '%s' to 1 replica (unpause).", name)
        except Exception as exc:
            logger.warning("Could not scale up '%s': %s", name, exc)

    async def exec_in_workload(
        self, name: str, command: list[str]
    ) -> ExecResult:
        """Execute a command in a running pod.

        Args:
            name: Pod name or label-based name. If it looks like a deployment
                  name (no random suffix), we find the actual pod first.
        """
        loop = asyncio.get_event_loop()
        pod_name = await self._resolve_pod_name(name)
        if not pod_name:
            return ExecResult(exit_code=1, output=f"No running pod found for '{name}'")

        try:
            result = await loop.run_in_executor(
                None,
                lambda: _k8s_stream(
                    self._core_v1.connect_get_namespaced_pod_exec,
                    pod_name,
                    self._namespace,
                    command=command,
                    stderr=True,
                    stdout=True,
                    stdin=False,
                    tty=False,
                ),
            )
            return ExecResult(exit_code=0, output=result or "")
        except Exception as exc:
            return ExecResult(exit_code=1, output=str(exc))

    async def list_workloads(
        self, label_filter: str | None = None
    ) -> list[WorkloadInfo]:
        """List running pods, optionally filtered by name pattern."""
        loop = asyncio.get_event_loop()
        try:
            # If label_filter looks like a specific name, use it as a label selector
            label_selector = f"app={label_filter}" if label_filter else ""
            pods = await loop.run_in_executor(
                None,
                lambda: self._core_v1.list_namespaced_pod(
                    namespace=self._namespace,
                    label_selector=label_selector,
                    field_selector="status.phase=Running",
                ),
            )
            return [
                WorkloadInfo(
                    name=pod.metadata.name,
                    status=self._pod_status(pod),
                    attrs={"phase": pod.status.phase if pod.status else "Unknown"},
                )
                for pod in pods.items
            ]
        except Exception as exc:
            logger.error("Error listing pods: %s", exc)
            return []

    async def ensure_network(self, workload_name: str) -> None:
        """No-op for Kubernetes — Services handle DNS automatically."""
        pass

    def resolve_hostname(self, name: str, engine: str) -> str:
        """K8s Service DNS for local pods, Ray head IP for ray_vllm."""
        if engine == "ray_vllm":
            return RAY_HEAD_HOST
        # Within the same namespace, short name resolves via CoreDNS
        return name

    # ──────────── Internal Helpers ────────────────

    async def _resolve_pod_name(self, name: str) -> str | None:
        """Find the actual running pod name for a given workload name."""
        loop = asyncio.get_event_loop()
        try:
            pods = await loop.run_in_executor(
                None,
                lambda: self._core_v1.list_namespaced_pod(
                    namespace=self._namespace,
                    label_selector=f"app={name}",
                    field_selector="status.phase=Running",
                ),
            )
            if pods.items:
                return pods.items[0].metadata.name
        except Exception as exc:
            logger.warning("Could not resolve pod for '%s': %s", name, exc)
        return None

    @staticmethod
    def _pod_status(pod) -> str:
        """Map K8s pod phase/conditions to a Docker-like status string."""
        if not pod.status:
            return "unknown"
        phase = pod.status.phase or "Unknown"
        if phase == "Running":
            # Check if all containers are ready
            if pod.status.conditions:
                for c in pod.status.conditions:
                    if c.type == "Ready" and c.status == "True":
                        return "running"
            return "running"
        elif phase == "Pending":
            return "created"
        elif phase == "Succeeded":
            return "exited"
        elif phase == "Failed":
            return "dead"
        return phase.lower()
