"""
Backend abstraction for container/workload orchestration.

Defines the protocol that both Docker and Kubernetes backends implement,
allowing the orchestrator to manage inference workloads without coupling
to a specific container runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class WorkloadInfo:
    """Runtime information about a managed workload (container or pod)."""
    name: str
    status: str                  # "running", "paused", "exited", "pending", etc.
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecResult:
    """Result of executing a command inside a workload."""
    exit_code: int
    output: str


@runtime_checkable
class OrchestrationBackend(Protocol):
    """Protocol for container/pod lifecycle management.

    Both Docker and Kubernetes backends implement this interface.
    The orchestrator delegates all runtime-specific operations here.
    """

    @property
    def supports_pause(self) -> bool:
        """Whether this backend supports pause/unpause (cgroup freeze).

        Docker supports this natively. Kubernetes does not — it must
        scale to 0 instead, which requires a full model reload on resume.
        """
        ...

    async def get_workload(self, name: str) -> WorkloadInfo | None:
        """Get info about a workload by name. Returns None if not found."""
        ...

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
        """Create and start a new workload."""
        ...

    async def stop_workload(self, name: str, timeout: int = 10) -> None:
        """Stop a running workload."""
        ...

    async def remove_workload(self, name: str) -> None:
        """Remove a workload entirely."""
        ...

    async def pause_workload(self, name: str) -> None:
        """Pause a workload (freeze processes). Only supported by Docker."""
        ...

    async def unpause_workload(self, name: str) -> None:
        """Unpause a previously paused workload."""
        ...

    async def exec_in_workload(
        self, name: str, command: list[str]
    ) -> ExecResult:
        """Execute a command inside a running workload."""
        ...

    async def list_workloads(
        self, label_filter: str | None = None
    ) -> list[WorkloadInfo]:
        """List workloads, optionally filtered by label/name pattern."""
        ...

    async def ensure_network(self, workload_name: str) -> None:
        """Ensure a workload is on the correct network. No-op for K8s."""
        ...

    def resolve_hostname(self, name: str, engine: str) -> str:
        """Resolve the hostname used to reach a workload's HTTP endpoint.

        Args:
            name: The workload/container name.
            engine: The model engine type ("vllm", "ray_vllm", "comfyui", "diffusers").

        Returns:
            Hostname or IP to use in HTTP URLs.
        """
        ...
