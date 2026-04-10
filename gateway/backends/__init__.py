"""
Backend factory for container orchestration.

Selects Docker or Kubernetes backend based on the ORCHESTRATION_BACKEND
environment variable. Dependencies are imported lazily so that a Docker
deployment never needs `pip install kubernetes` and vice versa.
"""

from __future__ import annotations

import os

from gateway.backends.base import ExecResult, OrchestrationBackend, WorkloadInfo

__all__ = [
    "ExecResult",
    "OrchestrationBackend",
    "WorkloadInfo",
    "create_backend",
]

ORCHESTRATION_BACKEND: str = os.getenv("ORCHESTRATION_BACKEND", "docker")


def create_backend() -> OrchestrationBackend:
    """Create the orchestration backend based on configuration.

    Returns a DockerBackend or KubernetesBackend instance.
    """
    backend_type = ORCHESTRATION_BACKEND.lower()

    if backend_type == "docker":
        from gateway.backends.docker_backend import DockerBackend
        return DockerBackend()

    if backend_type == "kubernetes":
        from gateway.backends.kubernetes_backend import KubernetesBackend
        return KubernetesBackend()

    raise ValueError(
        f"Unknown ORCHESTRATION_BACKEND={backend_type!r}. "
        f"Expected 'docker' or 'kubernetes'."
    )
