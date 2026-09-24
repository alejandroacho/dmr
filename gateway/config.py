"""
Central configuration for the Smart Gateway.
Defines VRAM profiles, available models, and system parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from gateway.schemas import (
    ContainerState,
    ModelSlot,
    ProfileMode,
    SwapStrategy,
)


# ─────────────────── Environment Variables ──────────────────────

GATEWAY_HOST: str = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT: int = int(os.getenv("GATEWAY_PORT", "8000"))

# IP of the Ray head node — used to reach vllm serve running inside the Ray cluster
RAY_HEAD_HOST: str = os.getenv("RAY_HEAD_HOST", "192.168.200.12")

# ─── Spark cluster (spark_vllm_docker) ───
# Container that eugr/spark-vllm-docker's launch-cluster.sh leaves running on
# every node. Its foreground process is `sleep infinity`, so vllm serve runs as
# an exec'd child: the Gateway starts/stops that process, never the container.
# Bring the containers up out-of-band with:
#   cd ~/spark-vllm-docker && HF_HOME=~/hf-cache ./run-recipe.sh <recipe> -d
SPARK_CLUSTER_CONTAINER: str = os.getenv("SPARK_CLUSTER_CONTAINER", "vllm_node")

# Worker nodes of the Spark cluster (rank >= 1), comma-separated.
# Multi-node vLLM needs one `vllm serve --headless` per worker, and those live
# on another machine's Docker daemon, so they are reached over SSH.
SPARK_WORKER_HOSTS: list[str] = [
    h.strip()
    for h in os.getenv("SPARK_WORKER_HOSTS", "192.168.200.13").split(",")
    if h.strip()
]

# SSH identity used to reach the worker nodes. Mounted read-only into the
# Gateway container; grants docker access on the workers, so keep it scoped.
SPARK_SSH_USER: str = os.getenv("SPARK_SSH_USER", "alejandroacho")
SPARK_SSH_KEY: str = os.getenv("SPARK_SSH_KEY", "/ssh/id_spark")

# Port used to coordinate the multi-node vLLM group (matches the launcher's
# MASTER_PORT in ~/spark-vllm-docker/.env).
SPARK_MASTER_PORT: int = int(os.getenv("SPARK_MASTER_PORT", "29501"))

# Engines whose backend is a process exec'd inside an already-running
# container rather than a workload the Gateway itself creates. For these the
# Gateway manages the *process*; removing or recreating the container is never
# correct.
EXEC_ENGINES: tuple[str, ...] = ("ray_vllm", "spark_cluster")

# Public base URL for generated assets (images, videos).
# Must be reachable from clients like OpenWebUI.
GATEWAY_PUBLIC_URL: str = os.getenv("GATEWAY_PUBLIC_URL", "http://192.168.1.125:8000")

# Local path where model weights reside
MODELS_PATH: str = os.getenv("MODELS_PATH", "/home/alejandroacho/Models")

# Host /proc/meminfo, bind-mounted into the container. On unified-memory
# systems (GB10) this is the real memory budget: NVML answers "Not Supported"
# for device memory, and its per-process accounting cannot see the inference
# processes, which live in another container's PID namespace.
HOST_MEMINFO_PATH: str = os.getenv("HOST_MEMINFO_PATH", "/host/meminfo")


def _detect_system_ram_gb(default: int = 512) -> int:
    """Total RAM in GiB, read from the host when the mount is available.

    Preferred over a hardcoded figure: SYSTEM_RAM_GB caps the VRAM budget, so
    an over-estimate (128 on a 121 GiB box) silently inflates the free margin.
    """
    for path in (HOST_MEMINFO_PATH, "/proc/meminfo"):
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // (1024 * 1024)
        except (OSError, ValueError, IndexError):
            continue
    return default


# Total system RAM; determines swap strategy and caps the VRAM budget.
SYSTEM_RAM_GB: int = int(os.getenv("SYSTEM_RAM_GB", "0")) or _detect_system_ram_gb()

# Maximum timeout for swaps before aborting (seconds)
# GB10 can take ~4 min to load large models (weights + KV cache + warmup)
SWAP_TIMEOUT_S: int = int(os.getenv("SWAP_TIMEOUT_S", "600"))

# nvidia-smi polling interval (seconds)
VRAM_POLL_INTERVAL_S: float = float(os.getenv("VRAM_POLL_INTERVAL_S", "2.0"))

# Safety VRAM margin to always keep free (MB)
VRAM_SAFETY_MARGIN_MB: int = int(os.getenv("VRAM_SAFETY_MARGIN_MB", "4096"))

# ── Media node (node 3) ──
# Dedicated GB10 that serves MiniMax-H3 video+audio and nothing else. It is a
# separate machine with its own 124 GB unified pool, so it never participates in
# this Gateway's VRAM profiles or container swapping.
# MEDIA_NODE_IP is the single place the node's address is configured (see .env);
# MEDIA_NODE_HOST still overrides it for the rare case of addressing the adapter
# and the asset URLs differently. Currently on WiFi — switch MEDIA_NODE_IP to the
# wired 192.168.1.x address once cabled.
MEDIA_NODE_IP: str = os.getenv("MEDIA_NODE_IP", "192.168.8.147")
MEDIA_NODE_HOST: str = os.getenv("MEDIA_NODE_HOST", MEDIA_NODE_IP)
MEDIA_NODE_PORT: int = int(os.getenv("MEDIA_NODE_PORT", "8010"))

# Docker socket
DOCKER_SOCKET: str = os.getenv("DOCKER_SOCKET", "unix:///var/run/docker.sock")

# Docker network for inference containers
DOCKER_NETWORK: str = os.getenv("DOCKER_NETWORK", "blackwell_net")

# Retry-After header for 503 during swaps (seconds)
RETRY_AFTER_SECONDS: int = int(os.getenv("RETRY_AFTER_SECONDS", "5"))

# How often to emit retry/wait log lines (seconds); avoids log spam during long swaps
RETRY_LOG_INTERVAL_S: int = int(os.getenv("RETRY_LOG_INTERVAL_S", "10"))

# Maximum queued requests during a swap
MAX_QUEUE_SIZE: int = int(os.getenv("MAX_QUEUE_SIZE", "200"))

# Long Polling mode vs immediate 503
LONG_POLLING_ENABLED: bool = os.getenv("LONG_POLLING_ENABLED", "true").lower() == "true"

# Maximum Long Polling timeout (seconds)
LONG_POLLING_TIMEOUT_S: int = int(os.getenv("LONG_POLLING_TIMEOUT_S", "600"))

# Orchestration backend: "docker" (default) or "kubernetes"
ORCHESTRATION_BACKEND: str = os.getenv("ORCHESTRATION_BACKEND", "docker")

# Kubernetes namespace (only used when ORCHESTRATION_BACKEND=kubernetes)
K8S_NAMESPACE: str = os.getenv("K8S_NAMESPACE", "blackwell")


# ─────────────────── Model Definitions ───────────────────────

@dataclass
class ModelDefinition:
    """Metadata for a model available in the cluster."""
    name: str
    container_image: str           # Docker image
    container_name: str            # Container name
    vram_required_mb: int
    port: int
    quantization: str = "Q8_0"
    tensor_parallel_size: int = 2
    # Either a token budget or "auto" to let vLLM derive it from the checkpoint.
    max_model_len: int | str = 128000
    kv_cache_dtype: str = "fp8"
    model_path: str = ""           # Path within MODELS_PATH (local)
    hf_model_id: str = ""          # HuggingFace model ID (overrides model_path)
    engine: str = "vllm"           # vllm | ray_vllm | spark_cluster | comfyui | diffusers
    # For EXEC_ENGINES: container to exec the serve process into. Empty means
    # "the Ray head", resolved at runtime.
    exec_container: str = ""
    # For spark_cluster: number of serve processes to launch. >1 selects vLLM's
    # native multi-node mode (rank 0 here + one --headless rank per worker);
    # 1 means a single process that shards the model some other way.
    cluster_nodes: int = 1
    # For spark_cluster: the model shards through Ray instead of the native
    # multi-node mode, so a Ray cluster must already span the containers.
    requires_ray_cluster: bool = False
    extra_args: dict[str, Any] = field(default_factory=dict)
    # Prefix injected before model path in the container command.
    # Needed when the image ENTRYPOINT is a generic shell (e.g. entrypoint.sh
    # that does exec "$@") instead of the vllm binary itself.
    cmd_prefix: list[str] = field(default_factory=list)
    # Extra volumes to mount in addition to the model volume.
    # Format: {host_path: {"bind": container_path, "mode": "ro"|"rw"}}
    extra_volumes: dict[str, Any] = field(default_factory=dict)
    # Extra environment variables to inject into the container.
    extra_env: dict[str, str] = field(default_factory=dict)
    # Override the image ENTRYPOINT (e.g. ["vllm", "serve"] for images
    # whose default entrypoint is a shell like /bin/bash -c).
    container_entrypoint: list[str] | None = None
    # Remote node hostname/IP. Empty means the model runs as a local container
    # reachable by name on DOCKER_NETWORK. When set, the model lives on another
    # machine: the Gateway proxies to it but never manages its lifecycle.
    host: str = ""

    @property
    def required_ray_nodes(self) -> int:
        """Ray nodes this model needs to serve, or 0 if it does not use Ray.

        Two different topologies rely on Ray: `ray_vllm` models run their serve
        process inside the Ray head container, while `requires_ray_cluster`
        models shard through a Ray cluster spanning their own containers.
        Either way the cluster must span one node per tensor-parallel rank.
        """
        if self.engine == "ray_vllm" or self.requires_ray_cluster:
            return max(self.tensor_parallel_size, 1)
        return 0

    @property
    def is_remote(self) -> bool:
        return bool(self.host)

    @property
    def base_url(self) -> str:
        """Backend base URL — remote host if set, else the container's DNS name."""
        return f"http://{self.host or self.container_name}:{self.port}"

    def to_slot(self, state: ContainerState = ContainerState.STOPPED) -> ModelSlot:
        return ModelSlot(
            name=self.name,
            container_name=self.container_name,
            vram_allocated_mb=self.vram_required_mb,
            state=state,
            port=self.port,
            quantization=self.quantization,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            kv_cache_dtype=self.kv_cache_dtype,
        )


# ────── Model Catalog ──────

DEEPSEEK_V4_FLASH = ModelDefinition(
    name="deepseek-v4-flash",
    # Community B12X build for DGX Spark: eugr/spark-vllm-b12x, tagged locally
    # as vllm-node-b12x by `build-and-copy.sh --exp-b12x`.
    # The image is NOT started by the Gateway — launch-cluster.sh brings up one
    # `vllm_node` container per node (Ray + mods + host networking) and the
    # Gateway only starts/stops the vllm serve processes inside them.
    container_image="vllm-node-b12x:latest",
    container_name=SPARK_CLUSTER_CONTAINER,
    # 284B total / 13B active MoE, FP4 experts + FP8 dense, sharded TP=2 across
    # both Sparks. Not visible to local NVML, hence skip_vram_check below.
    vram_required_mb=220_000,
    port=8020,
    quantization="auto",           # native FP4/FP8 checkpoint, never --quantization
    tensor_parallel_size=2,
    max_model_len="auto",          # resolves to 1,048,576 tokens
    kv_cache_dtype="fp8",
    hf_model_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    engine="spark_cluster",
    exec_container=SPARK_CLUSTER_CONTAINER,
    cluster_nodes=2,
    # launch-cluster.sh exports these inside its launch script rather than via
    # `docker run -e`, so the container env does NOT carry them and the exec'd
    # serve process must set them itself.
    extra_env={
        "CUTE_DSL_ARCH": "sm_121a",
        "VLLM_USE_AOT_COMPILE": "1",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
        "VLLM_MEMORY_PROFILE_INCLUDE_ATTN": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "VLLM_USE_B12X_WO_PROJECTION": "1",
        "VLLM_USE_B12X_MHC": "1",
        "VLLM_USE_B12X_FP8_GEMM": "1",
        "VLLM_USE_B12X_MOE": "1",
        "VLLM_USE_B12X_SPARSE_INDEXER": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_MOE_SKIP_PADDING": "0",
        "B12X_MLA_SM120_UNIFIED": "1",
        "B12X_MOE_FORCE_A8": "1",
    },
    extra_args={
        "--host": "0.0.0.0",
        "--block-size": 256,
        "--max-num-seqs": 8,
        "--max-num-batched-tokens": 8192,
        "--gpu-memory-utilization": "0.85",
        "--tokenizer-mode": "deepseek_v4",
        "--tool-call-parser": "deepseek_v4",
        "--enable-auto-tool-choice": True,
        "--reasoning-parser": "deepseek_v4",
        "--reasoning-config": (
            '{"reasoning_parser":"deepseek_v4",'
            '"reasoning_start_str":"","reasoning_end_str":""}'
        ),
        # Passed as single tokens: vLLM parses the dotted nested form with "=".
        "--default-chat-template-kwargs.thinking=true": True,
        "--default-chat-template-kwargs.reasoning_effort=high": True,
        # Upstream dropped the instanttensor-hybrid-draft-loader mod from
        # this recipe (2026-09) in favour of the b12x loader, which is built
        # into the image and needs no patch at container launch.
        "--load-format": "b12x",
        "--moe-backend": "b12x",
        "--linear-backend": "b12x",
        "--attention-backend": "B12X",
        "--max-cudagraph-capture-size": 48,
        "--compilation-config": (
            '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
        ),
        "--speculative-config": (
            '{"method":"dspark","num_speculative_tokens":5,'
            '"draft_sample_method":"probabilistic",'
            '"attention_backend":"B12X"}'
        ),
    },
)


QWEN35_122B_FP8 = ModelDefinition(
    name="qwen3.5-122b",
    # Same cluster container as DeepSeek. Its two mods coexist: DeepSeek needs
    # instanttensor-hybrid-draft-loader (patches vLLM's model loader) and this
    # one needs fix-qwen3.5-chat-template, which only drops a jinja file into
    # /workspace, so one container launch serves both models.
    container_image="vllm-node-b12x:latest",
    container_name=SPARK_CLUSTER_CONTAINER,
    # 122B-A10B in Qwen's own FP8: ~127 GB of weights, ~64 GB per node.
    vram_required_mb=140_000,
    port=8021,
    quantization="auto",           # native FP8 checkpoint
    tensor_parallel_size=2,
    max_model_len=262144,
    kv_cache_dtype="auto",         # the recipe leaves KV in its default dtype
    hf_model_id="Qwen/Qwen3.5-122B-A10B-FP8",
    engine="spark_cluster",
    exec_container=SPARK_CLUSTER_CONTAINER,
    # Unlike DeepSeek, this recipe shards through Ray: a single serve process
    # drives both GPUs, so no headless rank and no --nnodes/--node-rank.
    cluster_nodes=1,
    requires_ray_cluster=True,
    extra_args={
        "--host": "0.0.0.0",
        "--distributed-executor-backend": "ray",
        "--gpu-memory-utilization": "0.8",
        "--max-num-batched-tokens": 8192,
        "--load-format": "instanttensor",
        "--enable-auto-tool-choice": True,
        "--tool-call-parser": "qwen3_coder",
        "--reasoning-parser": "qwen3",
        # Installed by the fix-qwen3.5-chat-template mod. Absolute path so the
        # process does not depend on its working directory.
        "--chat-template": "/workspace/unsloth.jinja",
    },
)


QWEN38_FLASH_NEXT = ModelDefinition(
    name="qwen3.8-flash-next",
    # Same cluster container and image as the other two. This recipe needs no
    # mods at all (`mods: []` upstream), so it imposes nothing on a launch that
    # already serves DeepSeek and Qwen3.5.
    container_image="vllm-node-b12x:latest",
    container_name=SPARK_CLUSTER_CONTAINER,
    # ~106 GB of NVFP4 weights plus a 256K KV cache at 0.7 utilization,
    # sharded across both nodes. Not visible to local NVML (skip_vram_check).
    vram_required_mb=150_000,
    port=8022,
    # NVFP4 + MXFP8 mixed checkpoint: vLLM needs the quantization named
    # explicitly here, unlike the native FP8/FP4 checkpoints above.
    quantization="modelopt_mixed",
    tensor_parallel_size=2,
    max_model_len=262144,
    kv_cache_dtype="fp8",
    hf_model_id="local-inference-lab/Qwen3.8-Flash-Next-NVFP4",
    engine="spark_cluster",
    exec_container=SPARK_CLUSTER_CONTAINER,
    # Native multi-node like DeepSeek (rank 0 here + one --headless rank on the
    # worker). The recipe passes no --distributed-executor-backend, so this one
    # does NOT need the containers launched in Ray mode.
    cluster_nodes=2,
    requires_ray_cluster=False,
    # launch-cluster.sh exports these inside its launch script rather than via
    # `docker run -e`, so the exec'd serve process must set them itself.
    extra_env={
        "CUTE_DSL_ARCH": "sm_121a",
        "SAFETENSORS_FAST_GPU": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        "VLLM_USE_AOT_COMPILE": "1",
        "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "B12X_POLICY_MODE": "auto",
        # Cluster-only: collectives ride the ConnectX-7 RoCE rails.
        "VLLM_ENABLE_ROCE_ALLREDUCE": "1",
        "VLLM_ROCE_ALLREDUCE_MAX_SIZE": "2MB",
    },
    extra_args={
        "--host": "0.0.0.0",
        "--pipeline-parallel-size": 1,
        "--dtype": "bfloat16",
        "--block-size": 16,
        "--max-num-seqs": 16,
        "--max-num-batched-tokens": 4096,
        "--gpu-memory-utilization": "0.7",
        # Hybrid attention/SSM model: the Mamba state cache must be aligned to
        # the attention block size or vLLM refuses to allocate it.
        "--mamba-cache-mode": "align",
        "--enable-chunked-prefill": True,
        "--load-format": "b12x",
        "--gdn-decode-kernel": "b12x",
        "--linear-backend": "b12x",
        "--moe-backend": "b12x",
        "--no-enable-flashinfer-autotune": True,
        # Multimodal checkpoint; the encoder replicates per rank instead of
        # sharding. Harmless for the text-only traffic the Gateway serves.
        "--mm-encoder-tp-mode": "data",
        "--enable-auto-tool-choice": True,
        "--tool-call-parser": "qwen3_xml",
        "--reasoning-parser": "qwen3",
        "--speculative-config": (
            '{"method":"mtp","num_speculative_tokens":4}'
        ),
        "--compilation-config": (
            '{"pass_config":{"fuse_act_quant":true}}'
        ),
    },
)

# ────── Media node models (node 3, remote) ──────
# MiniMax-H3: omni-modal DiT generating video with native stereo audio, up to 2K
# and ~15s. Both task checkpoints share the Qwen3-VL-32B conditioning encoder
# and both VAEs, so all of it (~63 GB) stays resident on node 3's GB10 — no
# swapping, and no interaction with this node's text profiles.
#
# fl2va  — t2va and first/last-frame-to-video+audio
# ref2va — omni-reference (9 images, 3 videos, 3 video soundtracks, 3 audios)
#
# Both entries point at the same adapter port: node 3 picks the checkpoint from
# the request. They are deliberately absent from ALL_MODELS and PROFILES —
# those drive local container lifecycle (including force-removal of "orphans").

MINIMAX_H3_FL2VA = ModelDefinition(
    name="minimax-h3-fl2va",
    # One image and one container serve all three families (see Dockerfile.media),
    # so these match ACE-Step's and HiDream's. Purely informational — `host` is
    # set, so this model is proxied to and never orchestrated from here.
    container_image="media-node:latest",
    container_name="media-node",
    vram_required_mb=42_470,        # fl2va INT8 convrot + NVFP4 encoder + both VAEs
    port=MEDIA_NODE_PORT,
    quantization="int8_convrot",
    tensor_parallel_size=1,
    max_model_len=0,
    kv_cache_dtype="none",
    model_path="minimax-h3",
    engine="comfyui",
    host=MEDIA_NODE_HOST,
)

MINIMAX_H3_REF2VA = ModelDefinition(
    name="minimax-h3-ref2va",
    container_image="media-node:latest",
    container_name="media-node",
    vram_required_mb=20_970,        # ref2va INT8 convrot (encoder + VAEs already loaded)
    port=MEDIA_NODE_PORT,
    quantization="int8_convrot",
    tensor_parallel_size=1,
    max_model_len=0,
    kv_cache_dtype="none",
    model_path="minimax-h3",
    engine="comfyui",
    host=MEDIA_NODE_HOST,
)

ACE_STEP_15_XL_TURBO = ModelDefinition(
    name="ace-step-1.5-xl-turbo",
    container_image="media-node:latest",
    container_name="media-node",
    vram_required_mb=19_900,        # DiT bf16 + both Qwen encoders + VAE
    port=MEDIA_NODE_PORT,
    quantization="bf16",
    tensor_parallel_size=1,
    max_model_len=0,
    kv_cache_dtype="none",
    model_path="ace-step-1.5",
    engine="comfyui",
    host=MEDIA_NODE_HOST,
)

HIDREAM_O1_IMAGE = ModelDefinition(
    name="hidream-o1-image",
    container_image="media-node:latest",
    container_name="media-node",
    vram_required_mb=8_100,         # all-in-one fp8_scaled checkpoint
    port=MEDIA_NODE_PORT,
    quantization="fp8_scaled",
    tensor_parallel_size=1,
    max_model_len=0,
    kv_cache_dtype="none",
    model_path="hidream-o1",
    engine="comfyui",
    host=MEDIA_NODE_HOST,
)

# Models served by the media node — exposed via /v1/models and proxied to,
# never orchestrated. One ComfyUI holds all of them and evicts as needed.
REMOTE_MEDIA_MODELS: list[ModelDefinition] = [
    MINIMAX_H3_FL2VA,
    MINIMAX_H3_REF2VA,
    ACE_STEP_15_XL_TURBO,
    HIDREAM_O1_IMAGE,
]

# Which Gateway endpoint drives each model.
MEDIA_MODEL_ENDPOINTS: dict[str, str] = {
    MINIMAX_H3_FL2VA.name: "/v1/av/generate",
    MINIMAX_H3_REF2VA.name: "/v1/av/generate",
    ACE_STEP_15_XL_TURBO.name: "/v1/audio/music",
    HIDREAM_O1_IMAGE.name: "/v1/images/generate",
}


# ────── Load Profiles ──────

@dataclass
class VRAMProfile:
    """VRAM distribution profile across models."""
    mode: ProfileMode
    description: str
    primary_models: list[ModelDefinition]
    secondary_models: list[ModelDefinition] = field(default_factory=list)
    labels: dict[str, ModelDefinition] = field(default_factory=dict)
    total_vram_required_mb: int = 0
    # Skip local VRAM check for profiles that run on the Ray cluster,
    # where memory is distributed across nodes and not visible to NVML.
    skip_vram_check: bool = False

    def __post_init__(self):
        all_models = self.primary_models + self.secondary_models
        self.total_vram_required_mb = sum(m.vram_required_mb for m in all_models)


PROFILE_DEEPSEEK = VRAMProfile(
    mode=ProfileMode.FOCUS,
    description=(
        "DeepSeek-V4-Flash 284B-A13B FP4 experts, TP=2 across both Sparks, "
        "1M context + dspark speculative decoding (~220 GB cluster-wide)"
    ),
    primary_models=[DEEPSEEK_V4_FLASH],
    secondary_models=[],
    labels={"chat": DEEPSEEK_V4_FLASH, "code": DEEPSEEK_V4_FLASH},
    # Weights are sharded across two nodes; local NVML sees only half.
    skip_vram_check=True,
)

PROFILE_QWEN35 = VRAMProfile(
    mode=ProfileMode.FOCUS,
    description=(
        "Qwen3.5-122B-A10B FP8, TP=2 across both Sparks via Ray, "
        "256K context (~140 GB cluster-wide)"
    ),
    primary_models=[QWEN35_122B_FP8],
    secondary_models=[],
    labels={"chat": QWEN35_122B_FP8, "code": QWEN35_122B_FP8},
    # Sharded across two nodes; local NVML sees only half.
    skip_vram_check=True,
)

PROFILE_QWEN38 = VRAMProfile(
    mode=ProfileMode.FOCUS,
    description=(
        "Qwen3.8-Flash-Next NVFP4, TP=2 across both Sparks (native multi-node, "
        "no Ray), 256K context + MTP speculative decoding (~150 GB cluster-wide)"
    ),
    primary_models=[QWEN38_FLASH_NEXT],
    secondary_models=[],
    labels={"chat": QWEN38_FLASH_NEXT, "code": QWEN38_FLASH_NEXT},
    # Sharded across two nodes; local NVML sees only half.
    skip_vram_check=True,
)

PROFILES: dict[str, VRAMProfile] = {
    "deepseek": PROFILE_DEEPSEEK,
    "qwen35": PROFILE_QWEN35,
    "qwen38": PROFILE_QWEN38,
}

# Flat list of every model in the catalog (used for orphan cleanup)
ALL_MODELS: list[ModelDefinition] = [
    DEEPSEEK_V4_FLASH,
    QWEN35_122B_FP8,
    QWEN38_FLASH_NEXT,
]


def get_swap_strategy() -> SwapStrategy:
    """Determines swap strategy based on available RAM."""
    if SYSTEM_RAM_GB >= 512:
        return SwapStrategy.PAUSE_UNPAUSE
    return SwapStrategy.STOP_START
