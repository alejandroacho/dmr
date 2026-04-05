#!/bin/bash
#
# Reset and restart this machine's Ray node from scratch.
#
# Usage:
#   bash reset_ray_node.sh --head
#   bash reset_ray_node.sh --worker <head_ip>
#
# Environment variables (optional overrides):
#   VLLM_IMAGE    Docker image to use  (default: blackwell-vllm:latest)
#   MN_IF_NAME    Network interface    (default: enp1s0f1np1)

set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 --head | --worker <head_ip>"
    exit 1
fi

NODE_TYPE="$1"
HEAD_IP="${2:-}"

if [[ "$NODE_TYPE" != "--head" && "$NODE_TYPE" != "--worker" ]]; then
    echo "Error: first argument must be --head or --worker"
    exit 1
fi

if [[ "$NODE_TYPE" == "--worker" && -z "$HEAD_IP" ]]; then
    echo "Error: --worker requires the head node IP as second argument"
    exit 1
fi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VLLM_IMAGE="${VLLM_IMAGE:-blackwell-vllm:latest}"
MN_IF_NAME="${MN_IF_NAME:-enp1s0f1np1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VLLM_HOST_IP=$(ip -4 addr show "$MN_IF_NAME" | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
if [[ -z "$VLLM_HOST_IP" ]]; then
    echo "Error: could not get IP from interface $MN_IF_NAME"
    exit 1
fi

if [[ "$NODE_TYPE" == "--head" ]]; then
    HEAD_IP="$VLLM_HOST_IP"
fi

echo "============================================"
echo "Node type  : $NODE_TYPE"
echo "This node  : $VLLM_HOST_IP (via $MN_IF_NAME)"
echo "Head node  : $HEAD_IP"
echo "Image      : $VLLM_IMAGE"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Stop and remove ALL ray-node-* containers on this machine
# ---------------------------------------------------------------------------
echo ""
echo ">>> Stopping all ray-node-* containers..."
RUNNING=$(docker ps --format '{{.Names}}' | grep -E '^ray-node-(head|worker)$' || true)
if [[ -n "$RUNNING" ]]; then
    echo "$RUNNING" | xargs docker stop
    echo "Stopped: $RUNNING"
else
    echo "No running ray-node-* containers found."
fi

echo ">>> Removing all ray-node-* containers (including stopped)..."
ALL=$(docker ps -a --format '{{.Names}}' | grep -E '^ray-node-(head|worker)$' || true)
if [[ -n "$ALL" ]]; then
    echo "$ALL" | xargs docker rm
    echo "Removed: $ALL"
else
    echo "No ray-node-* containers to remove."
fi

# ---------------------------------------------------------------------------
# 2. Start fresh Ray node
# ---------------------------------------------------------------------------
echo ""
echo ">>> Starting fresh $NODE_TYPE Ray node..."
bash "$SCRIPT_DIR/run_cluster.sh" "$VLLM_IMAGE" "$HEAD_IP" "$NODE_TYPE" \
    ~/.cache/huggingface \
    --detach \
    -e VLLM_HOST_IP="$VLLM_HOST_IP" \
    -e UCX_NET_DEVICES="$MN_IF_NAME" \
    -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
    -e OMPI_MCA_btl_tcp_if_include="$MN_IF_NAME" \
    -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" \
    -e TP_SOCKET_IFNAME="$MN_IF_NAME" \
    -e RAY_memory_monitor_refresh_ms=0 \
    -e RAY_memory_usage_threshold=0.99 \
    -e MASTER_ADDR="$HEAD_IP"

echo ""
echo "Done. To check cluster status:"
echo "  bash $SCRIPT_DIR/check_ray_cluster_status.sh"
