#!/bin/bash
#
# Reset the entire Ray cluster (head + worker) from the head node.
# Cleans DEAD/zombie nodes, restarts both containers, verifies state.
#
# Usage: bash reset_cluster.sh

set -uo pipefail

WORKER_HOST="${WORKER_HOST:-alejandroacho@192.168.200.13}"
HEAD_IP="${HEAD_IP:-192.168.200.12}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " Ray cluster full reset"
echo " Head:   $HEAD_IP (local)"
echo " Worker: $WORKER_HOST"
echo "============================================"

echo ""
echo ">>> [1/3] Resetting head node..."
bash "$SCRIPT_DIR/reset_ray_node.sh" --head

echo ""
echo ">>> [2/3] Resetting worker node..."
# Try systemd first; fall back to manual reset_ray_node.sh
ssh "$WORKER_HOST" "
    if systemctl is-enabled ray-node-worker >/dev/null 2>&1; then
        echo 'Using systemd service...'
        sudo systemctl restart ray-node-worker
    else
        echo 'No systemd service — using reset_ray_node.sh directly...'
        bash ~/reset_ray_node.sh --worker $HEAD_IP
    fi
"

echo ""
echo ">>> [3/3] Verifying cluster..."
echo "Waiting 15s for nodes to register..."
sleep 15

docker exec ray-node-head ray list nodes 2>&1 | grep -E "ALIVE|DEAD" | awk '{
    state = $5
    ip = $3
    id = substr($2, 1, 12)
    printf "  %-8s %s (%s)\n", state, ip, id
}'

echo ""
ALIVE=$(docker exec ray-node-head ray list nodes 2>&1 | grep -c "ALIVE")
DEAD=$(docker exec ray-node-head ray list nodes 2>&1 | grep -c "DEAD")
echo "Result: $ALIVE ALIVE, $DEAD DEAD"

if [ "$ALIVE" -ge 2 ] && [ "$DEAD" -eq 0 ]; then
    echo "Cluster healthy."
    exit 0
else
    echo "Warning: expected 2 ALIVE / 0 DEAD."
    exit 1
fi
