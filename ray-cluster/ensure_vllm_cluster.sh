#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Ensure the two-node vllm_node cluster is up.
#
#  Idempotent: a complete cluster is left strictly alone, so this is safe to
#  run from a boot unit and from a timer.
#
#  It exists because the nodes do not recover on their own after a power cut:
#  launch-cluster.sh creates the containers with `docker run --rm`, which
#  Docker refuses to combine with a restart policy, so a container that dies
#  is removed and never comes back.
# ─────────────────────────────────────────────────────────────
set -uo pipefail

LAUNCHER_DIR="${LAUNCHER_DIR:-$HOME/spark-vllm-docker}"
RECIPE="${RECIPE:-deepseek-v4-flash-0731}"
PORT="${PORT:-8020}"
WORKER="${WORKER:-192.168.200.13}"
CONTAINER="${CONTAINER:-vllm_node}"
HF_HOME_DIR="${HF_HOME_DIR:-$HOME/hf-cache}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes)

log() { echo "[ensure-vllm-cluster] $*"; }

head_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]
}

# Echoes: running | missing | unreachable
worker_state() {
    local out
    if ! out=$(ssh "${SSH_OPTS[@]}" "$WORKER" \
            "docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null || true" \
            2>/dev/null); then
        echo unreachable
        return
    fi
    if [ "$(printf '%s' "$out" | tr -d '[:space:]')" = "true" ]; then
        echo running
    else
        echo missing
    fi
}

worker=$(worker_state)

# A network blip must never be read as "the worker is gone": tearing down a
# healthy cluster costs a full weight reload for nothing.
if [ "$worker" = "unreachable" ]; then
    log "worker $WORKER unreachable — leaving the cluster untouched."
    exit 0
fi

if head_running && [ "$worker" = "running" ]; then
    log "cluster already complete (head + worker). Nothing to do."
    exit 0
fi

log "cluster incomplete (head=$(head_running && echo running || echo missing), worker=$worker). Relaunching."

# Tear down BOTH ranks first. The launcher checks only whether *some*
# container is already running and then prints "Cluster containers are already
# running. Skipping launch." — leaving the node that actually lost its
# container without a rank, while still starting the survivor's. A partial
# cluster cannot serve a TP=2 model, so the clean state is all-or-nothing.
docker stop "$CONTAINER" >/dev/null 2>&1 || true
ssh "${SSH_OPTS[@]}" "$WORKER" "docker stop $CONTAINER" >/dev/null 2>&1 || true

cd "$LAUNCHER_DIR" || { log "launcher dir '$LAUNCHER_DIR' not found."; exit 1; }
HF_HOME="$HF_HOME_DIR" ./run-recipe.sh "$RECIPE" --port "$PORT" -d
rc=$?

if [ $rc -ne 0 ]; then
    log "launcher exited $rc."
    exit $rc
fi

# The launcher reports success even when it created nothing, so verify.
for _ in $(seq 1 30); do
    if head_running && [ "$(worker_state)" = "running" ]; then
        log "cluster up on both nodes. The model loads in the background (~3-5 min)."
        exit 0
    fi
    sleep 2
done

log "ERROR: launcher finished but the containers are not both running."
exit 1
