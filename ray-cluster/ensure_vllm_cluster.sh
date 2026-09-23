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
# Fallback only. The recipe to relaunch is whichever model the Gateway has
# active (see resolve_active_recipe below); these values are used when the
# Gateway cannot be reached, e.g. at boot before it is up.
RECIPE="${RECIPE:-deepseek-v4-flash-0731}"
PORT="${PORT:-8020}"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"
WORKER="${WORKER:-192.168.200.13}"
CONTAINER="${CONTAINER:-vllm_node}"
HF_HOME_DIR="${HF_HOME_DIR:-$HOME/hf-cache}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes)

log() { echo "[ensure-vllm-cluster] $*"; }

# Gateway profile key -> launcher recipe. The Gateway knows which profile is
# active but not which recipe launched it, so the mapping lives here. A key
# missing from this table leaves RECIPE/PORT at their defaults.
declare -A PROFILE_RECIPES=(
    [deepseek]="deepseek-v4-flash-0731"
    [qwen35]="qwen3.5-122b-fp8"
    [qwen38]="qwen3.8-flash-next-nvfp4-cluster"
)

# Relaunching always used to bring back DeepSeek, whatever was actually
# loaded. When this fired during a model swap it exec'd DeepSeek into the
# container where the incoming model was still loading, and the two fought
# over the node's memory. Ask the Gateway what it has active instead.
resolve_active_recipe() {
    local json key port
    json=$(curl -sf --max-time 5 "$GATEWAY_URL/v1/profiles/active" 2>/dev/null) || {
        log "Gateway unreachable — falling back to recipe '$RECIPE' on port $PORT."
        return 0
    }
    key=$(printf '%s' "$json" | python3 -c \
        'import json,sys; print(json.load(sys.stdin).get("key",""))' 2>/dev/null) || return 0
    port=$(printf '%s' "$json" | python3 -c \
        'import json,sys; m=json.load(sys.stdin).get("models") or [{}]; print(m[0].get("port",""))' \
        2>/dev/null) || return 0
    if [ -n "$key" ] && [ -n "${PROFILE_RECIPES[$key]:-}" ]; then
        RECIPE="${PROFILE_RECIPES[$key]}"
        [ -n "$port" ] && PORT="$port"
        log "Gateway reports profile '$key' active — relaunching recipe '$RECIPE' on port $PORT."
    else
        log "Gateway profile '$key' has no recipe mapping — keeping '$RECIPE' on port $PORT."
    fi
}

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

resolve_active_recipe

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
