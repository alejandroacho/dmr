# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dockerfile — Blackwell Smart Gateway
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FROM python:3.12-slim AS base

# Metadata
LABEL maintainer="Alejandro Acho"
LABEL description="Blackwell Orchestrator & Smart Gateway"
LABEL version="1.0.0"

# Default environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ─── System dependencies ─────────────────────────
# openssh-client: multi-node models (engine=spark_cluster) run one vllm serve
# per node, and the worker nodes' Docker daemons are only reachable over SSH.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# ─── Python dependencies ──────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Source code ───────────────────────────────────
COPY gateway/ ./gateway/

# ─── Health check ────────────────────────────────────
HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:8000/health || exit 1

# ─── Port ──────────────────────────────────────────
EXPOSE 8000

# ─── Entrypoint ──────────────────────────────────────
CMD ["python", "-m", "uvicorn", "gateway.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
