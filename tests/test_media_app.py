"""
Tests for gateway.media_app — the media-only Gateway for a standalone node.

The point of this app is what it *doesn't* do: no orchestrator, no profile
swapping, no container lifecycle. Those tests matter most — on a media node,
the main app's startup would force-remove the container serving the models.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import gateway.media_app as media_app
import gateway.media_node as media_node
from gateway.schemas import GPUInfo, VRAMReport


# ──────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────

def _vram_report() -> VRAMReport:
    return VRAMReport(
        gpus=[GPUInfo(index=0, name="NVIDIA GB10", vram_total_mb=131072,
                      vram_used_mb=44173, vram_free_mb=86899,
                      temperature_c=79, utilization_pct=96)],
        total_vram_mb=131072, total_used_mb=44173, total_free_mb=86899, healthy=True,
    )


@pytest.fixture()
def stub_vram():
    """Replaces the module-level monitor so NVML is never touched."""
    monitor = MagicMock()
    monitor.start = AsyncMock()
    monitor.stop = AsyncMock()
    monitor.latest = _vram_report()
    monitor.query_gpus = AsyncMock(return_value=_vram_report())
    with patch.object(media_app, "vram_monitor", monitor):
        yield monitor


@pytest.fixture()
def client(stub_vram):
    with patch.object(media_node, "shutdown", AsyncMock()):
        with TestClient(media_app.app) as c:
            yield c


def _probe(state: str):
    """state: "ready" | "loading" | "unreachable"."""
    payloads = {
        "ready": {"url": "http://media-node:8010", "reachable": True, "ready": True,
                  "http_status": 200, "node": {"status": "ok", "ready": True}},
        "loading": {"url": "http://media-node:8010", "reachable": True, "ready": False,
                    "http_status": 503, "node": {"status": "loading", "ready": False}},
        "unreachable": {"url": "http://media-node:8010", "reachable": False, "ready": False,
                        "error": "ClientConnectorError: nope"},
    }
    return patch.object(media_node, "probe",
                        AsyncMock(return_value=(state == "ready", payloads[state])))


# ──────────────────────────────────────────────────────
#  What must NOT be here
# ──────────────────────────────────────────────────────

def test_no_orchestration_surface(client):
    """A media node must not expose profile swapping or container control.

    The main Gateway's /admin/container/{name}/remove and profile switching would
    stop or delete the container that serves the models.
    """
    paths = client.get("/openapi.json").json()["paths"]
    forbidden = [p for p in paths
                 if "/admin/" in p or "profile" in p or "/status/swap" in p]
    assert forbidden == [], f"media node must not expose {forbidden}"


def test_no_text_inference_surface(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/chat/completions" not in paths


def test_media_routes_present(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/av/generate" in paths
    # Forced on for this app: no local video backend to shadow
    assert "/v1/videos/generate" in paths
    assert "/status/media-node" in paths


# ──────────────────────────────────────────────────────
#  Health
# ──────────────────────────────────────────────────────

def test_health_ok_when_backend_ready(client):
    with _probe("ready"):
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["role"] == "media-node"
    assert body["backend"]["ready"] is True
    assert body["models"] == [
        "minimax-h3-fl2va", "minimax-h3-ref2va", "ace-step-1.5-xl-turbo", "qwen-image-2.1",
    ]
    assert body["vram"]["total_used_mb"] == 44173


def test_health_is_503_while_weights_load(client):
    """The adapter answers on its port minutes before it can generate.

    Returning 200 there marks the container healthy too early and lets clients
    fire requests the backend rejects.
    """
    with _probe("loading"):
        resp = client.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "loading"
    assert body["backend"]["reachable"] is True
    assert body["backend"]["ready"] is False


def test_health_degraded_when_backend_unreachable(client):
    with _probe("unreachable"):
        resp = client.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["backend"]["reachable"] is False


def test_vram_endpoint(client, stub_vram):
    resp = client.get("/status/vram")

    assert resp.status_code == 200
    assert resp.json()["gpus"][0]["name"] == "NVIDIA GB10"
    stub_vram.query_gpus.assert_awaited()


# ──────────────────────────────────────────────────────
#  Model discovery
# ──────────────────────────────────────────────────────

def test_models_lists_all_modalities_and_aliases(client):
    data = client.get("/v1/models").json()["data"]
    ids = [m["id"] for m in data]

    # Nothing is swapped out, so everything the node can serve is always listed
    for expected in ("minimax-h3-fl2va", "minimax-h3-ref2va",
                     "ace-step-1.5-xl-turbo", "qwen-image-2.1"):
        assert expected in ids

    by_id = {m["id"]: m for m in data}
    assert by_id["minimax-h3-fl2va"]["endpoint"] == "/v1/av/generate"
    assert by_id["ace-step-1.5-xl-turbo"]["endpoint"] == "/v1/audio/music"
    assert by_id["qwen-image-2.1"]["endpoint"] == "/v1/images/generate"

    # Every entry must be actionable — an endpoint-less model is undiscoverable
    assert all(m.get("endpoint") for m in data)

    aliases = {m["id"]: m["alias_for"] for m in data if "alias_for" in m}
    assert aliases == {
        "av": "minimax-h3-fl2va",
        "video": "minimax-h3-fl2va",
        "music": "ace-step-1.5-xl-turbo",
        "image": "qwen-image-2.1",
    }


def test_all_modality_routes_present(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/v1/av/generate", "/v1/audio/music", "/v1/images/generate"):
        assert path in paths


# ──────────────────────────────────────────────────────
#  Generation passes through to the adapter
# ──────────────────────────────────────────────────────

def test_generate_proxies_to_the_adapter(client):
    session = MagicMock()
    session.closed = False
    resp_obj = MagicMock()
    resp_obj.status = 200
    resp_obj.json = AsyncMock(return_value={"video_base64": "AAA=", "mode": "t2va",
                                            "seed": 1, "num_frames": 124, "fps": 24})
    resp_obj.__aenter__ = AsyncMock(return_value=resp_obj)
    resp_obj.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(return_value=resp_obj)

    with patch.object(media_node, "_get_session", lambda: session):
        r = client.post("/v1/av/generate", json={"prompt": "a storm at sea"})

    assert r.status_code == 200
    assert r.json()["data"]["mode"] == "t2va"
    assert session.post.call_args[0][0].endswith("/generate")


def test_legacy_video_path_reaches_h3(client):
    """/v1/videos/generate is aliased here, so old clients keep working."""
    session = MagicMock()
    session.closed = False
    resp_obj = MagicMock()
    resp_obj.status = 200
    resp_obj.json = AsyncMock(return_value={"video_base64": "AAA=", "mode": "t2va", "seed": 2})
    resp_obj.__aenter__ = AsyncMock(return_value=resp_obj)
    resp_obj.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(return_value=resp_obj)

    with patch.object(media_node, "_get_session", lambda: session):
        r = client.post("/v1/videos/generate", json={"prompt": "x"})

    assert r.status_code == 200
    assert r.json()["data"]["mode"] == "t2va"
