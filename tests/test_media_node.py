"""
Tests for gateway.media_node — the remote MiniMax-H3 media node (node 3).

The media node is another machine, so everything here is about the proxy
boundary: route registration, request pass-through, and error translation.
No Docker, no VRAM, no ComfyUI.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gateway.media_node as media_node
from gateway.config import (
    ACE_STEP_15_XL_TURBO,
    ALL_MODELS,
    QWEN_IMAGE_21,
    MEDIA_MODEL_ENDPOINTS,
    MINIMAX_H3_FL2VA,
    MINIMAX_H3_REF2VA,
    REMOTE_MEDIA_MODELS,
)


# ──────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────

class FakeResponse:
    """Simulates an aiohttp response context manager."""

    def __init__(self, status=200, body="", json_data=None):
        self.status = status
        self._body = body
        self._json = json_data or {}

    async def text(self):
        return self._body

    async def json(self):
        return self._json

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patch_node(response=None, exc=None):
    """Patches the module's session so no real HTTP happens."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    if exc is not None:
        session.post = MagicMock(side_effect=exc)
        session.get = MagicMock(side_effect=exc)
    else:
        session.post = MagicMock(return_value=response)
        session.get = MagicMock(return_value=response)
    return patch.object(media_node, "_get_session", lambda: session), session


def _client(alias: bool = False) -> TestClient:
    with patch.object(media_node, "ALIAS_LEGACY_VIDEO_PATH", alias):
        app = FastAPI()
        media_node.attach(app)
    return TestClient(app)


def _ok_payload() -> dict:
    return {
        "video_base64": base64.b64encode(b"fake-mp4-bytes").decode(),
        "filename": "h3_00001_.mp4",
        "mode": "t2va",
        "has_audio": True,
        "fps": 24,
        "num_frames": 124,
        "width": 1344,
        "height": 768,
        "seed": 42,
    }


# ──────────────────────────────────────────────────────
#  Model catalog
# ──────────────────────────────────────────────────────

def test_media_models_are_remote():
    assert MINIMAX_H3_FL2VA.is_remote
    assert MINIMAX_H3_FL2VA.base_url == f"http://{MINIMAX_H3_FL2VA.host}:{MINIMAX_H3_FL2VA.port}"


def test_media_models_excluded_from_local_lifecycle():
    """ALL_MODELS drives orphan container *removal* — a remote model in there
    would have the Gateway force-delete the media node's container."""
    for model in REMOTE_MEDIA_MODELS:
        assert model not in ALL_MODELS
    assert not any(m.is_remote for m in ALL_MODELS)
    assert REMOTE_MEDIA_MODELS == [
        MINIMAX_H3_FL2VA, MINIMAX_H3_REF2VA, ACE_STEP_15_XL_TURBO, QWEN_IMAGE_21,
    ]


def test_every_media_model_has_an_endpoint():
    """A model listed with no endpoint is undiscoverable in practice."""
    for model in REMOTE_MEDIA_MODELS:
        assert MEDIA_MODEL_ENDPOINTS.get(model.name), f"{model.name} has no endpoint"


def test_local_models_keep_container_dns_urls():
    from gateway.config import QWEN3_5_4B

    assert not QWEN3_5_4B.is_remote
    assert QWEN3_5_4B.base_url == f"http://{QWEN3_5_4B.container_name}:{QWEN3_5_4B.port}"


# ──────────────────────────────────────────────────────
#  Route registration
# ──────────────────────────────────────────────────────

def test_canonical_routes_registered():
    paths = _client().get("/openapi.json").json()["paths"]
    assert "/v1/av/generate" in paths
    assert "/status/media-node" in paths
    # Not aliased unless asked
    assert "/v1/videos/generate" not in paths


def test_legacy_alias_when_enabled():
    paths = _client(alias=True).get("/openapi.json").json()["paths"]
    assert "/v1/videos/generate" in paths


def test_alias_never_shadows_an_existing_route():
    """A Gateway already serving LTX video must keep its own handler."""
    with patch.object(media_node, "ALIAS_LEGACY_VIDEO_PATH", True):
        app = FastAPI()

        @app.post("/v1/videos/generate")
        async def ltx():
            return {"backend": "ltx"}

        media_node.attach(app)

    handlers = [r for r in app.routes if getattr(r, "path", None) == "/v1/videos/generate"]
    assert len(handlers) == 1
    assert handlers[0].endpoint.__name__ == "ltx"


# ──────────────────────────────────────────────────────
#  Generation pass-through
# ──────────────────────────────────────────────────────

def test_generate_forwards_and_wraps_result():
    patcher, session = _patch_node(FakeResponse(200, json_data=_ok_payload()))
    with patcher:
        resp = _client().post("/v1/av/generate", json={"prompt": "a fox howling at dusk"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["mode"] == "t2va"
    assert body["data"]["has_audio"] is True
    assert "processing_time_ms" in body["data"]

    # Forwarded to the node's /generate with H3 defaults applied
    url = session.post.call_args[0][0]
    payload = session.post.call_args.kwargs["json"]
    assert url.endswith("/generate")
    assert payload["prompt"] == "a fox howling at dusk"
    assert (payload["width"], payload["height"]) == (1344, 768)
    assert payload["num_frames"] == 124 and payload["fps"] == 24
    assert payload["sampler"] == "res_multistep" and payload["scheduler"] == "simple"
    # response_format is a Gateway-side concern, not the node's
    assert "response_format" not in payload


def test_unreachable_node_returns_503():
    patcher, _ = _patch_node(exc=aiohttp.ClientConnectorError(MagicMock(), OSError("no route")))
    with patcher:
        resp = _client().post("/v1/av/generate", json={"prompt": "x"})

    assert resp.status_code == 503
    assert "unreachable" in resp.json()["detail"]


def test_node_still_loading_propagates_503():
    """503 must survive as 503 so clients retry while the 63 GB loads."""
    patcher, _ = _patch_node(FakeResponse(503, body="ComfyUI not ready yet"))
    with patcher:
        resp = _client().post("/v1/av/generate", json={"prompt": "x"})

    assert resp.status_code == 503


def test_bad_request_propagates_400():
    patcher, _ = _patch_node(FakeResponse(400, body="ref2va accepts at most 9 reference images"))
    with patcher:
        resp = _client().post("/v1/av/generate", json={"prompt": "x"})

    assert resp.status_code == 400


def test_backend_error_becomes_502():
    patcher, _ = _patch_node(FakeResponse(500, body="ComfyUI execution failed"))
    with patcher:
        resp = _client().post("/v1/av/generate", json={"prompt": "x"})

    assert resp.status_code == 502


def test_reference_fields_are_forwarded():
    img = base64.b64encode(b"png").decode()
    patcher, session = _patch_node(FakeResponse(200, json_data={**_ok_payload(), "mode": "ref2va"}))
    with patcher:
        resp = _client().post("/v1/av/generate", json={
            "prompt": "<Picture 1> dancing in the rain",
            "ref_images": [img, img],
            "ref_image_size": "max",
        })

    assert resp.status_code == 200
    payload = session.post.call_args.kwargs["json"]
    assert payload["ref_images"] == [img, img]
    assert payload["ref_image_size"] == "max"


def test_invalid_ref_image_size_rejected_before_dispatch():
    patcher, session = _patch_node(FakeResponse(200, json_data=_ok_payload()))
    with patcher:
        resp = _client().post("/v1/av/generate", json={"prompt": "x", "ref_image_size": "huge"})

    assert resp.status_code == 422
    session.post.assert_not_called()


# ──────────────────────────────────────────────────────
#  URL response format
# ──────────────────────────────────────────────────────

def test_url_format_requires_asset_dir():
    patcher, _ = _patch_node(FakeResponse(200, json_data=_ok_payload()))
    with patcher, patch.object(media_node, "MEDIA_ASSET_DIR", ""):
        resp = _client().post("/v1/av/generate",
                              json={"prompt": "x", "response_format": "url"})

    assert resp.status_code == 400
    assert "MEDIA_ASSET_DIR" in resp.json()["detail"]


def test_url_format_writes_file_and_returns_link(tmp_path):
    patcher, _ = _patch_node(FakeResponse(200, json_data=_ok_payload()))
    with patcher, \
         patch.object(media_node, "MEDIA_ASSET_DIR", str(tmp_path)), \
         patch.object(media_node, "MEDIA_PUBLIC_URL", "http://192.168.1.125:8000"):
        resp = _client().post("/v1/av/generate",
                              json={"prompt": "x", "response_format": "url"})

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["video_url"].startswith("http://192.168.1.125:8000/assets/media/")
    assert data["size_bytes"] == len(b"fake-mp4-bytes")
    # base64 is dropped in URL mode — the point is not shipping 50 MB of JSON
    assert "video_base64" not in data

    written = list(tmp_path.glob("*.mp4"))
    assert len(written) == 1
    assert written[0].read_bytes() == b"fake-mp4-bytes"


# ──────────────────────────────────────────────────────
#  Status endpoint
# ──────────────────────────────────────────────────────

def test_status_reports_reachable_node():
    patcher, _ = _patch_node(FakeResponse(200, json_data={"status": "ok", "ready": True}))
    with patcher:
        resp = _client().get("/status/media-node")

    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is True
    assert body["node"]["ready"] is True


def test_status_reports_unreachable_node():
    patcher, _ = _patch_node(exc=aiohttp.ClientConnectorError(MagicMock(), OSError("down")))
    with patcher:
        resp = _client().get("/status/media-node")

    assert resp.status_code == 503
    body = resp.json()
    assert body["reachable"] is False
    assert body["ready"] is False


def test_reachable_but_loading_is_not_ready():
    """The adapter answers on its port long before the 63 GB are loaded.

    Reporting that as healthy would send generations into a backend that 503s.
    """
    patcher, _ = _patch_node(FakeResponse(503, json_data={"status": "loading", "ready": False}))
    with patcher:
        resp = _client().get("/status/media-node")

    assert resp.status_code == 503
    body = resp.json()
    assert body["reachable"] is True     # the port answers...
    assert body["ready"] is False        # ...but it isn't usable yet


def test_ready_only_when_backend_says_so():
    patcher, _ = _patch_node(FakeResponse(200, json_data={"status": "ok", "ready": True}))
    with patcher:
        resp = _client().get("/status/media-node")

    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_validation_errors_reach_the_caller_as_4xx():
    """A 422 from the node is the caller's bad input, not a bad gateway.

    Mapping it to 502 tells them to blame the infrastructure for their own
    out-of-range request.
    """
    patcher, _ = _patch_node(FakeResponse(
        422, body='{"detail":[{"loc":["body","num_frames"],"msg":"Input should be less than or equal to 362"}]}'))
    with patcher:
        resp = _client().post("/v1/av/generate", json={"prompt": "x"})

    assert resp.status_code == 422
    # and the node's detail is unwrapped, not double-encoded JSON
    detail = resp.json()["detail"]
    assert isinstance(detail, list)
    assert "362" in detail[0]["msg"]


def test_backend_500_still_becomes_502():
    patcher, _ = _patch_node(FakeResponse(500, body="ComfyUI execution failed"))
    with patcher:
        resp = _client().post("/v1/av/generate", json={"prompt": "x"})
    assert resp.status_code == 502


# ──────────────────────────────────────────────────────
#  Client disconnect
# ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disconnect_aborts_the_upstream_call():
    """There are two hops: dropping the caller here does not reach the node.

    Unless the Gateway abandons its own upstream request, the node keeps a GPU
    busy for nobody and later jobs pile up inside ComfyUI, invisible to the
    node's serialisation. That is what wedged the node for nine hours.
    """
    import asyncio as _asyncio
    from fastapi import HTTPException

    started = _asyncio.Event()
    cancelled = _asyncio.Event()

    async def never_returns(*a, **kw):
        started.set()
        try:
            await _asyncio.sleep(3600)
        except _asyncio.CancelledError:
            cancelled.set()
            raise

    request = MagicMock()
    request.is_disconnected = AsyncMock(return_value=True)

    with patch.object(media_node, "_dispatch", wraps=media_node._dispatch) as _, \
         patch.object(media_node, "DISCONNECT_POLL_S", 0.01):
        # Replace the inner (no-request) branch with a call that hangs
        real = media_node._dispatch

        async def dispatch(path, payload, label, req=None):
            if req is None:
                return await never_returns()
            return await real(path, payload, label, req)

        with patch.object(media_node, "_dispatch", dispatch):
            with pytest.raises(HTTPException) as exc:
                await real("/generate", {}, "av", request)

    assert exc.value.status_code == 499
    await _asyncio.sleep(0.05)
    assert cancelled.is_set(), "the upstream request must be cancelled, closing the connection"


@pytest.mark.asyncio
async def test_connected_caller_gets_the_result():
    import asyncio as _asyncio

    request = MagicMock()
    request.is_disconnected = AsyncMock(return_value=False)
    real = media_node._dispatch

    async def dispatch(path, payload, label, req=None):
        if req is None:
            await _asyncio.sleep(0.02)
            return {"seed": 1}, 20.0
        return await real(path, payload, label, req)

    with patch.object(media_node, "_dispatch", dispatch), \
         patch.object(media_node, "DISCONNECT_POLL_S", 0.01):
        result, elapsed = await real("/generate", {}, "av", request)

    assert result == {"seed": 1}


# ──────────────────────────────────────────────────────
#  Upstream timeout
# ──────────────────────────────────────────────────────

async def test_no_total_timeout_on_the_upstream_session():
    """A ceiling here aborts a 55-minute generation the GPU already paid for.

    aiohttp treats total=None as "no limit"; 0 is NOT the same thing — it is a
    falsy value that would be passed straight through as a 0-second deadline,
    which is why the config uses `or None` rather than the raw value.
    """
    with patch.object(media_node, "_session", None):
        try:
            session = media_node._get_session()
            assert session.timeout.total is None
            # The connect timeout must survive, or an unreachable node hangs
            # instead of failing fast with a 503.
            assert session.timeout.connect == media_node.MEDIA_NODE_CONNECT_TIMEOUT_S
        finally:
            await media_node.shutdown()


async def test_a_configured_timeout_is_still_applied():
    with patch.object(media_node, "_session", None), \
         patch.object(media_node, "MEDIA_NODE_TIMEOUT_S", 42):
        try:
            assert media_node._get_session().timeout.total == 42
        finally:
            await media_node.shutdown()


def test_upstream_timeout_becomes_504_not_500():
    """Regression: aiohttp raises asyncio.TimeoutError on the total timeout, which
    is not an aiohttp.ClientError, so it escaped the handler as a bare 500 that
    told the caller nothing about what had happened."""
    patcher, _ = _patch_node(exc=TimeoutError())
    with patcher, patch.object(media_node, "MEDIA_NODE_TIMEOUT_S", 30):
        resp = _client().post("/v1/av/generate", json={"prompt": "x"})

    assert resp.status_code == 504
    detail = resp.json()["detail"]
    assert "MEDIA_NODE_TIMEOUT_S=30" in detail
    # The node keeps running the job — say so, rather than implying it was undone.
    assert "may still be running" in detail


def test_connect_failure_is_still_503_not_504():
    """aiohttp's connect timeout subclasses both ClientError and TimeoutError, so
    clause order decides which one wins. It is an unreachable node, not a slow
    generation."""
    patcher, _ = _patch_node(exc=aiohttp.ServerTimeoutError("connect timed out"))
    with patcher:
        resp = _client().post("/v1/av/generate", json={"prompt": "x"})

    assert resp.status_code == 503
    assert "unreachable" in resp.json()["detail"]


def test_image_route_forwards_qwen_defaults():
    patcher, session = _patch_node(FakeResponse(200, json_data={
        "images": ["cG5n"], "model": "qwen-image-2.1",
    }))
    with patcher:
        response = _client().post('/v1/images/generate', json={"prompt": "a lighthouse"})
    assert response.status_code == 200
    payload = session.post.call_args.kwargs['json']
    assert payload['scheduler'] == 'simple'
    assert 'variant' not in payload
    assert 'noise_scale' not in payload
    assert 'response_format' not in payload
    assert response.json()['data']['model'] == 'qwen-image-2.1'
