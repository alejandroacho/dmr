"""
Tests for gateway.proxy — Retry Logic on Connection Failures.
Validates Bug 5 fix: proxy retries transient connection errors.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from tests.conftest import TEST_MODEL_A
from gateway.proxy import InferenceProxy


# ──────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────

def _make_proxy() -> InferenceProxy:
    proxy = InferenceProxy()
    proxy._session = MagicMock(spec=aiohttp.ClientSession)
    # Speed up retries for tests
    proxy.CONNECT_RETRIES = 3
    proxy.CONNECT_RETRY_DELAY = 0.01
    return proxy


class FakeResponse:
    """Simulates an aiohttp response context manager."""

    def __init__(self, status=200, body=None, json_data=None):
        self.status = status
        self._body = body or ""
        self._json = json_data or {}

    async def text(self):
        return self._body

    async def json(self):
        return self._json

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# ──────────────────────────────────────────────────────
#  Chat completion retries
# ──────────────────────────────────────────────────────

class TestChatCompletionRetry:

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        proxy = _make_proxy()
        expected = {"choices": [{"message": {"content": "hello"}}]}
        proxy._session.post = MagicMock(
            return_value=FakeResponse(status=200, json_data=expected)
        )

        result = await proxy.chat_completion(
            TEST_MODEL_A, {"messages": []}, stream=False
        )

        assert result == expected

    @pytest.mark.asyncio
    async def test_retries_on_connection_error_then_succeeds(self):
        """Should retry on OSError and succeed on later attempt."""
        proxy = _make_proxy()
        call_count = 0

        def _alternating_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise aiohttp.ClientConnectorError(
                    connection_key=MagicMock(), os_error=OSError("Connection refused")
                )
            return FakeResponse(status=200, json_data={"ok": True})

        proxy._session.post = _alternating_post

        result = await proxy.chat_completion(
            TEST_MODEL_A, {"messages": []}, stream=False
        )

        assert result == {"ok": True}
        assert call_count == 3  # Failed 2x, succeeded on 3rd

    @pytest.mark.asyncio
    async def test_returns_error_after_all_retries_exhausted(self):
        """After CONNECT_RETRIES failures, should return error dict."""
        proxy = _make_proxy()

        def _always_fail(*args, **kwargs):
            raise OSError("Connection refused")

        proxy._session.post = _always_fail

        result = await proxy.chat_completion(
            TEST_MODEL_A, {"messages": []}, stream=False
        )

        assert "error" in result
        assert result["error"]["code"] == 502
        assert "unreachable after 3 retries" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_retries_on_timeout_error(self):
        """asyncio.TimeoutError should trigger retries."""
        proxy = _make_proxy()
        call_count = 0

        def _timeout_then_ok(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            return FakeResponse(status=200, json_data={"done": True})

        proxy._session.post = _timeout_then_ok

        result = await proxy.chat_completion(
            TEST_MODEL_A, {"messages": []}, stream=False
        )

        assert result == {"done": True}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_http_error_status(self):
        """HTTP 4xx/5xx is not a connection error — should NOT retry."""
        proxy = _make_proxy()
        call_count = 0

        def _http_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeResponse(status=422, body="Validation error")

        proxy._session.post = _http_error

        result = await proxy.chat_completion(
            TEST_MODEL_A, {"messages": []}, stream=False
        )

        assert "error" in result
        assert result["error"]["code"] == 422
        assert call_count == 1  # No retries for HTTP errors


# ──────────────────────────────────────────────────────
#  Streaming retries
# ──────────────────────────────────────────────────────

class TestStreamRetry:

    @pytest.mark.asyncio
    async def test_stream_retries_on_connect_error(self):
        """Streaming should retry on connection failure."""
        proxy = _make_proxy()
        call_count = 0

        class FakeStreamResponse:
            def __init__(self):
                self.status = 200
                self.content = self

            async def iter_any(self):
                yield b"data: {\"chunk\": 1}\n\n"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        def _fail_then_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aiohttp.ClientConnectorError(
                    connection_key=MagicMock(), os_error=OSError("refused")
                )
            return FakeStreamResponse()

        proxy._session.post = _fail_then_stream

        chunks = []
        gen = proxy._stream_response(
            "http://fake:8001/v1/chat/completions", {"messages": []}
        )
        async for chunk in gen:
            chunks.append(chunk)

        assert call_count == 2
        assert len(chunks) >= 1
        assert b"chunk" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_error_after_retries_exhausted(self):
        """Should yield error event after all retry attempts fail."""
        proxy = _make_proxy()

        def _always_fail(*args, **kwargs):
            raise OSError("Connection refused")

        proxy._session.post = _always_fail

        chunks = []
        gen = proxy._stream_response(
            "http://fake:8001/v1/chat/completions", {"messages": []}
        )
        async for chunk in gen:
            chunks.append(chunk)

        assert len(chunks) == 1
        decoded = chunks[0].decode()
        assert "data:" in decoded
        error_data = json.loads(decoded.split("data: ")[1].strip())
        assert error_data["error"]["code"] == 502

    @pytest.mark.asyncio
    async def test_mid_stream_failure_is_not_replayed(self):
        """Regression: retrying after chunks were already yielded re-POSTed the
        request, so the client received the completion twice, concatenated.
        Once output is in flight the only safe move is to report the break."""
        proxy = _make_proxy()
        call_count = 0

        class BreakingStream:
            def __init__(self):
                self.status = 200
                self.content = self

            async def iter_any(self):
                yield b'data: {"choices":[{"delta":{"content":"Hola"}}]}\n\n'
                raise aiohttp.ServerDisconnectedError()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        def _break_midway(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return BreakingStream()

        proxy._session.post = _break_midway

        chunks = []
        async for chunk in proxy._stream_response(
            "http://fake:8001/v1/chat/completions", {"messages": []}
        ):
            chunks.append(chunk)

        assert call_count == 1, "must not re-POST once output has been sent"
        assert b"Hola" in chunks[0]
        assert chunks.count(chunks[0]) == 1, "content must not be duplicated"
        error = json.loads(chunks[-1].decode().split("data: ")[1].strip())
        assert error["error"]["code"] == 502
        assert "interrupted" in error["error"]["message"].lower()


# ──────────────────────────────────────────────────────
#  Served-model-name resolution
# ──────────────────────────────────────────────────────

class TestServedModelName:
    """The Gateway also adopts serve processes started by the cluster
    launcher, whose recipes pass no --served-model-name. vLLM then registers
    the bare HuggingFace id and every request 404s."""

    @pytest.mark.asyncio
    async def test_uses_backend_id_when_configured_name_is_absent(self):
        proxy = _make_proxy()
        proxy._session.get = MagicMock(
            return_value=FakeResponse(
                status=200,
                json_data={"data": [{"id": "deepseek-ai/DeepSeek-V4-Flash-0731"}]},
            )
        )

        name = await proxy._served_model_name(TEST_MODEL_A)
        assert name == "deepseek-ai/DeepSeek-V4-Flash-0731"

    @pytest.mark.asyncio
    async def test_prefers_configured_name_when_backend_offers_it(self):
        proxy = _make_proxy()
        proxy._session.get = MagicMock(
            return_value=FakeResponse(
                status=200,
                json_data={"data": [{"id": "other"}, {"id": TEST_MODEL_A.name}]},
            )
        )

        assert await proxy._served_model_name(TEST_MODEL_A) == TEST_MODEL_A.name

    @pytest.mark.asyncio
    async def test_result_is_cached_and_forgettable(self):
        proxy = _make_proxy()
        calls = 0

        def _get(*args, **kwargs):
            nonlocal calls
            calls += 1
            return FakeResponse(status=200, json_data={"data": [{"id": "hf/id"}]})

        proxy._session.get = _get

        assert await proxy._served_model_name(TEST_MODEL_A) == "hf/id"
        assert await proxy._served_model_name(TEST_MODEL_A) == "hf/id"
        assert calls == 1, "should not re-query the backend on every request"

        proxy.forget_served_names()
        assert await proxy._served_model_name(TEST_MODEL_A) == "hf/id"
        assert calls == 2

    @pytest.mark.asyncio
    async def test_unreachable_backend_falls_back_without_caching(self):
        """While the model is still loading /v1/models refuses connections;
        caching that answer would pin the wrong name for the whole run."""
        proxy = _make_proxy()

        def _fail(*args, **kwargs):
            raise OSError("connection refused")

        proxy._session.get = _fail

        assert await proxy._served_model_name(TEST_MODEL_A) == TEST_MODEL_A.name
        assert proxy._served_names == {}
