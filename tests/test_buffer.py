"""
Tests for gateway.request_buffer — Buffer Drain Logic.
Validates Bug 4 fix: buffered requests are drained and resolved
after a swap completes.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.request_buffer import (
    BufferOverflowError,
    RadixPrefixCache,
    RequestBuffer,
    BufferedRequest,
)
from gateway.schemas import AgentRequest


# ──────────────────────────────────────────────────────
#  RequestBuffer core
# ──────────────────────────────────────────────────────

class TestRequestBuffer:

    @pytest.mark.asyncio
    async def test_enqueue_and_drain(self):
        """Enqueued requests should be returned by drain_all."""
        buf = RequestBuffer()

        req = AgentRequest(messages=[{"role": "user", "content": "test"}])
        future = await buf.enqueue(req)

        assert buf.queue_size >= 1

        drained = await buf.drain_all()
        assert len(drained) == 1
        assert drained[0].request == req
        assert not drained[0].future.done()

    @pytest.mark.asyncio
    async def test_resolve_completes_future(self):
        """resolve() should set the Future's result."""
        buf = RequestBuffer()
        req = AgentRequest(messages=[{"role": "user", "content": "hello"}])
        future = await buf.enqueue(req)

        drained = await buf.drain_all()
        result_data = {"choices": [{"message": {"content": "world"}}]}
        await buf.resolve(drained[0], result_data)

        assert future.done()
        assert await asyncio.wait_for(future, timeout=1.0) == result_data

    @pytest.mark.asyncio
    async def test_reject_all_sets_exception(self):
        """reject_all() should make all futures raise."""
        buf = RequestBuffer()
        req = AgentRequest(messages=[{"role": "user", "content": "test"}])
        future = await buf.enqueue(req)

        await buf.reject_all("swap failed")

        with pytest.raises(RuntimeError, match="swap failed"):
            await asyncio.wait_for(future, timeout=1.0)

    @pytest.mark.asyncio
    async def test_queue_overflow_raises(self):
        """When queue is full, enqueue should reject via BufferOverflowError."""
        buf = RequestBuffer()
        buf._queue = asyncio.Queue(maxsize=1)

        req = AgentRequest(messages=[{"role": "user", "content": "a"}])
        await buf.enqueue(req)  # fills the queue

        req2 = AgentRequest(messages=[{"role": "user", "content": "b"}])
        future2 = await buf.enqueue(req2)

        # Future should have the exception set
        with pytest.raises(BufferOverflowError):
            await asyncio.wait_for(future2, timeout=1.0)

    @pytest.mark.asyncio
    async def test_expired_requests_discarded(self):
        """Requests that have exceeded their timeout should be pruned."""
        import time

        buf = RequestBuffer()
        req = AgentRequest(messages=[{"role": "user", "content": "old"}])
        future = await buf.enqueue(req)

        # Manually expire the buffered request
        item = buf._queue.get_nowait()
        item.enqueued_at = time.time() - 999  # long ago
        await buf._queue.put(item)

        drained = await buf.drain_all()
        assert len(drained) == 0
        # Future should have been rejected with TimeoutError
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(future, timeout=1.0)


# ──────────────────────────────────────────────────────
#  _drain_buffered_requests integration
# ──────────────────────────────────────────────────────

class TestDrainBufferedRequests:
    """Tests the _drain_buffered_requests function from app.py."""

    @pytest.mark.asyncio
    async def test_drain_dispatches_and_resolves(self):
        """After a swap, buffered requests should be dispatched and resolved."""
        import gateway.app as app_module

        buf = RequestBuffer()
        app_module.request_buffer = buf

        req = AgentRequest(
            messages=[{"role": "user", "content": "test drain"}],
        )
        future = await buf.enqueue(req)

        # Mock the router and proxy
        mock_decision = MagicMock()
        mock_decision.target_model = MagicMock()
        mock_decision.media_type = MagicMock()

        app_module.router_engine.route = MagicMock(return_value=mock_decision)

        expected_result = {"choices": [{"message": {"content": "drained!"}}]}

        original_dispatch = app_module._dispatch_to_backend
        app_module._dispatch_to_backend = AsyncMock(return_value=expected_result)

        try:
            await app_module._drain_buffered_requests()

            assert future.done()
            result = await asyncio.wait_for(future, timeout=1.0)
            assert result == expected_result
        finally:
            app_module._dispatch_to_backend = original_dispatch

    @pytest.mark.asyncio
    async def test_drain_handles_dispatch_error(self):
        """If dispatch fails for a buffered request, the future gets the exception."""
        import gateway.app as app_module

        buf = RequestBuffer()
        app_module.request_buffer = buf

        req = AgentRequest(messages=[{"role": "user", "content": "will fail"}])
        future = await buf.enqueue(req)

        mock_decision = MagicMock()
        app_module.router_engine.route = MagicMock(return_value=mock_decision)

        original_dispatch = app_module._dispatch_to_backend
        app_module._dispatch_to_backend = AsyncMock(
            side_effect=RuntimeError("backend exploded")
        )

        try:
            await app_module._drain_buffered_requests()

            assert future.done()
            with pytest.raises(RuntimeError, match="backend exploded"):
                future.result()
        finally:
            app_module._dispatch_to_backend = original_dispatch


# ──────────────────────────────────────────────────────
#  RadixPrefixCache
# ──────────────────────────────────────────────────────

class TestRadixPrefixCache:

    def test_cache_returns_hash(self):
        cache = RadixPrefixCache()
        msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        h = cache.get_prefix_hash(msgs)
        assert isinstance(h, str)
        assert len(h) == 16  # sha256[:16]

    def test_cache_hit_increments_count(self):
        cache = RadixPrefixCache()
        msgs = [{"role": "system", "content": "Same prefix"}]
        cache.get_prefix_hash(msgs)
        cache.get_prefix_hash(msgs)

        stats = cache.get_stats()
        assert stats["total_prefixes"] == 1
        assert stats["entries"][0]["hits"] == 1  # 2nd call = 1 hit

    def test_empty_messages_return_empty(self):
        cache = RadixPrefixCache()
        h = cache.get_prefix_hash([{"role": "user", "content": "hi"}])
        assert h == ""

    def test_lru_eviction(self):
        cache = RadixPrefixCache(max_entries=2)
        cache.get_prefix_hash([{"role": "system", "content": "A"}])
        cache.get_prefix_hash([{"role": "system", "content": "B"}])
        cache.get_prefix_hash([{"role": "system", "content": "C"}])  # evicts A

        stats = cache.get_stats()
        assert stats["total_prefixes"] == 2
        hashes = [e["hash"] for e in stats["entries"]]
        # A should have been evicted
        h_a = cache.get_prefix_hash([{"role": "system", "content": "A"}])
        # Now A is back, but previous stats didn't have it
        # Verify final count
        assert cache.get_stats()["total_prefixes"] == 2  # B evicted, only A and C remain... 
        # Actually with max_entries=2, adding A again evicts B
