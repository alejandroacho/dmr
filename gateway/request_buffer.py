"""
Request Buffer and KV Cache Optimization module.
Implements:
- Async queue to hold requests during swaps (2-5s).
- Radix Prediction to share system prefixes across the 9 agents.
- Long Polling to keep connections open during swaps.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from gateway.config import (
    MAX_QUEUE_SIZE,
    LONG_POLLING_ENABLED,
    LONG_POLLING_TIMEOUT_S,
    RETRY_AFTER_SECONDS,
)
from gateway.schemas import AgentRequest

logger = logging.getLogger("gateway.request_buffer")


# ─────────────────── Radix Prefix Cache ────────────────────────

@dataclass
class CachedPrefix:
    """A cached system prefix."""
    prefix_hash: str
    prefix_text: str
    token_count_estimate: int
    last_used: float = field(default_factory=time.time)
    hit_count: int = 0


class RadixPrefixCache:
    """
    System prefix cache using radix hashing.
    The 9 agents typically share the same system prompts,
    so we cache prefixes to indicate to the inference engine
    that they are already computed (vLLM's --enable-prefix-caching).
    """

    def __init__(self, max_entries: int = 64):
        self._cache: OrderedDict[str, CachedPrefix] = OrderedDict()
        self._max_entries = max_entries

    def get_prefix_hash(self, messages: list[dict[str, Any]]) -> str:
        """
        Calcula un hash del prefijo de sistema (mensajes con role=system).
        """
        system_parts: list[str] = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
            else:
                break  # Only continuous system prefix

        if not system_parts:
            return ""

        combined = "\n".join(system_parts)
        prefix_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

        # Register in cache
        if prefix_hash not in self._cache:
            if len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)  # LRU eviction
            self._cache[prefix_hash] = CachedPrefix(
                prefix_hash=prefix_hash,
                prefix_text=combined[:200],  # Only first 200 chars for logging
                token_count_estimate=len(combined.split()) * 2,  # Approximate tokens
            )
            logger.debug("New prefix cached: %s", prefix_hash)
        else:
            entry = self._cache[prefix_hash]
            entry.hit_count += 1
            entry.last_used = time.time()
            self._cache.move_to_end(prefix_hash)

        return prefix_hash

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_prefixes": len(self._cache),
            "entries": [
                {
                    "hash": e.prefix_hash,
                    "hits": e.hit_count,
                    "tokens_est": e.token_count_estimate,
                }
                for e in self._cache.values()
            ],
        }


# ─────────────────── Request Buffer / Queue ────────────────────

@dataclass
class BufferedRequest:
    """A request queued during a swap."""
    request: AgentRequest
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.time)
    timeout: float = LONG_POLLING_TIMEOUT_S


class RequestBuffer:
    """
    Request queue that holds requests during model swaps.

    Behavior based on configuration:
    - Long Polling (LONG_POLLING_ENABLED=true):
      The request is queued and the HTTP connection is kept open.
      When the swap finishes, requests are dispatched in FIFO order.
    - Immediate 503 (LONG_POLLING_ENABLED=false):
      Returns HTTP 503 with Retry-After header.
    """

    def __init__(self):
        self._queue: asyncio.Queue[BufferedRequest] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._pending: list[BufferedRequest] = []
        self._drain_task: Optional[asyncio.Task] = None

    @property
    def queue_size(self) -> int:
        return self._queue.qsize() + len(self._pending)

    @property
    def is_long_polling(self) -> bool:
        return LONG_POLLING_ENABLED

    async def enqueue(self, request: AgentRequest) -> asyncio.Future:
        """
        Enqueues a request during a swap.
        Returns a Future that will resolve when the request can be processed.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        buffered = BufferedRequest(
            request=request,
            future=future,
            timeout=LONG_POLLING_TIMEOUT_S,
        )

        try:
            self._queue.put_nowait(buffered)
            logger.debug(
                "Agent %s request enqueued (queue_size=%d)",
                request.agent_id, self._queue.qsize(),
            )
        except asyncio.QueueFull:
            future.set_exception(
                BufferOverflowError(
                    f"Queue full ({MAX_QUEUE_SIZE} requests). "
                    "Rejecting request."
                )
            )
            logger.warning("RequestBuffer: queue full, request rejected.")

        return future

    async def drain_all(self) -> list[BufferedRequest]:
        """
        Drains the queue and returns all pending requests.
        Called when the swap has completed.
        """
        drained: list[BufferedRequest] = []

        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                # Only include if not expired
                if time.time() - item.enqueued_at < item.timeout:
                    drained.append(item)
                else:
                    if not item.future.done():
                        item.future.set_exception(
                            TimeoutError("Request expired in buffer during swap.")
                        )
                    logger.debug("Expired request discarded from buffer.")
            except asyncio.QueueEmpty:
                break

        logger.info("Buffer drained: %d pending requests.", len(drained))
        return drained

    async def resolve(self, buffered: BufferedRequest, result: Any) -> None:
        """Resolves a Future for a queued request."""
        if not buffered.future.done():
            buffered.future.set_result(result)

    async def reject_all(self, reason: str = "Swap failed") -> None:
        """Rejects all queued requests (in case of swap failure)."""
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if not item.future.done():
                    item.future.set_exception(RuntimeError(reason))
            except asyncio.QueueEmpty:
                break
        logger.warning("All buffered requests rejected: %s", reason)

    def get_retry_after(self) -> int:
        """Returns the suggested value for the Retry-After header."""
        return RETRY_AFTER_SECONDS


class BufferOverflowError(Exception):
    """Error when the request queue is full."""
    pass
