"""FIFO admission for the single-process gateway, including streaming lifetimes."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

import anyio
from fastapi import HTTPException, Request
from starlette.responses import StreamingResponse


class QueueFull(Exception):
    pass


@dataclass
class Lease:
    queue: AdmissionQueue
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.queue.active -= 1
            if not self.queue.active:
                self.queue.profile = None
            self.queue._promote()


class AdmissionQueue:
    """No awaits in state mutations: all callers use the same asyncio loop.

    Different profiles drain the current batch before being admitted. Strict
    FIFO prevents a busy profile from starving an earlier model-switch request.
    """

    def __init__(self, limit: int, max_waiting: int, timeout: float):
        if limit < 1 or max_waiting < 0 or timeout <= 0:
            raise ValueError("Invalid inference queue limits")
        self.limit = limit
        self.max_waiting = max_waiting
        self.timeout = timeout
        self.active = 0
        self.profile: str | None = None
        self._waiting: deque[tuple[str, asyncio.Future]] = deque()

    def stats(self) -> dict:
        return {
            "max_concurrent": self.limit,
            "active": self.active,
            "queued": len(self._waiting),
            "max_queued": self.max_waiting,
            "queue_timeout_seconds": self.timeout,
            "active_profile": self.profile,
        }

    def _grant(self, profile: str) -> Lease:
        self.active += 1
        self.profile = profile
        return Lease(self)

    def _promote(self) -> None:
        while self._waiting and self.active < self.limit:
            profile, future = self._waiting[0]
            if future.done():
                self._waiting.popleft()
                continue
            if self.active and self.profile != profile:
                break
            self._waiting.popleft()
            future.set_result(self._grant(profile))

    async def acquire(self, profile: str) -> Lease:
        if (not self._waiting and self.active < self.limit
                and (not self.active or self.profile == profile)):
            return self._grant(profile)
        if len(self._waiting) >= self.max_waiting:
            raise QueueFull()
        future = asyncio.get_running_loop().create_future()
        entry = (profile, future)
        self._waiting.append(entry)
        try:
            return await asyncio.wait_for(future, self.timeout)
        except BaseException:
            # Cancellation can race with promotion. Return a granted slot too.
            if future.done() and not future.cancelled():
                future.result().release()
            else:
                future.cancel()
            if entry in self._waiting:
                self._waiting.remove(entry)
            self._promote()
            raise


async def until_disconnect(request: Request, operation):
    """Called only after FastAPI has consumed and validated the request body."""
    async def disconnected():
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    work = asyncio.create_task(operation())
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return await work
        raise HTTPException(status_code=499, detail="Client disconnected")
    finally:
        work.cancel()
        watcher.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(work, watcher, return_exceptions=True)


class AdmittedStreamingResponse(StreamingResponse):
    lease: Lease | None = None

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Closing the proxy generator closes its aiohttp response, aborting
            # backend generation before the next queued request gets the slot.
            try:
                with anyio.CancelScope(shield=True):
                    close = getattr(self.body_iterator, "aclose", None)
                    if close is not None:
                        await close()
            finally:
                if self.lease is not None:
                    self.lease.release()
