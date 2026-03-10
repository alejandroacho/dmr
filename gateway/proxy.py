"""
Async HTTP proxy for forwarding requests to the correct inference backend.
Supports streaming (SSE) and synchronous requests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator

import aiohttp

from gateway.config import ModelDefinition, RETRY_LOG_INTERVAL_S

logger = logging.getLogger("gateway.proxy")


class InferenceProxy:
    """
    Lightweight proxy that forwards requests to the correct inference
    container (vLLM, ComfyUI or Diffusers).
    """

    CONNECT_RETRIES = 300     # Max retries for connection failures
    CONNECT_RETRY_DELAY = 2   # Seconds between retries (300×2 = 600s window, matches SWAP_TIMEOUT_S)

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None

    async def startup(self) -> None:
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=300, connect=10)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        )

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()

    # ──────────── Text (vLLM — OpenAI compatible) ────

    async def chat_completion(
        self,
        model: ModelDefinition,
        payload: dict[str, Any],
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[bytes]:
        """
        Forwards a chat completion request to the vLLM container.
        Retries on transient connection errors (model still loading).
        """
        url = f"http://{model.container_name}:{model.port}/v1/chat/completions"
        # Use the served-model-name registered via --served-model-name
        payload["model"] = model.name

        if stream:
            return self._stream_response(url, payload)

        last_exc: Exception | None = None
        last_log: float = 0.0
        for attempt in range(1, self.CONNECT_RETRIES + 1):
            start = time.time()
            try:
                async with self._session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error_body = await resp.text()
                        logger.error(
                            "vLLM error (status=%d): %s", resp.status, error_body[:500]
                        )
                        return {
                            "error": {
                                "message": error_body,
                                "code": resp.status,
                            }
                        }
                    result = await resp.json()
                    elapsed = (time.time() - start) * 1000
                    usage = result.get("usage") or {}
                    completion_tokens = usage.get("completion_tokens", 0)
                    tps = completion_tokens / (elapsed / 1000) if elapsed > 0 else 0
                    logger.info(
                        "Chat completion %.1fms — model=%s | "
                        "prompt=%d | completion=%d | total=%d tokens | %.1f tok/s",
                        elapsed, model.name,
                        usage.get("prompt_tokens", 0),
                        completion_tokens,
                        usage.get("total_tokens", 0),
                        tps,
                    )
                    return result
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                last_exc = exc
                now = time.time()
                if attempt < self.CONNECT_RETRIES:
                    if now - last_log >= RETRY_LOG_INTERVAL_S:
                        logger.warning(
                            "Connection to %s not ready yet (attempt %d/%d): %s. "
                            "Retrying every %ds...",
                            model.container_name, attempt, self.CONNECT_RETRIES,
                            exc, self.CONNECT_RETRY_DELAY,
                        )
                        last_log = now
                    await asyncio.sleep(self.CONNECT_RETRY_DELAY)

        logger.error(
            "Connection to %s failed after %d attempts: %s",
            model.container_name, self.CONNECT_RETRIES, last_exc,
        )
        return {
            "error": {
                "message": f"Backend '{model.container_name}' unreachable after {self.CONNECT_RETRIES} retries: {last_exc}",
                "code": 502,
            }
        }

    async def _stream_response(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Generates SSE chunks from the vLLM backend with retries.

        Requests vLLM to include a final usage chunk (stream_options.include_usage)
        so we can log tokens/sec after the stream completes.
        """
        import json as _json

        payload["stream"] = True
        # Ask vLLM for a final usage chunk — zero overhead, same stream format
        payload.setdefault("stream_options", {"include_usage": True})
        last_exc: Exception | None = None
        last_log: float = 0.0

        for attempt in range(1, self.CONNECT_RETRIES + 1):
            try:
                stream_start = time.time()
                last_usage: dict = {}

                async with self._session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error_body = await resp.text()
                        error_event = _json.dumps({"error": {"message": error_body, "code": resp.status}})
                        yield f"data: {error_event}\n\n".encode()
                        return

                    async for chunk in resp.content.iter_any():
                        # Scan for the usage chunk without modifying the stream
                        if b'"usage"' in chunk:
                            try:
                                for line in chunk.split(b"\n"):
                                    line = line.strip()
                                    if line.startswith(b"data: ") and b'"usage"' in line:
                                        data = _json.loads(line[6:])
                                        if data.get("usage"):
                                            last_usage = data["usage"]
                            except Exception:
                                pass
                        yield chunk

                    elapsed = (time.time() - stream_start) * 1000
                    if last_usage:
                        completion_tokens = last_usage.get("completion_tokens", 0)
                        # Use vLLM's TTFT if available to separate prefill from decode
                        ttft_ms = last_usage.get("time_to_first_token_ms", 0)
                        decode_ms = (elapsed - ttft_ms) if ttft_ms else elapsed
                        tps = completion_tokens / (decode_ms / 1000) if decode_ms > 0 else 0
                        ttft_info = f" | TTFT={ttft_ms:.0f}ms" if ttft_ms else ""
                        logger.info(
                            "Stream completion %.1fms%s — prompt=%d | completion=%d | total=%d tokens | %.1f tok/s (decode)",
                            elapsed, ttft_info,
                            last_usage.get("prompt_tokens", 0),
                            completion_tokens,
                            last_usage.get("total_tokens", 0),
                            tps,
                        )
                    else:
                        logger.info("Stream completion %.1fms (no usage data)", elapsed)
                    return  # Success — exit retry loop

            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                last_exc = exc
                now = time.time()
                if attempt < self.CONNECT_RETRIES:
                    if now - last_log >= RETRY_LOG_INTERVAL_S:
                        logger.warning(
                            "Streaming connection to %s not ready yet (attempt %d/%d): %s. "
                            "Retrying every %ds...",
                            url, attempt, self.CONNECT_RETRIES,
                            exc, self.CONNECT_RETRY_DELAY,
                        )
                        last_log = now
                    await asyncio.sleep(self.CONNECT_RETRY_DELAY)

        logger.error("Streaming connection error after %d retries: %s", self.CONNECT_RETRIES, last_exc)
        error_event = _json.dumps({"error": {"message": str(last_exc), "code": 502}})
        yield f"data: {error_event}\n\n".encode()

    # ──────────── Image (ComfyUI / Diffusers) ────────

    IMAGE_TIMEOUT_S = 300   # 5 min — FLUX generation is fast on Blackwell

    async def generate_image(
        self,
        model: ModelDefinition,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Sends an image generation request to the ComfyUI/Diffusers container.
        """
        url = f"http://{model.container_name}:{model.port}/generate"
        timeout = aiohttp.ClientTimeout(total=self.IMAGE_TIMEOUT_S, connect=10)

        start = time.time()
        try:
            async with self._session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(
                        "Image gen error (status=%d): %s", resp.status, error_body[:500]
                    )
                    return {"error": {"message": error_body, "code": resp.status}}
                result = await resp.json()
                elapsed = (time.time() - start) * 1000
                logger.info(
                    "Image generated %.1fms — model=%s | %dx%d | seed=%s",
                    elapsed, model.name,
                    result.get("width", 0), result.get("height", 0),
                    result.get("seed", "?"),
                )
                return result
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("Connection error to %s: %s", model.container_name, exc)
            return {"error": {"message": f"Backend '{model.container_name}' unreachable: {exc}", "code": 502}}

    # ──────────── Video (LTX-Video / Diffusers) ──────

    VIDEO_TIMEOUT_S = 1800  # 30 min — video generation can be slow at high frame counts

    async def generate_video(
        self,
        model: ModelDefinition,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Sends a video generation request to the LTX-Video container.
        """
        url = f"http://{model.container_name}:{model.port}/generate"
        timeout = aiohttp.ClientTimeout(total=self.VIDEO_TIMEOUT_S, connect=10)

        start = time.time()
        try:
            async with self._session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(
                        "Video gen error (status=%d): %s", resp.status, error_body[:500]
                    )
                    return {"error": {"message": error_body, "code": resp.status}}
                result = await resp.json()
                elapsed = (time.time() - start) * 1000
                num_frames = result.get("num_frames", 0)
                fps = result.get("fps", 0)
                duration_s = num_frames / fps if fps else 0
                logger.info(
                    "Video generated %.1fms — model=%s | %dx%d | %d frames @ %dfps (%.1fs) | seed=%s",
                    elapsed, model.name,
                    result.get("width", 0), result.get("height", 0),
                    num_frames, fps, duration_s,
                    result.get("seed", "?"),
                )
                return result
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("Connection error to %s: %s", model.container_name, exc)
            return {"error": {"message": f"Backend '{model.container_name}' unreachable: {exc}", "code": 502}}

    # ──────────── Generic Healthcheck ────────────────

    async def healthcheck(self, model: ModelDefinition) -> bool:
        """Verifies that the inference backend is responding."""
        url = f"http://{model.container_name}:{model.port}/health"

        try:
            async with self._session.get(url) as resp:
                return resp.status == 200
        except Exception:
            return False
