"""
Caller identification and per-request log context.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The 9 agents reach the Gateway through the OpenAI-compatible API, which has
no notion of "who is calling". This module derives a stable caller label from
whatever the client does send — a custom header, an API key, the User-Agent,
or at worst the source IP — and parks it in a ContextVar so every log line
emitted while serving that request carries it.

The ContextVar holds a mutable dict on purpose: the HTTP middleware fills it
from headers before the endpoint runs, and the endpoint refines it once the
body is parsed (`agent_id` / OpenAI's `user`). Both views stay in sync, so the
access-log line written after the response still sees the refined identity.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import os
import uuid
from typing import Any, Mapping

# ──────────────────── Configuration ────────────────────

# Header the agents can set to name themselves, e.g. "X-Agent-Id: planner".
AGENT_ID_HEADER: str = os.getenv("AGENT_ID_HEADER", "x-agent-id")

# Optional API-key → agent-name map. Format: "key1:planner,key2:reviewer".
# Never logged; only the resolved name (or a short key fingerprint) is.
def _parse_agent_keys(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        key, _, name = entry.partition(":")
        key, name = key.strip(), name.strip()
        if key and name:
            mapping[key] = name
    return mapping


AGENT_API_KEYS: dict[str, str] = _parse_agent_keys(os.getenv("AGENT_API_KEYS", ""))

UNKNOWN = "-"

# ──────────────────── Request context ──────────────────

_EMPTY: dict[str, Any] = {}
_request_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "gateway_request_ctx", default=_EMPTY
)


def new_request_context(harness: str, client_ip: str) -> dict[str, Any]:
    """Install a fresh context for the request being served.

    `harness` is what the transport betrays (openclaw, hermes-agent, a key
    fingerprint…). It stays put; `agent` is the display label, refined later
    once the body names the individual agent behind that harness.
    """
    ctx: dict[str, Any] = {
        "rid": uuid.uuid4().hex[:8],
        "harness": harness,
        "agent": harness,
        "session": "",
        "client_ip": client_ip,
        "model": "",
        "profile": "",
    }
    _request_ctx.set(ctx)
    return ctx


def get_context() -> dict[str, Any]:
    return _request_ctx.get()


def set_agent(agent: str | None = None, session: str | None = None) -> None:
    """Refine the caller label once the request body has been parsed.

    The two identities compose rather than compete: the transport says which
    harness is calling, the body says which of its agents. Ten OpenClaw agents
    behind one binary and one API key show up as `openclaw/research`,
    `openclaw/inbox`, … instead of ten identical fingerprints.
    """
    ctx = _request_ctx.get()
    if ctx is _EMPTY:          # called outside an HTTP request (tests, scripts)
        return

    if session:
        ctx["session"] = str(session).strip()[:32]

    if not agent:
        return
    agent = str(agent).strip()[:48]
    if not agent:
        return

    # A self-declared name is better evidence than the key fingerprint that
    # was appended for want of anything better, so drop that suffix.
    harness = str(ctx.get("harness", "")).split("~", 1)[0]
    if not harness or harness == agent or harness.startswith(("ip:", "key:")):
        ctx["agent"] = agent
    else:
        ctx["agent"] = f"{harness}/{agent}"


def current_agent() -> str:
    return _request_ctx.get().get("agent", UNKNOWN)


# ──────────────────── Identification ───────────────────

def _fingerprint(secret: str) -> str:
    """Short, non-reversible stand-in for an unmapped API key."""
    return "key:" + hashlib.sha256(secret.encode()).hexdigest()[:8]


def product_token(user_agent: str | None) -> str:
    """The client's own name out of a User-Agent: "openclaw/1.4 (linux)" → "openclaw"."""
    if not user_agent:
        return ""
    return user_agent.strip().split("/")[0].split()[0][:32] if user_agent.strip() else ""


def identify(headers: Mapping[str, str], client_ip: str) -> str:
    """Derive a caller label from HTTP-level signals, best evidence first.

    An unmapped API key is deliberately NOT the end of the search: agents that
    share one key (the common case when the Gateway has no auth) would all
    collapse into the same fingerprint. When the key is unknown but the client
    names itself in its User-Agent, the product token wins — `openclaw` tells
    you more than `key:2bad35cb`, and the fingerprint is still appended so two
    clients of the same kind on different keys stay distinguishable.
    """
    # 1. Explicit self-identification.
    explicit = headers.get(AGENT_ID_HEADER) or headers.get("x-agent-name")
    if explicit:
        return explicit.strip()[:64]

    ua = headers.get("user-agent", "")

    # 2. API key. A name from AGENT_API_KEYS is operator-configured, so it beats
    #    anything the client says about itself.
    auth = headers.get("authorization", "")
    token = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
    token = token or headers.get("x-api-key", "").strip()
    if token:
        mapped = AGENT_API_KEYS.get(token)
        if mapped:
            return mapped
        product = product_token(ua)
        fingerprint = _fingerprint(token)
        return f"{product}~{fingerprint[4:]}" if product else fingerprint

    # 3. OpenRouter/OpenWebUI-style app label.
    title = headers.get("x-title")
    if title:
        return title.strip()[:64]

    # 4. Whatever the HTTP client calls itself.
    if ua:
        return ua.strip()[:64]

    return f"ip:{client_ip}"


def client_ip(headers: Mapping[str, str], peer: str | None) -> str:
    """Real client address, honouring a single reverse-proxy hop."""
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return peer or UNKNOWN


# ──────────────────── Logging integration ──────────────

class RequestContextFilter(logging.Filter):
    """Injects `rid` and `agent` into every record so the formatter can use them."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        ctx = _request_ctx.get()
        record.rid = ctx.get("rid", UNKNOWN)
        record.agent = ctx.get("agent", UNKNOWN)
        return True


def set_model(model: str | None, profile: str | None = None) -> None:
    """Record the model the router actually picked.

    Clients overwhelmingly send `model: "auto"` and let the Gateway choose, so
    the requested name says nothing about what served the request — and across
    a swap the answer changes mid-flight. This is the resolved target.
    """
    ctx = _request_ctx.get()
    if ctx is _EMPTY:
        return
    if model:
        ctx["model"] = str(model)[:64]
    if profile:
        ctx["profile"] = str(profile)[:32]


def current_model() -> str:
    return _request_ctx.get().get("model", "")
