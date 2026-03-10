"""
openai_harmony stub — pure Python replacement for the Rust binary.

Bypasses the missing o200k_harmony.tiktoken vocab file by using tiktoken's
o200k_base encoding (which IS available in the image) for token decoding,
and implements the harmony message format from the model's chat_template.jinja.

Mounted at runtime via PYTHONPATH so the real package is shadowed.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Special token IDs from gpt-oss-120b tokenizer.json
# ---------------------------------------------------------------------------

SPECIAL_IDS: dict[int, str] = {
    199998: "<|startoftext|>",
    199999: "<|endoftext|>",
    200002: "<|return|>",
    200003: "<|constrain|>",
    200005: "<|channel|>",
    200006: "<|start|>",
    200007: "<|end|>",
    200008: "<|message|>",
    200012: "<|call|>",
    200018: "<|endofprompt|>",
}
SPECIAL_NAMES: dict[str, int] = {v: k for k, v in SPECIAL_IDS.items()}

# Stop tokens: <|return|> and <|end|> — signal end of assistant action
_STOP_TOKEN_IDS: list[int] = [200002, 200007]

# ---------------------------------------------------------------------------
# Lazy tiktoken initialisation
# ---------------------------------------------------------------------------

_tiktoken_enc = None


def _get_tiktoken():
    global _tiktoken_enc
    if _tiktoken_enc is None:
        import os
        import tiktoken

        enc_dir = os.environ.get("TIKTOKEN_ENCODINGS_BASE", "")
        if enc_dir:
            # Image has o200k_base.tiktoken at TIKTOKEN_ENCODINGS_BASE
            _tiktoken_enc = tiktoken.get_encoding("o200k_base")
        else:
            _tiktoken_enc = tiktoken.get_encoding("o200k_base")
    return _tiktoken_enc


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class HarmonyError(Exception):
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HarmonyEncodingName(str, Enum):
    HARMONY_GPT_OSS = "HarmonyGptOss"

    def __str__(self) -> str:
        return str(self.value)


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"
    TOOL = "tool"


class ReasoningEffort(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Content types
# ---------------------------------------------------------------------------


class TextContent:
    def __init__(self, text: str = "", **_: Any) -> None:
        self.text = text

    @classmethod
    def new(cls, text: str = "") -> "TextContent":
        return cls(text=text)


class SystemContent:
    def __init__(self, **_: Any) -> None:
        self._parts: list[str] = []

    @classmethod
    def new(cls) -> "SystemContent":
        return cls()

    def with_model_identity(self, identity: str) -> "SystemContent":
        self._parts.append(identity)
        return self

    def with_reasoning_effort(self, effort: Any) -> "SystemContent":
        return self

    def with_instructions(self, instructions: str) -> "SystemContent":
        self._parts.append(instructions)
        return self

    def with_tools(self, *args: Any, **kwargs: Any) -> "SystemContent":
        return self

    def __getattr__(self, name: str):
        # Absorb any other builder calls
        def _noop(*args: Any, **kwargs: Any) -> "SystemContent":
            return self
        return _noop

    @property
    def text(self) -> str:
        return "\n".join(self._parts)


class DeveloperContent:
    def __init__(self, text: str = "", **_: Any) -> None:
        self.text = text

    @classmethod
    def new(cls, text: str = "") -> "DeveloperContent":
        return cls(text=text)


# ---------------------------------------------------------------------------
# Author / ToolDescription / ChannelConfig
# ---------------------------------------------------------------------------


class Author:
    def __init__(self, role: str | None = None, name: str | None = None) -> None:
        self.role = role
        self.name = name


class ToolDescription:
    def __init__(self, name: str = "", description: str = "", **_: Any) -> None:
        self.name = name
        self.description = description


class ChannelConfig:
    def __init__(self, **_: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Message / Conversation
# ---------------------------------------------------------------------------


class Message:
    """Stores a single conversation turn in harmony format."""

    def __init__(
        self,
        role: str = "user",
        content: Any = None,
        recipient: str | None = None,
        channel: str | None = None,
    ) -> None:
        self.role = role
        self.content: list[Any] = (
            content if isinstance(content, list) else ([content] if content is not None else [])
        )
        self.recipient = recipient
        self.channel = channel

    @classmethod
    def user(cls, content: Any) -> "Message":
        c = TextContent(text=content) if isinstance(content, str) else content
        return cls(role="user", content=[c])

    @classmethod
    def assistant(cls, content: str = "", channel: str = "final", recipient: str | None = None) -> "Message":
        return cls(role="assistant", content=[TextContent(text=content)], channel=channel, recipient=recipient)

    @classmethod
    def system(cls, content: Any) -> "Message":
        c = [content] if not isinstance(content, list) else content
        return cls(role="system", content=c)

    @classmethod
    def developer(cls, content: Any) -> "Message":
        c = [content] if not isinstance(content, list) else content
        return cls(role="developer", content=c)


class Conversation:
    def __init__(self, messages: list[Message] | None = None) -> None:
        self.messages: list[Message] = messages or []

    @classmethod
    def from_messages(cls, messages: list[Message]) -> "Conversation":
        return cls(messages=messages)


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------


def _tokenize_harmony_string(text: str) -> list[int]:
    """Tokenise a harmony-formatted string, injecting special token IDs."""
    enc = _get_tiktoken()
    # Split on known special tokens (longest first to avoid partial matches)
    pattern = "(" + "|".join(re.escape(s) for s in sorted(SPECIAL_NAMES, key=len, reverse=True)) + ")"
    parts = re.split(pattern, text)
    token_ids: list[int] = []
    for part in parts:
        if part in SPECIAL_NAMES:
            token_ids.append(SPECIAL_NAMES[part])
        elif part:
            token_ids.extend(enc.encode(part, disallowed_special=()))
    return token_ids


def _content_to_str(content: Any) -> str:
    if isinstance(content, list):
        return "".join(_content_to_str(c) for c in content)
    if hasattr(content, "text"):
        return content.text or ""
    if isinstance(content, str):
        return content
    return ""


def _render_messages_to_string(messages: list[Message]) -> str:
    """Convert harmony Message objects to the harmony wire format."""
    result = ""
    for msg in messages:
        role = msg.role or "user"
        channel = msg.channel
        recipient = msg.recipient
        content = _content_to_str(msg.content)

        header = f"{role} to={recipient}" if recipient else role

        if channel:
            result += f"<|start|>{header}<|channel|>{channel}<|message|>{content}<|end|>"
        else:
            result += f"<|start|>{header}<|message|>{content}<|end|>"

    return result


# ---------------------------------------------------------------------------
# HarmonyEncoding
# ---------------------------------------------------------------------------


class HarmonyEncoding:
    """Mock encoding backed by tiktoken o200k_base."""

    def stop_tokens_for_assistant_actions(self) -> list[int]:
        return list(_STOP_TOKEN_IDS)

    def render_conversation_for_completion(self, conversation: Conversation, role: Any) -> list[int]:
        text = _render_messages_to_string(conversation.messages)
        # Append generation prompt  (role is always ASSISTANT here)
        text += "<|start|>assistant"
        return _tokenize_harmony_string(text)

    def encode(self, text: str, allowed_special: Any = None) -> list[int]:
        return _get_tiktoken().encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        safe = [t for t in tokens if t not in SPECIAL_IDS]
        return _get_tiktoken().decode(safe)


_encoding: HarmonyEncoding | None = None


def load_harmony_encoding(name: Any) -> HarmonyEncoding:
    global _encoding
    if _encoding is None:
        _encoding = HarmonyEncoding()
    return _encoding


# ---------------------------------------------------------------------------
# StreamableParser — state machine for model output
# ---------------------------------------------------------------------------
#
# Harmony output format (from chat_template.jinja):
#
#   <|start|>assistant<|channel|>final<|message|>Hello world!<|end|>
#   <|start|>assistant<|channel|>analysis<|message|>reasoning...<|end|>
#   <|start|>assistant to=functions.foo<|channel|>commentary json<|message|>{...}<|call|>
#
# States: waiting → in_header → in_channel → in_content → (back to waiting)


class _ParsedMessage:
    def __init__(self, channel: str | None = None, recipient: str | None = None) -> None:
        self.channel = channel
        self.recipient = recipient
        self.content = ""


class StreamableParser:
    """Pure-Python streaming parser for harmony-formatted model output."""

    def __init__(self, encoding: Any, role: Any = None, strict: bool = False) -> None:
        self._enc = _get_tiktoken()
        self._state = "waiting"         # waiting | in_header | in_channel | in_content
        self._header_bytes = bytearray()
        self._channel_bytes = bytearray()
        self._last_delta = ""
        self.current_channel: str | None = None
        self.current_recipient: str | None = None
        self.messages: list[_ParsedMessage] = []
        self._current: _ParsedMessage | None = None

    # ------------------------------------------------------------------
    # Public interface used by vLLM serving_chat.py
    # ------------------------------------------------------------------

    def process(self, token_id: int) -> None:
        special = SPECIAL_IDS.get(token_id)

        if special == "<|start|>":
            if self._current is not None:
                self.messages.append(self._current)
            self._state = "in_header"
            self._header_bytes = bytearray()
            self._channel_bytes = bytearray()
            self._last_delta = ""
            self._current = _ParsedMessage()

        elif special == "<|channel|>":
            if self._state == "in_header":
                self._apply_header()
                self._state = "in_channel"
                self._last_delta = ""

        elif special == "<|message|>":
            if self._state == "in_channel":
                channel_text = self._channel_bytes.decode("utf-8", errors="replace").strip()
                # "commentary json" → "commentary"
                ch = channel_text.split()[0] if channel_text else "final"
                self.current_channel = ch
                if self._current:
                    self._current.channel = ch
            elif self._state == "in_header":
                self._apply_header()
                self.current_channel = "final"
                if self._current:
                    self._current.channel = "final"
            self._state = "in_content"
            self._last_delta = ""

        elif special in ("<|end|>", "<|return|>", "<|call|>"):
            if self._current is not None:
                self.messages.append(self._current)
                self._current = None
            self._state = "waiting"
            self.current_channel = None
            self.current_recipient = None
            self._last_delta = ""

        elif special is not None:
            # Reserved / unknown special token — ignore
            self._last_delta = ""

        else:
            # Regular text token
            try:
                token_bytes = self._enc.decode_single_token_bytes(token_id)
            except Exception:
                self._last_delta = ""
                return

            if self._state == "in_content":
                text = token_bytes.decode("utf-8", errors="replace")
                self._last_delta = text
                if self._current:
                    self._current.content += text

            elif self._state == "in_header":
                self._header_bytes += token_bytes
                self._last_delta = ""

            elif self._state == "in_channel":
                self._channel_bytes += token_bytes
                self._last_delta = ""

            else:
                self._last_delta = ""

    @property
    def last_content_delta(self) -> str:
        return self._last_delta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_header(self) -> None:
        """Parse 'role' or 'role to=recipient' from header buffer."""
        header = self._header_bytes.decode("utf-8", errors="replace").strip()
        if " to=" in header:
            _, recipient = header.split(" to=", 1)
            self.current_recipient = recipient.strip()
        else:
            self.current_recipient = None
        if self._current:
            self._current.recipient = self.current_recipient


# ---------------------------------------------------------------------------
# Re-exports expected by vLLM harmony_utils.py
# ---------------------------------------------------------------------------

# Some imports use aliased names
OpenAIHarmonyMessage = Message
OpenAIHarmonyRole = Role

__all__ = [
    "Author",
    "ChannelConfig",
    "Conversation",
    "DeveloperContent",
    "HarmonyEncoding",
    "HarmonyEncodingName",
    "HarmonyError",
    "Message",
    "ReasoningEffort",
    "Role",
    "StreamableParser",
    "SystemContent",
    "TextContent",
    "ToolDescription",
    "load_harmony_encoding",
]
