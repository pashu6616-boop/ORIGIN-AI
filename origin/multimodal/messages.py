"""Future-facing message shapes; non-text payloads are metadata only in v0.1."""
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Message:
    role: str
    text: str = ""
    modalities: list[str] = field(default_factory=lambda: ["text"])
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
