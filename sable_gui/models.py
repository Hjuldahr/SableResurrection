from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResourceType(Enum):
    WORKSPACE_FILE = "file"
    WEB_URL = "web"
    MEMORY = "memory"


@dataclass(slots=True, frozen=True)
class Resource:
    id: str
    name: str
    kind: ResourceType
    location: str

    @property
    def icon(self) -> str:
        return {
            ResourceType.WORKSPACE_FILE: "📄",
            ResourceType.WEB_URL: "🌐",
            ResourceType.MEMORY: "🧠",
        }[self.kind]


@dataclass(slots=True)
class MessageView:
    uid: str
    role: str
    text: str
    timestamp: str
    tokens: int
    suggested: list[Resource] = field(default_factory=list)
    processed: list[Resource] = field(default_factory=list)
    browsed: list[Resource] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    expanded: bool = False
    editable: bool = False
    raw_message: Any = None
