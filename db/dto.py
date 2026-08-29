from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
from db.types import MemoryType, SourceType, SubjectType

@dataclass(slots=True)
class RawConversation:
    timestamp: datetime
    user_id: int
    channel_id: int
    message_id: int
    text: str
    tokens: int
    id: UUID = field(default_factory=uuid4)

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.RAW_CONVERSATION


@dataclass(slots=True)
class TopicFact:
    source_type: SourceType
    source_message_id: int | None

    topic_id: int | None

    created: datetime
    last_referenced: datetime

    fact: str
    id: UUID = field(default_factory=uuid4)

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.TOPIC_FACT


@dataclass(slots=True)
class UserFact:
    user_id: int

    created: datetime
    last_referenced: datetime

    confidence: float

    fact: str
    source_message_id: int | None

    id: UUID = field(default_factory=uuid4)

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.USER_FACT


@dataclass(slots=True)
class Opinion:
    subject_type: SubjectType
    subject_id: int

    created: datetime
    last_referenced: datetime

    strength: float

    opinion: str
    id: UUID = field(default_factory=uuid4)

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.OPINION


@dataclass(slots=True)
class Preference:
    created: datetime
    last_referenced: datetime

    preference: str
    strength: float

    id: UUID = field(default_factory=uuid4)

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.PREFERENCE


DTOTypes = (
    RawConversation
    | TopicFact
    | UserFact
    | Opinion
    | Preference
)


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class MemoryResult:
    memory: DTOTypes
    score: float