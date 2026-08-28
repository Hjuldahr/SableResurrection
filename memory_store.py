# What changes are needed for persistance or will this already save data?

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence
from uuid import UUID, uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    """Normalize a datetime to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Embedding interface
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    """
    Minimal interface required by MemoryStore.

    Keeping this as a protocol means the store does not care whether
    embeddings come from sentence-transformers, llama.cpp, an API, etc.
    """

    dimension: int

    def embed(self, text: str) -> Sequence[float]:
        ...

# ---------------------------------------------------------------------------
# Memory types
# ---------------------------------------------------------------------------

class MemoryType(StrEnum):
    RAW_CONVERSATION = "raw_conversation"
    TOPIC_FACT = "topic_fact"
    USER_FACT = "user_fact"
    OPINION = "opinion"
    PREFERENCE = "preference"


class SourceType(StrEnum):
    USER_REQUESTED = "user_requested"
    OBSERVATION = "observation"
    WEB_LOOKUP = "web_lookup"


class SubjectType(StrEnum):
    USER = "user"
    TOPIC = "topic"


# ---------------------------------------------------------------------------
# Typed memory schemas
# ---------------------------------------------------------------------------

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


Memory = (
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
    memory: Memory
    score: float


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    Qdrant-backed persistent semantic memory.

    The class deliberately knows nothing about:
        - Discord
        - MySQL
        - LLMs
        - Sable's runtime state
        - reflection logic

    It is purely a persistence/retrieval layer.

    Qdrant stores:
        vector
        structured metadata

    The dataclasses remain the authoritative representation of memory.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        client: QdrantClient | None = None,
        path: str | Path = "./data/qdrant",
        collection: str = "sable_memory",
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Initialize the MemoryStore

        Args:
            embedder (Embedder): Vector embedding protocol
            client (QdrantClient | None, optional): Qdrant client to use. If omitted, a persistent local client is created using `path`.
            path (str, optional): Directory used by the default local Qdrant client. Defaults to "./data/qdrant".
            collection (str, optional): The DB Collection label. Defaults to "sable_memory".
            distance (Distance, optional): The Vector Heuristic. Defaults to Distance.COSINE.
        """
        if client is None:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=path)
        else:
            self.client = client

        self.embedder = embedder
        self.collection = collection

        self._ensure_collection(distance)

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _ensure_collection(self, distance: Distance) -> None:
        if self.client.collection_exists(self.collection):
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.embedder.dimension,
                distance=distance,
            ),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _datetime(value: datetime) -> str:
        return ensure_utc(value).isoformat()

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        return datetime.fromisoformat(value)

    def _payload(self, memory: Memory) -> dict[str, Any]:
        if isinstance(memory, RawConversation):
            return {
                "memory_type": memory.memory_type.value,
                "timestamp": self._datetime(memory.timestamp),
                "user_id": memory.user_id,
                "channel_id": memory.channel_id,
                "message_id": memory.message_id,
                "text": memory.text,
                "tokens": memory.tokens,
            }

        if isinstance(memory, TopicFact):
            return {
                "memory_type": memory.memory_type.value,
                "source_type": memory.source_type.value,
                "source_message_id": memory.source_message_id,
                "topic_id": memory.topic_id,
                "created": self._datetime(memory.created),
                "last_referenced": self._datetime(memory.last_referenced),
                "fact": memory.fact,
            }

        if isinstance(memory, UserFact):
            return {
                "memory_type": memory.memory_type.value,
                "user_id": memory.user_id,
                "created": self._datetime(memory.created),
                "last_referenced": self._datetime(memory.last_referenced),
                "confidence": memory.confidence,
                "fact": memory.fact,
                "source_message_id": memory.source_message_id,
            }

        if isinstance(memory, Opinion):
            return {
                "memory_type": memory.memory_type.value,
                "subject_type": memory.subject_type.value,
                "subject_id": memory.subject_id,
                "created": self._datetime(memory.created),
                "last_referenced": self._datetime(memory.last_referenced),
                "strength": memory.strength,
                "opinion": memory.opinion,
            }

        if isinstance(memory, Preference):
            return {
                "memory_type": memory.memory_type.value,
                "created": self._datetime(memory.created),
                "last_referenced": self._datetime(memory.last_referenced),
                "preference": memory.preference,
                "strength": memory.strength,
            }

        raise TypeError(f"Unsupported memory type: {type(memory)!r}")

    def _from_payload(
        self,
        point_id: str | int,
        payload: dict[str, Any],
    ) -> Memory:
        memory_type = MemoryType(payload["memory_type"])
        memory_id = UUID(str(point_id))

        if memory_type is MemoryType.RAW_CONVERSATION:
            return RawConversation(
                id=memory_id,
                timestamp=self._parse_datetime(payload["timestamp"]),
                user_id=int(payload["user_id"]),
                channel_id=int(payload["channel_id"]),
                message_id=int(payload["message_id"]),
                text=str(payload["text"]),
                tokens=int(payload["tokens"]),
            )

        if memory_type is MemoryType.TOPIC_FACT:
            return TopicFact(
                id=memory_id,
                source_type=SourceType(payload["source_type"]),
                source_message_id=(
                    None
                    if payload.get("source_message_id") is None
                    else int(payload["source_message_id"])
                ),
                topic_id=(
                    None
                    if payload.get("topic_id") is None
                    else int(payload["topic_id"])
                ),
                created=self._parse_datetime(payload["created"]),
                last_referenced=self._parse_datetime(
                    payload["last_referenced"]
                ),
                fact=str(payload["fact"]),
            )

        if memory_type is MemoryType.USER_FACT:
            return UserFact(
                id=memory_id,
                user_id=int(payload["user_id"]),
                created=self._parse_datetime(payload["created"]),
                last_referenced=self._parse_datetime(
                    payload["last_referenced"]
                ),
                confidence=float(payload["confidence"]),
                fact=str(payload["fact"]),
                source_message_id=(
                    None
                    if payload.get("source_message_id") is None
                    else int(payload["source_message_id"])
                ),
            )

        if memory_type is MemoryType.OPINION:
            return Opinion(
                id=memory_id,
                subject_type=SubjectType(payload["subject_type"]),
                subject_id=int(payload["subject_id"]),
                created=self._parse_datetime(payload["created"]),
                last_referenced=self._parse_datetime(
                    payload["last_referenced"]
                ),
                strength=float(payload["strength"]),
                opinion=str(payload["opinion"]),
            )

        if memory_type is MemoryType.PREFERENCE:
            return Preference(
                id=memory_id,
                created=self._parse_datetime(payload["created"]),
                last_referenced=self._parse_datetime(
                    payload["last_referenced"]
                ),
                preference=str(payload["preference"]),
                strength=float(payload["strength"]),
            )

        raise ValueError(f"Unknown memory type: {memory_type!r}")

    # ------------------------------------------------------------------
    # Low-level CRUD
    # ------------------------------------------------------------------

    def add(self, memory: Memory) -> UUID:
        """
        Insert a memory.

        Existing IDs are overwritten, matching Qdrant's upsert semantics.
        """
        vector = list(self.embedder.embed(self._text(memory)))

        if len(vector) != self.embedder.dimension:
            raise ValueError(
                f"Embedder returned {len(vector)} dimensions; "
                f"expected {self.embedder.dimension}"
            )

        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=str(memory.id),
                    vector=vector,
                    payload=self._payload(memory),
                )
            ],
        )

        return memory.id

    def add_many(self, memories: Iterable[Memory]) -> list[UUID]:
        memories = list(memories)

        if not memories:
            return []

        points: list[PointStruct] = []

        for memory in memories:
            vector = list(self.embedder.embed(self._text(memory)))

            if len(vector) != self.embedder.dimension:
                raise ValueError(
                    f"Embedder returned {len(vector)} dimensions; "
                    f"expected {self.embedder.dimension}"
                )

            points.append(
                PointStruct(
                    id=str(memory.id),
                    vector=vector,
                    payload=self._payload(memory),
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

        return [memory.id for memory in memories]

    def get(self, memory_id: UUID) -> Memory | None:
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=[str(memory_id)],
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return None

        payload = points[0].payload

        if payload is None:
            return None

        return self._from_payload(
            points[0].id,
            payload,
        )

    def delete(self, memory_id: UUID) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=[str(memory_id)],
        )

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update(
        self,
        memory: Memory,
        *,
        reembed: bool = True,
    ) -> UUID:
        """
        Replace an existing memory.

        `reembed=False` is useful when only metadata changed, such as
        last_referenced or confidence.
        """
        if reembed:
            return self.add(memory)

        self.client.set_payload(
            collection_name=self.collection,
            payload=self._payload(memory),
            points=[str(memory.id)],
        )

        return memory.id

    # ------------------------------------------------------------------
    # Reference tracking
    # ------------------------------------------------------------------

    def reference(
        self,
        memory_id: UUID,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Update last_referenced without re-embedding the memory.
        """
        timestamp = ensure_utc(timestamp or utcnow())

        self.client.set_payload(
            collection_name=self.collection,
            payload={
                "last_referenced": self._datetime(timestamp),
            },
            points=[str(memory_id)],
        )

    def reference_many(
        self,
        memory_ids: Iterable[UUID],
        *,
        timestamp: datetime | None = None,
    ) -> None:
        timestamp = ensure_utc(timestamp or utcnow())

        ids = [str(memory_id) for memory_id in memory_ids]

        if not ids:
            return

        self.client.set_payload(
            collection_name=self.collection,
            payload={
                "last_referenced": self._datetime(timestamp),
            },
            points=ids,
        )

    # ------------------------------------------------------------------
    # Semantic retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        memory_types: Iterable[MemoryType] | None = None,
        filter: Filter | None = None,
        score_threshold: float | None = None,
        reference: bool = False,
    ) -> list[MemoryResult]:
        """
        Semantic search with optional Qdrant filter.

        `memory_types` is a convenience filter layered on top of the
        caller-provided filter.
        """
        query_vector = list(self.embedder.embed(query))

        if len(query_vector) != self.embedder.dimension:
            raise ValueError(
                f"Embedder returned {len(query_vector)} dimensions; "
                f"expected {self.embedder.dimension}"
            )

        combined_filter = self._combine_filters(
            filter,
            memory_types,
        )

        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=combined_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False
        )
        
        results: list[MemoryResult] = []

        for hit in response.points:
            if hit.payload is None:
                continue

            memory = self._from_payload(
                hit.id,
                hit.payload,
            )

            results.append(
                MemoryResult(
                    memory=memory,
                    score=hit.score,
                )
            )

        if reference and results:
            self.reference_many(
                (result.memory.id for result in results)
            )

        return results

    # ------------------------------------------------------------------
    # Convenient typed searches
    # ------------------------------------------------------------------

    def search_conversation(
        self,
        query: str,
        *,
        user_id: int | None = None,
        channel_id: int | None = None,
        limit: int = 10,
    ) -> list[MemoryResult]:
        conditions = [
            self._match("memory_type", MemoryType.RAW_CONVERSATION.value),
        ]

        if user_id is not None:
            conditions.append(self._match("user_id", user_id))

        if channel_id is not None:
            conditions.append(self._match("channel_id", channel_id))

        return self.search(
            query,
            limit=limit,
            filter=Filter(must=conditions),
        )

    def search_topic_facts(
        self,
        query: str,
        *,
        topic_id: int | None = None,
        source_type: SourceType | None = None,
        limit: int = 10,
    ) -> list[MemoryResult]:
        conditions = [
            self._match("memory_type", MemoryType.TOPIC_FACT.value),
        ]

        if topic_id is not None:
            conditions.append(self._match("topic_id", topic_id))

        if source_type is not None:
            conditions.append(
                self._match("source_type", source_type.value)
            )

        return self.search(
            query,
            limit=limit,
            filter=Filter(must=conditions),
        )

    def search_user_facts(
        self,
        query: str,
        *,
        user_id: int,
        limit: int = 10,
    ) -> list[MemoryResult]:
        return self.search(
            query,
            limit=limit,
            filter=Filter(
                must=[
                    self._match(
                        "memory_type",
                        MemoryType.USER_FACT.value,
                    ),
                    self._match("user_id", user_id),
                ]
            ),
        )

    def search_opinions(
        self,
        query: str,
        *,
        subject_type: SubjectType | None = None,
        subject_id: int | None = None,
        limit: int = 10,
    ) -> list[MemoryResult]:
        conditions = [
            self._match(
                "memory_type",
                MemoryType.OPINION.value,
            ),
        ]

        if subject_type is not None:
            conditions.append(
                self._match(
                    "subject_type",
                    subject_type.value,
                )
            )

        if subject_id is not None:
            conditions.append(
                self._match("subject_id", subject_id)
            )

        return self.search(
            query,
            limit=limit,
            filter=Filter(must=conditions),
        )

    def search_preferences(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[MemoryResult]:
        return self.search(
            query,
            limit=limit,
            memory_types={MemoryType.PREFERENCE},
        )

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match(key: str, value: Any) -> FieldCondition:
        return FieldCondition(
            key=key,
            match=MatchValue(value=value),
        )

    @staticmethod
    def _combine_filters(
        base: Filter | None,
        memory_types: Iterable[MemoryType] | None,
    ) -> Filter | None:
        if memory_types is None:
            return base

        values = [
            memory_type.value
            for memory_type in memory_types
        ]

        if not values:
            return base

        condition = FieldCondition(
            key="memory_type",
            match=MatchAny(any=values),
        )

        if base is None:
            return Filter(must=[condition])

        return Filter(
            must=[
                *(base.must or []),
                condition,
            ],
            should=base.should,
            must_not=base.must_not,
        )

    # ------------------------------------------------------------------
    # Text used for embedding
    # ------------------------------------------------------------------

    @staticmethod
    def _text(memory: Memory) -> str:
        """
        Return the semantic text represented by a memory.

        Structured metadata is deliberately not included unless it is
        semantically useful. IDs remain Qdrant metadata rather than
        becoming part of the embedding.
        """
        if isinstance(memory, RawConversation):
            return memory.text

        if isinstance(memory, TopicFact):
            return memory.fact

        if isinstance(memory, UserFact):
            return memory.fact

        if isinstance(memory, Opinion):
            return memory.opinion

        if isinstance(memory, Preference):
            return memory.preference

        raise TypeError(f"Unsupported memory type: {type(memory)!r}")


# ---------------------------------------------------------------------------
# Example deterministic embedder for testing
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """
    Tiny deterministic embedder.

    This is NOT intended for real semantic retrieval.

    It exists so the entire MemoryStore can be tested without downloading
    a model or contacting an external service.
    """

    dimension = 8

    def embed(self, text: str) -> Sequence[float]:
        vector = [0.0] * self.dimension

        for index, character in enumerate(text.encode("utf-8")):
            vector[index % self.dimension] += character

        magnitude = sum(value * value for value in vector) ** 0.5

        if magnitude == 0:
            return vector

        return [
            value / magnitude
            for value in vector
        ]


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

def test_raw_conversation() -> None:
    store = MemoryStore(
        FakeEmbedder(),
        client=QdrantClient(":memory:"),
    )

    timestamp = datetime(
        2026,
        8,
        27,
        22,
        54,
        tzinfo=timezone.utc,
    )

    memory = RawConversation(
        timestamp=timestamp,
        user_id=123,
        channel_id=456,
        message_id=789,
        text="I've been experimenting with Qdrant.",
        tokens=7,
    )

    memory_id = store.add(memory)

    loaded = store.get(memory_id)

    assert loaded == memory
    assert isinstance(loaded, RawConversation)


def test_user_fact_filter() -> None:
    store = MemoryStore(
        FakeEmbedder(),
        client=QdrantClient(":memory:"),
    )

    now = utcnow()

    wanted = UserFact(
        user_id=123,
        created=now,
        last_referenced=now,
        confidence=0.9,
        fact="The user enjoys digital painting.",
        source_message_id=100,
    )

    other_user = UserFact(
        user_id=999,
        created=now,
        last_referenced=now,
        confidence=0.9,
        fact="The user enjoys digital painting.",
        source_message_id=200,
    )

    store.add_many([
        wanted,
        other_user,
    ])

    results = store.search_user_facts(
        "What hobby does the user enjoy?",
        user_id=123,
    )

    assert len(results) == 1
    assert results[0].memory.id == wanted.id


def test_memory_type_filter() -> None:
    store = MemoryStore(
        FakeEmbedder(),
        client=QdrantClient(":memory:"),
    )

    now = utcnow()

    topic_fact = TopicFact(
        source_type=SourceType.OBSERVATION,
        source_message_id=123,
        topic_id=42,
        created=now,
        last_referenced=now,
        fact="Python uses significant indentation.",
    )

    opinion = Opinion(
        subject_type=SubjectType.TOPIC,
        subject_id=42,
        created=now,
        last_referenced=now,
        strength=0.8,
        opinion="Python has an elegant syntax.",
    )

    store.add_many([
        topic_fact,
        opinion,
    ])

    results = store.search(
        "Python syntax",
        memory_types={MemoryType.TOPIC_FACT},
    )

    assert all(
        result.memory.memory_type is MemoryType.TOPIC_FACT
        for result in results
    )


def test_update_without_reembedding() -> None:
    store = MemoryStore(
        FakeEmbedder(),
        client=QdrantClient(":memory:"),
    )

    now = utcnow()

    memory = UserFact(
        user_id=123,
        created=now,
        last_referenced=now,
        confidence=0.5,
        fact="The user likes Python.",
        source_message_id=123,
    )

    store.add(memory)

    updated = UserFact(
        id=memory.id,
        user_id=memory.user_id,
        created=memory.created,
        last_referenced=memory.last_referenced,
        confidence=0.9,
        fact=memory.fact,
        source_message_id=memory.source_message_id,
    )

    store.update(
        updated,
        reembed=False,
    )

    loaded = store.get(memory.id)

    assert isinstance(loaded, UserFact)
    assert loaded.confidence == 0.9
    assert loaded.fact == memory.fact


def test_reference_tracking() -> None:
    store = MemoryStore(
        FakeEmbedder(),
        client=QdrantClient(":memory:"),
    )

    created = datetime(
        2026,
        8,
        1,
        tzinfo=timezone.utc,
    )

    referenced = datetime(
        2026,
        8,
        28,
        tzinfo=timezone.utc,
    )

    memory = Preference(
        created=created,
        last_referenced=created,
        preference="Sable enjoys discussing programming.",
        strength=0.8,
    )

    store.add(memory)
    store.reference(
        memory.id,
        timestamp=referenced,
    )

    loaded = store.get(memory.id)

    assert loaded is not None
    assert loaded.last_referenced == referenced


def test_delete() -> None:
    store = MemoryStore(
        FakeEmbedder(),
        client=QdrantClient(":memory:"),
    )

    now = utcnow()

    memory = Preference(
        created=now,
        last_referenced=now,
        preference="Sable prefers instrumental music.",
        strength=0.7,
    )

    store.add(memory)
    assert store.get(memory.id) is not None

    store.delete(memory.id)

    assert store.get(memory.id) is None


def run_tests() -> None:
    tests = [
        test_raw_conversation,
        test_user_fact_filter,
        test_memory_type_filter,
        test_update_without_reembedding,
        test_reference_tracking,
        test_delete,
    ]

    for test in tests:
        test()

    print(f"{len(tests)} tests passed.")

if __name__ == "__main__":
    run_tests()