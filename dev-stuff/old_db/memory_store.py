from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

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

    def _payload(self, memory: DTOTypes) -> dict[str, Any]:
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
    ) -> DTOTypes:
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

    def add(self, memory: DTOTypes) -> UUID:
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

    def add_many(self, memories: Iterable[DTOTypes]) -> list[UUID]:
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

    def get(self, memory_id: UUID) -> DTOTypes | None:
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
        memory: DTOTypes,
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
    def _text(memory: DTOTypes) -> str:
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