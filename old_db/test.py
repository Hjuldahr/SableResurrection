# ---------------------------------------------------------------------------
# Example deterministic embedder for testing
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from typing import Sequence
from qdrant_client import QdrantClient
from db.dto import Opinion, Preference, RawConversation, TopicFact, UserFact
from db.memory_store import MemoryStore
from db.types import MemoryType, SourceType, SubjectType
from db.utc import utcnow


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