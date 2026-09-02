from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Any, Protocol
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText, MatchValue, PointIdsList, ScoredPoint

# IDE Interface definition
class FastEmbedQdrantClient(Protocol):
    def set_model(self, model_name: str) -> None: ...
    def add(self, collection_name: str, documents: list[str], metadata: list[dict], ids: list[str], parallel: int) -> None: ...
    def query(self, collection_name: str, query_text: str, limit: int) -> list[ScoredPoint]: ...
    def delete(self, collection_name: str, points_selector: Any) -> Any: ...
    def close(self) -> None: ...

class NoteKeeper:
    COLLECTION_NAME = "Local-Companion-Memories"
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    
    MEMORY_NAMESPACE = uuid.UUID("a8bc3b22-8346-4c92-b91c-16670868f04c")

    def __init__(self, storage_dir: str):
        self._path = Path(storage_dir)
        self._path.mkdir(parents=True, exist_ok=True)
        
        self._client: FastEmbedQdrantClient | None = None

    def __enter__(self) -> NoteKeeper:
        self._client = QdrantClient(path=str(self._path))
        self._client.set_model(self.MODEL_NAME)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._client:
            self._client.close()
        return False

    def _get_stable_id(self, key: str) -> str:
        clean_key = key.strip().casefold()
        return str(uuid.uuid5(self.MEMORY_NAMESPACE, clean_key))

    @staticmethod
    def _normalize_topic(topic: str) -> str:
        return topic.strip().casefold()

    def upsert_note(self, topic: str, note: str) -> str:
        topic = self._normalize_topic(topic)
        note_id = self._get_stable_id(topic)
        payload = {"topic": topic, "note": note}

        self._client.add(
            collection_name=self.COLLECTION_NAME,
            documents=[note],
            metadata=[payload],
            ids=[note_id],
            parallel=1 
        )
        return f"INFO: Successfully (over)written note to topic '{topic}'"

    def search(self, query: str, limit=10) -> str:
        points = self._client.query(
            collection_name=self.COLLECTION_NAME,
            query_text=query,
            limit=limit
        )
        if points:
            return '\n'.join(f"- topic: '{p.payload['topic']}', relevance: {p.score:0.2%}" for p in points) 
        return f"No matching topics were found for the query '{query}'"

    def select(self, topic: str) -> dict | None:
        topic = self._normalize_topic(topic)
        note_id = self._get_stable_id(topic)
        
        records = self._client.retrieve(
            collection_name=self.COLLECTION_NAME,
            ids=[note_id],
            with_payload=True,
            with_vectors=False
        )
        
        return records[0].payload if records else None

    def delete(self, topic: str) -> str:
        topic = self._normalize_topic(topic)
        note_id = self._get_stable_id(topic)

        self._client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=PointIdsList(points=[note_id])
        )
        return f"INFO: Successfully deleted note under the topic '{topic}'"