from __future__ import annotations
from enum import StrEnum


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