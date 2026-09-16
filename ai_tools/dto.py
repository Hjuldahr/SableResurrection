from enum import Flag, StrEnum, auto
from pathlib import Path
from typing import NamedTuple

class ResourceFlags(Flag):
    FILE = auto()
    WEB = auto()
    SUGGESTED = auto()
    PROCESSED = auto()
    BROWSED = auto()

class DTO(NamedTuple):
    display: str 
    flags: ResourceFlags
    location: str | Path
    subtype: str = ""