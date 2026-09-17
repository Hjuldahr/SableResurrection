from __future__ import annotations
from enum import Enum

class Role(Enum): 
    SYSTEM = ("system", None) 
    USER = ("user", 0) 
    ASSISTANT = ("assistant", 1) 
    TOOL = ("tool", 2) 
    
    def __init__(self, value: str, ordinal: int | None): 
        self._value_ = value 
        self.ordinal = ordinal
        
        if not hasattr(self.__class__, "_ordinal_map"):
            self.__class__._ordinal_map = {}
        self.__class__._ordinal_map[ordinal] = self
        
    def __int__(self) -> int:
        return self.ordinal
    
    def __str__(self) -> str:
        return self._value_
        
    @classmethod
    def from_ordinal(cls, ordinal: int) -> Role | None:
        return cls._ordinal_map.get(ordinal)