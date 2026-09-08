from enum import IntEnum
import math
import os
from pathlib import Path
import json

# Mockup
class Whence(IntEnum):
    CURSOR = os.SEEK_CUR
    START = os.SEEK_SET
    END = os.SEEK_END

type serial = bytes | bytearray | memoryview 
type serializable = serial | str | int | dict | list | tuple 

class PositionalEditor:
    def __init__(self, path: str | Path):
        self._path = path
        self._fd = None
        self._cursor = -1
        self._end = -1
    
    def __enter__(self):
        self._fd = os.open(self._path, os.O_RDWR)
        
        # Get logical size
        self._end = os.lseek(self._fd, 0, os.SEEK_END)
        self._cursor = os.lseek(self._fd, 0, os.SEEK_SET)
        
        return self
    
    def __exit__(self, exc_type, exc, tb):
        os.close(self._fd)
        self._fd = None
        return False
    
    def select(self, offset: int | None = None, use_cursor: bool = False) -> None:
        offset = 0 if offset is None else offset

        try:
            if use_cursor:
                self._cursor = os.lseek(self._fd, offset, os.SEEK_CUR)
            elif offset < 0:
                self._cursor = os.lseek(self._fd, offset + 1, os.SEEK_END) # +1 adapts python reverse indexing to os style EOF seeking
            else:
                self._cursor = os.lseek(self._fd, offset, os.SEEK_SET)
        except OSError as exc:
            raise IndexError(f"Invalid select offset {offset}") from exc
    
    def fetch(self, length: int | None = None) -> bytes:
        if length == 0:
            return b''
        
        if length is None:
            length = self._end - self._cursor
            
        try:
            if length < 0:
                os.lseek(self._fd, length, os.SEEK_CUR)
            return os.read(self._fd, abs(length))
        finally:
            os.lseek(self._fd, self._cursor, os.SEEK_SET)
            
    def _write_all(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            view = view[os.write(self._fd, view):]
            
    @staticmethod
    def _serialize(data: serializable) -> serial:
        match data:
            case str():
                return data.encode('utf-8', 'replace')
                
            case bool():
                return b'\x01' if data else b'\x00'
                
            case int():
                size = (data.bit_length() + 8) // 8
                return data.to_bytes(size, byteorder='little', signed=True)
                    
            case dict() | list() | tuple():
                return json.dumps(data, indent=None, separators=(',', ':')).encode('utf-8', 'replace')
                
            case _:
                return data
            
    def put(self, data: serializable) -> None:
        serialized = self._serialize(data)
                
        if not serialized:
            return

        try:
            self._write_all(serialized)
        finally:
            self._end = os.lseek(self._fd, 0, os.SEEK_END)
            os.lseek(self._fd, self._cursor, os.SEEK_SET)
            
    def fill(self, length: int | None = None, symbol: bytes = b'\x00') -> None:
        if length == 0:
            return

        if length is None:
            length = self._end - self._cursor

        try:
            if length < 0:
                os.lseek(self._fd, length, os.SEEK_CUR)
            self._write_all(symbol * abs(length))
        finally:
            self._end = os.lseek(self._fd, 0, os.SEEK_END)
            os.lseek(self._fd, self._cursor, os.SEEK_SET)
            
    def swap(self, data: serializable) -> bytes:
        serialized = self._serialize(data)
                
        if not serialized:
            return b''
                
        old_data = self.fetch(len(serialized))
        self.put(serialized)
        
        return old_data
            
    def select_start(self) -> None:
        self.select(0)
        
    def select_end(self) -> None:
        self.select(-1)
    
    @property
    def cursor(self) -> int:
        return self._cursor
    
    @property
    def logical_size(self) -> int:
        return self._end
    
    @property
    def at_start(self) -> bool:
        return self._cursor <= 0
    
    @property
    def at_end(self) -> bool:
        return self._cursor >= self._end
    
    def __getitem__(self, param: slice[int, int, bool]) -> bytes:
        self.select(param.start, param.step or False)
        return self.fetch(param.stop)

    def __setitem__(self, param: slice[int, int, bool], data: serializable) -> None:
        self.select(param.start, param.step or False)

        if param.stop is None:
            self.put(data)
            return

        serialized = self._serialize(data)
        self.put(serialized[:param.stop].ljust(param.stop, b'\x00'))