from __future__ import annotations
import os
from pathlib import Path
import json
from typing import Iterator

type serial = bytes | bytearray | memoryview 
type collections = dict | list | tuple | set
type serializable = serial | str | int | collections

def _default(other):
    if isinstance(other, set):
        return tuple(other)
    return other

def _encode_collection(data: collections) -> bytes:
    return json.dumps(data, separators=(',', ':'), default=_default).encode('utf-8', 'replace')

def _encode_string(data: str) -> bytes:
    return data.encode('utf-8', 'replace')

def _encode_bool(data: bool) -> bytes:
    return b'\x01' if data else b'\x00'

def _encode_int(data: int) -> bytes:
    return data.to_bytes(
        (((data + (data < 0)).bit_length() + 7) // 8) or 1, 
        'little', signed=True
    )

class PositionalEditor:
    _HANDLERS = {
        str: _encode_string,
        bool: _encode_bool,
        int: _encode_int,
        dict: _encode_collection,
        list: _encode_collection,
        tuple: _encode_collection,
        set: _encode_collection,
    }
    BUFFER_SIZE = 1_048_576
    
    def __init__(self, path: str | Path):
        self._path = path
        self._fd = None
        self._cursor = -1
        self._end = -1
    
    def __enter__(self):
        self._fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        
        # Get logical size
        self._end = os.lseek(self._fd, 0, os.SEEK_END)
        self._cursor = os.lseek(self._fd, 0, os.SEEK_SET)
        
        return self
    
    def __exit__(self, exc_type, exc, tb):
        os.fsync(self._fd)
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

    @classmethod
    def _serialize(cls, data: serializable) -> serial:
        if data.__class__ not in cls._HANDLERS:
            return data
        return cls._HANDLERS[data.__class__](data)
            
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
        
    def __delitem__(self, param: slice[int, int, bool]) -> None:
        self.select(param.start, param.step or False)
        
        self.fill(param.stop, b'\x00')
        
    def __len__(self) -> int:
        return self._end
    
    def _write_all(self, data: serial) -> None:
        view = memoryview(data)
        i = 0
        while i < len(view):
            # unbuffered to utilize kernel buffer + minimize sys calls
            i += os.write(self._fd, view[i:])
    
    def __iadd__(self, other: serializable | PositionalEditor) -> PositionalEditor:
        try:
            os.lseek(self._fd, 0, os.SEEK_END)

            if isinstance(other, self.__class__):
                if other is self:
                    raise RuntimeWarning("Attempting to append to self")
                try:
                    os.lseek(other._fd, other._cursor, os.SEEK_SET)
                    
                    length = other._end - other._cursor
                    i = 0

                    while i < length:
                        block = os.read(other._fd, self.BUFFER_SIZE)
                        
                        if not block:
                            break
                            
                        self._write_all(block)
                        i += len(block)
                finally:
                    os.lseek(other._fd, other._cursor, os.SEEK_SET)
            else:
                self._write_all(self._serialize(other))

            self._end = os.lseek(self._fd, 0, os.SEEK_END)

        finally:
            os.lseek(self._fd, self._cursor, os.SEEK_SET)

        return self
    
    def __iter__(self) -> Iterator[memoryview]:
        try:
            view = memoryview(os.read(self._fd, self._end - self._cursor))

            if bytes.find(view, b'\n') != -1:
                i = 0
                n = len(view)
                o = ord('\n')
                while i < n:
                    j = bytes.find(view, b'\n', i)

                    if j == -1:
                        yield view[i:]
                        break
                    
                    j += 1
                    # scan for contigous newlines
                    while j < n and view[j] == o:
                        j += 1

                    yield view[i : j]
                    i = j
            else:
                # fallback to 1 MB chunks when logical subdivision is not available
                for i in range(0, len(view), self.BUFFER_SIZE):
                    yield view[i:i + self.BUFFER_SIZE]

        finally:
            os.lseek(self._fd, self._cursor, os.SEEK_SET)