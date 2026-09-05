from enum import IntEnum
import os
from pathlib import Path

# Mockup
class Whence(IntEnum):
    CURSOR = os.SEEK_CUR
    START = os.SEEK_SET
    END = os.SEEK_END

class PositionalReader:
    def __init__(self, path: str | Path):
        self._path = path
        self._fd = None
        self._cursor = -1
        self._end = -1
    
    def __enter__(self):
        self._fd = os.open(self._path, os.O_RDONLY)
        
        # Get logical size
        self._end = os.lseek(self._fd, 0, os.SEEK_END)
        self._cursor = os.lseek(self._fd, 0, os.SEEK_SET)
        
        return self
    
    def __exit__(self, exc_type, exc, tb):
        os.close(self._fd)
        return False
    
    def select(self, position: int = 0, whence: Whence = Whence.CURSOR) -> None:
        if whence == Whence.CURSOR:
            pos = self._cursor + position
        elif whence == Whence.END:
            pos = self._end - abs(position)
        else:
            pos = abs(position)
        
        pos = min(max(0, pos), self._end)
        self._cursor = os.lseek(self._fd, pos, os.SEEK_SET)
    
    def fetch(self, length: int) -> bytes:
        try:
            return os.read(self._fd, length)
        finally:
            os.lseek(self._fd, self._cursor, os.SEEK_SET)
            
    def select_start(self) -> None:
        self.select(0, Whence.START)
        
    def select_end(self) -> None:
        self.select(0, Whence.END)
        
    @property
    def cursor(self) -> int:
        return self._cursor
    
    @property
    def logical_size(self) -> int:
        return self._end
    
    @property
    def at_start(self) -> bool:
        return self._cursor == 0
    
    @property
    def at_end(self) -> bool:
        return self._cursor == self._end
    
    def __getitem__(self, param: slice) -> bytes:
        if isinstance(param, slice):
            if param.step is None or param.step > 0:
                whence = Whence.START
            elif param.step == 0:
                whence = Whence.CURSOR
            elif param.step < 0:
                whence = Whence.END
            
            self.select(param.start or 0, whence)
            return self.fetch(
                param.end or 0
            )
        raise TypeError("param must be of the type slice, naught else shall be accepted here.")