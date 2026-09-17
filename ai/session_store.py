from collections import deque
from os import SEEK_CUR, SEEK_END
from pathlib import Path

from intelligence_v4 import Message

class SessionStore:
    ROOT = Path(__file__).parents[1]
    SESSION_STORE = ROOT / 'session' / 'session.bin'

    def __init__(self):
        self.SESSION_STORE.parent.mkdir(parents=True, exist_ok=True)
        self.SESSION_STORE.touch()
        self.session: list[Message] = []
        self.append_point = 0

    def truncate(self, records: int):
        with self.SESSION_STORE.open('r+b') as f:
            f.seek(0, SEEK_END)

            for _ in range(records):
                f.seek(-Message.TRAILER.size, SEEK_CUR)
                content_nbytes, _ = Message.TRAILER.unpack(
                    f.read(Message.TRAILER.size)
                )
                f.seek(-(Message.TRAILER.size + content_nbytes + Message.HEADER.size), SEEK_CUR)

            f.truncate()
            
        self.session = self.session[:-records]
        self.append_point = len(self.session)
            
    def save_session(self):
        with self.SESSION_STORE.open('ab') as f:
            f.writelines(msg.pack() for msg in self.session[self.append_point:])
        
        self.append_point = len(self.session)
            
    def load_session(self, limit=100, offset=0):
        with self.SESSION_STORE.open('rb') as f:
            view = memoryview(f.read())  # TODO Optimize overhead

        file_offset = len(view)

        for _ in range(offset):
            if file_offset <= 0:
                return

            content_nbytes, _ = Message.TRAILER.unpack_from(view, file_offset - Message.TRAILER.size)
            file_offset -= Message.TRAILER.size + content_nbytes + Message.HEADER.size

        for _ in range(limit):
            if file_offset <= 0:
                break

            msg, file_offset = Message.unpack(view, file_offset)
            self.session.insert(0, msg)
            
        self.append_point = len(self.session)    
            
    def fetch_context(self, max_message_tokens) -> list[Message]:
        temp = []

        token_total = 0
        for msg in reversed(self.session):
            if token_total + msg.ntokens > max_message_tokens:
                break

            temp.append(msg)
            token_total += msg.ntokens

        temp.reverse()
        return temp