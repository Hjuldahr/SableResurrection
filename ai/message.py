from datetime import datetime, timezone
import struct
from typing import Any
import uuid

from ai.role import Role

class Message:
    __slots__ = ('uid', 'dt', 'role', 'content', 'ntokens', 'chat_meta', 'ui_meta')

    NAMESPACE = uuid.UUID('90f7925d-9fd2-4661-ba1d-ff6a58555fe2')

    HEADER = struct.Struct('<H7B')
    TRAILER = struct.Struct('<IH')

    def __init__(
        self,
        dt: datetime | None = None,
        role: Role = Role.SYSTEM,
        content: str = "",
        ntokens: int = 0,
        *,
        chat_meta: dict[str, Any] | None = None,
        ui_meta: dict[str, Any] | None = None
    ):
        self.dt = dt if dt is not None else datetime.now(timezone.utc)
        self.uid = uuid.uuid5(self.NAMESPACE, self.dt.strftime("%Y%m%d%H%M%S%f"))
        self.role = role
        self.ntokens = ntokens
        self.content = content
        self.chat_meta = chat_meta
        self.ui_meta = ui_meta

    @property
    def content_bytes(self) -> bytes:
        return self.content.encode('utf-8', 'replace')

    def pack(self) -> bytes:
        dt = self.dt
        cb = self.content_bytes
        content_nbytes = len(cb)

        buffer = bytearray(self.HEADER.size + content_nbytes + self.TRAILER.size)
        offset = 0

        self.HEADER.pack_into(
            buffer, offset, 
            dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond // 1000, int(self.role),
        )
        offset += self.HEADER.size

        buffer[offset:offset + content_nbytes] = cb
        offset += content_nbytes

        self.TRAILER.pack_into(buffer, offset, content_nbytes, self.ntokens)

        return bytes(buffer)

    @property
    def byte_count(self) -> int:
        return self.HEADER.size + len(self.content_bytes) + self.TRAILER.size

    @classmethod
    def unpack(cls, view: memoryview, rev_offset: int):
        rev_offset -= cls.TRAILER.size
        content_nbytes, ntokens = cls.TRAILER.unpack_from(view, rev_offset)

        rev_offset -= content_nbytes
        content = str(view[rev_offset:rev_offset + content_nbytes], 'utf-8', 'replace')

        rev_offset -= cls.HEADER.size
        year, month, day, hour, minute, second, millisecond, role_ordinal = cls.HEADER.unpack_from(view, rev_offset)

        dt = datetime(year, month, day, hour, minute, second, millisecond * 1000, timezone.utc)
        role = Role.from_ordinal(role_ordinal)

        return cls(dt, role, content, ntokens), rev_offset

    def to_msg_obj(self) -> dict[str, Any]:
        return {
            **(self.chat_meta if self.chat_meta is not None else {}),
            'role': self.role.value,
            'content': self.content,
        }