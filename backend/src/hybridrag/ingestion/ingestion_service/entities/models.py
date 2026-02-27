from dataclasses import dataclass
from typing import Optional
from enum import Enum
import hashlib

class EventType(str, Enum):
    FILE_ADDED   = "file_added"
    FILE_UPDATED = "file_updated"
    FILE_DELETED = "file_deleted"

@dataclass
class FileEvent:
    event_type: EventType
    bucket: str
    key: str
    etag: Optional[str] = None
    version_id: Optional[str] = None

@dataclass
class Chunk:
    file_id: str
    key: str
    text: str
    _chunk_id: Optional[str] = None

    @property
    def chunk_id(self) -> str:
        if self._chunk_id:
            return self._chunk_id
        return f"{self.file_id}__chunk_{hashlib.md5(self.text.encode()).hexdigest()[:8]}"