import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional
from confluent_kafka import Producer

@dataclass
class FileProcessMessage:
    action: str  # added | deleted | updated
    result: str  # success | failed | duplicated | skipped
    bucket: str
    key: str
    message: str
    chunks: Optional[int] = None
    etag: Optional[str] = None
    version_id: Optional[str] = None
    file_id: Optional[str] = None
    reason: Optional[str] = None
    ts: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["ts"]:
            d["ts"] = time.time()
        return {k: v for k, v in d.items() if v is not None}

class KafkaNotifier:
    def __init__(self, bootstrap: Optional[str] = None, topic: Optional[str] = None):
        self.bootstrap = bootstrap or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        self.topic = topic or os.getenv("INDEXING_STATUS_TOPIC")
        self._p = Producer({"bootstrap.servers": self.bootstrap})

    def publish(self, msg: FileProcessMessage) -> None:
        payload = json.dumps(msg.to_dict(), ensure_ascii=False).encode("utf-8")
        key = msg.key.encode("utf-8", errors="ignore")

        def _delivery(err, _m):
            _ = err
            _ = _m
        self._p.produce(self.topic, value=payload, key=key, on_delivery=_delivery)
        self._p.poll(0)

    def flush(self, timeout: float = 3.0) -> None:
        self._p.flush(timeout)