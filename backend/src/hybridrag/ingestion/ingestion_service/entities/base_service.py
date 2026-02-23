import time
import signal
import json
import threading
from abc import ABC, abstractmethod
from typing import List, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from confluent_kafka import Consumer

def build_retry(attempts: int = 3, wait_min: float = 1, wait_max: float = 10):
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(min=wait_min, max=wait_max),
        reraise=True,
    )

class BaseBatchKafkaService(ABC):
    def __init__(
        self,
        kafka_bootstrap: str,
        input_topic: str,
        consumer_group: str,
        batch_size: int = 10,
        batch_interval: float = 2.0,
    ):
        self.kafka_bootstrap = kafka_bootstrap
        self.input_topic = input_topic
        self.consumer_group = consumer_group
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self._buf: List[Tuple[dict, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._consumer = None

    @abstractmethod
    def process_batch(self, items: List[Any]) -> None:
        ...

    def _kafka_loop(self):
        self._consumer = Consumer({
            "bootstrap.servers": self.kafka_bootstrap,
            "group.id": self.consumer_group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        })
        self._consumer.subscribe([self.input_topic])
        while not self._stop.is_set():
            msg = self._consumer.poll(1.0)
            if not msg:
                continue
            if msg.error():
                continue
            payload = json.loads(msg.value().decode("utf-8"))
            with self._lock:
                self._buf.append((payload, msg))

    def _flush(self):
        with self._lock:
            if not self._buf:
                return
            batch = self._buf[: self.batch_size]
            self._buf = self._buf[self.batch_size :]
        items = [p for (p, _) in batch]
        msgs  = [m for (_, m) in batch]
        self.process_batch(items)
        if self._consumer and msgs:
            self._consumer.commit(message=msgs[-1], asynchronous=False)

    def _batch_loop(self):
        while not self._stop.is_set():
            time.sleep(self.batch_interval)
            self._flush()

    def run(self):
        def shutdown(*_):
            self._stop.set()
            self._flush()
            if self._consumer:
                self._consumer.close()
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        threading.Thread(target=self._kafka_loop, daemon=True).start()
        threading.Thread(target=self._batch_loop, daemon=True).start()
        while not self._stop.is_set():
            time.sleep(1)