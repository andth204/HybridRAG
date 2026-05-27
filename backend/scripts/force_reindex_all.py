"""Publish FILE_UPDATED Kafka events with force=true for every file in file_index_state.

Use after switching INGEST_PIPELINE to re-populate the new store.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from confluent_kafka import Producer
from src.config.settings import settings


def main() -> int:
    producer = Producer({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS})
    conn = psycopg2.connect(settings.DATABASE_URL)
    rows = []
    with conn.cursor() as cur:
        cur.execute("SELECT bucket, object_key, etag, version_id FROM file_index_state")
        rows = cur.fetchall()
    conn.close()
    print(f"Re-indexing {len(rows)} files via Kafka topic={settings.INDEXING_INPUT_TOPIC}")

    for bucket, key, etag, version_id in rows:
        event = {
            "event_type": "file_updated",
            "bucket": bucket,
            "key": key,
            "etag": etag,
            "version_id": version_id,
            "force": True,
            "requested_action": "reindexed",
        }
        producer.produce(
            settings.INDEXING_INPUT_TOPIC,
            json.dumps(event).encode("utf-8"),
        )
        print(f"  [queue] {bucket}/{key}")
    producer.flush(timeout=10)
    print("All events flushed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
