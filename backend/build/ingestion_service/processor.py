import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Optional, List
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import unquote_plus

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from entities import (
    BaseBatchKafkaService,
    build_retry,
    Chunk,
    EventType,
    AsyncMinioClient,
    BM25Store,
    FAISSStore,
)
from helper.extractor import fetch_and_extract
from helper.state_repo import FileStateRepo, FileState
from helper.notifier import KafkaNotifier, FileProcessMessage
from src.hybridrag.ingestion.embedding import embedder
from src.hybridrag.ingestion.chunking.splitter import TextSplitter

logger = logging.getLogger(__name__)
_retry = build_retry(attempts=3, wait_min=1, wait_max=10)


@dataclass(frozen=True)
class MinioEvent:
    event_type: EventType
    bucket: str
    key: str
    etag: Optional[str] = None
    version_id: Optional[str] = None

def _uuid_for(bucket: str, key: str, etag_or_version: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{bucket}/{key}:{etag_or_version}"))

def _decode_minio_key(key: Optional[str]) -> Optional[str]:
    if not isinstance(key, str) or not key:
        return key
    return unquote_plus(key)

def _parse_minio_event(raw: dict) -> Optional[MinioEvent]:
    try:
        recs = raw.get("Records") or []
        if not recs:
            return None

        r = recs[0]
        event_name = (r.get("eventName") or "").lower()

        s3 = r.get("s3") or {}
        bucket = (s3.get("bucket") or {}).get("name")

        obj = s3.get("object") or {}
        key = _decode_minio_key(obj.get("key"))
        etag = obj.get("eTag") or obj.get("etag")
        version_id = obj.get("versionId") or obj.get("versionid")

        if not bucket or not key:
            return None

        if "removed" in event_name or "delete" in event_name:
            et = EventType.FILE_DELETED
        elif "created" in event_name or "put" in event_name:
            et = EventType.FILE_ADDED
        else:
            et = EventType.FILE_UPDATED

        if isinstance(etag, str):
            etag = etag.strip('"')

        return MinioEvent(event_type=et, bucket=bucket, key=key, etag=etag, version_id=version_id)

    except Exception as e:
        logger.error(f"[IngestionService] parse minio event failed: {e} | raw={raw}")
        return None


def _action_from_event_type(t: EventType) -> str:
    if t == EventType.FILE_DELETED:
        return "deleted"
    if t == EventType.FILE_UPDATED:
        return "updated"
    return "added"


class IngestionService(BaseBatchKafkaService):
    def __init__(
        self,
        kafka_bootstrap: str,
        input_topic: str,
        consumer_group: str,
        batch_size: int,
        batch_interval: float,
        minio: AsyncMinioClient,
        bm25_cache: str,
        faiss_cache: str,
        pg_dsn: str,
        state_schema: str = "public",
        state_table: str = "file_index_state",
        status_topic: Optional[str] = None,
    ):
        super().__init__(kafka_bootstrap, input_topic, consumer_group, batch_size, batch_interval)
        self.minio = minio
        self.splitter = TextSplitter()
        self.bm25 = BM25Store(cache_dir=bm25_cache)
        self.faiss = FAISSStore(
            embedding_fn=embedder.embed,
            embedding_dim=embedder.get_dimension(),
            cache_dir=faiss_cache,
        )
        self.state = FileStateRepo(pg_dsn, schema=state_schema, table=state_table)
        self.notifier = KafkaNotifier(bootstrap=kafka_bootstrap, topic=status_topic)

    def process_batch(self, items: List[Any]) -> None:
        asyncio.run(self._process_batch(items))

    async def _process_batch(self, items: List[Any]) -> None:
        events: List[MinioEvent] = []
        for it in tqdm(items, desc="Parsing events", unit="msg"):
            try:
                raw = it if isinstance(it, dict) else json.loads(it)
                ev = _parse_minio_event(raw) if "Records" in raw else None
                if ev:
                    events.append(ev)
            except Exception as e:
                logger.warning(f"[IngestionService] Bad message: {e} | raw={it}")
        if not events:
            return

        results = []
        with tqdm(total=len(events), desc="Indexing files", unit="file") as pbar:
            for coro in asyncio.as_completed([self._handle_with_notify(ev) for ev in events]):
                r = await coro
                results.append(r)
                pbar.set_postfix({
                    "result": r.get("result", "?"),
                    "key": r.get("key", ""),
                })
                pbar.update(1)

        stats = {"success": 0, "deleted": 0, "duplicated": 0, "skipped": 0, "failed": 0}
        for r in results:
            stats[r.get("result", "failed")] = stats.get(r.get("result", "failed"), 0) + 1
        logger.info(f"[IngestionService] batch — {stats}")

        self.notifier.flush()

    async def _handle_with_notify(self, ev: MinioEvent) -> dict:
        action = _action_from_event_type(ev.event_type)
        try:
            res = await self._handle(ev)
            self.notifier.publish(FileProcessMessage(
                action=action,
                result=res["result"],
                bucket=ev.bucket,
                key=ev.key,
                message=res["message"],
                chunks=res.get("chunks"),
                etag=res.get("etag"),
                version_id=res.get("version_id"),
                file_id=res.get("file_id"),
                reason=res.get("reason"),
            ))
            return {**res, "key": ev.key}
        except Exception as e:
            logger.exception(f"[IngestionService] failed key={ev.key}: {e}")
            msg = FileProcessMessage(
                action=action,
                result="failed",
                bucket=ev.bucket,
                key=ev.key,
                message=f"{action.upper()} failed",
                reason=str(e),
                etag=ev.etag,
                version_id=ev.version_id,
            )
            self.notifier.publish(msg)
            return {"result": "failed", "message": msg.message, "reason": str(e)}

    async def _handle(self, ev: MinioEvent) -> dict:
        # DELETE
        if ev.event_type == EventType.FILE_DELETED:
            self._purge_key(ev.key)
            self.state.delete(ev.bucket, ev.key)
            return {"result": "deleted", "message": "Deleted file successfully"}

        # ADD/UPDATE
        etag, version_id = await self._resolve_version(ev)

        current = self.state.get(ev.bucket, ev.key)
        if current and (current.etag == etag) and (current.version_id == version_id):
            return {
                "result": "duplicated",
                "message": "Duplicate file (same content/version), skipped indexing",
                "reason": "same_version",
                "etag": etag,
                "version_id": version_id,
                "file_id": current.file_id,
            }
        if current:
            self._purge_key(ev.key)
        return await self._index(ev.bucket, ev.key, etag=etag, version_id=version_id)

    def _purge_key(self, key: str) -> None:
        b = self.bm25.delete_by_key(key)
        f = self.faiss.delete_by_key(key)

    @_retry
    async def _resolve_version(self, ev: MinioEvent) -> tuple[Optional[str], Optional[str]]:
        if ev.version_id or ev.etag:
            etag = ev.etag.strip('"') if isinstance(ev.etag, str) else ev.etag
            return etag, ev.version_id

        info = await self.minio.info(ev.bucket, ev.key)
        etag = info.get("etag")
        if isinstance(etag, str):
            etag = etag.strip('"')
        return etag, None

    @_retry
    async def _fetch_text(self, bucket: str, key: str) -> Optional[str]:
        return await fetch_and_extract(self.minio, bucket, key)

    @_retry
    async def _index_stores(self, chunks: list[Chunk]) -> None:
        self.bm25.add_chunks(chunks)
        await self.faiss.add_chunks(chunks)

    async def _index(self, bucket: str, key: str, etag: Optional[str], version_id: Optional[str]) -> dict:
        full_text = await self._fetch_text(bucket, key)
        if not full_text:
            return {
                "result": "skipped",
                "message": "Skipped (empty/unsupported file)",
                "reason": "empty_or_unsupported",
                "etag": etag,
                "version_id": version_id,
            }

        version_tag = (version_id or etag or "noversion")
        file_id = _uuid_for(bucket, key, version_tag)
        parts = self.splitter.split_text(full_text)
        chunks = [Chunk(file_id=file_id, key=key, text=t, index=i) for i, t in enumerate(parts)]
        logger.info(f"[IngestionService] '{key}' -> {len(chunks)} chunks (version={version_tag})")

        await self._index_stores(chunks)
        self.state.upsert(FileState(bucket=bucket, key=key, etag=etag, version_id=version_id, file_id=file_id))
        return {
            "result": "success",
            "message": "Indexed file successfully",
            "chunks": len(chunks),
            "etag": etag,
            "version_id": version_id,
            "file_id": file_id,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

    pg_dsn = (
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )

    IngestionService(
        kafka_bootstrap=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
        input_topic=os.getenv("INDEXING_INPUT_TOPIC"),
        consumer_group=os.getenv("INDEXING_CONSUMER_GROUP"),
        batch_size=int(os.getenv("INDEXING_BATCH_SIZE")),
        batch_interval=float(os.getenv("INDEXING_BATCH_INTERVAL")),
        minio=AsyncMinioClient(),
        bm25_cache=os.getenv("BM25_CACHE_DIR"),
        faiss_cache=os.getenv("FAISS_CACHE_DIR"),
        pg_dsn=pg_dsn,
        state_schema=os.getenv("INDEX_STATE_SCHEMA"),
        state_table=os.getenv("INDEX_STATE_TABLE"),
        status_topic=os.getenv("INDEXING_STATUS_TOPIC"),
    ).run()