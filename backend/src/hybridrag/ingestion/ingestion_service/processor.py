import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import unquote_plus
from tqdm import tqdm
from entities import (
    AsyncMinioClient,
    BaseBatchKafkaService,
    BM25Store,
    Chunk,
    EventType,
    FAISSStore,
    build_retry,
)
from helper.extractor import fetch_and_extract
from helper.notifier import FileProcessMessage, KafkaNotifier
from helper.state_repo import FileState, FileStateRepo
from src.config.settings import settings
from src.hybridrag.ingestion.chunking.splitter import TextSplitter
from src.hybridrag.ingestion.embedding import embedder

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
        records = raw.get("Records") or []
        if not records:
            return None

        record = records[0]
        event_name = (record.get("eventName") or "").lower()
        s3 = record.get("s3") or {}
        bucket = (s3.get("bucket") or {}).get("name")
        obj = s3.get("object") or {}
        key = _decode_minio_key(obj.get("key"))
        etag = obj.get("eTag") or obj.get("etag")
        version_id = obj.get("versionId") or obj.get("versionid")
        if not bucket or not key:
            return None

        if "removed" in event_name or "delete" in event_name:
            event_type = EventType.FILE_DELETED
        elif "created" in event_name or "put" in event_name:
            event_type = EventType.FILE_ADDED
        else:
            event_type = EventType.FILE_UPDATED

        if isinstance(etag, str):
            etag = etag.strip('"')
        return MinioEvent(event_type=event_type, bucket=bucket, key=key, etag=etag, version_id=version_id)
    except Exception:
        return None


def _action_label(event_type: EventType) -> str:
    return {
        EventType.FILE_DELETED: "deleted",
        EventType.FILE_UPDATED: "updated",
    }.get(event_type, "added")

def _build_chunk(file_id: str, key: str, text: str, i: int) -> Chunk:
    return Chunk(file_id=file_id, key=key, text=text, _chunk_id=f"{file_id}__chunk_{i:06d}")


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
            index_type=getattr(settings, "FAISS_INDEX_TYPE", "hnsw"),
            hnsw_m=int(getattr(settings, "FAISS_HNSW_M", 32)),
            hnsw_ef_construction=int(getattr(settings, "FAISS_HNSW_EF_CONSTRUCTION", 200)),
            hnsw_ef_search=int(getattr(settings, "FAISS_HNSW_EF_SEARCH", 64)),
            ivf_nlist=int(getattr(settings, "FAISS_IVF_NLIST", 4096)),
            ivf_nprobe=int(getattr(settings, "FAISS_IVF_NPROBE", 16)),
        )
        self.state = FileStateRepo(pg_dsn, schema=state_schema, table=state_table)
        self.notifier = KafkaNotifier(bootstrap=kafka_bootstrap, topic=status_topic)
        self._concurrency = max(1, int(getattr(settings, "INDEXING_CONCURRENCY", 8)))
        self._index_lock: asyncio.Lock | None = None
        self._sem: asyncio.Semaphore | None = None

    def _ensure_async_primitives(self) -> None:
        self._index_lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(self._concurrency)

    def process_batch(self, items: list[Any]) -> None:
        asyncio.run(self._process_batch(items))

    async def _process_batch(self, items: list[Any]) -> None:
        self._ensure_async_primitives()
        events: list[MinioEvent] = []
        for item in tqdm(items, desc="Parsing events", unit="msg"):
            raw = item if isinstance(item, dict) else json.loads(item)
            if "Records" in raw and (event := _parse_minio_event(raw)):
                events.append(event)

        if not events:
            return

        async def _guarded(ev: MinioEvent) -> dict:
            async with self._sem:
                return await self._handle_with_notify(ev)

        results: list[dict] = []
        with tqdm(total=len(events), desc="Indexing files", unit="file") as pbar:
            for coro in asyncio.as_completed([_guarded(ev) for ev in events]):
                result = await coro
                results.append(result)
                pbar.set_postfix(result=result.get("result", "?"), key=result.get("key", ""))
                pbar.update(1)

        stats: dict[str, int] = {}
        for r in results:
            k = r.get("result", "failed")
            stats[k] = stats.get(k, 0) + 1

        logger.info("Batch complete -- %s", stats)
        self.notifier.flush()

    async def _handle_with_notify(self, ev: MinioEvent) -> dict:
        action = _action_label(ev.event_type)
        try:
            result = await self._handle(ev)
            self.notifier.publish(
                FileProcessMessage(
                    action=action,
                    result=result["result"],
                    bucket=ev.bucket,
                    key=ev.key,
                    message=result["message"],
                    chunks=result.get("chunks"),
                    etag=result.get("etag"),
                    version_id=result.get("version_id"),
                    file_id=result.get("file_id"),
                    reason=result.get("reason"),
                )
            )
            return {**result, "key": ev.key}
        except Exception as e:
            logger.exception("[FAILED] key=%s action=%s error=%s", ev.key, action, e)
            self.notifier.publish(
                FileProcessMessage(
                    action=action,
                    result="failed",
                    bucket=ev.bucket,
                    key=ev.key,
                    message=f"{action.upper()} failed",
                    reason=str(e),
                    etag=ev.etag,
                    version_id=ev.version_id,
                )
            )
            return {"result": "failed", "message": f"{action.upper()} failed", "reason": str(e), "key": ev.key}

    async def _handle(self, ev: MinioEvent) -> dict:
        if ev.event_type == EventType.FILE_DELETED:
            await self._purge(ev.key)
            self.state.delete(ev.bucket, ev.key)
            return {"result": "deleted", "message": "Deleted file successfully"}

        etag, version_id = await self._resolve_version(ev)
        current = self.state.get(ev.bucket, ev.key)
        if current and current.etag == etag and current.version_id == version_id:
            return {
                "result": "duplicated",
                "message": "Duplicate file (same content/version), skipped indexing",
                "reason": "same_version",
                "etag": etag,
                "version_id": version_id,
                "file_id": current.file_id,
            }
        if current:
            await self._purge(ev.key)
        return await self._index(ev.bucket, ev.key, etag=etag, version_id=version_id)

    async def _purge(self, key: str) -> None:
        async with self._index_lock:
            await asyncio.to_thread(self.bm25.delete_by_key, key)
            await asyncio.to_thread(self.faiss.delete_by_key, key)

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
        await self.faiss.precompute_embeddings(chunks)
        async with self._index_lock:
            await asyncio.to_thread(self.bm25.add_chunks, chunks)
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
        version_tag = version_id or etag or "noversion"
        file_id = _uuid_for(bucket, key, version_tag)
        parts = self.splitter.split_text(full_text)
        chunks = [_build_chunk(file_id=file_id, key=key, text=text, i=i) for i, text in enumerate(parts)]

        logger.info("'%s' -> %d chunks (version=%s)", key, len(chunks), version_tag)

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
    print("-> Service has started.")
    
    IngestionService(
        kafka_bootstrap=settings.KAFKA_BOOTSTRAP_SERVERS,
        input_topic=settings.INDEXING_INPUT_TOPIC,
        consumer_group=settings.INDEXING_CONSUMER_GROUP,
        batch_size=settings.INDEXING_BATCH_SIZE,
        batch_interval=settings.INDEXING_BATCH_INTERVAL,
        minio=AsyncMinioClient(),
        bm25_cache=str(settings.BM25_CACHE_DIR),
        faiss_cache=str(settings.FAISS_CACHE_DIR),
        pg_dsn=settings.DATABASE_URL,
        state_schema=settings.INDEX_STATE_SCHEMA,
        state_table=settings.INDEX_STATE_TABLE,
        status_topic=settings.INDEXING_STATUS_TOPIC,
    ).run()