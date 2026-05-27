"""Weaviate ingestion service (Phase 2 alternative pipeline).

Mirrors :mod:`processor` but uses the hierarchical splitter, the
metadata extractor, and a single Weaviate-backed store (no FAISS/BM25
pair). The two paths can run side-by-side; ``INGEST_PIPELINE`` selects
which one is wired into a deployment.

We deliberately import the small Kafka/event helpers from
``processor`` instead of duplicating them so any future change to the
event format only happens in one place. The two services share:
``MinioEvent``, ``_parse_minio_event``, ``_parse_manual_event``,
``_action_label``, ``_decode_minio_key``, ``_uuid_for``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any, Optional

from tqdm import tqdm

from entities import (
    AsyncMinioClient,
    BaseBatchKafkaService,
    EventType,
    build_retry,
)
from helper.extractor import fetch_and_extract
from helper.notifier import FileProcessMessage, KafkaNotifier
from helper.state_repo import FileState, FileStateRepo

from src.config.settings import settings
from src.hybridrag.ingestion.chunking.hierarchical import (
    HierarchicalChunk,
    HierarchicalSplitter,
)
from src.hybridrag.ingestion.embedding import embedder
from src.hybridrag.ingestion.ingestion_service.entities.weaviate_store import (
    WeaviateStore,
)
from src.hybridrag.ingestion.metadata.extractor import extract_metadata

# Reuse the event-parsing surface from the legacy processor to keep the
# two pipelines in lockstep. The import is deliberate -- duplicating
# `_parse_minio_event` here would create two sources of truth.
from processor import (  # type: ignore[import-not-found]
    MinioEvent,
    _action_label,
    _parse_manual_event,
    _parse_minio_event,
    _uuid_for,
)

logger = logging.getLogger(__name__)
_retry = build_retry(attempts=3, wait_min=1, wait_max=10)


def _chunk_to_weaviate_dict(
    chunk: HierarchicalChunk,
    *,
    file_id: str,
    key: str,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a :class:`HierarchicalChunk` into the dict shape that
    :meth:`WeaviateStore.add_chunks` expects.

    ``extra_meta`` (the output of :func:`extract_metadata`) is merged
    last so per-chunk inferred values (campus / year / faculty / major /
    doc_type) win over anything baked into ``chunk.extra``.
    """
    base: dict[str, Any] = {
        "chunk_id":    chunk.chunk_id,
        "parent_id":   chunk.parent_id,
        "file_id":     file_id,
        "key":         key,
        "content":     chunk.content,
        "section":     chunk.section,
        "header_path": list(chunk.header_path or []),
        "chunk_level": chunk.chunk_level,
        "is_table":    bool(chunk.is_table),
    }
    # Pull doc-level metadata from chunk.extra (e.g. campus inferred
    # at split-time), then overlay chunk-level metadata.
    if chunk.extra:
        for k, v in chunk.extra.items():
            if k in {"campus", "doc_type", "year", "faculty", "major"} and v is not None:
                base[k] = v
    if extra_meta:
        for k, v in extra_meta.items():
            if v is not None:
                base[k] = v
    return base


class WeaviateIngestionService(BaseBatchKafkaService):
    """Kafka-driven ingestion service that writes into Weaviate.

    Public surface mirrors :class:`IngestionService` so the two can be
    swapped behind the ``INGEST_PIPELINE`` flag without touching the
    deployment harness.
    """

    def __init__(
        self,
        kafka_bootstrap: str,
        input_topic: str,
        consumer_group: str,
        batch_size: int,
        batch_interval: float,
        minio: AsyncMinioClient,
        pg_dsn: str,
        state_schema: str = "public",
        state_table: str = "file_index_state",
        status_topic: Optional[str] = None,
    ):
        super().__init__(kafka_bootstrap, input_topic, consumer_group, batch_size, batch_interval)
        self.minio = minio
        self.splitter = HierarchicalSplitter()
        self.store = WeaviateStore(
            embedding_fn=embedder.embed,
            embedding_dim=embedder.get_dimension(),
        )
        self.state = FileStateRepo(pg_dsn, schema=state_schema, table=state_table)
        self.notifier = KafkaNotifier(bootstrap=kafka_bootstrap, topic=status_topic)
        self._concurrency = max(1, int(getattr(settings, "INDEXING_CONCURRENCY", 8)))
        self._index_lock: asyncio.Lock | None = None
        self._sem: asyncio.Semaphore | None = None
        # Persistent event loop so AsyncOpenAI + Weaviate v4 internal httpx /
        # gRPC clients survive across batches. `asyncio.run` would close the
        # loop after each call and orphan those connections, causing the next
        # batch to hang on stale transports.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_lock = threading.Lock()

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                # Reset async primitives created on a different loop.
                self._index_lock = None
                self._sem = None
            # Bind the loop to the current thread so libraries that read
            # `asyncio.get_event_loop()` (AsyncOpenAI, httpx) pick it up.
            try:
                asyncio.set_event_loop(self._loop)
            except Exception:
                pass
            return self._loop

    # ------------------------------------------------- async primitives
    def _ensure_async_primitives(self) -> None:
        if self._index_lock is None:
            self._index_lock = asyncio.Lock()
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._concurrency)

    # ------------------------------------------------------- batch entry
    def process_batch(self, items: list[Any]) -> None:
        loop = self._get_loop()
        try:
            loop.run_until_complete(self._process_batch(items))
        except Exception:
            logger.exception("process_batch: unhandled error; resetting loop + store")
            self._reset_for_next_batch()
            raise

    def _reset_for_next_batch(self) -> None:
        """Best-effort tear-down so a broken state does not poison subsequent
        batches. Closes the Weaviate client + event loop; both are lazily
        rebuilt on next ``process_batch``.
        """
        with self._loop_lock:
            if self._loop is not None and not self._loop.is_closed():
                try:
                    self._loop.run_until_complete(asyncio.sleep(0))
                except Exception:
                    pass
                try:
                    self._loop.close()
                except Exception:
                    pass
            self._loop = None
        self._index_lock = None
        self._sem = None
        try:
            self.store.close()
        except Exception:
            pass

    def shutdown(self) -> None:
        """Tear down resources. Called by the Kafka consumer loop on stop."""
        self._reset_for_next_batch()
        try:
            self.notifier.flush()
        except Exception:
            pass

    async def _process_batch(self, items: list[Any]) -> None:
        self._ensure_async_primitives()
        events: list[MinioEvent] = []
        for item in tqdm(items, desc="Parsing events", unit="msg"):
            raw = item if isinstance(item, dict) else json.loads(item)
            if isinstance(raw, dict) and (event := _parse_manual_event(raw)):
                events.append(event)
                continue
            if "Records" in raw and (event := _parse_minio_event(raw)):
                events.append(event)

        if not events:
            return

        logger.info("_process_batch: dispatching %d events (concurrency=%d)", len(events), self._concurrency)

        async def _guarded(ev: MinioEvent) -> dict:
            async with self._sem:
                logger.info("[start] key=%s event=%s", ev.key, ev.event_type)
                try:
                    res = await self._handle_with_notify(ev)
                    logger.info("[done] key=%s result=%s", ev.key, res.get("result"))
                    return res
                except Exception as exc:
                    logger.exception("[crash] key=%s err=%s", ev.key, exc)
                    raise

        results: list[dict] = []
        for coro in asyncio.as_completed([_guarded(ev) for ev in events]):
            try:
                result = await coro
            except Exception as exc:
                logger.exception("[batch] event failed: %s", exc)
                continue
            results.append(result)

        stats: dict[str, int] = {}
        for r in results:
            k = r.get("result", "failed")
            stats[k] = stats.get(k, 0) + 1

        logger.info("Batch complete -- %s", stats)
        self.notifier.flush()

    # ---------------------------------------------------- per-file path
    async def _handle_with_notify(self, ev: MinioEvent) -> dict:
        default_action = ev.requested_action or _action_label(ev.event_type)
        try:
            result = await self._handle(ev)
            action = str(result.get("action") or default_action)
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
            logger.exception("[FAILED] key=%s action=%s error=%s", ev.key, default_action, e)
            self.notifier.publish(
                FileProcessMessage(
                    action=default_action,
                    result="failed",
                    bucket=ev.bucket,
                    key=ev.key,
                    message=f"{default_action.upper()} failed",
                    reason=str(e),
                    etag=ev.etag,
                    version_id=ev.version_id,
                )
            )
            return {"result": "failed", "message": f"{default_action.upper()} failed", "reason": str(e), "key": ev.key}

    async def _handle(self, ev: MinioEvent) -> dict:
        if ev.event_type == EventType.FILE_DELETED:
            await self._purge(ev.key)
            self.state.delete(ev.bucket, ev.key)
            return {
                "action": "deleted",
                "result": "deleted",
                "message": f'Deleted "{ev.key}" successfully.',
            }

        etag, version_id = await self._resolve_version(ev)
        current = self.state.get(ev.bucket, ev.key)
        if current and current.etag == etag and current.version_id == version_id and not ev.force:
            return {
                "action": "updated",
                "result": "duplicated",
                "message": f'Duplicate file "{ev.key}" detected.',
                "reason": "same_version",
                "etag": etag,
                "version_id": version_id,
                "file_id": current.file_id,
            }
        if current:
            await self._purge(ev.key)
            result = await self._index(ev.bucket, ev.key, etag=etag, version_id=version_id)
            if result.get("result") == "success":
                if ev.requested_action == "reindexed":
                    result["message"] = f'Re-indexed "{ev.key}" successfully.'
                else:
                    result["message"] = f'Updated "{ev.key}" successfully.'
            return {**result, "action": ev.requested_action or "updated"}

        result = await self._index(ev.bucket, ev.key, etag=etag, version_id=version_id)
        if result.get("result") == "success" and ev.requested_action == "reindexed":
            result["message"] = f'Re-indexed "{ev.key}" successfully.'
        if ev.requested_action:
            return {**result, "action": ev.requested_action}
        return result

    # ---------------------------------------------------- helper actions
    async def _purge(self, key: str) -> None:
        async with self._index_lock:
            await asyncio.to_thread(self.store.delete_by_key, key)

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

    async def _index(self, bucket: str, key: str, etag: Optional[str], version_id: Optional[str]) -> dict:
        full_text = await self._fetch_text(bucket, key)
        # DEBUG: dump first 30 codepoints of key + full_text to verify in-worker decode
        if key and (key.startswith("Đ") or key.startswith("Th")):
            logger.warning(
                "WORKER-EXTRACT key_cp=%s text_first30_cp=%s",
                [hex(ord(c)) for c in key[:8]],
                [hex(ord(c)) for c in (full_text or "")[:30]],
            )
        if not full_text:
            return {
                "result": "skipped",
                "message": f'Skipped "{key}" (empty or unsupported file).',
                "reason": "empty_or_unsupported",
                "etag": etag,
                "version_id": version_id,
            }

        version_tag = version_id or etag or "noversion"
        file_id = _uuid_for(bucket, key, version_tag)

        parents, children = self.splitter.split(
            full_text,
            file_id=file_id,
            key=key,
            doc_meta={"file_id": file_id, "key": key},
        )

        filename = os.path.basename(key)
        enriched: list[dict[str, Any]] = []
        for chunk in (*parents, *children):
            extra_meta = extract_metadata(
                text=chunk.content,
                header_path=chunk.header_path,
                filename=filename,
            )
            enriched.append(
                _chunk_to_weaviate_dict(chunk, file_id=file_id, key=key, extra_meta=extra_meta)
            )

        if not enriched:
            return {
                "result": "skipped",
                "message": f'Skipped "{key}" (no chunks produced).',
                "reason": "empty_chunks",
                "etag": etag,
                "version_id": version_id,
            }

        logger.info(
            "'%s' -> %d parents + %d children (version=%s)",
            key, len(parents), len(children), version_tag,
        )

        await self._index_stores(enriched)
        self.state.upsert(
            FileState(bucket=bucket, key=key, etag=etag, version_id=version_id, file_id=file_id)
        )
        return {
            "result": "success",
            "message": f'Indexed "{key}" successfully.',
            "chunks": len(enriched),
            "etag": etag,
            "version_id": version_id,
            "file_id": file_id,
        }

    @_retry
    async def _index_stores(self, enriched: list[dict[str, Any]]) -> None:
        # `precompute_embeddings` warms the embedding cache outside the
        # critical section so the write under the lock is purely Weaviate I/O.
        await self.store.precompute_embeddings(enriched)
        async with self._index_lock:
            await self.store.add_chunks(enriched)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")
    print("-> Weaviate ingestion service has started.")

    WeaviateIngestionService(
        kafka_bootstrap=settings.KAFKA_BOOTSTRAP_SERVERS,
        input_topic=settings.INDEXING_INPUT_TOPIC,
        consumer_group=settings.INDEXING_CONSUMER_GROUP,
        batch_size=settings.INDEXING_BATCH_SIZE,
        batch_interval=settings.INDEXING_BATCH_INTERVAL,
        minio=AsyncMinioClient(),
        pg_dsn=settings.DATABASE_URL,
        state_schema=settings.INDEX_STATE_SCHEMA,
        state_table=settings.INDEX_STATE_TABLE,
        status_topic=settings.INDEXING_STATUS_TOPIC,
    ).run()
