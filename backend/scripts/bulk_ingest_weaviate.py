"""Direct bulk ingest into Weaviate.

Reads file_index_state, fetches each file from MinIO, runs hierarchical
chunking + metadata extraction, writes to Weaviate.

Run inside the ingestion-service container:
    docker exec utehy-ingestion-service python scripts/bulk_ingest_weaviate.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid as uuidlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# Add ingestion_service dir for `entities` / `helper` imports
INGEST_DIR = Path(__file__).parent.parent / "src" / "hybridrag" / "ingestion" / "ingestion_service"
sys.path.insert(0, str(INGEST_DIR))

import psycopg2
from src.config.settings import settings
from src.hybridrag.ingestion.chunking.hierarchical import HierarchicalSplitter
from src.hybridrag.ingestion.embedding import embedder
from src.hybridrag.ingestion.ingestion_service.entities.minio_common import AsyncMinioClient
from src.hybridrag.ingestion.ingestion_service.entities.weaviate_store import WeaviateStore
from src.hybridrag.ingestion.metadata.extractor import extract_metadata
from helper.extractor import fetch_and_extract  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("bulk_ingest")


def file_id_for(bucket: str, key: str, etag: str | None) -> str:
    return str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, f"{bucket}/{key}:{etag or 'noversion'}"))


async def ingest_one(
    store: WeaviateStore,
    splitter: HierarchicalSplitter,
    minio: AsyncMinioClient,
    bucket: str,
    key: str,
    etag: str | None,
) -> int:
    text = await fetch_and_extract(minio, bucket, key)
    if not text:
        log.warning("empty: %s/%s", bucket, key)
        return 0

    fid = file_id_for(bucket, key, etag)
    parents, children = splitter.split(text, file_id=fid, key=key, doc_meta={"file_id": fid, "key": key})

    enriched = []
    for chunk in parents + children:
        meta = extract_metadata(
            text=chunk.content,
            header_path=chunk.header_path,
            filename=Path(key).name,
        )
        enriched.append(
            {
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id or "",
                "file_id": fid,
                "key": key,
                "content": chunk.content,
                "section": chunk.section,
                "header_path": chunk.header_path,
                "chunk_level": chunk.chunk_level,
                "is_table": chunk.is_table,
                **meta,
            }
        )

    # Replace any existing chunks for this key
    removed = store.delete_by_key(key)
    if removed:
        log.info("  deleted %d existing chunks for %s", removed, key)

    await store.precompute_embeddings(enriched)
    inserted = await store.add_chunks(enriched)
    log.info("  inserted %d chunks (%d parents + %d children) for %s", inserted, len(parents), len(children), key)
    return inserted


async def main() -> int:
    conn = psycopg2.connect(settings.DATABASE_URL)
    rows = []
    with conn.cursor() as cur:
        cur.execute("SELECT bucket, object_key, etag FROM file_index_state ORDER BY object_key")
        rows = cur.fetchall()
    conn.close()
    log.info("Bulk ingest into Weaviate: %d files", len(rows))

    store = WeaviateStore(embedding_fn=embedder.embed, embedding_dim=embedder.get_dimension())
    splitter = HierarchicalSplitter()
    minio = AsyncMinioClient()

    total = 0
    for bucket, key, etag in rows:
        log.info("==> %s/%s", bucket, key)
        try:
            n = await ingest_one(store, splitter, minio, bucket, key, etag)
            total += n
        except Exception:
            log.exception("FAILED %s/%s", bucket, key)

    log.info("Done. Total chunks inserted: %d", total)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
