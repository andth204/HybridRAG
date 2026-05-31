"""Re-ingest the local sample corpus into the Weaviate collection.

Bypasses the MinIO+Kafka indexer pipeline and pushes files directly via
:class:`HierarchicalSplitter` → :class:`WeaviateStore`. Required after:

* converting score files from TSV to markdown pipe (v3.5 task 1), and
* flipping ``route_tables_to_row_chunks`` to ``True`` by default.

Idempotent: ``delete_by_key`` runs first for each file, so existing
chunks attributed to the same filename are replaced.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from src.config.settings import settings  # noqa: E402
from src.hybridrag.ingestion.chunking.hierarchical import (  # noqa: E402
    HierarchicalSplitter,
)
from src.hybridrag.ingestion.embedding import embedder  # noqa: E402
from src.hybridrag.ingestion.ingestion_service.entities.weaviate_store import (  # noqa: E402
    WeaviateStore,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
log = logging.getLogger("reingest")


def _chunk_to_dict(chunk, *, file_id: str, key: str) -> dict:
    """Project a HierarchicalChunk into the dict shape WeaviateStore expects."""
    return {
        "chunk_id":    chunk.chunk_id,
        "parent_id":   chunk.parent_id,
        "file_id":     file_id,
        "key":         key,
        "content":     chunk.content,
        "section":     chunk.section,
        "header_path": chunk.header_path,
        "chunk_level": chunk.chunk_level,
        "is_table":    chunk.is_table,
    }


async def reingest() -> int:
    splitter = HierarchicalSplitter(route_tables_to_row_chunks=True)
    store = WeaviateStore(
        embedding_fn=embedder.embed,
        embedding_dim=embedder.get_dimension(),
    )

    data_dir = Path(settings.DATA_DIR) / "samples"
    files = sorted(
        list(data_dir.glob("*.md")) + list(data_dir.glob("*.txt"))
    )
    log.info("found %d source files in %s", len(files), data_dir)

    total_inserted = 0
    for fpath in files:
        key = fpath.name
        # Stable per-file id (same scheme as the MinIO event ingest path
        # uses, only with a synthetic bucket prefix since this is local).
        file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"local/samples/{key}"))

        text = fpath.read_text(encoding="utf-8")

        # Idempotent: drop any existing rows for this key first.
        deleted = store.delete_by_key(key)
        if deleted:
            log.info("  [%s] deleted %d existing chunks", key, deleted)

        parents, children = splitter.split(text, file_id=file_id, key=key)
        all_chunks = [
            _chunk_to_dict(c, file_id=file_id, key=key)
            for c in (*parents, *children)
        ]
        if not all_chunks:
            log.warning("  [%s] no chunks produced; skipping", key)
            continue

        inserted = await store.add_chunks(all_chunks)
        log.info(
            "  [%s] parents=%d children=%d inserted=%d",
            key, len(parents), len(children), inserted,
        )
        total_inserted += inserted

    log.info("=" * 60)
    log.info("Total chunks inserted: %d", total_inserted)
    return total_inserted


if __name__ == "__main__":
    inserted = asyncio.run(reingest())
    sys.exit(0 if inserted > 0 else 1)
