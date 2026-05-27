"""One-shot direct ingest bypassing Kafka consumer worker.

Reads files from MinIO, splits, embeds, inserts via WeaviateStore directly.
Use after `client.collections.delete('DocChunk')` to repopulate cleanly.
Workaround for the long-lived-worker UTF-8 corruption bug.
"""
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "hybridrag" / "ingestion" / "ingestion_service"))


async def main():
    from entities.minio_common import AsyncMinioClient
    from helper.extractor import fetch_and_extract
    from src.hybridrag.ingestion.chunking.hierarchical import HierarchicalSplitter
    from src.hybridrag.ingestion.metadata.extractor import extract_metadata
    from src.hybridrag.ingestion.ingestion_service.weaviate_processor import _chunk_to_weaviate_dict, _uuid_for
    from src.hybridrag.ingestion.ingestion_service.entities.weaviate_store import WeaviateStore
    from src.hybridrag.ingestion.ingestion_service.helper.state_repo import FileStateRepo, FileState
    from src.hybridrag.ingestion.embedding.openai import embedder
    from src.config.settings import settings

    mc = AsyncMinioClient()
    splitter = HierarchicalSplitter()
    store = WeaviateStore(embedding_fn=embedder.embed, embedding_dim=embedder.get_dimension())
    state = FileStateRepo(settings.DATABASE_URL)

    files = [
        'qa_fb.md',
        'wiki.md',
        'tuyen_sinh_247.md',
        'Điểm 2023.md',
        'Điểm 2024.md',
        'Điểm 2025.md',
        'Thông báo tuyển sinh đại học chính quy năm 2026.txt',
    ]
    for key in files:
        text = await fetch_and_extract(mc, 'utehy', key)
        if not text:
            print(f'skip empty {key}', flush=True)
            continue
        info = await mc.info('utehy', key)
        etag = info.get('etag') or 'noversion'
        file_id = _uuid_for('utehy', key, etag)
        parents, children = splitter.split(text, file_id=file_id, key=key, doc_meta={'file_id': file_id, 'key': key})
        enriched = []
        for chunk in (*parents, *children):
            em = extract_metadata(text=chunk.content, header_path=chunk.header_path, filename=os.path.basename(key))
            d = _chunk_to_weaviate_dict(chunk, file_id=file_id, key=key, extra_meta=em)
            enriched.append(d)
        store.delete_by_key(key)
        await store.precompute_embeddings(enriched)
        inserted = await store.add_chunks(enriched)
        state.upsert(FileState(bucket='utehy', key=key, etag=etag, version_id=None, file_id=file_id))
        print(f'{key}: {inserted} chunks', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
