"""
Re-index all source documents into BM25 and FAISS stores.
Run from: backend/  with: python scripts/reindex_all.py
"""
import sys
import asyncio
import hashlib
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybridrag.ingestion.chunking.splitter import TextSplitter
from src.hybridrag.ingestion.embedding.openai import OpenAIEmbedder
from src.hybridrag.ingestion.ingestion_service.entities.bm25_store import BM25Store
from src.hybridrag.ingestion.ingestion_service.entities.faiss_store import FAISSStore
from src.hybridrag.ingestion.ingestion_service.entities.models import Chunk
from src.config.settings import settings


def make_file_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"local://{key}"))


async def main():
    samples_dir = Path(__file__).parent.parent / "data" / "samples"
    files = sorted(samples_dir.iterdir())

    print(f"\nRe-indexing {len(files)} files from {samples_dir}")
    print("=" * 60)

    splitter = TextSplitter()
    embedder = OpenAIEmbedder()

    bm25 = BM25Store(cache_dir=str(settings.BM25_CACHE_DIR))
    faiss = FAISSStore(
        embedding_fn=embedder.embed,
        embedding_dim=embedder.get_dimension(),
        cache_dir=str(settings.FAISS_CACHE_DIR),
    )

    total_bm25 = 0
    total_faiss = 0

    for fpath in files:
        if not fpath.is_file():
            continue
        key = fpath.name
        print(f"\n  [{key}]")

        # Delete old chunks for this file
        del_b = bm25.delete_by_key(key)
        del_f = faiss.delete_by_key(key)
        if del_b or del_f:
            print(f"    Deleted: BM25={del_b}  FAISS={del_f} old chunks")

        # Read and split
        text = fpath.read_text(encoding="utf-8")
        parts = splitter.split_text(text)
        print(f"    Split into {len(parts)} chunks  (file size={len(text):,} chars)")

        file_id = make_file_id(key)
        chunks = [
            Chunk(file_id=file_id, key=key, text=part)
            for part in parts
        ]

        # Pre-compute embeddings (batch API call)
        print(f"    Embedding {len(chunks)} chunks...", end=" ", flush=True)
        await faiss.precompute_embeddings(chunks)
        print("done")

        # Add to stores
        added_b = bm25.add_chunks(chunks)
        added_f = await faiss.add_chunks(chunks)
        print(f"    Added: BM25={added_b}  FAISS={added_f}")
        total_bm25 += added_b
        total_faiss += added_f

    print("\n" + "=" * 60)
    print(f"  Total added:  BM25={total_bm25}  FAISS={total_faiss}")

    # Verify chunk distribution
    print("\n  Chunk distribution in BM25 store:")
    from collections import Counter
    counter = Counter(v["key"] for v in bm25.meta.values())
    for k, cnt in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"    {k:<55}: {cnt:>4} chunks")


if __name__ == "__main__":
    asyncio.run(main())
