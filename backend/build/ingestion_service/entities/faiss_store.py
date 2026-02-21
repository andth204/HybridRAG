import inspect
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Callable, Any
import hashlib
import faiss


def _chunk_id_to_int64(chunk_id: str) -> np.int64:
    h = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(h, byteorder="big", signed=True)
    return np.int64(val)


class FAISSStore:
    def __init__(self, embedding_fn: Callable[[List[str]], Any], embedding_dim: int, cache_dir: str = "./faiss_cache"):
        self._embed = embedding_fn
        self._dim = int(embedding_dim)
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        self._index_file = d / "index.faiss"
        self._meta_file = d / "meta.pkl"
        self._index = None
        self._meta: List[dict] = [] # meta entries: {"chunk_id": str, "vid": int, "file_id": str, "key": str, "text": str}
        self._load()

    def _load(self) -> None:
        if self._index_file.exists() and self._meta_file.exists():
            self._index = faiss.read_index(str(self._index_file))
            with open(self._meta_file, "rb") as f:
                self._meta = pickle.load(f) or []
        else:
            base = faiss.IndexFlatIP(self._dim)
            self._index = faiss.IndexIDMap2(base)
            self._meta = []

    def _save(self) -> None:
        faiss.write_index(self._index, str(self._index_file))
        with open(self._meta_file, "wb") as f:
            pickle.dump(self._meta, f)

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n == 0, 1.0, n)

    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        if inspect.iscoroutinefunction(self._embed):
            result = await self._embed(texts)
        else:
            result = self._embed(texts)
        return np.asarray(result, dtype=np.float32)

    async def add_chunks(self, chunks) -> int:
        existing = {m["chunk_id"] for m in self._meta}
        new = [c for c in chunks if c.chunk_id not in existing]
        if not new:
            return 0

        vecs = self._norm(await self._get_embeddings([c.text for c in new]))
        ids = np.array([_chunk_id_to_int64(c.chunk_id) for c in new], dtype=np.int64)
        meta_by_vid = {int(m["vid"]): m["chunk_id"] for m in self._meta if "vid" in m}
        for c, vid in zip(new, ids):
            old = meta_by_vid.get(int(vid))
            if old is not None and old != c.chunk_id:
                raise RuntimeError(f"FAISS id collision: vid={int(vid)} for {c.chunk_id} collides with {old}")

        self._index.add_with_ids(vecs, ids)

        for c, vid in zip(new, ids):
            self._meta.append(
                {
                    "chunk_id": c.chunk_id,
                    "vid": int(vid),
                    "file_id": c.file_id,
                    "key": c.key,
                    "text": c.text,
                }
            )
        self._save()
        return len(new)

    def delete_by_key(self, key: str) -> int:
        ids = [m["vid"] for m in self._meta if m.get("key") == key]
        if not ids:
            return 0
        self._index.remove_ids(np.asarray(ids, dtype=np.int64))
        before = len(self._meta)
        self._meta = [m for m in self._meta if m.get("key") != key]
        self._save()
        removed = before - len(self._meta)
        return removed

    def has_key(self, key: str) -> bool:
        return any(m.get("key") == key for m in self._meta)

    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self._index is None or self._index.ntotal == 0:
            return []

        q = self._norm(await self._get_embeddings([query]))
        top_k = max(1, int(top_k))
        scores, ids = self._index.search(q, min(top_k, int(self._index.ntotal)))
        meta_by_vid = {int(m["vid"]): m for m in self._meta}

        out: List[Dict] = []
        for s, vid in zip(scores[0], ids[0]):
            if int(vid) < 0:
                continue
            m = meta_by_vid.get(int(vid))
            if m:
                out.append({**m, "score": float(s)})
        return out