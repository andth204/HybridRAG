import hashlib
import inspect
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import faiss
import numpy as np

def _chunk_id_to_int64(chunk_id: str) -> np.int64:
    h = hashlib.blake2b(chunk_id.encode(), digest_size=8).digest()
    val = int.from_bytes(h, byteorder="big", signed=True)
    return np.int64(-2 if val == -1 else val)


class FAISSStore:
    def __init__(
        self,
        embedding_fn: Callable[[List[str]], Any],
        embedding_dim: int,
        cache_dir: str = "./faiss_cache",
        *,
        index_type: str = "hnsw",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        ivf_nlist: int = 4096,
        ivf_nprobe: int = 16,
    ):
        self._embed = embedding_fn
        self._dim = int(embedding_dim)
        self._index_type = (index_type or "hnsw").lower()
        self._hnsw_m = int(hnsw_m)
        self._hnsw_ef_construction = int(hnsw_ef_construction)
        self._hnsw_ef_search = int(hnsw_ef_search)
        self._ivf_nlist = int(ivf_nlist)
        self._ivf_nprobe = int(ivf_nprobe)
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        self._index_file = d / "index.faiss"
        self._meta_file = d / "meta.pkl"
        self._index: Optional[faiss.Index] = None
        self._meta: List[dict] = []
        self._meta_by_vid: Dict[int, dict] = {}
        self._tombstone_vids: set[int] = set()
        self._tombstone_vid_to_chunk_id: Dict[int, str] = {}
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if self._index_file.exists() and self._meta_file.exists():
            self._index = faiss.read_index(str(self._index_file))
            with open(self._meta_file, "rb") as f:
                meta_obj = pickle.load(f)

            if isinstance(meta_obj, list):
                self._meta = meta_obj or []
                self._tombstone_vids = set()
                self._tombstone_vid_to_chunk_id = {}
            else:
                self._meta = meta_obj.get("meta") or []
                self._tombstone_vids = set(meta_obj.get("tombstones") or [])
                self._tombstone_vid_to_chunk_id = dict(meta_obj.get("tombstone_chunk_ids") or {})
            self._rebuild_meta_cache()
            self._apply_runtime_params()
            return
        self._index = self._build_new_index()
        self._meta = []
        self._meta_by_vid = {}
        self._tombstone_vids = set()
        self._tombstone_vid_to_chunk_id = {}
        self._apply_runtime_params()

    def _save(self) -> None:
        if self._index is None:
            return

        tmp_index = self._index_file.with_suffix(".faiss.tmp")
        faiss.write_index(self._index, str(tmp_index))
        tmp_index.replace(self._index_file)
        tmp_meta = self._meta_file.with_suffix(".pkl.tmp")
        with open(tmp_meta, "wb") as f:
            pickle.dump(
                {
                    "meta": self._meta,
                    "tombstones": list(self._tombstone_vids),
                    "tombstone_chunk_ids": self._tombstone_vid_to_chunk_id,
                },
                f,
            )
        tmp_meta.replace(self._meta_file)

    def _build_new_index(self) -> faiss.Index:
        if self._index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self._dim)
            ivf = faiss.IndexIVFFlat(quantizer, self._dim, self._ivf_nlist, faiss.METRIC_INNER_PRODUCT)
            return faiss.IndexIDMap2(ivf)
        hnsw = faiss.IndexHNSWFlat(self._dim, self._hnsw_m, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexIDMap2(hnsw)

    def _unwrap(self) -> faiss.Index:
        assert self._index is not None
        return getattr(self._index, "index", self._index)

    def _apply_runtime_params(self) -> None:
        if self._index is None:
            return
        base = self._unwrap()
        if hasattr(base, "hnsw"):
            try:
                base.hnsw.efConstruction = self._hnsw_ef_construction
                base.hnsw.efSearch = self._hnsw_ef_search
            except Exception:
                pass
        if hasattr(base, "nprobe"):
            try:
                base.nprobe = self._ivf_nprobe
            except Exception:
                pass


    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n == 0, 1.0, n)

    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        result = await self._embed(texts) if inspect.iscoroutinefunction(self._embed) else self._embed(texts)
        return np.asarray(result, dtype=np.float32)

    def _rebuild_meta_cache(self) -> None:
        self._meta_by_vid = {int(m["vid"]): m for m in self._meta if "vid" in m}

    async def precompute_embeddings(self, chunks) -> None:
        existing_ids = {m["chunk_id"] for m in self._meta}
        new = [c for c in chunks if c.chunk_id not in existing_ids and c.chunk_id not in self._embed_cache]
        if not new:
            return

        texts = [c.text for c in new]
        vecs = self._norm(await self._get_embeddings(texts))
        for c, vec in zip(new, vecs):
            self._embed_cache[c.chunk_id] = vec

    async def add_chunks(self, chunks) -> int:
        if self._index is None:
            self._index = self._build_new_index()
            self._apply_runtime_params()

        existing_ids = {m["chunk_id"] for m in self._meta}
        new = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new:
            self._embed_cache.clear()
            return 0

        cached = [self._embed_cache.get(c.chunk_id) for c in new]
        if all(v is not None for v in cached):
            vecs = np.stack(cached, axis=0).astype(np.float32)
        else:
            texts = [c.text for c in new]
            vecs = self._norm(await self._get_embeddings(texts))

        self._embed_cache.clear()
        ids = np.array([_chunk_id_to_int64(c.chunk_id) for c in new], dtype=np.int64)
        for c, vid in zip(new, ids):
            ivid = int(vid)
            for t_vid, t_cid in list(self._tombstone_vid_to_chunk_id.items()):
                if t_cid == c.chunk_id:
                    self._tombstone_vids.discard(t_vid)
                    del self._tombstone_vid_to_chunk_id[t_vid]
                    break

            existing_meta = self._meta_by_vid.get(ivid)
            if existing_meta and existing_meta["chunk_id"] != c.chunk_id:
                raise RuntimeError(
                    f"FAISS id collision: vid={ivid} chunk={c.chunk_id} vs {existing_meta['chunk_id']}"
                )
            ghost = self._tombstone_vid_to_chunk_id.get(ivid)
            if ghost and ghost != c.chunk_id:
                raise RuntimeError(f"FAISS id collision (tombstone): vid={ivid} chunk={c.chunk_id} vs deleted {ghost}")

        if self._index_type == "ivf":
            base = self._unwrap()
            if hasattr(base, "is_trained") and not base.is_trained:
                try:
                    base.train(vecs)
                except Exception as e:
                    raise RuntimeError(f"IVF training failed: {e}") from e

        self._index.add_with_ids(vecs, ids)
        for c, vid in zip(new, ids):
            m = {"chunk_id": c.chunk_id, "vid": int(vid), "file_id": c.file_id, "key": c.key, "text": c.text}
            self._meta.append(m)
            self._meta_by_vid[int(vid)] = m

        self._save()
        return len(new)

    def delete_by_key(self, key: str) -> int:
        if self._index is None:
            return 0

        to_remove = [m for m in self._meta if m.get("key") == key]
        if not to_remove:
            return 0

        ids = np.asarray([int(m["vid"]) for m in to_remove], dtype=np.int64)
        for m in to_remove:
            vid = int(m["vid"])
            self._tombstone_vids.add(vid)
            self._tombstone_vid_to_chunk_id[vid] = m.get("chunk_id", "")
        try:
            self._index.remove_ids(ids)
        except Exception:
            pass

        before = len(self._meta)
        self._meta = [m for m in self._meta if m.get("key") != key]
        self._rebuild_meta_cache()
        self._save()
        return before - len(self._meta)

    def has_key(self, key: str) -> bool:
        return any(m.get("key") == key for m in self._meta)
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        nprobe: int | None = None,
        ef_search: int | None = None,
        oversample: int = 5,
    ) -> List[Dict]:
        if self._index is None or self._index.ntotal == 0:
            return []

        top_k = max(1, int(top_k))
        base = self._unwrap()
        if self._index_type == "ivf" and hasattr(base, "nprobe"):
            base.nprobe = int(nprobe) if nprobe is not None else self._ivf_nprobe
        if self._index_type == "hnsw" and hasattr(base, "hnsw") and ef_search is not None:
            try:
                base.hnsw.efSearch = int(ef_search)
            except Exception:
                pass

        q = self._norm(await self._get_embeddings([query]))
        search_k = min(self._index.ntotal, top_k * max(1, int(oversample)))
        scores, ids = self._index.search(q, search_k)
        out: List[Dict] = []
        for s, vid in zip(scores[0], ids[0]):
            ivid = int(vid)
            if ivid == -1 or ivid in self._tombstone_vids:
                continue
            m = self._meta_by_vid.get(ivid)
            if m:
                out.append({**m, "score": float(s)})
            if len(out) >= top_k:
                break
        return out