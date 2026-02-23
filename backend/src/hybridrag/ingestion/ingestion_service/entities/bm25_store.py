import pickle
import math
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter, defaultdict

def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Store:
    def __init__(self, cache_dir: str = "./bm25_cache", k1: float = 1.5, b: float = 0.75):
        p = Path(cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        self._file = p / "bm25.pkl"
        self.k1 = float(k1)
        self.b = float(b)
        self.meta: Dict[str, Dict[str, Any]] = {}
        self.dl: Dict[str, int] = {}
        self.doc_terms: Dict[str, List[str]] = {}
        self.df: Dict[str, int] = {}
        self.postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.N: int = 0
        self.avgdl: float = 0.0
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with open(self._file, "rb") as f:
                data = pickle.load(f) or {}

            self.meta = data.get("meta", {}) or {}
            self.dl = data.get("dl", {}) or {}
            self.doc_terms = data.get("doc_terms", {}) or {}
            self.df = data.get("df", {}) or {}
            postings_raw = data.get("postings", {}) or {}
            self.postings = defaultdict(dict, postings_raw)
            self.N = len(self.meta)
            self.avgdl = (sum(self.dl.values()) / self.N) if self.N else 0.0
        except Exception as e:
            self.meta, self.dl, self.doc_terms, self.df = {}, {}, {}, {}
            self.postings = defaultdict(dict)
            self.N, self.avgdl = 0, 0.0

    def _save(self) -> None:
        with open(self._file, "wb") as f:
            pickle.dump(
                {
                    "meta": self.meta,
                    "dl": self.dl,
                    "doc_terms": self.doc_terms,
                    "df": self.df,
                    "postings": dict(self.postings),
                },
                f,
            )

    def _recalc_stats(self) -> None:
        self.N = len(self.meta)
        self.avgdl = (sum(self.dl.values()) / self.N) if self.N else 0.0

    def add_chunks(self, chunks) -> int:
        added = 0
        for c in chunks:
            cid = c.chunk_id
            if cid in self.meta:
                continue

            tokens = _tokenize(c.text)
            tf = Counter(tokens)
            unique_terms = list(tf.keys())

            # store meta
            self.meta[cid] = {
                "chunk_id": cid,
                "file_id": c.file_id,
                "key": c.key,
                "text": c.text,
            }
            self.dl[cid] = len(tokens)
            self.doc_terms[cid] = unique_terms
            for term, freq in tf.items():
                if cid not in self.postings[term]:
                    self.df[term] = self.df.get(term, 0) + 1
                self.postings[term][cid] = int(freq)

            added += 1

        if added:
            self._recalc_stats()
            self._save()
        return added

    def _delete_chunk_id(self, cid: str) -> None:
        if cid not in self.meta:
            return

        terms = self.doc_terms.get(cid, [])
        for term in terms:
            posting = self.postings.get(term)
            if not posting:
                continue

            # remove tf entry
            if cid in posting:
                del posting[cid]

                # update df
                self.df[term] = max(0, self.df.get(term, 1) - 1)
                if not posting:
                    self.postings.pop(term, None)
                    self.df.pop(term, None)

        self.meta.pop(cid, None)
        self.dl.pop(cid, None)
        self.doc_terms.pop(cid, None)

    def delete_by_key(self, key: str) -> int:
        cids = [cid for cid, m in self.meta.items() if m.get("key") == key]
        if not cids:
            return 0

        for cid in cids:
            self._delete_chunk_id(cid)

        self._recalc_stats()
        self._save()
        return len(cids)

    def has_key(self, key: str) -> bool:
        return any(m.get("key") == key for m in self.meta.values())

    def _idf(self, df: int) -> float:
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_terms = _tokenize(query)
        if not self.N or not q_terms:
            return []

        scores: Dict[str, float] = defaultdict(float)
        for term in q_terms:
            posting = self.postings.get(term)
            if not posting:
                continue

            df = self.df.get(term, 0)
            if df <= 0:
                continue

            idf = self._idf(df)
            for cid, tf in posting.items():
                dl = self.dl.get(cid, 0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                scores[cid] += idf * (tf * (self.k1 + 1.0)) / (denom or 1.0)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(1, int(top_k))]

        out: List[Dict] = []
        for cid, s in ranked:
            m = self.meta.get(cid)
            if m:
                out.append({**m, "score": float(s)})
        return out