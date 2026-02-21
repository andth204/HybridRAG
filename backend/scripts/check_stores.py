import os, sys, pickle
from pathlib import Path
from dotenv import load_dotenv
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
BM25_CACHE_DIR  = os.getenv("BM25_CACHE_DIR")
FAISS_CACHE_DIR = os.getenv("FAISS_CACHE_DIR")

def check_bm25():
    file = Path(BM25_CACHE_DIR) / "bm25.pkl"
    if not file.exists():
        print(f"[BM25] Chưa có cache. (path={file})")
        return
    data = pickle.load(open(file, "rb")) or {}
    meta = data.get("meta", {}) or {}
    print(f"\n[BM25] Tổng: {len(meta)} chunks (path={file})")
    print("-" * 60)
    for i, (cid, m) in enumerate(list(meta.items())[:50]):
        print(f"  [{i}] chunk_id : {cid}")
        print(f"       file_id  : {m.get('file_id')}")
        print(f"       key      : {m.get('key')}")
        t = (m.get("text") or "")
        print(f"       text     : {t[:100]}...")
        print()

def check_faiss():
    meta_file = Path(FAISS_CACHE_DIR) / "meta.pkl"
    idx_file  = Path(FAISS_CACHE_DIR) / "index.faiss"
    if not meta_file.exists() or not idx_file.exists():
        print(f"[FAISS] Chưa có cache. (path={meta_file})")
        return
    import faiss
    raw   = pickle.load(open(meta_file, "rb"))
    index = faiss.read_index(str(idx_file))
    if isinstance(raw, dict):
        meta_items = list(raw.get("id2meta", {}).values())
    else:
        meta_items = raw
    print(f"\n[FAISS] Tổng: {index.ntotal} vectors / {len(meta_items)} meta")
    print("-" * 60)
    for i, m in enumerate(meta_items):
        print(f"  [{i}] chunk_id : {m['chunk_id']}")
        print(f"       file_id  : {m['file_id']}")
        print(f"       key      : {m['key']}")
        print(f"       text     : {m['text'][:100]}...")
        print()

if __name__ == "__main__":
    check_bm25()
    check_faiss()