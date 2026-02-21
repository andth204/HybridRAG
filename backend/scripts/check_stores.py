import os, sys, pickle
from pathlib import Path
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
BM25_CACHE_DIR  = os.getenv("BM25_CACHE_DIR",  "./bm25_cache")
FAISS_CACHE_DIR = os.getenv("FAISS_CACHE_DIR", "./faiss_cache")


def check_bm25():
    file = Path(BM25_CACHE_DIR) / "corpus.pkl"
    if not file.exists():
        print(f"[BM25] Chưa có cache. (path={file})")
        return
    corpus = pickle.load(open(file, "rb"))
    print(f"\n[BM25] Tổng: {len(corpus)} chunks")
    print("-" * 60)
    for i, c in enumerate(corpus):
        print(f"  [{i}] chunk_id : {c['chunk_id']}")
        print(f"       file_id  : {c['file_id']}")
        print(f"       key      : {c['key']}")
        print(f"       text     : {c['text'][:100]}...")
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