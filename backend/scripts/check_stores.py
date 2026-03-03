import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
import faiss

BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BACKEND_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

def resolve_dir(p: str | None) -> Path:
    if not p:
        raise RuntimeError("BM25_CACHE_DIR/FAISS_CACHE_DIR chưa được set trong .env")
    pth = Path(p)
    if not pth.is_absolute():
        pth = (BACKEND_DIR / pth).resolve()
    return pth

BM25_CACHE_DIR  = resolve_dir(os.getenv("BM25_CACHE_DIR"))
FAISS_CACHE_DIR = resolve_dir(os.getenv("FAISS_CACHE_DIR"))

def check_bm25(limit: int = 50, text_len: int = 200):
    file = BM25_CACHE_DIR / "bm25.pkl"
    if not file.exists():
        print(f"[BM25] Chưa có cache. (path={file})")
        return

    data = pickle.load(open(file, "rb")) or {}
    meta = (data.get("meta", {}) or {})

    print(f"\n[BM25] Tổng: {len(meta)} chunks (path={file})")
    print("-" * 80)

    for i, (chunk_id, m) in enumerate(list(meta.items())[:limit]):
        key = m.get("key")
        file_id = m.get("file_id")
        txt = (m.get("text") or "").replace("\n", " ").strip()

        print(f"[{i}] chunk_id: {chunk_id}")
        print(f"    key    : {key}")
        print(f"    file_id : {file_id}")
        print(f"    text   : {txt[:text_len]}{'...' if len(txt) > text_len else ''}")
        print()

def check_faiss(limit: int = 50, text_len: int = 200):
    meta_file = FAISS_CACHE_DIR / "meta.pkl"
    idx_file  = FAISS_CACHE_DIR / "index.faiss"
    if not meta_file.exists() or not idx_file.exists():
        print(f"[FAISS] Chưa có cache. (path={meta_file})")
        return

    raw = pickle.load(open(meta_file, "rb"))
    index = faiss.read_index(str(idx_file))
    if isinstance(raw, dict):
        meta_items = list((raw.get("id2meta") or {}).values())
    else:
        meta_items = list(raw or [])

    print(f"\n[FAISS] Tổng: {index.ntotal} vectors / {len(meta_items)} meta (path={meta_file})")
    print("-" * 80)
    for i, m in enumerate(meta_items[:limit]):
        chunk_id = m.get("chunk_id")
        key = m.get("key")
        file_id = m.get("file_id")
        txt = (m.get("text") or "").replace("\n", " ").strip()

        print(f"[{i}] chunk_id: {chunk_id}")
        print(f"    key    : {key}")
        print(f"    file_id : {file_id}")
        print(f"    text   : {txt[:text_len]}{'...' if len(txt) > text_len else ''}")
        print()

if __name__ == "__main__":
    check_bm25(limit=50, text_len=200)
    check_faiss(limit=50, text_len=200)