"""
Đánh giá module truy hồi Hybrid RAG - gọi trực tiếp không qua HTTP API
Dataset: backend/data/data_test.json (50 mẫu)
Metrics: HR@K, Precision@K, Recall@K, MRR@K
So sánh: BM25-only vs Vector-only vs Hybrid (BM25+Vec+RRF)
Ghi chú: Reranker bị tắt (USE_RERANKER=False) vì không có GPU CUDA
"""
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybridrag.retrieval.hybrid import HybridSearcher
from src.hybridrag.retrieval.bm25_search import BM25Searcher
from src.hybridrag.retrieval.vector_search import VectorSearcher

K_VALUES = [1, 3, 5]


def normalize_source(s: str) -> str:
    return Path(s).stem.strip().lower()


def relevant_sources(sample: Dict) -> List[str]:
    return [normalize_source(f) for f in sample.get("source_files", [])]


def compute_metrics(retrieved_list: List[List[str]], relevant_list: List[List[str]], k_values: List[int]) -> Dict:
    """
    Metrics tính tại source-file level (deduplicate chunk từ cùng file).
    retrieved_list: list of source names per query (có thể trùng lặp từ cùng file)
    relevant_list:  list of ground-truth source names per query
    """
    n = len(retrieved_list)
    metrics = {}
    rr_sum = 0.0
    max_k = max(k_values)

    for k in k_values:
        hits, prec_sum, rec_sum = 0, 0.0, 0.0
        for retrieved, relevant in zip(retrieved_list, relevant_list):
            # Deduplicate: lấy unique sources theo thứ tự xuất hiện, top-k unique
            seen, unique_top_k = set(), []
            for r in retrieved:
                if r not in seen:
                    seen.add(r)
                    unique_top_k.append(r)
                if len(unique_top_k) >= k:
                    break
            rel_set = set(relevant)
            tp  = len(set(unique_top_k) & rel_set)
            hit = tp > 0
            hits     += int(hit)
            prec_sum += tp / k
            rec_sum  += tp / len(rel_set) if rel_set else 0
        metrics[f"HR@{k}"]        = hits / n
        metrics[f"Precision@{k}"] = prec_sum / n
        metrics[f"Recall@{k}"]    = rec_sum / n

    # MRR tính trên unique sources
    for retrieved, relevant in zip(retrieved_list, relevant_list):
        rel_set = set(relevant)
        seen = set()
        rank = 0
        for r in retrieved[:max_k * 3]:  # tìm trong nhiều results hơn
            if r not in seen:
                seen.add(r)
                rank += 1
                if r in rel_set:
                    rr_sum += 1.0 / rank
                    break
            if rank >= max_k:
                break
    metrics[f"MRR@{max_k}"] = rr_sum / n
    return metrics


def extract_sources_from_results(results) -> List[str]:
    """Trích xuất và chuẩn hóa tên source từ kết quả tìm kiếm."""
    sources = []
    for r in results:
        # Thử nhiều trường khác nhau tùy structure của result
        src = (
            r.get("source") or
            r.get("metadata", {}).get("source") or
            r.get("metadata", {}).get("file_name") or
            r.get("key") or ""
        )
        sources.append(normalize_source(src))
    return sources


async def eval_bm25(samples: List[Dict], k: int = 8) -> tuple:
    searcher = BM25Searcher()
    searcher.load_index()
    retrieved_list, latencies = [], []
    for i, s in enumerate(samples):
        t0 = time.perf_counter()
        results = await searcher.search(s["question"], top_k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved_list.append(extract_sources_from_results(results))
        if (i + 1) % 10 == 0:
            print(f"  BM25: {i+1}/{len(samples)}")
    return retrieved_list, latencies


async def eval_vector(samples: List[Dict], k: int = 8) -> tuple:
    searcher = VectorSearcher()
    searcher.load_index()
    retrieved_list, latencies = [], []
    for i, s in enumerate(samples):
        t0 = time.perf_counter()
        results = await searcher.search(s["question"], top_k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved_list.append(extract_sources_from_results(results))
        if (i + 1) % 10 == 0:
            print(f"  Vector: {i+1}/{len(samples)}")
    return retrieved_list, latencies


async def eval_hybrid(samples: List[Dict], k: int = 8) -> tuple:
    searcher = HybridSearcher()
    searcher.load_indexes()
    retrieved_list, latencies = [], []
    for i, s in enumerate(samples):
        t0 = time.perf_counter()
        results = await searcher.search(s["question"], vector_k=k, bm25_k=k, use_reranker=False)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved_list.append(extract_sources_from_results(results))
        if (i + 1) % 10 == 0:
            print(f"  Hybrid: {i+1}/{len(samples)}")
    return retrieved_list, latencies


def print_table(all_metrics: Dict[str, Dict]):
    max_k = max(K_VALUES)
    header = f"  {'Mode':<28}" + "".join(f"  HR@{k}" for k in K_VALUES) + "".join(f"  P@{k}" for k in K_VALUES) + "".join(f"  R@{k}" for k in K_VALUES) + f"  MRR@{max_k}  Lat(ms)"
    print("\n" + "="*100)
    print(header)
    print("="*100)
    for mode, m in all_metrics.items():
        row = f"  {mode:<28}"
        for k in K_VALUES:
            row += f"  {m[f'HR@{k}']:.4f}"
        for k in K_VALUES:
            row += f"  {m[f'Precision@{k}']:.4f}"
        for k in K_VALUES:
            row += f"  {m[f'Recall@{k}']:.4f}"
        row += f"  {m[f'MRR@{max_k}']:.4f}  {m['avg_latency_ms']:.1f}"
        print(row)
    print("="*100)


async def main():
    data_path = Path(__file__).parent.parent / "data" / "data_test.json"
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    samples = dataset["samples"]
    K = 8

    print(f"\nDanh gia module Hybrid RAG Retrieval")
    print(f"Dataset: {dataset['dataset_name']} | {len(samples)} mau | K={K}")
    print(f"USE_RERANKER=False (no CUDA GPU)")
    print("="*60)

    relevant_list = [relevant_sources(s) for s in samples]
    all_metrics: Dict[str, Dict] = {}
    all_retrieved: Dict[str, List] = {}

    # BM25
    print("\n[1/3] BM25-only...")
    ret_bm25, lat_bm25 = await eval_bm25(samples, k=K)
    m_bm25 = compute_metrics(ret_bm25, relevant_list, K_VALUES)
    m_bm25["avg_latency_ms"] = sum(lat_bm25) / len(lat_bm25)
    all_metrics["BM25-only"] = m_bm25
    all_retrieved["BM25-only"] = ret_bm25

    # Vector
    print("\n[2/3] Vector-only (FAISS + text-embedding-3-small)...")
    ret_vec, lat_vec = await eval_vector(samples, k=K)
    m_vec = compute_metrics(ret_vec, relevant_list, K_VALUES)
    m_vec["avg_latency_ms"] = sum(lat_vec) / len(lat_vec)
    all_metrics["Vector-only"] = m_vec
    all_retrieved["Vector-only"] = ret_vec

    # Hybrid
    print("\n[3/3] Hybrid (BM25 + Vector + RRF, no rerank)...")
    ret_hyb, lat_hyb = await eval_hybrid(samples, k=K)
    m_hyb = compute_metrics(ret_hyb, relevant_list, K_VALUES)
    m_hyb["avg_latency_ms"] = sum(lat_hyb) / len(lat_hyb)
    all_metrics["Hybrid (no rerank)"] = m_hyb
    all_retrieved["Hybrid (no rerank)"] = ret_hyb

    print_table(all_metrics)

    # Theo do kho
    print("\n  HR@3 theo do kho (Hybrid):")
    by_diff: Dict[str, Dict] = {}
    for i, s in enumerate(samples):
        d = s.get("difficulty", "medium")
        by_diff.setdefault(d, {"hit": 0, "total": 0})
        by_diff[d]["total"] += 1
        if any(r in relevant_list[i] for r in ret_hyb[i][:3]):
            by_diff[d]["hit"] += 1
    for d, stat in sorted(by_diff.items()):
        print(f"    {d:<12}: {stat['hit']}/{stat['total']}  ({100*stat['hit']/stat['total']:.1f}%)")

    # Theo category
    print("\n  HR@3 theo category (Hybrid, top 5):")
    by_cat: Dict[str, Dict] = {}
    for i, s in enumerate(samples):
        cat = s.get("category", "other")
        by_cat.setdefault(cat, {"hit": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if any(r in relevant_list[i] for r in ret_hyb[i][:3]):
            by_cat[cat]["hit"] += 1
    for cat, stat in sorted(by_cat.items(), key=lambda x: -x[1]["total"])[:7]:
        print(f"    {cat:<35}: {stat['hit']}/{stat['total']}  ({100*stat['hit']/stat['total']:.1f}%)")

    out_path = Path(__file__).parent.parent / "data" / "eval_retrieval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nKet qua luu: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
