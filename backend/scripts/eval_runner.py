"""
Golden eval runner for HybridRAG (Phase 0.5 baseline).

Runs the golden query set through HybridSearcher (+ optional AnswerGenerator)
and reports retrieval recall@K, MRR, keyword coverage, and latency percentiles.

Outputs:
  - <output>.json : raw per-record details + aggregates
  - <output>.md   : human-readable markdown report

Usage (from backend/):
  python scripts/eval_runner.py                            # full run with gen
  python scripts/eval_runner.py --no-gen                   # retrieval only
  python scripts/eval_runner.py --limit 10 --top-k 5       # smoke test
  python scripts/eval_runner.py --input data/eval/foo.jsonl --output data/eval/baselines/foo

NOTE: This script does NOT modify any indexing/retrieval code. It honors
USE_RERANKER from settings and skips generation if OPENAI_API_KEY is empty
or --no-gen is set.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Project import path: backend/ as root so `from src.hybridrag... import ...` works.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.config.settings import settings  # noqa: E402
from src.hybridrag.retrieval.hybrid import HybridSearcher  # noqa: E402

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - tqdm listed in requirements but be safe
    def tqdm(it: Iterable, **_kwargs: Any) -> Iterable:  # type: ignore
        return it

log = logging.getLogger("eval_runner")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT = BACKEND_ROOT / "data" / "eval" / "golden_v0.jsonl"
DEFAULT_OUTPUT_BASE = BACKEND_ROOT / "data" / "eval" / "baselines" / "v0"

# Used only when --input file is missing (dev convenience). Real run requires
# the real golden file produced by the parallel agent.
MOCK_RECORDS: List[Dict[str, Any]] = [
    {
        "id": "mock-001",
        "query": "Điểm chuẩn ngành CNTT năm 2024 là bao nhiêu?",
        "expected_keywords": ["điểm chuẩn", "công nghệ thông tin", "2024"],
        "expected_source": "diem_chuan_2024.md",
        "intent": "score_lookup",
        "category": "admission_score",
    },
    {
        "id": "mock-002",
        "query": "Trường có những phương thức xét tuyển nào?",
        "expected_keywords": ["xét tuyển", "phương thức"],
        "expected_source": "phuong_thuc_tuyen_sinh.md",
        "intent": "admission_method",
        "category": "admission",
    },
    {
        "id": "mock-003",
        "query": "Học phí ngành Kế toán bao nhiêu một kỳ?",
        "expected_keywords": ["học phí", "kế toán"],
        "expected_source": "hoc_phi.md",
        "intent": "tuition_lookup",
        "category": "tuition",
    },
]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_golden(path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    """Returns (records, used_mock). Falls back to MOCK_RECORDS if file missing."""
    if not path.exists():
        log.warning("Input file %s not found - falling back to MOCK_RECORDS. "
                    "Real run requires the golden_v0.jsonl from the parallel agent.", path)
        return list(MOCK_RECORDS), True

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed JSONL at %s:%d (%s)", path, lineno, exc)
                continue
            if not isinstance(obj, dict):
                log.warning("Skipping non-object record at %s:%d", path, lineno)
                continue
            records.append(obj)
    return records, False


def basename_norm(path_like: str) -> str:
    """Normalize a path-like string to its basename (stem-friendly), lowercased."""
    if not isinstance(path_like, str):
        return ""
    cleaned = path_like.strip()
    if not cleaned:
        return ""
    # strip query/fragment first
    cleaned = cleaned.split("?", 1)[0].split("#", 1)[0]
    # split on both Unix and Windows separators
    for sep in ("\\", "/"):
        if sep in cleaned:
            cleaned = cleaned.rsplit(sep, 1)[-1]
    return cleaned.strip().lower()


def stem_norm(path_like: str) -> str:
    """Return basename without extension, lowercased."""
    base = basename_norm(path_like)
    if not base:
        return ""
    if "." in base:
        return base.rsplit(".", 1)[0]
    return base


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    # Nearest-rank, clamped to [0, n-1]. Good enough at small n.
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return float(s[k])


def find_rank(retrieved: List[Dict[str, Any]], expected_source: str) -> int:
    """Return 1-based rank of first match, 0 if not found.

    Matches on basename (with extension) or stem (without extension).
    """
    if not expected_source or not retrieved:
        return 0
    target_base = basename_norm(expected_source)
    target_stem = stem_norm(expected_source)
    for idx, doc in enumerate(retrieved, start=1):
        key_val = doc.get("key") if isinstance(doc, dict) else None
        if not isinstance(key_val, str) or not key_val.strip():
            # fall back to other source-like fields
            for fallback_field in ("source", "file_name", "title", "doc_id", "id"):
                v = doc.get(fallback_field) if isinstance(doc, dict) else None
                if isinstance(v, str) and v.strip():
                    key_val = v
                    break
        if not isinstance(key_val, str) or not key_val.strip():
            continue
        cand_base = basename_norm(key_val)
        cand_stem = stem_norm(key_val)
        if cand_base and (cand_base == target_base or cand_stem == target_stem):
            return idx
    return 0


def keyword_coverage(answer: str, expected_keywords: List[str]) -> Optional[float]:
    if not expected_keywords:
        return None
    if not answer:
        return 0.0
    low = answer.lower()
    hits = sum(1 for kw in expected_keywords if isinstance(kw, str) and kw.strip() and kw.lower() in low)
    return hits / len(expected_keywords)


def get_doc_content(doc: Dict[str, Any]) -> str:
    for k in ("content", "text", "chunk", "document"):
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# ---------------------------------------------------------------------------
# Per-record evaluation
# ---------------------------------------------------------------------------
async def eval_one(
    record: Dict[str, Any],
    *,
    searcher: HybridSearcher,
    generator: Any,
    top_k: int,
    run_gen: bool,
) -> Dict[str, Any]:
    rid = str(record.get("id", ""))
    query = record.get("query") or record.get("question") or ""
    expected_source = record.get("expected_source") or ""
    expected_keywords = record.get("expected_keywords") or []
    intent = record.get("intent") or "unknown"
    category = record.get("category") or "unknown"

    result: Dict[str, Any] = {
        "id": rid,
        "query": query,
        "intent": intent,
        "category": category,
        "expected_source": expected_source,
        "expected_keywords": expected_keywords,
        "retrieved": [],
        "rank": 0,
        "recall_at_k": 0,
        "mrr": 0.0,
        "retrieval_ms": 0.0,
        "generation_ms": None,
        "answer": None,
        "keyword_coverage": None,
        "error": None,
    }

    if not query:
        result["error"] = "empty_query"
        return result

    # Retrieval
    retrieved: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    try:
        retrieved = await searcher.search(
            query=query,
            rerank_top_k=top_k,
        )
    except Exception as exc:  # noqa: BLE001 - intentional broad catch per spec
        result["error"] = f"retrieval_failed: {type(exc).__name__}: {exc}"
        result["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        return result
    result["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    # Persist a compact view of retrieved docs
    compact_retrieved: List[Dict[str, Any]] = []
    for doc in retrieved[:max(top_k, 5)]:
        if not isinstance(doc, dict):
            continue
        compact_retrieved.append({
            "key": doc.get("key", ""),
            "rrf_score": doc.get("rrf_score"),
            "rerank_score": doc.get("rerank_score"),
            "vector_score": doc.get("vector_score"),
            "bm25_score": doc.get("bm25_score"),
        })
    result["retrieved"] = compact_retrieved

    # Rank + recall + MRR
    rank = find_rank(retrieved[:top_k], expected_source)
    result["rank"] = rank
    result["recall_at_k"] = 1 if rank > 0 else 0
    result["mrr"] = (1.0 / rank) if rank > 0 else 0.0

    # Optional generation
    if run_gen and generator is not None and retrieved:
        t1 = time.perf_counter()
        try:
            answer = await generator.answer_text(
                query=query,
                retrieved_docs=retrieved,
                timeout=30.0,
            )
            result["answer"] = answer
            result["keyword_coverage"] = keyword_coverage(answer, expected_keywords)
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"generation_failed: {type(exc).__name__}: {exc}"
        result["generation_ms"] = (time.perf_counter() - t1) * 1000

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(
    rows: List[Dict[str, Any]],
    *,
    top_k: int,
    gen_enabled: bool,
) -> Dict[str, Any]:
    n = len(rows)
    success_rows = [r for r in rows if not r.get("error")]
    n_success = len(success_rows)

    recall_vals = [r["recall_at_k"] for r in success_rows]
    mrr_vals = [r["mrr"] for r in success_rows]
    ret_lat = [r["retrieval_ms"] for r in success_rows]

    mean_recall = (sum(recall_vals) / n_success) if n_success else 0.0
    mean_mrr = (sum(mrr_vals) / n_success) if n_success else 0.0

    # Generation aggregates (only over rows that actually produced answers)
    gen_rows = [r for r in success_rows if r.get("generation_ms") is not None]
    gen_lat = [r["generation_ms"] for r in gen_rows]
    kw_cov_vals = [r["keyword_coverage"] for r in gen_rows if r.get("keyword_coverage") is not None]
    mean_kw_cov: Optional[float] = (sum(kw_cov_vals) / len(kw_cov_vals)) if kw_cov_vals else None

    # Per-intent / per-category breakdowns
    def _breakdown(field: str) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in success_rows:
            groups[str(r.get(field, "unknown"))].append(r)
        out: Dict[str, Dict[str, Any]] = {}
        for key, grp in groups.items():
            recall = sum(g["recall_at_k"] for g in grp) / len(grp)
            mrr = sum(g["mrr"] for g in grp) / len(grp)
            kw_grp = [g["keyword_coverage"] for g in grp if g.get("keyword_coverage") is not None]
            out[key] = {
                "count": len(grp),
                "recall_at_k": recall,
                "mrr": mrr,
                "keyword_coverage": (sum(kw_grp) / len(kw_grp)) if kw_grp else None,
            }
        return out

    failures = [
        {
            "id": r["id"],
            "query": r["query"],
            "intent": r.get("intent"),
            "category": r.get("category"),
            "expected_source": r.get("expected_source"),
            "top3_keys": [d.get("key", "") for d in (r.get("retrieved") or [])[:3]],
            "error": r.get("error"),
        }
        for r in rows
        if r.get("error") or r.get("recall_at_k") == 0
    ]

    return {
        "total": n,
        "success": n_success,
        "errors": n - n_success,
        f"recall_at_{top_k}": mean_recall,
        "mrr": mean_mrr,
        "keyword_coverage": mean_kw_cov,
        "retrieval_ms_p50": percentile(ret_lat, 50.0),
        "retrieval_ms_p95": percentile(ret_lat, 95.0),
        "generation_ms_p50": percentile(gen_lat, 50.0) if gen_enabled and gen_lat else None,
        "generation_ms_p95": percentile(gen_lat, 95.0) if gen_enabled and gen_lat else None,
        "gen_runs": len(gen_lat),
        "by_intent": _breakdown("intent"),
        "by_category": _breakdown("category"),
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(v: Optional[float], spec: str = ".4f") -> str:
    if v is None:
        return "n/a"
    try:
        return format(v, spec)
    except Exception:
        return str(v)


def write_markdown(
    path: Path,
    *,
    aggregates: Dict[str, Any],
    rows: List[Dict[str, Any]],
    top_k: int,
    gen_enabled: bool,
    input_path: Path,
    used_mock: bool,
) -> None:
    lines: List[str] = []
    lines.append(f"# HybridRAG eval report (golden v0)")
    lines.append("")
    lines.append(f"- Input: `{input_path}`{' (MOCK DATA - real run pending)' if used_mock else ''}")
    lines.append(f"- Records: **{aggregates['total']}** (success: {aggregates['success']}, errors: {aggregates['errors']})")
    lines.append(f"- top_k: **{top_k}**")
    lines.append(f"- Reranker: **{'on' if settings.USE_RERANKER else 'off'}**")
    lines.append(f"- Generation: **{'on' if gen_enabled else 'off'}**")
    lines.append("")

    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| recall@{top_k} | {_fmt(aggregates[f'recall_at_{top_k}'])} |")
    lines.append(f"| MRR | {_fmt(aggregates['mrr'])} |")
    lines.append(f"| keyword_coverage | {_fmt(aggregates['keyword_coverage'])} |")
    lines.append(f"| retrieval ms p50 | {_fmt(aggregates['retrieval_ms_p50'], '.1f')} |")
    lines.append(f"| retrieval ms p95 | {_fmt(aggregates['retrieval_ms_p95'], '.1f')} |")
    lines.append(f"| generation ms p50 | {_fmt(aggregates['generation_ms_p50'], '.1f')} |")
    lines.append(f"| generation ms p95 | {_fmt(aggregates['generation_ms_p95'], '.1f')} |")
    lines.append(f"| gen runs | {aggregates['gen_runs']} |")
    lines.append("")

    # Per-intent
    lines.append("## Per-intent breakdown")
    lines.append("")
    lines.append(f"| Intent | Count | recall@{top_k} | MRR | keyword_coverage |")
    lines.append("|---|---|---|---|---|")
    for intent, m in sorted(aggregates["by_intent"].items(), key=lambda x: -x[1]["count"]):
        lines.append(
            f"| {intent} | {m['count']} | {_fmt(m['recall_at_k'])} | {_fmt(m['mrr'])} | "
            f"{_fmt(m['keyword_coverage'])} |"
        )
    lines.append("")

    # Per-category
    lines.append("## Per-category breakdown")
    lines.append("")
    lines.append(f"| Category | Count | recall@{top_k} | MRR | keyword_coverage |")
    lines.append("|---|---|---|---|---|")
    for cat, m in sorted(aggregates["by_category"].items(), key=lambda x: -x[1]["count"]):
        lines.append(
            f"| {cat} | {m['count']} | {_fmt(m['recall_at_k'])} | {_fmt(m['mrr'])} | "
            f"{_fmt(m['keyword_coverage'])} |"
        )
    lines.append("")

    # Failures
    lines.append(f"## Failure cases (recall@{top_k}=0 or error) — {len(aggregates['failures'])} records")
    lines.append("")
    if aggregates["failures"]:
        lines.append("| id | intent | category | expected_source | top-3 retrieved keys | error |")
        lines.append("|---|---|---|---|---|---|")
        for f in aggregates["failures"]:
            top3 = " ; ".join(f.get("top3_keys") or []) or "-"
            err = (f.get("error") or "").replace("|", "/")
            lines.append(
                f"| {f['id']} | {f.get('intent')} | {f.get('category')} | "
                f"{f.get('expected_source')} | {top3} | {err or '-'} |"
            )
    else:
        lines.append("_No failures._")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main async entry
# ---------------------------------------------------------------------------
async def run(
    *,
    input_path: Path,
    output_base: Path,
    top_k: int,
    limit: int,
    no_gen: bool,
) -> Dict[str, Any]:
    records, used_mock = load_golden(input_path)
    if limit and limit > 0:
        records = records[:limit]

    # Decide whether generation runs
    has_api_key = bool(settings.OPENAI_API_KEY)
    gen_enabled = (not no_gen) and has_api_key
    if no_gen:
        log.info("Generation skipped (--no-gen).")
    elif not has_api_key:
        log.warning("OPENAI_API_KEY missing - skipping generation phase.")

    # Searcher (load indexes; honors USE_RERANKER from settings)
    log.info("Loading indexes (USE_RERANKER=%s)...", settings.USE_RERANKER)
    searcher = HybridSearcher()
    searcher.load_indexes()

    generator = None
    if gen_enabled:
        # Local import so --no-gen runs don't pull AsyncOpenAI client unnecessarily
        from src.hybridrag.chat.answer import AnswerGenerator  # noqa: WPS433
        generator = AnswerGenerator()

    log.info("Evaluating %d records (top_k=%d, gen=%s)...",
             len(records), top_k, "on" if gen_enabled else "off")

    rows: List[Dict[str, Any]] = []
    for record in tqdm(records, desc="eval", unit="q"):
        row = await eval_one(
            record,
            searcher=searcher,
            generator=generator,
            top_k=top_k,
            run_gen=gen_enabled,
        )
        rows.append(row)

    aggregates = aggregate(rows, top_k=top_k, gen_enabled=gen_enabled)

    # Write outputs
    out_json = output_base.with_suffix(".json")
    out_md = output_base.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "input": str(input_path),
            "used_mock": used_mock,
            "top_k": top_k,
            "limit": limit,
            "use_reranker": settings.USE_RERANKER,
            "generation_enabled": gen_enabled,
            "generate_model": settings.GENERATE_MODEL if gen_enabled else None,
            "rerank_top_k": settings.RERANK_TOP_K,
            "vector_search_k": settings.VECTOR_SEARCH_K,
            "elastic_search_k": settings.ELASTIC_SEARCH_K,
        },
        "aggregates": aggregates,
        "records": rows,
    }
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(
        out_md,
        aggregates=aggregates,
        rows=rows,
        top_k=top_k,
        gen_enabled=gen_enabled,
        input_path=input_path,
        used_mock=used_mock,
    )

    # 5-line summary to stdout
    print("\n=== eval_runner summary ===")
    print(f"records           : {aggregates['total']} (errors: {aggregates['errors']})")
    print(f"recall@{top_k:<11}: {_fmt(aggregates[f'recall_at_{top_k}'])}   "
          f"MRR: {_fmt(aggregates['mrr'])}")
    print(f"retrieval ms p50/p95 : {_fmt(aggregates['retrieval_ms_p50'], '.1f')} / "
          f"{_fmt(aggregates['retrieval_ms_p95'], '.1f')}")
    print(f"generation ms p50/p95: {_fmt(aggregates['generation_ms_p50'], '.1f')} / "
          f"{_fmt(aggregates['generation_ms_p95'], '.1f')}  "
          f"(kw_cov={_fmt(aggregates['keyword_coverage'])})")
    print(f"failures          : {len(aggregates['failures'])} (json={out_json}, md={out_md})")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run golden eval set through HybridRAG.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help=f"Path to golden JSONL (default: {DEFAULT_INPUT})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_BASE,
                   help=f"Output basename (no extension). Will write .json + .md "
                        f"(default: {DEFAULT_OUTPUT_BASE})")
    p.add_argument("--no-gen", action="store_true",
                   help="Skip AnswerGenerator (retrieval metrics only, no OpenAI cost).")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top K retrieved docs to score recall against (default: 5).")
    p.add_argument("--limit", type=int, default=0,
                   help="Evaluate only first N records (0 = all).")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose logging.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    output_base: Path = args.output
    if output_base.suffix in {".json", ".md"}:
        output_base = output_base.with_suffix("")

    try:
        asyncio.run(run(
            input_path=args.input,
            output_base=output_base,
            top_k=args.top_k,
            limit=args.limit,
            no_gen=args.no_gen,
        ))
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
