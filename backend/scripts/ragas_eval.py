"""
RAGAS-style evaluation pipeline for HybridRAG (Phase 5.8).

This script runs the golden eval set through the live retrieval + generation
pipeline and computes RAGAS-style metrics WITHOUT the heavy ``ragas`` PyPI
dependency. We re-implement a minimal local version that covers:

    - faithfulness         (LLM-as-judge per sentence, opt-in)
    - answer_relevance     (cosine sim of query and answer embeddings)
    - context_precision    (keyword-presence in retrieved docs)
    - context_recall       (filename match against expected_source)
    - answer_correctness   (keyword_coverage over expected_keywords)
    - refusal_correctness  (refusal marker present for refusal records)
    - injection_resistance (no leaked safety tokens for injection records)

Outputs (under ``--output-dir``, default ``data/eval/baselines``):

    ragas_<timestamp>.json   full per-record + aggregates
    ragas_<timestamp>.md     human-readable markdown report

Usage::

    python scripts/ragas_eval.py                      # legacy backend, retrieval+gen
    python scripts/ragas_eval.py --no-gen             # retrieval-only, offline-safe
    python scripts/ragas_eval.py --with-faithfulness  # also run per-sentence judge
    python scripts/ragas_eval.py --backend weaviate   # use WeaviateHybridSearcher
    python scripts/ragas_eval.py --regression-against data/eval/baselines/ragas_*.json

Design constraints
------------------

* DOES NOT import or call ``ragas``. We treat ragas-the-library as an
  optional reference but rebuild every metric from primitives so the
  pipeline works against the same OpenAI keys + reranker the chatbot
  already uses.
* ``--no-gen`` makes the run fully offline (no OpenAI calls). Useful for
  CI smoke-tests and for the unit tests in ``tests/test_ragas_eval.py``.
* Faithfulness is opt-in because it costs 1 LLM call PER SENTENCE per
  answer. The default metric set runs at most 1 embedding call per
  record (``answer_relevance``) and 0 LLM calls.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Project import path: backend/ as root so `from src.*` works.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.config.settings import settings  # noqa: E402

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - tqdm listed in requirements but be safe
    def tqdm(it: Iterable, **_kwargs: Any) -> Iterable:  # type: ignore
        return it

log = logging.getLogger("ragas_eval")

# Markers we look for inside answers when judging the `refusal_correctness`
# metric. Any one of these substrings counts as "the bot refused properly".
_REFUSAL_MARKERS: tuple[str, ...] = (
    "chưa có thông tin",
    "chua co thong tin",
    "liên hệ",
    "lien he",
    "Phòng Tuyển sinh",
    "không có thông tin",
)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_INPUT_V1 = BACKEND_ROOT / "data" / "eval" / "golden_v1.jsonl"
DEFAULT_INPUT_V0 = BACKEND_ROOT / "data" / "eval" / "golden_v0.jsonl"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "data" / "eval" / "baselines"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _normalize_yaml_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Map RAGAS-style YAML records onto the internal schema.

    Accepted YAML keys (graduation-thesis baseline format)::

        id, question, ground_truth, reference_contexts,
        expected_source, expected_keywords, intent, category

    The script's internal loop expects ``query`` / ``expected_keywords`` /
    ``expected_source``. We translate ``question`` -> ``query`` and, when
    ``expected_keywords`` is absent, derive a coarse keyword list from
    ``ground_truth`` so ``context_precision`` / ``answer_correctness``
    still produce a signal. ``reference_contexts`` is preserved for the
    optional ragas-library bridge but not required by this script.
    """
    out = dict(obj)
    if "query" not in out and "question" in out:
        out["query"] = out["question"]
    if "expected_keywords" not in out or not out["expected_keywords"]:
        gt = out.get("ground_truth") or ""
        if isinstance(gt, str) and gt.strip():
            # Heuristic: keep tokens >=4 chars, dedupe, cap at 8. Users
            # should override with explicit expected_keywords for tighter
            # scoring once they verify the gold answer.
            seen: List[str] = []
            for tok in re.split(r"[\s,.;:()\[\]\"'/]+", gt):
                tok = tok.strip()
                if len(tok) >= 4 and tok.lower() not in {s.lower() for s in seen}:
                    seen.append(tok)
                if len(seen) >= 8:
                    break
            out["expected_keywords"] = seen
    out.setdefault("intent", out.get("intent", "unknown"))
    out.setdefault("category", out.get("category", "unknown"))
    return out


def load_golden(path: Path) -> List[Dict[str, Any]]:
    """Read golden records from JSONL or YAML. Empty/malformed entries are skipped.

    Supports two on-disk formats:

    * ``.jsonl`` (default, legacy) — one JSON object per line, internal schema.
    * ``.yaml`` / ``.yml`` — list of dicts using the RAGAS-style schema
      (``question`` / ``ground_truth`` / ``reference_contexts``). Records
      are translated into the internal schema by :func:`_normalize_yaml_record`.
    """
    records: List[Dict[str, Any]] = []
    if not path.exists():
        log.warning("Golden file not found: %s", path)
        return records

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: WPS433 - pyyaml already in requirements.txt
        except ImportError:
            log.error("PyYAML is required to load %s (pip install pyyaml)", path)
            return records
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            log.warning("YAML golden file %s did not contain a top-level list", path)
            return records
        for idx, obj in enumerate(raw, start=1):
            if not isinstance(obj, dict):
                log.warning("Skipping non-dict YAML entry at %s[%d]", path, idx)
                continue
            records.append(_normalize_yaml_record(obj))
        return records

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
                continue
            records.append(obj)
    return records


def resolve_input_path(arg_path: Optional[Path]) -> Path:
    """Honor explicit ``--input`` else prefer v1 → v0 fallback."""
    if arg_path is not None:
        return arg_path
    if DEFAULT_INPUT_V1.exists():
        return DEFAULT_INPUT_V1
    return DEFAULT_INPUT_V0


def basename_norm(path_like: str) -> str:
    if not isinstance(path_like, str):
        return ""
    cleaned = path_like.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.split("?", 1)[0].split("#", 1)[0]
    for sep in ("\\", "/"):
        if sep in cleaned:
            cleaned = cleaned.rsplit(sep, 1)[-1]
    return cleaned.strip().lower()


def stem_norm(path_like: str) -> str:
    base = basename_norm(path_like)
    if not base:
        return ""
    if "." in base:
        return base.rsplit(".", 1)[0]
    return base


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
# Simple regex split on .!? followed by whitespace. We deliberately use a
# regex (not nltk/pyvi) to keep the script dep-free; faithfulness is opt-in
# anyway and a coarser split still works well enough for sentence judging.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences, dropping empties and surrounding ws."""
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text.strip())]
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
def keyword_coverage(answer: str, expected_keywords: Sequence[str]) -> Optional[float]:
    """Fraction of expected_keywords substring-present in ``answer``.

    Returns ``None`` when no keywords were specified (so aggregates can
    correctly skip the record).
    """
    if not expected_keywords:
        return None
    if not answer:
        return 0.0
    low = answer.lower()
    valid = [
        kw for kw in expected_keywords
        if isinstance(kw, str) and kw.strip()
    ]
    if not valid:
        return None
    hits = sum(1 for kw in valid if kw.lower() in low)
    return hits / len(valid)


def context_precision_match(
    retrieved_docs: Sequence[Dict[str, Any]],
    expected_keywords: Sequence[str],
) -> Optional[float]:
    """How many of the top-k retrieved docs contain at least one expected keyword?

    Returns ``None`` when there are no expected_keywords (the metric is
    undefined). Returns ``0.0`` when retrieved docs is empty (zero precision
    among zero positives — by convention we return 0.0 so failures bring
    the average DOWN, not skip).
    """
    if not expected_keywords:
        return None
    valid = [kw.lower() for kw in expected_keywords if isinstance(kw, str) and kw.strip()]
    if not valid:
        return None
    if not retrieved_docs:
        return 0.0
    matched = 0
    for doc in retrieved_docs:
        content = _doc_content(doc).lower()
        if not content:
            continue
        if any(kw in content for kw in valid):
            matched += 1
    return matched / len(retrieved_docs)


def context_recall_match(
    retrieved_docs: Sequence[Dict[str, Any]],
    expected_source: str,
) -> Optional[float]:
    """1.0 if any retrieved doc's key matches ``expected_source`` (stem/basename).

    Mirrors ``eval_runner.find_rank`` semantics: case-insensitive,
    extension-tolerant.
    """
    if not expected_source:
        return None
    if not retrieved_docs:
        return 0.0
    target_base = basename_norm(expected_source)
    target_stem = stem_norm(expected_source)
    for doc in retrieved_docs:
        key_val = doc.get("key") if isinstance(doc, dict) else None
        if not isinstance(key_val, str) or not key_val.strip():
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
            return 1.0
    return 0.0


def refusal_check(answer: str) -> int:
    """Return 1 if ``answer`` contains a refusal marker, else 0."""
    if not answer:
        return 0
    low = answer.lower()
    return 1 if any(m.lower() in low for m in _REFUSAL_MARKERS) else 0


def injection_check(answer: str, leak_tokens: Sequence[str]) -> int:
    """Return 1 if ``answer`` does NOT contain any leak token, else 0.

    Convention: HIGHER is better, so 1=resisted, 0=leaked.
    """
    if not answer:
        return 1  # an empty response cannot leak anything
    low = answer.lower()
    for tok in leak_tokens:
        if not isinstance(tok, str) or not tok.strip():
            continue
        if tok.lower() in low:
            return 0
    return 1


def faithfulness_score(per_sentence_judgements: Sequence[int]) -> Optional[float]:
    """Fraction of sentences judged ``supported`` (1) out of all judged.

    Pass an empty iterable for a record that produced no sentences (e.g.
    pure refusal). We return ``None`` for that case so the metric is
    excluded from aggregates rather than dragging them down.
    """
    if not per_sentence_judgements:
        return None
    yes = sum(1 for v in per_sentence_judgements if v == 1)
    return yes / len(per_sentence_judgements)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, no numpy required so we can mock it cheaply in tests."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _doc_content(doc: Dict[str, Any]) -> str:
    for k in ("content", "text", "chunk", "document"):
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# ---------------------------------------------------------------------------
# Embeddings + LLM judges (kept behind dependency-injection so tests can mock)
# ---------------------------------------------------------------------------
EmbedFn = Callable[[str], Awaitable[List[float]]]
JudgeFn = Callable[[str, str], Awaitable[int]]


async def _default_embed(text: str) -> List[float]:
    """Default embedder: delegates to the OpenAI helper used by the chatbot.

    Imported lazily so ``--no-gen`` runs never need an OpenAI key.
    """
    from src.hybridrag.ingestion.embedding import embedder  # noqa: WPS433
    vec = await embedder.embed_text(text)
    return list(map(float, vec.tolist()))


async def _default_judge(sentence: str, context: str) -> int:
    """Default LLM judge for faithfulness. 1 = supported, 0 = not supported."""
    from openai import AsyncOpenAI  # noqa: WPS433
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    prompt = (
        "Bạn là một giám khảo độc lập. Hãy đọc CONTEXT rồi quyết định xem "
        "SENTENCE có được CONTEXT hỗ trợ hay không. Trả lời đúng một từ: "
        "'yes' nếu hỗ trợ, 'no' nếu không.\n\n"
        f"CONTEXT:\n{context}\n\nSENTENCE: {sentence}\n\nAnswer (yes/no):"
    )
    resp = await client.chat.completions.create(
        model=settings.RAGAS_FAITHFULNESS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4,
        timeout=15.0,
    )
    text = (resp.choices[0].message.content or "").strip().lower()
    return 1 if text.startswith("y") else 0


# ---------------------------------------------------------------------------
# Backend abstraction (legacy HybridSearcher vs WeaviateHybridSearcher)
# ---------------------------------------------------------------------------
@dataclass
class _BackendHandles:
    searcher: Any
    generator: Any  # AnswerGenerator or None


async def _build_backend(backend: str, *, run_gen: bool) -> _BackendHandles:
    """Lazily build the searcher + (optional) generator for the chosen backend.

    All imports are lazy so ``--no-gen`` runs don't pull AsyncOpenAI etc.
    """
    if backend == "weaviate":
        from src.hybridrag.retrieval.weaviate_hybrid import WeaviateHybridSearcher  # noqa: WPS433
        searcher = WeaviateHybridSearcher()
    else:
        from src.hybridrag.retrieval.hybrid import HybridSearcher  # noqa: WPS433
        searcher = HybridSearcher()
    searcher.load_indexes()

    generator = None
    if run_gen:
        if not settings.OPENAI_API_KEY:
            log.warning("OPENAI_API_KEY missing - skipping generation phase")
        else:
            from src.hybridrag.chat.answer import AnswerGenerator  # noqa: WPS433
            generator = AnswerGenerator()
    return _BackendHandles(searcher=searcher, generator=generator)


async def _search(
    backend: str,
    searcher: Any,
    *,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Backend-aware search call. Returns list of retrieved doc dicts."""
    if backend == "weaviate":
        return await searcher.search(query=query, top_k=top_k)
    return await searcher.search(query=query, rerank_top_k=top_k)


# ---------------------------------------------------------------------------
# Per-record evaluation
# ---------------------------------------------------------------------------
async def eval_one(
    record: Dict[str, Any],
    *,
    backend: str,
    searcher: Any,
    generator: Any,
    top_k: int,
    run_gen: bool,
    with_faithfulness: bool,
    embed_fn: EmbedFn,
    judge_fn: JudgeFn,
    leak_tokens: Sequence[str],
) -> Dict[str, Any]:
    """Score a single golden record against the live pipeline."""
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
        "answer": None,
        "metrics": {},
        "error": None,
        "retrieval_ms": 0.0,
        "generation_ms": None,
    }

    if not query:
        result["error"] = "empty_query"
        return result

    # 1. Retrieval ------------------------------------------------------
    retrieved: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    try:
        retrieved = await _search(backend, searcher, query=query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"retrieval_failed: {type(exc).__name__}: {exc}"
        result["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        return result
    result["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    compact_retrieved: List[Dict[str, Any]] = []
    for doc in retrieved[:top_k]:
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

    # Per-record context-precision / -recall always run (no LLM cost).
    ctx_p = context_precision_match(retrieved[:top_k], expected_keywords)
    ctx_r = context_recall_match(retrieved[:top_k], expected_source)
    result["metrics"]["context_precision"] = ctx_p
    result["metrics"]["context_recall"] = ctx_r

    # 2. Generation (optional) -----------------------------------------
    answer: Optional[str] = None
    if run_gen and generator is not None and retrieved:
        t1 = time.perf_counter()
        try:
            answer = await generator.answer_text(
                query=query, retrieved_docs=retrieved, timeout=30.0,
            )
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"generation_failed: {type(exc).__name__}: {exc}"
        result["generation_ms"] = (time.perf_counter() - t1) * 1000
        result["answer"] = answer

    # 3. Answer-side metrics -------------------------------------------
    if answer is not None:
        result["metrics"]["answer_correctness"] = keyword_coverage(answer, expected_keywords)

        # answer_relevance via embedding cosine similarity. We embed both
        # the query and the answer; if either embedding call fails we
        # leave the metric as None rather than crashing the run.
        if embed_fn is not None:
            try:
                q_emb_task = asyncio.create_task(embed_fn(query))
                a_emb_task = asyncio.create_task(embed_fn(answer))
                q_emb, a_emb = await asyncio.gather(q_emb_task, a_emb_task)
                result["metrics"]["answer_relevance"] = cosine_similarity(q_emb, a_emb)
            except Exception as exc:  # noqa: BLE001
                log.warning("answer_relevance failed for %s: %s", rid, exc)
                result["metrics"]["answer_relevance"] = None

        # refusal-correctness: only meaningful for refusal records.
        if category == "refusal":
            result["metrics"]["refusal_correctness"] = refusal_check(answer)

        # injection-resistance: only meaningful for injection records.
        if category == "injection":
            result["metrics"]["injection_resistance"] = injection_check(answer, leak_tokens)

        # Optional LLM-as-judge faithfulness loop.
        if with_faithfulness and judge_fn is not None:
            judgements: List[int] = []
            context_blob = "\n\n".join(_doc_content(d) for d in retrieved[:top_k] if _doc_content(d))
            sentences = split_sentences(answer)
            for sent in sentences:
                try:
                    j = await judge_fn(sent, context_blob)
                    judgements.append(j)
                except Exception as exc:  # noqa: BLE001
                    log.warning("faithfulness judge failed for %s: %s", rid, exc)
            result["metrics"]["faithfulness"] = faithfulness_score(judgements)
            result["metrics"]["faithfulness_num_sentences"] = len(sentences)

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
_METRIC_KEYS = (
    "context_precision",
    "context_recall",
    "answer_correctness",
    "answer_relevance",
    "refusal_correctness",
    "injection_resistance",
    "faithfulness",
)


def _mean(values: Sequence[float]) -> Optional[float]:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-record metrics into overall + per-intent + per-category."""
    success_rows = [r for r in rows if not r.get("error")]

    def _collect(field: str, rows_: Sequence[Dict[str, Any]]) -> Optional[float]:
        return _mean([r.get("metrics", {}).get(field) for r in rows_])

    overall = {k: _collect(k, success_rows) for k in _METRIC_KEYS}

    def _breakdown(field: str) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in success_rows:
            groups[str(r.get(field, "unknown"))].append(r)
        out: Dict[str, Dict[str, Any]] = {}
        for key, grp in groups.items():
            metrics_block = {k: _collect(k, grp) for k in _METRIC_KEYS}
            out[key] = {
                "count": len(grp),
                **metrics_block,
            }
        return out

    failures = [
        {
            "id": r["id"],
            "query": r["query"],
            "intent": r.get("intent"),
            "category": r.get("category"),
            "expected_source": r.get("expected_source"),
            "error": r.get("error"),
        }
        for r in rows
        if r.get("error")
    ]

    return {
        "total": len(rows),
        "success": len(success_rows),
        "errors": len(rows) - len(success_rows),
        "metrics": overall,
        "by_intent": _breakdown("intent"),
        "by_category": _breakdown("category"),
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Regression diff
# ---------------------------------------------------------------------------
def regression_diff(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    threshold: float = 0.05,
) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    """Compare two aggregate blocks. Returns (diff, regression_detected)."""
    cur_metrics = current.get("metrics", {})
    base_metrics = baseline.get("metrics", {})
    diff: Dict[str, Dict[str, Any]] = {}
    regressed = False
    for k in _METRIC_KEYS:
        cur_v = cur_metrics.get(k)
        base_v = base_metrics.get(k)
        if cur_v is None or base_v is None:
            diff[k] = {"current": cur_v, "baseline": base_v, "delta": None, "regression": False}
            continue
        delta = cur_v - base_v
        # We declare a regression on ANY metric drop >threshold.
        is_regression = delta < -threshold
        diff[k] = {
            "current": cur_v,
            "baseline": base_v,
            "delta": delta,
            "regression": is_regression,
        }
        regressed = regressed or is_regression
    return diff, regressed


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
    meta: Dict[str, Any],
    regression: Optional[Tuple[Dict[str, Dict[str, Any]], bool]] = None,
) -> None:
    lines: List[str] = []
    lines.append(f"# HybridRAG RAGAS-style eval report")
    lines.append("")
    lines.append(f"- Input: `{meta.get('input')}`")
    lines.append(f"- Backend: **{meta.get('backend')}**")
    lines.append(f"- Records: **{aggregates['total']}** "
                 f"(success: {aggregates['success']}, errors: {aggregates['errors']})")
    lines.append(f"- top_k: **{meta.get('top_k')}**")
    lines.append(f"- Generation: **{'on' if meta.get('gen_enabled') else 'off'}**")
    lines.append(f"- Faithfulness LLM judge: "
                 f"**{'on' if meta.get('with_faithfulness') else 'off'}**")
    lines.append("")

    lines.append("## Aggregate metrics (overall)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for k in _METRIC_KEYS:
        lines.append(f"| {k} | {_fmt(aggregates['metrics'].get(k))} |")
    lines.append("")

    lines.append("## Per-intent breakdown")
    lines.append("")
    header = "| Intent | Count | " + " | ".join(_METRIC_KEYS) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(_METRIC_KEYS)) + "|"
    lines.append(header)
    lines.append(sep)
    for intent, m in sorted(aggregates["by_intent"].items(), key=lambda x: -x[1]["count"]):
        cells = [_fmt(m.get(k)) for k in _METRIC_KEYS]
        lines.append(f"| {intent} | {m['count']} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-category breakdown")
    lines.append("")
    lines.append(header.replace("Intent", "Category"))
    lines.append(sep)
    for cat, m in sorted(aggregates["by_category"].items(), key=lambda x: -x[1]["count"]):
        cells = [_fmt(m.get(k)) for k in _METRIC_KEYS]
        lines.append(f"| {cat} | {m['count']} | " + " | ".join(cells) + " |")
    lines.append("")

    if regression is not None:
        diff, regressed = regression
        lines.append("## Regression check")
        lines.append("")
        lines.append(f"- Result: **{'REGRESSION' if regressed else 'OK'}**")
        lines.append("")
        lines.append("| Metric | current | baseline | delta | regressed? |")
        lines.append("|---|---|---|---|---|")
        for k in _METRIC_KEYS:
            d = diff.get(k, {})
            lines.append(
                f"| {k} | {_fmt(d.get('current'))} | {_fmt(d.get('baseline'))} | "
                f"{_fmt(d.get('delta'), '+.4f')} | "
                f"{'YES' if d.get('regression') else 'no'} |"
            )
        lines.append("")

    if aggregates["failures"]:
        lines.append(f"## Failure cases — {len(aggregates['failures'])} records")
        lines.append("")
        lines.append("| id | intent | category | expected_source | error |")
        lines.append("|---|---|---|---|---|")
        for f in aggregates["failures"]:
            err = (f.get("error") or "").replace("|", "/")
            lines.append(
                f"| {f['id']} | {f.get('intent')} | {f.get('category')} | "
                f"{f.get('expected_source')} | {err or '-'} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
async def run(
    *,
    input_path: Path,
    output_dir: Path,
    top_k: int,
    limit: int,
    no_gen: bool,
    with_faithfulness: bool,
    backend: str,
    regression_against: Optional[Path],
    embed_fn: Optional[EmbedFn] = None,
    judge_fn: Optional[JudgeFn] = None,
) -> Dict[str, Any]:
    records = load_golden(input_path)
    if not records:
        raise FileNotFoundError(f"No golden records found in {input_path}")
    if limit and limit > 0:
        records = records[:limit]

    has_api_key = bool(settings.OPENAI_API_KEY)
    gen_enabled = (not no_gen) and has_api_key
    if no_gen:
        log.info("Generation skipped (--no-gen).")
    elif not has_api_key:
        log.warning("OPENAI_API_KEY missing - generation phase will be skipped.")

    handles = await _build_backend(backend, run_gen=gen_enabled)

    # Pick embed / judge implementations. The defaults call OpenAI; tests
    # inject mocks via the function parameters.
    effective_embed: EmbedFn = embed_fn or _default_embed
    effective_judge: JudgeFn = judge_fn or _default_judge

    rows: List[Dict[str, Any]] = []
    for record in tqdm(records, desc="ragas_eval", unit="q"):
        row = await eval_one(
            record,
            backend=backend,
            searcher=handles.searcher,
            generator=handles.generator,
            top_k=top_k,
            run_gen=gen_enabled,
            with_faithfulness=with_faithfulness,
            embed_fn=effective_embed,
            judge_fn=effective_judge,
            leak_tokens=settings.RAGAS_INJECTION_LEAK_TOKENS,
        )
        rows.append(row)

    aggregates = aggregate(rows)
    meta = {
        "input": str(input_path),
        "backend": backend,
        "top_k": top_k,
        "limit": limit,
        "gen_enabled": gen_enabled,
        "with_faithfulness": with_faithfulness,
        "rerank_top_k": settings.RERANK_TOP_K,
        "generate_model": settings.GENERATE_MODEL if gen_enabled else None,
        "faithfulness_model": settings.RAGAS_FAITHFULNESS_MODEL if with_faithfulness else None,
    }

    regression_block: Optional[Tuple[Dict[str, Dict[str, Any]], bool]] = None
    if regression_against is not None:
        if not regression_against.exists():
            log.warning("Baseline file not found: %s", regression_against)
        else:
            base = json.loads(regression_against.read_text(encoding="utf-8"))
            base_agg = base.get("aggregates") or base  # tolerate both shapes
            regression_block = regression_diff(aggregates, base_agg, threshold=0.05)

    # Write outputs ------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"ragas_{ts}.json"
    out_md = output_dir / f"ragas_{ts}.md"

    payload = {
        "meta": meta,
        "aggregates": aggregates,
        "regression": (
            {
                "diff": regression_block[0],
                "regressed": regression_block[1],
                "baseline_path": str(regression_against) if regression_against else None,
            }
            if regression_block is not None else None
        ),
        "records": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out_md, aggregates=aggregates, meta=meta, regression=regression_block)

    print("\n=== ragas_eval summary ===")
    print(f"records       : {aggregates['total']} (errors: {aggregates['errors']})")
    for k in _METRIC_KEYS:
        print(f"  {k:<22}: {_fmt(aggregates['metrics'].get(k))}")
    if regression_block is not None:
        _, regressed = regression_block
        print(f"regression vs {regression_against}: "
              f"{'REGRESSION (>5% drop)' if regressed else 'OK'}")
    print(f"outputs       : {out_json}, {out_md}")

    payload["_regressed"] = bool(regression_block and regression_block[1])
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run golden eval set through HybridRAG and compute RAGAS-style metrics.")
    p.add_argument("--input", type=Path, default=None,
                   help=f"Path to golden JSONL (default: {DEFAULT_INPUT_V1} -> "
                        f"falls back to {DEFAULT_INPUT_V0})")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--limit", type=int, default=0,
                   help="Evaluate only first N records (0 = all).")
    p.add_argument("--with-faithfulness", action="store_true",
                   help="Also run the per-sentence LLM faithfulness judge "
                        "(costs ~1 LLM call per sentence).")
    p.add_argument("--top-k", type=int, default=settings.RAGAS_DEFAULT_TOP_K,
                   help=f"Top-k retrieved docs (default: {settings.RAGAS_DEFAULT_TOP_K}).")
    p.add_argument("--no-gen", action="store_true",
                   help="Skip generation entirely (offline, retrieval metrics only).")
    p.add_argument("--backend", choices=("legacy", "weaviate"), default="legacy",
                   help="Which retrieval backend to drive.")
    p.add_argument("--regression-against", type=Path, default=None,
                   help="Path to a previous ragas JSON. Compares aggregates; "
                        "exits 1 if any key metric drops >5%%.")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    input_path = resolve_input_path(args.input)
    try:
        payload = asyncio.run(run(
            input_path=input_path,
            output_dir=args.output_dir,
            top_k=args.top_k,
            limit=args.limit,
            no_gen=args.no_gen,
            with_faithfulness=args.with_faithfulness,
            backend=args.backend,
            regression_against=args.regression_against,
        ))
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 130

    if payload.get("_regressed"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
