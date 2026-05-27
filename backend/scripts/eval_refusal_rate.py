"""
Refusal / Grounded / Partial / Handoff rate evaluator.

Sends each curated query to the live HybridRAG chat endpoint
(``POST /api/v1/chat/answer``), auto-classifies the response into
{grounded, partial, refusal, handoff, chitchat, error}, then reports
per-category match rates against the expected labels.

Designed for thesis-style evaluation: stdlib + ``requests`` + ``pyyaml`` only,
synchronous, resilient (per-query try/except), no DB / no async.

Usage
-----

    cd backend
    python scripts/eval_refusal_rate.py \\
        [--api-url http://localhost:8000] \\
        [--queries-file data/eval/refusal_eval_queries.yaml] \\
        [--output data/eval/refusal_results_<ts>.json] \\
        [--token <jwt>] \\
        [--timeout 60]

The script prints a Markdown breakdown to stdout and writes the full JSON
report to ``--output`` (default ``backend/data/eval/refusal_results_<ts>.json``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import yaml


# ---------------------------------------------------------------------- #
# Paths / constants
# ---------------------------------------------------------------------- #
# The script lives in ``backend/scripts/``; the repo's ``backend`` dir is its
# parent. We resolve paths relative to this file so the script works regardless
# of the caller's cwd — but the spec also asks that it work from ``backend/``,
# which is naturally true since defaults resolve there too.
BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES_FILE = BACKEND_DIR / "data" / "eval" / "refusal_eval_queries.yaml"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "eval"

# Substring of ``settings.REFUSAL_MESSAGE`` that the bot ALWAYS emits when it
# cannot answer. Compared case-insensitively against the start of the answer.
REFUSAL_MARKER = "chưa có thông tin"

# Hedge words / phrases that signal a partial / under-confident answer.
PARTIAL_MARKERS = (
    "một phần",
    "chưa có thông tin chi tiết",
    "có thể",
    "chưa rõ",
)

# Markers that the human-handoff fallback was triggered.
HANDOFF_MARKERS = ("Hotline", "Phòng Tuyển sinh")

# Citation pattern — the answer generator embeds ``[1]``, ``[2]``, ... after
# claims it grounded in retrieved chunks. Presence of at least one marker is
# what separates "grounded" from "free-form text".
CITATION_RE = re.compile(r"\[\d+\]")

VALID_CLASSES = {"grounded", "partial", "refusal", "handoff", "chitchat", "error"}


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the HybridRAG chatbot's refusal / grounded / partial / "
            "handoff rate against a curated YAML query set."
        )
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running API (default: %(default)s).",
    )
    parser.add_argument(
        "--queries-file",
        default=str(DEFAULT_QUERIES_FILE),
        help="Path to the YAML query set (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to write the JSON report (default: "
            "data/eval/refusal_results_<timestamp>.json)."
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer JWT for the chat endpoint. If omitted, the script tries "
             "unauthenticated and exits with a clear error on 401.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request HTTP timeout in seconds (default: %(default)s).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------- #
# YAML loading + validation
# ---------------------------------------------------------------------- #
def load_queries(path: Path) -> list[dict[str, Any]]:
    """Load and lightly validate the query set.

    Each entry must have ``id``, ``category``, ``query``, ``expected_class``.
    Bad entries are skipped with a stderr warning rather than aborting the
    whole run — typos in the YAML shouldn't lose hours of LLM calls.
    """
    if not path.exists():
        sys.stderr.write(f"[fatal] queries file not found: {path}\n")
        sys.exit(2)

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, list):
        sys.stderr.write(
            f"[fatal] {path} must be a YAML list of query entries.\n"
        )
        sys.exit(2)

    cleaned: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            sys.stderr.write(f"[warn] entry #{i} is not a mapping, skipping\n")
            continue
        missing = [k for k in ("id", "category", "query", "expected_class") if k not in entry]
        if missing:
            sys.stderr.write(
                f"[warn] entry #{i} missing keys {missing!r}, skipping\n"
            )
            continue
        qid = str(entry["id"])
        if qid in seen_ids:
            sys.stderr.write(f"[warn] duplicate id {qid!r}, skipping\n")
            continue
        seen_ids.add(qid)
        cleaned.append(
            {
                "id": qid,
                "category": str(entry["category"]),
                "query": str(entry["query"]),
                "expected_class": str(entry["expected_class"]),
            }
        )
    return cleaned


# ---------------------------------------------------------------------- #
# HTTP call
# ---------------------------------------------------------------------- #
def call_chat_answer(
    *,
    api_url: str,
    question: str,
    token: str | None,
    timeout: float,
) -> tuple[dict[str, Any] | None, int | None, str | None, int]:
    """Hit POST /api/v1/chat/answer once.

    Returns ``(body, status_code, error_message, latency_ms)``. Either
    ``body`` is set (success path) or ``error_message`` is set.
    """
    url = api_url.rstrip("/") + "/api/v1/chat/answer"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {"question": question, "search_mode": "hybrid"}

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return None, None, f"network_error: {exc}", latency_ms
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if resp.status_code == 401:
        return None, 401, "unauthorized", latency_ms
    if resp.status_code >= 400:
        return None, resp.status_code, f"http_{resp.status_code}: {resp.text[:300]}", latency_ms

    try:
        body = resp.json()
    except ValueError as exc:
        return None, resp.status_code, f"bad_json: {exc}", latency_ms
    return body, resp.status_code, None, latency_ms


# ---------------------------------------------------------------------- #
# Classification
# ---------------------------------------------------------------------- #
def classify_response(body: dict[str, Any]) -> str:
    """Bucket the chat response into one of ``VALID_CLASSES`` (minus error).

    Order of checks matters:
      1. refusal (substring match against the canonical REFUSAL_MESSAGE)
         -> if it also contains handoff markers, escalate to ``handoff``.
      2. chitchat (intent says so, OR no retrieval + short non-refusal text).
      3. partial (long-enough answer that contains hedge words).
      4. grounded (long-enough, retrieved>0, has at least one ``[N]`` citation).
      5. fallback: ``partial`` — the answer is present but doesn't meet
         the strict "grounded" bar (no citation OR no retrieved docs).
    """
    answer = (body.get("answer") or "").strip()
    retrieved_count = int(body.get("retrieved_count") or 0)
    intent = (body.get("intent") or "").strip().lower()

    answer_lower = answer.lower()
    # Check the first sentence for the refusal marker. We use a 200-char
    # window because the bot occasionally prepends a clarifier sentence.
    is_refusal = REFUSAL_MARKER in answer_lower[:200]
    contains_handoff = any(m in answer for m in HANDOFF_MARKERS)

    if is_refusal:
        return "handoff" if contains_handoff else "refusal"

    # Chitchat: explicit intent OR no retrieval + short non-refusal text.
    if intent == "chitchat":
        return "chitchat"
    if retrieved_count == 0 and 0 < len(answer) <= 200:
        return "chitchat"

    if len(answer) <= 30:
        # Too short to be meaningfully grounded; if it isn't a refusal /
        # chitchat we treat it as partial (degenerate output).
        return "partial"

    contains_hedge = any(m in answer_lower for m in PARTIAL_MARKERS)
    if contains_hedge:
        return "partial"

    has_citation = bool(CITATION_RE.search(answer))
    if retrieved_count > 0 and has_citation:
        return "grounded"

    # Answer exists, no hedge, but missing the citation / retrieval evidence
    # we require for "grounded" -> downgrade to "partial".
    return "partial"


def expected_matches_actual(expected: str, actual: str) -> bool:
    """Match policy.

    - "refusal" expected matches both ``refusal`` and ``handoff`` actuals,
      because handoff is a styled subtype of refusal (same upstream signal,
      different presentation).
    - everything else is an exact string compare.
    """
    if expected == "refusal":
        return actual in ("refusal", "handoff")
    return expected == actual


# ---------------------------------------------------------------------- #
# Aggregation + reporting
# ---------------------------------------------------------------------- #
def aggregate(details: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-query rows up into overall + per-category buckets."""
    by_class: dict[str, int] = {c: 0 for c in VALID_CLASSES}
    by_category: dict[str, dict[str, int]] = {}

    matches = 0
    for row in details:
        actual = row["actual"]
        by_class[actual] = by_class.get(actual, 0) + 1
        cat = row["category"]
        bucket = by_category.setdefault(
            cat,
            {"total": 0, "match": 0, **{c: 0 for c in VALID_CLASSES}},
        )
        bucket["total"] += 1
        bucket[actual] = bucket.get(actual, 0) + 1
        if row["match"]:
            matches += 1
            bucket["match"] += 1

    total = len(details)
    match_rate = round(matches / total, 4) if total else 0.0
    return {
        "total": total,
        "matches": matches,
        "match_rate": match_rate,
        "by_class": by_class,
        "by_category": by_category,
    }


def render_markdown_table(summary: dict[str, Any]) -> str:
    """Pretty-print the per-category table to stdout.

    Columns: Category | Total | Grounded | Partial | Refusal | Handoff | Match%
    We pad to fixed widths so the output is easy to eyeball in a terminal.
    """
    headers = ["Category", "Total", "Grounded", "Partial", "Refusal", "Handoff", "Match%"]
    widths = [22, 6, 9, 8, 8, 8, 7]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]

    # Sort categories alphabetically for stable output.
    for cat in sorted(summary["by_category"].keys()):
        b = summary["by_category"][cat]
        total = b["total"]
        rate = (b["match"] / total * 100) if total else 0.0
        lines.append(
            fmt_row(
                [
                    cat,
                    str(total),
                    str(b.get("grounded", 0)),
                    str(b.get("partial", 0)),
                    str(b.get("refusal", 0)),
                    str(b.get("handoff", 0)),
                    f"{rate:.0f}%",
                ]
            )
        )

    total = summary["total"]
    overall_rate = summary["match_rate"] * 100
    bc = summary["by_class"]
    lines.append(fmt_row(["-" * w for w in widths]))
    lines.append(
        fmt_row(
            [
                "OVERALL",
                str(total),
                str(bc.get("grounded", 0)),
                str(bc.get("partial", 0)),
                str(bc.get("refusal", 0)),
                str(bc.get("handoff", 0)),
                f"{overall_rate:.0f}%",
            ]
        )
    )
    # Tack on a small extras line for chitchat / error since they don't fit
    # the main 4-column schema but are still useful signal.
    extras = (
        f"(chitchat={bc.get('chitchat', 0)}, error={bc.get('error', 0)}, "
        f"matches={summary['matches']}/{total})"
    )
    lines.append(extras)
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    queries_path = Path(args.queries_file)
    queries = load_queries(queries_path)
    if not queries:
        sys.stderr.write("[fatal] no valid queries loaded; aborting\n")
        return 2

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.output:
        output_path = Path(args.output)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"refusal_results_{timestamp}.json"

    print(
        f"[info] api_url={args.api_url}  queries={len(queries)}  "
        f"queries_file={queries_path}\n"
    )

    details: list[dict[str, Any]] = []
    auth_warned = False
    for i, q in enumerate(queries, start=1):
        body, status_code, err, latency_ms = call_chat_answer(
            api_url=args.api_url,
            question=q["query"],
            token=args.token,
            timeout=args.timeout,
        )

        # First-request 401 with no token: bail loudly. The user wants a
        # clear error rather than 20 silent failures.
        if status_code == 401 and not args.token and not auth_warned:
            sys.stderr.write(
                "\n[fatal] API returned 401 Unauthorized.\n"
                "        Re-run with --token <jwt> (Bearer access token).\n"
            )
            return 1
        if status_code == 401:
            # Token present but rejected — same fatal exit.
            sys.stderr.write(
                "\n[fatal] API returned 401 Unauthorized even with --token.\n"
                "        Check that the token is valid and not expired.\n"
            )
            return 1

        if err is not None or body is None:
            actual = "error"
            answer_preview = ""
            retrieved_count = 0
            sources: list[str] = []
            intent = None
            intent_score = None
            error_msg = err
        else:
            actual = classify_response(body)
            answer = (body.get("answer") or "").strip()
            answer_preview = answer[:240] + ("..." if len(answer) > 240 else "")
            retrieved_count = int(body.get("retrieved_count") or 0)
            sources = list(body.get("sources") or [])
            intent = body.get("intent")
            intent_score = body.get("intent_score")
            error_msg = None

        match = expected_matches_actual(q["expected_class"], actual)
        row = {
            "id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "expected": q["expected_class"],
            "actual": actual,
            "match": match,
            "latency_ms": latency_ms,
            "retrieved_count": retrieved_count,
            "sources": sources,
            "intent": intent,
            "intent_score": intent_score,
            "answer_preview": answer_preview,
            "error": error_msg,
        }
        details.append(row)

        status_tag = "OK " if match else "MISS"
        if actual == "error":
            status_tag = "ERR "
        print(
            f"  [{i:>2}/{len(queries)}] {status_tag} {q['id']:<14} "
            f"expected={q['expected_class']:<8} actual={actual:<9} "
            f"lat={latency_ms}ms"
        )

    summary = aggregate(details)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "api_url": args.api_url,
        "queries_file": str(queries_path),
        "total": summary["total"],
        "matches": summary["matches"],
        "match_rate": summary["match_rate"],
        "by_class": summary["by_class"],
        "by_category": summary["by_category"],
        "details": details,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print()
    print(render_markdown_table(summary))
    print()
    print(f"[info] wrote {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
