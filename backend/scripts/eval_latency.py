"""Latency profiling harness for the HybridRAG chat pipeline.

Sends a representative set of queries to ``POST /api/v1/chat/answer`` and
records wall-clock latency around the call. After each answer the script
fetches the freshly persisted assistant message via
``GET /api/v1/chat/sessions/{id}/messages`` so it can pull the stage
timings (``rewrite_ms``, ``route_ms``, ``search_ms``, ``generate_ms``)
that the API stores on ``ChatMessage.metadata`` -- the public
``ChatAnswerResponse`` schema does not expose them directly.

The first ``--n-warmup`` runs are dropped from the aggregated percentile
report to discount cold-start / cache warm-up effects. The full per-run
trace (including the warm-up runs) is still written to the output JSON
for transparency.

Usage::

    cd backend
    python scripts/eval_latency.py \\
        --api-url http://localhost:8000 \\
        --token <jwt> \\
        --queries-file data/eval/latency_eval_queries.yaml \\
        --n-warmup 2 \\
        --n-runs 20 \\
        --output data/eval/latency_results_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_QUERIES_FILE = Path("data/eval/latency_eval_queries.yaml")
DEFAULT_N_WARMUP = 2
DEFAULT_N_RUNS = 20
ANSWER_ENDPOINT = "/api/v1/chat/answer"
MESSAGES_ENDPOINT_TMPL = "/api/v1/chat/sessions/{session_id}/messages"
REQUEST_TIMEOUT_S = 120.0

STAGE_KEYS = ("total_ms", "rewrite_ms", "route_ms", "search_ms", "generate_ms")


# --------------------------------------------------------------------------- #
# Stats helpers (pure stdlib -- no numpy)
# --------------------------------------------------------------------------- #
def percentile(data: list[float], p: float) -> float:
    """Linear-interpolation percentile (matches numpy default)."""
    if not data:
        return float("nan")
    s = sorted(data)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    return {
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "p99": round(percentile(values, 99), 2),
        "mean": round(statistics.fmean(values), 2),
        "std": round(statistics.pstdev(values) if len(values) > 1 else 0.0, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "n": len(values),
    }


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_queries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a YAML list of query entries")
    items: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}[{idx}]: expected mapping, got {type(entry)}")
        if "query" not in entry:
            raise ValueError(f"{path}[{idx}]: missing 'query' field")
        items.append(
            {
                "id": str(entry.get("id", f"q{idx + 1}")),
                "category": str(entry.get("category", "uncategorised")),
                "query": str(entry["query"]),
            }
        )
    if not items:
        raise ValueError(f"{path}: no queries defined")
    return items


def build_run_plan(
    queries: list[dict[str, Any]], n_runs: int
) -> list[dict[str, Any]]:
    """Cycle through queries until we have exactly ``n_runs`` entries."""
    plan: list[dict[str, Any]] = []
    if n_runs <= 0:
        return plan
    for i in range(n_runs):
        q = queries[i % len(queries)]
        plan.append(
            {
                "run_index": i,
                "query_id": q["id"],
                "category": q["category"],
                "query": q["query"],
            }
        )
    return plan


# --------------------------------------------------------------------------- #
# API calls
# --------------------------------------------------------------------------- #
def _auth_headers(token: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def call_answer(
    api_url: str, token: str | None, question: str, session_id: str | None
) -> tuple[float, dict[str, Any]]:
    """POST /chat/answer and return (wall_ms, parsed_json)."""
    payload: dict[str, Any] = {"question": question, "search_mode": "hybrid"}
    if session_id:
        payload["session_id"] = session_id
    url = api_url.rstrip("/") + ANSWER_ENDPOINT
    t0 = time.perf_counter()
    resp = requests.post(
        url,
        json=payload,
        headers=_auth_headers(token),
        timeout=REQUEST_TIMEOUT_S,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    return wall_ms, resp.json()


def fetch_last_assistant_metadata(
    api_url: str, token: str | None, session_id: str
) -> dict[str, Any] | None:
    """Pull the most recent assistant message and return its ``metadata``.

    The /answer response model does NOT expose ``metadata``, but the
    persisted ChatMessage does. We fetch the session history, take the
    last role=='assistant' item, and return ``metadata`` (a dict).
    """
    url = api_url.rstrip("/") + MESSAGES_ENDPOINT_TMPL.format(session_id=session_id)
    try:
        resp = requests.get(url, headers=_auth_headers(token), timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    items = (resp.json() or {}).get("items", [])
    for msg in reversed(items):
        if msg.get("role") == "assistant":
            md = msg.get("metadata")
            return md if isinstance(md, dict) else None
    return None


# --------------------------------------------------------------------------- #
# Per-query execution
# --------------------------------------------------------------------------- #
def execute_run(
    api_url: str,
    token: str | None,
    run: dict[str, Any],
    session_id: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run_index": run["run_index"],
        "query_id": run["query_id"],
        "category": run["category"],
        "query": run["query"],
    }
    try:
        wall_ms, body = call_answer(api_url, token, run["query"], session_id)
    except requests.HTTPError as exc:  # 4xx / 5xx
        status = exc.response.status_code if exc.response is not None else "?"
        record["error"] = f"HTTPError {status}"
        return record
    except requests.RequestException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    except (ValueError, json.JSONDecodeError) as exc:
        record["error"] = f"DecodeError: {exc}"
        return record

    record["total_ms"] = round(wall_ms, 2)
    record["session_id"] = body.get("session_id")
    record["retrieved_count"] = body.get("retrieved_count")
    record["route_name"] = body.get("route_name")
    record["intent"] = body.get("intent")

    sid = body.get("session_id")
    metadata: dict[str, Any] | None = None
    if sid:
        metadata = fetch_last_assistant_metadata(api_url, token, sid)

    if metadata:
        record["metadata_source"] = "session_messages"
        for key in ("rewrite_ms", "route_ms", "search_ms", "generate_ms"):
            if key in metadata:
                try:
                    record[key] = float(metadata[key])
                except (TypeError, ValueError):
                    pass
    else:
        record["metadata_source"] = "wallclock_only"

    return record


# --------------------------------------------------------------------------- #
# Aggregation + rendering
# --------------------------------------------------------------------------- #
def aggregate(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stages: dict[str, dict[str, float]] = {}
    for key in STAGE_KEYS:
        values = [r[key] for r in records if isinstance(r.get(key), (int, float))]
        stages[key] = summarise([float(v) for v in values])
    return stages


def render_table(stages: dict[str, dict[str, float]]) -> str:
    header = f"{'Stage':<14}{'n':>5}{'p50':>10}{'p95':>10}{'p99':>10}{'mean':>10}{'std':>8}{'min':>10}{'max':>10}"
    lines = [header, "-" * len(header)]
    for key in STAGE_KEYS:
        s = stages.get(key, {})

        def _fmt(name: str) -> str:
            v = s.get(name)
            if v is None or (isinstance(v, float) and v != v):  # NaN check
                return "-"
            return f"{v:.1f}" if isinstance(v, float) else str(v)

        lines.append(
            f"{key:<14}{_fmt('n'):>5}{_fmt('p50'):>10}{_fmt('p95'):>10}"
            f"{_fmt('p99'):>10}{_fmt('mean'):>10}{_fmt('std'):>8}"
            f"{_fmt('min'):>10}{_fmt('max'):>10}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latency profiler for the HybridRAG chat pipeline."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--token", default=None, help="JWT for the Authorization header."
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=DEFAULT_QUERIES_FILE,
        help="YAML file with query entries.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=DEFAULT_N_WARMUP,
        help="Discard the first N runs from the percentile aggregate.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help="Total number of runs (cycled through the query set).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the JSON report. Default: data/eval/latency_results_<ts>.json",
    )
    parser.add_argument(
        "--session-mode",
        choices=("fresh", "sticky"),
        default="fresh",
        help="'fresh' = each call creates a new session; 'sticky' reuses the first session.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.n_warmup < 0:
        print("ERROR: --n-warmup must be >= 0", file=sys.stderr)
        return 2
    if args.n_runs <= 0:
        print("ERROR: --n-runs must be > 0", file=sys.stderr)
        return 2
    if args.n_warmup >= args.n_runs:
        print(
            f"ERROR: --n-warmup ({args.n_warmup}) must be < --n-runs ({args.n_runs})",
            file=sys.stderr,
        )
        return 2

    if not args.queries_file.exists():
        print(f"ERROR: queries file not found: {args.queries_file}", file=sys.stderr)
        return 2

    queries = load_queries(args.queries_file)
    plan = build_run_plan(queries, args.n_runs)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or Path(f"data/eval/latency_results_{ts}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[eval_latency] api={args.api_url} queries={len(queries)} "
        f"runs={args.n_runs} warmup={args.n_warmup} session_mode={args.session_mode}",
        file=sys.stderr,
    )

    records: list[dict[str, Any]] = []
    sticky_session: str | None = None
    for run in plan:
        sid_for_call = sticky_session if args.session_mode == "sticky" else None
        record = execute_run(args.api_url, args.token, run, sid_for_call)
        records.append(record)
        if args.session_mode == "sticky" and sticky_session is None:
            sticky_session = record.get("session_id")

        if "error" in record:
            tag = f"ERR {record['error']}"
        else:
            tag = (
                f"total={record.get('total_ms', '?')}ms"
                f" rewrite={record.get('rewrite_ms', '-')}"
                f" route={record.get('route_ms', '-')}"
                f" search={record.get('search_ms', '-')}"
                f" generate={record.get('generate_ms', '-')}"
            )
        print(
            f"  [{run['run_index'] + 1:>3}/{args.n_runs}] "
            f"{run['query_id']:<10} {run['category']:<10} {tag}",
            file=sys.stderr,
        )

    # Drop warm-up runs and errored runs from percentile aggregation.
    measured = [r for r in records[args.n_warmup:] if "error" not in r]
    errored = sum(1 for r in records if "error" in r)

    stages = aggregate(measured)
    report = {
        "timestamp": ts,
        "api_url": args.api_url,
        "queries_file": str(args.queries_file),
        "total_runs": len(records),
        "skipped_warmup": min(args.n_warmup, len(records)),
        "errored": errored,
        "measured_runs": len(measured),
        "stages": stages,
        "per_query": records,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("", file=sys.stderr)
    print(render_table(stages))
    print("")
    print(f"[eval_latency] wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
