"""Visual smoke test for the Phase 4C clarifier.

Runs six hardcoded queries that should exercise every ``reason``
branch of :class:`hybridrag.chat.clarifier.Clarifier`, plus one
"no clarification" baseline. Prints the result as JSON-ish output
so the on-call dev can eyeball what would have been sent back to
the frontend.

Run from the backend root::

    python scripts/check_clarifier.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# Force UTF-8 stdout so Vietnamese display strings render on Windows consoles.
for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

# Ensure ``src.*`` is importable when executing the script directly.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from src.hybridrag.chat import clarifier as clarifier_mod  # noqa: E402
from src.hybridrag.chat.clarifier import Clarifier  # noqa: E402


# ----------------------------------------------------------------- #
# Build a deterministic ``resolve_all`` shim so the smoke output
# does not depend on the live entity dictionary or fuzzy scores.
# ----------------------------------------------------------------- #
SCENARIO_MAP: dict[str, dict[str, list[dict]]] = {
    "ambiguous_major": {
        "major": [
            {"canonical": "ky_thuat_phan_mem",        "display": "Kỹ thuật phần mềm",                "_score": 88},
            {"canonical": "cong_nghe_ky_thuat_co_khi","display": "Công nghệ kỹ thuật cơ khí",        "_score": 86},
            {"canonical": "cong_nghe_ky_thuat_dien_dien_tu","display": "Công nghệ kỹ thuật điện, điện tử", "_score": 84},
        ],
    },
    "missing_year": {
        "major": [
            {"canonical": "cong_nghe_thong_tin", "display": "Công nghệ thông tin", "_score": 95},
        ],
    },
    "missing_major": {},
    "ambiguous_campus": {
        "major": [],
        "campus": [
            {"canonical": "co_so_1", "display": "Cơ sở 1", "_score": 92},
            {"canonical": "co_so_2", "display": "Cơ sở 2", "_score": 89},
        ],
    },
    "low_recall": {},
    "no_clarify": {
        "major": [
            {"canonical": "cong_nghe_thong_tin", "display": "Công nghệ thông tin", "_score": 95},
        ],
    },
}


def _build_resolver(scenario: str):
    mapping = SCENARIO_MAP.get(scenario, {})

    def fake_resolve_all(text: str, *, min_score: int = 80, max_ngram: int = 4, entity_types=None):
        if entity_types is None:
            return dict(mapping)
        return {k: v for k, v in mapping.items() if k in entity_types}

    return fake_resolve_all


# ----------------------------------------------------------------- #
# Sample queries — one per ``reason`` plus a "no clarify" baseline.
# ----------------------------------------------------------------- #
SAMPLES: list[dict] = [
    {
        "label":   "ambiguous_major",
        "query":   "Học phí ngành kỹ thuật bao nhiêu?",
        "intent":  "tuition_lookup",
        "slots":   {"year": 2024},
        "docs":    None,
    },
    {
        "label":   "missing_year",
        "query":   "Điểm chuẩn CNTT là bao nhiêu?",
        "intent":  "score_lookup",
        "slots":   {},
        "docs":    None,
    },
    {
        "label":   "missing_major",
        "query":   "Điểm chuẩn năm 2024 là bao nhiêu?",
        "intent":  "score_lookup",
        "slots":   {},
        "docs":    None,
    },
    {
        "label":   "ambiguous_campus",
        "query":   "Cơ sở nào dạy CNTT?",
        "intent":  "program_info",
        "slots":   {},
        "docs":    None,
    },
    {
        "label":   "low_recall",
        "query":   "Trường có gì hay không?",
        "intent":  "general_qa",
        "slots":   {"major": "cong_nghe_thong_tin", "year": 2024},
        "docs":    [{"rerank_score": 0.12, "content": "Một đoạn không liên quan lắm."}],
    },
    {
        "label":   "no_clarify",
        "query":   "Học phí CNTT là bao nhiêu?",
        "intent":  "tuition_lookup",
        "slots":   {"year": 2024},
        "docs":    None,
    },
]


def _to_jsonable(req) -> object:
    if req is None:
        return None
    d = dataclasses.asdict(req)
    return d


def main() -> int:
    print(f"[clarifier-smoke] year-now={datetime.now(timezone.utc).year}")
    print(f"[clarifier-smoke] running {len(SAMPLES)} scenarios\n")

    clf = Clarifier()
    for s in SAMPLES:
        scenario = s["label"]
        clarifier_mod.resolve_all = _build_resolver(scenario)  # type: ignore[assignment]

        req = clf.check(
            query=s["query"],
            intent=s["intent"],
            session_slots=s["slots"],
            retrieval_docs=s["docs"],
        )

        print(f"=== {scenario} ===")
        print(f"  query   : {s['query']!r}")
        print(f"  intent  : {s['intent']}")
        print(f"  slots   : {s['slots']}")
        if s["docs"]:
            print(f"  docs    : {s['docs']}")
        print("  result  :")
        print("  " + json.dumps(_to_jsonable(req), ensure_ascii=False, indent=2).replace("\n", "\n  "))
        print()

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
