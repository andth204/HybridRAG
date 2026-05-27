"""Evaluate the keyword-first intent classifier against the golden set.

Reads ``backend/data/eval/golden_v0.jsonl``, runs every query through
:class:`KeywordIntentClassifier`, and prints:

* total / per-intent accuracy,
* the confusion matrix (true × predicted),
* a sample of misclassifications for the worst intents,
* the per-intent keyword count (provenance — how much vocabulary each
  intent is backed by in ``intent_keywords.yaml``).

Exit code is always 0 — this is a diagnostic, not a CI gate. No
network calls are made; the classifier is pure-local.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 on stdout so Windows ``cp1252`` console doesn't choke on
# Vietnamese diacritics or fancy punctuation. No-op on POSIX.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from src.hybridrag.router.intent_classifier import KeywordIntentClassifier
from src.hybridrag.router.intents import Intent


# ---------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------- #
def _golden_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "eval" / "golden_v0.jsonl"


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _pad(s: str, width: int) -> str:
    """Truncate or right-pad a string to ``width`` for table layout."""
    if len(s) > width:
        return s[: width - 1] + "…"
    return s.ljust(width)


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #
def main() -> int:
    golden = _golden_path()
    if not golden.exists():
        print(f"golden file not found: {golden}")
        return 0

    records = _load_records(golden)
    if not records:
        print("golden file empty")
        return 0

    clf = KeywordIntentClassifier()

    # Provenance: how many keywords does each intent have?
    print("Per-intent keyword counts from intent_keywords.yaml:")
    for label, count in clf.intent_keyword_counts.items():
        print(f"  {_pad(label, 18)} {count:>3}")
    print()

    labels = [i.value for i in Intent]
    confusion: dict[str, Counter] = {lab: Counter() for lab in labels}
    per_intent: dict[str, dict[str, int]] = {
        lab: {"correct": 0, "total": 0} for lab in labels
    }
    errors: list[dict[str, str]] = []

    correct = 0
    for rec in records:
        truth = rec.get("intent")
        query = rec.get("query") or ""
        if truth not in per_intent:
            # Skip labels we don't know about — keeps the matrix sane.
            continue
        pred = clf.classify(query).intent.value
        confusion[truth][pred] += 1
        per_intent[truth]["total"] += 1
        if pred == truth:
            per_intent[truth]["correct"] += 1
            correct += 1
        else:
            errors.append(
                {
                    "id": str(rec.get("id", "")),
                    "truth": truth,
                    "pred": pred,
                    "query": query,
                }
            )

    total = sum(p["total"] for p in per_intent.values())
    overall_acc = (correct / total) if total else 0.0

    # ------------------------------------------------------------------ #
    # Per-intent accuracy
    # ------------------------------------------------------------------ #
    print("Per-intent accuracy:")
    print(f"  {_pad('intent', 18)} {'n':>3} {'correct':>8} {'acc':>7}")
    print("  " + "-" * 40)
    for lab in labels:
        n = per_intent[lab]["total"]
        c = per_intent[lab]["correct"]
        acc = (c / n) if n else 0.0
        print(
            f"  {_pad(lab, 18)} {n:>3} {c:>8} {acc * 100:>6.1f}%"
        )
    print("  " + "-" * 40)
    print(f"  {_pad('OVERALL', 18)} {total:>3} {correct:>8} {overall_acc * 100:>6.1f}%")
    print()

    # ------------------------------------------------------------------ #
    # Confusion matrix
    # ------------------------------------------------------------------ #
    print("Confusion matrix (rows = truth, cols = pred):")
    col_w = 8
    header = "  " + _pad("", 18) + "".join(_pad(lab[:7], col_w) for lab in labels)
    print(header)
    for true_lab in labels:
        row = "  " + _pad(true_lab, 18)
        for pred_lab in labels:
            count = confusion[true_lab][pred_lab]
            row += _pad(str(count) if count else "·", col_w)
        print(row)
    print()

    # ------------------------------------------------------------------ #
    # Worst-performing intents + sample errors
    # ------------------------------------------------------------------ #
    if errors:
        # Group errors per truth label
        by_truth: dict[str, list[dict[str, str]]] = defaultdict(list)
        for e in errors:
            by_truth[e["truth"]].append(e)

        print(f"Misclassifications ({len(errors)} of {total}):")
        for lab in labels:
            errs = by_truth.get(lab, [])
            if not errs:
                continue
            print(f"  [{lab}] {len(errs)} errors:")
            for e in errs[:5]:
                qprev = e["query"]
                if len(qprev) > 80:
                    qprev = qprev[:77] + "..."
                print(f"    {e['id']:>4} → pred={e['pred']:<18} | {qprev}")
            if len(errs) > 5:
                print(f"    ... and {len(errs) - 5} more")
    print()
    print(f"Target: keyword classifier accuracy >= 75% — current: {overall_acc * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
