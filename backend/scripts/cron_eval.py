#!/usr/bin/env python
"""Phase 6.7 — Weekly cron eval runner.

Runs `ragas_eval.py`, compares to baseline, alerts on regression, rotates
baseline on pass, archives history.

Usage:
    python scripts/cron_eval.py [--golden data/eval/golden_v1.jsonl]
                                [--baseline data/eval/baselines/latest.json]
                                [--regression-threshold 0.05]
                                [--slack-webhook URL]
                                [--with-gen]
                                [--dry-run]
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

log = logging.getLogger("cron_eval")

KEY_METRICS = (
    "context_precision",
    "context_recall",
    "answer_correctness",
    "refusal_correctness",
    "injection_resistance",
)


def run_eval(golden_path: str, out_dir: pathlib.Path, with_gen: bool) -> pathlib.Path:
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_json = out_dir / f"ragas_{timestamp}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/ragas_eval.py",
        "--input", golden_path,
        "--output-dir", str(out_dir),
    ]
    if not with_gen:
        cmd.append("--no-gen")
    log.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("ragas_eval.py failed: rc=%d stderr=%s", proc.returncode, proc.stderr[-2000:])
        raise RuntimeError(f"ragas_eval failed rc={proc.returncode}")
    # Find newest ragas_*.json in out_dir
    candidates = sorted(out_dir.glob("ragas_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("ragas_eval produced no json output")
    return candidates[0]


def diff_against_baseline(
    current: dict,
    baseline: dict | None,
    threshold: float,
) -> list[str]:
    if not baseline:
        return []
    cur_agg = current.get("aggregates", {})
    base_agg = baseline.get("aggregates", {})
    alerts: list[str] = []
    for metric in KEY_METRICS:
        cur = cur_agg.get(metric)
        base = base_agg.get(metric)
        if cur is None or base is None:
            continue
        if base <= 0.01:
            continue
        drop = (base - cur) / base
        if drop > threshold:
            alerts.append(
                f"[ALERT] {metric} dropped {base:.3f} → {cur:.3f} ({drop*100:+.1f}%)"
            )
    return alerts


def post_slack(webhook: str, text: str) -> None:
    if not webhook:
        return
    try:
        req = Request(
            webhook,
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(req, timeout=5)
    except URLError as exc:
        log.warning("Slack post failed: %s", exc)


def rotate_baseline(
    current_path: pathlib.Path,
    latest_path: pathlib.Path,
    history_dir: pathlib.Path,
) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%d")
    archive = history_dir / f"ragas_{stamp}.json"
    shutil.copy2(current_path, archive)
    shutil.copy2(current_path, latest_path)
    log.info("Baseline rotated: %s + history %s", latest_path, archive)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default="data/eval/golden_v1.jsonl")
    p.add_argument("--baseline", default="data/eval/baselines/latest.json")
    p.add_argument("--regression-threshold", type=float, default=0.05)
    p.add_argument("--slack-webhook", default=os.environ.get("SLACK_ALERT_WEBHOOK", ""))
    p.add_argument("--with-gen", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = pathlib.Path("data/eval/baselines")
    history_dir = out_dir / "history"
    baseline_path = pathlib.Path(args.baseline)
    baseline: dict | None = None
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Baseline parse failed (%s); ignoring", exc)
    else:
        log.info("No baseline at %s — first run will create it", baseline_path)

    if args.dry_run:
        log.info("[dry-run] Would run eval against %s", args.golden)
        return 0

    try:
        current_path = run_eval(args.golden, out_dir, with_gen=args.with_gen)
    except Exception as exc:
        log.error("Eval run failed: %s", exc)
        post_slack(args.slack_webhook, f"HybridRAG cron eval CRASHED: {exc}")
        return 2

    try:
        current = json.loads(current_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Current eval output parse failed: %s", exc)
        return 2

    alerts = diff_against_baseline(current, baseline, args.regression_threshold)
    if alerts:
        msg = "HybridRAG cron eval REGRESSION:\n" + "\n".join(alerts)
        log.warning(msg)
        post_slack(args.slack_webhook, msg)
        return 1

    rotate_baseline(current_path, baseline_path, history_dir)
    log.info("Eval pass; baseline updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
