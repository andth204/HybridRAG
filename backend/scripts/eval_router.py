"""
Đánh giá module định tuyến câu hỏi (Router)
Dataset: backend/data/router_test_30.json
Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
"""
import sys
import json
import time
import asyncio
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybridrag.router.keywords import KeywordRouter
from src.hybridrag.router.samples import ROUTES

try:
    from src.hybridrag.router.semantic import SemanticRouter
    from src.config.settings import settings
    SEMANTIC_AVAILABLE = True
except ImportError as _e:
    SEMANTIC_AVAILABLE = False
    print(f"[INFO] SemanticRouter unavailable: {_e}")


def compute_metrics(y_true, y_pred, labels):
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    metrics = {}
    for label in labels:
        tp = cm[label][label]
        fp = sum(cm[t][label] for t in labels if t != label)
        fn = sum(cm[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1,
                          "tp": tp, "fp": fp, "fn": fn}
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    return accuracy, metrics, cm


def print_confusion_matrix(cm, labels):
    print("\nConfusion Matrix:")
    header = f"{'':>20}" + "".join(f"  Pred:{l:<12}" for l in labels)
    print(header)
    for true_label in labels:
        row = f"  True:{true_label:<14}" + "".join(f"  {cm[true_label][pred_label]:<16}" for pred_label in labels)
        print(row)


def eval_keyword_router(samples, labels):
    router = KeywordRouter(routes=ROUTES)
    y_true, y_pred, scores = [], [], []
    latencies = []

    for s in samples:
        t0 = time.perf_counter()
        score, route = router.guide(s["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        y_true.append(s["expected_route"])
        y_pred.append(route)
        scores.append(score)

    accuracy, metrics, cm = compute_metrics(y_true, y_pred, labels)
    return accuracy, metrics, cm, latencies, y_true, y_pred


async def eval_semantic_router(samples, labels):
    router = SemanticRouter(routes=ROUTES, embeddings_dir=settings.ROUTER_EMBEDDINGS_DIR)
    y_true, y_pred, scores = [], [], []
    latencies = []

    for s in samples:
        t0 = time.perf_counter()
        score, route = await router.guide(s["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        y_true.append(s["expected_route"])
        y_pred.append(route)
        scores.append(score)

    accuracy, metrics, cm = compute_metrics(y_true, y_pred, labels)
    return accuracy, metrics, cm, latencies, y_true, y_pred


def print_report(router_name, accuracy, metrics, cm, latencies, labels):
    print(f"\n{'='*60}")
    print(f"  {router_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy*100:.2f}%  ({int(accuracy*len(latencies))}/{len(latencies)} correct)")
    print(f"  Avg latency: {sum(latencies)/len(latencies):.2f} ms")
    print(f"  P95 latency: {sorted(latencies)[int(0.95*len(latencies))]:.2f} ms")

    print(f"\n  {'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*52}")
    for label in labels:
        m = metrics[label]
        print(f"  {label:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    macro_p = sum(metrics[l]["precision"] for l in labels) / len(labels)
    macro_r = sum(metrics[l]["recall"]    for l in labels) / len(labels)
    macro_f = sum(metrics[l]["f1"]        for l in labels) / len(labels)
    print(f"  {'Macro avg':<20} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f:>10.4f}")

    print_confusion_matrix(cm, labels)


def print_error_analysis(y_true, y_pred, samples):
    errors = [(i, s) for i, (t, p, s) in enumerate(zip(y_true, y_pred, samples)) if t != p]
    if errors:
        print(f"\nError analysis ({len(errors)} misclassified):")
        for _, s in errors:
            print(f"  [{s['id']}] Q: {s['question'][:60]}...")
            print(f"         Expected: {s['expected_route']:12}  Group: {s.get('group', '-')}")


def group_accuracy(y_true, y_pred, samples):
    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for t, p, s in zip(y_true, y_pred, samples):
        g = s.get("group", "unknown")
        group_stats[g]["total"] += 1
        if t == p:
            group_stats[g]["correct"] += 1
    return group_stats


def difficulty_accuracy(y_true, y_pred, samples):
    diff_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for t, p, s in zip(y_true, y_pred, samples):
        d = s.get("muc_do", "unknown")
        diff_stats[d]["total"] += 1
        if t == p:
            diff_stats[d]["correct"] += 1
    return diff_stats


async def main():
    data_path = Path(__file__).parent.parent / "data" / "router_test_30.json"
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset["samples"]
    labels = dataset["route_labels"]

    print(f"\nDataset: {dataset['dataset_name']}")
    print(f"Samples: {dataset['sample_count']}  |  Distribution: {dataset['distribution']}")

    # ── Keyword Router ─────────────────────────────────────────────
    acc_kw, metrics_kw, cm_kw, lat_kw, y_true_kw, y_pred_kw = eval_keyword_router(samples, labels)
    print_report("KeywordRouter", acc_kw, metrics_kw, cm_kw, lat_kw, labels)
    print_error_analysis(y_true_kw, y_pred_kw, samples)

    diff_kw = difficulty_accuracy(y_true_kw, y_pred_kw, samples)
    print("\n  Accuracy by difficulty (KeywordRouter):")
    for d, s in sorted(diff_kw.items()):
        print(f"    {d:<15}: {s['correct']}/{s['total']}  ({100*s['correct']/s['total']:.1f}%)")

    # ── Semantic Router ────────────────────────────────────────────
    print("\nLoading SemanticRouter (requires embeddings cache)...")
    try:
        if not SEMANTIC_AVAILABLE:
            raise ImportError("FAISS not available in this Python environment")
        acc_sm, metrics_sm, cm_sm, lat_sm, y_true_sm, y_pred_sm = await eval_semantic_router(samples, labels)
        print_report("SemanticRouter", acc_sm, metrics_sm, cm_sm, lat_sm, labels)
        print_error_analysis(y_true_sm, y_pred_sm, samples)

        diff_sm = difficulty_accuracy(y_true_sm, y_pred_sm, samples)
        print("\n  Accuracy by difficulty (SemanticRouter):")
        for d, s in sorted(diff_sm.items()):
            print(f"    {d:<15}: {s['correct']}/{s['total']}  ({100*s['correct']/s['total']:.1f}%)")

        # ── So sánh tổng hợp ─────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  So sánh tổng hợp")
        print(f"{'='*60}")
        print(f"  {'Router':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Avg Lat(ms)':>12}")
        print(f"  {'-'*54}")
        macro_f_kw = sum(metrics_kw[l]["f1"] for l in labels) / len(labels)
        macro_f_sm = sum(metrics_sm[l]["f1"] for l in labels) / len(labels)
        print(f"  {'KeywordRouter':<20} {acc_kw*100:>9.2f}% {macro_f_kw:>10.4f} {sum(lat_kw)/len(lat_kw):>12.2f}")
        print(f"  {'SemanticRouter':<20} {acc_sm*100:>9.2f}% {macro_f_sm:>10.4f} {sum(lat_sm)/len(lat_sm):>12.2f}")

        results = {
            "keyword_router": {
                "accuracy": acc_kw,
                "macro_f1": macro_f_kw,
                "metrics": metrics_kw,
                "avg_latency_ms": sum(lat_kw)/len(lat_kw),
            },
            "semantic_router": {
                "accuracy": acc_sm,
                "macro_f1": macro_f_sm,
                "metrics": metrics_sm,
                "avg_latency_ms": sum(lat_sm)/len(lat_sm),
            }
        }

    except Exception as e:
        print(f"  SemanticRouter skipped: {e}")
        results = {
            "keyword_router": {
                "accuracy": acc_kw,
                "macro_f1": sum(metrics_kw[l]["f1"] for l in labels) / len(labels),
                "metrics": metrics_kw,
                "avg_latency_ms": sum(lat_kw)/len(lat_kw),
            }
        }

    # Save JSON results
    out_path = Path(__file__).parent.parent / "data" / "eval_router_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
