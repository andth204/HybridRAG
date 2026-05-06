"""
Đánh giá module viết lại truy vấn theo ngữ cảnh (QueryReflection)
Dataset: backend/data/rewriter_test_40.json (40 mẫu, 10 mỗi loại)
Metrics:
  - Rewrite rate: tỷ lệ câu hỏi được viết lại (khác câu gốc)
  - Preservation rate: tỷ lệ câu độc lập được giữ nguyên
  - Context capture score: GPT-4o-mini đánh giá chất lượng viết lại
  - Latency: thời gian xử lý
"""
import sys
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybridrag.rewriter.core import QueryReflection
from src.config.settings import settings
from openai import AsyncOpenAI


def load_test_cases() -> List[Dict]:
    data_path = Path(__file__).parent.parent / "data" / "rewriter_test_40.json"
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset["samples"]


async def evaluate_with_judge(client: AsyncOpenAI, original: str, rewritten: str, history: List[Dict], expected: str) -> Tuple[int, str]:
    """Dùng GPT-4o-mini làm judge đánh giá chất lượng viết lại (1-5)."""
    history_str = "\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in history) or "(không có)"
    prompt = f"""Bạn là chuyên gia đánh giá chất lượng viết lại câu hỏi cho hệ thống RAG.

Lịch sử hội thoại:
{history_str}

Câu hỏi gốc: "{original}"
Câu hỏi viết lại: "{rewritten}"
Câu hỏi tham chiếu (mong đợi): "{expected}"

Hãy chấm điểm câu hỏi viết lại từ 1-5 theo tiêu chí:
5 - Xuất sắc: Câu viết lại độc lập, đầy đủ ngữ cảnh, tương đương câu tham chiếu
4 - Tốt: Câu viết lại tốt, có thể thiếu một chi tiết nhỏ
3 - Trung bình: Câu có cải thiện nhưng vẫn còn mơ hồ
2 - Yếu: Câu viết lại không tốt hơn câu gốc đáng kể
1 - Sai: Câu viết lại làm sai lệch ý nghĩa hoặc tệ hơn câu gốc

Chỉ trả về một số nguyên từ 1-5, không giải thích."""

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
            ),
            timeout=10.0
        )
        score_str = response.choices[0].message.content.strip()
        score = int(score_str[0]) if score_str and score_str[0].isdigit() else 3
        return min(5, max(1, score)), score_str
    except Exception as e:
        return 3, f"error: {e}"


async def main():
    TEST_CASES = load_test_cases()
    rewriter = QueryReflection()
    judge_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    results = []
    latencies = []

    print(f"\nĐánh giá module QueryReflection ({len(TEST_CASES)} mẫu kiểm thử)")
    print("=" * 70)

    for tc in TEST_CASES:
        t0 = time.perf_counter()
        rewritten = await rewriter.reflect(tc["current_query"], tc["chat_history"])
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        was_rewritten = rewritten != tc["current_query"]
        is_correct_behavior = True
        if tc["type"] in ("independent", "smalltalk", "no_history"):
            is_correct_behavior = not was_rewritten
        else:
            is_correct_behavior = was_rewritten

        score, score_raw = await evaluate_with_judge(
            judge_client,
            tc["current_query"],
            rewritten,
            tc["chat_history"],
            tc["expected_standalone"]
        )

        results.append({
            "id": tc["id"],
            "type": tc["type"],
            "original": tc["current_query"],
            "rewritten": rewritten,
            "expected": tc["expected_standalone"],
            "was_rewritten": was_rewritten,
            "correct_behavior": is_correct_behavior,
            "judge_score": score,
            "latency_ms": lat_ms,
        })

        status = "✓" if is_correct_behavior else "✗"
        print(f"[{tc['id']}] {status} Type={tc['type']:<18} Score={score}/5  Lat={lat_ms:.0f}ms")
        print(f"       Original : {tc['current_query'][:60]}")
        print(f"       Rewritten: {rewritten[:60]}")
        print()

    # ── Tổng kết ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)

    type_groups = {}
    for r in results:
        t = r["type"]
        type_groups.setdefault(t, []).append(r)

    overall_correct = sum(1 for r in results if r["correct_behavior"])
    avg_score       = sum(r["judge_score"] for r in results) / len(results)
    avg_lat         = sum(latencies) / len(latencies)
    p95_lat         = sorted(latencies)[int(0.95 * len(latencies))]

    print(f"  Tổng mẫu:           {len(results)}")
    print(f"  Correct behavior:   {overall_correct}/{len(results)} ({100*overall_correct/len(results):.1f}%)")
    print(f"  Avg judge score:    {avg_score:.2f}/5")
    print(f"  Avg latency:        {avg_lat:.1f} ms")
    print(f"  P95 latency:        {p95_lat:.1f} ms")

    print(f"\n  Phân tích theo loại:")
    for t, grp in type_groups.items():
        correct = sum(1 for r in grp if r["correct_behavior"])
        avg_s   = sum(r["judge_score"] for r in grp) / len(grp)
        print(f"    {t:<20}: {correct}/{len(grp)} correct  avg_score={avg_s:.2f}")

    # Rewrite rate cho context_dependent
    dep_group = type_groups.get("context_dependent", [])
    if dep_group:
        rewrite_rate = sum(1 for r in dep_group if r["was_rewritten"]) / len(dep_group)
        print(f"\n  Rewrite rate (context_dependent): {rewrite_rate*100:.1f}%")

    # Preservation rate cho independent + smalltalk + no_history
    preserve_types = ["independent", "smalltalk", "no_history"]
    preserve_group = [r for r in results if r["type"] in preserve_types]
    if preserve_group:
        preserve_rate = sum(1 for r in preserve_group if not r["was_rewritten"]) / len(preserve_group)
        print(f"  Preservation rate (non-dependent): {preserve_rate*100:.1f}%")

    # Save
    out_path = Path(__file__).parent.parent / "data" / "eval_rewriter_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": len(results),
                "correct_behavior_rate": overall_correct / len(results),
                "avg_judge_score": avg_score,
                "avg_latency_ms": avg_lat,
                "p95_latency_ms": p95_lat,
            },
            "by_type": {
                t: {
                    "count": len(grp),
                    "correct": sum(1 for r in grp if r["correct_behavior"]),
                    "avg_score": sum(r["judge_score"] for r in grp) / len(grp),
                }
                for t, grp in type_groups.items()
            },
            "samples": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Kết quả lưu → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
