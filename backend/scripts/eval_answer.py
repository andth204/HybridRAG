"""
Đánh giá module sinh câu trả lời - gọi trực tiếp không qua HTTP API
Dataset: backend/data/data_test.json (50 mẫu)
Metrics:
  - ROUGE-L
  - LLM-as-Judge (GPT-4o-mini): Faithfulness, Relevance, Completeness (1-5)
  - Answer Rate
"""
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybridrag.retrieval.hybrid import HybridSearcher
from src.hybridrag.chat.answer import AnswerGenerator
from src.config.settings import settings
from openai import AsyncOpenAI


def rouge_l(hypothesis: str, reference: str) -> float:
    def lcs_len(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    lcs = lcs_len(hyp, ref)
    p = lcs / len(hyp)
    r = lcs / len(ref)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


async def llm_judge(client: AsyncOpenAI, question: str, reference: str, generated: str) -> Dict[str, int]:
    prompt = (
        f'Cau hoi: "{question}"\n'
        f'Dap an tham chieu: "{reference}"\n'
        f'Cau tra loi sinh ra: "{generated}"\n\n'
        "Cham diem 3 tieu chi, moi tieu chi 1-5:\n"
        "1. Faithfulness: thong tin co chinh xac, khong bia dat?\n"
        "2. Relevance: co dung trong tam cau hoi?\n"
        "3. Completeness: co bao phu du thong tin quan trong?\n\n"
        "Chi tra ve 3 so nguyen cach nhau dau phay, vi du: 4,5,3"
    )
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            ),
            timeout=15.0
        )
        raw = resp.choices[0].message.content.strip()
        nums = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        if len(nums) >= 3:
            return {
                "faithfulness": min(5, max(1, nums[0])),
                "relevance":    min(5, max(1, nums[1])),
                "completeness": min(5, max(1, nums[2])),
            }
    except Exception:
        pass
    return {"faithfulness": 3, "relevance": 3, "completeness": 3}


async def main():
    data_path = Path(__file__).parent.parent / "data" / "data_test.json"
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    samples = dataset["samples"]

    print(f"\nDanh gia module Answer Generation")
    print(f"Dataset: {dataset['dataset_name']} | {len(samples)} mau")
    print("="*70)

    # Load searcher + generator mot lan
    searcher = HybridSearcher()
    searcher.load_indexes()
    generator = AnswerGenerator()
    judge = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    results = []
    for i, s in enumerate(samples):
        print(f"  [{i+1:>2}/{len(samples)}] {s['id']} ({s['difficulty']})...", end=" ", flush=True)

        # Retrieve
        t0 = time.perf_counter()
        docs = await searcher.search(s["question"])
        search_ms = (time.perf_counter() - t0) * 1000

        # Generate
        answer_parts = []
        t1 = time.perf_counter()
        async for chunk in generator.stream_answer(
            query=s["question"],
            retrieved_docs=docs,
            timeout=30.0,
        ):
            answer_parts.append(chunk)
        gen_ms = (time.perf_counter() - t1) * 1000
        generated = "".join(answer_parts).strip()

        rl = rouge_l(generated, s["reference_answer"])
        scores = await llm_judge(judge, s["question"], s["reference_answer"], generated)

        results.append({
            "id":          s["id"],
            "category":    s.get("category", ""),
            "difficulty":  s.get("difficulty", ""),
            "question":    s["question"],
            "reference":   s["reference_answer"],
            "generated":   generated,
            "rouge_l":     rl,
            "judge":       scores,
            "n_docs":      len(docs),
            "search_ms":   search_ms,
            "gen_ms":      gen_ms,
            "answered":    len(generated) > 20,
        })

        avg_judge = sum(scores.values()) / 3
        print(f"ROUGE-L={rl:.3f}  Judge={avg_judge:.2f}/5  Gen={gen_ms:.0f}ms")

    # Tong ket
    print("\n" + "="*70)
    print("TONG KET")
    print("="*70)
    n = len(results)
    answered   = sum(1 for r in results if r["answered"])
    avg_rl     = sum(r["rouge_l"] for r in results) / n
    avg_faith  = sum(r["judge"]["faithfulness"] for r in results) / n
    avg_rel    = sum(r["judge"]["relevance"]    for r in results) / n
    avg_comp   = sum(r["judge"]["completeness"] for r in results) / n
    avg_search = sum(r["search_ms"] for r in results) / n
    avg_gen    = sum(r["gen_ms"]    for r in results) / n
    p95_gen    = sorted(r["gen_ms"] for r in results)[int(0.95 * n)]

    print(f"  Total:          {n}")
    print(f"  Answer rate:    {answered}/{n} ({100*answered/n:.1f}%)")
    print(f"  Avg ROUGE-L:    {avg_rl:.4f}")
    print(f"  Faithfulness:   {avg_faith:.2f}/5")
    print(f"  Relevance:      {avg_rel:.2f}/5")
    print(f"  Completeness:   {avg_comp:.2f}/5")
    print(f"  Avg search:     {avg_search:.0f} ms")
    print(f"  Avg generate:   {avg_gen:.0f} ms")
    print(f"  P95 generate:   {p95_gen:.0f} ms")

    print(f"\n  ROUGE-L + Judge theo do kho:")
    by_diff: Dict[str, List] = {}
    for r in results:
        by_diff.setdefault(r["difficulty"], []).append(r)
    for diff, grp in sorted(by_diff.items()):
        rl   = sum(r["rouge_l"] for r in grp) / len(grp)
        judg = sum(sum(r["judge"].values())/3 for r in grp) / len(grp)
        print(f"    {diff:<10}: ROUGE-L={rl:.4f}  Judge={judg:.2f}/5  (n={len(grp)})")

    # Top 3 tot nhat va kem nhat
    sorted_by_judge = sorted(results, key=lambda r: sum(r["judge"].values()), reverse=True)
    print(f"\n  Top 3 cau tra loi tot nhat (theo Judge):")
    for r in sorted_by_judge[:3]:
        print(f"    [{r['id']}] {r['question'][:55]}...")
        print(f"      ROUGE-L={r['rouge_l']:.3f}  Faith={r['judge']['faithfulness']}  Rel={r['judge']['relevance']}  Comp={r['judge']['completeness']}")

    print(f"\n  3 cau tra loi kem nhat (theo Judge):")
    for r in sorted_by_judge[-3:]:
        print(f"    [{r['id']}] {r['question'][:55]}...")
        print(f"      ROUGE-L={r['rouge_l']:.3f}  Faith={r['judge']['faithfulness']}  Rel={r['judge']['relevance']}  Comp={r['judge']['completeness']}")
        print(f"      Generated: {r['generated'][:120]}...")

    out_path = Path(__file__).parent.parent / "data" / "eval_answer_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": n,
                "answer_rate": answered / n,
                "avg_rouge_l": avg_rl,
                "avg_faithfulness": avg_faith,
                "avg_relevance":    avg_rel,
                "avg_completeness": avg_comp,
                "avg_search_ms":    avg_search,
                "avg_gen_ms":       avg_gen,
                "p95_gen_ms":       p95_gen,
            },
            "by_difficulty": {
                diff: {
                    "count":    len(grp),
                    "avg_rouge_l": sum(r["rouge_l"] for r in grp) / len(grp),
                    "avg_judge":   sum(sum(r["judge"].values())/3 for r in grp)/len(grp),
                }
                for diff, grp in by_diff.items()
            },
            "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Ket qua luu: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
