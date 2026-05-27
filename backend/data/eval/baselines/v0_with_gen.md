# HybridRAG eval report (golden v0)

- Input: `D:\my-projects\nlp\HybridRAG\backend\data\eval\golden_v0.jsonl`
- Records: **50** (success: 50, errors: 0)
- top_k: **5**
- Reranker: **off**
- Generation: **on**

## Aggregate metrics

| Metric | Value |
|---|---|
| recall@5 | 0.8800 |
| MRR | 0.7273 |
| keyword_coverage | 0.3590 |
| retrieval ms p50 | 173.0 |
| retrieval ms p95 | 292.1 |
| generation ms p50 | 2103.4 |
| generation ms p95 | 5120.4 |
| gen runs | 50 |

## Per-intent breakdown

| Intent | Count | recall@5 | MRR | keyword_coverage |
|---|---|---|---|---|
| general_qa | 10 | 1.0000 | 0.7917 | 0.3250 |
| score_lookup | 8 | 0.7500 | 0.5625 | 0.6562 |
| program_info | 8 | 0.7500 | 0.6667 | 0.1875 |
| tuition_lookup | 6 | 1.0000 | 0.9167 | 0.0833 |
| admission_method | 6 | 0.8333 | 0.4778 | 0.2667 |
| deadline | 4 | 1.0000 | 1.0000 | 0.5625 |
| compare | 4 | 0.7500 | 0.5625 | 0.4000 |
| chitchat | 4 | 1.0000 | 1.0000 | 0.5000 |

## Per-category breakdown

| Category | Count | recall@5 | MRR | keyword_coverage |
|---|---|---|---|---|
| score | 8 | 0.7500 | 0.5625 | 0.6562 |
| major | 8 | 0.7500 | 0.6667 | 0.1875 |
| contact | 7 | 1.0000 | 0.9048 | 0.6429 |
| tuition | 6 | 1.0000 | 0.9167 | 0.0833 |
| method | 6 | 0.8333 | 0.4778 | 0.2667 |
| deadline | 4 | 1.0000 | 1.0000 | 0.5625 |
| compare | 4 | 0.7500 | 0.5625 | 0.4000 |
| campus | 4 | 1.0000 | 0.6458 | 0.1875 |
| refusal | 3 | 1.0000 | 1.0000 | 0.0000 |

## Failure cases (recall@5=0 or error) — 6 records

| id | intent | category | expected_source | top-3 retrieved keys | error |
|---|---|---|---|---|---|
| q003 | score_lookup | score | Điểm 2023.md | tuyen_sinh_247.md ; qa_fb.md ; Điểm 2025.md | - |
| q007 | score_lookup | score | Điểm 2024.md | tuyen_sinh_247.md ; qa_fb.md ; tuyen_sinh_247.md | - |
| q015 | program_info | major | qa_fb.md | tuyen_sinh_247.md ; tuyen_sinh_247.md ; tuyen_sinh_247.md | - |
| q022 | program_info | major | Thông báo tuyển sinh đại học chính quy năm 2026.txt | qa_fb.md ; qa_fb.md ; qa_fb.md | - |
| q026 | admission_method | method | tuyen_sinh_247.md | qa_fb.md ; qa_fb.md ; qa_fb.md | - |
| q036 | compare | compare | tuyen_sinh_247.md | qa_fb.md ; qa_fb.md ; qa_fb.md | - |
