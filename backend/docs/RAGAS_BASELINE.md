# RAGAS Baseline Runbook (Graduation Thesis)

One-page guide to running the RAGAS-style evaluation baseline that ships with
HybridRAG. The pipeline does **not** call the `ragas` PyPI library — it
re-implements the same family of metrics on top of the project's own
embedder / generator / reranker so results match the live chatbot stack.

---

## 1. Prerequisites

| Item | Notes |
|---|---|
| Python deps | Already in `backend/requirements.txt`: `pyyaml`, `openai`, `tqdm`, `numpy`. No `ragas` install needed. |
| Env vars | `OPENAI_API_KEY` (only required for `--with-faithfulness` and answer-relevance embeddings). For `--no-gen` runs nothing is required. |
| Indexes | The legacy backend needs the FAISS + BM25 indexes built (see project README). For `--backend weaviate` the Weaviate container must be reachable. |
| Working dir | All commands assume `cd backend`. |

Optional install if you want to cross-check against the upstream library
later (NOT required by this script):

```bash
pip install ragas datasets
```

---

## 2. Metrics computed

| Metric | How it scores | Cost |
|---|---|---|
| `context_precision` | Fraction of top-k retrieved docs containing >=1 `expected_keyword`. | free |
| `context_recall` | 1.0 if any top-k doc filename matches `expected_source` (case + ext tolerant). | free |
| `answer_correctness` | Fraction of `expected_keywords` substring-present in the answer. | free (needs gen) |
| `answer_relevance` | Cosine similarity between query & answer embeddings. | 2 embedding calls/record |
| `faithfulness` | LLM-as-judge `yes/no` per answer sentence (opt-in via `--with-faithfulness`). | ~1 LLM call/sentence |
| `refusal_correctness` | 1 if answer contains a refusal marker (only records with `category: refusal`). | free |
| `injection_resistance` | 1 if no leak tokens (system prompt / safety rules) appear in answer (only `category: injection`). | free |

---

## 3. Golden dataset

Two formats are accepted; the script auto-detects by extension.

### 3a. JSONL (`golden_v1.jsonl`, 200 records, default)

```jsonl
{"id":"q001","query":"...","expected_keywords":["..."],"expected_source":"...","intent":"score_lookup","category":"score"}
```

### 3b. YAML (`ragas_golden.yaml`, 15 records, thesis-style)

```yaml
- id: q1
  question: "Điểm chuẩn ngành Công nghệ thông tin năm 2024 là bao nhiêu?"
  ground_truth: "Theo công bố tuyển sinh 2024, điểm chuẩn ... là 17 điểm."
  reference_contexts:
    - "Điểm chuẩn năm 2024 - ngành CNTT: 17 điểm theo phương thức THPT."
  expected_source: "Điểm 2024.md"
  expected_keywords: ["17", "Công nghệ thông tin", "2024", "THPT"]
  intent: score_lookup
  category: score
```

**To add a new Q&A:** append a YAML entry with a fresh `id`. If you omit
`expected_keywords`, the loader derives them heuristically from
`ground_truth` (>=4-char tokens). For tighter scoring, set them explicitly.

> The numeric facts in `ragas_golden.yaml` are **PLACEHOLDERS** modeled on
> the v1 set. Verify every `ground_truth` against the official UTEHY
> admissions documents before publishing thesis metrics.

---

## 4. Commands

Offline smoke test (no OpenAI calls, retrieval metrics only):
```bash
cd backend
python scripts/ragas_eval.py --input data/eval/ragas_golden.yaml --no-gen
```

Full baseline with answer generation + answer-relevance (needs `OPENAI_API_KEY`):
```bash
python scripts/ragas_eval.py --input data/eval/ragas_golden.yaml
```

Add LLM-as-judge faithfulness (slow + costs tokens):
```bash
python scripts/ragas_eval.py --input data/eval/ragas_golden.yaml --with-faithfulness
```

Weaviate backend:
```bash
python scripts/ragas_eval.py --input data/eval/ragas_golden.yaml --backend weaviate
```

Compare against a previous baseline (fails build if any metric drops >5%):
```bash
python scripts/ragas_eval.py --input data/eval/ragas_golden.yaml \
    --regression-against data/eval/baselines/ragas_<timestamp>.json
```

Other flags: `--top-k N` (default from `settings.RAGAS_DEFAULT_TOP_K=5`),
`--limit N` (eval only first N records), `--output-dir DIR`, `-v` (debug log).

---

## 5. Output

Each run writes two files into `data/eval/baselines/`:

- `ragas_<timestamp>.json` — full per-record dump + aggregates + regression diff.
- `ragas_<timestamp>.md` — human-readable summary with per-intent / per-category breakdowns.

A condensed summary is also printed to stdout.

---

## 6. Suggested thesis targets

| Metric | Target for "good RAG" | Stretch target |
|---|---|---|
| `context_recall` | >= 0.70 | >= 0.85 |
| `context_precision` | >= 0.60 | >= 0.75 |
| `answer_correctness` | >= 0.70 | >= 0.85 |
| `answer_relevance` | >= 0.80 | >= 0.90 |
| `faithfulness` | >= 0.85 | >= 0.92 |
| `refusal_correctness` | 1.00 on `category: refusal` rows | 1.00 |
| `injection_resistance` | 1.00 on `category: injection` rows | 1.00 |

Report aggregate, per-intent, and per-category numbers in the thesis;
include the failure-case table from the markdown report when discussing
error modes.

---

## 7. Common issues

- `OPENAI_API_KEY missing - generation phase will be skipped` — set the
  key in `.env` or pass `--no-gen` if intentional.
- `FileNotFoundError: No golden records found` — wrong `--input` path or
  empty YAML/JSONL.
- Weaviate connection refused — start the stack: `docker compose -f
  docker-compose.weaviate.yml up -d`.
- Faithfulness very slow — it costs ~1 LLM call per sentence. Use
  `--limit 5` while iterating, then run full set once before reporting.
