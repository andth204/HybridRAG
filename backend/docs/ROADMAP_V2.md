# HybridRAG → Admissions Advisor — Full Roadmap V2

Status: Plan (not started)
Owner: dthan
Target: Production-grade admissions advisory chatbot for UTEHY
Estimated effort: 10-12 weeks solo full-time

---

## Context

Current system: HybridRAG (BM25 + FAISS + RRF + Jina reranker) with FastAPI backend, OpenAI embeddings, GPT-4o-mini generation. Storage is local pickle files. Known issues:

- Coreference fails across multi-turn (bug: ask "campus 1" then "what majors there" → bot answers about campus 3).
- Flat chunking, no parent-child hierarchy.
- No metadata filter (campus, year, faculty).
- Local BM25/FAISS = no multi-process safety, full-file pickle write on every update.
- Latency p95 first-token ~1.5-3.5s, full answer ~2-6s.
- No eval framework, no feedback loop, no monitoring.
- No tool calling, no slot filling, no clarification.

This roadmap fixes all of the above in 6 phases.

---

## Architecture target

```
User Q
  │
  ├─► SSE status: rewriting
  ├─► load history (16 msg, async pool)
  ├─► slot_filler (extract campus/major/year, carry-over)
  ├─► QueryReflection rewrite (gpt-4o-mini, parallel with retrieval speculative)
  ├─► entity resolver (synonym → canonical)
  ├─► intent classifier (8 intents, keyword first, semantic fallback)
  │
  ├─► branch by intent:
  │     ├─ chitchat → stream LLM
  │     ├─ score_lookup / tuition_lookup → TOOL CALL (Postgres structured)
  │     ├─ compare → tool + RAG hybrid
  │     └─ general_qa / program_info / ... → RAG
  │
  ├─► RAG path:
  │     ├─ Weaviate hybrid(query, alpha=0.6, filters={campus,year,...}) → child chunks
  │     ├─ Jina reranker (top 20 → top 5 children)
  │     ├─ expand child → parent (dedup)
  │     └─ AnswerGenerator stream (citation-enforced prompt)
  │
  ├─► post-process verifier (number check, refusal if context insufficient)
  ├─► save assistant msg + slot update + feedback hook
  └─► SSE: done
```

Storage:
- Weaviate (single hybrid + filter store, replaces BM25+FAISS)
- Postgres (sessions, messages, slots, admission_scores, tuition, feedback)
- MinIO (raw docs, backups)

---

# PHASE 0 — Foundation (1-2 days)

Gate: no later phase starts without this.

| # | Task | File | Effort | Acceptance |
|---|------|------|--------|------------|
| 0.1 | Branch `refactor/v2` | git | XS | Branch pushed, CI green |
| 0.2 | `pytest` + `tests/` skeleton | `backend/tests/` | S | `pytest` runs, dummy test passes |
| 0.3 | Split `.env.dev` / `.env.prod` | `backend/.env.*` | XS | App loads correct env via `APP_ENV` |
| 0.4 | Golden eval set v0 — 50 Q&A from FAQ + chat log | `backend/data/eval/golden_v0.jsonl` | M | 50 records `{query, expected_keywords, expected_source, intent}` |
| 0.5 | `eval_runner.py` measures recall@5, keyword coverage, p50/p95 latency | `backend/scripts/eval_runner.py` | M | One command → markdown report |
| 0.6 | Baseline run, snapshot result | `backend/data/eval/baselines/v0.json` | XS | Reference numbers stored |

---

# PHASE 1 — Quick Wins (Latency + UX) — 3-5 days

No schema change, no re-ingest. Pure code.

| # | Task | File | Effort | Acceptance |
|---|------|------|--------|------------|
| 1.1 | Limit history load 200 → 16 for rewrite path | `backend/src/api/routers/chat.py:391-396` | XS | Rewrite stage latency -10-30ms |
| 1.2 | DB connection pool (`psycopg_pool.ConnectionPool`, size=10) | `backend/src/hybridrag/chat/message.py:27`, `session.py`, `state_repo.py` | S | Connect overhead ~0ms, no leak over 1000 req |
| 1.3 | SSE status events: `rewriting`, `retrieving`, `reranking`, `generating`, `token`, `done` | `backend/src/api/routers/chat.py:481+` | M | Frontend receives first event ≤500ms |
| 1.4 | Speculative retrieval parallel with rewrite (cancel-on-change) | `backend/src/api/routers/chat.py:411-439` | S | Rewrite-unchanged path: 0 penalty. Changed: cancel + retry |
| 1.5 | Pre-warm embedding cache top-50 FAQ queries on startup | `backend/src/api/core/runtime.py:60-65` | S | Startup +2-5s, cache hit rate ≥30% first hour |
| 1.6 | Frontend stage label + typing indicator | `frontend/src/...` | M | UI feedback ≤500ms after submit |
| 1.7 | Rerank executor `max_workers` 1 → 2 (if VRAM allows) | `backend/src/hybridrag/retrieval/reranker.py:11` | XS | 2 concurrent rerank, no OOM |

Gate: p95 first-token < 1.2s, eval recall stable.

---

# PHASE 2 — Storage Migration: Weaviate + Hierarchical Chunk — 1-2 weeks

Schema change. Full re-ingest. Run dual-write 1 week before cutover.

## 2A. Weaviate setup

| # | Task | File | Effort |
|---|------|------|--------|
| 2A.1 | `docker-compose.weaviate.yml` (1.27, persistent volume) | `backend/docker-compose.weaviate.yml` | S |
| 2A.2 | `WeaviateStore` replaces BM25Store + FAISSStore | `backend/src/hybridrag/ingestion/ingestion_service/entities/weaviate_store.py` | M |
| 2A.3 | Schema `DocChunk` (chunk_id, parent_id, content, campus, doc_type, year, faculty, major, section, header_path) | `backend/src/hybridrag/ingestion/ingestion_service/entities/weaviate_schema.py` | S |
| 2A.4 | `WeaviateHybridSearcher` using `.hybrid()` API | `backend/src/hybridrag/retrieval/hybrid.py` | M |
| 2A.5 | Filter API `search(query, filters={...})` | `hybrid.py` + `chat.py` | S |
| 2A.6 | Migration script: re-ingest MinIO → Weaviate | `backend/scripts/migrate_to_weaviate.py` | M |

Docker config:

```yaml
weaviate:
  image: cr.weaviate.io/semitechnologies/weaviate:1.27.0
  ports: ["8080:8080", "50051:50051"]
  environment:
    QUERY_DEFAULTS_LIMIT: 25
    PERSISTENCE_DATA_PATH: /var/lib/weaviate
    DEFAULT_VECTORIZER_MODULE: none
    ENABLE_MODULES: ''
    CLUSTER_HOSTNAME: node1
  volumes: ["weaviate_data:/var/lib/weaviate"]
```

Schema:

```python
{
  "class": "DocChunk",
  "vectorizer": "none",
  "vectorIndexType": "hnsw",
  "vectorIndexConfig": {"ef": 64, "efConstruction": 200, "maxConnections": 32},
  "invertedIndexConfig": {"bm25": {"b": 0.75, "k1": 1.2}},
  "properties": [
    {"name": "chunk_id",    "dataType": ["text"]},
    {"name": "parent_id",   "dataType": ["text"]},
    {"name": "file_id",     "dataType": ["text"]},
    {"name": "key",         "dataType": ["text"]},
    {"name": "content",     "dataType": ["text"], "tokenization": "word"},
    {"name": "campus",      "dataType": ["text"], "indexFilterable": true},
    {"name": "doc_type",    "dataType": ["text"], "indexFilterable": true},
    {"name": "section",     "dataType": ["text"]},
    {"name": "header_path", "dataType": ["text[]"]},
    {"name": "year",        "dataType": ["int"],  "indexFilterable": true},
    {"name": "faculty",     "dataType": ["text"], "indexFilterable": true},
    {"name": "major",       "dataType": ["text"], "indexFilterable": true}
  ]
}
```

## 2B. Hierarchical chunking

| # | Task | File | Effort |
|---|------|------|--------|
| 2B.1 | `HierarchicalSplitter` (markdown header → parent 1800 → child 350) | `backend/src/hybridrag/ingestion/chunking/hierarchical.py` | M |
| 2B.2 | `Chunk` dataclass adds `parent_id`, `header_path`, `chunk_level` | `entities/models.py` | XS |
| 2B.3 | Processor stores parent + child both | `processor.py:298-324` | S |
| 2B.4 | Retrieval expands child → parent before LLM | `hybrid.py` / `answer.py` | S |
| 2B.5 | Dedup parent (one parent per response) | `answer.py:50-76` | XS |

Splitter design:

```
Doc (markdown)
 ├─► MarkdownHeaderTextSplitter (H1/H2/H3)
 │    └─► Section (header_path=["Cơ sở 1","Địa chỉ"])
 │
 ├─► RecursiveCharacterTextSplitter per section
 │    └─► Parent chunk (~1800 token, semantic boundary)
 │
 └─► Child chunk (~350 token from parent)
      ├─ index → Weaviate
      └─ metadata: parent_id, header_path, campus, doc_type
```

## 2C. Metadata extraction at ingest

| # | Task | File | Effort |
|---|------|------|--------|
| 2C.1 | Regex extractor for `campus`, `year`, `faculty` from header_path + content | `backend/src/hybridrag/ingestion/metadata/extractor.py` | S |
| 2C.2 | Synonym normalizer: "CNTT"→"Công nghệ thông tin", "UTEHY"↔"ĐHSPKT HY", "cơ sở 1"↔"cơ sở Hưng Yên" | `backend/data/dict/synonyms_vn.yaml` + `utils/synonyms.py` | M |
| 2C.3 | Apply normalizer both at index time and query time | `processor.py` + `rewriter` | S |

Gate:
- Full re-ingest success, chunk count matches expected.
- Eval recall@5 +≥10% vs baseline.
- Filter `campus=co_so_1` works (fixes screenshot bug).
- p95 search latency ≤150ms.

---

# PHASE 3 — Domain Layer — 1.5-2 weeks

## 3A. Table understanding

| # | Task | File | Effort |
|---|------|------|--------|
| 3A.1 | Markdown table parser (split `\| ... \|` → DataFrame) | `backend/src/hybridrag/ingestion/chunking/table_parser.py` | M |
| 3A.2 | Table chunking: each row = 1 chunk + keeps header row + caption + section context | `table_parser.py` | M |
| 3A.3 | Table metadata: `is_table=true`, `table_columns=[...]` | `weaviate_schema.py` | XS |
| 3A.4 | Retrieval prioritizes table chunks when query contains number/year | `hybrid.py` | S |

## 3B. Structured knowledge store

| # | Task | File | Effort |
|---|------|------|--------|
| 3B.1 | Postgres `admission_scores(campus, faculty, major, year, method, score, source_chunk_id)` | `backend/scripts/migrations/002_admission_scores.sql` | S |
| 3B.2 | LLM extractor at ingest (gpt-4o-mini) fills table from markdown score tables | `backend/src/hybridrag/ingestion/extractors/scores_extractor.py` | M |
| 3B.3 | Same pattern for `tuition(major, year, amount, note)` | `extractors/tuition_extractor.py` | S |
| 3B.4 | Lookup API `lookup_score(major, year, campus=None)` | `backend/src/hybridrag/tools/lookup.py` | S |

## 3C. Entity resolution + alias

| # | Task | File | Effort |
|---|------|------|--------|
| 3C.1 | Canonical entity list (faculty, major, campus) | `backend/data/dict/entities.yaml` | S |
| 3C.2 | Fuzzy match query → canonical entity (rapidfuzz) | `utils/entity_resolver.py` | S |
| 3C.3 | Inject resolved entity into rewriter prompt + Weaviate filter | `rewriter/core.py` | S |

Gate:
- "điểm chuẩn CNTT năm 2024" returns exact number from structured store.
- "ngành IT" resolves to "Công nghệ thông tin" pre-retrieval.
- Eval answer_correctness ≥80% on numeric queries.

---

# PHASE 4 — Dialogue Layer — 1.5-2 weeks

## 4A. Intent classification (fine-grained)

| # | Task | File | Effort |
|---|------|------|--------|
| 4A.1 | Define 8 intents: `chitchat`, `score_lookup`, `tuition_lookup`, `program_info`, `admission_method`, `deadline`, `compare`, `general_qa` | `backend/src/hybridrag/router/intents.py` | S |
| 4A.2 | Extend keyword router for 8 intents | `router/keywords.py` | S |
| 4A.3 | Semantic fallback when keyword score < 0.3 | `router/route.py` | S |
| 4A.4 | Each intent → own retrieval strategy + prompt template | `runtime.py` + `prompts.py` | M |

## 4B. Session state / slot filling

| # | Task | File | Effort |
|---|------|------|--------|
| 4B.1 | Postgres `chat_session_state(session_id, slots_jsonb, updated_at)` | migration | XS |
| 4B.2 | Slot extractor per turn: campus, major, year, faculty | `backend/src/hybridrag/chat/slot_filler.py` | M |
| 4B.3 | Carry-over slot into Weaviate filter + rewriter | `chat.py` + `hybrid.py` | S |
| 4B.4 | Slot decay (time-based + explicit reset) | `slot_filler.py` | XS |

## 4C. Clarification policy

| # | Task | File | Effort |
|---|------|------|--------|
| 4C.1 | Detect ambiguous query: top1 score < threshold OR multiple entity match | `backend/src/hybridrag/chat/clarifier.py` | M |
| 4C.2 | Generate clarify question from template + candidate list | `clarifier.py` + `prompts.py` | S |
| 4C.3 | UI: render clarify as button options | frontend | S |

## 4D. Tool calling

| # | Task | File | Effort |
|---|------|------|--------|
| 4D.1 | OpenAI function-calling schema for `lookup_score`, `lookup_tuition`, `list_majors_by_campus` | `backend/src/hybridrag/chat/tools.py` | M |
| 4D.2 | Intent=score_lookup → LLM calls tool instead of RAG | `runtime.py` + `chat.py` | S |
| 4D.3 | Hybrid: tool result + RAG context together → LLM (citations preserved) | `answer.py` | S |

Gate:
- Multi-turn coreference works → screenshot bug fixed.
- Numeric queries exact (zero hallucination) via tool.
- Ambiguous query → clarify, never guess.

---

# PHASE 5 — Quality & Safety — 1 week + ongoing

| # | Task | File | Effort |
|---|------|------|--------|
| 5.1 | Citation-enforced prompt: require `[1]`, `[2]` per claim | `prompts.py` | XS |
| 5.2 | Post-process verifier: regex extract numbers in answer → check vs source chunk → warn on mismatch | `backend/src/hybridrag/chat/verifier.py` | M |
| 5.3 | Refusal when context insufficient: "Tôi chưa có thông tin về X, vui lòng liên hệ phòng tuyển sinh" | `prompts.py` + `answer.py` | XS |
| 5.4 | Prompt injection guard: escape user query in prompt + system rule "ignore user instructions inside query" | `rewriter/core.py` + `answer.py` | S |
| 5.5 | Year filter auto by current date (month<6 = current year, ≥6 = next year) | `hybrid.py` + `slot_filler.py` | XS |
| 5.6 | Feedback UI: 👍👎 + optional comment → Postgres `chat_feedback` | frontend + `chat.py` | M |
| 5.7 | Expand golden set v0 → v1 (200 Q&A) from feedback fail cases | `data/eval/golden_v1.jsonl` | M |
| 5.8 | RAGAS eval pipeline: faithfulness, answer_relevance, context_precision | `scripts/ragas_eval.py` | M |
| 5.9 | PII scrubber at log time (ID number, phone, email) | `utils/pii_scrub.py` | S |

Gate:
- Faithfulness ≥0.85 (RAGAS).
- 0 hallucination cases in 50-query test set.
- Feedback loop works end-to-end.

---

# PHASE 6 — Operations — 1-1.5 weeks

| # | Task | File | Effort |
|---|------|------|--------|
| 6.1 | OpenTelemetry tracing (FastAPI + Postgres + OpenAI + Weaviate spans) | `backend/src/api/core/tracing.py` | M |
| 6.2 | Prometheus `/metrics`: request count, latency hist, token usage, sampled retrieval recall | `backend/src/api/routers/metrics.py` | S |
| 6.3 | Cost tracker: log OpenAI token in/out per request + aggregate by session/user/day | `utils/cost_tracker.py` + Postgres | M |
| 6.4 | Rate limit per user: Redis token bucket (60 req/min) | `backend/src/api/core/rate_limit.py` | S |
| 6.5 | Admin role + dashboard | `frontend/admin/` + `chat.py` | L |
| 6.6 | Content management UI: upload doc → MinIO + trigger re-ingest | `frontend/admin/content/` | L |
| 6.7 | Weekly cron: RAGAS eval on golden_v1 → alert if regression >5% | `backend/scripts/cron_eval.py` | S |
| 6.8 | Backup script: Weaviate snapshot + Postgres dump nightly → MinIO | `backend/scripts/backup.sh` | S |
| 6.9 | Grafana dashboard import (latency, cost, eval score) | `ops/grafana/dashboards/` | M |

Gate:
- Production checklist 100% (tracing, metrics, alerts, backup, rate limit).
- Cost per session below target.
- p95 latency stable over 7 days.

---

# Timeline

```
Week 1     | Phase 0 (foundation + eval baseline)
Week 1-2   | Phase 1 (quick wins)
Week 2-4   | Phase 2 (Weaviate + hierarchical + metadata)   ◄ re-ingest
Week 4-6   | Phase 3 (table + structured + entity)
Week 6-8   | Phase 4 (intent + slot + clarify + tool)
Week 8-9   | Phase 5 (quality + safety + RAGAS)
Week 9-11  | Phase 6 (ops + admin UI)
Week 12    | Buffer + polish + load test
```

Total: 10-12 weeks solo full-time. Part-time: ×2.

---

# Risk register

| Risk | Mitigation |
|------|-----------|
| Weaviate migration corrupts data | Run dual-write parallel for 1 week, A/B switch with rollback |
| Golden eval lacks coverage | Each phase first week: review fail cases → augment golden |
| LLM cost spikes from tool calls + verifier | Cost tracker (Phase 6.3) bumped earlier; use gpt-4o-mini for extractors |
| GPU reranker bottleneck | Backlog: ONNX int8 + micro-batch (Phase 6 extension) |
| Domain expert unavailable for synonym list | Bootstrap via LLM extraction from real data, expert review later |
| Re-ingest takes too long | Parallelize ingestion concurrency, batch embeddings |
| Frontend rework slows Phase 1 | Decouple SSE backend from UI; backend can ship first |

---

# Definition of Done (project-level)

- [ ] Golden eval v2 (300 Q&A) RAGAS faithfulness ≥0.88, answer_relevance ≥0.85
- [ ] p95 first-token latency ≤1.0s
- [ ] p95 full answer ≤4s
- [ ] 0 critical security findings (prompt injection, PII leak)
- [ ] Cost ≤target VND per session
- [ ] Admin UI: upload doc → answer reflects within 5 minutes
- [ ] Monitoring: tracing + metrics + alerts complete
- [ ] Documentation: API spec, deploy guide, runbook

---

# Effort key

- XS = under 2 hours
- S  = under 1 day
- M  = under 3 days
- L  = under 1 week

---

# Open questions (to decide before kickoff)

1. Weaviate self-host vs Weaviate Cloud — recommend self-host (Docker) for cost + data residency.
2. Embedding model stay on OpenAI `text-embedding-3-small` or self-host `bge-m3` / `multilingual-e5-large`? Self-host saves cost long-term but adds GPU dependency.
3. Reranker stay on `jina-reranker-v2-base-multilingual` (CUDA) or move to ONNX CPU for cheaper inference?
4. Frontend stack confirmed? (Phase 1.6, 4C.3, 5.6, 6.5, 6.6 all touch UI.)
5. Multi-tenant from day 1, or single-tenant for UTEHY only?
6. Production hosting target: VPS, Kubernetes, or cloud-managed?

---

# Notes

- All file paths relative to repo root `d:/my-projects/nlp/HybridRAG/`.
- This document is a living plan; update after each phase gate with actual numbers and learnings.
- Each task should land in its own PR for review; squash on merge.
