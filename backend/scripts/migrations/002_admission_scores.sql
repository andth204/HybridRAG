-- Phase 3B: structured admission knowledge
--
-- Two tables that the dialogue / tool layer (Phase 4) calls instead of
-- routing numeric questions through RAG:
--
--   * admission_scores  — one row per (campus, major, year, method,
--                         subject_combo) admission cutoff
--   * tuition           — one row per (major, year, unit) tuition rate
--
-- Both tables are populated by an LLM extractor at ingest time
-- (see backend/src/hybridrag/ingestion/extractors/*) and are queried
-- by backend/src/hybridrag/tools/lookup.py.
--
-- Idempotent on purpose: CREATE TABLE IF NOT EXISTS and CREATE INDEX
-- IF NOT EXISTS so the migration script can be replayed safely.

CREATE TABLE IF NOT EXISTS admission_scores (
    id BIGSERIAL PRIMARY KEY,
    campus TEXT NOT NULL DEFAULT '',
    faculty TEXT NOT NULL DEFAULT '',
    major_canonical TEXT NOT NULL,
    major_code TEXT,
    year INT NOT NULL,
    method TEXT NOT NULL DEFAULT '',
    subject_combo TEXT,
    score NUMERIC(5,2),
    note TEXT,
    source_file TEXT,
    source_chunk_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (campus, major_canonical, year, method, subject_combo)
);

CREATE INDEX IF NOT EXISTS ix_admission_scores_year_major  ON admission_scores (year, major_canonical);
CREATE INDEX IF NOT EXISTS ix_admission_scores_campus_year ON admission_scores (campus, year);
CREATE INDEX IF NOT EXISTS ix_admission_scores_faculty     ON admission_scores (faculty);

CREATE TABLE IF NOT EXISTS tuition (
    id BIGSERIAL PRIMARY KEY,
    major_canonical TEXT NOT NULL,
    year INT NOT NULL,
    amount_vnd BIGINT,
    unit TEXT NOT NULL DEFAULT 'per_credit',
    note TEXT,
    source_file TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (major_canonical, year, unit)
);

CREATE INDEX IF NOT EXISTS ix_tuition_year_major ON tuition (year, major_canonical);
