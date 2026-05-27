-- Phase 4B: per-session slot state for dialogue manager
--
-- One row per chat session, holding the dialogue manager's slot frame:
--   campus, major, year, faculty, doc_type.
--
-- The ``slots`` column is JSONB so the application can evolve the slot
-- shape without DDL churn. Each top-level key is one slot name; the
-- value is an object with the following fields (mirrored by
-- ``src.hybridrag.chat.session_state.SlotValue``):
--
--   {
--     "value":      <canonical key | int | string>,   -- the resolved entity
--     "display":    <user-facing name>,               -- e.g. "Cơ sở 1"
--     "set_at":     "2026-05-21T12:34:56+00:00",      -- ISO-8601 UTC
--     "confidence": 0.95,                              -- 0..1 (1.0 for regex)
--     "turn":       3                                  -- turn index when set
--   }
--
-- ``last_intent`` and ``last_query`` are denormalized hints for the
-- dialogue layer (clarifier, rewriter context). The slot store is the
-- source of truth — these two fields are advisory only.
--
-- Idempotent on purpose so ``scripts/apply_migrations.py`` can replay.

CREATE TABLE IF NOT EXISTS chat_session_state (
    session_id UUID PRIMARY KEY,
    slots JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_intent TEXT,
    last_query TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_chat_session_state_updated_at ON chat_session_state (updated_at DESC);
