-- Phase 5.6: chat feedback (thumbs up / down + optional comment)
--
-- One row per feedback event. We deliberately allow multiple rows per
-- (user, message) pair so a user who revises their opinion has both
-- ratings visible to whoever audits the feedback later.
--
-- ``comment`` is PII-scrubbed at the API layer BEFORE storage
-- (see :func:`src.api.routers.chat.create_message_feedback`). The
-- backing column is plain ``TEXT`` because we expect Vietnamese
-- free-text up to a few hundred chars and want full-text search
-- compatibility down the road.
--
-- Idempotent on purpose so ``scripts/apply_migrations.py`` can replay.

CREATE TABLE IF NOT EXISTS chat_feedback (
    id BIGSERIAL PRIMARY KEY,
    message_id UUID NOT NULL,
    session_id UUID NOT NULL,
    user_id UUID,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_chat_feedback_message ON chat_feedback (message_id);
CREATE INDEX IF NOT EXISTS ix_chat_feedback_session ON chat_feedback (session_id, created_at DESC);
