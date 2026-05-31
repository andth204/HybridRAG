-- v3.5: long-session conversation summary.
--
-- Stores a rolling LLM-generated summary on chat_session_state so the
-- rewriter and answer-compose paths can see beyond the most recent
-- ~K_REWRITE user turns. The summary is updated only when the session
-- crosses settings.SUMMARY_TRIGGER_TURN (default 12) and re-generated
-- every settings.SUMMARY_REFRESH_EVERY turns thereafter — short
-- sessions never pay the LLM cost.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS so re-running the migration
-- against an already-upgraded DB is safe.

ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS conversation_summary  TEXT        DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS summary_updated_at    TIMESTAMPTZ DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS summary_turn_count    INT         DEFAULT 0;
