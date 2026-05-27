-- Phase 6.3: OpenAI token usage + cost tracking
CREATE TABLE IF NOT EXISTS chat_cost_log (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID,
    user_id UUID,
    model TEXT NOT NULL,
    tokens_in INT NOT NULL DEFAULT 0,
    tokens_out INT NOT NULL DEFAULT 0,
    cost_usd NUMERIC(10, 6) NOT NULL DEFAULT 0,
    feature TEXT NOT NULL DEFAULT 'answer',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_chat_cost_log_created_at ON chat_cost_log (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_chat_cost_log_session   ON chat_cost_log (session_id);
CREATE INDEX IF NOT EXISTS ix_chat_cost_log_user      ON chat_cost_log (user_id);
