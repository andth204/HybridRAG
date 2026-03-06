CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Bảng người dùng (lưu thông tin đăng nhập Google OAuth)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    google_id VARCHAR(255) UNIQUE,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100),
    role VARCHAR(20) NOT NULL DEFAULT 'user' CHECK (role IN ('manager', 'user')),
    is_blocked BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trigger cập nhật thời gian cập nhật
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Bảng phiên trò chuyện
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER update_chat_sessions_updated_at
    BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Bảng tin nhắn
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE, -- Gắn với phiên
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')), -- Giới hạn vai trò
    content TEXT NOT NULL, -- Nội dung tin nhắn
    parent_message_id UUID REFERENCES chat_messages(id), -- Nếu hỗ trợ branch hoặc edit
    revision_number INTEGER DEFAULT 1, -- Phiên bản chỉnh sửa
    is_edited BOOLEAN DEFAULT FALSE,
    metadata JSONB, -- Dữ liệu RAG: sources, retrieved_docs, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes để tăng hiệu suất truy vấn
CREATE INDEX idx_users_google_id ON users(google_id); -- Truy vấn nhanh theo Google ID
CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id); -- Liệt kê session theo user
CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id); -- Lấy lịch sử theo session
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at DESC); -- Sắp xếp tin nhắn mới nhất trước