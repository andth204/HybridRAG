"""Phase 5.6 — Tests for the feedback endpoint.

Covers ``POST /api/v1/chat/messages/{message_id}/feedback``:

* Happy path: creates a row and returns 201 + payload.
* PII scrubbing: phone / email tokens in the comment are redacted
  before persistence.
* Pydantic-level validation: an invalid rating returns 422.
* Authorization: a message belonging to a different user's session
  returns 404 (never leaks the existence of foreign messages).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.core.dependencies import AuthContext, get_auth_context
from src.api.routers import chat as chat_module
from src.hybridrag.chat.feedback import FeedbackRecord
from src.hybridrag.chat.message import ChatMessage
from src.hybridrag.chat.session import ChatSession


# -------------------------------------------------------------------- #
# Fakes
# -------------------------------------------------------------------- #
@dataclass
class _StoredFeedback:
    message_id: str
    session_id: str
    user_id: Optional[str]
    rating: str
    comment: Optional[str]


class _FakeFeedbackRepo:
    """Captures create() calls in-memory; mimics the real repo's API."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.records: list[_StoredFeedback] = []
        self._next_id = 1

    def create(
        self,
        *,
        message_id: str,
        session_id: str,
        user_id: str | None,
        rating: str,
        comment: str | None = None,
    ) -> FeedbackRecord:
        if rating not in {"up", "down"}:
            raise ValueError("rating must be 'up' or 'down'")
        stored = _StoredFeedback(
            message_id=message_id,
            session_id=session_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
        )
        self.records.append(stored)
        record = FeedbackRecord(
            id=self._next_id,
            message_id=message_id,
            session_id=session_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
        )
        self._next_id += 1
        return record


class _FakeSessionRepo:
    def __init__(self, *, owner_user_id: str, session_id: str) -> None:
        self.owner_user_id = owner_user_id
        self.session_id = session_id

    def get(self, session_id: str, user_id: Optional[str] = None) -> Optional[ChatSession]:
        if session_id != self.session_id:
            return None
        if user_id and user_id != self.owner_user_id:
            return None
        return ChatSession(
            id=self.session_id,
            user_id=self.owner_user_id,
            title=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )


class _FakeMessageRepo:
    def __init__(
        self,
        *,
        message_id: str,
        session_id: str,
    ) -> None:
        self.message_id = message_id
        self.session_id = session_id

    def get(self, message_id: str) -> Optional[ChatMessage]:
        if message_id != self.message_id:
            return None
        return ChatMessage(
            id=self.message_id,
            session_id=self.session_id,
            role="assistant",
            content="dummy",
            parent_message_id=None,
            revision_number=1,
            is_edited=False,
            metadata=None,
            created_at=datetime.now(timezone.utc),
        )


# -------------------------------------------------------------------- #
# Harness
# -------------------------------------------------------------------- #
def _build_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    owner_user_id: str,
    caller_user_id: str | None = None,
    message_exists: bool = True,
) -> tuple[TestClient, _FakeFeedbackRepo, str, str]:
    """Spin up a FastAPI app with chat router and stubbed repos."""
    session_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    fb_repo = _FakeFeedbackRepo()
    if message_exists:
        msg_repo = _FakeMessageRepo(message_id=message_id, session_id=session_id)
    else:
        # Repo returns None for everything → 404 path.
        class _EmptyMsgRepo:
            def get(self, _: str) -> None:
                return None
        msg_repo = _EmptyMsgRepo()  # type: ignore[assignment]

    sess_repo = _FakeSessionRepo(owner_user_id=owner_user_id, session_id=session_id)

    monkeypatch.setattr(chat_module, "ChatFeedbackRepo", lambda *a, **kw: fb_repo)
    monkeypatch.setattr(chat_module, "ChatMessageRepo", lambda *a, **kw: msg_repo)
    monkeypatch.setattr(chat_module, "ChatSessionRepo", lambda *a, **kw: sess_repo)

    app = FastAPI()
    app.include_router(chat_module.router)
    auth_user = caller_user_id or owner_user_id
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        user_id=auth_user,
        access_token="test-token",
        user_role="user",
    )
    return TestClient(app), fb_repo, session_id, message_id


# -------------------------------------------------------------------- #
# Tests
# -------------------------------------------------------------------- #
def test_post_feedback_creates_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: 201 + persisted row + JSON body mirrors the record."""
    owner = str(uuid.uuid4())
    client, fb_repo, session_id, message_id = _build_harness(
        monkeypatch, owner_user_id=owner
    )
    resp = client.post(
        f"/api/v1/chat/messages/{message_id}/feedback",
        json={"rating": "up", "comment": "rất hữu ích"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["rating"] == "up"
    assert body["message_id"] == message_id
    assert body["session_id"] == session_id
    assert body["user_id"] == owner
    assert body["comment"] == "rất hữu ích"
    # Repo received the create() call.
    assert len(fb_repo.records) == 1
    stored = fb_repo.records[0]
    assert stored.rating == "up"
    assert stored.comment == "rất hữu ích"
    assert stored.user_id == owner


def test_post_feedback_scrubs_pii(monkeypatch: pytest.MonkeyPatch) -> None:
    """A comment containing a phone number is scrubbed before storage."""
    owner = str(uuid.uuid4())
    client, fb_repo, _, message_id = _build_harness(
        monkeypatch, owner_user_id=owner
    )
    raw_comment = "Bạn gọi mình ở 0987654321 nhé"
    resp = client.post(
        f"/api/v1/chat/messages/{message_id}/feedback",
        json={"rating": "down", "comment": raw_comment},
    )
    assert resp.status_code == 201, resp.text
    # The phone is gone from BOTH the stored row and the response body.
    stored = fb_repo.records[0]
    assert "0987654321" not in (stored.comment or "")
    assert "***" in (stored.comment or ""), (
        f"expected scrub redact marker in stored comment, got {stored.comment!r}"
    )
    assert "0987654321" not in resp.json().get("comment", "")


def test_post_feedback_invalid_rating(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pydantic rejects unknown rating values with 422."""
    owner = str(uuid.uuid4())
    client, fb_repo, _, message_id = _build_harness(
        monkeypatch, owner_user_id=owner
    )
    resp = client.post(
        f"/api/v1/chat/messages/{message_id}/feedback",
        json={"rating": "other", "comment": "..."},
    )
    # FastAPI/Pydantic surfaces validation errors as 422 by default.
    assert resp.status_code in (400, 422), resp.text
    # The repo was never touched.
    assert fb_repo.records == []


def test_post_feedback_message_not_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    """A message belonging to someone else's session yields a clean 404."""
    owner = str(uuid.uuid4())
    intruder = str(uuid.uuid4())
    client, fb_repo, _, message_id = _build_harness(
        monkeypatch,
        owner_user_id=owner,
        caller_user_id=intruder,
    )
    resp = client.post(
        f"/api/v1/chat/messages/{message_id}/feedback",
        json={"rating": "up"},
    )
    assert resp.status_code == 404, resp.text
    # The repo must never have been called for a foreign-owned message.
    assert fb_repo.records == []


def test_post_feedback_message_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-existent message also returns 404 (no information leak)."""
    owner = str(uuid.uuid4())
    client, fb_repo, _, message_id = _build_harness(
        monkeypatch, owner_user_id=owner, message_exists=False
    )
    resp = client.post(
        f"/api/v1/chat/messages/{message_id}/feedback",
        json={"rating": "up"},
    )
    assert resp.status_code == 404, resp.text
    assert fb_repo.records == []
