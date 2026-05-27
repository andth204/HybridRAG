"""Tests for ``src.hybridrag.router.intent_router``.

These tests use light-weight stubs for both the keyword and semantic
classifiers so the dispatch logic can be inspected in isolation
(without needing a YAML, golden file, or embedder).
"""
from __future__ import annotations

from typing import Optional

import pytest

from src.hybridrag.router.intent_classifier import KeywordIntentClassifier
from src.hybridrag.router.intent_router import IntentRouter
from src.hybridrag.router.intents import (
    INTENT_FALLBACK_THRESHOLD,
    Intent,
    IntentResult,
)


# ---------------------------------------------------------------- #
# Stubs
# ---------------------------------------------------------------- #
class _StubKeyword:
    """Minimal stand-in for ``KeywordIntentClassifier`` — just returns
    a pre-canned :class:`IntentResult` and records every call."""

    def __init__(self, result: IntentResult) -> None:
        self._result = result
        self.calls: list[str] = []
        # ``IntentRouter._ambiguous`` peeks at ``_rules``; an empty list
        # makes the helper short-circuit to ``False`` (= not ambiguous).
        self._rules: list = []

    def classify(self, query: str) -> IntentResult:
        self.calls.append(query)
        return self._result


class _StubSemantic:
    def __init__(self, result: Optional[IntentResult]) -> None:
        self._result = result
        self.calls: list[str] = []

    async def classify(self, query: str) -> IntentResult:
        self.calls.append(query)
        if self._result is None:
            raise RuntimeError("semantic blow-up")
        return self._result


# ---------------------------------------------------------------- #
# Dispatch logic
# ---------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_high_keyword_returns_immediately() -> None:
    """High-confidence keyword result is returned without touching semantic."""
    kw = _StubKeyword(
        IntentResult(
            intent=Intent.SCORE_LOOKUP,
            score=0.9,
            matched=["diem chuan"],
            source="keyword",
        )
    )
    sem = _StubSemantic(
        IntentResult(intent=Intent.GENERAL_QA, score=0.5, source="semantic")
    )
    router = IntentRouter(keyword=kw, semantic=sem)

    out = await router.classify("điểm chuẩn CNTT 2024")
    assert out.intent is Intent.SCORE_LOOKUP
    assert out.score == 0.9
    assert sem.calls == []  # semantic never invoked
    assert kw.calls == ["điểm chuẩn CNTT 2024"]


@pytest.mark.asyncio
async def test_low_keyword_invokes_semantic_and_takes_higher_score() -> None:
    """Below threshold → consult semantic; pick the higher-scoring one."""
    kw = _StubKeyword(
        IntentResult(
            intent=Intent.GENERAL_QA,
            score=0.1,
            source="fallback",
        )
    )
    sem = _StubSemantic(
        IntentResult(
            intent=Intent.DEADLINE,
            score=0.72,
            source="semantic",
        )
    )
    router = IntentRouter(keyword=kw, semantic=sem)
    out = await router.classify("khi nào trường mở cổng đăng ký")
    assert out.intent is Intent.DEADLINE
    assert out.score == 0.72
    assert sem.calls, "semantic must be consulted when keyword is below threshold"


@pytest.mark.asyncio
async def test_low_keyword_semantic_lower_keeps_keyword() -> None:
    """If semantic loses on score, keep the keyword (default) result."""
    kw_res = IntentResult(
        intent=Intent.GENERAL_QA,
        score=0.15,
        source="fallback",
    )
    sem_res = IntentResult(
        intent=Intent.CHITCHAT,
        score=0.05,
        source="semantic",
    )
    router = IntentRouter(keyword=_StubKeyword(kw_res), semantic=_StubSemantic(sem_res))
    out = await router.classify("some odd query")
    assert out is kw_res or (out.intent is kw_res.intent and out.score == kw_res.score)


@pytest.mark.asyncio
async def test_semantic_fallback_does_not_override_keyword() -> None:
    """When semantic returns ``source='fallback'`` keep the keyword result."""
    kw_res = IntentResult(
        intent=Intent.GENERAL_QA,
        score=0.2,
        source="fallback",
    )
    sem_res = IntentResult(
        intent=Intent.GENERAL_QA,
        score=0.0,
        source="fallback",
    )
    router = IntentRouter(keyword=_StubKeyword(kw_res), semantic=_StubSemantic(sem_res))
    out = await router.classify("unknown")
    assert out is kw_res


@pytest.mark.asyncio
async def test_semantic_exception_is_safe() -> None:
    """If the semantic classifier crashes we fall back to the keyword pick."""
    kw_res = IntentResult(
        intent=Intent.GENERAL_QA,
        score=0.05,
        source="fallback",
    )
    router = IntentRouter(
        keyword=_StubKeyword(kw_res),
        semantic=_StubSemantic(None),  # will raise
    )
    out = await router.classify("anything")
    assert out is kw_res


@pytest.mark.asyncio
async def test_ambiguity_triggers_semantic() -> None:
    """Even if keyword is above threshold, near-tie among top intents
    should still consult the semantic classifier — and pick whichever
    side has the higher score."""

    # We need a real ``KeywordIntentClassifier`` so the ambiguity helper
    # has rules to look at. We craft a query that triggers both
    # ``compare`` and ``score_lookup`` hits closely.
    real_kw = KeywordIntentClassifier()

    class _ForcedAmbig(KeywordIntentClassifier):
        """Reuse the rule bank from the real classifier; force the
        ``classify`` return value to look high-confidence so we exercise
        only the ambiguity branch."""

        def __init__(self, base: KeywordIntentClassifier) -> None:
            # Copy compiled state from ``base`` — we don't load again.
            self._yaml_path = base._yaml_path
            self._rules = base._rules

        def classify(self, query: str) -> IntentResult:  # type: ignore[override]
            return IntentResult(
                intent=Intent.SCORE_LOOKUP,
                score=0.6,
                matched=["diem chuan"],
                source="keyword",
            )

    sem_res = IntentResult(intent=Intent.COMPARE, score=0.95, source="semantic")
    router = IntentRouter(
        keyword=_ForcedAmbig(real_kw),
        semantic=_StubSemantic(sem_res),
    )

    # Query containing both score_lookup and compare keywords →
    # the helper sees a near-tie and calls semantic.
    out = await router.classify("so sánh điểm chuẩn")
    assert out.intent is Intent.COMPARE
    assert out.score == 0.95


@pytest.mark.asyncio
async def test_lazy_semantic_instantiation() -> None:
    """If we never need the semantic classifier it is never built."""
    kw_res = IntentResult(
        intent=Intent.SCORE_LOOKUP,
        score=0.95,
        source="keyword",
    )
    router = IntentRouter(keyword=_StubKeyword(kw_res))
    assert router._sem is None
    out = await router.classify("điểm chuẩn CNTT")
    assert out is kw_res
    assert router._sem is None
