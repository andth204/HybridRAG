"""Clarification policy (Phase 4C).

When a user query is ambiguous, the chatbot should ask ONE focused
follow-up question instead of guessing. Two broad ambiguity classes
are handled here:

A. **Entity ambiguity** — the query mentions an entity type but more
   than one candidate resolves with comparable confidence. Examples:
     - "Học phí ngành kỹ thuật"  → which engineering major?
     - "Điểm chuẩn năm trước"    → which year?
     - "Cơ sở dạy CNTT"          → cơ sở 1 / 2 / 3?

B. **Low recall** — the retrieval layer returned documents but the
   top-1 score is below a useful threshold, so we cannot give a
   confident answer.

The :class:`Clarifier` is **opt-in**: callers decide whether to
short-circuit the answer pipeline with a clarification turn. The
class itself never raises — ``check()`` returns either a
:class:`ClarificationRequest` or ``None``.

This module is fully standalone — it does NOT depend on the runtime,
chat router or retrieval modules. Integration with the live chat
pipeline happens in a later step.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml
from rapidfuzz import fuzz

from src.config.prompts import CLARIFICATION_QUESTIONS
from src.config.settings import settings
from src.hybridrag.utils.entity_resolver import resolve_all
from src.hybridrag.utils.synonyms import _normalize_key


log = logging.getLogger(__name__)


# Heuristic thresholds (override-able via constructor)
DEFAULT_ENTITY_MIN_SCORE = 70   # below this, treat candidate as weak
DEFAULT_AMBIGUITY_GAP    = 5    # top1 - topN within this gap → ambiguous
DEFAULT_LOW_RECALL_THRESHOLD = 0.30  # top retrieval score below → clarify
DEFAULT_MAX_OPTIONS = 4          # candidates per clarification

# Intents that should have a year slot bound before answering.
_YEAR_REQUIRED_INTENTS = {"score_lookup", "tuition_lookup"}

# Intents that should have a major bound before answering.
_MAJOR_REQUIRED_INTENTS = {"score_lookup", "tuition_lookup"}

# Years to offer when "missing_year" fires (current, -1, -2).
_YEAR_OFFSETS = (0, -1, -2)

# Curated popular majors fallback (used only if entities.yaml fails to load).
_POPULAR_MAJOR_FALLBACK = (
    {"value": "cong_nghe_thong_tin", "label": "Công nghệ thông tin"},
    {"value": "ky_thuat_phan_mem",   "label": "Kỹ thuật phần mềm"},
    {"value": "khoa_hoc_may_tinh",   "label": "Khoa học máy tính"},
    {"value": "quan_tri_kinh_doanh", "label": "Quản trị kinh doanh"},
)

# Hints that the user is asking about a "major-like" thing without
# narrowing to one specific ngành. If at least one of these appears
# AND multiple major candidates resolve close to each other, we ask.
_MAJOR_NOUN_HINTS = (
    "nganh",        # ngành
    "khoa",
    "ky thuat",     # kỹ thuật
    "cong nghe",    # công nghệ
    "chuyen nganh", # chuyên ngành
    "khoi",         # khối
)

# Year-like tokens already present in the query → skip missing_year.
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


# -------------------------------------------------------------------- #
# Public data classes
# -------------------------------------------------------------------- #
@dataclass
class ClarificationRequest:
    """Result of :meth:`Clarifier.check`.

    Attributes:
        reason:    Tag identifying why clarification is needed; one of
                   ``"ambiguous_major"``, ``"missing_year"``,
                   ``"missing_major"``, ``"ambiguous_campus"``,
                   ``"low_recall"``.
        question:  Rendered Vietnamese question to send back to the user.
        options:   Structured candidate list ``[{"value":..,"label":..}]``.
                   Empty for ``low_recall`` (free-form clarification).
        slot:      Which session slot the user's answer should populate
                   (``"major"``, ``"year"``, ``"campus"``). ``None`` for
                   ``low_recall``.
        extra:     Optional debug info — top retrieval score, raw scores,
                   intent, etc. Not surfaced to the end user.
    """

    reason: str
    question: str
    options: list[dict[str, str]] = field(default_factory=list)
    slot: Optional[str] = None
    extra: dict[str, Any] | None = None


# -------------------------------------------------------------------- #
# Helpers (module-level so they're independently testable)
# -------------------------------------------------------------------- #
def _normalized_query(text: str) -> str:
    """Diacritic / case-folded query for keyword hint detection."""
    return _normalize_key(text or "")


def _record_score(record: dict[str, Any], query_norm: str) -> float:
    """Best fuzzy score between ``query_norm`` and any alias of ``record``.

    ``resolve_all`` returns the canonical records but drops the internal
    score. To detect "top1 vs top2 within ambiguity_gap" we re-score
    each record against the (already normalized) query using
    ``rapidfuzz.fuzz.WRatio`` over canonical / display / aliases / code.

    If the record already carries an explicit ``_score`` / ``score``
    field (e.g. injected by a unit test mock), that wins — no recompute.
    """
    for key in ("_score", "score"):
        val = record.get(key)
        if isinstance(val, (int, float)):
            return float(val)

    if not query_norm:
        return 0.0

    candidates: list[str] = []
    for f in ("canonical", "display"):
        v = record.get(f)
        if isinstance(v, str):
            candidates.append(v)
    canonical = record.get("canonical")
    if isinstance(canonical, str) and "_" in canonical:
        candidates.append(canonical.replace("_", " "))
    for a in record.get("aliases", []) or []:
        if isinstance(a, str):
            candidates.append(a)
    code = record.get("code")
    if isinstance(code, str):
        candidates.append(code)

    best = 0.0
    for cand in candidates:
        norm = _normalize_key(cand)
        if not norm:
            continue
        s = fuzz.WRatio(query_norm, norm)
        if s > best:
            best = s
    return float(best)


def _doc_score(doc: dict[str, Any]) -> Optional[float]:
    """Extract the most-trustworthy score from a retrieval doc dict.

    Tries (in order): ``rerank_score``, ``rrf_score``, ``vector_score``,
    ``score``. Returns ``None`` if none of these keys carry a number.
    """
    for key in ("rerank_score", "rrf_score", "vector_score", "score"):
        v = doc.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _format_options_text(options: Iterable[dict[str, str]]) -> str:
    """Render ``options`` as a numbered list (``1. label\\n2. label\\n...``)."""
    lines: list[str] = []
    for i, opt in enumerate(options, start=1):
        label = opt.get("label") or opt.get("value") or ""
        if not label:
            continue
        lines.append(f"{i}. {label}")
    return "\n".join(lines)


def _render_question(reason: str, options: list[dict[str, str]]) -> str:
    """Format ``CLARIFICATION_QUESTIONS[reason]`` with the options block."""
    template = CLARIFICATION_QUESTIONS.get(reason)
    if not template:
        log.warning("clarifier.unknown_reason reason=%s", reason)
        return CLARIFICATION_QUESTIONS["low_recall"]
    options_text = _format_options_text(options) if options else ""
    try:
        return template.format(options_text=options_text)
    except KeyError:
        # Template has no {options_text} placeholder (low_recall) — return as-is.
        return template


def _record_to_option(record: dict[str, Any]) -> dict[str, str]:
    """Convert an entity record to a UI-friendly option dict."""
    value = str(record.get("canonical") or "")
    label = str(record.get("display") or value.replace("_", " ").strip())
    return {"value": value, "label": label}


# -------------------------------------------------------------------- #
# Popular-majors fallback list loader
# -------------------------------------------------------------------- #
def _load_popular_majors(max_options: int) -> list[dict[str, str]]:
    """Return up to ``max_options`` curated majors from ``entities.yaml``.

    Strategy: the canonical list inside ``entities.yaml`` for the
    ``major`` type is curated by hand; we pick the ones in our
    ``_POPULAR_MAJOR_FALLBACK`` priority order if they exist, then top
    up from the remaining majors in YAML order.
    """
    yaml_path = settings.BASE_DIR / "data" / "dict" / "entities.yaml"
    records: list[dict[str, Any]] = []
    if yaml_path.exists():
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            majors = data.get("major") or []
            if isinstance(majors, list):
                records = [r for r in majors if isinstance(r, dict) and r.get("canonical")]
        except yaml.YAMLError as exc:
            log.warning("clarifier.popular_majors yaml error: %s", exc)

    if not records:
        return [dict(o) for o in _POPULAR_MAJOR_FALLBACK[:max_options]]

    by_canonical = {r["canonical"]: r for r in records}
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    # 1. Priority list first.
    for o in _POPULAR_MAJOR_FALLBACK:
        canon = o["value"]
        rec = by_canonical.get(canon)
        if rec is None:
            continue
        out.append(_record_to_option(rec))
        seen.add(canon)
        if len(out) >= max_options:
            return out
    # 2. Then top up with the remaining YAML majors in declared order.
    for r in records:
        canon = r["canonical"]
        if canon in seen:
            continue
        out.append(_record_to_option(r))
        seen.add(canon)
        if len(out) >= max_options:
            break
    return out


# -------------------------------------------------------------------- #
# Clarifier
# -------------------------------------------------------------------- #
class Clarifier:
    """Policy gate that decides whether to ask the user a clarifying question.

    The class is stateless aside from configured thresholds — a single
    instance can be reused across requests and threads.
    """

    def __init__(
        self,
        entity_min_score: int = DEFAULT_ENTITY_MIN_SCORE,
        ambiguity_gap: int = DEFAULT_AMBIGUITY_GAP,
        low_recall_threshold: float = DEFAULT_LOW_RECALL_THRESHOLD,
        max_options: int = DEFAULT_MAX_OPTIONS,
    ) -> None:
        self.entity_min_score = entity_min_score
        self.ambiguity_gap = ambiguity_gap
        self.low_recall_threshold = low_recall_threshold
        self.max_options = max(1, int(max_options))

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def check(
        self,
        *,
        query: str,
        intent: str | None = None,
        session_slots: dict[str, Any] | None = None,
        retrieval_docs: list[dict[str, Any]] | None = None,
    ) -> Optional[ClarificationRequest]:
        """Run the clarification policy chain. First match wins.

        Order:
          1. Ambiguous major (entity ambiguity).
          2. Missing year (slot ambiguity).
          3. Missing major (slot ambiguity).
          4. Ambiguous campus (entity ambiguity).
          5. Low retrieval recall (generic).

        Args:
            query:           The user's current question (post-rewrite).
            intent:          One of the 8 router intents (or ``None``).
            session_slots:   Slots already bound from prior turns
                             (``{"campus":..,"year":..,"major":..}``).
            retrieval_docs:  Top-N docs from the retrieval layer (may be
                             ``None`` if retrieval has not run yet).

        Returns:
            A :class:`ClarificationRequest` if the system should ask a
            follow-up, otherwise ``None``.

        Disabled per product decision (2026-05-22): user prefers the
        bot to always answer based on retrieval rather than ask
        clarifying questions back. The internal helpers
        (``_check_ambiguous_entity``, ``_check_missing_year`` etc.)
        are kept intact so the policy can be re-enabled in the future
        by removing this short-circuit.
        """
        return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _check_ambiguous_entity(
        self,
        query: str,
        kind: str,
        current_value: Any | None,
    ) -> Optional[ClarificationRequest]:
        """Detect ambiguity among the top-K candidates of one entity type."""
        if current_value:
            # User already pinned this slot — no ambiguity to resolve.
            return None

        if not query:
            return None

        # For majors, only trigger if the query mentions a major-like noun.
        # This avoids false positives on chitchat or one-word queries.
        if kind == "major" and not self._mentions_major_noun(query):
            return None

        try:
            hits = resolve_all(query, entity_types=[kind])
        except Exception as exc:  # noqa: BLE001
            log.debug("clarifier.resolve_all_failed kind=%s err=%s", kind, exc)
            return None

        records: list[dict[str, Any]] = list(hits.get(kind, []))
        if len(records) < 2:
            return None

        query_norm = _normalized_query(query)

        # If the query contains ANY record's full alias / canonical /
        # display as a substring, treat that as an unambiguous match and
        # skip the clarifier entirely. This avoids false-positive
        # ambiguity when the user already typed the full entity name
        # but rapidfuzz's WRatio scores siblings with a shared prefix
        # within the ``ambiguity_gap`` window (e.g. "Công nghệ thông
        # tin" vs "Công nghệ may" vs "Công nghệ giáo dục" — all start
        # with "công nghệ"). Containment is a stronger signal than
        # fuzzy similarity for this case.
        for rec in records:
            aliases: list[str] = []
            for f in ("canonical", "display"):
                v = rec.get(f)
                if isinstance(v, str) and v.strip():
                    aliases.append(v)
            canonical = rec.get("canonical")
            if isinstance(canonical, str) and "_" in canonical:
                aliases.append(canonical.replace("_", " "))
            for a in rec.get("aliases", []) or []:
                if isinstance(a, str) and a.strip():
                    aliases.append(a)
            for cand in aliases:
                norm = _normalize_key(cand)
                # Require the candidate be non-trivial (>=2 tokens or
                # >=8 chars) so a generic word like "may" doesn't
                # accidentally win against "công nghệ may".
                if not norm or (len(norm) < 8 and " " not in norm):
                    continue
                if norm in query_norm:
                    return None  # exact match — no clarification needed

        scored: list[tuple[float, dict[str, Any]]] = []
        for rec in records:
            s = _record_score(rec, query_norm)
            if s >= self.entity_min_score:
                scored.append((s, rec))

        if len(scored) < 2:
            return None

        scored.sort(key=lambda p: -p[0])
        top_score = scored[0][0]
        within_gap: list[tuple[float, dict[str, Any]]] = [
            (s, r) for (s, r) in scored if (top_score - s) <= self.ambiguity_gap
        ]

        if len(within_gap) < 2:
            return None

        chosen = within_gap[: self.max_options]
        options = [_record_to_option(rec) for (_s, rec) in chosen]
        reason = "ambiguous_major" if kind == "major" else f"ambiguous_{kind}"
        slot_name = kind  # "major" → "major", "campus" → "campus"
        question = _render_question(reason, options)
        return ClarificationRequest(
            reason=reason,
            question=question,
            options=options,
            slot=slot_name,
            extra={
                "kind": kind,
                "top_score": top_score,
                "scores": [s for (s, _r) in chosen],
            },
        )

    def _check_missing_year(
        self,
        query: str,
        intent: str | None,
        session_year: Any | None,
    ) -> Optional[ClarificationRequest]:
        """Trigger ``missing_year`` for year-required intents lacking a year."""
        if intent not in _YEAR_REQUIRED_INTENTS:
            return None
        if session_year is not None:
            return None
        if _YEAR_RE.search(query or ""):
            return None

        current = datetime.now(timezone.utc).year
        years = [current + off for off in _YEAR_OFFSETS]
        options: list[dict[str, str]] = [
            {"value": str(y), "label": f"Năm {y}"} for y in years
        ]
        question = _render_question("missing_year", options)
        return ClarificationRequest(
            reason="missing_year",
            question=question,
            options=options,
            slot="year",
            extra={"intent": intent, "current_year": current},
        )

    def _check_missing_major(
        self,
        query: str,
        intent: str | None,
        session_major: Any | None,
    ) -> Optional[ClarificationRequest]:
        """Trigger ``missing_major`` when an intent that needs a major has none."""
        if intent not in _MAJOR_REQUIRED_INTENTS:
            return None
        if session_major:
            return None

        # If the query already resolves to at least one major, this is
        # handled either by ``ambiguous_major`` (multiple candidates) or
        # by the downstream answer path (single confident candidate).
        try:
            hits = resolve_all(query or "", entity_types=["major"])
        except Exception as exc:  # noqa: BLE001
            log.debug("clarifier.resolve_all_failed kind=major err=%s", exc)
            hits = {}
        if hits.get("major"):
            return None

        options = _load_popular_majors(self.max_options)
        if not options:
            return None
        question = _render_question("missing_major", options)
        return ClarificationRequest(
            reason="missing_major",
            question=question,
            options=options,
            slot="major",
            extra={"intent": intent},
        )

    def _check_low_recall(
        self,
        retrieval_docs: list[dict[str, Any]] | None,
    ) -> Optional[ClarificationRequest]:
        """Generic clarify when the top retrieval score is below threshold."""
        if not retrieval_docs:
            return None
        top_doc = retrieval_docs[0]
        score = _doc_score(top_doc)
        if score is None:
            return None
        if score >= self.low_recall_threshold:
            return None
        question = _render_question("low_recall", [])
        return ClarificationRequest(
            reason="low_recall",
            question=question,
            options=[],
            slot=None,
            extra={"top_score": score, "threshold": self.low_recall_threshold},
        )

    # ------------------------------------------------------------------ #
    # Small utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mentions_major_noun(query: str) -> bool:
        """Heuristic: does ``query`` contain a major-like noun?"""
        if not query:
            return False
        norm = _normalize_key(query)
        return any(hint in norm for hint in _MAJOR_NOUN_HINTS)


__all__ = [
    "Clarifier",
    "ClarificationRequest",
    "DEFAULT_ENTITY_MIN_SCORE",
    "DEFAULT_AMBIGUITY_GAP",
    "DEFAULT_LOW_RECALL_THRESHOLD",
    "DEFAULT_MAX_OPTIONS",
]
