"""Slot extraction + carry-over logic for the Phase 4B dialogue manager.

The slot filler is responsible for:

1. **Extraction** — pulling new slot values from the *current* user
   query via regex (year) and the fuzzy entity resolver (campus,
   major, faculty, doc_type).
2. **Decay** — dropping any slot whose ``turn`` field is older than
   ``decay_turns`` ago, so the bot doesn't keep filtering by a campus
   the user only mentioned 10 turns earlier.
3. **Reset detection** — recognising explicit "quên đi" / "reset" /
   "ignore previous" intents and wiping the entire slot frame.
4. **Coreference expansion** — see :func:`expand_coreference`. A free
   function used by the rewriter to substitute "ngành đó" with the
   currently-active major display name *before* the rewriter LLM call.

The filler delegates all I/O to
``src.hybridrag.chat.session_state.SessionStateRepo`` so unit tests
can mock the repo without touching Postgres.

Slot frame stored shape (mirrored by :class:`SlotValue`)::

    {
      "campus":   {"value": "co_so_1", "display": "Cơ sở 1", "set_at": ISO, "confidence": 0.95, "turn": 3},
      "major":    {"value": "cong_nghe_thong_tin", "display": "Công nghệ thông tin", ...},
      "year":     {"value": 2024, "display": "2024", ...},
      "faculty":  {...},
      "doc_type": {...}
    }
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.hybridrag.chat.session_state import (
    SessionState,
    SessionStateRepo,
    SlotValue,
)
from src.hybridrag.utils.entity_resolver import resolve_all


# Years before 2000 are almost certainly not admission cohorts; the
# ``20\d{2}`` lower-bound also keeps us from picking up arbitrary 4-digit
# numbers (e.g. tuition amounts) that happen to land in the year range.
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

# Entity types the slot filler maps directly from ``resolve_all`` output.
# Order matters: we record the highest-scoring hit per slot.
_ENTITY_SLOTS: tuple[tuple[str, str], ...] = (
    ("campus", "campus"),
    ("major", "major"),
    ("faculty", "faculty"),
    ("doc_type", "doc_type"),
)


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp for ``SlotValue.set_at`` round-trips."""
    return datetime.now(timezone.utc).isoformat()


def _strip_diacritics(s: str) -> str:
    """Lowercase + accent-strip + đ→d, used only by reset detection.

    We never persist this — it's a one-shot helper so "quên đi" matches
    "quen di" without the synonyms module overhead.
    """
    if not s:
        return ""
    s = s.lower().replace("đ", "d").replace("Đ", "d")
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


class SlotFiller:
    """Extracts slot updates from a user query and merges them into session state.

    Public surface:

    * ``extract_slots(query, *, turn_index)`` — pure function, no I/O.
    * ``update(session_id, query, *, turn_index)`` — read state, extract,
      decay, merge, write back. Returns the new ``SessionState``.
    * ``filters_for_retrieval(state)`` — project the current slot frame
      into a flat ``{slot: value}`` dict for Weaviate / SQL filters.

    Behaviour knobs:

    * ``decay_turns`` — a slot whose ``turn`` is more than this many
      turns behind ``turn_index`` is dropped from the merged result.
      Default ``DEFAULT_DECAY_TURNS = 6``.
    * Reset phrases (Vietnamese + English) live in ``RESET_PHRASES``.
    """

    DEFAULT_DECAY_TURNS: int = 6

    # Lower-cased + accent-stripped phrases that trigger a full slot reset.
    # We match by accent-stripped substring so users with broken IME can
    # still hit "quen di" and get the same behaviour.
    RESET_PHRASES: set[str] = {
        "quên đi",
        "quen di",
        "reset",
        "xóa hết",
        "xoa het",
        "bỏ qua",
        "bo qua",
        "ignore previous",
    }

    def __init__(
        self,
        repo: SessionStateRepo | None = None,
        *,
        decay_turns: int = DEFAULT_DECAY_TURNS,
    ) -> None:
        self.repo = repo or SessionStateRepo()
        # Always >= 1 turn so a value extracted this turn never decays
        # immediately because of a misconfigured 0.
        self.decay_turns = max(1, int(decay_turns))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract_slots(
        self, query: str, *, turn_index: int
    ) -> dict[str, SlotValue]:
        """Extract new slot values from a single user query.

        Pure: no DB I/O. Returns only the slots whose extraction
        succeeded on *this* query — the caller is responsible for
        merging into the previous frame.

        Confidence model:

        * ``year`` regex match → 1.0 (deterministic).
        * Entity resolver hits → 1.0 (the underlying ``resolve_all``
          already filters by ``min_score=80``). The score isn't exposed
          on the return record, so we treat any returned record as
          high-confidence — the dialogue layer can override later.
        """
        if not query or not query.strip():
            return {}

        now_iso = _utc_now_iso()
        out: dict[str, SlotValue] = {}

        # ---- year ----
        year_match = _YEAR_RE.search(query)
        if year_match:
            year_int = int(year_match.group(1))
            out["year"] = SlotValue(
                value=year_int,
                display=str(year_int),
                set_at=now_iso,
                confidence=1.0,
                turn=turn_index,
            )

        # ---- entities (campus / major / faculty / doc_type) ----
        hits = resolve_all(query)
        for etype, slot_name in _ENTITY_SLOTS:
            recs = hits.get(etype) or []
            if not recs:
                continue
            top = recs[0]
            canonical = top.get("canonical")
            if not canonical:
                continue
            out[slot_name] = SlotValue(
                value=canonical,
                display=top.get("display") or canonical,
                set_at=now_iso,
                confidence=1.0,
                turn=turn_index,
            )

        return out

    def update(
        self, session_id: str, query: str, *, turn_index: int
    ) -> SessionState:
        """Read state, extract this-turn slots, merge with decay, write back."""
        state = self.repo.get(session_id)

        # Explicit reset: nuke everything and return an empty frame.
        if self._is_reset(query):
            self.repo.reset(session_id)
            return self.repo.get(session_id)

        new_slots = self.extract_slots(query, turn_index=turn_index)
        decayed = self._decay(state.slots, turn_index=turn_index)
        # New wins on key collision — last extraction is most recent
        # signal from the user.
        merged: dict[str, SlotValue] = {**decayed, **new_slots}

        state.slots = merged
        state.last_query = query
        return self.repo.upsert(state)

    def filters_for_retrieval(self, state: SessionState) -> dict[str, Any]:
        """Project current slots into a flat filter dict.

        Output keys correspond to Weaviate / Postgres filter columns:
        ``campus``, ``year``, ``faculty``, ``major``, ``doc_type``.
        Only slots that have a non-null ``value`` are included so the
        caller can pass the dict straight into a query builder without
        re-checking for empties.
        """
        out: dict[str, Any] = {}
        for name in SessionStateRepo.SUPPORTED_SLOTS:
            sv = state.slots.get(name)
            if sv is None:
                continue
            if sv.value is None:
                continue
            out[name] = sv.value
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _is_reset(self, query: str) -> bool:
        if not query:
            return False
        # Match accent-stripped lowercase so "Quên đi" / "quen di" / "QUEN DI"
        # all hit the same set. Use word-boundary regex so a substring inside
        # an unrelated word (e.g. "quen di chuyen") does not trigger reset.
        normalized = _strip_diacritics(query)
        for phrase in self.RESET_PHRASES:
            pat = r"(?:^|\W)" + re.escape(_strip_diacritics(phrase)) + r"(?:\W|$)"
            if re.search(pat, normalized):
                return True
        return False

    def _decay(
        self, slots: dict[str, SlotValue], *, turn_index: int
    ) -> dict[str, SlotValue]:
        """Drop slots whose ``turn`` field is older than ``decay_turns`` behind.

        Edge case: legacy rows / corrupted JSON may lack a ``turn``
        field (defaults to ``0`` via :meth:`SlotValue.from_jsonable`).
        We treat that as "ancient" and let it decay naturally — this
        also drains stale state automatically after a deploy that adds
        the ``turn`` field.
        """
        out: dict[str, SlotValue] = {}
        for name, sv in slots.items():
            slot_turn = int(sv.turn or 0)
            age = turn_index - slot_turn
            if age <= self.decay_turns:
                out[name] = sv
        return out


# -------------------------------------------------------------------- #
# Coreference expansion
# -------------------------------------------------------------------- #
# Each tuple is ``(compiled regex, slot name)``. The regex is applied
# case-insensitively against the raw user query; on first match we
# substitute the matched span with the slot's ``display`` value
# (preserving the original Vietnamese surface form for the rewriter
# downstream).
#
# Patterns intentionally avoid greedy matches so two coref expressions
# in the same query (e.g. "ngành đó ở đó") each rewrite independently.
COREF_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "ở đó" / "ở kia" / "tại đó" → campus
    (re.compile(r"\b(ở|tại)\s+(đó|kia|đấy)\b", re.IGNORECASE), "campus"),
    # "cơ sở đó" → campus  (must come before generic "ngành đó")
    (re.compile(r"\bcơ\s*sở\s+(đó|này|kia|đấy)\b", re.IGNORECASE), "campus"),
    # "ngành đó/này/kia" → major
    (re.compile(r"\bngành\s+(đó|này|kia|đấy)\b", re.IGNORECASE), "major"),
    # "năm đó/này/kia" → year
    (re.compile(r"\bnăm\s+(đó|này|kia|đấy)\b", re.IGNORECASE), "year"),
    # "khoa đó/này/kia" → faculty
    (re.compile(r"\bkhoa\s+(đó|này|kia|đấy)\b", re.IGNORECASE), "faculty"),
]


def _case_preserving_replacement(prefix: str, display: str):
    """Build a ``re.sub`` callback that mirrors the matched word's casing.

    "Ngành đó" / "ngành đó" both rewrite into their natural form so the
    rewriter LLM doesn't see a randomly-cased sentence. We inspect the
    first letter of the match: upper → capitalize the Vietnamese
    prefix; lower → leave it lower-cased.
    """

    def repl(match: "re.Match[str]") -> str:
        matched = match.group(0)
        leading = next((ch for ch in matched if ch.isalpha()), "")
        if leading and leading.isupper():
            cased_prefix = prefix[:1].upper() + prefix[1:]
        else:
            cased_prefix = prefix
        return f"{cased_prefix} {display}"

    return repl


def expand_coreference(query: str, state: SessionState) -> str:
    """Rewrite implicit coreference (``ở đó``, ``ngành đó``, …) using session slots.

    For each pattern that matches, we look up the corresponding slot in
    ``state.slots``; if the slot has a non-empty ``display`` value we
    substitute. If the slot isn't set the original surface stays
    untouched — the rewriter / clarifier downstream will handle it.

    Case is preserved from the matched surface form: ``"Ngành đó"`` →
    ``"Ngành <Major>"`` while ``"ngành đó"`` → ``"ngành <Major>"``. The
    rewritten sentence stays visually consistent with the user's
    capitalisation.

    Examples::

        state.major.display = "Công nghệ thông tin"
        "Ngành đó học phí bao nhiêu?"
            → "Ngành Công nghệ thông tin học phí bao nhiêu?"

        state.campus.display = "Cơ sở 1"
        "Ngành nào dạy ở đó?"
            → "Ngành nào dạy ở Cơ sở 1?"

    No-state / no-coref: returns the query unchanged.
    """
    if not query:
        return query

    out = query
    for pattern, slot_name in COREF_PATTERNS:
        sv = state.slots.get(slot_name) if state and state.slots else None
        if sv is None or not sv.display:
            continue
        display = sv.display

        if slot_name == "campus":
            # Replace the whole match span ("ở đó" / "cơ sở đó") with
            # the explicit "ở <display>" form. The rewriter expects a
            # natural Vietnamese phrase, not "ở Cơ sở 1 đó".
            out = pattern.sub(_case_preserving_replacement("ở", display), out)
        elif slot_name == "major":
            out = pattern.sub(_case_preserving_replacement("ngành", display), out)
        elif slot_name == "year":
            out = pattern.sub(_case_preserving_replacement("năm", display), out)
        elif slot_name == "faculty":
            out = pattern.sub(_case_preserving_replacement("khoa", display), out)

    return out


__all__ = [
    "SlotFiller",
    "expand_coreference",
    "COREF_PATTERNS",
]
