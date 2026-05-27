"""Phase 5.2 — Post-process answer verifier.

After an LLM produces an answer that contains numeric claims (scores, tuition
amounts, deadlines, years), this module:

1. Extracts every numeric claim from the answer along with its surrounding
   context and the bracket-style citations ``[n]`` attached to it.
2. For each claim, checks whether the cited source chunk actually contains the
   same number (in a separator-tolerant form).
3. Returns a :class:`VerificationReport` enumerating verified / unverified /
   uncited claims with an overall verdict.
4. Optionally annotates the answer with a small Vietnamese warning marker so
   that the caller (chat orchestrator) can decide whether to surface it.

This is a safety net: the LLM is already instructed to cite sources (Phase 5.1),
but verification catches the residual cases where the model invents a number
that does not appear in the cited chunk.

The module deliberately has **no heavy dependencies** (no LangChain, no
OpenAI). Only the standard library + the project's :mod:`src.config.settings`
(loaded lazily, with a literal fallback) are used.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Regexes
# --------------------------------------------------------------------------- #
# Matches a numeric token. Supports:
#   * Integer with thousand separators using ``.`` or ``,`` (e.g. ``1.234.567``,
#     ``1,234,567``).
#   * Integer or decimal with either ``.`` or ``,`` as decimal point
#     (e.g. ``21.5``, ``21,5``).
#   * Bare integers (``2024``, ``42``).
# Word boundaries keep us from matching e.g. ``v2``'s ``2`` from inside an
# identifier, while still allowing the number to be flanked by punctuation.
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.,])                                # left boundary: no word char, no separator
    (?:
        \d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?      # 1.234.567 / 1,234,567 / 1.234.567,89
      | \d+(?:[.,]\d+)?                        # 21 / 21.5 / 21,5
    )
    (?![\w])                                   # right boundary: no word char (digits or letters)
    """,
    re.VERBOSE,
)

# Matches a citation group ``[1]`` / ``[1,2]`` / ``[1, 2]`` / ``[ 1 ]``.
_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")

# A citation is considered "attached" to a number if it begins within this many
# characters after the number's last digit (whitespace counts).
_CITATION_DISTANCE_LIMIT = 30

# Width of the context window captured around each claim (used for debugging
# and human-readable notes).
_CONTEXT_RADIUS = 40


# --------------------------------------------------------------------------- #
# Refusal marker — pulled from settings if available, else literal fallback
# --------------------------------------------------------------------------- #
def _refusal_marker() -> str:
    """Return the leading phrase that flags an answer as a refusal.

    Phase 5.3 will introduce ``settings.REFUSAL_MESSAGE``. Until that lands,
    fall back to the documented Vietnamese phrase so this module is
    self-contained.
    """
    try:
        from src.config.settings import settings as _s
        candidate = getattr(_s, "REFUSAL_MESSAGE", None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    except Exception:  # pragma: no cover — settings always importable in this repo
        pass
    return "Xin lỗi, hiện mình chưa có thông tin chính thức"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class NumericClaim:
    """A single numeric assertion lifted out of the answer text."""

    text: str
    """The original surface form (e.g. ``"21,5"``, ``"1.234.567"``)."""

    value: float
    """The normalized floating-point value."""

    surrounding: str
    """Roughly ``±_CONTEXT_RADIUS`` characters of context around the claim."""

    citation_ids: list[int] = field(default_factory=list)
    """The 1-based citation indices attached to this number (in source order)."""


@dataclass
class ClaimVerdict:
    """The verdict for one :class:`NumericClaim`."""

    claim: NumericClaim
    status: str  # "verified" | "unverified" | "no_citation"
    matched_sources: list[int] = field(default_factory=list)
    note: Optional[str] = None


@dataclass
class VerificationReport:
    """Summary of verifying every numeric claim in an answer."""

    overall: str  # "ok" | "warning" | "refusal"
    claims: list[ClaimVerdict]
    refusal_detected: bool


# --------------------------------------------------------------------------- #
# Number normalization & matching
# --------------------------------------------------------------------------- #
def normalize_number(raw: str) -> float:
    """Parse a numeric token into a ``float`` regardless of locale convention.

    Examples
    --------
    >>> normalize_number("1.234.567")
    1234567.0
    >>> normalize_number("1,234,567.89")
    1234567.89
    >>> normalize_number("21,5")
    21.5
    >>> normalize_number("21.5")
    21.5
    """
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("normalize_number: empty input")

    has_dot = "." in cleaned
    has_comma = "," in cleaned

    if has_dot and has_comma:
        # Both present — the *last* one is the decimal separator.
        last_dot = cleaned.rfind(".")
        last_comma = cleaned.rfind(",")
        if last_dot > last_comma:
            # "1,234,567.89" — comma = thousand, dot = decimal
            cleaned = cleaned.replace(",", "")
        else:
            # "1.234.567,89" — dot = thousand, comma = decimal
            cleaned = cleaned.replace(".", "").replace(",", ".")
    elif has_comma:
        parts = cleaned.split(",")
        # Decimal comma if there's exactly one comma AND the part after it is
        # 1–2 digits (typical European decimal). Otherwise treat as thousands.
        if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif has_dot:
        parts = cleaned.split(".")
        # Multiple dots OR groups of exactly 3 digits after the first dot → it's
        # a thousand separator. A single ``.`` followed by ≤2 digits is a
        # decimal point.
        if len(parts) > 2 or (
            len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3
        ):
            # e.g. "1.234.567" or "1.234" → 1234567 / 1234
            cleaned = cleaned.replace(".", "")
        # else: leave as-is — "21.5" parses fine.

    return float(cleaned)


def _canonical_digits(value: float, surface: str) -> set[str]:
    """Return canonical string forms suitable for substring lookup in a doc.

    We generate several representations so that ``"21,5"`` in the answer can
    match ``"21.5"`` in the doc and vice-versa. Trailing zeros after the
    decimal point are stripped so ``21.50`` and ``21.5`` are equivalent.
    """
    out: set[str] = set()

    # Always include the original surface form (lets us match exotic shapes
    # like ``1.234.567``).
    out.add(surface.strip())

    # Integer-valued floats — render as plain integer.
    if value.is_integer():
        intval = int(value)
        out.add(str(intval))
        # Thousand-separator variants for large integers.
        s = str(intval)
        if len(s) > 3:
            grouped_dot = _group_thousands(s, sep=".")
            grouped_comma = _group_thousands(s, sep=",")
            out.add(grouped_dot)
            out.add(grouped_comma)
        return out

    # Non-integer: emit dot-decimal and comma-decimal, stripped of trailing 0s.
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." in text:
        int_part, dec_part = text.split(".", 1)
        out.add(f"{int_part}.{dec_part}")
        out.add(f"{int_part},{dec_part}")
        if len(int_part) > 3:
            out.add(f"{_group_thousands(int_part, sep='.')},{dec_part}")
            out.add(f"{_group_thousands(int_part, sep=',')}.{dec_part}")
    else:
        out.add(text)
    return out


def _group_thousands(int_str: str, *, sep: str) -> str:
    """Insert thousand separators into a string of digits."""
    if int_str.startswith("-"):
        return "-" + _group_thousands(int_str[1:], sep=sep)
    n = len(int_str)
    if n <= 3:
        return int_str
    head = n % 3
    parts: list[str] = []
    if head:
        parts.append(int_str[:head])
    parts.extend(int_str[i : i + 3] for i in range(head, n, 3))
    return sep.join(parts)


# --------------------------------------------------------------------------- #
# Claim extraction
# --------------------------------------------------------------------------- #
def _collect_citations_after(answer: str, start: int) -> tuple[list[int], int]:
    """Collect citation ids attached to the number ending at ``start``.

    Spec rules:
      * A citation is "attached" if it begins within
        ``_CITATION_DISTANCE_LIMIT`` chars after the number's last digit.
      * Adjacent brackets such as ``[1][2]`` count as one logical group.
      * Between the number and the first bracket — and between consecutive
        brackets in the group — only whitespace and punctuation are allowed.
        A letter or digit in that gap terminates the group.

    Returns ``(ids, last_match_end)``. ``last_match_end`` is the offset just
    past the final consumed citation (= ``start`` if no citations matched), so
    the outer loop can advance past anything it already attributed.
    """
    ids: list[int] = []
    cursor = start
    last_end = start
    window_end = min(len(answer), start + _CITATION_DISTANCE_LIMIT)
    while True:
        m = _CITATION_RE.search(answer, cursor, window_end)
        if not m:
            break
        # The first bracket must start within the original 30-char window;
        # `re.search` already enforces that by construction. For subsequent
        # brackets we additionally require the gap to be whitespace/punct only.
        gap = answer[cursor : m.start()]
        if gap and re.search(r"[A-Za-z0-9]", gap):
            break
        for token in m.group(1).split(","):
            token = token.strip()
            if token.isdigit():
                ids.append(int(token))
        last_end = m.end()
        cursor = m.end()
    # Dedupe while preserving order
    seen: set[int] = set()
    deduped: list[int] = []
    for i in ids:
        if i not in seen:
            deduped.append(i)
            seen.add(i)
    return deduped, last_end


def _is_bare_year(raw: str, value: float) -> bool:
    """Return ``True`` for a 4-digit integer in the plausible-year range.

    Bare years that lack a citation are very common in narrative text
    (``"năm 2024"``) and would otherwise drown the report in
    ``no_citation`` warnings. Cited years are still extracted — the caller
    only filters via this helper for uncited tokens.
    """
    if "." in raw or "," in raw:
        return False
    if not raw.isdigit() or len(raw) != 4:
        return False
    return 1900.0 <= value <= 2100.0


def extract_numeric_claims(answer: str) -> list[NumericClaim]:
    """Find every numeric token in ``answer`` and capture its citation ids.

    The matcher consumes the answer left-to-right. After each number is found,
    we look at the next ``_CITATION_DISTANCE_LIMIT`` characters for citation
    brackets and treat them as belonging to that number. The scan then resumes
    *after* the consumed citations, so a citation cannot be assigned to two
    different numbers.

    Bare 4-digit years in the 1900-2100 range with no attached citation are
    treated as narrative context (e.g. ``"năm 2024"``) and skipped to keep the
    report focused on the numbers a human would actually want to check.
    """
    if not isinstance(answer, str) or not answer:
        return []

    claims: list[NumericClaim] = []
    cursor = 0
    while cursor < len(answer):
        m = _NUMBER_RE.search(answer, cursor)
        if not m:
            break
        raw = m.group(0)
        try:
            value = normalize_number(raw)
        except ValueError:
            cursor = m.end()
            continue

        ids, consumed_end = _collect_citations_after(answer, m.end())

        if not ids and _is_bare_year(raw, value):
            # Skip the narrative-year false positive but still advance so we
            # don't re-match it on the next iteration.
            cursor = m.end()
            continue

        left = max(0, m.start() - _CONTEXT_RADIUS)
        right = min(len(answer), m.end() + _CONTEXT_RADIUS)
        surrounding = answer[left:right].strip()

        claims.append(
            NumericClaim(
                text=raw,
                value=value,
                surrounding=surrounding,
                citation_ids=ids,
            )
        )
        cursor = consumed_end if consumed_end > m.end() else m.end()
    return claims


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _doc_content(doc: dict[str, Any]) -> str:
    """Pull the text body from a retrieval doc. Mirrors ``AnswerGenerator``'s rules."""
    if not isinstance(doc, dict):
        return ""
    for key in ("content", "text", "chunk", "document"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _doc_contains_claim(doc_text: str, claim: NumericClaim) -> bool:
    """Check whether ``doc_text`` contains the numeric value of ``claim``.

    Strategy: build a small set of canonical representations of the claim's
    value and look for any of them as a substring of the doc. Also detect any
    numeric token inside the doc and compare its parsed value.
    """
    if not doc_text:
        return False

    # Cheap pass — direct substring with multiple separator variants.
    candidates = _canonical_digits(claim.value, claim.text)
    for cand in candidates:
        if cand and cand in doc_text:
            return True

    # Expensive pass — parse every number in the doc and compare values.
    for m in _NUMBER_RE.finditer(doc_text):
        try:
            v = normalize_number(m.group(0))
        except ValueError:
            continue
        if _values_match(v, claim.value):
            return True
    return False


def _values_match(a: float, b: float) -> bool:
    """Two numeric values match if they are equal up to a small tolerance.

    Tolerance is the smaller of:
      * absolute 1e-6, or
      * 0.01 % of the larger magnitude (handles float-precision artefacts on
        large integers).
    """
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= max(1e-6, scale * 1e-4)


def verify_answer(
    answer: str,
    retrieved_docs: list[dict[str, Any]],
) -> VerificationReport:
    """Verify every numeric claim in ``answer`` against ``retrieved_docs``.

    ``retrieved_docs`` must be in citation order — ``[1]`` refers to
    ``retrieved_docs[0]``, ``[2]`` to ``retrieved_docs[1]`` and so on.
    The convention mirrors ``AnswerGenerator.build_context``.

    Returns
    -------
    VerificationReport
        Aggregated verdict per claim plus an overall verdict.
    """
    refusal_detected = False
    marker = _refusal_marker()
    if isinstance(answer, str) and marker and answer.lstrip().startswith(marker):
        refusal_detected = True

    if refusal_detected:
        return VerificationReport(
            overall="refusal",
            claims=[],
            refusal_detected=True,
        )

    if not isinstance(answer, str):
        answer = "" if answer is None else str(answer)

    claims = extract_numeric_claims(answer)
    if not claims:
        return VerificationReport(
            overall="ok",
            claims=[],
            refusal_detected=False,
        )

    docs = retrieved_docs or []
    verdicts: list[ClaimVerdict] = []
    any_unverified = False

    for claim in claims:
        if not claim.citation_ids:
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    status="no_citation",
                    note="Không có trích dẫn đi kèm số liệu.",
                )
            )
            any_unverified = True
            continue

        matched: list[int] = []
        out_of_range: list[int] = []
        for cid in claim.citation_ids:
            idx = cid - 1
            if idx < 0 or idx >= len(docs):
                out_of_range.append(cid)
                continue
            doc_text = _doc_content(docs[idx])
            if _doc_contains_claim(doc_text, claim):
                matched.append(cid)

        if matched:
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    status="verified",
                    matched_sources=matched,
                )
            )
        else:
            note_parts: list[str] = []
            if out_of_range:
                note_parts.append(
                    "Trích dẫn vượt ngoài số tài liệu hiện có: "
                    + ", ".join(f"[{c}]" for c in out_of_range)
                )
            else:
                note_parts.append(
                    "Không tìm thấy số liệu trong tài liệu được trích dẫn."
                )
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    status="unverified",
                    note="; ".join(note_parts),
                )
            )
            any_unverified = True

    overall = "warning" if any_unverified else "ok"
    return VerificationReport(
        overall=overall,
        claims=verdicts,
        refusal_detected=False,
    )


# --------------------------------------------------------------------------- #
# Answer annotation
# --------------------------------------------------------------------------- #
def annotate_answer(answer: str, report: VerificationReport) -> str:
    """Append a Vietnamese warning marker if any claim in ``report`` is unverified.

    Returns the answer unchanged when ``report.overall`` is ``"ok"`` or
    ``"refusal"``.

    The annotation is inserted BEFORE the trailing
    ``[Thông tin tham chiếu]`` block (if present), so the frontend's
    reference parser still sees a clean block ending the message and
    can render the clickable source-download buttons. The verifier
    warning ends up as part of the body text, which is exactly what
    the user should see.
    """
    if not isinstance(answer, str):
        return answer
    if report.overall != "warning":
        return answer

    unverified_texts: list[str] = []
    seen: set[str] = set()
    for v in report.claims:
        if v.status in ("unverified", "no_citation"):
            t = v.claim.text
            if t not in seen:
                unverified_texts.append(t)
                seen.add(t)
    if not unverified_texts:
        return answer

    joined = ", ".join(unverified_texts)
    note = f"\n\n_Lưu ý: một số con số chưa được kiểm chứng tự động: {joined}._"

    # Locate the trailing reference heading. Match both diacritic
    # variants ("Thông tin tham chiếu" / "Thong tin tham chieu") so a
    # diacritic-stripped LLM emit still parses correctly. The heading
    # MUST sit on its own line for our split to work.
    heading_re = re.compile(
        r"\n\s*\[\s*Th[ôo]ng\s+tin\s+tham\s+chi[ếe]u\s*\]\s*\n",
        flags=re.IGNORECASE,
    )
    m = heading_re.search(answer)
    if m is None:
        return answer + note
    # Splice the warning between the body and the reference block.
    return answer[: m.start()] + note + answer[m.start():]


__all__ = [
    "NumericClaim",
    "ClaimVerdict",
    "VerificationReport",
    "normalize_number",
    "extract_numeric_claims",
    "verify_answer",
    "annotate_answer",
]
