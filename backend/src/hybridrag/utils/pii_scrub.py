"""Phase 5.9 — PII scrubber for logs and observability metadata.

Defense-in-depth: chat content is still stored verbatim in the primary
database, but any field that is forwarded to OBSERVABILITY (structured logs,
tracing spans, Prometheus labels) goes through this module first. The goal is
to redact Vietnamese-specific PII surfaces:

  * Phone numbers — mobile prefixes ``09xx``, ``08xx``, ``07xx``, ``05xx``,
    ``03xx`` and the international ``+84`` form, with optional spaces / dots /
    dashes inside.
  * National-ID numbers — 9-digit CMND and 12-digit CCCD.
  * Email addresses.
  * Bank-card numbers — loose 13-19 digit blocks.

Implementation notes
--------------------
Order matters. Phone digits can overlap with CMND digits, and card digits with
both. The :func:`scrub` function applies the regexes in this order:

    1. emails  (so the local-part doesn't leak into the phone matcher)
    2. CCCD / CMND (12- or 9-digit blocks)
    3. phones  (Vietnamese pattern, may include separators)
    4. cards   (very loose; over-matches on purpose)

The regexes are compiled once at import time and the module deliberately has
no heavy dependencies.
"""
from __future__ import annotations

import re
from typing import Optional


# --------------------------------------------------------------------------- #
# Compiled regexes
# --------------------------------------------------------------------------- #

# Email — RFC-ish but pragmatic. We accept the common local-part chars and
# require at least one dot in the domain. Word boundary on both ends keeps us
# from latching onto fragments.
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])"
    r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
    r"(?![\w.-])"
)

# CCCD (12 digits) or CMND (9 digits). The negative lookarounds ensure we
# don't peel just the first 9 digits off a 12-digit string or vice versa, and
# that we don't catch a number embedded inside a longer digit sequence
# (cards etc., which are handled later).
_ID_RE = re.compile(r"(?<!\d)(?:\d{12}|\d{9})(?!\d)")

# Vietnamese phone numbers. Mobile prefixes after the country/access code:
#   09x, 08x, 07x, 05x, 03x  (Viettel/MobiFone/Vinaphone/Vietnamobile/Gmobile)
# Country code may be ``+84`` or ``84`` (without leading 0), or just ``0``.
# Between digits we accept zero or one of space, dot, dash. We require the
# total digit count to be 10 (national, starting 0) or 11 (when starting +84
# the rest is 9 digits — so total digit-glyph count matches).
_PHONE_RE = re.compile(
    r"(?<![\w])"
    r"(?:\+?84[\s.\-]?|0)"
    r"(?:3|5|7|8|9)"             # mobile prefix
    r"(?:[\s.\-]?\d){8}"         # 8 more digits separated by optional sep
    r"(?!\d)"
)

# Card-like 13-19 digit blocks, possibly grouped with spaces/dashes (e.g. visa
# ``4111 1111 1111 1111`` or ``4111-1111-1111-1111``). The first and last
# characters must be digits.
_CARD_RE = re.compile(
    r"(?<![\w])"
    r"\d(?:[ \-]?\d){12,18}"
    r"(?![\w])"
)


# Regexes used in detect() — kept identical to the scrubbing versions.
_KIND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("emails", _EMAIL_RE),
    ("ids", _ID_RE),
    ("phones", _PHONE_RE),
    ("cards", _CARD_RE),
]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def scrub(text: str, *, redact: str = "***") -> str:
    """Redact PII tokens in ``text`` and return the result.

    Parameters
    ----------
    text:
        Free-form text that may contain Vietnamese phone numbers, ID card
        numbers, emails, or bank-card numbers. Non-string input is returned
        unchanged.
    redact:
        Placeholder substituted in place of each PII match. Defaults to
        ``"***"``.
    """
    if not isinstance(text, str) or not text:
        return text

    # Order matters — emails first (otherwise the ``@`` is fine but the
    # local-part / domain digits could be eaten by later passes). Then
    # CCCD/CMND, then phones, then loose cards.
    out = _EMAIL_RE.sub(redact, text)
    out = _ID_RE.sub(redact, out)
    out = _PHONE_RE.sub(redact, out)
    out = _CARD_RE.sub(redact, out)
    return out


def detect(text: str) -> dict[str, list[str]]:
    """Return the PII tokens found in ``text``, grouped by kind.

    The returned dict always contains the four kinds ``emails``, ``ids``,
    ``phones`` and ``cards`` — empty lists when nothing matches. This is
    convenient for logging *what* kind of PII a user submitted without logging
    the value itself (count + kind only).

    The text is mutated between passes (kind-by-kind redactions) so that the
    same span is not reported under two categories — e.g. a CCCD is not also
    reported as a phone or card.
    """
    found: dict[str, list[str]] = {"emails": [], "ids": [], "phones": [], "cards": []}
    if not isinstance(text, str) or not text:
        return found

    working = text
    for kind, pattern in _KIND_PATTERNS:
        matches = pattern.findall(working)
        if matches:
            found[kind].extend(matches)
            working = pattern.sub(" ", working)
    return found


def scrub_dict(
    data: dict,
    *,
    keys: Optional[set[str]] = None,
    redact: str = "***",
) -> dict:
    """Recursively scrub every string value in ``data``.

    The returned dictionary mirrors the structure of ``data`` (dicts inside
    dicts, lists inside lists) but with string leaves passed through
    :func:`scrub`. Non-string / non-container values are passed through
    untouched.

    Parameters
    ----------
    data:
        The mapping to scrub. Original is not mutated.
    keys:
        Optional set of key names to restrict scrubbing to. Matching is
        case-insensitive. When ``None`` (default) every string value is
        scrubbed.
    redact:
        Placeholder for matched PII tokens. Forwarded to :func:`scrub`.
    """
    if not isinstance(data, dict):
        return data  # defensive: callers may pass through plain values

    lowered_keys: Optional[set[str]] = (
        {k.lower() for k in keys} if keys is not None else None
    )

    def _walk(value: object, *, scrub_strings: bool) -> object:
        if isinstance(value, dict):
            return {
                k: _walk(
                    v,
                    scrub_strings=_should_scrub(k, scrub_strings),
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [_walk(item, scrub_strings=scrub_strings) for item in value]
        if isinstance(value, tuple):
            return tuple(_walk(item, scrub_strings=scrub_strings) for item in value)
        if isinstance(value, str) and scrub_strings:
            return scrub(value, redact=redact)
        return value

    def _should_scrub(key: object, parent_scrub: bool) -> bool:
        """Decide whether the children of ``key`` should be scrubbed.

        When a key filter is set, scrubbing is enabled for any value whose key
        path includes a whitelisted key. We track this via the ``parent_scrub``
        flag so nested dicts under e.g. ``"msg"`` still get cleaned.
        """
        if lowered_keys is None:
            return True
        if parent_scrub:
            return True
        if isinstance(key, str) and key.lower() in lowered_keys:
            return True
        return False

    return {
        k: _walk(v, scrub_strings=_should_scrub(k, parent_scrub=False))
        for k, v in data.items()
    }


__all__ = ["scrub", "detect", "scrub_dict"]
