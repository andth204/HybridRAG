"""Defense-in-depth helpers for prompt-injection mitigation (Phase 5.4).

The functions in this module sanitize untrusted text BEFORE it is embedded
into an LLM prompt. They are intentionally conservative: we do NOT censor
benign user content, we only:

1. Trim/normalize obvious prompt-protocol scaffolds (chat templates,
   instruction tokens).
2. Strip invisible / zero-width characters that could smuggle hidden
   directives past a human reviewer.
3. Escape ``<`` / ``>`` so user text cannot close the XML-style trust
   boundary tags our prompt templates use (``<user_question>``,
   ``<context_doc>``, ``<history>``, ...).
4. Truncate over-long inputs to ``PROMPT_INPUT_MAX_CHARS`` to bound the
   attack surface and protect against context-length exhaustion.

The companion system rule (``QUY TẮC AN TOÀN``) in the RAG / rewriter
prompts instructs the model to treat the wrapped blocks as DATA, never
as instructions. Sanitization here is the second line of defense.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Final

from src.config.settings import settings

log = logging.getLogger(__name__)

# Zero-width / formatting characters used in homoglyph or hidden-prompt
# attacks. Stripped outright.
_ZERO_WIDTH_CHARS: Final[str] = "​‌‍﻿"
_ZERO_WIDTH_RE: Final[re.Pattern[str]] = re.compile(f"[{_ZERO_WIDTH_CHARS}]")

# Chat-template / instruction tokens shipped by various base models.
# Sanitizing these prevents a user from impersonating a higher-trust
# role inside the prompt body.
_INSTRUCTION_TOKENS: Final[tuple[str, ...]] = (
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "</s>",
    "<\\s>",
    "<s>",
)

# Heuristic single-line injection scaffolds. A line that STARTS with one
# of these is collapsed to an inert marker so the model still sees
# something (preserving sentence structure) but the imperative verb is
# defanged. We match case-insensitive and after any leading punctuation.
_INJECTION_LINE_PREFIXES: Final[tuple[str, ...]] = (
    "system:",
    "assistant:",
    "user:",
    "ignore previous",
    "ignore all previous",
    "ignore the above",
    "ignore the previous",
    "disregard previous",
    "disregard all previous",
    "disregard the above",
    "forget previous",
    "forget the above",
    "jailbreak",
    "dan mode",
    "act as dan",
    "you are dan",
    "bo qua chi thi",
    "bo qua huong dan",
    "bo qua quy tac",
    "quen chi thi",
)

_LEADING_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"^[\s\-\*\>\#\|\.\)\(\[\]\:]+")


def _default_max_chars() -> int:
    raw = getattr(settings, "PROMPT_INPUT_MAX_CHARS", 4000)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 4000
    return value if value > 0 else 4000


def _strip_instruction_tokens(text: str) -> str:
    """Remove special chat-template tokens regardless of position."""
    for token in _INSTRUCTION_TOKENS:
        if token in text:
            text = text.replace(token, " ")
    return text


def _neutralize_injection_lines(text: str) -> str:
    """Collapse imperative injection prefixes on a per-line basis.

    We do NOT delete the line — the prompt's safety preamble + XML
    wrapping is the primary defense. We just strip the leading
    imperative so a single regex-style filter on the model side can
    short-circuit cleanly. Bodies are kept so semantics survive.
    """
    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = _LEADING_PUNCT_RE.sub("", line).lower()
        for prefix in _INJECTION_LINE_PREFIXES:
            if stripped.startswith(prefix):
                # Replace just the matched prefix (case-insensitive) with
                # a neutral placeholder so the model can SEE that
                # something was scrubbed.
                pattern = re.compile(re.escape(prefix), re.IGNORECASE)
                line = pattern.sub("[scrubbed]", line, count=1)
                break
        out_lines.append(line)
    return "\n".join(out_lines)


def _escape_angle_brackets(text: str) -> str:
    """Escape ``<`` / ``>`` to HTML entities.

    The RAG / rewriter prompts wrap untrusted content in XML-style trust
    boundary tags (``<user_question>``, ``<context_doc>``, ``<history>``).
    If user content contained a literal ``</user_question>`` it could
    close the boundary from inside and inject directives that the model
    would then treat as system instructions.
    """
    return text.replace("<", "&lt;").replace(">", "&gt;")


def sanitize_user_text(text: str | None) -> str:
    """Sanitize a single untrusted text blob for safe prompt embedding.

    Steps (in order):
        1. Return early on empty input.
        2. Normalize unicode to NFC (folds combining-mark variants).
        3. Strip zero-width / BOM characters.
        4. Strip chat-template instruction tokens.
        5. Truncate to ``PROMPT_INPUT_MAX_CHARS`` (default 4000); logs a
           warning if anything was truncated.
        6. Neutralize per-line injection imperatives (``ignore previous``,
           ``system:``, ``jailbreak``, ...).
        7. Escape ``<`` / ``>`` so user text cannot close the wrapping
           trust-boundary XML tags.
    """
    if not text:
        return ""

    cleaned = unicodedata.normalize("NFC", text)
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = _strip_instruction_tokens(cleaned)

    max_chars = _default_max_chars()
    if len(cleaned) > max_chars:
        log.warning(
            "sanitize_user_text: truncating input from %d to %d chars",
            len(cleaned),
            max_chars,
        )
        cleaned = cleaned[:max_chars]

    cleaned = _neutralize_injection_lines(cleaned)
    cleaned = _escape_angle_brackets(cleaned)
    return cleaned


__all__ = ["sanitize_user_text"]
