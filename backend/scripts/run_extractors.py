"""Populate KG (admission_scores) from converted markdown score files.

Reads ``data/samples/Điểm 20*.md`` (now in markdown pipe format), splits
each file into per-method sections (THPT / học bạ / ĐGNL) and passes
each section to the LLM extractor with an explicit method hint. Without
the split the model tends to drop the "Xét điểm học bạ" section when a
single file contains three tables, so we drive each table separately.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

# Allow ``python scripts/run_extractors.py`` from backend/ root.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from src.hybridrag.ingestion.extractors.scores_extractor import (  # noqa: E402
    extract_scores_from_text,
)
from src.hybridrag.ingestion.extractors.tuition_extractor import (  # noqa: E402
    extract_tuition_from_text,
)
from src.hybridrag.kg.scores_repo import AdmissionScoresRepo  # noqa: E402
from src.hybridrag.kg.tuition_repo import TuitionRepo  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
log = logging.getLogger("run_extractors")


SCORE_FILES = [
    BACKEND_ROOT / "data" / "samples" / "Điểm 2023.md",
    BACKEND_ROOT / "data" / "samples" / "Điểm 2024.md",
    BACKEND_ROOT / "data" / "samples" / "Điểm 2025.md",
]


# Files known to mention "học phí" — fed to the tuition extractor.
# Anything that doesn't contain a real tuition figure simply returns
# ``rows=[]`` and is a no-op (we still pay one LLM call per file).
TUITION_FILES = [
    BACKEND_ROOT / "data" / "samples" / "tuyen_sinh_247.md",
    BACKEND_ROOT / "data" / "samples" / "qa_fb.md",
]


# Map H2 section heading patterns → method code that the extractor's
# SYSTEM_PROMPT whitelists. The hint is prepended to the section text
# so the model emits the right ``method`` field even when the heading
# doesn't spell it out (e.g. "Xét điểm học bạ" → "XHCT").
_METHOD_HINT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"thi\s*tốt\s*nghi[eệ]p", re.IGNORECASE), "THPT"),
    (re.compile(r"học\s*bạ", re.IGNORECASE), "XHCT"),
    (re.compile(r"ĐGNL|ĐGTD|đánh\s*giá\s*năng\s*lực", re.IGNORECASE), "HSA"),
]


def _split_h2_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown by H2 headings. Returns ``[(heading, body), ...]``.

    Body includes the heading line. Pre-H2 prose is dropped (just the
    H1 title in these files).
    """
    lines = text.splitlines(keepends=True)
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_buf: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "".join(current_buf)))
            current_heading = line.strip().lstrip("#").strip()
            current_buf = [line]
        else:
            if current_heading is not None:
                current_buf.append(line)
    if current_heading is not None:
        sections.append((current_heading, "".join(current_buf)))
    return sections


def _method_for_heading(heading: str) -> str | None:
    for pattern, method in _METHOD_HINT_PATTERNS:
        if pattern.search(heading):
            return method
    return None


def _section_with_method_hint(body: str, method: str) -> str:
    """Prepend a one-liner that pins the ``method`` field for this table."""
    hint = (
        f"NOTE FOR EXTRACTOR: every row in this section uses method=\"{method}\". "
        f"Set method=\"{method}\" on every row you emit from this section.\n\n"
    )
    return hint + body


def populate_scores() -> dict[str, int]:
    """Extract + upsert each score file. Returns ``{source_file: row_count}``."""
    repo = AdmissionScoresRepo()
    counts: dict[str, int] = {}

    for path in SCORE_FILES:
        if not path.exists():
            log.warning("file not found: %s", path)
            continue

        source_file = path.name
        log.info("processing %s ...", source_file)

        text = path.read_text(encoding="utf-8")

        # Idempotent: drop existing rows attributed to this file so re-run
        # doesn't accumulate duplicates (UNIQUE constraint would prevent
        # exact duplicates but score edits would leave stale rows).
        deleted = repo.delete_by_source(source_file)
        if deleted:
            log.info("  deleted %d existing rows for %s", deleted, source_file)

        upserted = 0
        sections = _split_h2_sections(text)
        if not sections:
            log.warning("  no H2 sections found in %s; skipping", source_file)
            counts[source_file] = 0
            continue

        for heading, body in sections:
            method = _method_for_heading(heading)
            if method is None:
                log.warning("  heading %r has no method mapping; skipping", heading)
                continue

            hinted_body = _section_with_method_hint(body, method)
            rows = extract_scores_from_text(hinted_body, source_file=source_file)
            if not rows:
                log.warning("  no rows extracted from section %r", heading)
                continue

            section_upserted = 0
            for row in rows:
                # Defensive: enforce the section's method even if the model
                # ignored the hint.
                if not row.get("method"):
                    row["method"] = method
                try:
                    repo.upsert(**row)
                    section_upserted += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning("    upsert failed for %r: %s", row, exc)
            log.info(
                "  section=%r method=%s extracted=%d upserted=%d",
                heading,
                method,
                len(rows),
                section_upserted,
            )
            upserted += section_upserted

        counts[source_file] = upserted

    return counts


def populate_tuition() -> dict[str, int]:
    """Extract + upsert tuition rows from each known-to-mention-tuition file."""
    repo = TuitionRepo()
    counts: dict[str, int] = {}

    for path in TUITION_FILES:
        if not path.exists():
            log.warning("file not found: %s", path)
            continue

        source_file = path.name
        log.info("tuition: processing %s ...", source_file)

        text = path.read_text(encoding="utf-8")

        # Idempotent: drop existing rows attributed to this source so a
        # re-run picks up any edits without duplicates.
        deleted = repo.delete_by_source(source_file)
        if deleted:
            log.info("  deleted %d existing tuition rows for %s", deleted, source_file)

        rows = extract_tuition_from_text(text, source_file=source_file)
        if not rows:
            log.warning("  no tuition rows extracted from %s", source_file)
            counts[source_file] = 0
            continue

        upserted = 0
        for row in rows:
            try:
                repo.upsert(**row)
                upserted += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("  tuition upsert failed for %r: %s", row, exc)
        log.info("  extracted=%d upserted=%d", len(rows), upserted)
        counts[source_file] = upserted

    return counts


def main() -> int:
    log.info("=== Populating admission_scores ===")
    score_counts = populate_scores()
    log.info("=== Populating tuition ===")
    tuition_counts = populate_tuition()

    score_total = sum(score_counts.values())
    tuition_total = sum(tuition_counts.values())
    log.info("=" * 60)
    log.info("Scores: %d rows", score_total)
    for src, n in score_counts.items():
        log.info("  %s: %d", src, n)
    log.info("Tuition: %d rows", tuition_total)
    for src, n in tuition_counts.items():
        log.info("  %s: %d", src, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
