"""Walk a directory of markdown / text sources and populate the KG tables.

Workflow per file:

1.  Pick which extractor(s) to run based on filename heuristics:
    * filename has a year token (20XX) OR "điểm" / "diem" → score extractor.
    * filename has "học phí" / "hoc phi" / "tuition" → tuition extractor.
    * neither matched → run BOTH (safer than skipping new sources).

2.  Read the file. If it's larger than ``--max-size-kb`` (default 200KB),
    truncate to ``--truncate-kb`` (default 100KB) and log a warning.
    Truncation was chosen over recursive chunking because the score /
    tuition tables in our corpus all live near the top of the source
    file (within the first 30-50KB); the structural cost of a chunk
    loop isn't justified.

3.  Hand the text to the relevant extractor, then upsert each row via
    the matching repo. ``--dry-run`` skips the upsert and just prints
    what would happen.

4.  Track per-file error state. Exit code is non-zero if any file
    errored; we still keep processing the rest so a single bad source
    doesn't stop the world.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

from src.hybridrag.ingestion.extractors.scores_extractor import (
    extract_scores_from_text,
)
from src.hybridrag.ingestion.extractors.tuition_extractor import (
    extract_tuition_from_text,
)
from src.hybridrag.kg.scores_repo import AdmissionScoresRepo
from src.hybridrag.kg.tuition_repo import TuitionRepo


log = logging.getLogger("extract_kg")


# Pattern shared with the metadata extractor: a 4-digit 20XX year.
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# Filename heuristics for choosing extractors. We accept diacritic-free
# variants too because the corpus has both ``Điểm 2024.md`` and the
# ASCII-folded form that VFS imports often produce.
_SCORE_FILENAME_PATTERN = re.compile(r"đi[eể]m|diem|score", re.IGNORECASE)
_TUITION_FILENAME_PATTERN = re.compile(
    r"học[\s_\-]*phí|hoc[\s_\-]*phi|tuition", re.IGNORECASE,
)


def _strip_diacritics(s: str) -> str:
    """Best-effort drop of Vietnamese diacritics for filename heuristics."""
    s = s.replace("đ", "d").replace("Đ", "d")
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _pick_extractors(filename: str, only: str) -> tuple[bool, bool]:
    """Return ``(run_score, run_tuition)`` flags for a given filename."""
    if only == "score":
        return True, False
    if only == "tuition":
        return False, True

    name = filename
    name_ascii = _strip_diacritics(filename)
    has_year = bool(YEAR_PATTERN.search(name))
    score_hit = (
        has_year
        or _SCORE_FILENAME_PATTERN.search(name) is not None
        or _SCORE_FILENAME_PATTERN.search(name_ascii) is not None
    )
    tuition_hit = (
        _TUITION_FILENAME_PATTERN.search(name) is not None
        or _TUITION_FILENAME_PATTERN.search(name_ascii) is not None
    )

    if not score_hit and not tuition_hit:
        return True, True  # Default: try both.
    return score_hit, tuition_hit


def _iter_sources(source_dir: Path) -> Iterable[Path]:
    """Yield every ``*.md`` / ``*.txt`` file under ``source_dir`` (sorted)."""
    if not source_dir.exists():
        return []
    return sorted(
        p for p in source_dir.iterdir() if p.suffix.lower() in {".md", ".txt"}
    )


def _read_text(
    path: Path, *, max_size_kb: int, truncate_kb: int,
) -> str:
    """Read a file, applying the truncation policy described in the module docstring."""
    size = path.stat().st_size
    text = path.read_text(encoding="utf-8")
    if size > max_size_kb * 1024:
        truncate_chars = truncate_kb * 1024
        log.warning(
            "%s: %d bytes > %dKB; truncating to first %dKB",
            path.name, size, max_size_kb, truncate_kb,
        )
        return text[:truncate_chars]
    return text


def _process_file(
    path: Path,
    *,
    only: str,
    dry_run: bool,
    scores_repo: AdmissionScoresRepo | None,
    tuition_repo: TuitionRepo | None,
    max_size_kb: int,
    truncate_kb: int,
) -> tuple[int, int, bool]:
    """Run the chosen extractors on a single file.

    Returns ``(score_rows, tuition_rows, errored)``.
    """
    score_rows = 0
    tuition_rows = 0
    errored = False

    try:
        text = _read_text(path, max_size_kb=max_size_kb, truncate_kb=truncate_kb)
    except Exception as exc:  # noqa: BLE001 — file IO can fail in many ways
        log.error("%s: failed to read: %r", path.name, exc)
        return 0, 0, True

    run_score, run_tuition = _pick_extractors(path.name, only)

    if run_score:
        try:
            rows = extract_scores_from_text(text, source_file=path.name)
            score_rows = len(rows)
            if not dry_run and scores_repo is not None:
                for row in rows:
                    scores_repo.upsert(**row)
        except Exception as exc:  # noqa: BLE001
            log.error("%s: score extractor failed: %r", path.name, exc)
            errored = True

    if run_tuition:
        try:
            rows = extract_tuition_from_text(text, source_file=path.name)
            tuition_rows = len(rows)
            if not dry_run and tuition_repo is not None:
                for row in rows:
                    tuition_repo.upsert(**row)
        except Exception as exc:  # noqa: BLE001
            log.error("%s: tuition extractor failed: %r", path.name, exc)
            errored = True

    return score_rows, tuition_rows, errored


def run(
    *,
    source: Path,
    limit: int = 0,
    only: str = "both",
    dry_run: bool = False,
    max_size_kb: int = 200,
    truncate_kb: int = 100,
) -> int:
    """Execute the extraction loop. Returns the desired process exit code."""
    files = list(_iter_sources(source))
    if limit > 0:
        files = files[:limit]
    if not files:
        print(f"[extract_kg] no source files under {source}")
        return 0

    scores_repo = None if dry_run else AdmissionScoresRepo()
    tuition_repo = None if dry_run else TuitionRepo()

    error_count = 0
    for path in files:
        score_n, tuition_n, errored = _process_file(
            path,
            only=only,
            dry_run=dry_run,
            scores_repo=scores_repo,
            tuition_repo=tuition_repo,
            max_size_kb=max_size_kb,
            truncate_kb=truncate_kb,
        )
        prefix = "[dry-run] " if dry_run else ""
        print(
            f"{prefix}[{path.name}] extracted {score_n} score rows, "
            f"{tuition_n} tuition rows"
        )
        if errored:
            error_count += 1

    return 1 if error_count else 0


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LLM extractors over markdown sources and populate "
            "the admission_scores / tuition tables."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/samples"),
        help="Directory to walk for .md/.txt sources (default: data/samples).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N files (0 = no limit).",
    )
    parser.add_argument(
        "--only",
        choices=("score", "tuition", "both"),
        default="both",
        help="Restrict which extractor(s) to run (default: both, with per-file heuristics).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extraction summary without writing to the DB.",
    )
    parser.add_argument(
        "--max-size-kb",
        type=int,
        default=200,
        help="Files larger than this get truncated (default: 200).",
    )
    parser.add_argument(
        "--truncate-kb",
        type=int,
        default=100,
        help="Truncate oversized files to this prefix (default: 100).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_argparser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return run(
        source=args.source,
        limit=args.limit,
        only=args.only,
        dry_run=args.dry_run,
        max_size_kb=args.max_size_kb,
        truncate_kb=args.truncate_kb,
    )


if __name__ == "__main__":
    sys.exit(main())
