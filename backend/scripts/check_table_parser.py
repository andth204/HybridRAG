"""Visual dry-run of the markdown table parser.

Reads ``data/samples/Điểm 2024.md`` and prints what the
:mod:`hybridrag.ingestion.chunking.table_parser` module sees in it.

Run from the backend root::

    python scripts/check_table_parser.py
"""
from __future__ import annotations

import sys
from pathlib import Path


# Force stdout/stderr to UTF-8 so we can print Vietnamese filenames /
# table contents on Windows consoles (default cp1252 raises on ``Đ``).
for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

# Make ``src.*`` importable when executing the script directly.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from src.hybridrag.ingestion.chunking.table_parser import (  # noqa: E402
    find_tables,
    split_text_keeping_tables,
    table_to_row_chunks,
)


SAMPLE = BACKEND_DIR / "data" / "samples" / "Điểm 2024.md"


def _report_tables(label: str, text: str, file_id: str) -> int:
    tables = find_tables(text)
    print(f"--- {label} ---")
    print(f"  source            : {label}")
    print(f"  characters        : {len(text)}")
    print(f"  markdown tables   : {len(tables)}")

    total_chunks = 0
    for i, t in enumerate(tables):
        first_row = t.rows[0] if t.rows else []
        print(f"  table[{i}]")
        print(f"    caption        : {t.caption!r}")
        print(f"    columns        : {len(t.header)} -> {t.header}")
        print(f"    rows           : {len(t.rows)}")
        print(f"    alignments     : {t.alignments}")
        print(f"    first row      : {first_row}")
        chunks = table_to_row_chunks(t, file_id=file_id, table_index=i)
        total_chunks += len(chunks)
        if chunks:
            print(f"    first chunk_id : {chunks[0].chunk_id}")
            preview = chunks[0].content.replace("\n", " | ")
            if len(preview) > 140:
                preview = preview[:137] + "..."
            print(f"    first chunk md : {preview}")
    print(f"  TableChunks total : {total_chunks}")

    segments = list(split_text_keeping_tables(text))
    prose_segments = sum(1 for _seg, tbl in segments if tbl is None)
    table_segments = sum(1 for _seg, tbl in segments if tbl is not None)
    print(
        f"  split segments    : {len(segments)} "
        f"(prose={prose_segments}, table={table_segments})"
    )
    print()
    return len(tables)


def _synthetic_demo() -> str:
    """A small inline markdown sample the script always knows how to parse.

    Useful when the on-disk sample uses tab-separated rows (as the legacy
    Điểm files do) and therefore yields zero markdown tables.
    """
    return (
        "## Bảng điểm chuẩn 2024 (mẫu)\n"
        "\n"
        "| STT | Mã ngành | Tên ngành | Tổ hợp | Điểm chuẩn |\n"
        "| :--- | :---: | :--- | :---: | ---: |\n"
        "| 1 | 7480201 | Công nghệ thông tin | A00 | 17 |\n"
        "| 2 | 7480101 | Khoa học máy tính | A00 | 17 |\n"
        "| 3 | 7480103 | Kỹ thuật Phần mềm | A00 | 17 |\n"
        "\n"
        "Ghi chú: bảng minh hoạ cho dry-run.\n"
    )


def main() -> int:
    if not SAMPLE.exists():
        print(f"[FAIL] sample not found: {SAMPLE}")
        return 1

    text = SAMPLE.read_text(encoding="utf-8")
    found = _report_tables(SAMPLE.name, text, file_id="diem_2024")

    if found == 0:
        print(
            "[note] The on-disk sample contains no markdown pipe-tables "
            "(this corpus uses tab-separated rows). Falling back to an "
            "in-script synthetic sample so the parser surface can still "
            "be visually verified."
        )
        print()
        _report_tables("synthetic-demo", _synthetic_demo(), file_id="demo")

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
