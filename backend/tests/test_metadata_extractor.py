"""Tests for ``src.hybridrag.ingestion.metadata.extractor``."""
from __future__ import annotations

import pytest

from src.hybridrag.ingestion.metadata.extractor import extract_metadata
from src.hybridrag.utils.entity_resolver import reload_entities


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    """Drop the cached entity table before each test."""
    reload_entities()


# ---------------------------------------------------------------- #
# Year extraction
# ---------------------------------------------------------------- #
def test_extract_year_from_filename() -> None:
    meta = extract_metadata(
        text="Bảng điểm chuẩn năm 2024 của trường.",
        header_path=None,
        filename="Điểm 2024.md",
    )
    assert meta.get("year") == 2024


def test_extract_year_from_text() -> None:
    """When no year is in the filename / headers, derive from content."""
    meta = extract_metadata(
        text="Thông tin học phí năm học 2025-2026 dự kiến.",
        header_path=None,
        filename="hoc_phi.md",
    )
    # Earliest 20XX wins → 2025.
    assert meta.get("year") == 2025


def test_extract_year_from_header_first() -> None:
    """A year in the header_path takes precedence over older years in text."""
    meta = extract_metadata(
        text="Tham chiếu lại số liệu năm 2024.",
        header_path=["Tuyển sinh 2023"],
        filename="qa.md",
    )
    # Earliest year across all sources wins → 2023.
    assert meta.get("year") == 2023


def test_extract_year_missing_returns_no_key() -> None:
    meta = extract_metadata(
        text="Trường có cơ sở 1 tại Khoái Châu.",
        header_path=["Cơ sở"],
        filename="campuses.md",
    )
    assert "year" not in meta


# ---------------------------------------------------------------- #
# Campus extraction
# ---------------------------------------------------------------- #
def test_extract_campus_from_header() -> None:
    meta = extract_metadata(
        text="Địa chỉ liên hệ.",
        header_path=["Cơ sở 1", "Liên hệ"],
        filename="qa.md",
    )
    assert meta.get("campus") == "co_so_1"


def test_extract_campus_from_text_when_header_missing() -> None:
    meta = extract_metadata(
        text=(
            "Ngành Công nghệ kỹ thuật ô tô được đào tạo tại cơ sở 1 "
            "(xã Dân Tiến, Khoái Châu, Hưng Yên)."
        ),
        header_path=[],
        filename="qa_fb.md",
    )
    assert meta.get("campus") == "co_so_1"


def test_extract_campus_three() -> None:
    meta = extract_metadata(
        text="Cơ sở 3 của UTEHY tại Hải Dương đào tạo các ngành Kinh tế.",
        header_path=[],
        filename="qa_fb.md",
    )
    assert meta.get("campus") == "co_so_3"


# ---------------------------------------------------------------- #
# doc_type extraction
# ---------------------------------------------------------------- #
def test_extract_doc_type_from_filename() -> None:
    meta = extract_metadata(
        text="Bảng điểm chuẩn các ngành năm 2023.",
        header_path=[],
        filename="Điểm 2023.md",
    )
    assert meta.get("doc_type") == "diem_chuan"


def test_extract_doc_type_tuyen_sinh() -> None:
    meta = extract_metadata(
        text="Đề án tuyển sinh năm 2025 của UTEHY.",
        header_path=[],
        filename="tuyen_sinh_247.md",
    )
    assert meta.get("doc_type") == "tuyen_sinh"


def test_extract_doc_type_tuition() -> None:
    meta = extract_metadata(
        text="Học phí khối kỹ thuật.",
        header_path=[],
        filename="hoc_phi.md",
    )
    assert meta.get("doc_type") == "hoc_phi"


# ---------------------------------------------------------------- #
# Faculty / major extraction
# ---------------------------------------------------------------- #
def test_extract_faculty() -> None:
    """A passage talking about a major also surfaces the faculty grouping."""
    meta = extract_metadata(
        text="Điểm chuẩn ngành Công nghệ thông tin năm 2024 theo điểm thi THPT là 17.",
        header_path=[],
        filename="Điểm 2024.md",
    )
    assert meta.get("faculty") == "cntt"


def test_extract_major() -> None:
    meta = extract_metadata(
        text="Ngành Kỹ thuật phần mềm tuyển 210 chỉ tiêu năm 2025.",
        header_path=[],
        filename="tuyen_sinh_247.md",
    )
    assert meta.get("major") == "ky_thuat_phan_mem"


# ---------------------------------------------------------------- #
# Empty / minimal input
# ---------------------------------------------------------------- #
def test_extract_empty_input() -> None:
    meta = extract_metadata(text="", header_path=None, filename=None)
    assert meta == {}
