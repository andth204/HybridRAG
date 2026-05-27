"""Tests for ``src.hybridrag.chat.verifier`` (Phase 5.2 answer verifier)."""
from __future__ import annotations

from src.hybridrag.chat.verifier import (
    ClaimVerdict,
    NumericClaim,
    VerificationReport,
    annotate_answer,
    extract_numeric_claims,
    normalize_number,
    verify_answer,
)


# ---------------------------------------------------------------- #
# Claim extraction
# ---------------------------------------------------------------- #
def test_extract_simple_claim() -> None:
    """A score with a single citation produces one cited claim."""
    answer = "Điểm chuẩn ngành CNTT năm 2024 là 21,5 [1]."
    claims = extract_numeric_claims(answer)
    assert len(claims) == 1
    claim = claims[0]
    assert claim.text == "21,5"
    assert claim.value == 21.5
    assert claim.citation_ids == [1]


def test_extract_year_claim() -> None:
    """A 4-digit year that carries a citation IS a verifiable claim."""
    answer = "Năm 2024 [1]."
    claims = extract_numeric_claims(answer)
    assert len(claims) == 1
    claim = claims[0]
    assert claim.text == "2024"
    assert claim.value == 2024.0
    assert claim.citation_ids == [1]


def test_extract_multi_citation() -> None:
    """``[1, 2]`` and ``[1,2]`` and ``[1][2]`` shapes all flatten to a list."""
    for shape in (
        "Điểm là 21,5 [1, 2].",
        "Điểm là 21,5 [1,2].",
        "Điểm là 21,5 [1][2].",
    ):
        claims = extract_numeric_claims(shape)
        assert len(claims) == 1, shape
        assert claims[0].citation_ids == [1, 2], shape


def test_no_citation_claim() -> None:
    """A bare number without a citation is still extracted (status set later)."""
    claims = extract_numeric_claims("Điểm là 22")
    assert len(claims) == 1
    assert claims[0].text == "22"
    assert claims[0].value == 22.0
    assert claims[0].citation_ids == []


# ---------------------------------------------------------------- #
# Verification
# ---------------------------------------------------------------- #
def _docs(*texts: str) -> list[dict[str, object]]:
    """Cheap helper to build the ``retrieved_docs`` list expected by ``verify_answer``."""
    return [{"key": f"doc-{i+1}", "content": t} for i, t in enumerate(texts)]


def test_verify_pass() -> None:
    """Doc contains the number — verdict is verified, overall OK."""
    answer = "Điểm chuẩn năm 2024 là 21,5 [1]."
    docs = _docs("Năm 2024 điểm chuẩn ngành CNTT là 21.5 điểm.")
    report = verify_answer(answer, docs)

    assert report.overall == "ok"
    assert report.refusal_detected is False
    assert len(report.claims) == 1
    verdict = report.claims[0]
    assert verdict.status == "verified"
    assert verdict.matched_sources == [1]


def test_verify_fail() -> None:
    """Doc does not contain the claimed number — verdict is unverified."""
    answer = "Điểm chuẩn là 23,5 [1]."
    docs = _docs("Năm 2024 điểm chuẩn ngành CNTT là 21.5 điểm.")
    report = verify_answer(answer, docs)

    assert report.overall == "warning"
    assert len(report.claims) == 1
    verdict = report.claims[0]
    assert verdict.status == "unverified"
    assert verdict.matched_sources == []
    assert verdict.note  # has a human-readable explanation


def test_refusal_detected() -> None:
    """The canonical Vietnamese refusal phrase short-circuits verification.

    The marker is pulled from ``settings.REFUSAL_MESSAGE`` (Phase 5.3) when
    available, otherwise from the literal fallback defined in the verifier.
    Either way, an answer that starts with that phrase is classified as a
    refusal and no claims are extracted.
    """
    from src.hybridrag.chat.verifier import _refusal_marker

    answer = _refusal_marker() + " Liên hệ phòng tuyển sinh để biết thêm."
    report = verify_answer(answer, [])
    assert report.overall == "refusal"
    assert report.refusal_detected is True
    assert report.claims == []


# ---------------------------------------------------------------- #
# Number normalization
# ---------------------------------------------------------------- #
def test_normalize_number_thousands() -> None:
    """Locale-mixed digit groups all parse to the same float."""
    assert normalize_number("1.234.567") == 1234567.0
    assert normalize_number("1,234,567.89") == 1234567.89
    assert normalize_number("21,5") == 21.5
    assert normalize_number("21.5") == 21.5


# ---------------------------------------------------------------- #
# Annotation
# ---------------------------------------------------------------- #
def test_annotate_unverified() -> None:
    """When the report is a warning, the marker is appended."""
    answer = "Điểm chuẩn là 23,5 [1]."
    docs = _docs("Năm 2024 điểm chuẩn ngành CNTT là 21.5 điểm.")
    report = verify_answer(answer, docs)
    annotated = annotate_answer(answer, report)

    assert annotated.startswith(answer)
    assert annotated != answer
    assert "Lưu ý" in annotated
    assert "23,5" in annotated  # the unverified number is named


def test_annotate_ok_unchanged() -> None:
    """When everything verifies, the answer is returned verbatim."""
    answer = "Điểm chuẩn là 21,5 [1]."
    docs = _docs("Điểm chuẩn là 21.5 điểm.")
    report = verify_answer(answer, docs)
    assert report.overall == "ok"
    assert annotate_answer(answer, report) == answer


def test_annotate_refusal_unchanged() -> None:
    """Refusal answers must not be tagged with the warning marker either."""
    answer = "Xin lỗi, hiện mình chưa có thông tin chính thức về vấn đề này."
    report = verify_answer(answer, [])
    assert annotate_answer(answer, report) == answer


# ---------------------------------------------------------------- #
# Citation distance & invalid indices
# ---------------------------------------------------------------- #
def test_citation_distance_limit() -> None:
    """A citation that sits past 30 chars after the number is NOT attached."""
    answer = (
        "Điểm là 22, [1] và rất nhiều thông tin khác về tuyển sinh năm nay. [2]"
    )
    claims = extract_numeric_claims(answer)
    # The number "22" should pick up [1] only — [2] is far away.
    cited_22 = [c for c in claims if c.text == "22"]
    assert len(cited_22) == 1
    assert cited_22[0].citation_ids == [1]


def test_index_out_of_range() -> None:
    """Citing ``[5]`` when only two docs exist surfaces as unverified + note."""
    answer = "Điểm là 21,5 [5]."
    docs = _docs(
        "Năm 2024 điểm chuẩn ngành CNTT là 21.5 điểm.",
        "Năm 2023 điểm chuẩn ngành CNTT là 20.0 điểm.",
    )
    report = verify_answer(answer, docs)
    assert report.overall == "warning"
    assert len(report.claims) == 1
    verdict = report.claims[0]
    assert verdict.status == "unverified"
    assert verdict.note is not None
    # The note should mention the out-of-range index so an engineer can debug.
    assert "[5]" in verdict.note or "5" in verdict.note
