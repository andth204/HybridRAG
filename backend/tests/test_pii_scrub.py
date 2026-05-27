"""Tests for ``src.hybridrag.utils.pii_scrub`` (Phase 5.9 PII scrubber)."""
from __future__ import annotations

from src.hybridrag.utils.pii_scrub import detect, scrub, scrub_dict


# ---------------------------------------------------------------- #
# Phone scrubbing
# ---------------------------------------------------------------- #
def test_scrub_phone_basic() -> None:
    """Bare 10-digit VN mobile is replaced verbatim."""
    assert scrub("Liên hệ 0987654321") == "Liên hệ ***"


def test_scrub_phone_separators() -> None:
    """Phones with spaces and dashes are also caught."""
    assert scrub("Số 0987 654 321 và 0987-654-321") == "Số *** và ***"


def test_scrub_international_phone() -> None:
    """``+84`` form is recognised, with or without a separator after the prefix."""
    assert scrub("+84 987 654 321") == "***"
    assert scrub("+84987654321") == "***"


# ---------------------------------------------------------------- #
# Email scrubbing
# ---------------------------------------------------------------- #
def test_scrub_email() -> None:
    """A simple ``user@domain.tld`` token is redacted."""
    assert scrub("Gửi mail tới abc@example.com") == "Gửi mail tới ***"


# ---------------------------------------------------------------- #
# Vietnamese ID scrubbing
# ---------------------------------------------------------------- #
def test_scrub_cccd() -> None:
    """12-digit CCCD is matched as an ID, not a phone or card."""
    assert scrub("CCCD 012345678901") == "CCCD ***"


def test_scrub_cmnd_9digit() -> None:
    """9-digit CMND is matched as an ID."""
    assert scrub("CMND 123456789") == "CMND ***"


# ---------------------------------------------------------------- #
# No false positives on common content
# ---------------------------------------------------------------- #
def test_scrub_no_year_collision() -> None:
    """A bare year + score must not trigger any PII pattern."""
    text = "Năm 2024 điểm chuẩn 21,5"
    assert scrub(text) == text


# ---------------------------------------------------------------- #
# Mixed input
# ---------------------------------------------------------------- #
def test_scrub_mixed() -> None:
    """Multiple PII tokens in one string all get redacted."""
    text = "Email abc@d.com, SĐT 0987654321, CCCD 012345678901"
    assert scrub(text) == "Email ***, SĐT ***, CCCD ***"


# ---------------------------------------------------------------- #
# detect()
# ---------------------------------------------------------------- #
def test_detect_returns_kinds() -> None:
    """detect() yields kind→[matches] without mutating the input."""
    text = "Email abc@d.com, SĐT 0987654321, CCCD 012345678901"
    out = detect(text)

    assert out["emails"] == ["abc@d.com"]
    assert out["ids"] == ["012345678901"]
    assert out["phones"] == ["0987654321"]
    # Cards is documented as "loose"; the three PII tokens above must not be
    # double-counted under cards.
    assert out["cards"] == []
    # All four kinds are always present, even when empty.
    assert set(out.keys()) == {"emails", "ids", "phones", "cards"}


# ---------------------------------------------------------------- #
# scrub_dict() — recursive structure handling
# ---------------------------------------------------------------- #
def test_scrub_dict_recursive() -> None:
    """Nested dicts and lists are walked; only string leaves are scrubbed."""
    data = {
        "msg": "call 0987654321",
        "meta": {"user_email": "a@b.com"},
        "history": ["abc@d.com", "no pii here"],
        "count": 7,
    }
    cleaned = scrub_dict(data)

    assert cleaned["msg"] == "call ***"
    assert cleaned["meta"] == {"user_email": "***"}
    assert cleaned["history"] == ["***", "no pii here"]
    assert cleaned["count"] == 7  # non-string preserved


def test_scrub_dict_with_key_filter() -> None:
    """When ``keys`` is supplied, only those (case-insensitive) get scrubbed."""
    data = {
        "msg": "call 0987654321",
        "meta": {"user_email": "a@b.com"},
    }
    cleaned = scrub_dict(data, keys={"msg"})

    assert cleaned["msg"] == "call ***"
    # meta.user_email must be left intact when not on the whitelist.
    assert cleaned["meta"] == {"user_email": "a@b.com"}


# ---------------------------------------------------------------- #
# Cards — loose match by design
# ---------------------------------------------------------------- #
def test_card_loose_match() -> None:
    """16-digit Visa-style block with spaces is redacted by the card pattern."""
    assert scrub("4111 1111 1111 1111") == "***"
