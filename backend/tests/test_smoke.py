"""Smoke tests verifying the test harness and basic repo layout."""
import sys
from pathlib import Path


def test_python_version() -> None:
    """The project requires Python 3.10+."""
    assert sys.version_info >= (3, 10), (
        f"Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}"
    )


def test_repo_layout(repo_root: Path) -> None:
    """Core source and data directories must exist relative to the backend root."""
    assert (repo_root / "src" / "hybridrag").is_dir(), (
        f"Missing src/hybridrag directory under {repo_root}"
    )
    assert (repo_root / "data" / "samples").is_dir(), (
        f"Missing data/samples directory under {repo_root}"
    )
