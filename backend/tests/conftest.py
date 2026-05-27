"""Shared pytest fixtures for the HybridRAG backend test suite."""
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the backend repo root (the directory containing `src/`, `data/`, ...)."""
    return Path(__file__).parent.parent


@pytest.fixture
def samples_dir(repo_root: Path) -> Path:
    """Absolute path to the sample-documents directory used by ingestion tests."""
    return repo_root / "data" / "samples"
