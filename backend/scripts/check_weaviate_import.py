"""Smoke-test that the Weaviate modules import cleanly.

This script does NOT open a connection to Weaviate. It only verifies that:
  - weaviate-client (>=4.9, <5) is installed in the active environment, and
  - WeaviateStore and the schema module load without import errors.

Run from the backend root:
    python scripts/check_weaviate_import.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src.*` importable when executing the script directly.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    try:
        import weaviate  # noqa: F401
    except ImportError as e:
        print(
            "[FAIL] weaviate-client is not installed. "
            "Install it with: pip install 'weaviate-client>=4.9,<5'"
        )
        print(f"       Underlying error: {e}")
        return 1

    print(f"[ok] weaviate-client v{weaviate.__version__}")

    try:
        from src.hybridrag.ingestion.ingestion_service.entities import weaviate_schema
        from src.hybridrag.ingestion.ingestion_service.entities.weaviate_store import (
            WeaviateStore,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] failed to import Weaviate modules: {e}")
        return 1

    print(f"[ok] imported {weaviate_schema.__name__}")
    print(f"[ok] imported WeaviateStore from {WeaviateStore.__module__}")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
