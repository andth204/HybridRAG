"""Migration helper: walk a local document directory and push the
hierarchical chunks into Weaviate.

Used during Phase 2 integration to compare the new pipeline against the
legacy FAISS+BM25 path without re-driving the production Kafka /
MinIO flow. The script connects directly to Weaviate -- if the server
is unreachable, it exits non-zero with a clear message. It NEVER
auto-starts Docker.

Usage (from ``backend/``):

    python scripts/migrate_to_weaviate.py                 # default: data/samples
    python scripts/migrate_to_weaviate.py --source data/eval/docs --limit 5
    python scripts/migrate_to_weaviate.py --dry-run       # split + count only
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Make ``src.*`` importable when executing the script directly.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.config.settings import settings  # noqa: E402  (import after sys.path tweak)
from src.hybridrag.ingestion.chunking.hierarchical import HierarchicalSplitter  # noqa: E402
from src.hybridrag.ingestion.embedding import embedder  # noqa: E402
from src.hybridrag.ingestion.ingestion_service.entities.weaviate_store import (  # noqa: E402
    WeaviateStore,
)
from src.hybridrag.ingestion.metadata.extractor import extract_metadata  # noqa: E402


SUPPORTED_SUFFIXES = {".md", ".txt"}


def _stable_file_id(filename: str) -> str:
    """Deterministic per-filename UUID so re-runs are idempotent.

    We append ``:local`` (instead of an etag) because the migration
    script does not go through MinIO.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}:local"))


def _iter_source_files(source: Path, limit: int) -> list[Path]:
    files: list[Path] = []
    if source.is_file():
        if source.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(source)
    else:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(path)
    if limit and limit > 0:
        files = files[:limit]
    return files


def _check_weaviate_reachable(url: str) -> None:
    """Probe the Weaviate /v1/.well-known/ready endpoint before doing work.

    Raises SystemExit(2) with a human-readable message if the server is
    not reachable -- we want to fail fast rather than after the user has
    waited for the splitter to chew through everything.
    """
    try:
        import requests  # local import so the rest of the script is import-light
    except ImportError:
        # If `requests` is missing, fall back to a socket probe so the
        # script remains useful in stripped-down environments.
        from urllib.parse import urlparse
        import socket
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 8080)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            try:
                s.connect((host, port))
            except OSError as exc:
                print(
                    f"[fatal] Weaviate is not reachable at {url} ({exc}). "
                    "Start it with `docker compose -f docker-compose.weaviate.yml up -d` "
                    "and retry.",
                    file=sys.stderr,
                )
                raise SystemExit(2)
        return

    ready_url = url.rstrip("/") + "/v1/.well-known/ready"
    try:
        resp = requests.get(ready_url, timeout=3.0)
    except requests.RequestException as exc:
        print(
            f"[fatal] Weaviate is not reachable at {url} ({exc}). "
            "Start it with `docker compose -f docker-compose.weaviate.yml up -d` "
            "and retry.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if resp.status_code != 200:
        print(
            f"[fatal] Weaviate is not reachable at {url} "
            f"(HTTP {resp.status_code} on /v1/.well-known/ready). "
            "Start it with `docker compose -f docker-compose.weaviate.yml up -d` "
            "and retry.",
            file=sys.stderr,
        )
        raise SystemExit(2)


async def _migrate(
    *,
    source: Path,
    limit: int,
    dry_run: bool,
) -> dict[str, Any]:
    files = _iter_source_files(source, limit)
    if not files:
        print(f"[warn] No .md/.txt files found under {source}")
        return {"files": 0, "parents": 0, "children": 0, "errors": 0}

    splitter = HierarchicalSplitter()
    store: WeaviateStore | None = None
    if not dry_run:
        store = WeaviateStore(
            embedding_fn=embedder.embed,
            embedding_dim=embedder.get_dimension(),
        )

    total_parents = 0
    total_children = 0
    errors = 0

    for path in files:
        filename = path.name
        key = filename
        file_id = _stable_file_id(filename)
        print(f"\n  [{filename}]  file_id={file_id}")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"    [skip] read error: {exc}")
            errors += 1
            continue

        if not text.strip():
            print("    [skip] empty file")
            continue

        try:
            parents, children = splitter.split(
                text,
                file_id=file_id,
                key=key,
                doc_meta={"file_id": file_id, "key": key},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    [fail] splitter raised: {exc}")
            errors += 1
            continue

        print(f"    parents={len(parents)}  children={len(children)}  chars={len(text):,}")
        total_parents += len(parents)
        total_children += len(children)

        if dry_run:
            continue

        # Build the enriched dicts and stream them into Weaviate.
        enriched: list[dict[str, Any]] = []
        for chunk in (*parents, *children):
            meta = extract_metadata(
                text=chunk.content,
                header_path=chunk.header_path,
                filename=filename,
            )
            d: dict[str, Any] = {
                "chunk_id":    chunk.chunk_id,
                "parent_id":   chunk.parent_id,
                "file_id":     file_id,
                "key":         key,
                "content":     chunk.content,
                "section":     chunk.section,
                "header_path": list(chunk.header_path or []),
                "chunk_level": chunk.chunk_level,
                "is_table":    bool(chunk.is_table),
            }
            for k, v in meta.items():
                if v is not None:
                    d[k] = v
            enriched.append(d)

        if not enriched:
            continue
        try:
            # Replace any previous chunks for this key so the migration
            # is idempotent across re-runs.
            removed = await asyncio.to_thread(store.delete_by_key, key)
            if removed:
                print(f"    removed {removed} stale chunks for key={key}")
            await store.precompute_embeddings(enriched)
            inserted = await store.add_chunks(enriched)
            print(f"    inserted {inserted} objects")
        except Exception as exc:  # noqa: BLE001
            print(f"    [fail] Weaviate write error: {exc}")
            errors += 1

    if store is not None:
        store.close()

    return {
        "files": len(files),
        "parents": total_parents,
        "children": total_children,
        "errors": errors,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Push local .md/.txt documents into a Weaviate DocChunk collection."
    )
    p.add_argument(
        "--source",
        default=str(BACKEND_DIR / "data" / "samples"),
        help="Directory (or single file) to migrate. Default: data/samples",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N files (0=all). Default: 0",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Split + count without writing to Weaviate.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    source = Path(args.source).resolve()
    if not source.exists():
        print(f"[fatal] --source path does not exist: {source}", file=sys.stderr)
        return 2

    if not args.dry_run:
        _check_weaviate_reachable(settings.WEAVIATE_URL)

    print(f"Migrating from {source}  (limit={args.limit or 'all'}, dry_run={args.dry_run})")
    print("=" * 60)
    t0 = time.perf_counter()
    stats = asyncio.run(_migrate(source=source, limit=args.limit, dry_run=args.dry_run))
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(
        f"  files={stats['files']}  parents={stats['parents']}  "
        f"children={stats['children']}  errors={stats['errors']}  "
        f"elapsed={elapsed:.2f}s"
    )
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
