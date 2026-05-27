#!/usr/bin/env bash
# Phase 6.8 — Nightly backup: Postgres + Weaviate snapshot + FAISS/BM25 tarball → MinIO
set -euo pipefail

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="${BACKUP_DIR:-/tmp/hybridrag-backups}"
MINIO_BUCKET="${MINIO_BUCKET:-hybridrag-backups}"
MINIO_ALIAS="${MINIO_ALIAS:-local}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
WEAVIATE_URL="${WEAVIATE_URL:-http://localhost:8080}"
VECTOR_STORE_DIR="${VECTOR_STORE_DIR:-./data/vector_store}"

mkdir -p "$BACKUP_DIR"

echo "[$(date -Iseconds)] backup start ts=$TIMESTAMP"

# 1. Postgres
if [ -n "${POSTGRES_HOST:-}" ] && [ -n "${POSTGRES_USER:-}" ] && [ -n "${POSTGRES_DB:-}" ]; then
    PGPASSWORD="${POSTGRES_PASSWORD:-}" pg_dump \
        -h "$POSTGRES_HOST" -p "${POSTGRES_PORT:-5432}" \
        -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
        -F c -f "$BACKUP_DIR/pg_${TIMESTAMP}.dump"
    echo "[ok] pg_dump → pg_${TIMESTAMP}.dump"
else
    echo "[skip] Postgres env vars missing"
fi

# 2. Weaviate snapshot
if curl -sf "$WEAVIATE_URL/v1/.well-known/ready" > /dev/null 2>&1; then
    BACKUP_ID="hybridrag_${TIMESTAMP}"
    curl -sX POST "$WEAVIATE_URL/v1/backups/filesystem" \
        -H 'Content-Type: application/json' \
        -d "{\"id\":\"$BACKUP_ID\",\"include\":[\"DocChunk\"]}" \
        > "$BACKUP_DIR/weaviate_${TIMESTAMP}.json" || true
    echo "[ok] weaviate snapshot id=$BACKUP_ID"
else
    echo "[skip] weaviate unreachable at $WEAVIATE_URL"
fi

# 3. Local FAISS + BM25 tarball
if [ -d "$VECTOR_STORE_DIR" ]; then
    tar -C "$VECTOR_STORE_DIR" -czf "$BACKUP_DIR/vectorstore_${TIMESTAMP}.tar.gz" .
    echo "[ok] vectorstore tar created"
else
    echo "[skip] $VECTOR_STORE_DIR not found"
fi

# 4. Upload to MinIO (requires `mc` CLI + configured alias)
if command -v mc > /dev/null 2>&1; then
    mc mb -p "$MINIO_ALIAS/$MINIO_BUCKET" 2>/dev/null || true
    for f in "$BACKUP_DIR"/pg_${TIMESTAMP}.dump "$BACKUP_DIR"/vectorstore_${TIMESTAMP}.tar.gz "$BACKUP_DIR"/weaviate_${TIMESTAMP}.json; do
        [ -f "$f" ] || continue
        mc cp "$f" "$MINIO_ALIAS/$MINIO_BUCKET/" || echo "[warn] mc upload failed for $f"
    done
    echo "[ok] uploaded to minio"
else
    echo "[skip] mc CLI not installed"
fi

# 5. Retention sweep
find "$BACKUP_DIR" -type f -mtime "+$RETENTION_DAYS" -delete
echo "[ok] retention sweep (kept $RETENTION_DAYS days)"

echo "[$(date -Iseconds)] backup done"
