from __future__ import annotations
import asyncio
import io
from datetime import datetime
from functools import lru_cache
from typing import Any
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from minio.commonconfig import CopySource
from minio.error import S3Error
from pydantic import BaseModel
from src.api.core.dependencies import AuthContext, get_auth_context
from src.api.core.storage import (
    build_minio_client,
    minio_object_exists,
    resolve_bucket,
    scope_object_key,
    unscope_object_key,
)
from src.config.settings import settings
from src.hybridrag.ingestion.ingestion_service.helper.state_repo import FileStateRepo

router = APIRouter(prefix="/api/v1/files", tags=["files"])


class FileAcceptedResponse(BaseModel):
    bucket: str
    key: str
    scoped_key: str
    action: str
    message: str
    status_endpoint: str


class FileStatusResponse(BaseModel):
    bucket: str
    key: str
    scoped_key: str
    minio_exists: bool
    indexed: bool
    status: str
    file_id: str | None = None
    etag: str | None = None
    version_id: str | None = None
    updated_at: datetime | None = None


class FileListItem(BaseModel):
    bucket: str
    key: str
    scoped_key: str
    file_id: str
    etag: str | None = None
    version_id: str | None = None
    updated_at: datetime | None = None


class FilesListResponse(BaseModel):
    items: list[FileListItem]


@lru_cache(maxsize=1)
def get_minio_client():
    return build_minio_client()


@lru_cache(maxsize=1)
def get_file_state_repo() -> FileStateRepo:
    return FileStateRepo(
        settings.DATABASE_URL,
        schema=settings.INDEX_STATE_SCHEMA,
        table=settings.INDEX_STATE_TABLE,
    )


def _parse_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


async def _assert_bucket_exists_or_404(bucket: str) -> None:
    minio_client = get_minio_client()
    try:
        exists = await asyncio.to_thread(minio_client.bucket_exists, bucket)
    except S3Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MinIO error while checking bucket: {exc}",
        ) from exc
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bucket '{bucket}' not found",
        )


def _build_status_endpoint(*, key: str, bucket: str) -> str:
    return f"/api/v1/files/{key}/status?bucket={bucket}"


async def _reindex_existing_object(
    *,
    minio_client,
    user_id: str,
    bucket: str,
    raw_key: str,
) -> FileAcceptedResponse:
    scoped_key = scope_object_key(user_id, raw_key)
    exists = await asyncio.to_thread(minio_object_exists, minio_client, bucket, scoped_key)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Object not found in MinIO",
        )

    await asyncio.to_thread(minio_client.copy_object, bucket, scoped_key, CopySource(bucket, scoped_key))
    key_for_path = unscope_object_key(user_id, scoped_key)
    return FileAcceptedResponse(
        bucket=bucket,
        key=key_for_path,
        scoped_key=scoped_key,
        action="reindexed",
        message="Existing object touch accepted, re-indexing event emitted",
        status_endpoint=_build_status_endpoint(key=key_for_path, bucket=bucket),
    )


@router.post("/index", status_code=status.HTTP_202_ACCEPTED, response_model=FileAcceptedResponse)
async def index_file(
    request: Request,
    file: UploadFile | None = File(default=None),
    bucket: str | None = Form(default=None),
    key: str | None = Form(default=None),
    auth: AuthContext = Depends(get_auth_context),
) -> FileAcceptedResponse:
    minio_client = get_minio_client()
    raw_bucket = _parse_optional_text(bucket)
    raw_key = _parse_optional_text(key)

    try:
        # Swagger-documented flow: multipart upload from local file.
        if file is not None:
            target_bucket = resolve_bucket(raw_bucket)
            await _assert_bucket_exists_or_404(target_bucket)

            filename = _parse_optional_text(getattr(file, "filename", None)) or "upload.bin"
            public_key = raw_key or filename
            scoped_key = scope_object_key(auth.user_id, public_key)
            data = await file.read()
            if not data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty",
                )

            put_stream = io.BytesIO(data)
            put_content_type = _parse_optional_text(getattr(file, "content_type", None)) or "application/octet-stream"
            await asyncio.to_thread(
                minio_client.put_object,
                target_bucket,
                scoped_key,
                put_stream,
                len(data),
                put_content_type,
            )
            key_for_path = unscope_object_key(auth.user_id, scoped_key)
            return FileAcceptedResponse(
                bucket=target_bucket,
                key=key_for_path,
                scoped_key=scoped_key,
                action="uploaded",
                message="File uploaded to MinIO, indexing event accepted",
                status_endpoint=_build_status_endpoint(key=key_for_path, bucket=target_bucket),
            )

        # Optional form-only key touch mode (no upload).
        if raw_key:
            target_bucket = resolve_bucket(raw_bucket)
            await _assert_bucket_exists_or_404(target_bucket)
            return await _reindex_existing_object(
                minio_client=minio_client,
                user_id=auth.user_id,
                bucket=target_bucket,
                raw_key=raw_key,
            )

        # Backward-compatible JSON mode on the same endpoint.
        content_type = (request.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="JSON payload must be an object",
                )
            json_bucket = _parse_optional_text(payload.get("bucket"))
            json_key = _parse_optional_text(payload.get("key"))
            if not json_key:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="key is required for JSON payload",
                )
            target_bucket = resolve_bucket(json_bucket)
            await _assert_bucket_exists_or_404(target_bucket)
            return await _reindex_existing_object(
                minio_client=minio_client,
                user_id=auth.user_id,
                bucket=target_bucket,
                raw_key=json_key,
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide multipart 'file' upload or object 'key' to re-index",
        )
    except HTTPException:
        raise
    except S3Error as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MinIO operation failed: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.delete("/{object_key:path}", status_code=status.HTTP_202_ACCEPTED, response_model=FileAcceptedResponse)
async def delete_file(
    object_key: str,
    bucket: str | None = Query(default=None),
    auth: AuthContext = Depends(get_auth_context),
) -> FileAcceptedResponse:
    try:
        target_bucket = resolve_bucket(bucket)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    await _assert_bucket_exists_or_404(target_bucket)
    scoped_key = scope_object_key(auth.user_id, object_key)
    minio_client = get_minio_client()
    exists = await asyncio.to_thread(minio_object_exists, minio_client, target_bucket, scoped_key)
    if exists:
        try:
            await asyncio.to_thread(minio_client.remove_object, target_bucket, scoped_key)
        except S3Error as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"MinIO delete failed: {exc}",
            ) from exc

    key_for_path = unscope_object_key(auth.user_id, scoped_key)
    return FileAcceptedResponse(
        bucket=target_bucket,
        key=key_for_path,
        scoped_key=scoped_key,
        action="delete_requested",
        message="Delete accepted; ingestion service will process delete event asynchronously",
        status_endpoint=f"/api/v1/files/{key_for_path}/status?bucket={target_bucket}",
    )


@router.get("", response_model=FilesListResponse)
async def list_files(
    bucket: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    auth: AuthContext = Depends(get_auth_context),
) -> FilesListResponse:
    try:
        target_bucket = resolve_bucket(bucket)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    state_repo = get_file_state_repo()
    prefix = f"{auth.user_id}/"
    states = await asyncio.to_thread(
        state_repo.list,
        bucket=target_bucket,
        key_prefix=prefix,
        limit=limit,
        offset=offset,
    )
    items = [
        FileListItem(
            bucket=item.bucket,
            key=unscope_object_key(auth.user_id, item.key),
            scoped_key=item.key,
            file_id=item.file_id,
            etag=item.etag,
            version_id=item.version_id,
            updated_at=item.updated_at,
        )
        for item in states
    ]
    return FilesListResponse(items=items)


@router.get("/{object_key:path}/status", response_model=FileStatusResponse)
async def file_status(
    object_key: str,
    bucket: str | None = Query(default=None),
    auth: AuthContext = Depends(get_auth_context),
) -> FileStatusResponse:
    try:
        target_bucket = resolve_bucket(bucket)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    scoped_key = scope_object_key(auth.user_id, object_key)
    minio_client = get_minio_client()
    state_repo = get_file_state_repo()

    minio_exists = await asyncio.to_thread(minio_object_exists, minio_client, target_bucket, scoped_key)
    state_item = await asyncio.to_thread(state_repo.get, target_bucket, scoped_key)
    indexed = state_item is not None

    if indexed:
        resolved_status = "indexed"
    elif minio_exists:
        resolved_status = "pending"
    else:
        resolved_status = "not_found"

    return FileStatusResponse(
        bucket=target_bucket,
        key=unscope_object_key(auth.user_id, scoped_key),
        scoped_key=scoped_key,
        minio_exists=minio_exists,
        indexed=indexed,
        status=resolved_status,
        file_id=(state_item.file_id if state_item else None),
        etag=(state_item.etag if state_item else None),
        version_id=(state_item.version_id if state_item else None),
        updated_at=(state_item.updated_at if state_item else None),
    )