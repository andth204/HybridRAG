from __future__ import annotations
import asyncio
import io
import json
import logging
import mimetypes
import time
import unicodedata
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Any
from urllib.parse import quote
from confluent_kafka import Consumer, KafkaError, Producer
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from minio.error import S3Error
from pydantic import BaseModel
from src.api.core.dependencies import AuthContext, get_auth_context, get_manager_auth_context
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
logger = logging.getLogger(__name__)


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
    status: str
    size_bytes: int = 0
    file_id: str | None = None
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


@lru_cache(maxsize=1)
def get_kafka_input_producer() -> Producer:
    return Producer({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS})


def _parse_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _basename(path_like: str) -> str:
    normalized = path_like.strip().replace("\\", "/")
    return normalized.rsplit("/", 1)[-1].strip()


def _uses_global_document_scope(auth: AuthContext) -> bool:
    return auth.user_role == "manager"


def _document_storage_key(auth: AuthContext, object_key: str) -> str:
    normalized_key = object_key.strip().lstrip("/")
    if not normalized_key:
        raise ValueError("object_key must not be empty")
    if _uses_global_document_scope(auth):
        return normalized_key
    return scope_object_key(auth.user_id, normalized_key)


def _document_public_key(auth: AuthContext, storage_key: str) -> str:
    if _uses_global_document_scope(auth):
        return storage_key
    return unscope_object_key(auth.user_id, storage_key)


def _document_key_prefix(auth: AuthContext) -> str | None:
    if _uses_global_document_scope(auth):
        return None
    return f"{auth.user_id}/"


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


def _publish_indexing_input_event(
    *,
    bucket: str,
    key: str,
    event_type: str,
    etag: str | None = None,
    version_id: str | None = None,
    force: bool = False,
    requested_action: str | None = None,
) -> None:
    if not settings.KAFKA_BOOTSTRAP_SERVERS.strip() or not settings.INDEXING_INPUT_TOPIC.strip():
        raise RuntimeError("Kafka input topic is not configured")

    payload: dict[str, Any] = {
        "event_type": event_type,
        "bucket": bucket,
        "key": key,
        "force": force,
    }
    if etag:
        payload["etag"] = etag
    if version_id:
        payload["version_id"] = version_id
    if requested_action:
        payload["requested_action"] = requested_action

    producer = get_kafka_input_producer()
    delivery_error: str | None = None

    def _delivery(err, _msg):
        nonlocal delivery_error
        if err is not None:
            delivery_error = str(err)

    producer.produce(
        settings.INDEXING_INPUT_TOPIC,
        key=key.encode("utf-8", errors="ignore"),
        value=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        on_delivery=_delivery,
    )
    producer.poll(0)
    producer.flush(3.0)
    if delivery_error:
        raise RuntimeError(delivery_error)


def _build_download_headers(filename: str) -> dict[str, str]:
    safe_name = (
        filename.strip()
        .replace("\r", "")
        .replace("\n", "")
        .replace('"', "")
    ) or "download"
    ascii_fallback = unicodedata.normalize(
        "NFKD",
        safe_name.replace("đ", "d").replace("Đ", "D"),
    ).encode("ascii", "ignore").decode("ascii").strip()
    if not ascii_fallback:
        ascii_fallback = "download"
    encoded_name = quote(safe_name, safe="")
    return {
        "Content-Disposition": (
            f'attachment; filename="{ascii_fallback}"; '
            f"filename*=UTF-8''{encoded_name}"
        ),
        "Cache-Control": "no-store",
    }


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _resolve_file_status(*, indexed: bool, minio_exists: bool) -> str:
    if indexed and minio_exists:
        return "indexed"
    if indexed and not minio_exists:
        return "deleting"
    if minio_exists:
        return "indexing"
    return "not_found"


def _list_minio_objects(*, minio_client, bucket: str, prefix: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in minio_client.list_objects(bucket, prefix=prefix, recursive=True):
        object_name = _parse_optional_text(getattr(item, "object_name", None))
        if not object_name:
            continue
        etag = getattr(item, "etag", None)
        if isinstance(etag, str):
            etag = etag.strip('"')
        items.append(
            {
                "key": object_name,
                "size_bytes": int(getattr(item, "size", 0) or 0),
                "updated_at": getattr(item, "last_modified", None),
                "etag": etag,
            }
        )
    return items


async def _stream_minio_object(*, minio_client, bucket: str, scoped_key: str, download_name: str) -> StreamingResponse:
    try:
        response = await asyncio.to_thread(minio_client.get_object, bucket, scoped_key)
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Referenced file not found",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MinIO download failed: {exc}",
        ) from exc

    content_type = response.headers.get("Content-Type") or mimetypes.guess_type(download_name)[0] or "application/octet-stream"

    async def iterator():
        try:
            while True:
                chunk = await asyncio.to_thread(response.read, 64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            response.close()
            response.release_conn()

    return StreamingResponse(
        iterator(),
        media_type=content_type,
        headers=_build_download_headers(download_name),
    )


async def _resolve_reference_download(
    *,
    user_id: str,
    bucket: str,
    reference_name: str,
):
    normalized_name = _parse_optional_text(reference_name)
    if not normalized_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="reference_name must not be empty",
        )

    minio_client = get_minio_client()
    exact_scoped_key = scope_object_key(user_id, normalized_name)
    if await asyncio.to_thread(minio_object_exists, minio_client, bucket, exact_scoped_key):
        return await _stream_minio_object(
            minio_client=minio_client,
            bucket=bucket,
            scoped_key=exact_scoped_key,
            download_name=_basename(normalized_name) or normalized_name,
        )

    state_repo = get_file_state_repo()
    matches = await asyncio.to_thread(
        state_repo.find_by_basename,
        bucket=bucket,
        key_prefix=f"{user_id}/",
        basename=_basename(normalized_name),
        limit=2,
    )
    if len(matches) == 1:
        match = matches[0]
        return await _stream_minio_object(
            minio_client=minio_client,
            bucket=bucket,
            scoped_key=match.key,
            download_name=_basename(match.key) or normalized_name,
        )
    if len(matches) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Multiple files match reference '{normalized_name}'. Use a unique file name.",
        )

    shared_matches = await asyncio.to_thread(
        state_repo.find_by_basename,
        bucket=bucket,
        basename=_basename(normalized_name),
        limit=3,
    )
    if len(shared_matches) == 1:
        match = shared_matches[0]
        return await _stream_minio_object(
            minio_client=minio_client,
            bucket=bucket,
            scoped_key=match.key,
            download_name=_basename(match.key) or normalized_name,
        )
    if len(shared_matches) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Multiple indexed files share the name '{normalized_name}'. Use unique file names in MinIO.",
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Referenced file '{normalized_name}' was not found",
    )


async def _reindex_existing_object(
    *,
    minio_client,
    auth: AuthContext,
    bucket: str,
    raw_key: str,
) -> FileAcceptedResponse:
    scoped_key = _document_storage_key(auth, raw_key)
    try:
        stat = await asyncio.to_thread(minio_client.stat_object, bucket, scoped_key)
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Object not found in MinIO",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MinIO stat failed: {exc}",
        ) from exc

    etag = _parse_optional_text(getattr(stat, "etag", None))
    if etag is not None:
        etag = etag.strip('"')
    version_id = _parse_optional_text(getattr(stat, "version_id", None))

    try:
        await asyncio.to_thread(
            _publish_indexing_input_event,
            bucket=bucket,
            key=scoped_key,
            event_type="file_updated",
            etag=etag,
            version_id=version_id,
            force=True,
            requested_action="reindexed",
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Kafka re-index publish failed: {exc}",
        ) from exc

    key_for_path = _document_public_key(auth, scoped_key)
    return FileAcceptedResponse(
        bucket=bucket,
        key=key_for_path,
        scoped_key=scoped_key,
        action="reindexed",
        message="Re-index request accepted",
        status_endpoint=_build_status_endpoint(key=key_for_path, bucket=bucket),
    )


@router.post("/index", status_code=status.HTTP_202_ACCEPTED, response_model=FileAcceptedResponse)
async def index_file(
    request: Request,
    file: UploadFile | None = File(default=None),
    bucket: str | None = Form(default=None),
    key: str | None = Form(default=None),
    auth: AuthContext = Depends(get_manager_auth_context),
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
            scoped_key = _document_storage_key(auth, public_key)
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
            key_for_path = _document_public_key(auth, scoped_key)
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
                auth=auth,
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
                auth=auth,
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


@router.get("", response_model=FilesListResponse)
async def list_files(
    bucket: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    auth: AuthContext = Depends(get_manager_auth_context),
) -> FilesListResponse:
    try:
        target_bucket = resolve_bucket(bucket)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    await _assert_bucket_exists_or_404(target_bucket)

    minio_client = get_minio_client()
    state_repo = get_file_state_repo()
    prefix = _document_key_prefix(auth)
    states = await asyncio.to_thread(
        state_repo.list,
        bucket=target_bucket,
        key_prefix=prefix,
        limit=max(limit + offset, 5_000),
        offset=0,
    )
    minio_objects = await asyncio.to_thread(
        _list_minio_objects,
        minio_client=minio_client,
        bucket=target_bucket,
        prefix=prefix or "",
    )

    state_by_key = {item.key: item for item in states}
    minio_by_key = {item["key"]: item for item in minio_objects}

    merged_items: list[FileListItem] = []
    for scoped_key in set(state_by_key) | set(minio_by_key):
        state_item = state_by_key.get(scoped_key)
        minio_item = minio_by_key.get(scoped_key)
        minio_exists = minio_item is not None
        indexed = state_item is not None
        merged_items.append(
            FileListItem(
                bucket=target_bucket,
                key=_document_public_key(auth, scoped_key),
                scoped_key=scoped_key,
                status=_resolve_file_status(indexed=indexed, minio_exists=minio_exists),
                size_bytes=int((minio_item or {}).get("size_bytes", 0) or 0),
                file_id=(state_item.file_id if state_item else None),
                etag=((state_item.etag if state_item and state_item.etag else None) or (minio_item or {}).get("etag")),
                version_id=(state_item.version_id if state_item else None),
                updated_at=((state_item.updated_at if state_item and state_item.updated_at else None) or (minio_item or {}).get("updated_at")),
            )
        )

    merged_items.sort(
        key=lambda item: (
            item.updated_at.timestamp() if item.updated_at else 0.0,
            item.key.lower(),
        ),
        reverse=True,
    )
    return FilesListResponse(items=merged_items[offset : offset + limit])


@router.get("/by-name/{reference_name:path}/download")
async def download_file_by_reference_name(
    reference_name: str,
    bucket: str | None = Query(default=None),
    auth: AuthContext = Depends(get_auth_context),
):
    try:
        target_bucket = resolve_bucket(bucket)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    return await _resolve_reference_download(
        user_id=auth.user_id,
        bucket=target_bucket,
        reference_name=reference_name,
    )


@router.get("/events/stream")
async def stream_file_status_events(
    request: Request,
    bucket: str | None = Query(default=None),
    auth: AuthContext = Depends(get_manager_auth_context),
):
    try:
        target_bucket = resolve_bucket(bucket)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not settings.KAFKA_BOOTSTRAP_SERVERS.strip() or not settings.INDEXING_STATUS_TOPIC.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Kafka status stream is not configured",
        )

    user_prefix = _document_key_prefix(auth)
    consumer = Consumer(
        {
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "group.id": f"api-file-status-{auth.user_id}-{uuid.uuid4()}",
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([settings.INDEXING_STATUS_TOPIC])

    async def iterator():
        last_ping_at = time.monotonic()
        try:
            yield _sse(
                "ready",
                {
                    "bucket": target_bucket,
                    "topic": settings.INDEXING_STATUS_TOPIC,
                },
            )

            while True:
                if await request.is_disconnected():
                    break

                message = await asyncio.to_thread(consumer.poll, 1.0)
                if message is None:
                    if time.monotonic() - last_ping_at >= 15:
                        last_ping_at = time.monotonic()
                        yield _sse("ping", {"ts": time.time()})
                    continue

                if message.error():
                    if message.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    yield _sse("error", {"message": str(message.error())})
                    continue

                try:
                    raw_payload = json.loads(message.value().decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(raw_payload, dict):
                    continue

                scoped_key = _parse_optional_text(raw_payload.get("key"))
                if not scoped_key:
                    continue
                if user_prefix and not scoped_key.startswith(user_prefix):
                    continue

                event_bucket = _parse_optional_text(raw_payload.get("bucket"))
                if event_bucket != target_bucket:
                    continue

                yield _sse(
                    "status",
                    {
                        "bucket": event_bucket,
                        "key": _document_public_key(auth, scoped_key),
                        "scoped_key": scoped_key,
                        "action": _parse_optional_text(raw_payload.get("action")) or "added",
                        "result": _parse_optional_text(raw_payload.get("result")) or "failed",
                        "message": _parse_optional_text(raw_payload.get("message")) or "File processing event received",
                        "chunks": raw_payload.get("chunks"),
                        "etag": _parse_optional_text(raw_payload.get("etag")),
                        "version_id": _parse_optional_text(raw_payload.get("version_id")),
                        "file_id": _parse_optional_text(raw_payload.get("file_id")),
                        "reason": _parse_optional_text(raw_payload.get("reason")),
                        "ts": raw_payload.get("ts"),
                    },
                )
        finally:
            consumer.close()

    return StreamingResponse(
        iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{object_key:path}/status", response_model=FileStatusResponse)
async def file_status(
    object_key: str,
    bucket: str | None = Query(default=None),
    auth: AuthContext = Depends(get_manager_auth_context),
) -> FileStatusResponse:
    try:
        target_bucket = resolve_bucket(bucket)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    scoped_key = _document_storage_key(auth, object_key)
    minio_client = get_minio_client()
    state_repo = get_file_state_repo()

    minio_exists = await asyncio.to_thread(minio_object_exists, minio_client, target_bucket, scoped_key)
    state_item = await asyncio.to_thread(state_repo.get, target_bucket, scoped_key)
    indexed = state_item is not None

    resolved_status = _resolve_file_status(indexed=indexed, minio_exists=minio_exists)

    return FileStatusResponse(
        bucket=target_bucket,
        key=_document_public_key(auth, scoped_key),
        scoped_key=scoped_key,
        minio_exists=minio_exists,
        indexed=indexed,
        status=resolved_status,
        file_id=(state_item.file_id if state_item else None),
        etag=(state_item.etag if state_item else None),
        version_id=(state_item.version_id if state_item else None),
        updated_at=(state_item.updated_at if state_item else None),
    )


@router.delete("/{object_key:path}", status_code=status.HTTP_202_ACCEPTED, response_model=FileAcceptedResponse)
async def delete_file(
    object_key: str,
    bucket: str | None = Query(default=None),
    auth: AuthContext = Depends(get_manager_auth_context),
) -> FileAcceptedResponse:
    try:
        target_bucket = resolve_bucket(bucket)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    await _assert_bucket_exists_or_404(target_bucket)
    scoped_key = _document_storage_key(auth, object_key)
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
    message = "Delete accepted; cleanup event queued for ingestion service"
    try:
        await asyncio.to_thread(
            _publish_indexing_input_event,
            bucket=target_bucket,
            key=scoped_key,
            event_type="file_deleted",
            force=True,
        )
    except RuntimeError as exc:
        logger.warning(
            "Delete cleanup publish failed for bucket=%s key=%s: %s",
            target_bucket,
            scoped_key,
            exc,
        )
        message = "Delete accepted in MinIO, but cleanup queue failed; retry delete if the item remains stuck"

    key_for_path = _document_public_key(auth, scoped_key)
    return FileAcceptedResponse(
        bucket=target_bucket,
        key=key_for_path,
        scoped_key=scoped_key,
        action="delete_requested",
        message=message,
        status_endpoint=f"/api/v1/files/{key_for_path}/status?bucket={target_bucket}",
    )
