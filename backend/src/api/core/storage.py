from __future__ import annotations
from urllib.parse import urlparse
from minio import Minio
from minio.error import S3Error
from src.config.settings import settings


def build_minio_client() -> Minio:
    raw_endpoint = settings.MINIO_ENDPOINT.strip()
    if raw_endpoint.startswith(("http://", "https://")):
        parsed = urlparse(raw_endpoint)
        endpoint = parsed.netloc
        secure = parsed.scheme == "https"
    else:
        endpoint = raw_endpoint
        secure = settings.MINIO_SECURE

    access_key = settings.MINIO_ACCESS_KEY or settings.MINIO_ROOT_USER
    secret_key = settings.MINIO_SECRET_KEY or settings.MINIO_ROOT_PASSWORD
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError("MinIO settings are not configured")

    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def resolve_bucket(raw_bucket: str | None) -> str:
    bucket = (raw_bucket or settings.MINIO_BUCKET_NAME).strip()
    if not bucket:
        raise ValueError("bucket is empty and MINIO_BUCKET_NAME is not configured")
    return bucket


def scope_object_key(user_id: str, object_key: str) -> str:
    key = object_key.strip().lstrip("/")
    if not key:
        raise ValueError("object_key must not be empty")
    prefix = f"{user_id}/"
    return key if key.startswith(prefix) else f"{prefix}{key}"


def unscope_object_key(user_id: str, object_key: str) -> str:
    prefix = f"{user_id}/"
    if object_key.startswith(prefix):
        return object_key[len(prefix) :]
    return object_key


def minio_object_exists(client: Minio, bucket: str, object_key: str) -> bool:
    try:
        client.stat_object(bucket, object_key)
        return True
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            return False
        raise