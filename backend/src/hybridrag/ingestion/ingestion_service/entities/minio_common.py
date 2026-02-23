from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Optional, Union
import aioboto3
from botocore.config import Config
from src.config.settings import settings

class AsyncMinioClient:
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "us-east-1",
        connect_timeout: int = 10,
        read_timeout: int = 30,
        max_attempts: int = 3,
    ):
        self.endpoint_url = endpoint_url or settings.MINIO_ENDPOINT
        self.access_key = access_key or settings.MINIO_ACCESS_KEY
        self.secret_key = secret_key or settings.MINIO_SECRET_KEY

        if not self.endpoint_url or not self.endpoint_url.startswith(("http://", "https://")):
            raise ValueError("MINIO_ENDPOINT phải dạng http(s)://host:port")
        if not self.access_key or not self.secret_key:
            raise ValueError("Thiếu MINIO_ACCESS_KEY / MINIO_SECRET_KEY")

        self._session = aioboto3.Session()
        self._config = Config(
            region_name=region,
            retries={"max_attempts": max_attempts, "mode": "adaptive"},
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )

    @asynccontextmanager
    async def _client(self):
        async with self._session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=self._config,
        ) as s3:
            yield s3

    async def get(self, bucket: str, key: str) -> BytesIO:
        async with self._client() as s3:
            resp = await s3.get_object(Bucket=bucket, Key=key)
            return BytesIO(await resp["Body"].read())

    async def info(self, bucket: str, key: str) -> dict:
        async with self._client() as s3:
            r = await s3.head_object(Bucket=bucket, Key=key)
            etag = r.get("ETag")
            if isinstance(etag, str):
                etag = etag.strip('"')
            return {
                "last_modified": r.get("LastModified"),
                "content_length": r.get("ContentLength"),
                "content_type": r.get("ContentType"),
                "etag": etag,
            }

    async def delete(self, bucket: str, key: Union[str, List[str]]) -> None:
        async with self._client() as s3:
            if isinstance(key, str):
                await s3.delete_object(Bucket=bucket, Key=key)
            elif isinstance(key, list) and key:
                await s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": k} for k in key]},
                )