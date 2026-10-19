"""
S3 / MinIO storage client for artifact persistence.
Integration Standard #7: S3-compatible storage.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import structlog

from app.config import get_settings

logger = structlog.get_logger()


class StorageClient:
    """
    Client for saving and retrieving generated audio and manifests.
    Connects to MinIO/S3 when configured, or falls back to local storage.
    """

    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self.bucket = self.settings.s3_bucket

        try:
            from minio import Minio
            # Strip http:// or https:// if provided in endpoint
            endpoint = self.settings.s3_endpoint
            secure = False
            if endpoint.startswith("https://"):
                endpoint = endpoint[len("https://"):]
                secure = True
            elif endpoint.startswith("http://"):
                endpoint = endpoint[len("http://"):]

            self.client = Minio(
                endpoint,
                access_key=self.settings.s3_access_key,
                secret_key=self.settings.s3_secret_key,
                secure=secure,
            )
            # Ensure bucket exists if connection succeeds
            try:
                if not self.client.bucket_exists(self.bucket):
                    self.client.make_bucket(self.bucket)
                    logger.info("minio_bucket_created", bucket=self.bucket)
            except Exception as e:
                logger.warning("minio_connection_failed_using_local", error=str(e))
                self.client = None

        except ImportError:
            logger.warning("minio_library_not_installed_using_local")
            self.client = None

    def upload_file(self, local_path: Path | str, object_name: str) -> str:
        """
        Upload a file to S3/MinIO.

        Returns:
            URL or path string for accessing the artifact.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        if self.client:
            try:
                content_type = "audio/wav" if local_path.suffix == ".wav" else "application/json"
                self.client.fput_object(
                    self.bucket,
                    object_name,
                    str(local_path),
                    content_type=content_type,
                )
                logger.info("file_uploaded_to_s3", bucket=self.bucket, object_name=object_name)
                return f"{self.settings.s3_endpoint}/{self.bucket}/{object_name}"
            except Exception as e:
                logger.error("s3_upload_failed_falling_back_to_local", error=str(e))

        # Fallback to local URL path
        return f"/output/{object_name}"


# Singleton instance
_storage_client: Optional[StorageClient] = None


def get_storage_client() -> StorageClient:
    """Get or create singleton storage client."""
    global _storage_client
    if _storage_client is None:
        _storage_client = StorageClient()
    return _storage_client
