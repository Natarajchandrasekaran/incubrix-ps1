"""
Storage module for artifacts and audio files.
"""
from app.storage.client import StorageClient, get_storage_client

__all__ = ["StorageClient", "get_storage_client"]
