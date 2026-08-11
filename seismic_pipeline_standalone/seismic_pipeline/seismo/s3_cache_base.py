"""Shared S3 client pooling for cache managers."""
from __future__ import annotations

from typing import Any, Dict, Optional


class S3ClientCacheMixin:
    """Mixin providing lazy, cached boto3 S3 client creation."""

    _CLIENT_CACHE: Dict[tuple, Any] = {}

    s3_config: Optional[Dict[str, Any]]

    def _s3_client_cache_key(self) -> tuple:
        cfg = self.s3_config or {}
        return (
            cfg.get("endpoint_url"),
            cfg.get("aws_access_key_id"),
            cfg.get("aws_secret_access_key"),
            cfg.get("region_name"),
        )

    def _get_s3_client(self):
        """Return a cached boto3 S3 client or None if config is missing."""
        if not self.s3_config:
            return None
        import boto3

        key = self._s3_client_cache_key()
        if key not in self._CLIENT_CACHE:
            self._CLIENT_CACHE[key] = boto3.client(
                self.s3_config.get("service_name", "s3"),
                **{k: v for k, v in self.s3_config.items() if k != "service_name" and v is not None},
            )
        return self._CLIENT_CACHE[key]
