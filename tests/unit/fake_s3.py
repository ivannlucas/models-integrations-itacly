"""In-memory stand-in for the boto3 S3 client calls ArtifactStore makes (single bucket)."""
from __future__ import annotations

import io
from pathlib import Path

from botocore.exceptions import ClientError


class FakeS3:
    """Supports list_objects_v2 pagination, download_file, upload_file, get_object, put_object."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_upload_on: str | None = None   # upload_file raises for keys ending with this

    # ── helpers for tests ──
    def put(self, key: str, data: bytes) -> None:
        """Stores ``data`` under ``key``."""
        self.objects[key] = data

    def get(self, key: str) -> bytes:
        """Returns the bytes stored under ``key``."""
        return self.objects[key]

    def keys(self) -> list[str]:
        """Returns the stored keys, sorted."""
        return sorted(self.objects)

    # ── boto3 surface ──
    def get_paginator(self, name: str):
        """Returns a one-page paginator for ``list_objects_v2``."""
        assert name == "list_objects_v2"
        fake = self

        class _Paginator:
            def paginate(
                self, Bucket: str, Prefix: str,  # noqa: N803  # pylint: disable=invalid-name
            ):  # boto3 names
                """Returns a single page with every object whose key starts with ``Prefix``."""
                _ = Bucket
                contents = [{"Key": k, "Size": len(v)} for k, v in sorted(fake.objects.items())
                            if k.startswith(Prefix)]
                return [{"Contents": contents}] if contents else [{}]

        return _Paginator()

    def download_file(self, bucket: str, key: str, path: str) -> None:
        """Writes the object stored under ``key`` to ``path``."""
        _ = bucket
        Path(path).write_bytes(self.objects[key])

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        """Stores the file at ``path`` under ``key`` (raises if ``fail_upload_on`` matches)."""
        _ = bucket
        if self.fail_upload_on and key.endswith(self.fail_upload_on):
            raise ConnectionError(f"s3 caído subiendo {key}")
        self.objects[key] = Path(path).read_bytes()

    def get_object(
        self, Bucket: str, Key: str,  # noqa: N803  # pylint: disable=invalid-name
    ) -> dict:  # boto3 names
        """Returns ``{"Body": stream}`` or raises a NoSuchKey ClientError."""
        _ = Bucket
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(
        self, Bucket: str, Key: str, Body,  # noqa: N803  # pylint: disable=invalid-name
    ) -> dict:  # boto3 names
        """Stores ``Body`` (bytes or str, utf-8 encoded) under ``Key``."""
        _ = Bucket
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        return {}
