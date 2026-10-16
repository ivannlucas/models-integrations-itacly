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
        self.objects[key] = data

    def get(self, key: str) -> bytes:
        return self.objects[key]

    def keys(self) -> list[str]:
        return sorted(self.objects)

    # ── boto3 surface ──
    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        fake = self

        class _Paginator:
            def paginate(self, Bucket: str, Prefix: str):  # noqa: N803 — boto3 names
                contents = [{"Key": k, "Size": len(v)} for k, v in sorted(fake.objects.items())
                            if k.startswith(Prefix)]
                return [{"Contents": contents}] if contents else [{}]

        return _Paginator()

    def download_file(self, bucket: str, key: str, path: str) -> None:
        Path(path).write_bytes(self.objects[key])

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        if self.fail_upload_on and key.endswith(self.fail_upload_on):
            raise ConnectionError(f"s3 caído subiendo {key}")
        self.objects[key] = Path(path).read_bytes()

    def get_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket: str, Key: str, Body) -> dict:  # noqa: N803
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        return {}
