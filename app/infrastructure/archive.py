"""Bounded ZIP extraction for the ZIPs users send to /train and batch /predict.

``ZipFile.extractall`` already neutralises ``..`` and absolute member paths (no zip-slip), but it
writes whatever the archive expands to: a few MB can inflate to hundreds of GB and fill the pod's
disk. ``safe_extract_zip`` streams every member and stops at configurable limits.

The limits are sized for the real datasets of the plugins that take ZIPs (images, video frames,
audio, COCO/CVAT annotations), which barely compress, so the compression ratio is what catches a
zip bomb; total size and file count are a backstop. All three can be raised per deployment.
"""
from __future__ import annotations

import logging
import os
import shutil
import zipfile

from app.domain.services.exceptions import ArchiveLimitExceededError

logger = logging.getLogger(__name__)

GIB = 1024 ** 3
# Defaults; ARCHIVE_MAX_TOTAL_BYTES / ARCHIVE_MAX_FILES / ARCHIVE_MAX_RATIO override them (see
# .env.example). Read on every call, not at import: .env is loaded by artifact_store, which some
# plugins import after this module.
DEFAULT_MAX_TOTAL_BYTES = 10 * GIB
DEFAULT_MAX_FILES = 200_000
# Expanded/compressed size per member. JPEG/PNG/WAV/MP4 stay near 1x and JSON/CSV around 10-50x;
# zip bombs are in the thousands. Small members are exempt (a tiny file can have any ratio).
DEFAULT_MAX_RATIO = 200
RATIO_EXEMPT_BYTES = 1024 ** 2
_CHUNK = 1024 ** 2


def _limit(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        logger.warning("%s=%r no es un entero; se usa %d", name, raw, default)
        return default


def _member_path(dest_real: str, name: str) -> str:
    path = os.path.realpath(os.path.join(dest_real, name))
    if path != dest_real and not path.startswith(dest_real + os.sep):
        raise ArchiveLimitExceededError(f"El ZIP contiene una ruta fuera del destino: {name!r}")
    return path


def safe_extract_zip(zip_path: str, dest_dir: str) -> None:
    """Extract *zip_path* into *dest_dir* within the size, file-count and ratio limits.

    Raises ``ArchiveLimitExceededError`` (→ 413) when a limit is exceeded and ``ValueError``
    (→ 400 in /train) when the file is not a valid ZIP. Partially extracted files are left in
    *dest_dir*; callers already extract into a temp dir that they remove in ``finally``.
    """
    max_total = _limit("ARCHIVE_MAX_TOTAL_BYTES", DEFAULT_MAX_TOTAL_BYTES)
    max_files = _limit("ARCHIVE_MAX_FILES", DEFAULT_MAX_FILES)
    max_ratio = _limit("ARCHIVE_MAX_RATIO", DEFAULT_MAX_RATIO)
    try:
        zf = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"El fichero no es un ZIP válido: {exc}") from exc

    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > max_files:
            raise ArchiveLimitExceededError(
                f"El ZIP contiene {len(members)} ficheros; el máximo es {max_files} (ARCHIVE_MAX_FILES)."
            )
        declared = sum(m.file_size for m in members)
        if declared > max_total:
            raise ArchiveLimitExceededError(
                f"El ZIP ocupa {declared / GIB:.1f} GiB descomprimido; el máximo es "
                f"{max_total / GIB:.1f} GiB (ARCHIVE_MAX_TOTAL_BYTES)."
            )
        os.makedirs(dest_dir, exist_ok=True)
        free = shutil.disk_usage(dest_dir).free
        if declared > free:
            raise ArchiveLimitExceededError(
                f"No hay espacio en disco para descomprimir el ZIP ({declared / GIB:.1f} GiB "
                f"necesarios, {free / GIB:.1f} GiB libres)."
            )

        dest_real = os.path.realpath(dest_dir)
        written_total = 0
        for member in zf.infolist():
            target = _member_path(dest_real, member.filename)
            if member.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            # Count what is really written: the sizes in the ZIP headers are not trusted.
            member_cap = max(member.compress_size * max_ratio, RATIO_EXEMPT_BYTES)
            written = 0
            with zf.open(member) as src, open(target, "wb") as dst:
                while chunk := src.read(_CHUNK):
                    written += len(chunk)
                    written_total += len(chunk)
                    if written > member_cap:
                        raise ArchiveLimitExceededError(
                            f"'{member.filename}' se expande más de {max_ratio}x su tamaño "
                            "comprimido (posible zip bomb, ARCHIVE_MAX_RATIO)."
                        )
                    if written_total > max_total:
                        raise ArchiveLimitExceededError(
                            f"El ZIP supera {max_total / GIB:.1f} GiB descomprimido "
                            "(ARCHIVE_MAX_TOTAL_BYTES)."
                        )
                    dst.write(chunk)
    logger.info("Extracted %d files (%.1f MiB) from %s", len(members), written_total / 1024 ** 2, zip_path)
