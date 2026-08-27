"""Helpers to persist and validate artifact metadata for reproducible runs."""

from __future__ import annotations

import json
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def relative_posix_path(path: Path, project_root: Path) -> str:
    """Express ``path`` relative to ``project_root`` using forward slashes.

    Storing portable, root-relative POSIX paths keeps the artifact metadata
    independent from the training machine: a path such as
    ``data/split/dataset.csv`` resolves correctly on Windows, Linux and macOS.
    Falls back to the POSIX form of the original path when it lives outside the
    project root.
    """

    resolved = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def portable_path_name(raw_path: str) -> str:
    """Return the final path component, tolerating Windows or POSIX separators.

    ``Path("C:\\\\...\\\\config-feedforward.txt").name`` returns the whole string
    on Linux because backslashes are not separators there. Normalizing
    backslashes to forward slashes first makes basename extraction reliable on
    every platform, including legacy metadata that stored absolute Windows paths.
    """

    return PurePosixPath(str(raw_path).replace("\\", "/")).name


def _stored_path_is_absolute(normalized_path: str) -> bool:
    """Detect absolute paths in either POSIX or Windows form."""

    return PurePosixPath(normalized_path).is_absolute() or PureWindowsPath(normalized_path).is_absolute()


def resolve_stored_path(raw_path: str, project_root: Path | None) -> Path:
    """Resolve a metadata-stored path to a usable filesystem path.

    Modern metadata stores root-relative POSIX paths; legacy metadata may store
    absolute paths. Relative paths are resolved against ``project_root`` (or the
    current working directory when it is not provided).
    """

    normalized = str(raw_path).replace("\\", "/")
    if _stored_path_is_absolute(normalized):
        return Path(raw_path)
    base = Path(project_root) if project_root is not None else Path.cwd()
    return base / normalized


def model_metadata_path(model_path: Path) -> Path:
    """Return sidecar metadata path for a serialized model artifact."""

    return model_path.with_suffix(f"{model_path.suffix}.metadata.json")


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 checksum for a file.

    Args:
        file_path: Path to file.
        chunk_size: Read chunk size in bytes.

    Returns:
        str: Hex digest.
    """

    digest = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        while True:
            block = file_obj.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _relativize(path: Path, project_root: Path) -> str:
    """Return `path` relative to `project_root` when possible, else its name.

    Metadata must be portable across machines: absolute paths from the training
    host (e.g. ``C:\\Users\\IA\\...``) are meaningless on an auditor's checkout
    and previously broke inference. We therefore store repository-relative
    paths so the artifact travels with the repo.
    """

    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_training_metadata(
    *,
    project_root: Path,
    dataset_path: Path,
    neat_config_path: Path,
    sample_size: int,
    generations: int,
    random_state: int,
    strategies: tuple[str, ...],
    input_columns: tuple[str, ...],
    uses_temperature_as_policy_input: bool,
    model_sha256: str,
    dataset_sha256: str,
    neat_config_sha256: str,
    physical_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Build metadata payload for a trained winner genome artifact.

    Args:
        physical_ranges: Min/max physical bounds (per column) of the training
            scaler. Persisting them in the artifact makes inference
            self-contained: it no longer needs to re-derive bounds from the
            input dataset filename or a colocated scaler file.
    """

    payload: dict[str, Any] = {
        # schema_version 2 adds the embedded `physical_ranges` block so that
        # inference becomes independent from the input dataset naming.
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        # Paths are stored relative to project root for portability. Inference
        # never reads the training dataset (physical ranges are embedded here),
        # so these fields are provenance only.
        "project_root": ".",
        "dataset_path": _relativize(dataset_path, project_root),
        "neat_config_path": _relativize(neat_config_path, project_root),
        "sample_size": int(sample_size),
        "generations": int(generations),
        "random_state": int(random_state),
        "strategies": list(strategies),
        "input_columns": list(input_columns),
        "uses_temperature_as_policy_input": bool(uses_temperature_as_policy_input),
        "model_sha256": str(model_sha256),
        "dataset_sha256": str(dataset_sha256),
        "neat_config_sha256": str(neat_config_sha256),
    }
    if physical_ranges is not None:
        payload["physical_ranges"] = {
            str(column): [float(bounds[0]), float(bounds[1])]
            for column, bounds in physical_ranges.items()
        }
    return payload


def write_json_metadata(output_path: Path, payload: dict[str, Any]) -> None:
    """Persist a metadata dictionary as UTF-8 JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def validate_model_metadata_for_inference(
    *,
    model_path: Path,
    neat_config_path: Path,
    expected_input_columns: tuple[str, ...],
    expected_strategies: tuple[str, ...] | None,
    logger: logging.Logger,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Validate model sidecar metadata before running inference.

    Args:
        project_root: Base directory used to resolve root-relative metadata
            paths (schema_version 2). When omitted, relative paths are resolved
            against the current working directory.

    Raises:
        FileNotFoundError: If sidecar metadata does not exist.
        ValueError: If metadata is malformed or incompatible.
    """

    metadata_path = model_metadata_path(model_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata sidecar not found: {metadata_path}. "
            "Run training with metadata generation enabled."
        )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_keys = {
        "schema_version",
        "dataset_path",
        "neat_config_path",
        "sample_size",
        "generations",
        "random_state",
        "strategies",
        "input_columns",
        "uses_temperature_as_policy_input",
        "model_sha256",
        "dataset_sha256",
        "neat_config_sha256",
    }
    missing = sorted(required_keys.difference(payload.keys()))
    if missing:
        raise ValueError(f"Model metadata is incomplete. Missing keys: {missing}")

    if int(payload["schema_version"]) not in (1, 2):
        raise ValueError(
            f"Unsupported model metadata schema_version={payload['schema_version']}. "
            "Expected schema_version 1 or 2."
        )

    expected_config_name = neat_config_path.name
    metadata_config_name = portable_path_name(str(payload["neat_config_path"]))
    if metadata_config_name != expected_config_name:
        logger.warning(
            "Model metadata NEAT config name ('%s') differs from the runtime config "
            "name ('%s'). Continuing, but verify they are compatible.",
            metadata_config_name,
            expected_config_name,
        )

    if bool(payload["uses_temperature_as_policy_input"]):
        raise ValueError(
            "Incompatible model metadata: policy uses process_temperature_c as input. "
            "Current inference policy requires temperature to be excluded from ML inputs "
            "and used only in emissions simulation."
        )

    metadata_input_columns = tuple(str(column) for column in payload["input_columns"])
    if metadata_input_columns != expected_input_columns:
        raise ValueError(
            "Input feature order mismatch between trained model metadata and runtime policy. "
            f"Metadata={metadata_input_columns}, runtime={expected_input_columns}."
        )

    if "process_temperature_c" in metadata_input_columns:
        raise ValueError(
            "Invalid metadata: process_temperature_c cannot be part of neural policy inputs."
        )

    if len(metadata_input_columns) != 8:
        raise ValueError(
            "Invalid metadata: neural policy expects exactly 8 ordered input columns."
        )

    if expected_strategies is not None:
        metadata_strategies = tuple(str(strategy) for strategy in payload["strategies"])
        if metadata_strategies != expected_strategies:
            raise ValueError(
                "Strategy order mismatch between trained model metadata and runtime inference. "
                f"Metadata={metadata_strategies}, runtime={expected_strategies}."
            )

    # The loaded genome file MUST be the exact artifact referenced by metadata:
    # this is the only hash that affects inference correctness.
    runtime_model_hash = compute_sha256(model_path)
    if runtime_model_hash != str(payload["model_sha256"]):
        raise ValueError(
            "Model hash mismatch. The loaded model file is not the exact artifact referenced "
            "by metadata."
        )

    # Training-dataset provenance is a soft, best-effort check. Inference does
    # NOT read the training dataset (physical ranges are embedded in metadata),
    # and the stored path is machine-relative, so a missing or changed training
    # file must never block a portable inference run. We only log provenance.
    dataset_path_str = str(payload["dataset_path"])
    dataset_candidates = [
        Path(dataset_path_str),
        model_path.resolve().parents[2] / dataset_path_str
        if len(model_path.resolve().parents) >= 3
        else Path(dataset_path_str),
    ]
    resolved_dataset = next((p for p in dataset_candidates if p.exists()), None)
    if resolved_dataset is None:
        logger.warning(
            "Training dataset referenced by metadata ('%s') is not present. "
            "This is expected on a portable checkout and does not affect inference "
            "(physical ranges are embedded in the model metadata).",
            dataset_path_str,
        )
    elif compute_sha256(resolved_dataset) != str(payload["dataset_sha256"]):
        logger.warning(
            "Training dataset hash differs from metadata provenance for %s. "
            "Inference is unaffected; regenerate the model to refresh provenance.",
            resolved_dataset,
        )

    # NEAT config compatibility: a mismatch may be cosmetic (comments) or
    # semantic. Warn rather than hard-fail so inference stays portable, while
    # still surfacing potential drift for auditing.
    runtime_neat_hash = compute_sha256(neat_config_path)
    if runtime_neat_hash != str(payload["neat_config_sha256"]):
        logger.warning(
            "NEAT config hash differs from metadata provenance. Verify the runtime "
            "config is structurally compatible with the trained genome (input/output "
            "counts and feed-forward mode)."
        )

    logger.info("Validated model metadata sidecar: %s", metadata_path)
    return payload
