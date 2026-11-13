"""Builds telemetry windows from a batch CSV or an inline list of raw telemetry rows."""
from __future__ import annotations

import pandas as pd

from app.domain.services.exceptions import InvalidTelemetryInputError
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.common import (
    TrainConfig,
    FeatureArtifacts,
)
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.data import load_telemetry
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.datasets import (
    AssetSequence,
    make_sequences,
)
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.features import (
    build_feature_matrix,
    engineer_row_features,
)
from app.plugins.ml46_dairy_fouling_clog_detection.constants import RAW_FIXED_COLUMNS, SEQ_LEN


def build_raw_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build a raw telemetry DataFrame from a list of row dicts (inline predict)."""
    return pd.DataFrame(rows)


def validate_raw_columns(raw_df: pd.DataFrame) -> None:
    """Reject a CSV/row-set missing required raw telemetry columns (input_contract.py parity).

    Without this, load_telemetry()/engineer_row_features() silently fill missing measured
    columns with defaults/NaN and score a degraded-but-plausible window instead of rejecting
    the request — see src/predict/input_contract.py::validate_input_contract_columns in the
    delivered model repo.
    """
    missing = [c for c in RAW_FIXED_COLUMNS if c not in raw_df.columns]
    if missing:
        raise InvalidTelemetryInputError(f"CSV falta columnas requeridas: {missing}")


# 0/1 flags: an empty cell means 0. The vendored code assumes they are always filled (the
# delivered dataset has no blanks): a blank maintenance_active ended up as int(NaN) in the
# training metrics ("cannot convert float NaN to integer").
_FLAG_COLUMNS = ("maintenance_active", "maintenance_planned", "clog_event", "fouling_onset_event")


def normalize_raw_telemetry(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Fill the blanks the vendored pipeline cannot handle and check the timestamps.

    Only empty cells change, so a file without blanks (like the delivered dataset) is scored
    exactly as before:

    * 0/1 flags (maintenance_active, maintenance_planned, clog_event, fouling_onset_event): 0.
    * maintenance_type / fault_type: ``other`` inside a maintenance period (the maintenance
      happened but its type/cause is unknown), ``none`` elsewhere. A period whose cells were
      all empty made ``mode().iloc[0]`` fail with IndexError when deriving maintenances.

    Unparseable timestamps are rejected here with the offending rows and the expected format
    (the vendored message was "Some telemetry timestamps could not be parsed.").
    """
    df = raw_df.copy()
    if "timestamp" in df.columns:
        # Same parse as the vendored load_telemetry, so exactly what it would reject is caught.
        parsed = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        bad = df.index[parsed.isna()]
        if len(bad):
            examples = ", ".join(
                f"fila {i + 2}: {df.at[i, 'timestamp']!r}" for i in bad[:3]
            )
            raise InvalidTelemetryInputError(
                f"{len(bad)} valor(es) de timestamp no se pueden leer ({examples}). Formato "
                "esperado: ISO 8601 con zona horaria, p. ej. 2024-01-01T00:00:00+00:00."
            )
    for column in _FLAG_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    in_maintenance = (
        df["maintenance_active"] == 1 if "maintenance_active" in df.columns
        else pd.Series(False, index=df.index)
    )
    for column in ("maintenance_type", "fault_type"):
        if column in df.columns:
            blank = df[column].isna() | (df[column].astype(str).str.strip() == "")
            df[column] = df[column].astype(object)
            df.loc[blank & in_maintenance, column] = "other"
            df.loc[blank & ~in_maintenance, column] = "none"
    return df


def no_window_reason(raw_df: pd.DataFrame, seq_len: int = SEQ_LEN) -> str:
    """Why a telemetry file produced no valid window, for the error message."""
    reasons = [
        f"Ninguna ventana válida: hacen falta {seq_len} filas seguidas (1 por minuto) del mismo "
        "activo y ciclo cuya última fila tenga phase = production y maintenance_active = 0."
    ]
    if "asset_id" in raw_df.columns and len(raw_df):
        rows = raw_df.groupby("asset_id").size()
        reasons.append(f"Filas por activo: máximo {int(rows.max())} ({len(rows)} activo(s)).")
    if "phase" in raw_df.columns:
        phases = sorted(raw_df["phase"].dropna().astype(str).unique().tolist())
        if "production" not in phases:
            reasons.append(
                f"Valores de phase encontrados: {phases[:8]}; ninguno es 'production' (los "
                "valores van en inglés: production, CIP_alkaline, CIP_acid, idle, maintenance)."
            )
    return " ".join(reasons)


def prepare_sequences(
    raw_df: pd.DataFrame,
    train_cfg: TrainConfig,
    feature_artifacts: FeatureArtifacts,
) -> tuple[dict[str, AssetSequence], list[int], list[str]]:
    """Run the full feature-engineering pipeline and group rows into per-asset AssetSequences.

    Returns (sequences, feature_indices_for_no_clock, asset_ids).
    """
    validate_raw_columns(raw_df)
    telemetry_df = load_telemetry(normalize_raw_telemetry(raw_df), train_cfg, require_targets=False)
    telemetry_df = engineer_row_features(telemetry_df)
    telemetry_df, full_feature_names, _ = build_feature_matrix(
        telemetry_df, feature_artifacts, [], train_cfg
    )
    sequences = make_sequences(telemetry_df, train_cfg, full_feature_names)

    feature_to_idx = {name: i for i, name in enumerate(full_feature_names)}
    feature_indices = [feature_to_idx[name] for name in feature_artifacts.no_clock_feature_names]
    asset_ids = sorted(telemetry_df["asset_id"].astype(str).unique().tolist())
    return sequences, feature_indices, asset_ids


def last_window_only(
    sequences: dict[str, AssetSequence],
    feature_indices: list[int],
    seq_len: int = SEQ_LEN,
) -> tuple[str, int]:
    """Return the (sequence_id, end_idx) of the single most recent valid window across all
    sequences.

    Used by predict_inline, which always scores the latest point in the submitted history.
    """
    _ = feature_indices
    best: tuple[str, int, pd.Timestamp] | None = None
    for sequence_id, seq in sequences.items():
        meta = seq.meta
        valid_end = (meta["phase"] == "production") & (
            meta["maintenance_active"].fillna(0).astype(int) == 0
        )
        for end_idx in range(len(seq.timestamps) - 1, seq_len - 2, -1):
            if bool(valid_end.iloc[end_idx]):
                ts = pd.Timestamp(meta["timestamp"].iloc[end_idx])
                # pylint: disable-next=unsubscriptable-object  # best is not None here
                if best is None or ts > best[2]:
                    best = (sequence_id, end_idx, ts)
                break
    if best is None:
        return "", -1
    return best[0], best[1]
