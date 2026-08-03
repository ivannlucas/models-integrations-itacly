import logging

import pandas as pd
import numpy as np

from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.constants import SENSOR_COLUMNS, WINDOW_SIZE

logger = logging.getLogger(__name__)


def apply_digital_twin(df: pd.DataFrame, ts1_mean_train: float) -> pd.DataFrame:
    offset = 65.0 - ts1_mean_train
    df = df.copy()
    df["TS1"] = df["TS1"] + offset
    if "TS2" in df.columns:
        df["TS2"] = df["TS2"] + offset
    return df


def _resample_10hz(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Bins Time_Segundos to 0.1s and averages readings that land in the same bin.

    Mirrors the training pipeline's clean_and_resample (Time.round(1) then
    groupby(...).mean()) and the reference predictor's
    apply_digital_twin_inference. Raw exports commonly sample faster than the
    10Hz the model was trained on with a Time_Segundos column that already
    carries finer-than-0.1s precision (e.g. 0.01s steps), so no two rows are
    exact duplicates and a plain groupby is a no-op. Rounding first is what
    actually creates the 10Hz bins instead of assuming they already exist.
    """
    df = df.copy()
    df["Time_Segundos"] = df["Time_Segundos"].round(1)
    return df.groupby(group_cols)[SENSOR_COLUMNS].mean().reset_index()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # kind="stable" makes the tiebreak for equal Time_Segundos values explicit
    # (keeps first-seen order). Pandas' default "quicksort" is NOT guaranteed
    # stable, so without this, ties could be reordered unpredictably even
    # before considering upload order.
    df = df.sort_values("Time_Segundos", kind="stable").copy()
    X = df[SENSOR_COLUMNS]
    rmean = X.rolling(5, min_periods=1).mean().add_suffix("_rmean")
    rstd = X.rolling(5, min_periods=1).std().fillna(0).add_suffix("_rstd")
    lag = X.shift(1).bfill().add_suffix("_lag1")
    return pd.concat([df, rmean, rstd, lag], axis=1)


def build_dataframe_from_sensors(
    sensor_data: dict[str, list[float]],
    time_segundos: list[float] | None,
    cycle_id: int | None,
    ts1_mean_train: float,
    apply_digital_twin_flag: bool,
) -> pd.DataFrame:
    n_steps = len(sensor_data["PS1"])
    if time_segundos is None:
        time_segundos = [round(i * 0.1, 1) for i in range(n_steps)]

    times = time_segundos[:n_steps] if len(time_segundos) > n_steps else time_segundos
    if len(times) < n_steps:
        times = times + [times[-1] + 0.1 * (i + 1) for i in range(n_steps - len(times))]

    row = {"Time_Segundos": times}
    for col in SENSOR_COLUMNS:
        vals = sensor_data.get(col, [0.0] * n_steps)
        row[col] = vals[:n_steps] if len(vals) > n_steps else vals + [0.0] * (n_steps - len(vals))

    df = pd.DataFrame(row)
    if cycle_id is not None:
        df["Cycle_ID"] = cycle_id

    if apply_digital_twin_flag:
        df = apply_digital_twin(df, ts1_mean_train)

    df = engineer_features(df)
    feature_cols = [c for c in df.columns if c not in ("Cycle_ID", "Time_Segundos")]
    return df[feature_cols]


def build_dataframe_from_csv(
    data_path: str,
    ts1_mean_train: float,
    apply_digital_twin_flag: bool,
) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    has_cycle_id = "Cycle_ID" in df.columns
    if has_cycle_id and df["Cycle_ID"].nunique() > 1:
        df_10hz = _resample_10hz(df, ["Cycle_ID", "Time_Segundos"])
        groups = []
        for cid in df_10hz["Cycle_ID"].unique():
            g = df_10hz[df_10hz["Cycle_ID"] == cid].copy()
            if apply_digital_twin_flag:
                g = apply_digital_twin(g, ts1_mean_train)
            g = engineer_features(g)
            groups.append(g)
        df_out = pd.concat(groups, ignore_index=True)
    else:
        group_cols = ["Cycle_ID", "Time_Segundos"] if has_cycle_id else ["Time_Segundos"]
        df_10hz = _resample_10hz(df, group_cols)
        if apply_digital_twin_flag:
            df_10hz = apply_digital_twin(df_10hz, ts1_mean_train)
        df_out = engineer_features(df_10hz)

    # Derived from df_out (post-resample), not the raw CSV: row counts change
    # during resampling, so the raw Cycle_ID series would no longer align.
    cycle_ids = df_out.get("Cycle_ID")
    drop_cols = [c for c in ["Time_Segundos", "date"] if c in df_out.columns]
    feature_cols = [c for c in df_out.columns if c not in drop_cols]
    return df_out[feature_cols], cycle_ids


def pad_or_truncate(X: np.ndarray, label: str = "") -> np.ndarray:
    if X.shape[0] < WINDOW_SIZE:
        if label:
            logger.warning(
                "[%s] Only %d/%d timesteps available; padding %d with zeros. "
                "Prediction quality for this cycle is not guaranteed.",
                label, X.shape[0], WINDOW_SIZE, WINDOW_SIZE - X.shape[0],
            )
        pad = np.zeros((WINDOW_SIZE - X.shape[0], X.shape[1]))
        X = np.vstack([X, pad])
    elif X.shape[0] > WINDOW_SIZE:
        if label:
            logger.warning(
                "[%s] %d timesteps available, expected %d; keeping only the first %d "
                "(sorted by Time_Segundos). This usually means several physical cycles "
                "share the same Cycle_ID value in the source data - check upstream.",
                label, X.shape[0], WINDOW_SIZE, WINDOW_SIZE,
            )
        X = X[:WINDOW_SIZE, :]
    return X
