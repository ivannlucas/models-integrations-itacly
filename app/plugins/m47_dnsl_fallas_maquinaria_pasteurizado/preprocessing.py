import pandas as pd
import numpy as np

from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.constants import SENSOR_COLUMNS, WINDOW_SIZE


def apply_digital_twin(df: pd.DataFrame, ts1_mean_train: float) -> pd.DataFrame:
    offset = 65.0 - ts1_mean_train
    df = df.copy()
    df["TS1"] = df["TS1"] + offset
    if "TS2" in df.columns:
        df["TS2"] = df["TS2"] + offset
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Time_Segundos").copy()
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
    cycle_ids = df.get("Cycle_ID")
    if cycle_ids is not None and cycle_ids.nunique() > 1:
        groups = []
        for cid in df["Cycle_ID"].unique():
            g = df[df["Cycle_ID"] == cid].copy()
            if apply_digital_twin_flag:
                g = apply_digital_twin(g, ts1_mean_train)
            g = engineer_features(g)
            groups.append(g)
        df_out = pd.concat(groups, ignore_index=True)
    else:
        if apply_digital_twin_flag:
            df = apply_digital_twin(df, ts1_mean_train)
        df_out = engineer_features(df)

    drop_cols = [c for c in ["Time_Segundos", "date"] if c in df_out.columns]
    feature_cols = [c for c in df_out.columns if c not in drop_cols]
    return df_out[feature_cols], cycle_ids


def pad_or_truncate(X: np.ndarray) -> np.ndarray:
    if X.shape[0] < WINDOW_SIZE:
        pad = np.zeros((WINDOW_SIZE - X.shape[0], X.shape[1]))
        X = np.vstack([X, pad])
    elif X.shape[0] > WINDOW_SIZE:
        X = X[:WINDOW_SIZE, :]
    return X
