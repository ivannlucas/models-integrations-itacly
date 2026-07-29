"""Feature engineering and input handling for ml40 — faithful port of the AI team's code.

The engineering functions replicate inbox/a40/codigo/.../src/data_processing/preprocess.py
line by line (same formulas, same lag windows, same dropna behaviour) so that predictions on
raw sensor CSVs match the original pipeline exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.domain.services.exceptions import (
    InsufficientCycleHistoryError,
    UnknownDiagnosisSystemError,
)
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.constants import (
    CYCLE_COLUMNS,
    ENGINEERED_MARKER,
    MIN_HISTORY_MINUTES,
    RAW_INPUT_COLUMNS,
    SYSTEM_SIGNATURE,
    SYSTEMS,
)


# ── System / input-mode detection ─────────────────────────────────────────────

def detect_system(columns, system: str | None = None) -> str:
    """Return which subsystem an input belongs to, validating explicit choices too."""
    if system is not None:
        if system not in SYSTEMS:
            raise UnknownDiagnosisSystemError(
                f"Sistema '{system}' no reconocido; debe ser uno de {list(SYSTEMS)}."
            )
        return system
    cols = set(columns)
    for candidate in SYSTEMS:
        if SYSTEM_SIGNATURE[candidate] <= cols:
            return candidate
    raise UnknownDiagnosisSystemError(
        "Las columnas de entrada no corresponden a ningún subsistema conocido: se esperan "
        f"{sorted(SYSTEM_SIGNATURE['refrigeracion'])} (refrigeracion) o "
        f"{sorted(SYSTEM_SIGNATURE['aireado'])} (aireado)."
    )


def is_engineered(df: pd.DataFrame, system: str) -> bool:
    """True if the CSV already contains the engineered features (splits-style input)."""
    return ENGINEERED_MARKER[system] in df.columns


def validate_raw_input(df: pd.DataFrame, system: str) -> None:
    """Check the raw-sensor contract before feature engineering."""
    required = CYCLE_COLUMNS + RAW_INPUT_COLUMNS[system]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"El CSV de entrada ({system}) no cumple el contrato de sensores crudos; "
            f"faltan columnas: {missing}"
        )


def validate_history(df: pd.DataFrame, system: str) -> None:
    """Reject cycles shorter than the minimum history the lag features need."""
    min_rows = MIN_HISTORY_MINUTES[system]
    counts = df.groupby("run_id").size()
    short = counts[counts < min_rows]
    if not short.empty:
        detail = ", ".join(f"run {rid}: {n} filas" for rid, n in short.items())
        raise InsufficientCycleHistoryError(
            f"El sistema {system} requiere al menos {min_rows} minutos de histórico por ciclo "
            f"(run_id) para calcular lags y medias móviles; ciclos insuficientes: {detail}."
        )


# ── Refrigeración (port of extract_refrigeration_features) ───────────────────

def extract_refrigeration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thermodynamic indicators for the refrigeration system."""
    df = df.copy()
    df = df.sort_values(["run_id", "time_min"])
    df["T_error"] = df["T_cab"] - df["T_set"]
    df["T_lift"] = df["T_cond_sat"] - df["T_evap_sat"]
    df["T_cond_approach"] = df["T_cond_sat"] - df["T_amb"]
    df["T_spread"] = df["T_cond_sat"] - df["T_evap_sat"]
    df["T_cab_meas_diff"] = df["T_cab"] - df["T_cab_meas"]

    df["P_ratio"] = df["P_dis_bar"] / (df["P_suc_bar"] + 0.1)
    df["Power_per_diff"] = df["P_comp_W"] / (df["T_lift"] + 0.1)
    df["Q_est"] = df["P_comp_W"] / (df["T_cab"] - df["T_evap_sat"] + 1e-5)
    df["Sensor_error"] = df["T_cab"] - df["T_cab_meas"]

    df["T_cab_grad"] = df.groupby("run_id")["T_cab"].diff().fillna(0)
    df["P_suc_rate"] = df.groupby("run_id")["P_suc_bar"].diff().fillna(0)

    df["Eff_vol"] = (df["T_evap_sat"] + 273.15) / (df["T_cond_sat"] + 273.15)
    df["P_suc_norm"] = df["P_suc_bar"] / df["T_amb"]
    df["P_dis_error"] = df["P_dis_bar"] - (df["T_cond_sat"] * 0.25)
    df["T_subcooling_approx"] = df["T_cond_sat"] - df["T_amb"]
    df["Power_to_Pratio"] = df["P_comp_W"] / (df["P_ratio"] + 0.1)

    df["P_dis_volatility"] = (
        df.groupby("run_id")["P_dis_bar"].transform(lambda x: x.rolling(window=5).std()).fillna(0)
    )

    df["pressure_ratio"] = df["P_dis_bar"] / (df["P_suc_bar"] + 0.1)
    df["specific_work"] = df["P_comp_W"] / (df["Q_evap_W"] + 0.1)
    df["cop_degradation"] = df["COP"] / df.groupby("run_id")["COP"].transform("max")

    df["EEI"] = df["P_comp_W"] / (df["T_cab"] - df["T_evap_sat"] + 0.1)
    df["Thermal_Load_Index"] = (df["T_cab"] - df["T_evap_sat"]) * df["P_ratio"]
    return df


def create_refrigeration_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Lags, deltas and rolling statistics for refrigeration (drops warm-up rows)."""
    df = df.copy().sort_values(["run_id", "time_min"])

    lag_features = ["P_dis_bar", "T_cond_sat"]
    lags = [5, 15, 45]
    rolling_features = ["P_dis_bar", "T_cond_sat", "EEI"]
    windows = [15, 30]

    for feature in lag_features:
        for lag in lags:
            df[f"{feature}_lag_{lag}"] = df.groupby("run_id")[feature].shift(lag)
            df[f"{feature}_delta_{lag}"] = df[feature] - df[f"{feature}_lag_{lag}"]

    if "P_dis_error" in df.columns:
        df["P_dis_error_lag_100"] = df.groupby("run_id")["P_dis_error"].shift(100)

    for feature in rolling_features:
        for w in windows:
            df[f"{feature}_roll_std_{w}"] = df.groupby("run_id")[feature].transform(
                lambda x, w=w: x.rolling(w, min_periods=w // 2).std()
            )
            roll_mean = df.groupby("run_id")[feature].rolling(w, min_periods=w // 2).mean()
            df[f"{feature}_roll_mean_{w}"] = roll_mean.reset_index(level=0, drop=True)

    df["P_dis_bar_roll_std_20"] = (
        df.groupby("run_id")["P_dis_bar"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    )
    df["P_dis_bar_roll_mean_20"] = (
        df.groupby("run_id")["P_dis_bar"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    )
    df["Pdis_instability_20"] = df["P_dis_bar_roll_std_20"] / (df["P_dis_bar_roll_mean_20"] + 1e-5)

    if "T_cond_approach" in df.columns:
        df["cond_approach_std_20"] = (
            df.groupby("run_id")["T_cond_approach"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
        )

    df = df.dropna(subset=["Pdis_instability_20", "P_dis_bar_delta_15"])
    return df


def physics_indicators_refrigeration(df: pd.DataFrame) -> pd.DataFrame:
    """Per-run early/mean physics summaries used by the neurosymbolic rules."""
    df = df.copy()
    df["early_P_dis_error"] = df.groupby("run_id")["P_dis_error"].transform(lambda x: x.iloc[:100].mean())
    df["mean_P_dis_bar"] = df.groupby("run_id")["P_dis_bar"].transform("mean")
    return df


# ── Aireado (port of extract_aireado_features) ────────────────────────────────

def extract_aireado_features(df: pd.DataFrame) -> pd.DataFrame:
    """Psychrometric indicators for the aeration/curing system."""
    df_ext = df.copy()

    def calculate_vpd(temp, rh):
        es = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
        ea = es * (rh / 100.0)
        return es - ea

    df_ext["VPD"] = calculate_vpd(df_ext["T_cab"], df_ext["RH_cab"])
    df_ext["RH_error"] = df_ext["RH_cab"] - 75.0
    df_ext["Air_Flow_Ratio"] = df_ext["N_fan_Hz"] / (df_ext["Kg_embutido"] + 1.0)
    df_ext["Evap_Eff_Index"] = (df_ext["T_cab"] - df_ext["T_evap_sat"]) / (df_ext["RH_cab"] + 0.1)
    df_ext["Specific_Power_Load"] = df_ext["P_comp_W"] / (df_ext["Kg_embutido"] + 1.0)
    df_ext["Encostramiento_Risk"] = df_ext["N_fan_Hz"] / (df_ext["RH_cab"] + 1.0)
    return df_ext


def create_aireado_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Lags, deltas and rolling mean for aeration (keeps warm-up NaN rows, as the original)."""
    features_to_lag = ["RH_cab", "T_cab", "N_fan_Hz", "Evap_Eff_Index"]
    lags_aireado = [10, 30, 60]
    df_lagged = df.copy()
    for feat in features_to_lag:
        for lag in lags_aireado:
            df_lagged[f"{feat}_lag_{lag}"] = df_lagged.groupby("run_id")[feat].shift(lag)
            df_lagged[f"{feat}_delta_{lag}"] = df_lagged[feat] - df_lagged[f"{feat}_lag_{lag}"]

    df_lagged["RH_roll_mean_20"] = df_lagged.groupby("run_id")["RH_cab"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    return df_lagged


# ── Entry point ───────────────────────────────────────────────────────────────

def apply_feature_engineering(df: pd.DataFrame, system: str) -> pd.DataFrame:
    """Full raw→engineered pipeline for a system (src/main.py::apply_feature_engineering)."""
    if system == "refrigeracion":
        df = extract_refrigeration_features(df)
        df = create_refrigeration_lags(df)
        df = physics_indicators_refrigeration(df)
    elif system == "aireado":
        df = extract_aireado_features(df)
        df = create_aireado_lags(df)
    return df


def prepare_input(df: pd.DataFrame, system: str | None = None) -> tuple[pd.DataFrame, str]:
    """Detect system and input mode, engineer features if the CSV is raw.

    Returns (engineered_df with a clean positional index, system).
    """
    system = detect_system(df.columns, system)
    if is_engineered(df, system):
        prepared = df.sort_values(["run_id", "time_min"]) if set(CYCLE_COLUMNS) <= set(df.columns) else df
    else:
        validate_raw_input(df, system)
        validate_history(df, system)
        prepared = apply_feature_engineering(df.sort_values(["run_id", "time_min"]), system)
        if prepared.empty:
            raise InsufficientCycleHistoryError(
                f"Tras la ingeniería de variables no queda ninguna fila válida ({system}): "
                "el histórico por ciclo es demasiado corto para las ventanas de lags."
            )
    return prepared.reset_index(drop=True), system
