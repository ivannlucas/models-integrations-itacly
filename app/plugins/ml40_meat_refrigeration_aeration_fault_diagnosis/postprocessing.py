"""Inference core + neurosymbolic layer for ml40 — faithful port of the AI team's code.

run_inference, apply_neurosymbolic_rules and apply_run_voting replicate
inbox/a40/codigo/.../src/predict/predictor.py and src/predict/postprocess.py. The
neurosymbolic thresholds come from the delivered {system}_thresholds.yaml artifacts instead
of being re-read from a config file on every call.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.constants import (
    CLASS_MAPPINGS,
    DROP_COLS,
    HEALTH_THRESHOLD_PCT,
    METADATA_COLS,
    REFRIGERACION_BINARY_COLS,
)

logger = logging.getLogger(__name__)


def run_inference(model, scaler, df: pd.DataFrame, system: str) -> tuple[np.ndarray, np.ndarray]:
    """Clean, align and scale the engineered frame, then predict (port of run_inference)."""
    to_drop = list(set(DROP_COLS[system] + METADATA_COLS))
    x_cleaned = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        missing = [c for c in expected_features if c not in x_cleaned.columns]
        if missing:
            raise ValueError(
                f"El CSV de entrada ({system}) no contiene las features que espera el modelo: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        x_cleaned = x_cleaned[expected_features]

    if system == "refrigeracion" and scaler is not None:
        if hasattr(scaler, "feature_names_in_"):
            cols = list(scaler.feature_names_in_)
            # Scaled values are float64; the source columns can be int64 (e.g. whole-number
            # sensor readings), so assigning back in place silently keeps the old dtype and
            # truncates/corrupts the scaled floats instead of raising — cast first.
            x_cleaned = x_cleaned.astype({c: "float64" for c in cols})
            x_cleaned.loc[:, cols] = scaler.transform(x_cleaned[cols])
        else:
            num_cols = [c for c in x_cleaned.columns if c not in REFRIGERACION_BINARY_COLS]
            x_cleaned = x_cleaned.astype({c: "float64" for c in num_cols})
            x_cleaned.loc[:, num_cols] = scaler.transform(x_cleaned[num_cols])

    y_pred = model.predict(x_cleaned)
    y_probs = model.predict_proba(x_cleaned)
    return y_pred, y_probs


def apply_neurosymbolic_rules(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    system: str,
    thresholds: dict,
) -> np.ndarray:
    """Correct ML predictions with hard physics rules (port of apply_neurosymbolic_rules).

    ``df`` must be the engineered, UNSCALED frame with a clean positional index aligned with
    ``y_pred`` (the original code indexes the numpy array with boolean masks from ``df``).
    """
    y_final = y_pred.copy()
    dyn = thresholds or {}
    mapping = CLASS_MAPPINGS[system]
    ids = {name: fault_id for fault_id, name in mapping.items()}

    if system == "refrigeracion":
        id_nc = ids.get("NON_CONDENSABLES")
        id_cf = ids.get("COND_FOUL_SEVERE")
        id_norm = ids.get("NORMAL")
        id_drift_plus = ids.get("SENSOR_DRIFT_PLUS")
        id_drift_minus = ids.get("SENSOR_DRIFT_MINUS")
        id_uc_severe = ids.get("UNDERCHARGE_SEVERE")
        id_ineff = ids.get("COMP_INEFFICIENCY")

        t_nc_low = dyn.get("nc_low", -0.264)
        t_cf_high = dyn.get("cf_high", 10.57)
        uc_p_gate = dyn.get("uc_p_gate", df["P_suc_bar"].quantile(0.1))
        eff_vol_limit = dyn.get("eff_vol_limit", 0.6)
        drift_limit = dyn.get("drift_limit", 2.0)

        if "T_cab_meas_diff" in df.columns:
            y_final[(df["T_cab_meas_diff"] > drift_limit).to_numpy()] = id_drift_minus
            y_final[(df["T_cab_meas_diff"] < -drift_limit).to_numpy()] = id_drift_plus

        if "P_suc_bar" in df.columns and "SH_K" in df.columns:
            mask_uc = (df["P_suc_bar"] < uc_p_gate) & (df["SH_K"] > 15)
            y_final[mask_uc.to_numpy()] = id_uc_severe

        if "Eff_vol" in df.columns:
            mask_eff = (df["Eff_vol"] < eff_vol_limit).to_numpy() & (y_pred == id_norm)
            y_final[mask_eff] = id_ineff

        # Batch-level thermomechanical profile: NC vs CF disambiguation. NOTE: this rule
        # depends on the MEAN over all in-scope rows of the batch, so predictions are
        # batch-dependent by the AI team's design (see manifest known_issues).
        if "early_P_dis_error" in df.columns and "T_cond_approach" in df.columns:
            in_scope_mask = (y_pred == id_nc) | (y_pred == id_cf)
            if in_scope_mask.any():
                mean_p_dis = df.loc[in_scope_mask, "early_P_dis_error"].mean()
                mean_approach = df.loc[in_scope_mask, "T_cond_approach"].mean()
                if mean_approach > t_cf_high or mean_p_dis > t_nc_low:
                    mask_pred_nc = y_pred == id_nc
                    mask_pred_cf = y_pred == id_cf
                    y_final[mask_pred_nc] = id_cf
                    y_final[mask_pred_cf] = id_nc

    elif system == "aireado":
        if "Encostramiento_Risk" in df.columns:
            limit_enc = dyn.get("encostramiento_risk", 0.90)
            mask_enc = (df["Encostramiento_Risk"] > limit_enc) & (df["RH_cab"] < 68)
            y_final[mask_enc.to_numpy()] = ids.get("ENCOSTRAMIENTO", 1)

        if "N_fan_Hz" in df.columns:
            limit_fan = dyn.get("fan_fail_hz", 5.0)
            # The original looks up "FALLO VENTILADOR" (with a space), which never matches the
            # mapping key FALLO_VENTILADOR and falls back to 3 — same class id, kept faithful.
            mask_vent = (df["N_fan_Hz"] < limit_fan).to_numpy() & (y_pred == ids.get("NORMAL", 0))
            y_final[mask_vent] = ids.get("FALLO VENTILADOR", 3)

    return y_final


def apply_run_voting(df: pd.DataFrame, y_pred: np.ndarray) -> np.ndarray:
    """Majority vote per cycle (port of apply_run_voting)."""
    df_temp = df.copy()
    df_temp["y_pred"] = y_pred
    return df_temp.groupby("run_id")["y_pred"].transform(lambda x: x.value_counts().idxmax()).values


def collapse_by_run(df: pd.DataFrame, system: str) -> pd.DataFrame:
    """Collapse row-level results into one diagnosis per run (port of save_predictions).

    Expects ``prediction`` and ``confidence`` columns already present. Keeps fault_id/fault
    when the input was labeled (evaluation mode).
    """
    reference_cols = ["fault_id", "fault"]
    result_cols = ["prediction", "confidence"]
    present_ref = [c for c in reference_cols if c in df.columns]
    existing_cols = ["run_id"] + present_ref + result_cols

    agg_rules: dict[str, str] = {}
    for col in existing_cols:
        if col == "run_id":
            continue
        agg_rules[col] = "mean" if col == "confidence" else "first"

    df_runs = df[existing_cols].groupby("run_id").agg(agg_rules).reset_index()
    mapping = CLASS_MAPPINGS[system]
    df_runs["prediction"] = df_runs["prediction"].astype(int)
    df_runs["prediction_name"] = df_runs["prediction"].map(mapping)
    return df_runs


def health_status(mean_confidence: float) -> str:
    """Stateless health flag: mean confidence (0-1) of the current request vs 75% threshold.

    The original implementation averages the last 50 predictions persisted in a local CSV;
    that mutable state is deliberately not reproduced here (manifest known_issues).
    """
    return "DEGRADADO" if mean_confidence * 100.0 < HEALTH_THRESHOLD_PCT else "ESTABLE"
