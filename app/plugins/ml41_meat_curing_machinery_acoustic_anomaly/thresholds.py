"""Decision thresholds for the 48 (machine, machine_id, snr) vit_tiny combinations.

Extracted verbatim from inbox/a41/codigo/reports/table_auc_por_caso.csv (rows where
experiment starts with "vit_tiny_"), which persists the audit_fnr threshold computed
during original training for each combination (see manifest.yaml golden_cases /
known_issues). These are NOT recomputed at inference time.
"""
from __future__ import annotations


def checkpoint_threshold(checkpoint: dict, machine: str, machine_id: str, snr: str) -> float | None:
    """Read model calibration; original delivered checkpoints lack the threshold key.

    An explicit None records training without labeled anomalies. Do not silently
    replace it with a threshold calibrated for different weights/statistics.
    """
    if "threshold" in checkpoint:
        value = checkpoint["threshold"]
        return float(value) if value is not None else None
    return THRESHOLDS[(machine, machine_id, snr)]


THRESHOLDS: dict[tuple[str, str, str], float] = {
    ("fan", "id_00", "-6_dB"): 4.419,
    ("fan", "id_00", "0_dB"): 6.6067,
    ("fan", "id_00", "6_dB"): 10.2847,
    ("fan", "id_02", "-6_dB"): 7.3025,
    ("fan", "id_02", "0_dB"): 10.2376,
    ("fan", "id_02", "6_dB"): 9.242,
    ("fan", "id_04", "-6_dB"): 5.8919,
    ("fan", "id_04", "0_dB"): 9.4075,
    ("fan", "id_04", "6_dB"): 12.9963,
    ("fan", "id_06", "-6_dB"): 11.7131,
    ("fan", "id_06", "0_dB"): 11.3415,
    ("fan", "id_06", "6_dB"): 13.8746,
    ("pump", "id_00", "-6_dB"): 9.6755,
    ("pump", "id_00", "0_dB"): 9.9809,
    ("pump", "id_00", "6_dB"): 13.001,
    ("pump", "id_02", "-6_dB"): 8.2043,
    ("pump", "id_02", "0_dB"): 9.0275,
    ("pump", "id_02", "6_dB"): 10.9228,
    ("pump", "id_04", "-6_dB"): 12.6533,
    ("pump", "id_04", "0_dB"): 14.6664,
    ("pump", "id_04", "6_dB"): 14.8859,
    ("pump", "id_06", "-6_dB"): 7.5552,
    ("pump", "id_06", "0_dB"): 11.7429,
    ("pump", "id_06", "6_dB"): 12.9401,
    ("slider", "id_00", "-6_dB"): 11.4625,
    ("slider", "id_00", "0_dB"): 12.9442,
    ("slider", "id_00", "6_dB"): 14.8671,
    ("slider", "id_02", "-6_dB"): 7.7734,
    ("slider", "id_02", "0_dB"): 9.8018,
    ("slider", "id_02", "6_dB"): 12.4161,
    ("slider", "id_04", "-6_dB"): 8.6493,
    ("slider", "id_04", "0_dB"): 9.4802,
    ("slider", "id_04", "6_dB"): 10.5612,
    ("slider", "id_06", "-6_dB"): 7.4486,
    ("slider", "id_06", "0_dB"): 9.3159,
    ("slider", "id_06", "6_dB"): 10.0663,
    ("valve", "id_00", "-6_dB"): 2.7621,
    ("valve", "id_00", "0_dB"): 2.8615,
    ("valve", "id_00", "6_dB"): 3.8462,
    ("valve", "id_02", "-6_dB"): 2.585,
    ("valve", "id_02", "0_dB"): 2.9286,
    ("valve", "id_02", "6_dB"): 3.3346,
    ("valve", "id_04", "-6_dB"): 3.0866,
    ("valve", "id_04", "0_dB"): 3.799,
    ("valve", "id_04", "6_dB"): 3.6437,
    ("valve", "id_06", "-6_dB"): 2.8994,
    ("valve", "id_06", "0_dB"): 2.9641,
    ("valve", "id_06", "6_dB"): 3.7268,
}
