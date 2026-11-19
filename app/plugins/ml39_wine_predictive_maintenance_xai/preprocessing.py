"""Input handling for ml39 — independent port of ml37's cycle preprocessing (same contract:
14 physical sensors, resample to 600 steps, pad to the 40-slot schema_v1 tensor). Plugins stay
self-contained in this repo, so this is a copy, not a cross-plugin import of ml37's module.
"""
from __future__ import annotations

import numpy as np

from app.domain.services.exceptions import DataContractError
from app.plugins.ml39_wine_predictive_maintenance_xai.constants import (
    F_SCHEMA,
    PHYSICAL_SENSORS,
    TIMESTEPS,
)


def resample_cycle_to_fixed_length(
    t_values: np.ndarray, sensor_series: dict[str, np.ndarray], *, timesteps: int = TIMESTEPS,
) -> np.ndarray:
    """One cycle's raw (t, sensor) readings -> a (timesteps, 14) array, linearly interpolated
    onto `timesteps` evenly-spaced points spanning [t_min, t_max] — same contract as ml37."""
    t_values = np.asarray(t_values, dtype=np.float64)
    if not np.isfinite(t_values).all():
        raise DataContractError("La columna temporal contiene valores no numéricos, NaN o Inf")
    if len(np.unique(t_values)) != len(t_values):
        raise DataContractError("La columna temporal contiene valores de tiempo duplicados")

    out = np.zeros((timesteps, len(PHYSICAL_SENSORS)), dtype=np.float32)
    if len(t_values) == 1:
        for position, name in enumerate(PHYSICAL_SENSORS):
            out[:, position] = float(sensor_series[name][0])
        return out

    t_new = np.linspace(float(t_values.min()), float(t_values.max()), timesteps)
    for position, name in enumerate(PHYSICAL_SENSORS):
        values = np.asarray(sensor_series[name], dtype=np.float64)
        if not np.isfinite(values).all():
            raise DataContractError(f"El sensor '{name}' contiene valores no numéricos, NaN o Inf")
        out[:, position] = np.interp(t_new, t_values, values).astype(np.float32)
    return out


def build_schema_v1_tensor(resampled_14: np.ndarray) -> np.ndarray:
    """(T, 14) physical sensors -> (T, 40) schema_v1 tensor: slots 0-13 the physical sensors,
    slots 14-39 (virtual + reserved) always 0.0 — the no_virtual production configuration."""
    timesteps = resampled_14.shape[0]
    out = np.zeros((timesteps, F_SCHEMA), dtype=np.float32)
    out[:, : len(PHYSICAL_SENSORS)] = resampled_14
    return out


def apply_scaler(x: np.ndarray, scaler: dict) -> np.ndarray:
    """StandardScaler via plain mean/scale arrays, scale==0 guarded to 1.0."""
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    scale = np.where(scale == 0.0, 1.0, scale)
    shape = x.shape
    x2d = np.asarray(x, dtype=np.float32).reshape(-1, shape[-1])
    return ((x2d - mean) / scale).reshape(shape).astype("float32")


def parse_readings(readings: list[dict[str, float]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """A list of ``{t, PS1, ..., VS1}`` dicts -> (t_values, {sensor: values}), validating the 14
    physical sensors are all present in every reading."""
    if not readings:
        raise DataContractError("Se requiere al menos una lectura ('readings')")
    expected = {"t", *PHYSICAL_SENSORS}
    for i, reading in enumerate(readings):
        missing = expected - set(reading.keys())
        if missing:
            raise DataContractError(f"Lectura {i}: faltan campos obligatorios {sorted(missing)}")
    t_values = np.array([r["t"] for r in readings], dtype=np.float64)
    sensor_series = {
        name: np.array([r[name] for r in readings], dtype=np.float64) for name in PHYSICAL_SENSORS
    }
    return t_values, sensor_series


def cycle_from_readings(readings: list[dict[str, float]]) -> np.ndarray:
    """readings -> schema_v1 (600, 40) tensor — shared by predict_inline and predict_batch."""
    t_values, sensor_series = parse_readings(readings)
    resampled = resample_cycle_to_fixed_length(t_values, sensor_series)
    return build_schema_v1_tensor(resampled)
