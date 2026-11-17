"""Feature extraction for ml38 — a faithful port of the AI team's signal processing.

Source: inbox/a38/codigo/src/data_processing/preprocess.py (time, spectral and physical vibration
features, get_rpm), src/data_processing/build_dataset.py (concatenation order) and
scripts/predict_raw.py (read_signal). The formulas are copied as they are, including the
nominal rpm of get_rpm (see manifest known_issues KI-07), because they are what the served model
was trained with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

from app.plugins.ml38_wine_machinery_fault_detection.constants import (
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    BAND_MID_HZ,
    EPS,
    FEATURE_NAMES,
    FS_HZ,
    OPERATING_CONFIGURATIONS,
    PEAK_TOL_HZ,
    RPM_BASE_MOTOR_2,
    RPM_BASE_OTHER,
    TRAIN_ELE_WINDOW_SAMPLES,
    TRAIN_VIB_WINDOW_SAMPLES,
)
from app.plugins.ml38_wine_machinery_fault_detection.exceptions import (
    FeatureContractError,
    UnsupportedOperatingConfigurationError,
)


def check_operating_configuration(motor: str, speed: int) -> None:
    """Reject a motor/speed pair that is not a documented NLN-EMP configuration."""
    allowed = OPERATING_CONFIGURATIONS.get(motor)
    if allowed is None or int(speed) not in allowed:
        options = "; ".join(
            f"{m}: {', '.join(str(s) for s in speeds)}"
            for m, speeds in OPERATING_CONFIGURATIONS.items()
        )
        raise UnsupportedOperatingConfigurationError(
            f"Configuración motor/velocidad no documentada: motor={motor!r}, speed={speed}. "
            f"Solo se admiten las configuraciones del dataset NLN-EMP con las que se entrenó el "
            f"modelo ({options}); motor y velocidad fijan la frecuencia de giro de las "
            f"características 1X/2X y no deben elegirse de forma arbitraria (memoria §8.2)."
        )


def get_rpm(motor: str, speed: int | str) -> float:
    """Reference shaft speed (rpm) — preprocess.get_rpm."""
    base = RPM_BASE_MOTOR_2 if motor == "Motor-2" else RPM_BASE_OTHER
    return base * (int(speed) / 100.0)


def extract_time_features(signal: np.ndarray) -> np.ndarray:
    """mean, std, max, min, rms, kurtosis, skewness — preprocess.extract_time_features."""
    return np.array([
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.sqrt(np.mean(signal ** 2)),
        kurtosis(signal, bias=False),
        skew(signal, bias=False),
    ])


def extract_spectral_features(signal: np.ndarray) -> np.ndarray:
    """Spectral energy and centroid (in FFT bins) — preprocess.extract_spectral_features."""
    spectrum = np.abs(rfft(signal))
    spectral_energy = np.sum(spectrum ** 2)
    spectral_centroid = np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + EPS)
    return np.array([spectral_energy, spectral_centroid])


def extract_physical_vibration_features(signal: np.ndarray, rpm: float) -> np.ndarray:
    """Band energies and 1X/2X peaks — preprocess.extract_physical_vibration_features."""
    n = len(signal)
    spectrum = np.abs(rfft(signal))
    freqs = rfftfreq(n, 1 / FS_HZ)
    total_energy = np.sum(spectrum ** 2) + EPS

    def band_energy(f_low: float, f_high: float) -> float:
        mask = (freqs >= f_low) & (freqs < f_high)
        return np.sum(spectrum[mask] ** 2) / total_energy

    def peak_near(f_target: float, tol: float = PEAK_TOL_HZ) -> float:
        mask = (freqs >= f_target - tol) & (freqs <= f_target + tol)
        return np.max(spectrum[mask]) if np.any(mask) else 0

    f_rot = rpm / 60.0
    amp_1x = peak_near(f_rot)
    amp_2x = peak_near(2 * f_rot)
    return np.array([
        band_energy(*BAND_LOW_HZ),
        band_energy(*BAND_MID_HZ),
        band_energy(*BAND_HIGH_HZ),
        amp_1x,
        amp_2x,
        amp_2x / (amp_1x + EPS),
    ])


def build_feature_vector(
    vib_signal: np.ndarray, ele_signal: np.ndarray, motor: str, speed: int | str
) -> np.ndarray:
    """The 24 model features of one vibration/current pair, in FEATURE_NAMES order.

    Same concatenation as build_dataset.build_processed_dataset and predictor.run_inference:
    [vib time + vib spectral, vib physical, ele time + ele spectral].
    """
    vib_basic = np.concatenate([extract_time_features(vib_signal),
                                extract_spectral_features(vib_signal)])
    vib_phys = extract_physical_vibration_features(vib_signal, get_rpm(motor, speed))
    ele_feats = np.concatenate([extract_time_features(ele_signal),
                                extract_spectral_features(ele_signal)])
    features = np.concatenate([vib_basic, vib_phys, ele_feats])
    if features.shape != (len(FEATURE_NAMES),):  # defensive: the port must keep the layout
        raise RuntimeError(f"Vector de características con forma {features.shape}")
    return features


def read_signal(path: str, window_index: int) -> np.ndarray:
    """Signal column of a raw CSV — scripts/predict_raw.read_signal.

    Column 0 is ``time``; ``window_index`` = 0 selects the first signal column after it. A CSV
    with a single column is used as is.
    """
    try:
        df = pd.read_csv(path, dtype=float)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        raise FeatureContractError(
            f"No se puede leer la señal raw {path}: se espera un CSV numérico con una columna "
            f"time y una o más columnas de señal ({exc})"
        ) from exc
    if window_index < 0:
        raise FeatureContractError(f"window_index debe ser >= 0 (recibido {window_index}).")
    column_index = window_index + 1 if df.shape[1] > 1 else 0
    if column_index >= df.shape[1]:
        raise FeatureContractError(
            f"window_index={window_index} no disponible en {path}; "
            f"el fichero contiene {df.shape[1]} columnas."
        )
    signal = df.iloc[:, column_index].to_numpy()
    if signal.size == 0 or not np.all(np.isfinite(signal)):
        raise FeatureContractError(
            f"La columna de señal seleccionada en {path} está vacía o tiene valores no numéricos."
        )
    return signal


def signal_length_warnings(n_vib: int, n_ele: int) -> list[str]:
    """Warnings when a raw signal does not have the training window length (KI-05)."""
    warnings = []
    for name, got, expected, secs in (
        ("vibración", n_vib, TRAIN_VIB_WINDOW_SAMPLES, 12),
        ("corriente", n_ele, TRAIN_ELE_WINDOW_SAMPLES, 15),
    ):
        if got != expected:
            warnings.append(
                f"La señal de {name} tiene {got} muestras y el modelo se entrenó con ventanas de "
                f"{expected} ({secs} s a {FS_HZ} Hz). La energía espectral, el centroide y las "
                f"amplitudes 1X/2X dependen de la longitud: la predicción queda fuera del dominio "
                f"de entrenamiento y no debe usarse para decidir."
            )
    return warnings


def feature_frame_from_table(df: pd.DataFrame) -> pd.DataFrame:
    """The 24 feature columns of a tabular input, in contract order (labels are ignored)."""
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise FeatureContractError(
            f"Faltan columnas de características: {missing}. Se esperan las 24 columnas de "
            f"data/processed/feature_dataset.parquet."
        )
    return df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")


def read_table(path: str) -> pd.DataFrame:
    """Read a CSV or parquet file with the feature table (the original pipeline uses parquet)."""
    try:
        if path.lower().endswith((".parquet", ".pq")):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except (ValueError, OSError, pd.errors.ParserError, ImportError) as exc:
        raise FeatureContractError(f"No se puede leer el fichero {path}: {exc}") from exc
