from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTEXT_RANGES: dict[str, tuple[float, float]] = {
    "temp_mean": (10.0, 29.5),
    "ambient_rh_mean": (36.0, 78.0),
    "grain_humidity_mean": (11.4, 13.4),
    "grain_drift_per_day": (-0.01, 0.03),
    "insect_mult": (0.85, 1.12),
    "mold_mult": (0.92, 1.12),
    "co2_offset": (-10.0, 18.0),
}


LABEL_NAMES = {0: "sano", 1: "insectos", 2: "moho_critico"}


def _ar1_noise(
    n: int,
    *,
    rng: np.random.Generator,
    phi: float,
    sigma: float,
) -> np.ndarray:
    e = np.zeros(n, dtype=float)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal(0.0, sigma)
    return e


def _apply_real_oscillations(
    base_signal: np.ndarray,
    *,
    rng: np.random.Generator,
    step_minutes: int,
    daily_amp: float = 35.0,
    noise_sd: float = 25.0,
    ar_phi: float = 0.82,
    drift_ppm_per_day: float = 0.0,
    spike_prob: float = 0.002,
    spike_scale: float = 160.0,
    amp_mod_strength: float = 0.22,
    second_harmonic_scale: float = 0.25,
    weekly_scale: float = 0.35,
    min_val: float = 350.0,
    max_val: float = 22000.0,
) -> np.ndarray:
    x = np.asarray(base_signal, dtype=float).copy()
    n = len(x)
    t = np.arange(n, dtype=float)
    steps_per_day = max(1, int(round(24 * 60 / step_minutes)))

    phase_d = rng.uniform(0.0, 2.0 * np.pi)
    phase_h2 = rng.uniform(0.0, 2.0 * np.pi)
    phase_w = rng.uniform(0.0, 2.0 * np.pi)
    phase_m = rng.uniform(0.0, 2.0 * np.pi)

    amp_mod = 1.0 + amp_mod_strength * np.sin(2.0 * np.pi * t / (steps_per_day * 5.0) + phase_m)
    daily = daily_amp * amp_mod * np.sin(2 * np.pi * t / steps_per_day + phase_d)
    harmonic = (daily_amp * second_harmonic_scale) * np.sin(4 * np.pi * t / steps_per_day + phase_h2)
    weekly = (daily_amp * weekly_scale) * np.sin(
        2 * np.pi * t / (steps_per_day * 7) + phase_w
    )

    e = np.zeros(n, dtype=float)
    for i in range(1, n):
        e[i] = ar_phi * e[i - 1] + rng.normal(0, noise_sd)

    hetero = np.clip(x, 350.0, None) / 1000.0
    e = e * (1.0 + 0.20 * hetero)

    drift_per_step = drift_ppm_per_day / steps_per_day
    drift = drift_per_step * t

    spikes = np.where(
        rng.random(n) < spike_prob,
        rng.normal(0.0, spike_scale, n),
        0.0,
    )

    y = x + daily + harmonic + weekly + e + drift + spikes
    return np.clip(y, min_val, max_val)


def _simulate_latent_path(
    class_id: int,
    *,
    n: int,
    rng: np.random.Generator,
    scenario: dict[str, Any],
) -> tuple[np.ndarray, int]:
    lag_low, lag_high = scenario["lag_hours"]
    lag = int(rng.integers(lag_low, lag_high + 1))
    lag = max(0, min(lag, n - 2))

    z = np.zeros(n, dtype=float)
    z0 = float(rng.uniform(0.04, 0.11))
    if lag > 0:
        z[:lag] = z0 + rng.normal(0.0, 0.004, lag)

    m = max(1, n - lag)
    u = np.linspace(0.0, 1.0, m)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))

    if class_id == 0:
        end = float(rng.uniform(0.10, 0.22))
        core = z0 + (end - z0) * (u**1.08)
        core += 0.012 * np.sin(2.0 * np.pi * 2.0 * u + phase)
    elif class_id == 1:
        end = float(rng.uniform(0.40, 0.76) * scenario["insect_mult"])
        end = min(end, 0.80)
        mix = float(rng.uniform(0.0, 1.0))
        profile_a = 0.78 * u + 0.22 * (u**1.55)
        profile_b = 0.60 * u + 0.40 * (u**2.00)
        core = z0 + (end - z0) * ((1.0 - mix) * profile_a + mix * profile_b)
        core += 0.020 * np.sin(2.0 * np.pi * 2.0 * u + phase)
    elif class_id == 2:
        end = float(rng.uniform(0.84, 0.97) * min(1.08, scenario["mold_mult"]))
        end = min(end, 0.995)
        k = float(rng.uniform(6.0, 9.5))
        x0 = float(rng.uniform(0.53, 0.74))
        logistic = 1.0 / (1.0 + np.exp(-k * (u - x0)))
        logistic = (logistic - logistic[0]) / max(logistic[-1] - logistic[0], 1e-9)
        core = z0 + (end - z0) * logistic
        core += 0.010 * np.sin(2.0 * np.pi * 1.6 * u + phase)
    else:
        raise ValueError(f"class_id invalido: {class_id}")

    if class_id == 1:
        pulse_count = int(rng.integers(1, 4))
        uu = np.arange(m, dtype=float)
        for _ in range(pulse_count):
            c = float(rng.uniform(0.30, 0.90) * m)
            w = float(rng.uniform(8.0, 24.0))
            a = float(rng.uniform(0.006, 0.025))
            core += a * np.exp(-0.5 * ((uu - c) / w) ** 2)

    z[lag:] = core
    z += _ar1_noise(n, rng=rng, phi=0.90, sigma={0: 0.0040, 1: 0.0054, 2: 0.0076}[class_id])
    z = np.clip(z, 0.0, 1.0)
    z = (
        pd.Series(z)
        .rolling(window=3, min_periods=1)
        .mean()
        .to_numpy()
    )
    return z, lag


def _sample_storage_context(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "temp_mean": float(rng.uniform(*CONTEXT_RANGES["temp_mean"])),
        "ambient_rh_mean": float(rng.uniform(*CONTEXT_RANGES["ambient_rh_mean"])),
        "grain_humidity_mean": float(rng.uniform(*CONTEXT_RANGES["grain_humidity_mean"])),
        "grain_drift_per_day": float(rng.uniform(*CONTEXT_RANGES["grain_drift_per_day"])),
        "insect_mult": float(rng.uniform(*CONTEXT_RANGES["insect_mult"])),
        "mold_mult": float(rng.uniform(*CONTEXT_RANGES["mold_mult"])),
        "co2_offset": float(rng.uniform(*CONTEXT_RANGES["co2_offset"])),
        "lag_hours": (
            int(rng.integers(10, 19)),
            int(rng.integers(72, 97)),
        ),
    }


def _simulate_sample_latent(
    *,
    sample_id: str,
    class_id: int,
    context: dict[str, Any],
    time_idx: pd.DatetimeIndex,
    step_minutes: int,
    osc_amp_mod_strength: float,
    osc_second_harmonic: float,
    osc_weekly_scale: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(time_idx)
    t = np.arange(n, dtype=float)
    steps_per_day = max(1, int(round(24 * 60 / step_minutes)))
    dt_hours = step_minutes / 60.0

    z, lag_hours = _simulate_latent_path(class_id, n=n, rng=rng, scenario=context)
    dz = np.diff(z, prepend=z[0])
    pos_dz = np.clip(dz, 0.0, None)

    mass_factor = float(rng.uniform(0.72, 1.45))
    ventilation_factor = float(rng.uniform(0.82, 1.22))
    if class_id == 1:
        scenario_mult = float(context["insect_mult"])
    elif class_id == 2:
        scenario_mult = float(context["mold_mult"])
    else:
        scenario_mult = 1.0

    rate_base_map = {0: 0.020, 1: 0.074, 2: 0.155}
    rate_z_gain_map = {0: 1.00, 1: 2.35, 2: 5.00}
    rate_acc_gain_map = {0: 7.0, 1: 13.0, 2: 34.0}

    co2_rate = (
        rate_base_map[class_id]
        + rate_z_gain_map[class_id] * z
        + rate_acc_gain_map[class_id] * pos_dz
    )
    co2_rate = co2_rate * mass_factor * scenario_mult / ventilation_factor

    co2_start = float(408.0 + context["co2_offset"] + rng.normal(0.0, 11.0))
    co2 = co2_start + np.cumsum(co2_rate * dt_hours)

    co2 = _apply_real_oscillations(
        co2,
        rng=rng,
        step_minutes=step_minutes,
        daily_amp={0: 24.0, 1: 31.0, 2: 28.0}[class_id],
        noise_sd={0: 16.0, 1: 23.0, 2: 22.0}[class_id],
        ar_phi={0: 0.80, 1: 0.87, 2: 0.90}[class_id],
        drift_ppm_per_day={0: 0.6, 1: 2.0, 2: 2.8}[class_id],
        spike_prob={0: 0.0010, 1: 0.0015, 2: 0.0018}[class_id],
        spike_scale={0: 55.0, 1: 90.0, 2: 95.0}[class_id],
        amp_mod_strength=osc_amp_mod_strength,
        second_harmonic_scale=osc_second_harmonic,
        weekly_scale=osc_weekly_scale,
        min_val=350.0,
        max_val=9000.0,
    )


    temp_daily = 2.1 * np.sin(2 * np.pi * t / steps_per_day + rng.uniform(0.0, 2.0 * np.pi))
    temp = (
        context["temp_mean"]
        + temp_daily
        + 0.45 * _ar1_noise(n, rng=rng, phi=0.84, sigma=0.25)
    )
    temp += {0: 0.4, 1: 2.4, 2: 6.2}[class_id] * z
    temp += {0: 0.0, 1: 0.45, 2: 1.2}[class_id] * pos_dz * steps_per_day

    ambient_rh_daily = 5.0 * np.sin(2 * np.pi * t / steps_per_day + rng.uniform(0.0, 2.0 * np.pi))
    ambient_rh = (
        context["ambient_rh_mean"]
        + ambient_rh_daily
        + 2.2 * _ar1_noise(n, rng=rng, phi=0.78, sigma=0.55)
    )
    ambient_rh += {0: 0.4, 1: 1.0, 2: 2.1}[class_id] * z

    grain_drift_per_step = context["grain_drift_per_day"] / steps_per_day
    grain_humidity = (
        context["grain_humidity_mean"]
        + grain_drift_per_step * t
        + 0.18 * _ar1_noise(n, rng=rng, phi=0.82, sigma=0.17)
    )
    grain_humidity += {0: 0.3, 1: 1.35, 2: 3.4}[class_id] * z
    grain_humidity += {0: 0.0, 1: 0.28, 2: 0.9}[class_id] * pos_dz * steps_per_day

    temp = np.clip(temp, 0.0, 55.0)
    ambient_rh = np.clip(ambient_rh, 15.0, 100.0)
    grain_humidity = np.clip(grain_humidity, 8.0, 24.0)

    sample_df = pd.DataFrame(
        {
            "sample_id": sample_id,
            "timestamp": time_idx,
            "step_idx": np.arange(n, dtype=np.int32),
            "co2_ppm": co2,
            "temp_c": temp,
            "ambient_rh_pct": ambient_rh,
            "humidity_grain_pct": grain_humidity,
            "target": class_id,
            "lag_hours": lag_hours,
        }
    )


    return sample_df


def _apply_temporal_labels_v2(
    dataset: pd.DataFrame,
    temporal_cfg: dict[str, Any],
) -> pd.DataFrame:
    out = dataset.copy().sort_values(["sample_id", "timestamp"]).reset_index(drop=True)
    out["target_global"] = out["target"].astype(int)
    out["target"] = out["target"].astype(int)
    out["phase_name"] = out["target"].map(LABEL_NAMES)
    out["healthy_until_step"] = 0
    out["transition_to_insect_end_step"] = np.nan
    out["transition_to_moho_end_step"] = np.nan

    healthy_min_hours = int(temporal_cfg.get("healthy_min_hours", 24))
    healthy_margin_hours = int(temporal_cfg.get("healthy_margin_hours", 18))
    insect_transition_min = int(temporal_cfg.get("insect_transition_min", 24))
    insect_transition_max = int(temporal_cfg.get("insect_transition_max", 84))
    moho_transition_min = int(temporal_cfg.get("moho_transition_min", 24))
    moho_transition_max = int(temporal_cfg.get("moho_transition_max", 72))

    for _, idx in out.groupby("sample_id", sort=False).groups.items():
        rows = out.loc[idx]
        global_target = int(rows["target_global"].iloc[0])
        steps = rows["step_idx"].to_numpy(dtype=int)
        lag_hours = int(rows["lag_hours"].iloc[0]) if "lag_hours" in rows.columns else 0
        co2 = rows["co2_ppm"].to_numpy(dtype=float)

        co2_start = float(np.nanpercentile(co2[: max(12, min(len(co2), 48))], 50))
        co2_end = float(np.nanpercentile(co2[max(0, len(co2) - 48) :], 75))
        co2_gain = max(0.0, co2_end - co2_start)
        normalized_gain = float(np.clip(co2_gain / 1200.0, 0.0, 1.0))

        healthy_until = int(max(healthy_min_hours, lag_hours - healthy_margin_hours))
        healthy_until = int(np.clip(healthy_until, healthy_min_hours, min(len(rows) - 2, 120)))

        temporal = np.zeros(len(rows), dtype=int)
        transition_to_insect_end = np.nan
        transition_to_moho_end = np.nan

        if global_target == 1:
            insect_transition = int(
                round(
                    insect_transition_min
                    + (1.0 - normalized_gain) * (insect_transition_max - insect_transition_min)
                )
            )
            transition_to_insect_end = int(
                np.clip(healthy_until + insect_transition, healthy_until + 1, len(rows) - 1)
            )
            ramp = np.linspace(0.0, 1.0, max(1, transition_to_insect_end - healthy_until), endpoint=False)
            ramp_idx = (steps - healthy_until).clip(min=0, max=max(0, len(ramp) - 1))
            temporal = np.where(
                steps < healthy_until,
                0,
                np.where(
                    steps >= transition_to_insect_end,
                    1,
                    np.where(ramp[ramp_idx] >= 0.5, 1, 0),
                ),
            )
        elif global_target == 2:
            insect_transition = int(
                round(
                    insect_transition_min
                    + (1.0 - normalized_gain) * (insect_transition_max - insect_transition_min)
                )
            )
            moho_transition = int(
                round(
                    moho_transition_min
                    + (1.0 - normalized_gain) * (moho_transition_max - moho_transition_min)
                )
            )
            transition_to_insect_end = int(
                np.clip(healthy_until + insect_transition, healthy_until + 1, len(rows) - 2)
            )
            transition_to_moho_end = int(
                np.clip(transition_to_insect_end + moho_transition, transition_to_insect_end + 1, len(rows) - 1)
            )
            temporal = np.select(
                [
                    steps < healthy_until,
                    steps < transition_to_insect_end,
                    steps < transition_to_moho_end,
                ],
                [0, 1, 1],
                default=2,
            )

        out.loc[idx, "target"] = temporal
        out.loc[idx, "phase_name"] = [LABEL_NAMES[int(v)] for v in temporal]
        out.loc[idx, "healthy_until_step"] = healthy_until
        out.loc[idx, "transition_to_insect_end_step"] = transition_to_insect_end
        out.loc[idx, "transition_to_moho_end_step"] = transition_to_moho_end

    return out


def generate_synthetic_dataset(synth_cfg: dict[str, Any]) -> pd.DataFrame:
    seed = int(synth_cfg.get("seed", 42))
    samples_per_class = int(synth_cfg.get("samples_per_class", 100))
    days = int(synth_cfg.get("days", 20))
    step_minutes = int(synth_cfg.get("step_minutes", 60))
    start = str(synth_cfg.get("start", "2026-01-01"))
    osc_amp_mod_strength = float(synth_cfg.get("osc_amp_mod_strength", 0.22))
    osc_second_harmonic = float(synth_cfg.get("osc_second_harmonic", 0.25))
    osc_weekly_scale = float(synth_cfg.get("osc_weekly_scale", 0.35))

    rng = np.random.default_rng(seed)
    periods = int(days * 24 * 60 / step_minutes)
    time_idx = pd.date_range(start=start, periods=periods, freq=f"{step_minutes}min")

    all_data: list[pd.DataFrame] = []

    for class_id in range(3):
        for i in range(samples_per_class):
            local_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
            sample_rng = np.random.default_rng(local_seed)
            sample_id = f"S_{class_id}_{i:04d}"
            context = _sample_storage_context(sample_rng)

            sample_df = _simulate_sample_latent(
                sample_id=sample_id,
                class_id=class_id,
                context=context,
                time_idx=time_idx,
                step_minutes=step_minutes,
                osc_amp_mod_strength=osc_amp_mod_strength,
                osc_second_harmonic=osc_second_harmonic,
                osc_weekly_scale=osc_weekly_scale,
                rng=sample_rng,
            )
            all_data.append(sample_df)

    out = pd.concat(all_data, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    temporal_cfg = dict(synth_cfg.get("temporal_labels_v2", {}) or {})
    if bool(temporal_cfg.get("enabled", False)):
        out = _apply_temporal_labels_v2(out, temporal_cfg)
    return out.sort_values(["sample_id", "timestamp"]).reset_index(drop=True)


def _build_summary(dataset: pd.DataFrame, step_minutes: int, days: int) -> pd.DataFrame:
    split_target_col = "target_global" if "target_global" in dataset.columns else "target"
    sample_meta = dataset.groupby("sample_id", sort=False)[[split_target_col]].first()
    rows: list[dict[str, Any]] = [
        {"section": "global", "metric": "rows", "value": int(len(dataset))},
        {"section": "global", "metric": "samples", "value": int(dataset["sample_id"].nunique())},
        {"section": "global", "metric": "step_minutes", "value": int(step_minutes)},
        {"section": "global", "metric": "days", "value": int(days)},
    ]

    for cls, cnt in sample_meta[split_target_col].value_counts().sort_index().items():
        rows.append({"section": "class_count", "metric": f"class_{int(cls)}", "value": int(cnt)})

    if "target" in dataset.columns:
        for cls, cnt in dataset["target"].value_counts().sort_index().items():
            rows.append({"section": "temporal_target_rows", "metric": f"class_{int(cls)}", "value": int(cnt)})


    return pd.DataFrame(rows)


def generate_and_export_synthetic_dataset(cfg: dict[str, Any]) -> dict[str, str]:
    synth_cfg = dict(cfg.get("synthetic_dataset", {}))
    dataset = generate_synthetic_dataset(synth_cfg)

    step_minutes = int(synth_cfg.get("step_minutes", 60))
    days = int(synth_cfg.get("days", 20))

    config_hash = hashlib.sha1(
        json.dumps({"synthetic_dataset": synth_cfg}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    dataset_version = f"synth_v_{config_hash}"
    dataset["dataset_version"] = dataset_version
    dataset["config_hash"] = config_hash

    raw_path = Path(cfg["paths"]["raw_dataset"])

    processed_dir = Path(cfg["paths"]["processed_dataset"]).parent
    summary_default = processed_dir / "dataset_infestacion_cereales_sintetico_summary.csv"
    metadata_default = processed_dir / "dataset_infestacion_cereales_sintetico_metadata.json"
    summary_path = Path(cfg["paths"].get("synthetic_summary", summary_default))
    metadata_path = Path(cfg["paths"].get("synthetic_metadata", metadata_default))

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(raw_path, index=False)
    summary_df = _build_summary(dataset, step_minutes=step_minutes, days=days)
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "dataset_version": dataset_version,
        "config_hash": config_hash,
        "paths": {
            "raw_dataset": str(raw_path),
            "summary_dataset": str(summary_path),
            "metadata_dataset": str(metadata_path),
        },
        "rows": int(len(dataset)),
        "samples": int(dataset["sample_id"].nunique()),
        "synthetic_dataset_config": synth_cfg,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "raw_dataset": str(raw_path),
        "summary_dataset": str(summary_path),
        "metadata_dataset": str(metadata_path),
        "rows": str(len(dataset)),
        "samples": str(dataset["sample_id"].nunique()),
        "dataset_version": dataset_version,
        "config_hash": config_hash,
    }
