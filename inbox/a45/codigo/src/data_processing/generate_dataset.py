"""Generación de dataset sintético para secuencias temporales de un secador de granos de flujo mixto."""

import os
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from src.utils.common import ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MixedFlowGrainDryerGenerator:
    """Generador de dataset sintético para un secador de granos de flujo mixto, con múltiples tipos de fallas y perfiles de sensor realistas."""

    def __init__(self, seed: Optional[int] = None, config: Optional[dict] = None):
        if config is None:
            raise ValueError("config is required for MixedFlowGrainDryerGenerator")

        self.config = deepcopy(config)
        self.config.setdefault("project", {})

        if seed is not None:
            self.config["project"]["seed"] = int(seed)

        self.config["project"].setdefault("seed", 42)
        self.project_seed = int(self.config["project"]["seed"])
        self.rng = np.random.RandomState(self.project_seed)

        self.data_cfg = self.config["data_generation"]
        self.paths_cfg = self.config.get("paths", {}) or {}

        required_data_keys = [
            "sampling_interval_sec",
            "cycle_duration_hours",
            "dryer",
            "sensors",
            "fault_types",
            "split_ratios",
            "fault_start",
            "fault_duration",
            "severity",
        ]
        missing_keys = [k for k in required_data_keys if k not in self.data_cfg]
        if missing_keys:
            raise KeyError(
                f"Missing required keys in config['data_generation']: {missing_keys}"
            )

        self.dryer = self.data_cfg["dryer"]
        self.sensors = self.data_cfg["sensors"]
        self.fault_types = self.data_cfg["fault_types"]
        self.fault_profiles = self.data_cfg.get("fault_profiles", {})

        self.sampling_interval = int(self.data_cfg["sampling_interval_sec"])
        self.n_steps = int(
            float(self.data_cfg["cycle_duration_hours"]) * 3600 / self.sampling_interval
        )

        self.sensor_ranges = {s["name"]: tuple(s["range"]) for s in self.sensors}
        self.sensor_noise_std = {
            s["name"]: float(s.get("noise_std", 0.0)) for s in self.sensors
        }
        self.fault_label_map = {f["name"]: int(f["label"]) for f in self.fault_types}

    def _clip_sensor(self, name: str, val: np.ndarray) -> np.ndarray:
        lo, hi = self.sensor_ranges.get(name, (-np.inf, np.inf))
        return np.clip(val, lo, hi)

    def _smooth_noise(self, std: float, n: int, alpha: float = 0.9) -> np.ndarray:
        if std <= 0:
            return np.zeros(n, dtype=float)

        burn_in = 100
        n_total = n + burn_in
        innovations = self.rng.normal(0.0, std * np.sqrt(1.0 - alpha**2), n_total)
        b = np.array([1.0])
        a = np.array([1.0, -alpha])
        noise = lfilter(b, a, innovations)
        return noise[burn_in:]

    def _apply_fault_to_process(self, fault_name: str, impact: float, state: dict) -> dict:
        if fault_name == "BURNER_DEGRADED":
            k = self.fault_profiles[fault_name]["gain_loss_factor"]
            state["eff_gain"] *= 1.0 - k * impact

        elif fault_name == "PLENUM_THERMAL_LEAK":
            k = self.fault_profiles[fault_name]["loss_increase"]
            state["eff_loss"] *= 1.0 + k * impact

        elif fault_name == "FILTER_CLOGGED":
            drop = self.fault_profiles[fault_name]["fan_speed_drop"]
            state["fan_speed"] = max(100.0, state["fan_speed"] - drop * impact)
            state["extra_resistance"] = 1.0 + (1.5 * impact)
            state["eff_loss"] *= 1.0 - (0.15 * impact)

        elif fault_name == "DISCHARGE_JAM":
            drop = self.fault_profiles[fault_name].get("discharge_drop_factor", 0.9)
            state["discharge_frequency"] *= 1.0 - drop * impact
            state["humidity_accumulation_factor"] = 1.0 + (1.2 * impact)

        return state

    def _apply_fault_to_sensors(self, fault_name: str, impact: float, obs: dict) -> dict:
        if fault_name == "HUMIDITY_SENSOR_DRIFT":
            obs["exhaust_air_humidity"] += (
                self.fault_profiles[fault_name]["exhaust_humidity_rise"] * impact
            )

        elif fault_name == "DISCHARGE_JAM":
            obs["exhaust_air_temp"] += self.fault_profiles[fault_name]["exhaust_temp_rise"] * impact
            obs["exhaust_air_humidity"] -= (
                self.fault_profiles[fault_name]["exhaust_humidity_drop"] * impact
            )

        elif fault_name == "PLENUM_THERMAL_LEAK":
            obs["plenum_temp"] -= self.fault_profiles[fault_name]["plenum_temp_drop"] * impact

        return obs

    def _generate_cycle(self, fault_name: str = "NORMAL", cycle_id: Optional[int] = None) -> pd.DataFrame:
        n = self.n_steps
        grain_type = self.rng.choice(list(self.dryer["grain_profiles"].keys()))
        profile = self.dryer["grain_profiles"][grain_type]
        setpoint = float(profile["setpoint"])

        time_hours = np.linspace(0, float(self.data_cfg["cycle_duration_hours"]), n)

        if grain_type == "CORN":
            base_temp = self.rng.uniform(-2.0, 6.0)
            temp_amplitude = self.rng.uniform(3.0, 7.0)
            base_humidity = self.rng.uniform(80.0, 92.0)
            humidity_amplitude = self.rng.uniform(4.0, 8.0)
        else:
            base_temp = self.rng.uniform(20.0, 28.0)
            temp_amplitude = self.rng.uniform(10.0, 15.0)
            base_humidity = self.rng.uniform(25.0, 38.0)
            humidity_amplitude = self.rng.uniform(12.0, 18.0)

        ambient_temp_arr = base_temp + temp_amplitude * np.sin(2 * np.pi * (time_hours - 4) / 24)
        ambient_humidity_arr = np.clip(
            base_humidity - humidity_amplitude * np.sin(2 * np.pi * (time_hours - 4) / 24),
            8.0,
            98.0,
        )

        grain_moisture_in = self.rng.uniform(*profile["moisture_in"])

        n_startup = int(self.dryer["startup_pct"] * n)
        n_shutdown = int(self.dryer["shutdown_pct"] * n)

        noises = {
            s["name"]: self._smooth_noise(self.sensor_noise_std[s["name"]], n)
            for s in self.sensors
        }

        plenum_temp_real = np.zeros(n, dtype=float)
        plenum_temp_real[0] = ambient_temp_arr[0]

        plenum_temp_measured = np.zeros(n, dtype=float)
        plenum_temp_measured[0] = ambient_temp_arr[0] + noises["plenum_temp"][0]

        burner_power = np.zeros(n, dtype=float)
        fan_speed = np.full(n, float(self.dryer["fan_speed_rpm_nominal"]), dtype=float)
        static_pressure = np.zeros(n, dtype=float)

        exhaust_air_temp_real = np.zeros(n, dtype=float)
        exhaust_air_temp_measured = np.zeros(n, dtype=float)
        exhaust_air_temp_measured[0] = ambient_temp_arr[0] + noises["exhaust_air_temp"][0]

        exhaust_air_humidity_real = np.zeros(n, dtype=float)
        exhaust_air_humidity_measured = np.zeros(n, dtype=float)
        exhaust_air_humidity_measured[0] = (
            ambient_humidity_arr[0] + 15.0 + noises["exhaust_air_humidity"][0]
        )

        discharge_frequency = np.zeros(n, dtype=float)
        hidden_grain_moisture = np.zeros(n, dtype=float)
        hidden_grain_moisture[0] = grain_moisture_in

        dt = float(self.sampling_interval)
        tau_real = float(self.dryer["thermal_inertia_tau"])
        kp = float(self.dryer["Kp"])

        f_exp = float(self.dryer["pressure_fan_speed_exponent"])
        f_div = float(self.dryer["pressure_fan_speed_divisor"])

        f_start, f_end, severity_max = -1, -1, 0.0
        if fault_name != "NORMAL":
            s_pct = self.rng.uniform(
                self.data_cfg["fault_start"]["min_pct"], self.data_cfg["fault_start"]["max_pct"]
            )
            d_pct = self.rng.uniform(
                self.data_cfg["fault_duration"]["min_pct"],
                self.data_cfg["fault_duration"]["max_pct"],
            )
            f_start = int(s_pct * n)
            f_end = min(int(f_start + (d_pct * n)), n)
            severity_max = self.rng.uniform(
                self.data_cfg["severity"]["min"], self.data_cfg["severity"]["max"]
            )

        smoothed_error = 0.0

        for i in range(1, n):
            current_impact = 0.0
            if fault_name != "NORMAL" and f_start <= i < f_end:
                current_impact = ((i - f_start) / float(f_end - f_start)) * severity_max

            base_discharge = max(5.0, 60.0 - (hidden_grain_moisture[i - 1] * profile["discharge_slope"]))
            if i < n_startup:
                base_discharge = 0.0
            elif i >= (n - n_shutdown):
                base_discharge *= np.linspace(1.0, 0.2, n_shutdown)[i - (n - n_shutdown)]

            state = {
                "eff_gain": float(self.dryer["C_gain"]),
                "eff_loss": float(self.dryer["C_loss"]),
                "fan_speed": float(self.dryer["fan_speed_rpm_nominal"]),
                "extra_resistance": 1.0,
                "discharge_frequency": base_discharge,
                "humidity_accumulation_factor": 1.0,
            }

            state = self._apply_fault_to_process(fault_name, current_impact, state)
            fan_speed[i] = state["fan_speed"]
            discharge_frequency[i] = state["discharge_frequency"]

            if i < (n - n_shutdown):
                error_raw = setpoint - plenum_temp_measured[i - 1]
                smoothed_error = 0.85 * smoothed_error + 0.15 * error_raw

                modulation = self.dryer["burner_power_modulation_pct"] * np.sin(2 * np.pi * i / 120)
                p_ideal = self.dryer["burner_power_pct_nominal"] + (smoothed_error * kp) + modulation
                burner_power[i] = np.clip(p_ideal, 0, 100)
            else:
                burner_power[i] = 0

            heat_gain = burner_power[i] * state["eff_gain"]
            heat_loss = (plenum_temp_real[i - 1] - ambient_temp_arr[i]) * state["eff_loss"]
            plenum_temp_real[i] = plenum_temp_real[i - 1] + (dt / tau_real) * (heat_gain - heat_loss)

            driving_force = max(0, hidden_grain_moisture[i - 1] - 8.0)
            temp_differential = max(0.1, plenum_temp_real[i] - ambient_temp_arr[i])
            psychro_factor = (temp_differential / 80.0) ** 1.8
            humidity_resistance = max(0.1, 1.0 - (ambient_humidity_arr[i] / 100.0))

            drying_effect = (
                driving_force
                * profile["drying_rate"]
                * psychro_factor
                * humidity_resistance
                * (fan_speed[i] / 1200.0)
            ) / state["humidity_accumulation_factor"]

            if i >= (n - n_shutdown):
                drying_effect *= 0.05

            hidden_grain_moisture[i] = max(8.0, hidden_grain_moisture[i - 1] - drying_effect)

            level_factor = (
                1.0 if i < (n - n_shutdown) else max(0, 1.0 - (i - (n - n_shutdown)) / n_shutdown)
            )
            pressure_seal = 1.0 if level_factor > 0.2 else (level_factor / 0.2)

            clog_factor = state.get("extra_resistance", 1.0)
            static_pressure[i] = (
                ((fan_speed[i] / f_div) ** f_exp) * clog_factor
                + (hidden_grain_moisture[i] * self.dryer["pressure_grain_moisture_coeff"])
            ) * profile["resistance"] * (level_factor**2) * pressure_seal

            cooling_attenuation = (
                1.0 if i < (n - n_shutdown) else max(0.1, 1.0 - (i - (n - n_shutdown)) / n_shutdown)
            )
            evaporation_humidity_contribution = drying_effect * 12000.0 * cooling_attenuation

            exhaust_air_humidity_real[i] = np.clip(
                ambient_humidity_arr[i] + 15.0 + evaporation_humidity_contribution,
                ambient_humidity_arr[i],
                98.5,
            )

            evaporative_cooling = (drying_effect * 180.0) * cooling_attenuation
            exhaust_air_temp_real[i] = (
                (plenum_temp_real[i] * self.dryer["exhaust_temp_plenum_coeff"]) - evaporative_cooling
            )
            exhaust_air_temp_real[i] = max(ambient_temp_arr[i] + 2.0, exhaust_air_temp_real[i])

            obs = {
                "plenum_temp": plenum_temp_real[i],
                "exhaust_air_temp": exhaust_air_temp_real[i],
                "exhaust_air_humidity": exhaust_air_humidity_real[i],
            }
            obs = self._apply_fault_to_sensors(fault_name, current_impact, obs)

            plenum_temp_measured[i] = obs["plenum_temp"] + noises["plenum_temp"][i]
            exhaust_air_temp_measured[i] = obs["exhaust_air_temp"] + noises["exhaust_air_temp"][i]
            exhaust_air_humidity_measured[i] = (
                obs["exhaust_air_humidity"] + noises["exhaust_air_humidity"][i]
            )

        phase = np.full(n, "drying", dtype=object)
        phase[:n_startup] = "heating"
        phase[n - n_shutdown :] = "cooling"

        if cycle_id is None:
            cycle_id = -1

        df = pd.DataFrame(
            {
                "plenum_temp": self._clip_sensor("plenum_temp", plenum_temp_measured),
                "exhaust_air_temp": self._clip_sensor("exhaust_air_temp", exhaust_air_temp_measured),
                "exhaust_air_humidity": self._clip_sensor(
                    "exhaust_air_humidity", exhaust_air_humidity_measured
                ),
                "static_pressure": self._clip_sensor(
                    "static_pressure", static_pressure + noises["static_pressure"]
                ),
                "burner_power": self._clip_sensor("burner_power", burner_power + noises["burner_power"]),
                "fan_speed": self._clip_sensor("fan_speed", fan_speed + noises["fan_speed"]),
                "discharge_frequency": self._clip_sensor(
                    "discharge_frequency", discharge_frequency + noises["discharge_frequency"]
                ),
                "grain_moisture_in": self._clip_sensor(
                    "grain_moisture_in", np.full(n, grain_moisture_in) + noises["grain_moisture_in"]
                ),
                "ambient_temp": self._clip_sensor("ambient_temp", ambient_temp_arr + noises["ambient_temp"]),
                "ambient_humidity": self._clip_sensor(
                    "ambient_humidity", ambient_humidity_arr + noises["ambient_humidity"]
                ),
                "setpoint_temp": np.full(n, setpoint),
                "phase": phase,
                "grain_type": grain_type,
                "fault_name": fault_name,
                "fault_label": self.fault_label_map.get(fault_name, -1),
                "cycle_id": cycle_id,
            }
        )

        base_time = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=int(cycle_id) * 12)
        df.insert(
            0,
            "timestamp",
            pd.date_range(base_time, periods=n, freq=f"{self.sampling_interval}s"),
        )
        return df      

    def generate_dataset(
        self,
        n_cycles: int,
        split_name: Optional[str] = None,
        normal_ratio: Optional[float] = None,
        fault_ratio: Optional[float] = None,
    ) -> pd.DataFrame:
        
        nr = float(normal_ratio) if normal_ratio is not None else None
        fr = float(fault_ratio) if fault_ratio is not None else None

        if split_name is not None:
            split_ratios = self.data_cfg.get("split_ratios", {})
            split_info = split_ratios.get(split_name, {})
            nr = float(split_info.get("normal_ratio", nr)) if split_info else nr
            fr = float(split_info.get("fault_ratio", fr)) if split_info else fr

        if nr is None or fr is None:
            raise ValueError("Por favor, proporcione normal_ratio o fault_ratio.")

        # Si solo se proporciona una de las proporciones, asumimos que la otra es el complemento a 1.0
        if nr is not None and fr is None:
            fr = 1.0 - nr
        if fr is not None and nr is None:
            nr = 1.0 - fr

        total = nr + fr
        if total != 1.0:
            raise ValueError("normal_ratio + fault_ratio debe ser igual a 1.0.")
        
        n_fault = int(n_cycles * fr)
        n_normal = n_cycles - n_fault

        plan = ["NORMAL"] * n_normal
        available_faults = [f["name"] for f in self.fault_types if f["name"] != "NORMAL"]
        plan += [self.rng.choice(available_faults) for _ in range(n_fault)]
        self.rng.shuffle(plan)

        all_dfs = []
        for i, f_name in enumerate(plan):
            all_dfs.append(self._generate_cycle(str(f_name), cycle_id=i))

        return pd.concat(all_dfs, ignore_index=True)

    def generate_full_dataset_from_splits(self) -> tuple[pd.DataFrame, dict]:
        """Genera los splits train/val/test y el dataset completo desde la configuración."""
        _raw = self.paths_cfg.get("raw_data", "data/raw/dryer_full_dataset.csv").rstrip("/\\")
        if os.path.isdir(_raw) or not os.path.splitext(_raw)[1]:
            raw_dir = ensure_dir(_raw)
            full_filename = "dryer_full_dataset.csv"
        else:
            raw_dir = ensure_dir(os.path.dirname(_raw))
            full_filename = os.path.basename(_raw)
        splits_dir = ensure_dir(self.paths_cfg.get("splits", "data/splits/"))

        full_path = os.path.join(raw_dir, full_filename)

        split_filenames = {
            "train": "train.csv",
            "val": "val.csv",
            "test": "test.csv",
        }

        splits = {
            "train": int(self.data_cfg.get("n_cycles_train", 0)),
            "val": int(self.data_cfg.get("n_cycles_val", 0)),
            "test": int(self.data_cfg.get("n_cycles_test", 0)),
        }

        dfs: list[pd.DataFrame] = []
        cycle_offset = 0

        split_ratio_cfg = self.data_cfg.get("split_ratios", {})
        available_faults = [
            f["name"]
            for f in self.fault_types
            if 1 <= int(f.get("label", -1)) <= 6
        ]

        for split_name, n_cycles in splits.items():
            if n_cycles <= 0:
                continue

            split_cfg = (
                split_ratio_cfg.get(split_name, {})
                if isinstance(split_ratio_cfg, dict)
                else {}
            )

            normal_ratio = float(
                split_cfg.get(
                    "normal_ratio", self.data_cfg.get("normal_ratio", 0.0)
                )
            )
            fault_ratio = float(
                split_cfg.get(
                    "fault_ratio", self.data_cfg.get("fault_ratio", 0.0)
                )
            )

            if min(normal_ratio, fault_ratio) < 0:
                raise ValueError(
                    f"Ratios inválidos en split '{split_name}': deben ser >= 0"
                )

            if len(available_faults) == 0 and fault_ratio > 0:
                fault_ratio = 0.0

            ratio_sum = normal_ratio + fault_ratio
            if ratio_sum <= 0:
                raise ValueError(
                    f"Split '{split_name}' sin probabilidad asignada (normal/fault)."
                )

            # Normalize locally — do NOT mutate self.data_cfg
            local_normal = normal_ratio / ratio_sum
            local_fault = fault_ratio / ratio_sum

            logger.info(
                "Ratios para split '%s' -> normal=%.3f, fallo=%.3f",
                split_name,
                local_normal,
                local_fault,
            )

            df_split = self.generate_dataset(
                n_cycles=n_cycles,
                normal_ratio=local_normal,
                fault_ratio=local_fault,
            )
            df_split = df_split.copy()
            df_split["cycle_id"] = df_split["cycle_id"] + cycle_offset
            df_split["timestamp"] = pd.to_datetime(df_split["timestamp"]) + pd.to_timedelta(
                cycle_offset * 12, unit="h"
            )

            split_path = os.path.join(splits_dir, split_filenames.get(split_name, f"{split_name}.csv"))
            df_split.to_csv(split_path, index=False)
            logger.info("Split '%s' guardado en %s", split_name, split_path)

            dfs.append(df_split)
            cycle_offset += n_cycles

        if not dfs:
            raise ValueError(
                "No hay ciclos configurados en n_cycles_train/val/test"
            )

        df_full = pd.concat(dfs, ignore_index=True)
        df_full.to_csv(full_path, index=False)

        summary = {
            "output_csv_path": full_path,
            "n_cycles_generados": int(df_full["cycle_id"].nunique()),
            "n_rows": int(len(df_full)),
            "rows_por_ciclo": int(self.n_steps),
            "fault_distribution": df_full["fault_name"].value_counts().to_dict(),
            "timestamp_min": str(df_full["timestamp"].min()),
            "timestamp_max": str(df_full["timestamp"].max()),
        }
        return df_full, summary


def run_dataset_generation(config: dict) -> pd.DataFrame:
    generator = MixedFlowGrainDryerGenerator(config=config)
    df_full, summary = generator.generate_full_dataset_from_splits()

    logger.info("Distribucion de fallas: %s", summary["fault_distribution"])

    return df_full

def run_xai_datasets_generation(config: dict) -> Dict[str, pd.DataFrame]:
    """Genera datasets específicos para la capa de explicabilidad XAI.

    Args:
        config: Diccionario de configuración completo del proyecto.

    Returns:
        Diccionario con DataFrames para cada dataset XAI generado.
    """
    xai_cfg = config.get("xai", {}).get("dataset_generation", {})
    output_dir = config.get("paths", {}).get("raw_data", "data/raw/")
    ensure_dir(output_dir)
    generated_datasets: Dict[str, pd.DataFrame] = {}

    for name, dataset_cfg in xai_cfg.items():
        seed = dataset_cfg.get("seed")
        n_cycles = dataset_cfg.get("n_cycles")
        fault_ratio = dataset_cfg.get("fault_ratio")

        if seed is None or n_cycles is None or fault_ratio is None:
            logger.warning(
                "Dataset '%s' missing seed, n_cycles, or fault_ratio. Skipping.",
                name,
            )
            continue

        generator = MixedFlowGrainDryerGenerator(config=config, seed=seed)
        dataset = generator.generate_dataset(
            n_cycles=n_cycles,
            normal_ratio=1.0 - fault_ratio,
            fault_ratio=fault_ratio,
        )

        output_path = f"{output_dir}/{name}.csv"
        dataset.to_csv(output_path, index=False)

        generated_datasets[name] = dataset

        logger.info(
            "Dataset '%s' generado con %d ciclos. Distribucion de fallas: %s",
            name,
            n_cycles,
            dataset["fault_name"].value_counts().to_dict(),
        )

    logger.info("Todos los datasets XAI generados y guardados en %s", output_dir)
    return generated_datasets
