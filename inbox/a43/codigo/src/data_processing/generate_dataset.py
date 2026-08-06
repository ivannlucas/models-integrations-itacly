"""Generación de dataset sintético de hornos industriales.

Simula ciclos operacionales con 13 variables de sensores IIoT,
incluyendo modos de operación normal y 6 tipos de falla.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.common import ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OvenDataGenerator:
    """Generador de datos sintéticos para hornos industriales de procesado
    de masas cerealísticas.

    Simula ciclos operacionales con 13 variables de sensores IIoT,
    incluyendo modos de operación normal y 6 tipos de falla.

    Args:
        seed: Semilla para la generación de números aleatorios.
        config: Diccionario de configuración completo del proyecto.
    """

    def __init__(self, config: Optional[dict] = None, seed: Optional[int] = None):
        if config is None:
            raise ValueError("Se requiere un diccionario de configuración para inicializar OvenDataGenerator.")
        self.config = config

        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(config["project"]["seed"])

        self.data_cfg = config["data_generation"]
        self.oven_cfg = self.data_cfg["oven"]
        self.sensors = self.data_cfg["sensors"]
        self.fault_types = self.data_cfg["fault_types"]

        self.sampling_interval = self.data_cfg["sampling_interval_sec"]
        self.cycle_duration = self.data_cfg["cycle_duration_hours"]
        self.n_steps = int(self.cycle_duration * 3600 / self.sampling_interval)

        self.sensor_names = [s["name"] for s in self.sensors]
        self.sensor_ranges = {
            s["name"]: tuple(s["range"]) for s in self.sensors if "range" in s
        }

        self.fault_start_min_pct, self.fault_start_max_pct = self._get_bounds(
            cfg_key="fault_start",
            default_min=0.10,
            default_max=0.35,
            hard_min=0.0,
            hard_max=0.95,
        )
        self.fault_duration_min_pct, self.fault_duration_max_pct = self._get_bounds(
            cfg_key="fault_duration",
            default_min=0.40,
            default_max=0.65,
            hard_min=0.01,
            hard_max=1.0,
        )
        self.severity_min, self.severity_max = self._get_bounds(
            cfg_key="severity",
            default_min=0.85,
            default_max=1.00,
            hard_min=0.0,
            hard_max=3.0,
            min_key="min",
            max_key="max",
        )

        self.paths = config["paths"]

    def _get_bounds(
        self,
        cfg_key: str,
        default_min: float,
        default_max: float,
        hard_min: float,
        hard_max: float,
        min_key: str = "min_pct",
        max_key: str = "max_pct",
    ) -> Tuple[float, float]:
        """Extrae y valida límites de configuración.

        Args:
            cfg_key: Clave en data_cfg.
            default_min: Valor mínimo por defecto.
            default_max: Valor máximo por defecto.
            hard_min: Límite inferior absoluto.
            hard_max: Límite superior absoluto.
            min_key: Clave para el valor mínimo.
            max_key: Clave para el valor máximo.

        Returns:
            Tupla (lo, hi) validada.
        """
        cfg = self.data_cfg.get(cfg_key, {})
        lo = float(cfg.get(min_key, default_min))
        hi = float(cfg.get(max_key, default_max))
        lo = float(np.clip(lo, hard_min, hard_max))
        hi = float(np.clip(hi, hard_min, hard_max))
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def _generate_ambient_conditions(self) -> Dict[str, float]:
        """Genera condiciones ambientales aleatorias para un ciclo."""
        temp_range = self.oven_cfg["temp_ambient_range"]
        hum_range = self.oven_cfg["humidity_range"]
        temp_ambient = self.rng.uniform(temp_range[0], temp_range[1])
        humidity = self.rng.uniform(hum_range[0], hum_range[1])
        return {"temp_ambient": temp_ambient, "humidity": humidity}

    def _generate_normal_cycle(self, ambient: Dict[str, float]) -> pd.DataFrame:
        """Genera un ciclo normal completo con las 13 variables de sensores.

        Args:
            ambient: Condiciones ambientales (temp_ambient, humidity).

        Returns:
            DataFrame con las 13 columnas de sensores.
        """
        n = self.n_steps
        setpoint = self.oven_cfg["temp_setpoint"]

        preheat_end = int(0.2 * n)
        cooking_end = int(0.8 * n)

        # Temperaturas de zona
        temps = {}
        for zone in range(1, 4):
            temp = np.zeros(n)
            zone_offset = (zone - 2) * 5

            for i in range(preheat_end):
                frac = i / preheat_end
                temp[i] = ambient["temp_ambient"] + frac * (
                    setpoint + zone_offset - ambient["temp_ambient"]
                )

            for i in range(preheat_end, cooking_end):
                temp[i] = setpoint + zone_offset + 15 * np.sin(
                    2 * np.pi * i / (n * 0.1)
                )

            for i in range(cooking_end, n):
                frac = (i - cooking_end) / (n - cooking_end)
                temp[i] = (setpoint + zone_offset) * np.exp(-2.5 * frac) + ambient[
                    "temp_ambient"
                ] * (1 - np.exp(-2.5 * frac))

            temp += self.rng.normal(0, 1.2, n)
            temps[f"temp_zona{zone}"] = np.clip(temp, 20, 250)

        # Temperatura salida de gases
        temp_gases = 0.55 * temps["temp_zona3"] + 0.15 * temps["temp_zona1"]
        temp_gases += self.rng.normal(0, 1.5, n)
        temp_gases = np.clip(temp_gases, 20, 300)

        # Presión cámara
        presion_camara = np.ones(n) * 3.0
        presion_camara[:preheat_end] = np.linspace(0.5, 3.0, preheat_end)
        presion_camara[cooking_end:] = np.linspace(3.0, 0.5, n - cooking_end)
        presion_camara += self.rng.normal(0, 0.3, n)
        presion_camara = np.clip(presion_camara, -5, 10)

        # Presión ventilación
        presion_vent = np.ones(n) * 8.0
        presion_vent[cooking_end:] = np.linspace(8.0, 15.0, n - cooking_end)
        presion_vent += self.rng.normal(0, 0.5, n)
        presion_vent = np.clip(presion_vent, 0, 50)

        # Potencia
        potencia = np.zeros(n)
        potencia[:preheat_end] = np.linspace(35, 45, preheat_end)
        potencia[preheat_end:cooking_end] = 28 + 4 * np.sin(
            2 * np.pi * np.arange(cooking_end - preheat_end) / 600
        )
        potencia[cooking_end:] = np.linspace(28, 6, n - cooking_end)
        potencia += self.rng.normal(0, 0.6, n)
        potencia = np.clip(potencia, 0, 50)

        # Flujo de gas
        flujo_gas = np.zeros(n)
        flujo_gas[:preheat_end] = np.linspace(18, 28, preheat_end)
        flujo_gas[preheat_end:cooking_end] = 22 + 2.5 * np.sin(
            2 * np.pi * np.arange(cooking_end - preheat_end) / 500
        )
        flujo_gas[cooking_end:] = np.linspace(22, 4, n - cooking_end)
        flujo_gas += self.rng.normal(0, 0.2, n)
        flujo_gas = np.clip(flujo_gas, 0, 30)

        # Humedad relativa interior de la cámara
        h_end_preheat = ambient["humidity"] * 0.6
        humidity = np.zeros(n)

        for i in range(n):
            if i < preheat_end:
                frac = i / preheat_end
                base = ambient["humidity"] * (1 - 0.4 * frac)
            elif i < cooking_end:
                frac = (i - preheat_end) / (cooking_end - preheat_end)
                evap = 25 * np.sin(np.pi * frac)
                oscillation = 5 * np.sin(2 * np.pi * i / 300)
                process_base = 40 + evap + oscillation
                blend = np.exp(-6 * frac)
                base = blend * h_end_preheat + (1 - blend) * process_base
            else:
                frac = (i - cooking_end) / (n - cooking_end)
                oscillation = 5 * np.sin(2 * np.pi * i / 500) * (1 - frac)
                base = 40 * (1 - frac) + ambient["humidity"] * frac + oscillation

            temp_effect = np.exp(-0.005 * (temps["temp_zona2"][i] - 120))
            humidity[i] = base * temp_effect

        humidity += self.rng.normal(0, 1.0, n)
        humidity = np.clip(humidity, 0, 100)

        # Temperatura ambiente
        temp_amb = ambient["temp_ambient"] + self.rng.normal(0, 0.5, n).cumsum() * 0.002
        temp_amb = np.clip(temp_amb, -10, 50)

        # Setpoint
        setpoint_arr = np.ones(n) * setpoint
        setpoint_arr[:preheat_end] = np.linspace(60, setpoint, preheat_end)
        setpoint_arr[cooking_end:] = np.linspace(setpoint, 40, n - cooking_end)

        # Posición válvula
        valve = np.zeros(n)
        valve[:preheat_end] = np.linspace(80, 60, preheat_end)
        valve[preheat_end:cooking_end] = 50 + 5 * np.sin(
            2 * np.pi * np.arange(cooking_end - preheat_end) / 800
        )
        valve[cooking_end:] = np.linspace(50, 10, n - cooking_end)
        valve += self.rng.normal(0, 0.5, n)
        valve = np.clip(valve, 0, 100)

        # Velocidad ventilador
        fan_speed = np.zeros(n)
        fan_speed[:preheat_end] = np.linspace(1300, 1000, preheat_end)
        fan_speed[preheat_end:cooking_end] = 1000 + 60 * np.sin(
            2 * np.pi * np.arange(cooking_end - preheat_end) / 600
        )
        fan_speed[cooking_end:] = np.linspace(1000, 1600, n - cooking_end)
        fan_speed += self.rng.normal(0, 4.0, n)
        fan_speed = np.clip(fan_speed, 0, 2000)

        return pd.DataFrame(
            {
                "temp_zona1": temps["temp_zona1"],
                "temp_zona2": temps["temp_zona2"],
                "temp_zona3": temps["temp_zona3"],
                "temp_salida_gases": temp_gases,
                "presion_camara": presion_camara,
                "presion_ventilacion": presion_vent,
                "potencia_kw": potencia,
                "flujo_gas": flujo_gas,
                "humedad_relativa": humidity,
                "temp_ambiente": temp_amb,
                "setpoint_temp": setpoint_arr,
                "posicion_valvula": valve,
                "velocidad_ventilador": fan_speed,
            }
        )

    def _sample_fault_window(self, n: int) -> Tuple[int, int]:
        """Muestrea ventana aleatoria de inicio y fin de falla.

        Args:
            n: Número total de pasos del ciclo.

        Returns:
            Tupla (fault_start, fault_end).
        """
        start_min = int(np.floor(self.fault_start_min_pct * n))
        start_max = int(np.floor(self.fault_start_max_pct * n))
        start_min = max(0, min(start_min, n - 2))
        start_max = max(start_min + 1, min(start_max, n - 1))
        fault_start = int(self.rng.randint(start_min, start_max + 1))

        dur_min = int(np.floor(self.fault_duration_min_pct * n))
        dur_max = int(np.floor(self.fault_duration_max_pct * n))
        dur_min = max(1, min(dur_min, n - fault_start))
        dur_max = max(dur_min, min(dur_max, n - fault_start))
        fault_duration = int(self.rng.randint(dur_min, dur_max + 1))

        fault_end = min(fault_start + fault_duration, n)
        if fault_end <= fault_start:
            fault_end = min(n, fault_start + 1)
        return fault_start, fault_end

    def _sample_severity(self) -> float:
        """Muestrea severidad aleatoria dentro de los límites configurados."""
        return float(self.rng.uniform(self.severity_min, self.severity_max))

    def _inject_fault(
        self,
        df: pd.DataFrame,
        fault_name: str,
        fault_window: Optional[Tuple[int, int]] = None,
        severity: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Inyecta una falla en un ciclo normal.

        Args:
            df: DataFrame del ciclo normal.
            fault_name: Nombre del tipo de falla.
            fault_window: Ventana (start, end) opcional.
            severity: Severidad opcional.

        Returns:
            Tuple con el DataFrame con falla inyectada y la información de la falla.
        """
        n = len(df)
        if fault_window is None:
            fault_start, fault_end = self._sample_fault_window(n)
        else:
            fault_start, fault_end = fault_window
            fault_start = int(np.clip(fault_start, 0, n - 1))
            fault_end = int(np.clip(fault_end, fault_start + 1, n))

        if severity is None:
            severity = self._sample_severity()
        severity = float(np.clip(severity, 0.0, 3.0))
        fault_info = {
            "fault_name": str(fault_name),
            "fault_start": int(fault_start),
            "fault_end": int(fault_end),
            "fault_duration": int(max(0, fault_end - fault_start)),
            "severity": float(severity),
        }

        dfm = df.copy()
        idx = np.arange(fault_start, fault_end)
        if len(idx) == 0:
            fault_info["fault_duration"] = 0
            return dfm, fault_info

        if fault_name == "AISLAMIENTO_DEGRADADO":
            ramp = np.linspace(0, severity, len(idx))
            dfm.loc[idx, "potencia_kw"] *= 1 + 0.55 * ramp
            dfm.loc[idx, "temp_zona3"] -= 35 * severity * ramp
            dfm.loc[idx, "temp_zona1"] += 18 * severity * ramp
            dfm.loc[idx, "temp_salida_gases"] += 20 * severity * ramp
            dfm.loc[idx, "humedad_relativa"] += 15 * severity * ramp

        elif fault_name == "RESISTENCIA_DEGRADADA":
            ramp = np.linspace(0, severity, len(idx))
            dfm.loc[idx, "potencia_kw"] *= 1 - 0.50 * ramp
            dfm.loc[idx, "temp_zona1"] -= 45 * severity * ramp
            dfm.loc[idx, "temp_zona2"] -= 35 * severity * ramp
            dfm.loc[idx, "temp_zona3"] -= 25 * severity * ramp
            dfm.loc[idx, "temp_salida_gases"] -= 20 * severity * ramp
            dfm.loc[idx, "humedad_relativa"] += 10 * severity * ramp

        elif fault_name == "REFRACTARIO_EROSIONADO":
            noise = self.rng.normal(0, 18 * severity, len(idx))
            dfm.loc[idx, "temp_zona2"] += noise
            dfm.loc[idx, "temp_zona1"] += self.rng.normal(0, 14 * severity, len(idx))
            dfm.loc[idx, "temp_zona3"] += self.rng.normal(0, 10 * severity, len(idx))
            dfm.loc[idx, "presion_camara"] += self.rng.normal(
                0, 2.5 * severity, len(idx)
            )
            burst_period = self.rng.randint(15, 30)
            dfm.loc[idx, "presion_ventilacion"] += 6 * severity * np.abs(
                np.sin(2 * np.pi * np.arange(len(idx)) / burst_period)
            )
            dfm.loc[idx, "humedad_relativa"] += self.rng.normal(
                0, 8 * severity, len(idx)
            )

        elif fault_name == "SENSOR_DESCALIBRADO":
            sensor_col = self.rng.choice(
                ["temp_zona1", "temp_zona2", "temp_zona3"]
            )
            mode = self.rng.choice(["frozen", "spike", "drift"])

            fault_info["fault_sensor"] = str(sensor_col).upper()
            fault_info["fault_mode"] = str(mode).upper()
            if mode == "frozen":
                dfm.loc[idx, sensor_col] = dfm.loc[fault_start, sensor_col]
                dfm.loc[idx, "potencia_kw"] += 8 * severity * np.sin(
                    2 * np.pi * np.arange(len(idx)) / 15
                )
                dfm.loc[idx, "temp_salida_gases"] += self.rng.normal(
                    0, 10 * severity, len(idx)
                )
            elif mode == "spike":
                spikes = self.rng.choice(
                    idx, size=max(1, len(idx) // 4), replace=False
                )
                dfm.loc[spikes, sensor_col] += self.rng.uniform(
                    25, 70, len(spikes)
                ) * severity
            else:
                drift = np.linspace(0, 60 * severity, len(idx))
                dfm.loc[idx, sensor_col] += drift
                dfm.loc[idx, "potencia_kw"] -= (
                    8 * severity * np.linspace(0, 1, len(idx))
                )

        elif fault_name == "VALVULA_OBSTRUIDA":
            ramp = np.linspace(0, severity, len(idx))
            dfm.loc[idx, "flujo_gas"] *= 1 - 0.45 * ramp

            # El flujo cae pero el controlador intenta abrir la valvula (desacoplo).
            dfm.loc[idx, "posicion_valvula"] += 15 * severity * ramp

            # Hotspot en entrada por combustion ineficiente.
            dfm.loc[idx, "temp_zona1"] += 25 * severity * ramp

            # Enfriamiento progresivo por falta de alcance termico.
            dfm.loc[idx, "temp_zona2"] -= 18 * severity * ramp
            dfm.loc[idx, "temp_zona3"] -= 22 * severity * ramp

            # Compensacion de aire del sistema de control.
            dfm.loc[idx, "presion_ventilacion"] += 6 * severity * ramp
            dfm.loc[idx, "presion_camara"] += 5 * severity * ramp

            # Al bajar temperatura en zonas 2 y 3, la HR tiende a subir.
            dfm.loc[idx, "humedad_relativa"] += 12 * severity * ramp

        elif fault_name == "VENTILADOR_DEFECTUOSO":
            ramp = np.linspace(0, severity, len(idx))
            dfm.loc[idx, "velocidad_ventilador"] *= 1 - 0.80 * ramp
            dfm.loc[idx, "presion_camara"] += 12 * severity * ramp
            dfm.loc[idx, "temp_zona1"] += 12 * severity * ramp
            dfm.loc[idx, "temp_zona2"] += 14 * severity * ramp
            dfm.loc[idx, "temp_zona3"] += 22 * severity * ramp
            dfm.loc[idx, "temp_salida_gases"] += 18 * severity * ramp
            dfm.loc[idx, "temp_ambiente"] += self.rng.normal(
                0, 1.5 * severity, len(idx)
            )
            dfm.loc[idx, "humedad_relativa"] += 20 * severity * ramp

        # Clamp a rangos de sensores
        for col, (lo, hi) in self.sensor_ranges.items():
            if col in dfm.columns:
                dfm[col] = dfm[col].clip(lo, hi)

        dfm["humedad_relativa"] = dfm["humedad_relativa"].clip(0, 100)

        fault_info["fault_name"] = fault_name

        return dfm, fault_info

    def generate_cycle(
        self,
        cycle_id: int,
        fault_name: str = "NORMAL",
    ) -> pd.DataFrame:
        """Genera un ciclo completo (normal o con falla).

        Args:
            cycle_id: Identificador del ciclo.
            fault_name: Tipo de falla a inyectar ("NORMAL" para sin falla).

        Returns:
            DataFrame con timestamp, cycle_id, sensores y fault_name.
        """
        ambient = self._generate_ambient_conditions()
        df = self._generate_normal_cycle(ambient)

        df["fault_name"] = "NORMAL"

        if fault_name != "NORMAL":
            df, fault_meta = self._inject_fault(df, fault_name)

            fault_start = int(fault_meta["fault_start"])
            fault_end = int(fault_meta["fault_end"])

            if fault_end > fault_start:
                label = fault_name
                if fault_name == "SENSOR_DESCALIBRADO":
                    label = (f"{fault_name}_{fault_meta.get('fault_sensor', 'unknown').upper()}_{fault_meta.get('fault_mode', 'unknown').upper()}")
                
                df.loc[fault_start:fault_end - 1, "fault_name"] = label

        base_time = pd.Timestamp("2026-01-01") + pd.Timedelta(
            hours=cycle_id * 12
        )
        df.insert(
            0,
            "timestamp",
            pd.date_range(
                base_time, periods=len(df), freq=f"{self.sampling_interval}s"
            ),
        )
        df.insert(1, "cycle_id", cycle_id)

        return df

    def _build_cycle_plan(
        self,
        n_cycles: int,
        normal_ratio: Optional[float] = None,
        fault_ratio: Optional[float] = None,
    ) -> List[str]:
        """Construye plan de ciclos (qué tipo de falla asignar a cada ciclo).

        Args:
            n_cycles: Número total de ciclos a generar.

        Returns:
            Lista de nombres de falla para cada ciclo.
        """
        normal_ratio = float(
            normal_ratio
            if normal_ratio is not None
            else self.data_cfg.get("normal_ratio", 0.0)
        )
        fault_ratio = float(
            fault_ratio
            if fault_ratio is not None
            else self.data_cfg.get("fault_ratio", 0.0)
        )

        if min(normal_ratio, fault_ratio) < 0:
            raise ValueError("normal_ratio/fault_ratio deben ser >= 0")

        fault_types = [
            f["name"]
            for f in self.fault_types
            if 1 <= int(f.get("label", -1)) <= 6
        ]
        if len(fault_types) == 0:
            fault_ratio = 0.0

        ratio_sum = normal_ratio + fault_ratio
        if ratio_sum <= 0:
            raise ValueError("La suma de ratios debe ser > 0")

        normal_ratio /= ratio_sum
        fault_ratio /= ratio_sum

        n_normal = int(np.floor(n_cycles * normal_ratio))
        n_fault = int(np.floor(n_cycles * fault_ratio))

        remainder = n_cycles - (n_normal + n_fault)
        if remainder > 0:
            if normal_ratio >= fault_ratio or len(fault_types) == 0:
                n_normal += remainder
            else:
                n_fault += remainder

        if len(fault_types) == 0 and n_fault > 0:
            n_normal += n_fault
            n_fault = 0

        plan: List[str] = []
        plan.extend(["NORMAL"] * n_normal)
        if n_fault > 0:
            plan.extend(
                [str(self.rng.choice(fault_types)) for _ in range(n_fault)]
            )

        if len(plan) != n_cycles:
            raise ValueError(
                f"Plan de ciclos inconsistente: {len(plan)} != {n_cycles}"
            )

        permutation = self.rng.permutation(n_cycles)
        plan = [plan[i] for i in permutation]
        return plan

    def generate_dataset(
        self,
        n_cycles: int,
        split_name: Optional[str] = "train",
        normal_ratio: Optional[float] = None,
        fault_ratio: Optional[float] = None,
    ) -> pd.DataFrame:
        """Genera un dataset con n_cycles ciclos.

        Args:
            n_cycles: Número de ciclos a generar.
            split_name: Nombre del split (para logging).

        Returns:
            DataFrame con todos los ciclos concatenados.
        """
        
        if split_name is not None:
            logger.info(
                "Generando %d ciclos para split '%s'...", n_cycles, split_name
            )
        else:
            logger.info("Generando dataset con %d ciclos...", n_cycles)

        cycle_plan = self._build_cycle_plan(
            n_cycles,
            normal_ratio=normal_ratio,
            fault_ratio=fault_ratio,
        )
        all_cycles = []

        for cycle_id, fault_name in enumerate(cycle_plan):
            df_cycle = self.generate_cycle(
                cycle_id=cycle_id,
                fault_name=fault_name,
            )
            all_cycles.append(df_cycle)

        dataset = pd.concat(all_cycles, ignore_index=True)
        return dataset

    def _resolve_split_ratios(
        self, split_name: str
    ) -> Tuple[float, float]:
        """Resuelve ratios normal/fault para un split específico.

        Args:
            split_name: Nombre del split ("train", "val", "test").

        Returns:
            Tupla (normal_ratio, fault_ratio) normalizados.
        """
        split_ratio_cfg = self.data_cfg.get("split_ratios", {})
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
                f"Ratios invalidos en split '{split_name}': deben ser >= 0"
            )

        available_faults = [
            f["name"]
            for f in self.fault_types
            if 1 <= int(f.get("label", -1)) <= 6
        ]
        if len(available_faults) == 0:
            fault_ratio = 0.0

        ratio_sum = normal_ratio + fault_ratio
        if ratio_sum <= 0:
            raise ValueError(
                f"Split '{split_name}' sin probabilidad asignada (normal/fault)."
            )

        normal_ratio, fault_ratio = normal_ratio / ratio_sum, fault_ratio / ratio_sum
        return normal_ratio, fault_ratio

    def generate_full_dataset_from_splits(self) -> Tuple[pd.DataFrame, dict]:
        """Genera dataset completo a partir de splits train/val/test.

        Returns:
            Tupla (DataFrame completo, dict resumen).
        """
        raw_dir = ensure_dir(os.path.dirname(self.paths["raw_data"]))
        splits_dir = ensure_dir(self.paths["splits"])

        full_filename = os.path.basename(self.paths["raw_data"])

        splits = {
            "train": int(self.data_cfg.get("n_cycles_train", 0)),
            "val": int(self.data_cfg.get("n_cycles_val", 0)),
            "test": int(self.data_cfg.get("n_cycles_test", 0)),
        }

        dfs = []
        cycle_offset = 0

        for split_name, n_cycles in splits.items():
            if n_cycles <= 0:
                continue

            normal_ratio, fault_ratio = self._resolve_split_ratios(split_name)
            self.data_cfg["normal_ratio"] = normal_ratio
            self.data_cfg["fault_ratio"] = fault_ratio

            logger.info(
                "Ratios para split '%s' -> normal=%.3f, fault=%.3f",
                split_name,
                normal_ratio,
                fault_ratio,
            )

            df_split = self.generate_dataset(
                n_cycles=n_cycles, split_name=split_name
            )

            df_split = df_split.copy()
            df_split["cycle_id"] = df_split["cycle_id"] + cycle_offset
            df_split["timestamp"] = pd.to_datetime(
                df_split["timestamp"]
            ) + pd.to_timedelta(cycle_offset * 12, unit="h")

            # Guardar split individual
            split_path = os.path.join(splits_dir, f"{split_name}.csv")
            df_split.to_csv(split_path, index=False)
            logger.info("Split '%s' guardado en %s", split_name, split_path)

            dfs.append(df_split)
            cycle_offset += n_cycles

        if not dfs:
            raise ValueError(
                "No hay ciclos configurados en n_cycles_train/val/test"
            )

        df_full = (
            pd.concat(dfs, ignore_index=True)
            .sort_values(["timestamp", "cycle_id"])
            .reset_index(drop=True)
        )

        # Guardar dataset completo
        full_path = os.path.join(str(raw_dir), full_filename)
        df_full.to_csv(full_path, index=False)

        resumen = {
            "output_csv_path": full_path,
            "n_cycles_generados": int(df_full["cycle_id"].nunique()),
            "n_rows": int(len(df_full)),
            "rows_por_ciclo": int(self.n_steps),
            "fault_distribution": df_full["fault_name"]
            .value_counts()
            .to_dict(),
            "timestamp_min": str(df_full["timestamp"].min()),
            "timestamp_max": str(df_full["timestamp"].max()),
        }

        logger.info(
            "Dataset completo: %s (%d filas, %d ciclos)",
            full_path,
            resumen["n_rows"],
            resumen["n_cycles_generados"],
        )

        return df_full, resumen


def run_dataset_generation(config: dict) -> pd.DataFrame:
    """Ejecuta la generacion completa del dataset sintetico.

    Args:
        config: Diccionario de configuracion completo del proyecto.

    Returns:
        DataFrame con el dataset completo generado.
    """
    generator = OvenDataGenerator(config=config)
    df_full, resumen = generator.generate_full_dataset_from_splits()

    logger.info("Distribucion de fallas: %s", resumen["fault_distribution"])

    return df_full

def run_xai_datasets_generation(seeds: List[int], config: dict) -> Dict[str, pd.DataFrame]:
    """Genera datasets específicos para la capa de explicabilidad XAI.

    Args:
        seeds: Lista de semillas para reproducibilidad.
        config: Diccionario de configuracion completo del proyecto.

    Returns:
        Diccionario con DataFrames para cada dataset XAI generado.
    """

    interpretability_val = OvenDataGenerator(config=config, seed=seeds[0]).generate_dataset(
        n_cycles=1500,
        split_name=None,
        normal_ratio=0.5,
        fault_ratio=0.5,
    )

    logger.info("Semilla %d - Dataset 'interpretability_val' generado con %d ciclos. Distribucion de fallas: %s",
        seeds[0],
        1500,
        interpretability_val["fault_name"].value_counts().to_dict()
    )

    interpretability_val.to_csv("data/raw/interpretability_val.csv", index=False)

    actions_system_eval = OvenDataGenerator(config=config, seed=seeds[1]).generate_dataset(
        n_cycles=500,
        split_name=None,
        normal_ratio=0.8,
        fault_ratio=0.2,
    )

    logger.info("Semilla %d - Dataset 'actions_system_eval' generado con %d ciclos. Distribucion de fallas: %s",
        seeds[1],
        500,
        actions_system_eval["fault_name"].value_counts().to_dict()
    )
    actions_system_eval.to_csv("data/raw/actions_system_eval.csv", index=False)

    xai_background = OvenDataGenerator(config=config, seed=seeds[2]).generate_dataset(
        n_cycles=25,
        split_name=None,
        normal_ratio=0.5,
        fault_ratio=0.5,
    )

    logger.info("Semilla %d - Dataset 'xai_background' generado con %d ciclos.", seeds[2], 25)
    xai_background.to_csv("data/raw/xai_background.csv", index=False)

    logger.info("Todos los datasets XAI generados y guardados en data/raw/")