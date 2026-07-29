"""
Pasteurization plant simulator (Digital Twin) - V3.4 PID/supervisor.

This module encapsulates the data_creation.ipynb logic:

    - Variable flow around 5000 L/h nominal (rejection-sampled, +-400 L/h,
      clamped to [3500, 5500] L/h)
    - Open-loop control heuristic: T_servicio approximates the compensation
      a PID/supervisor would apply for fouling, flow and inlet temperature,
      plus actuation noise (does not invert the thermal balance)
    - A final safety guard recalculates T_servicio only for the rare
      extreme cases where the heuristic saturates below the pasteurization
      threshold
    - Stochastic fouling and batch-to-batch milk property variability
    - Thermal, hydraulic and energy calculations for the pasteurization line

The public function ``generar_dataset_pasteurizacion`` keeps the original
pipeline contract and returns the generated dataset. When ``save=True``,
it writes the raw artifact:

    - data/raw/pasteurizacion_dataset_simulado.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.paths import (
    RAW_DATA_DIR,
    RAW_DATASET_PATH,
)


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration for the V3.4 PID/supervisor pasteurization simulator."""

    num_dias: int = 180
    freq_muestreo: int = 5
    seed: int = 1
    u_clean: float = 3500.0
    area: float = 15.0
    t_out_objetivo: float = 72.3
    margen_control: float = 0.30
    flow_operacion_l_h: float = 5000.0
    flow_std_l_h: float = 400.0
    flow_min_l_h: float = 3500.0
    flow_max_l_h: float = 5500.0
    flow_diseno_termico_l_h: float = 4500.0
    flow_ref_eficiencia_l_h: float = 5150.0
    t_servicio_min: float = 76.0
    t_servicio_max: float = 95.0
    r_fouling_inicial: float = 0.00001
    r_fouling_max_operativo: float = 0.0008
    cp_diseno: float = 3890.0
    rho_diseno: float = 1030.0
    k_hidraulica: float = 2e-8
    eta_caldera: float = 0.90
    eta_bomba_max: float = 0.78
    k_curva_bomba: float = 2.5e-7
    flow_bep_l_h: float = 4800.0
    p_fixed_kw: float = 15.0
    t_servicio_base: float = 79.0
    pid_coef_fouling: float = 25000.0
    pid_coef_flujo: float = 0.0010
    pid_coef_tin: float = 0.12
    pid_ruido_std: float = 0.40

    @property
    def t_out_control(self) -> float:
        return self.t_out_objetivo + self.margen_control

    @property
    def total_registros(self) -> int:
        return int((self.num_dias * 24 * 60) / self.freq_muestreo)


DEFAULT_CONFIG = SimulatorConfig()


def calcular_effectiveness(
    flujo_l_h: float,
    rho: float,
    cp: float,
    fouling_factor: float,
    config: SimulatorConfig = DEFAULT_CONFIG,
) -> tuple[float, float]:
    """
    Calculate heat exchanger effectiveness and cold-side heat capacity.

    Uses the epsilon-NTU formulation with a channeling correction when the
    actual flow exceeds the thermal design flow of 4500 L/h.
    """
    flujo_kg_s = (flujo_l_h / 3600.0) * (rho / 1000.0)

    if flujo_l_h > config.flow_diseno_termico_l_h:
        factor_canalizacion = (config.flow_diseno_termico_l_h / flujo_l_h) ** 0.85
        u_efectivo = config.u_clean * factor_canalizacion
    else:
        u_efectivo = config.u_clean

    u_dirty = 1.0 / ((1.0 / u_efectivo) + fouling_factor)
    c_min = flujo_kg_s * cp
    ntu = (u_dirty * config.area) / c_min
    effectiveness = ntu / (1.0 + ntu)
    return effectiveness, c_min


def calcular_t_out_y_q(
    t_in_leche: float,
    t_servicio: float,
    flujo_l_h: float,
    rho: float,
    cp: float,
    fouling_factor: float,
    config: SimulatorConfig = DEFAULT_CONFIG,
) -> tuple[float, float, float, float]:
    """Calculate outlet temperature and heat transferred for a given service temperature."""
    effectiveness, c_min = calcular_effectiveness(
        flujo_l_h=flujo_l_h,
        rho=rho,
        cp=cp,
        fouling_factor=fouling_factor,
        config=config,
    )
    q_transferido_w = effectiveness * c_min * (t_servicio - t_in_leche)
    t_out_leche = t_in_leche + (q_transferido_w / c_min)
    return t_out_leche, q_transferido_w, effectiveness, c_min


def calcular_t_servicio_minima(
    t_in_leche: float,
    flujo_l_h: float,
    rho: float,
    cp: float,
    fouling_factor: float,
    t_out_deseada: float,
    config: SimulatorConfig = DEFAULT_CONFIG,
) -> float:
    """
    Calculate the minimum service temperature required to reach a desired outlet temperature
    given the current physical state.
    """
    effectiveness, _ = calcular_effectiveness(
        flujo_l_h=flujo_l_h,
        rho=rho,
        cp=cp,
        fouling_factor=fouling_factor,
        config=config,
    )
    effectiveness = max(effectiveness, 1e-6)
    return t_in_leche + (t_out_deseada - t_in_leche) / effectiveness


def calcular_t_servicio_pid(
    t_in_leche: float,
    flujo_l_h: float,
    rho: float,
    cp: float,
    fouling_factor: float,
    config: SimulatorConfig = DEFAULT_CONFIG,
) -> float:
    """
    Open-loop operational control heuristic (approximate PID).

    Does not invert the thermal balance to impose T_out_leche. Generates
    T_servicio as an approximate control action depending on fouling, flow
    and inlet temperature, plus actuation noise. This allows feasible,
    marginal and infeasible cases to exist (matches data_creation.ipynb).
    """
    compensacion_fouling = fouling_factor * config.pid_coef_fouling
    compensacion_flujo = config.pid_coef_flujo * (flujo_l_h - config.flow_operacion_l_h)
    compensacion_tin = config.pid_coef_tin * (4.0 - t_in_leche)
    ruido_pid = np.random.normal(0.0, config.pid_ruido_std)

    t_servicio = (
        config.t_servicio_base
        + compensacion_fouling
        + compensacion_flujo
        + compensacion_tin
        + ruido_pid
    )

    return float(np.clip(t_servicio, config.t_servicio_min, config.t_servicio_max))


def generar_dataset_pasteurizacion(
    num_dias: int = 180,
    freq_muestreo: int = 5,
    seed: int = 1,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate the V3.4 raw pasteurization dataset.

    Implements the open-loop PID/supervisor control heuristic, variable
    flow and stochastic fouling from data_creation.ipynb. A final safety
    guard keeps T_out_leche compliant in the vast majority of production
    records, but does not force full 100% compliance (matches the notebook,
    which reports ~99.6% of records at or above 72.3 °C).

    Parameters
    ----------
    num_dias : int
        Number of simulation days.
    freq_muestreo : int
        Sampling frequency in minutes.
    seed : int
        Random seed for reproducibility.
    save : bool
        If True, saves the CSV to data/raw/pasteurizacion_dataset_simulado.csv.

    Returns
    -------
    pd.DataFrame
        Generated raw dataset with columns:
        Time_min, T_in_leche, F_flow, T_servicio, t_ciclo,
        Delta_P, E_consumo, T_out_leche, Is_Cleaning
    """
    config = SimulatorConfig(
        num_dias=num_dias,
        freq_muestreo=freq_muestreo,
        seed=seed,
    )

    np.random.seed(config.seed)
    data = []
    fouling_factor = config.r_fouling_inicial
    tiempo_desde_limpieza = 0

    cp_actual = config.cp_diseno
    rho_actual = config.rho_diseno

    print(
        f"Generando {config.total_registros} registros "
        f"(V3.4: PID/supervisor, datos de planta seguros)..."
    )

    for i in range(config.total_registros):
        # 1. Entradas de proceso.
        estacion = np.sin(2.0 * np.pi * i / config.total_registros)
        t_in_leche = 4.0 + (3.0 * estacion) + np.random.normal(0.0, 0.3)
        t_in_leche = float(np.clip(t_in_leche, 0.0, 10.0))

        while True:
            flujo_l_h = float(np.random.normal(config.flow_operacion_l_h, config.flow_std_l_h))
            if config.flow_min_l_h <= flujo_l_h <= config.flow_max_l_h:
                break

        # 2. Logica de limpieza CIP.
        necesita_limpieza = (
            tiempo_desde_limpieza > (9 * 60)
            or fouling_factor > config.r_fouling_max_operativo
        )

        if necesita_limpieza:
            is_cleaning = 1
            fouling_factor = config.r_fouling_inicial
            tiempo_desde_limpieza = 0
            cp_actual = config.cp_diseno + np.random.normal(0.0, 50.0)
            rho_actual = config.rho_diseno + np.random.normal(0.0, 10.0)
        else:
            is_cleaning = 0

        # 3. Control PID/supervisor de temperatura.
        t_servicio = calcular_t_servicio_pid(
            t_in_leche=t_in_leche,
            flujo_l_h=flujo_l_h,
            rho=rho_actual,
            cp=cp_actual,
            fouling_factor=fouling_factor,
            config=config,
        )

        t_out_leche, q_transferido_w, _, c_min = calcular_t_out_y_q(
            t_in_leche=t_in_leche,
            t_servicio=t_servicio,
            flujo_l_h=flujo_l_h,
            rho=rho_actual,
            cp=cp_actual,
            fouling_factor=fouling_factor,
            config=config,
        )

        # Guardia final de datos de planta: si un caso extremo no puede cumplir
        # por saturacion, se asume desviacion de valvula/recirculacion y se
        # recalcula hacia el setpoint de control (no se fuerza el resultado).
        if t_out_leche < config.t_out_objetivo:
            t_servicio_requerida = calcular_t_servicio_minima(
                t_in_leche=t_in_leche,
                flujo_l_h=flujo_l_h,
                rho=rho_actual,
                cp=cp_actual,
                fouling_factor=fouling_factor,
                t_out_deseada=config.t_out_control,
                config=config,
            )
            t_servicio = float(
                np.clip(t_servicio_requerida, config.t_servicio_min, config.t_servicio_max)
            )
            t_out_leche, q_transferido_w, _, c_min = calcular_t_out_y_q(
                t_in_leche=t_in_leche,
                t_servicio=t_servicio,
                flujo_l_h=flujo_l_h,
                rho=rho_actual,
                cp=cp_actual,
                fouling_factor=fouling_factor,
                config=config,
            )

        # 4. Fouling dinamico (despues de la termodinamica).
        if is_cleaning == 0:
            ruido_cinetico = np.random.normal(1.0, 0.05)
            tasa_deposicion = 3e-8 * np.exp(0.04 * t_servicio) * ruido_cinetico
            factor_arrastre = (flujo_l_h / config.flow_diseno_termico_l_h) ** 1.2
            tasa_neta = tasa_deposicion / factor_arrastre
            fouling_factor += tasa_neta
            tiempo_desde_limpieza += config.freq_muestreo

        # 5. Hidraulica y potencia.
        factor_obstruccion = 1.0 + (fouling_factor * 5000.0)
        delta_p_bar = config.k_hidraulica * (flujo_l_h ** 2) * factor_obstruccion
        delta_p_bar += np.random.normal(0.0, 0.02)
        delta_p_bar = max(delta_p_bar, 0.01)

        delta_p_pa = delta_p_bar * 100000.0
        flujo_m3_s = (flujo_l_h / 1000.0) / 3600.0

        eta_bomba = config.eta_bomba_max - (
            config.k_curva_bomba * (flujo_l_h - config.flow_bep_l_h) ** 2
        )
        eta_bomba = max(0.30, eta_bomba)

        potencia_bomba_w = (flujo_m3_s * delta_p_pa) / eta_bomba

        # Penalizacion energetica por canalizacion/maldistribucion a alto caudal:
        # operar por encima del caudal de referencia de eficiencia empeora el
        # rendimiento efectivo de generacion/distribucion de calor.
        if flujo_l_h > config.flow_ref_eficiencia_l_h:
            factor_canalizacion_energia = (
                config.flow_ref_eficiencia_l_h / flujo_l_h
            ) ** 0.85
        else:
            factor_canalizacion_energia = 1.0

        consumo_termico_w = q_transferido_w / (config.eta_caldera * factor_canalizacion_energia)

        e_consumo_kw = (consumo_termico_w + potencia_bomba_w) / 1000.0
        # Perdidas fijas independientes del caudal: penalizan operar por debajo
        # del caudal nominal, ya que se reparten entre menos litros producidos.
        e_consumo_kw += config.p_fixed_kw
        e_consumo_kw += np.random.normal(0.0, 5.0)

        # 6. Ruido de instrumentacion en temperatura de salida.
        # Medida instrumental: se anade ruido, pero no se fuerza a cumplir
        # 72.3 C, para que el modelo aprenda casos factibles y no factibles.
        t_out_medida = t_out_leche + np.random.normal(0.0, 0.05)
        t_out_medida = float(np.clip(t_out_medida, 40.0, 95.0))

        data.append(
            [
                i * config.freq_muestreo,
                round(t_in_leche, 2),
                round(flujo_l_h, 2),
                round(t_servicio, 2),
                round(tiempo_desde_limpieza, 0),
                round(delta_p_bar, 3),
                round(e_consumo_kw, 2),
                round(t_out_medida, 2),
                int(is_cleaning),
            ]
        )

    df = pd.DataFrame(
        data,
        columns=[
            "Time_min",
            "T_in_leche",
            "F_flow",
            "T_servicio",
            "t_ciclo",
            "Delta_P",
            "E_consumo",
            "T_out_leche",
            "Is_Cleaning",
        ],
    )

    if save:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATASET_PATH, index=False)

    return df
