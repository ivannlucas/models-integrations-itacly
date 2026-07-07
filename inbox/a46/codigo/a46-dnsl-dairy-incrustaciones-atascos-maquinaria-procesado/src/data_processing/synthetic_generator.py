
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MILK_TYPES: Tuple[str, ...] = ("whole", "semi_skim", "skim", "high_solids")
ASSET_FAMILIES: Tuple[str, ...] = ("standard_phe", "robust_phe", "compact_phe")
PROD_STAGE_NAMES: Tuple[str, ...] = ("stable", "incipient", "advanced")
STAGE_ID_TO_NAME = {-1: "not_production", 0: "stable", 1: "incipient", 2: "advanced"}
CIP_EVENT_TYPES = {"scheduled_CIP", "CIP_extra"}
UNPLANNED_EVENT_TYPES = {"unclog", "mechanical_clean", "CIP_extra"}
@dataclass(frozen=True)
class MilkSpec:
    protein_g_L: Tuple[float, float]
    fat_g_L: Tuple[float, float]
    solids_g_L: Tuple[float, float]
    Ca_mM: Tuple[float, float]
    PO4_mM: Tuple[float, float]
    pH: Tuple[float, float]
    fouling_factor: float

MILK_SPECS: Dict[str, MilkSpec] = {
    "whole": MilkSpec((30.0, 36.0), (30.0, 42.0), (115.0, 140.0), (12.0, 22.0), (6.0, 13.0), (6.45, 6.75), 1.00),
    "semi_skim": MilkSpec((30.0, 36.0), (10.0, 22.0), (105.0, 130.0), (12.0, 22.0), (6.0, 13.0), (6.45, 6.75), 0.90),
    "skim": MilkSpec((30.0, 36.0), (0.0, 5.0), (95.0, 120.0), (12.0, 22.0), (6.0, 13.0), (6.45, 6.75), 0.85),
    "high_solids": MilkSpec((32.0, 40.0), (15.0, 35.0), (140.0, 190.0), (14.0, 25.0), (7.0, 15.0), (6.40, 6.75), 1.25),
}


@dataclass(frozen=True)
class EquipConsts:
    A_heat_m2: float = 5.0
    k_dep_org_W_mK: float = 0.25
    k_dep_min_W_mK: float = 1.20
    rho_dep_org_kg_m3: float = 1050.0
    rho_dep_min_kg_m3: float = 1400.0

@dataclass(frozen=True)
class CleaningParams:
    cip_k_org_alk: float = 1.6e-3
    cip_k_min_alk: float = 5.5e-4
    cip_k_org_acid: float = 5.0e-4
    cip_k_min_acid: float = 1.45e-3

@dataclass
class GeneratorConfig:
    out_telemetry: str
    out_maintenance: str
    out_meta: str
    assets: int = 10
    cycles_per_asset: int = 18
    dt: int = 60
    seed: int = 7
    fouling_anchor_rf: float = 8.5e-4
    rf_stage_thr_incipient: Optional[float] = None
    rf_stage_thr_advanced: Optional[float] = None
    fouling_horizon_min: int = 30
    clog_horizon_min: int = 15
    unplanned_horizon_min: int = 60
    stable_hours_min: float = 2.0
    stable_hours_max: float = 18.0
    incipient_hours_min: float = 1.0
    incipient_hours_max: float = 8.0
    advanced_hours_min: float = 0.5
    advanced_hours_max: float = 6.0
    idle_hours_min: float = 0.25
    idle_hours_max: float = 2.0
    inter_cycle_gap_min: int = 180
    inter_cycle_gap_max: int = 1440
    emit_idle_phase: bool = False
    no_noise: bool = False
    no_cip_logs: bool = False

    def resolved_stage_thresholds(self) -> Tuple[float, float]:
        incipient = self.rf_stage_thr_incipient if self.rf_stage_thr_incipient is not None else 0.65 * self.fouling_anchor_rf
        advanced = self.rf_stage_thr_advanced if self.rf_stage_thr_advanced is not None else 1.30 * self.fouling_anchor_rf
        if advanced <= incipient:
            raise ValueError("rf_stage_thr_advanced must be greater than rf_stage_thr_incipient.")
        return float(incipient), float(advanced)

@dataclass
class AssetProfile:
    asset_id: str
    family: str
    UA_clean_WK: float
    flow_sp_base: float
    hot_loop_sp_base: float
    Th_sp_base: float
    Tc_sp_base: float
    dP0_kPa: float
    dP_flow_k: float
    dP_foul_k: float
    dP_foul_pow: float
    fouling_sensitivity: float
    clog_sensitivity: float
    cleaning_efficiency: float
    vibration_factor: float

@dataclass
class EpisodePlan:
    episode_id: str
    batch_id: str
    milk_type: str
    protein_g_L: float
    fat_g_L: float
    solids_g_L: float
    Ca_mM: float
    PO4_mM: float
    pH: float
    thermal_history_factor: float
    flow_sp_kg_s: float
    hot_loop_sp_kg_s: float
    Th_sp_C: float
    Tc_sp_C: float
    stable_s: int
    incipient_s: int
    advanced_s: int
    maintenance_type: str
    maintenance_s: int
    planned_cip_after: bool
    cip_alk_s: int
    cip_acid_s: int
    idle_s: int
    clog_present: bool
    clog_start_s: int
    clog_end_s: int
    clog_flow_drop_frac: float
    clog_dP_mult: float
    severity_index: float
    stage_targets: Dict[str, Tuple[float, float]]
    planned_cip_strength: float
    unplanned_cip_strength: float

    @property
    def production_s(self) -> int:
        return self.stable_s + self.incipient_s + self.advanced_s

    @property
    def total_s(self) -> int:
        return self.production_s + self.maintenance_s + self.cip_alk_s + self.cip_acid_s + self.idle_s

    def observed_s(self, emit_idle_phase: bool) -> int:
        return self.production_s + self.maintenance_s + self.cip_alk_s + self.cip_acid_s + (self.idle_s if emit_idle_phase else 0)

@dataclass
class DriftState:
    Th_in: float = 0.0
    Tc_in: float = 0.0
    Th_out: float = 0.0
    Tc_out: float = 0.0
    flow: float = 0.0
    P_in: float = 0.0
    P_out: float = 0.0
    vib: float = 0.0

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))

def smoothstep01(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)

def cp_milk_proxy(protein_g_L: float, fat_g_L: float, solids_g_L: float) -> float:
    base = 3900.0
    cp = base - 1.5 * (solids_g_L - 110.0) - 2.0 * (fat_g_L - 15.0)
    return float(np.clip(cp, 3300.0, 4200.0))

def eps_ntu_countercurrent(UA: float, Ch: float, Cc: float) -> float:
    Cmin = min(Ch, Cc)
    Cmax = max(Ch, Cc)
    if Cmin <= 0.0 or Cmax <= 0.0:
        return 0.0
    Cr = Cmin / Cmax
    NTU = UA / Cmin
    if abs(1.0 - Cr) < 1e-9:
        return NTU / (1.0 + NTU)
    num = 1.0 - math.exp(-NTU * (1.0 - Cr))
    den = 1.0 - Cr * math.exp(-NTU * (1.0 - Cr))
    return float(num / max(den, 1e-9))

def UA_from_deposit(m_org: float, m_min: float, UA_clean_WK: float, equip: EquipConsts) -> Tuple[float, float]:
    m_org = max(float(m_org), 0.0)
    m_min = max(float(m_min), 0.0)
    thickness_org = m_org / equip.rho_dep_org_kg_m3
    thickness_min = m_min / equip.rho_dep_min_kg_m3
    Rf_org = thickness_org / equip.k_dep_org_W_mK
    Rf_min = thickness_min / equip.k_dep_min_W_mK
    Rf = Rf_org + Rf_min
    UA = 1.0 / (1.0 / UA_clean_WK + (Rf / equip.A_heat_m2))
    return float(UA), float(Rf)

def masses_from_Rf(Rf: float, org_resistance_share: float, equip: EquipConsts) -> Tuple[float, float]:
    org_resistance_share = clamp(org_resistance_share, 0.05, 0.95)
    Rf_org = float(Rf) * org_resistance_share
    Rf_min = max(float(Rf) - Rf_org, 0.0)
    thickness_org = Rf_org * equip.k_dep_org_W_mK
    thickness_min = Rf_min * equip.k_dep_min_W_mK
    m_org = thickness_org * equip.rho_dep_org_kg_m3
    m_min = thickness_min * equip.rho_dep_min_kg_m3
    return float(m_org), float(m_min)

def hx_step(
    Th_in_C: float,
    Tc_in_C: float,
    m_hot_kg_s: float,
    m_prod_kg_s: float,
    UA_WK: float,
    protein_g_L: float,
    fat_g_L: float,
    solids_g_L: float,
) -> Tuple[float, float, float, float]:
    cp_hot = 4180.0
    cp_prod = cp_milk_proxy(protein_g_L, fat_g_L, solids_g_L)
    Ch = m_hot_kg_s * cp_hot
    Cc = m_prod_kg_s * cp_prod
    eff = eps_ntu_countercurrent(UA_WK, Ch, Cc)
    dT_in = Th_in_C - Tc_in_C
    Q = eff * min(Ch, Cc) * dT_in
    Th_out = Th_in_C - Q / max(Ch, 1e-9)
    Tc_out = Tc_in_C + Q / max(Cc, 1e-9)
    return float(Th_out), float(Tc_out), float(Q), float(cp_prod)

def dP_proxy_kPa(m_total: float, flow_kg_s: float, profile: AssetProfile) -> float:
    return float(profile.dP0_kPa + profile.dP_flow_k * (flow_kg_s ** 2) + profile.dP_foul_k * (max(m_total, 0.0) ** profile.dP_foul_pow))

def vibration_proxy_mm_s(
    phase: str,
    flow_kg_s: float,
    dP_kPa: float,
    Rf: float,
    clog_active: bool,
    clog_severity: float,
    profile: AssetProfile,
) -> float:
    if phase == "production":
        base = 0.9 + profile.vibration_factor * (0.10 * flow_kg_s + 0.0038 * dP_kPa + 105.0 * Rf)
        if clog_active:
            base += 1.7 + 3.2 * clog_severity
    elif phase in ("CIP_alkaline", "CIP_acid"):
        base = 0.55 + 0.08 * flow_kg_s + 0.0016 * dP_kPa
    elif phase == "maintenance":
        base = 0.22 + 0.01 * dP_kPa
    else:
        base = 0.14
    return float(clamp(base, 0.03, 20.0))

def shift_id_from_hour(hour: int) -> int:
    if 6 <= hour < 14:
        return 1
    if 14 <= hour < 22:
        return 2
    return 3

def stage_from_rf(Rf: float, phase: str, thr_incipient: float, thr_advanced: float) -> int:
    if phase != "production":
        return -1
    if Rf >= thr_advanced:
        return 2
    if Rf >= thr_incipient:
        return 1
    return 0

def severity_score_from_rf(Rf: float, thr_advanced: float) -> float:
    return float(clamp(Rf / max(2.0 * thr_advanced, 1e-9), 0.0, 1.0))

def precompute_ambient(rng: np.random.Generator, start: datetime, total_steps: int, dt_s: int) -> Tuple[np.ndarray, np.ndarray]:
    T = np.zeros(total_steps, dtype=float)
    RH = np.zeros(total_steps, dtype=float)
    for i in range(total_steps):
        t = start + timedelta(seconds=i * dt_s)
        hour = t.hour + t.minute / 60.0
        day = (t - start).days
        daily_T = 3.0 * math.sin(2.0 * math.pi * hour / 24.0)
        weekly_T = 1.0 * math.sin(2.0 * math.pi * day / 7.0)
        T[i] = clamp(20.0 + daily_T + weekly_T + rng.normal(0.0, 0.35), 12.0, 30.0)
        daily_RH = 12.0 * math.sin(2.0 * math.pi * (hour + 5.0) / 24.0)
        RH[i] = clamp(55.0 + daily_RH + rng.normal(0.0, 1.8), 30.0, 85.0)
    return T, RH

def ambient_from_time(rng: np.random.Generator, start: datetime, t: datetime) -> Tuple[float, float]:
    hour = t.hour + t.minute / 60.0 + t.second / 3600.0
    day = (t - start).days
    daily_T = 3.0 * math.sin(2.0 * math.pi * hour / 24.0)
    weekly_T = 1.0 * math.sin(2.0 * math.pi * day / 7.0)
    T = clamp(20.0 + daily_T + weekly_T + rng.normal(0.0, 0.35), 12.0, 30.0)
    daily_RH = 12.0 * math.sin(2.0 * math.pi * (hour + 5.0) / 24.0)
    RH = clamp(55.0 + daily_RH + rng.normal(0.0, 1.8), 30.0, 85.0)
    return float(T), float(RH)

def sample_duration_s(rng: np.random.Generator, lo_h: float, hi_h: float, dt_s: int) -> int:
    lo_h = max(float(lo_h), dt_s / 3600.0)
    hi_h = max(float(hi_h), lo_h)
    minutes = float(rng.uniform(lo_h * 60.0, hi_h * 60.0))
    steps = max(1, int(round((minutes * 60.0) / max(dt_s, 1))))
    return int(steps * dt_s)

def sample_asset_profile(rng: np.random.Generator, asset_id: str) -> AssetProfile:
    family = str(rng.choice(ASSET_FAMILIES, p=[0.45, 0.30, 0.25]))
    cfg = {
        "standard_phe": dict(UA=2500.0, flow=5.1, hot=5.3, Th=82.0, Tc=48.0, dP0=25.0, dPflow=1.20, dPfoul=820.0, pow=1.22, foul=1.00, clog=1.00, clean=1.00, vib=1.00),
        "robust_phe": dict(UA=2750.0, flow=5.8, hot=5.9, Th=80.0, Tc=47.0, dP0=22.0, dPflow=1.05, dPfoul=720.0, pow=1.18, foul=0.85, clog=0.85, clean=1.12, vib=0.95),
        "compact_phe": dict(UA=2250.0, flow=4.2, hot=4.5, Th=84.0, Tc=49.0, dP0=28.0, dPflow=1.45, dPfoul=960.0, pow=1.28, foul=1.15, clog=1.18, clean=0.92, vib=1.07),
    }[family]
    return AssetProfile(
        asset_id=asset_id,
        family=family,
        UA_clean_WK=float(cfg["UA"] * rng.uniform(0.94, 1.06)),
        flow_sp_base=float(cfg["flow"] * rng.uniform(0.90, 1.10)),
        hot_loop_sp_base=float(cfg["hot"] * rng.uniform(0.90, 1.10)),
        Th_sp_base=float(cfg["Th"] + rng.normal(0.0, 1.8)),
        Tc_sp_base=float(cfg["Tc"] + rng.normal(0.0, 1.5)),
        dP0_kPa=float(cfg["dP0"] * rng.uniform(0.92, 1.08)),
        dP_flow_k=float(cfg["dPflow"] * rng.uniform(0.93, 1.08)),
        dP_foul_k=float(cfg["dPfoul"] * rng.uniform(0.90, 1.10)),
        dP_foul_pow=float(cfg["pow"]),
        fouling_sensitivity=float(cfg["foul"] * rng.uniform(0.92, 1.10)),
        clog_sensitivity=float(cfg["clog"] * rng.uniform(0.92, 1.12)),
        cleaning_efficiency=float(cfg["clean"] * rng.uniform(0.92, 1.08)),
        vibration_factor=float(cfg["vib"] * rng.uniform(0.95, 1.08)),
    )

def sample_episode_plan(
    rng: np.random.Generator,
    asset: AssetProfile,
    equip: EquipConsts,
    cfg: GeneratorConfig,
    asset_id: str,
    cycle_idx: int,
    current_m_org: float,
    current_m_min: float,
    fouling_anchor_rf: float,
    rf_stage_thr_incipient: float,
    rf_stage_thr_advanced: float,
) -> EpisodePlan:
    milk_type = str(rng.choice(MILK_TYPES, p=[0.33, 0.27, 0.22, 0.18]))
    spec = MILK_SPECS[milk_type]
    protein = float(rng.uniform(*spec.protein_g_L))
    fat = float(rng.uniform(*spec.fat_g_L))
    solids = float(rng.uniform(*spec.solids_g_L))
    Ca = float(rng.uniform(*spec.Ca_mM))
    PO4 = float(rng.uniform(*spec.PO4_mM))
    pH = float(rng.uniform(*spec.pH))
    thermal = float(rng.uniform(0.90, 1.25))

    flow_sp = float(clamp(asset.flow_sp_base * rng.uniform(0.90, 1.12), 2.0, 9.0))
    hot_loop_sp = float(clamp(asset.hot_loop_sp_base * rng.uniform(0.92, 1.12), 2.0, 9.0))
    Th_sp = float(clamp(asset.Th_sp_base + rng.normal(0.0, 2.0), 70.0, 92.0))
    Tc_sp = float(clamp(asset.Tc_sp_base + rng.normal(0.0, 1.7), 35.0, 65.0))

    stable_s = sample_duration_s(rng, cfg.stable_hours_min, cfg.stable_hours_max, cfg.dt)
    incipient_s = sample_duration_s(rng, cfg.incipient_hours_min, cfg.incipient_hours_max, cfg.dt)
    advanced_s = sample_duration_s(rng, cfg.advanced_hours_min, cfg.advanced_hours_max, cfg.dt)

    severity = asset.fouling_sensitivity * spec.fouling_factor * thermal * (Th_sp / 80.0) ** 1.08 * (5.3 / max(flow_sp, 1e-6)) ** 0.20 * rng.uniform(0.92, 1.08)
    severity = float(clamp(severity, 0.70, 1.85))
    _, start_Rf = UA_from_deposit(current_m_org, current_m_min, asset.UA_clean_WK, equip)

    stable_hi = max(rf_stage_thr_incipient * 0.90, 0.55 * fouling_anchor_rf)
    stable_end_Rf = float(clamp(max(start_Rf * 1.03, start_Rf + rng.uniform(0.08, 0.28) * fouling_anchor_rf * severity), 0.04 * fouling_anchor_rf, stable_hi))
    incipient_hi = max(rf_stage_thr_advanced * 0.96, rf_stage_thr_incipient * 1.08)
    incipient_end_Rf = float(clamp(stable_end_Rf + rng.uniform(0.45, 0.90) * fouling_anchor_rf * severity, rf_stage_thr_incipient * 1.02, incipient_hi))
    advanced_end_Rf = float(clamp(incipient_end_Rf + rng.uniform(0.65, 1.55) * fouling_anchor_rf * severity, rf_stage_thr_advanced * 1.10, 3.60 * fouling_anchor_rf))

    stage_targets = {
        "stable": masses_from_Rf(stable_end_Rf, 0.82, equip),
        "incipient": masses_from_Rf(incipient_end_Rf, 0.76, equip),
        "advanced": masses_from_Rf(advanced_end_Rf, 0.68, equip),
    }

    clog_prob = 0.22 + 0.17 * (severity - 0.9) + 0.07 * (milk_type == "high_solids") + 0.10 * (asset.clog_sensitivity - 1.0)
    clog_prob = float(clamp(clog_prob, 0.10, 0.55))
    clog_present = bool(rng.random() < clog_prob)
    clog_duration_s = int(rng.integers(8, 31)) * 60 if clog_present else 0
    if clog_present:
        pad = int(min(25 * 60, max(5 * 60, advanced_s // 3)))
        start_min = stable_s + incipient_s + max(0, advanced_s - clog_duration_s - pad)
        end_min = stable_s + incipient_s + advanced_s
        clog_start_s = int(rng.integers(start_min // 60, max((end_min - clog_duration_s) // 60, start_min // 60) + 1)) * 60
        clog_end_s = min(clog_start_s + clog_duration_s, stable_s + incipient_s + advanced_s)
        clog_drop = float(rng.uniform(0.18, 0.55))
        clog_mult = float(rng.uniform(1.5, 4.2))
    else:
        clog_start_s = -1
        clog_end_s = -1
        clog_drop = 0.0
        clog_mult = 1.0

    mech_clean_trigger = advanced_end_Rf >= max(rf_stage_thr_advanced * 1.6, 2.85e-3) and rng.random() < 0.28
    if clog_present:
        if mech_clean_trigger and rng.random() < 0.30:
            maintenance_type = "mechanical_clean"
            maintenance_s = int(rng.integers(60, 181)) * 60
        else:
            maintenance_type = "unclog"
            maintenance_s = int(rng.integers(10, 46)) * 60
        planned_cip_after = True
    else:
        if mech_clean_trigger:
            maintenance_type = "mechanical_clean"
            maintenance_s = int(rng.integers(70, 211)) * 60
            planned_cip_after = True
        elif severity > 1.20 and rng.random() < 0.35:
            maintenance_type = "CIP_extra"
            maintenance_s = int(rng.integers(25, 61)) * 60
            planned_cip_after = False
        else:
            maintenance_type = "scheduled_CIP"
            maintenance_s = 0
            planned_cip_after = True

    cip_alk_s = int(rng.integers(15, 31)) * 60 if planned_cip_after else 0
    cip_acid_s = int(rng.integers(10, 26)) * 60 if planned_cip_after else 0
    idle_s = int(rng.integers(8, 31)) * 60

    return EpisodePlan(
        episode_id=f"{asset_id}_C{cycle_idx:05d}",
        batch_id=f"{asset_id}_B{cycle_idx:05d}",
        milk_type=milk_type,
        protein_g_L=protein,
        fat_g_L=fat,
        solids_g_L=solids,
        Ca_mM=Ca,
        PO4_mM=PO4,
        pH=pH,
        thermal_history_factor=thermal,
        flow_sp_kg_s=flow_sp,
        hot_loop_sp_kg_s=hot_loop_sp,
        Th_sp_C=Th_sp,
        Tc_sp_C=Tc_sp,
        stable_s=stable_s,
        incipient_s=incipient_s,
        advanced_s=advanced_s,
        maintenance_type=maintenance_type,
        maintenance_s=maintenance_s,
        planned_cip_after=planned_cip_after,
        cip_alk_s=cip_alk_s,
        cip_acid_s=cip_acid_s,
        idle_s=idle_s,
        clog_present=clog_present,
        clog_start_s=clog_start_s,
        clog_end_s=clog_end_s,
        clog_flow_drop_frac=clog_drop,
        clog_dP_mult=clog_mult,
        severity_index=severity,
        stage_targets=stage_targets,
        planned_cip_strength=float(rng.uniform(0.95, 1.35) * asset.cleaning_efficiency),
        unplanned_cip_strength=float(rng.uniform(1.15, 1.70) * asset.cleaning_efficiency),
    )

def apply_sensor_drift(drift: DriftState, dt_s: int, rng: np.random.Generator) -> None:
    h = dt_s / 3600.0
    drift.Th_in += float(rng.normal(0.0, 0.02 * math.sqrt(h)))
    drift.Tc_in += float(rng.normal(0.0, 0.02 * math.sqrt(h)))
    drift.Th_out += float(rng.normal(0.0, 0.02 * math.sqrt(h)))
    drift.Tc_out += float(rng.normal(0.0, 0.02 * math.sqrt(h)))
    drift.flow += float(rng.normal(0.0, 0.01 * math.sqrt(h)))
    drift.P_in += float(rng.normal(0.0, 0.20 * math.sqrt(h)))
    drift.P_out += float(rng.normal(0.0, 0.20 * math.sqrt(h)))
    drift.vib += float(rng.normal(0.0, 0.015 * math.sqrt(h)))

def add_sensor_noise(
    flow: float,
    Th_in: float,
    Tc_in: float,
    Th_out: float,
    Tc_out: float,
    P_in: float,
    P_out: float,
    vib: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float, float, float, float, float]:
    flow += float(rng.normal(0.0, 0.03))
    Th_in += float(rng.normal(0.0, 0.20))
    Tc_in += float(rng.normal(0.0, 0.20))
    Th_out += float(rng.normal(0.0, 0.20))
    Tc_out += float(rng.normal(0.0, 0.20))
    P_in += float(rng.normal(0.0, 0.80))
    P_out += float(rng.normal(0.0, 0.80))
    vib += float(rng.normal(0.0, 0.12))
    return flow, Th_in, Tc_in, Th_out, Tc_out, P_in, P_out, vib

def cip_step(dt_s: int, m_org: float, m_min: float, mode: str, strength: float, cleaning: CleaningParams) -> Tuple[float, float]:
    if mode == "alkaline":
        k_org = cleaning.cip_k_org_alk * strength
        k_min = cleaning.cip_k_min_alk * strength
    else:
        k_org = cleaning.cip_k_org_acid * strength
        k_min = cleaning.cip_k_min_acid * strength
    return (
        max(float(m_org) * math.exp(-k_org * dt_s), 0.0),
        max(float(m_min) * math.exp(-k_min * dt_s), 0.0),
    )

def build_maintenance_row(
    maintenance_id: int,
    asset_id: str,
    start_time: datetime,
    end_time: datetime,
    planned: int,
    fault_type: str,
    maintenance_type: str,
    corrective_action: str,
    severity_rf: float,
    notes: str,
) -> Dict[str, object]:
    return {
        "maintenance_id": f"M{maintenance_id:06d}",
        "asset_id": asset_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_min": round((end_time - start_time).total_seconds() / 60.0, 6),
        "planned": int(planned),
        "fault_type": fault_type,
        "maintenance_type": maintenance_type,
        "corrective_action": corrective_action,
        "severity_rf_at_start": round(float(severity_rf), 10),
        "notes": notes,
    }

def generate_asset(
    asset_idx: int,
    cfg: GeneratorConfig,
    equip: EquipConsts,
    cleaning: CleaningParams,
    start: datetime,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed + 1009 * asset_idx)
    thr_incipient, thr_advanced = cfg.resolved_stage_thresholds()
    asset_id = f"asset_{asset_idx:02d}"
    profile = sample_asset_profile(rng, asset_id)

    t = start + timedelta(minutes=int(rng.integers(0, 6 * 60 + 1)))
    global_step = 0
    cycle_idx = 0
    batch_elapsed_min = 0.0
    m_org, m_min = masses_from_Rf(rng.uniform(0.03, 0.08) * cfg.fouling_anchor_rf, 0.84, equip)
    Twall_state = float(profile.Th_sp_base - 3.0)
    drift = DriftState()
    prev_prod_stage = stage_from_rf(UA_from_deposit(m_org, m_min, profile.UA_clean_WK, equip)[1], "production", thr_incipient, thr_advanced)
    maintenance_counter = 0
    ambient_now_T = 20.0
    ambient_now_RH = 55.0

    rows: List[Dict[str, object]] = []
    maint_rows: List[Dict[str, object]] = []

    def advance_time(phase: str) -> None:
        nonlocal t, global_step, batch_elapsed_min
        t += timedelta(seconds=cfg.dt)
        global_step += 1
        if phase == "production":
            batch_elapsed_min += cfg.dt / 60.0

    def emit_row(
        *,
        cycle_id: str,
        cycle_index: int,
        episode: EpisodePlan,
        phase: str,
        cip_subphase: str,
        flow_true: float,
        pressure_in_true: float,
        pressure_out_true: float,
        Th_in_true: float,
        Tc_in_true: float,
        Th_out_true: float,
        Tc_out_true: float,
        vibration_true: float,
        UA: float,
        Rf: float,
        Q: float,
        clog_event: int,
        clog_onset_event: int,
        maintenance_active: int,
        maintenance_type: str,
        fault_type: str,
        downtime_unplanned: int,
        episode_progress: float,
        stage_progress: float,
        planned_event_type: str,
        planned_flag: int,
    ) -> None:
        nonlocal Twall_state, prev_prod_stage, ambient_now_T, ambient_now_RH

        apply_sensor_drift(drift, cfg.dt, rng)
        flow_meas = flow_true + drift.flow
        Th_in_meas = Th_in_true + drift.Th_in
        Tc_in_meas = Tc_in_true + drift.Tc_in
        Th_out_meas = Th_out_true + drift.Th_out
        Tc_out_meas = Tc_out_true + drift.Tc_out
        P_in_meas = pressure_in_true + drift.P_in
        P_out_meas = pressure_out_true + drift.P_out
        vib_meas = vibration_true + drift.vib
        if not cfg.no_noise:
            flow_meas, Th_in_meas, Tc_in_meas, Th_out_meas, Tc_out_meas, P_in_meas, P_out_meas, vib_meas = add_sensor_noise(
                flow_meas, Th_in_meas, Tc_in_meas, Th_out_meas, Tc_out_meas, P_in_meas, P_out_meas, vib_meas, rng
            )
        flow_meas = max(flow_meas, 0.0)
        vib_meas = max(vib_meas, 0.0)
        Th_out_meas = min(Th_out_meas, Th_in_meas)
        Tc_out_meas = max(Tc_out_meas, Tc_in_meas)
        dP_meas = max(P_in_meas - P_out_meas, 0.0)

        stage_id = stage_from_rf(Rf, phase, thr_incipient, thr_advanced)
        fouling_onset_event = 0
        if phase == "production" and prev_prod_stage == 0 and stage_id >= 1:
            fouling_onset_event = 1
        if phase == "production":
            prev_prod_stage = stage_id

        row = {
            "timestamp": t.isoformat(),
            "asset_id": asset_id,
            "asset_family": profile.family,
            "sequence_id": cycle_id,
            "cycle_id": cycle_id,
            "cycle_index": int(cycle_index),
            "phase": phase,
            "cip_subphase": cip_subphase,
            "shift_id": shift_id_from_hour(t.hour),
            "episode_id": episode.episode_id,
            "episode_progress": round(float(clamp(episode_progress, 0.0, 1.0)), 6),
            "stage_progress": round(float(clamp(stage_progress, 0.0, 1.0)), 6),
            "batch_id": episode.batch_id,
            "batch_elapsed_min": round(float(batch_elapsed_min), 6),
            "milk_type": episode.milk_type,
            "batch_thermal_history_factor": round(float(episode.thermal_history_factor), 6),
            "protein_g_L_nominal": round(float(episode.protein_g_L), 6),
            "fat_g_L_nominal": round(float(episode.fat_g_L), 6),
            "solids_g_L_nominal": round(float(episode.solids_g_L), 6),
            "Ca_mM_nominal": round(float(episode.Ca_mM), 6),
            "PO4_mM_nominal": round(float(episode.PO4_mM), 6),
            "pH_nominal": round(float(episode.pH), 6),
            "ambient_T_C": round(float(ambient_now_T), 6),
            "ambient_RH_pct": round(float(ambient_now_RH), 6),
            "flow_kg_s": round(float(flow_meas), 6),
            "pressure_in_kPa": round(float(P_in_meas), 6),
            "pressure_out_kPa": round(float(P_out_meas), 6),
            "dP_kPa": round(float(dP_meas), 6),
            "Th_in_C": round(float(Th_in_meas), 6),
            "Tc_in_C": round(float(Tc_in_meas), 6),
            "Th_out_C": round(float(Th_out_meas), 6),
            "Tc_out_C": round(float(Tc_out_meas), 6),
            "Twall_C": round(float(Twall_state), 6),
            "vibration_mm_s": round(float(vib_meas), 6),
            "flow_sp_kg_s": round(float(episode.flow_sp_kg_s), 6),
            "Th_sp_C": round(float(episode.Th_sp_C), 6),
            "Tc_sp_C": round(float(episode.Tc_sp_C), 6),
            "m_org_kg_m2": round(float(m_org), 10),
            "m_min_kg_m2": round(float(m_min), 10),
            "m_total_kg_m2": round(float(m_org + m_min), 10),
            "Rf_m2K_W": round(float(Rf), 10),
            "UA_W_K": round(float(UA), 6),
            "Q_W": round(float(Q), 6),
            "rf_stage_thr_incipient": round(float(thr_incipient), 10),
            "rf_stage_thr_advanced": round(float(thr_advanced), 10),
            "fouling_stage": int(stage_id),
            "fouling_stage_name": STAGE_ID_TO_NAME[int(stage_id)],
            "fouling_onset_event": int(fouling_onset_event),
            "fouling_alarm": int(phase == "production" and stage_id >= 1),
            "fouling_warning_alarm": int(phase == "production" and stage_id == 1),
            "fouling_critical_alarm": int(phase == "production" and stage_id == 2),
            "foul_score": round(float(severity_score_from_rf(Rf, thr_advanced)), 6),
            "clog_event": int(clog_event),
            "clog_onset_event": int(clog_onset_event),
            "clog_alarm": int(clog_event),
            "maintenance_active": int(maintenance_active),
            "maintenance_planned": int(planned_flag),
            "maintenance_type": maintenance_type,
            "fault_type": fault_type,
            "downtime_unplanned": int(downtime_unplanned),
            "planned_event_type": planned_event_type,
        }
        rows.append(row)
        advance_time(phase)

    for cycle_idx in range(1, cfg.cycles_per_asset + 1):
        cycle_id = f"{asset_id}_C{cycle_idx:05d}"
        batch_elapsed_min = 0.0
        episode = sample_episode_plan(
            rng=rng,
            asset=profile,
            equip=equip,
            cfg=cfg,
            asset_id=asset_id,
            cycle_idx=cycle_idx,
            current_m_org=m_org,
            current_m_min=m_min,
            fouling_anchor_rf=cfg.fouling_anchor_rf,
            rf_stage_thr_incipient=thr_incipient,
            rf_stage_thr_advanced=thr_advanced,
        )
        episode_observed_total_s = episode.observed_s(cfg.emit_idle_phase)
        stage_start = {
            "stable": (m_org, m_min),
            "incipient": episode.stage_targets["stable"],
            "advanced": episode.stage_targets["incipient"],
        }
        stage_end = {
            "stable": episode.stage_targets["stable"],
            "incipient": episode.stage_targets["incipient"],
            "advanced": episode.stage_targets["advanced"],
        }
        production_cursor_s = 0

        for stage_name in PROD_STAGE_NAMES:
            duration_s = getattr(episode, f"{stage_name}_s")
            n_steps = max(1, duration_s // cfg.dt)
            start_org, start_min = stage_start[stage_name]
            end_org, end_min = stage_end[stage_name]
            for idx in range(n_steps):
                production_cursor_s += cfg.dt
                frac = smoothstep01((idx + 1) / n_steps)
                m_org = float(lerp(start_org, end_org, frac))
                m_min = float(lerp(start_min, end_min, frac))
                UA, Rf = UA_from_deposit(m_org, m_min, profile.UA_clean_WK, equip)
                ambient_now_T, ambient_now_RH = ambient_from_time(rng, start, t)

                wave = math.sin(2.0 * math.pi * idx / max(n_steps, 1))
                flow_nom = episode.flow_sp_kg_s * (1.0 + 0.03 * wave + 0.015 * rng.normal())
                hot_loop_nom = episode.hot_loop_sp_kg_s * (
                    1.0 + 0.02 * math.cos(2.0 * math.pi * idx / max(n_steps, 1)) + 0.010 * rng.normal()
                )
                fouling_drag = 0.015 * stage_from_rf(Rf, "production", thr_incipient, thr_advanced) + 0.050 * (Rf / max(2.6 * cfg.fouling_anchor_rf, 1e-9))
                flow_true = max(flow_nom * (1.0 - fouling_drag), 0.6)

                clog_event = 0
                clog_onset_event = 0
                if episode.clog_present and episode.clog_start_s <= (production_cursor_s - cfg.dt) < episode.clog_end_s:
                    clog_event = 1
                    if production_cursor_s - cfg.dt == episode.clog_start_s:
                        clog_onset_event = 1
                    flow_true *= (1.0 - episode.clog_flow_drop_frac)

                Th_in_true = clamp(
                    episode.Th_sp_C
                    + 0.9 * math.sin(2.0 * math.pi * production_cursor_s / max(episode.production_s, 1))
                    + rng.normal(0.0, 0.12),
                    65.0,
                    95.0,
                )
                Tc_in_true = clamp(
                    episode.Tc_sp_C
                    + 0.5 * math.cos(2.0 * math.pi * production_cursor_s / max(episode.production_s, 1))
                    + 0.12 * (ambient_now_T - 20.0)
                    + rng.normal(0.0, 0.10),
                    30.0,
                    68.0,
                )
                Th_out_true, Tc_out_true, Q, _ = hx_step(
                    Th_in_true,
                    Tc_in_true,
                    max(hot_loop_nom, 0.5),
                    max(flow_true, 0.5),
                    UA,
                    episode.protein_g_L,
                    episode.fat_g_L,
                    episode.solids_g_L,
                )
                Twall_target = 0.56 * Th_in_true + 0.44 * Th_out_true + 0.45 * (episode.thermal_history_factor - 1.0)
                Twall_state = 0.88 * Twall_state + 0.12 * Twall_target
                dP = dP_proxy_kPa(m_org + m_min, flow_true, profile)
                if clog_event:
                    dP *= episode.clog_dP_mult
                pressure_out_true = 145.0 + 5.0 * math.sin(global_step / 1800.0) + rng.normal(0.0, 0.20)
                pressure_in_true = pressure_out_true + dP
                clog_sev = episode.clog_flow_drop_frac * episode.clog_dP_mult if clog_event else 0.0
                vibration_true = vibration_proxy_mm_s("production", flow_true, dP, Rf, bool(clog_event), clog_sev, profile)

                emit_row(
                    cycle_id=cycle_id,
                    cycle_index=cycle_idx,
                    episode=episode,
                    phase="production",
                    cip_subphase="none",
                    flow_true=flow_true,
                    pressure_in_true=pressure_in_true,
                    pressure_out_true=pressure_out_true,
                    Th_in_true=Th_in_true,
                    Tc_in_true=Tc_in_true,
                    Th_out_true=Th_out_true,
                    Tc_out_true=Tc_out_true,
                    vibration_true=vibration_true,
                    UA=UA,
                    Rf=Rf,
                    Q=Q,
                    clog_event=clog_event,
                    clog_onset_event=clog_onset_event,
                    maintenance_active=0,
                    maintenance_type="none",
                    fault_type="clogging" if clog_event else ("fouling" if stage_from_rf(Rf, "production", thr_incipient, thr_advanced) >= 1 else "none"),
                    downtime_unplanned=0,
                    episode_progress=production_cursor_s / max(episode_observed_total_s, 1),
                    stage_progress=(idx + 1) / n_steps,
                    planned_event_type="none",
                    planned_flag=0,
                )

        if episode.maintenance_type != "scheduled_CIP" and episode.maintenance_s > 0:
            maintenance_counter += 1
            maintenance_start = t
            severity_rf_start = UA_from_deposit(m_org, m_min, profile.UA_clean_WK, equip)[1]
            maint_steps = max(1, episode.maintenance_s // cfg.dt)
            for idx in range(maint_steps):
                ambient_now_T, ambient_now_RH = ambient_from_time(rng, start, t)
                frac = (idx + 1) / maint_steps
                if episode.maintenance_type == "CIP_extra":
                    cip_subphase = "alkaline" if frac <= 0.55 else "acid"
                    mode = "alkaline" if frac <= 0.55 else "acid"
                    m_org, m_min = cip_step(cfg.dt, m_org, m_min, mode, episode.unplanned_cip_strength, cleaning)
                    flow_true = max(episode.flow_sp_kg_s * 0.65 + rng.normal(0.0, 0.06), 0.4)
                    hot_loop_true = max(episode.hot_loop_sp_kg_s * 0.62 + rng.normal(0.0, 0.05), 0.4)
                    Th_in_true = clamp(52.0 + 7.0 * math.sin(frac * math.pi) + rng.normal(0.0, 0.18), 40.0, 68.0)
                    Tc_in_true = clamp(31.0 + rng.normal(0.0, 0.12), 20.0, 48.0)
                    fault_type = "fouling"
                elif episode.maintenance_type == "mechanical_clean":
                    cip_subphase = "none"
                    flow_true = 0.0
                    hot_loop_true = 0.0
                    Th_in_true = ambient_now_T + rng.uniform(2.0, 6.0)
                    Tc_in_true = ambient_now_T + rng.uniform(0.2, 3.0)
                    fault_type = "fouling"
                    target_org = m_org * 0.10
                    target_min = m_min * 0.08
                    m_org = lerp(m_org, target_org, 0.18 * smoothstep01(frac))
                    m_min = lerp(m_min, target_min, 0.20 * smoothstep01(frac))
                else:
                    cip_subphase = "none"
                    flow_true = 0.0
                    hot_loop_true = 0.0
                    Th_in_true = ambient_now_T + rng.uniform(2.0, 6.0)
                    Tc_in_true = ambient_now_T + rng.uniform(0.2, 3.0)
                    fault_type = "clogging" if episode.maintenance_type == "unclog" else "fouling"
                    m_org = lerp(m_org, m_org * 0.94, 0.25 * smoothstep01(frac))
                    m_min = lerp(m_min, m_min * 0.96, 0.18 * smoothstep01(frac))
                UA, Rf = UA_from_deposit(m_org, m_min, profile.UA_clean_WK, equip)
                Th_out_true, Tc_out_true, Q, _ = hx_step(
                    Th_in_true, Tc_in_true, max(hot_loop_true, 0.2), max(flow_true, 0.2), UA, episode.protein_g_L, episode.fat_g_L, episode.solids_g_L
                )
                Twall_target = 0.50 * Th_in_true + 0.50 * Th_out_true
                Twall_state = 0.90 * Twall_state + 0.10 * Twall_target
                dP = max(4.0 + 0.35 * (flow_true ** 2) + 120.0 * Rf, 0.0)
                pressure_out_true = 143.0 + 3.0 * math.sin(global_step / 1700.0) + rng.normal(0.0, 0.18)
                pressure_in_true = pressure_out_true + dP
                vibration_true = vibration_proxy_mm_s("maintenance", flow_true, dP, Rf, False, 0.0, profile)

                emit_row(
                    cycle_id=cycle_id,
                    cycle_index=cycle_idx,
                    episode=episode,
                    phase="maintenance",
                    cip_subphase=cip_subphase,
                    flow_true=flow_true,
                    pressure_in_true=pressure_in_true,
                    pressure_out_true=pressure_out_true,
                    Th_in_true=Th_in_true,
                    Tc_in_true=Tc_in_true,
                    Th_out_true=Th_out_true,
                    Tc_out_true=Tc_out_true,
                    vibration_true=vibration_true,
                    UA=UA,
                    Rf=Rf,
                    Q=Q,
                    clog_event=0,
                    clog_onset_event=0,
                    maintenance_active=1,
                    maintenance_type=episode.maintenance_type,
                    fault_type=fault_type,
                    downtime_unplanned=1,
                    episode_progress=(episode.production_s + (idx + 1) * cfg.dt) / max(episode_observed_total_s, 1),
                    stage_progress=(idx + 1) / maint_steps,
                    planned_event_type="none",
                    planned_flag=0,
                )

            maint_row = build_maintenance_row(
                maintenance_id=maintenance_counter,
                asset_id=asset_id,
                start_time=maintenance_start,
                end_time=t,
                planned=0,
                fault_type="clogging" if episode.maintenance_type == "unclog" else "fouling",
                maintenance_type=episode.maintenance_type,
                corrective_action={"unclog": "flush/unblock", "mechanical_clean": "open & manual clean", "CIP_extra": "unscheduled CIP"}[episode.maintenance_type],
                severity_rf=severity_rf_start,
                notes=f"synthetic unplanned intervention for {episode.episode_id}",
            )
            maint_row["cycle_id"] = cycle_id
            maint_row["cycle_index"] = int(cycle_idx)
            maint_rows.append(maint_row)

        if episode.planned_cip_after and (episode.cip_alk_s > 0 or episode.cip_acid_s > 0):
            planned_start = t
            if not cfg.no_cip_logs:
                maintenance_counter += 1
                planned_id = maintenance_counter
            else:
                planned_id = -1
            for mode, duration_s, phase_name in (("alkaline", episode.cip_alk_s, "CIP_alkaline"), ("acid", episode.cip_acid_s, "CIP_acid")):
                n_steps = max(1, duration_s // cfg.dt) if duration_s > 0 else 0
                for idx in range(n_steps):
                    ambient_now_T, ambient_now_RH = ambient_from_time(rng, start, t)
                    m_org, m_min = cip_step(cfg.dt, m_org, m_min, mode, episode.planned_cip_strength, cleaning)
                    UA, Rf = UA_from_deposit(m_org, m_min, profile.UA_clean_WK, equip)
                    flow_true = max(episode.flow_sp_kg_s * 0.72 + rng.normal(0.0, 0.06), 0.5)
                    hot_loop_true = max(episode.hot_loop_sp_kg_s * 0.70 + rng.normal(0.0, 0.06), 0.5)
                    if mode == "alkaline":
                        Th_in_true = clamp(55.0 + 7.0 * math.sin((idx + 1) / max(n_steps, 1) * math.pi) + rng.normal(0.0, 0.16), 40.0, 70.0)
                        Tc_in_true = clamp(32.0 + rng.normal(0.0, 0.12), 20.0, 50.0)
                    else:
                        Th_in_true = clamp(50.0 + 6.0 * math.cos((idx + 1) / max(n_steps, 1) * math.pi) + rng.normal(0.0, 0.16), 35.0, 65.0)
                        Tc_in_true = clamp(30.0 + rng.normal(0.0, 0.12), 20.0, 46.0)
                    Th_out_true, Tc_out_true, Q, _ = hx_step(Th_in_true, Tc_in_true, max(hot_loop_true, 0.2), max(flow_true, 0.2), UA, 32.0, 15.0, 115.0)
                    Twall_target = 0.50 * Th_in_true + 0.50 * Th_out_true
                    Twall_state = 0.90 * Twall_state + 0.10 * Twall_target
                    dP = max(8.0 + 0.5 * (flow_true ** 2) + 90.0 * Rf, 0.0)
                    pressure_out_true = 144.0 + 2.5 * math.sin(global_step / 2000.0) + rng.normal(0.0, 0.15)
                    pressure_in_true = pressure_out_true + dP
                    vibration_true = vibration_proxy_mm_s(phase_name, flow_true, dP, Rf, False, 0.0, profile)

                    emit_row(
                        cycle_id=cycle_id,
                        cycle_index=cycle_idx,
                        episode=episode,
                        phase=phase_name,
                        cip_subphase=mode,
                        flow_true=flow_true,
                        pressure_in_true=pressure_in_true,
                        pressure_out_true=pressure_out_true,
                        Th_in_true=Th_in_true,
                        Tc_in_true=Tc_in_true,
                        Th_out_true=Th_out_true,
                        Tc_out_true=Tc_out_true,
                        vibration_true=vibration_true,
                        UA=UA,
                        Rf=Rf,
                        Q=Q,
                        clog_event=0,
                        clog_onset_event=0,
                        maintenance_active=0,
                        maintenance_type="scheduled_CIP",
                        fault_type="preventive",
                        downtime_unplanned=0,
                        episode_progress=1.0,
                        stage_progress=(idx + 1) / max(n_steps, 1),
                        planned_event_type="scheduled_CIP",
                        planned_flag=1,
                    )
            if not cfg.no_cip_logs and planned_id >= 0:
                maint_row = build_maintenance_row(
                    maintenance_id=planned_id,
                    asset_id=asset_id,
                    start_time=planned_start,
                    end_time=t,
                    planned=1,
                    fault_type="preventive",
                    maintenance_type="scheduled_CIP",
                    corrective_action="CIP_cycle",
                    severity_rf=UA_from_deposit(m_org, m_min, profile.UA_clean_WK, equip)[1],
                    notes=f"scheduled cleaning after {episode.episode_id}",
                )
                maint_row["cycle_id"] = cycle_id
                maint_row["cycle_index"] = int(cycle_idx)
                maint_rows.append(maint_row)

        if cfg.emit_idle_phase and episode.idle_s > 0:
            idle_steps = max(1, episode.idle_s // cfg.dt)
            for idx in range(idle_steps):
                ambient_now_T, ambient_now_RH = ambient_from_time(rng, start, t)
                UA, Rf = UA_from_deposit(m_org, m_min, profile.UA_clean_WK, equip)
                flow_true = max(rng.uniform(0.0, 0.4), 0.0)
                Th_in_true = ambient_now_T + rng.uniform(1.0, 4.0)
                Tc_in_true = ambient_now_T + rng.uniform(0.0, 2.0)
                Th_out_true, Tc_out_true, Q, _ = hx_step(
                    Th_in_true, Tc_in_true, max(rng.uniform(0.0, 0.4), 0.2), max(flow_true, 0.2), UA, 32.0, 15.0, 115.0
                )
                Twall_target = 0.48 * Th_in_true + 0.52 * Th_out_true
                Twall_state = 0.92 * Twall_state + 0.08 * Twall_target
                dP = max(4.5 + 0.20 * (flow_true ** 2) + 60.0 * Rf, 0.0)
                pressure_out_true = 144.0 + 1.0 * math.sin(global_step / 2200.0) + rng.normal(0.0, 0.10)
                pressure_in_true = pressure_out_true + dP
                vibration_true = vibration_proxy_mm_s("idle", flow_true, dP, Rf, False, 0.0, profile)

                emit_row(
                    cycle_id=cycle_id,
                    cycle_index=cycle_idx,
                    episode=episode,
                    phase="idle",
                    cip_subphase="none",
                    flow_true=flow_true,
                    pressure_in_true=pressure_in_true,
                    pressure_out_true=pressure_out_true,
                    Th_in_true=Th_in_true,
                    Tc_in_true=Tc_in_true,
                    Th_out_true=Th_out_true,
                    Tc_out_true=Tc_out_true,
                    vibration_true=vibration_true,
                    UA=UA,
                    Rf=Rf,
                    Q=Q,
                    clog_event=0,
                    clog_onset_event=0,
                    maintenance_active=0,
                    maintenance_type="none",
                    fault_type="none",
                    downtime_unplanned=0,
                    episode_progress=1.0,
                    stage_progress=(idx + 1) / idle_steps,
                    planned_event_type="none",
                    planned_flag=0,
                )
        else:
            t += timedelta(seconds=episode.idle_s)

        if cycle_idx < cfg.cycles_per_asset:
            gap_min = int(rng.integers(cfg.inter_cycle_gap_min, cfg.inter_cycle_gap_max + 1))
            t += timedelta(minutes=gap_min)

        batch_elapsed_min = 0.0
        prev_prod_stage = stage_from_rf(UA_from_deposit(m_org, m_min, profile.UA_clean_WK, equip)[1], "production", thr_incipient, thr_advanced)

    tel_df = pd.DataFrame(rows)
    maint_df = pd.DataFrame(maint_rows)
    return tel_df, maint_df
