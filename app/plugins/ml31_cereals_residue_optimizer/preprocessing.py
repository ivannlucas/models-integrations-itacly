"""Economic model + parameter extraction for the ml31 LP optimizer.

Faithful port of the v2.0 delivered code:
  - PriceCostManager   <- src/economics/price_cost_manager.py
  - ParameterExtractor <- src/data_processing/parameter_extractor.py
  - filter_dominated / find_knee_point <- scripts/run_pareto_study.py

The only adaptation is that PriceCostManager receives already-loaded dicts (the
plugin loads the JSON via ArtifactStore) instead of re-reading files on each
request. All numeric logic (formulas, signs, coefficients) is unchanged so the
LP reproduces the delivered results exactly.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PriceCostManager (port of src/economics/price_cost_manager.py)
# ---------------------------------------------------------------------------
class PriceCostManager:
    """Manages prices, costs, and harvest indices per crop.

    Loads reference data from dicts (crop_economics.json / harvest_index.json)
    and supports user overrides passed through the request.
    """

    def __init__(self, economics: dict, harvest_indices: dict):
        self.economics = economics
        self.harvest_indices = harvest_indices
        self._price_overrides: Dict[str, float] = {}
        self._cost_overrides: Dict[str, Dict[str, float]] = {}

    def get_price(self, crop: str) -> float:
        """Return grain price in EUR/kg for *crop*."""
        if crop in self._price_overrides:
            return self._price_overrides[crop]
        crop_data = self.economics.get("crops", {}).get(crop, {})
        price = crop_data.get("price_grain")
        if price is None:
            raise KeyError(f"No price found for crop '{crop}'")
        return float(price)

    def get_cost(self, crop: str, regime: str) -> float:
        """Return variable cost in EUR/ha for *crop* and *regime* (secano|regadio)."""
        key = f"cost_{regime}"
        overrides = self._cost_overrides.get(crop, {})
        if regime in overrides:
            return float(overrides[regime])
        if key in overrides:
            return float(overrides[key])
        crop_data = self.economics.get("crops", {}).get(crop, {})
        cost = crop_data.get(key)
        if cost is None:
            raise KeyError(f"No cost '{key}' found for crop '{crop}'")
        return float(cost)

    def get_harvest_index(self, crop: str, regime: str, under_stress: bool = False) -> float:
        """Return Harvest Index for *crop*, *regime*, and stress condition."""
        hi_data = self.harvest_indices.get("crops", {}).get(crop, {})
        if regime == "regadio":
            key = "hi_regadio"
        elif under_stress:
            key = "hi_secano_stress"
        else:
            key = "hi_secano"
        value = hi_data.get(key)
        if value is None:
            raise KeyError(f"No harvest index '{key}' found for crop '{crop}'")
        return float(value)

    def override(self, overrides: dict) -> None:
        """Apply user-provided price/cost overrides on top of reference data."""
        price_overrides = overrides.get("price_overrides", {})
        if price_overrides:
            self._price_overrides.update(price_overrides)

        cost_overrides = overrides.get("cost_overrides", {})
        if cost_overrides:
            for crop, costs in cost_overrides.items():
                self._cost_overrides.setdefault(crop, {}).update(costs)


# ---------------------------------------------------------------------------
# ParameterExtractor (port of src/data_processing/parameter_extractor.py)
# ---------------------------------------------------------------------------
class ParameterExtractor:
    """Extracts agronomic parameters from the processed dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract_for_year(self, year: int) -> Dict:
        """Extract LP parameters for a single reference year.

        Yields are the MEAN over the whole dataset; surface/straw come from
        *year*. Raises ValueError if the year is absent.
        """
        year_df = self.df[self.df["Año"] == year]
        if year_df.empty:
            raise ValueError(f"No data found for reference year {year}")

        crops = self.df["Cultivo"].unique().tolist()

        yield_s: Dict[str, float] = (
            self.df.groupby("Cultivo")["Rend_Secano_kg_ha"].mean().to_dict()
        )
        yield_r: Dict[str, float] = (
            self.df.groupby("Cultivo")["Rend_Regadio_kg_ha"].mean().to_dict()
        )

        hist_secano: Dict[str, float] = (
            year_df.groupby("Cultivo")["Sup_Secano_ha"].sum().to_dict()
        )
        hist_regadio: Dict[str, float] = (
            year_df.groupby("Cultivo")["Sup_Regadio_ha"].sum().to_dict()
        )

        paja: Dict[str, float] = {}
        if "Prod_Paja_Cosechada_t" in year_df.columns:
            paja = year_df.groupby("Cultivo")["Prod_Paja_Cosechada_t"].sum().to_dict()

        total_secano_ha = float(year_df["Sup_Secano_ha"].sum())
        total_regadio_ha = float(year_df["Sup_Regadio_ha"].sum())

        spring_rain_mm = float(year_df["Lluvia_Primavera_mm"].iloc[0])

        return {
            "crops": crops,
            "yield_s": yield_s,
            "yield_r": yield_r,
            "total_secano_ha": total_secano_ha,
            "total_regadio_ha": total_regadio_ha,
            "hist_secano": hist_secano,
            "hist_regadio": hist_regadio,
            "paja": paja,
            "spring_rain_mm": spring_rain_mm,
        }

    def compute_harvest_fraction(
        self, pcm: PriceCostManager, max_fraction: float = 0.95
    ) -> Dict[str, float]:
        """Estimate per-crop fraction of generated straw the farmer removes.

        harvest_fraction[c] = sum(Prod_Paja_Cosechada_t)/sum(straw_generated),
        clipped to [0, max_fraction]. Residue left on soil scales with production.
        """
        straw_gen_total: Dict[str, float] = {}
        paja_total: Dict[str, float] = {}

        for _, row in self.df.iterrows():
            crop = row["Cultivo"]
            under_stress = row.get("Lluvia_Primavera_mm", 130) < 100

            prod_s = row.get("Sup_Secano_ha", 0.0) * row.get("Rend_Secano_kg_ha", 0.0) / 1000.0
            prod_r = row.get("Sup_Regadio_ha", 0.0) * row.get("Rend_Regadio_kg_ha", 0.0) / 1000.0

            hi_s = pcm.get_harvest_index(crop, "secano", under_stress)
            hi_r = pcm.get_harvest_index(crop, "regadio", False)
            coeff_s = (1.0 - hi_s) / hi_s if hi_s > 0 else 0.0
            coeff_r = (1.0 - hi_r) / hi_r if hi_r > 0 else 0.0

            straw_gen = prod_s * coeff_s + prod_r * coeff_r
            paja = row.get("Prod_Paja_Cosechada_t", 0.0) or 0.0

            straw_gen_total[crop] = straw_gen_total.get(crop, 0.0) + straw_gen
            paja_total[crop] = paja_total.get(crop, 0.0) + paja

        fractions: Dict[str, float] = {}
        for crop in self.df["Cultivo"].unique():
            gen = straw_gen_total.get(crop, 0.0)
            frac = paja_total.get(crop, 0.0) / gen if gen > 0 else 0.0
            fractions[crop] = float(min(max(frac, 0.0), max_fraction))

        return fractions

    def get_baseline_metrics(
        self,
        year: int,
        params: Dict,
        pcm: PriceCostManager,
        harvest_fraction: Optional[Dict[str, float]] = None,
        climate_factor: float = 1.0,
    ) -> Dict[str, float]:
        """Compute baseline (historical) production, residue, and benefit for *year*.

        Uses the same scalable residue model and economic data as the optimizer,
        with *climate_factor* applied to yields (baseline and optimizer share the
        same climate assumption).
        """
        if harvest_fraction is None:
            harvest_fraction = self.compute_harvest_fraction(pcm)

        total_prod = 0.0
        total_res = 0.0
        total_ben = 0.0
        under_stress = params["spring_rain_mm"] < 100

        for c in params["crops"]:
            xs = params["hist_secano"].get(c, 0.0)
            xr = params["hist_regadio"].get(c, 0.0)
            ys = params["yield_s"].get(c, 0.0)
            yr = params["yield_r"].get(c, 0.0)

            prod_s = xs * ys * climate_factor / 1000.0
            prod_r = xr * yr * climate_factor / 1000.0
            prod = prod_s + prod_r

            hi_s = pcm.get_harvest_index(c, "secano", under_stress)
            hi_r = pcm.get_harvest_index(c, "regadio", False)
            res_coeff_s = (1.0 - hi_s) / hi_s if hi_s > 0 else 0.0
            res_coeff_r = (1.0 - hi_r) / hi_r if hi_r > 0 else 0.0

            straw_gen = prod_s * res_coeff_s + prod_r * res_coeff_r
            res = straw_gen * (1.0 - harvest_fraction.get(c, 0.0))

            price = pcm.get_price(c)
            cost_s = pcm.get_cost(c, "secano")
            cost_r = pcm.get_cost(c, "regadio")
            ben = prod * price * 1000.0 - cost_s * xs - cost_r * xr

            total_prod += prod
            total_res += res
            total_ben += ben

        return {
            "total_production_t": round(total_prod, 2),
            "total_residue_t": round(total_res, 2),
            "total_benefit_eur": round(total_ben, 2),
        }


# ---------------------------------------------------------------------------
# Pareto helpers (port of scripts/run_pareto_study.py)
# ---------------------------------------------------------------------------
def filter_dominated(points: List[dict]) -> List[dict]:
    """Remove dominated points. A dominates B if A.benefit >= B.benefit AND
    A.residue <= B.residue with at least one strict inequality."""
    n = len(points)
    is_dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (points[j]["benefit_eur"] >= points[i]["benefit_eur"] and
                    points[j]["residue_t"] <= points[i]["residue_t"] and
                    (points[j]["benefit_eur"] > points[i]["benefit_eur"] or
                     points[j]["residue_t"] < points[i]["residue_t"])):
                is_dominated[i] = True
                break
    return [p for i, p in enumerate(points) if not is_dominated[i]]


def find_knee_point(front: List[dict]) -> Optional[dict]:
    """Find the knee (elbow) point of the Pareto front (max distance to the line
    connecting the two extremes in normalized benefit-residue space)."""
    if len(front) < 2:
        return front[0] if front else None

    benefits = np.array([p["benefit_eur"] for p in front])
    residues = np.array([p["residue_t"] for p in front])

    b_min, b_max = benefits.min(), benefits.max()
    r_min, r_max = residues.min(), residues.max()

    b_norm = (benefits - b_min) / (b_max - b_min + 1e-12)
    r_norm = (residues - r_min) / (r_max - r_min + 1e-12)

    p0 = np.array([b_norm[0], r_norm[0]])
    p1 = np.array([b_norm[-1], r_norm[-1]])
    line = p1 - p0
    line_len = np.linalg.norm(line)
    if line_len < 1e-12:
        return front[len(front) // 2]

    max_dist = -1.0
    knee_idx = 0
    for i in range(len(front)):
        pt = np.array([b_norm[i], r_norm[i]])
        v = pt - p0
        proj = np.dot(v, line) / (line_len ** 2)
        proj = np.clip(proj, 0, 1)
        closest = p0 + proj * line
        dist = np.linalg.norm(pt - closest)
        if dist > max_dist:
            max_dist = dist
            knee_idx = i

    return front[knee_idx]
