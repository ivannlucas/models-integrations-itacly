"""Linear Programming optimizer for cereal crop allocation (PuLP + CBC).

Faithful port of the v2.0 delivered code (src/optimization/linear_programming.py).
Decision variables: x_s[i] (rainfed ha), x_r[i] (irrigated ha) per crop.
Two modes: A = minimize_residue (with min_benefit), B = maximize_benefit (with
max_residue). The problem is linear and convex, so CBC returns the exact global
optimum deterministically — no random seeds.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pulp

from app.plugins.ml31_cereals_residue_optimizer.preprocessing import PriceCostManager


@dataclass
class LPSolution:
    """Holds the result of a single LP solve."""

    status: str
    solve_time_seconds: float
    crops: List[str]
    secano: Dict[str, float] = field(default_factory=dict)
    regadio: Dict[str, float] = field(default_factory=dict)
    production: Dict[str, float] = field(default_factory=dict)
    residue: Dict[str, float] = field(default_factory=dict)
    benefit: Dict[str, float] = field(default_factory=dict)
    total_production_t: float = 0.0
    total_residue_t: float = 0.0
    total_benefit_eur: float = 0.0
    objective_value: float = 0.0
    raw: Optional[Any] = None


class LPOptimizer:
    """Builds and solves the LP model for crop allocation."""

    def __init__(
        self,
        crops: List[str],
        price_cost_mgr: PriceCostManager,
        total_secano_ha: float,
        total_regadio_ha: float,
        yield_s: Dict[str, float],
        yield_r: Dict[str, float],
        climate_factor: float = 1.0,
        spring_rain_mm: float = 130.0,
        crop_constraints: Optional[Dict[str, Dict[str, float]]] = None,
        harvest_fraction: Optional[Dict[str, float]] = None,
        min_secano_use_ha: float = 0.0,
        min_regadio_use_ha: float = 0.0,
        hist_area: Optional[Dict[str, float]] = None,
        surface_tolerance_pct: Optional[float] = None,
    ):
        self.crops = crops
        self.pcm = price_cost_mgr
        self.S_s = total_secano_ha
        self.S_r = total_regadio_ha
        self.yield_s = yield_s
        self.yield_r = yield_r
        self.f_clima = climate_factor
        self.spring_rain_mm = spring_rain_mm
        self.crop_constraints = crop_constraints or {}
        self.harvest_fraction = harvest_fraction or {c: 0.0 for c in crops}
        self.min_secano_use_ha = min_secano_use_ha
        self.min_regadio_use_ha = min_regadio_use_ha
        self.hist_area = hist_area or {}
        self.surface_tolerance_pct = surface_tolerance_pct

        under_stress = spring_rain_mm < 100
        self.hi_s: Dict[str, float] = {}
        self.hi_r: Dict[str, float] = {}
        for c in crops:
            self.hi_s[c] = price_cost_mgr.get_harvest_index(c, "secano", under_stress)
            self.hi_r[c] = price_cost_mgr.get_harvest_index(c, "regadio", False)

        self.price: Dict[str, float] = {c: price_cost_mgr.get_price(c) for c in crops}
        self.cost_s: Dict[str, float] = {c: price_cost_mgr.get_cost(c, "secano") for c in crops}
        self.cost_r: Dict[str, float] = {c: price_cost_mgr.get_cost(c, "regadio") for c in crops}

        self.res_coeff_s: Dict[str, float] = {c: (1.0 - self.hi_s[c]) / self.hi_s[c] for c in crops}
        self.res_coeff_r: Dict[str, float] = {c: (1.0 - self.hi_r[c]) / self.hi_r[c] for c in crops}

    # ── constraint helpers ──────────────────────────────────────────────────
    def _get_min_production(self, crop: str) -> float:
        default = self.crop_constraints.get("_default", {}).get("min_production_t", 0.0)
        return self.crop_constraints.get(crop, {}).get("min_production_t", default)

    def _get_max_surface_pct(self, crop: str) -> float:
        default = self.crop_constraints.get("_default", {}).get("max_surface_pct", 100.0)
        return self.crop_constraints.get(crop, {}).get("max_surface_pct", default)

    def _get_min_surface_pct(self, crop: str) -> float:
        default = self.crop_constraints.get("_default", {}).get("min_surface_pct", 0.0)
        return self.crop_constraints.get(crop, {}).get("min_surface_pct", default)

    # ── build & solve ────────────────────────────────────────────────────────
    def solve(
        self,
        mode: str = "minimize_residue",
        min_benefit_eur: Optional[float] = None,
        max_residue_t: Optional[float] = None,
    ) -> LPSolution:
        """Build and solve the LP model and return an LPSolution."""
        prob = pulp.LpProblem("CropAllocation", pulp.LpMinimize)

        x_s = {c: pulp.LpVariable(f"x_s_{c}", lowBound=0) for c in self.crops}
        x_r = {c: pulp.LpVariable(f"x_r_{c}", lowBound=0) for c in self.crops}

        prod_s = {c: x_s[c] * self.yield_s.get(c, 0) * self.f_clima / 1000.0 for c in self.crops}
        prod_r = {c: x_r[c] * self.yield_r.get(c, 0) * self.f_clima / 1000.0 for c in self.crops}
        prod = {c: prod_s[c] + prod_r[c] for c in self.crops}

        res_pos = {
            c: (prod_s[c] * self.res_coeff_s[c] + prod_r[c] * self.res_coeff_r[c])
            * (1.0 - self.harvest_fraction.get(c, 0.0))
            for c in self.crops
        }

        benefit = {
            c: prod[c] * self.price[c] * 1000.0
            - self.cost_s[c] * x_s[c]
            - self.cost_r[c] * x_r[c]
            for c in self.crops
        }

        total_prod = pulp.lpSum(prod[c] for c in self.crops)
        total_res = pulp.lpSum(res_pos[c] for c in self.crops)
        total_benefit = pulp.lpSum(benefit[c] for c in self.crops)

        # R1 / R2: available land upper bounds
        prob += pulp.lpSum(x_s[c] for c in self.crops) <= self.S_s, "R1_total_secano"
        prob += pulp.lpSum(x_r[c] for c in self.crops) <= self.S_r, "R2_total_regadio"

        # R1b / R2b: land-use floors
        if self.min_secano_use_ha > 0:
            prob += pulp.lpSum(x_s[c] for c in self.crops) >= self.min_secano_use_ha, "R1b_min_secano_use"
        if self.min_regadio_use_ha > 0:
            prob += pulp.lpSum(x_r[c] for c in self.crops) >= self.min_regadio_use_ha, "R2b_min_regadio_use"

        total_surface = self.S_s + self.S_r

        for c in self.crops:
            min_prod = self._get_min_production(c)
            if min_prod > 0:
                prob += prod[c] >= min_prod, f"R3_min_prod_{c}"

            max_pct = self._get_max_surface_pct(c)
            if max_pct < 100.0:
                prob += (x_s[c] + x_r[c]) <= max_pct / 100.0 * total_surface, f"R4_max_pct_{c}"

            min_pct = self._get_min_surface_pct(c)
            if min_pct > 0.0:
                prob += (x_s[c] + x_r[c]) >= min_pct / 100.0 * total_surface, f"R4b_min_pct_{c}"

            if self.surface_tolerance_pct is not None and c in self.hist_area:
                tol = self.surface_tolerance_pct / 100.0
                h = self.hist_area[c]
                prob += (x_s[c] + x_r[c]) <= h * (1.0 + tol), f"R4c_hi_band_{c}"
                prob += (x_s[c] + x_r[c]) >= h * (1.0 - tol), f"R4c_lo_band_{c}"

        if mode == "minimize_residue":
            prob += total_res, "Obj_minimize_residue"
            if min_benefit_eur is not None:
                prob += total_benefit >= min_benefit_eur, "R6_min_benefit"
        elif mode == "maximize_benefit":
            prob += -total_benefit, "Obj_maximize_benefit"
            if max_residue_t is not None:
                prob += total_res <= max_residue_t, "R7_max_residue"
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'minimize_residue' or 'maximize_benefit'."
            )

        t0 = time.perf_counter()
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        elapsed = time.perf_counter() - t0

        status_map = {
            "Optimal": "OPTIMAL",
            "Infeasible": "INFEASIBLE",
            "Unbounded": "UNBOUNDED",
            "Not solved": "NOT_SOLVED",
        }
        status_str = status_map.get(pulp.LpStatus[prob.status], pulp.LpStatus[prob.status])

        sol = LPSolution(
            status=status_str,
            solve_time_seconds=round(elapsed, 4),
            crops=self.crops,
            raw=prob,
        )

        if status_str != "OPTIMAL":
            return sol

        for c in self.crops:
            sol.secano[c] = round(pulp.value(x_s[c]), 2)
            sol.regadio[c] = round(pulp.value(x_r[c]), 2)
            sol.production[c] = round(pulp.value(prod[c]), 2)
            sol.residue[c] = round(pulp.value(res_pos[c]), 2)
            sol.benefit[c] = round(pulp.value(benefit[c]), 2)

        sol.total_production_t = round(pulp.value(total_prod), 2)
        sol.total_residue_t = round(pulp.value(total_res), 2)
        sol.total_benefit_eur = round(pulp.value(total_benefit), 2)

        sol.objective_value = (
            sol.total_residue_t if mode == "minimize_residue" else sol.total_benefit_eur
        )
        return sol
