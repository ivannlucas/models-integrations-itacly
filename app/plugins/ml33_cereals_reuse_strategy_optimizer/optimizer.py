"""Exact emissions-minimizing assignment optimizer — the deployed decision engine.

Ported from ``src/predict/exact_optimizer.py`` in the original model delivery
(``inbox/a33``, originally delivered as
``a33-cnp-cereals-neuroevolutivo-reduccion-ambiental-residuos`` — see manifest.yaml). The formulas and
solver setup are kept numerically identical to the source so the manifest's golden cases
(extracted from the delivery's own ``data/predictions/inference_with_constraints.csv``)
reproduce exactly.

Deliberate deviation from the source: the original wraps the solver call in an OS-level
fd-1 redirect (``_suppress_native_stdout``) to silence HiGHS' native stdout banner. That
trick swaps the process-wide file descriptor 1, which is safe in a single-threaded CLI
script but not in a shared multi-threaded ASGI server — concurrent requests could
momentarily lose or interleave unrelated stdout/log output while one request's block is
solving. It is dropped here; ``options={"disp": False}`` already silences scipy's own
solver output. This changes nothing about the numeric result, only about log cosmetics.

Problem structure
------------------
Lots are processed in operational blocks: capacities reset every ``lots_per_day`` lots (a
simulated plant day). Within each block, every lot must be assigned to exactly one reuse
strategy and the total volume routed to each strategy cannot exceed its remaining daily
capacity. The objective is to minimize the total simulated CO2 emissions of the block.

Because the per-lot emission of each strategy is a constant once the lot is known, this is
a small generalized assignment problem solved to optimality as a mixed-integer linear
program (MILP) with ``scipy.optimize.milp``:

    minimize    sum_{i,s} e[i,s] * x[i,s]
    subject to  sum_s x[i,s] = 1                (each lot assigned once)
                sum_i vol_i * x[i,s] <= cap_s    (capacity per strategy)
                x[i,s] in {0, 1}

With four strategies and ``lots_per_day`` lots per block, each sub-problem has
``4 * lots_per_day`` binary variables and is solved in milliseconds. The biomass-combustion
strategy carries a very large default capacity, so every block is feasible under defaults;
a per-lot greedy fallback is retained only as a defensive guard for non-default capacities.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from app.plugins.ml33_cereals_reuse_strategy_optimizer.constants import (
    ASSIGNMENT_SOURCE_CAPACITY_FALLBACK,
    ASSIGNMENT_SOURCE_EXACT,
    REQUIRED_LOT_COLUMNS,
    STRATEGY_ORDER,
    STRATEGY_TEMPERATURE_C,
)

logger = logging.getLogger(__name__)


def strategy_emissions(
    volume_tons: float,
    moisture_pct: float,
    strategy: str,
    subproduct_type: str,
) -> float:
    """Simulated CO2 emissions (kg) for a (lot, strategy) pair.

    Identical business physics to the original delivery's training (``evolution.py``),
    inference (``inference.py``) and evaluation (``evaluate_inference.py``) — all four
    shared this exact formula. Temperature is derived deterministically from the
    strategy, never taken as a decision input (``season`` is likewise not read here —
    see the manifest's known_issues on the vestigial ``season`` column).

    Args:
        volume_tons: Lot volume in tons (physical units).
        moisture_pct: Lot moisture percentage.
        strategy: Candidate strategy label.
        subproduct_type: Residue category label.

    Returns:
        float: Estimated CO2 emissions in kg.
    """

    temperature_c = STRATEGY_TEMPERATURE_C.get(strategy, 180.0)
    emission_base = (temperature_c * 0.5) * volume_tons

    if strategy == "Biomass combustion":
        return emission_base + ((moisture_pct**1.5) * 2.0) - 50.0
    if strategy == "Animal feed":
        if subproduct_type in {"Husk", "Straw", "Silo dust"} or moisture_pct > 18.0:
            return emission_base * 1.8
        return emission_base * 0.4
    if strategy == "Biochar":
        if moisture_pct < 10.0:
            return emission_base * 0.2
        return emission_base * 1.2
    if strategy == "Composting":
        return emission_base * 0.8 + (volume_tons * 5.0)
    return emission_base


def _greedy_block(
    volumes: np.ndarray,
    moistures: np.ndarray,
    subproducts: np.ndarray,
    capacities: dict[str, float],
    fallback_strategy: str,
) -> tuple[list[str], list[str]]:
    """Per-lot greedy fallback used only when the MILP is infeasible.

    Assigns each lot to its lowest-emission feasible strategy in order, decrementing
    remaining capacity. Returns assignments and per-lot sources.
    """

    remaining = dict(capacities)
    assigned: list[str] = []
    sources: list[str] = []
    for i in range(len(volumes)):
        best_strategy = None
        best_emissions = float("inf")
        for strategy in STRATEGY_ORDER:
            if volumes[i] > remaining.get(strategy, 0.0):
                continue
            candidate = strategy_emissions(volumes[i], moistures[i], strategy, subproducts[i])
            if candidate < best_emissions:
                best_emissions = candidate
                best_strategy = strategy
        if best_strategy is None:
            assigned.append(fallback_strategy)
            sources.append(ASSIGNMENT_SOURCE_CAPACITY_FALLBACK)
        else:
            remaining[best_strategy] -= volumes[i]
            assigned.append(best_strategy)
            sources.append(ASSIGNMENT_SOURCE_EXACT)
    return assigned, sources


def solve_block(
    volumes: np.ndarray,
    moistures: np.ndarray,
    subproducts: np.ndarray,
    capacities: dict[str, float],
    fallback_strategy: str,
    logger_: logging.Logger | None = None,
) -> tuple[list[str], list[str]]:
    """Solve one capacity-reset block to optimality via MILP.

    Args:
        volumes: Lot volumes (tons) for the block.
        moistures: Lot moisture percentages for the block.
        subproducts: Residue categories for the block.
        capacities: Remaining capacity per strategy at block start.
        fallback_strategy: Strategy applied to any lot the MILP cannot place.
        logger_: Optional logger for infeasibility diagnostics.

    Returns:
        tuple[list[str], list[str]]: Assigned strategies and per-lot sources.
    """

    n_lots = len(volumes)
    n_strategies = len(STRATEGY_ORDER)
    if n_lots == 0:
        return [], []

    # Cost vector c[i * n_strategies + s] = emissions of lot i under strategy s.
    cost = np.empty(n_lots * n_strategies, dtype=float)
    for i in range(n_lots):
        for s, strategy in enumerate(STRATEGY_ORDER):
            cost[i * n_strategies + s] = strategy_emissions(
                float(volumes[i]), float(moistures[i]), strategy, str(subproducts[i])
            )

    # Assignment constraints: each lot assigned exactly once.
    assign_matrix = np.zeros((n_lots, n_lots * n_strategies), dtype=float)
    for i in range(n_lots):
        assign_matrix[i, i * n_strategies:(i + 1) * n_strategies] = 1.0
    assignment_constraint = LinearConstraint(assign_matrix, lb=1.0, ub=1.0)

    # Capacity constraints: volume routed to each strategy <= remaining capacity.
    capacity_matrix = np.zeros((n_strategies, n_lots * n_strategies), dtype=float)
    capacity_ub = np.empty(n_strategies, dtype=float)
    for s, strategy in enumerate(STRATEGY_ORDER):
        for i in range(n_lots):
            capacity_matrix[s, i * n_strategies + s] = float(volumes[i])
        capacity_ub[s] = float(capacities.get(strategy, 0.0))
    capacity_constraint = LinearConstraint(capacity_matrix, ub=capacity_ub)

    result = milp(
        c=cost,
        constraints=[assignment_constraint, capacity_constraint],
        integrality=np.ones_like(cost),
        bounds=Bounds(lb=0.0, ub=1.0),
        options={"disp": False},
    )

    if not result.success or result.x is None:
        active_logger = logger_ or logger
        active_logger.warning(
            "MILP block infeasible or unsolved (%s). Falling back to per-lot greedy.",
            getattr(result, "message", "no message"),
        )
        return _greedy_block(volumes, moistures, subproducts, capacities, fallback_strategy)

    solution = np.asarray(result.x, dtype=float).reshape(n_lots, n_strategies)
    chosen_indices = solution.argmax(axis=1)
    assigned = [STRATEGY_ORDER[int(idx)] for idx in chosen_indices]
    sources = [ASSIGNMENT_SOURCE_EXACT] * n_lots
    return assigned, sources


def candidate_breakdown(
    volume_tons: float,
    moisture_pct: float,
    subproduct_type: str,
    capacities: dict[str, float],
) -> list[dict[str, Any]]:
    """Per-lot explanation: estimated emissions for EVERY candidate strategy, not just
    the chosen one — the "why" for this deterministic solver.

    Exact, not an approximation (unlike SHAP): strategy_emissions() is the same pure
    function the MILP itself minimizes, so this is simply evaluating it once per
    candidate instead of only for the winner. Sorted ascending by emissions, so the
    chosen strategy normally appears first among the feasible ones.

    ``feasible`` here checks only whether the strategy's total daily capacity covers
    THIS lot's own volume in isolation (volume_tons <= capacities[strategy]) — it does
    NOT account for contention with the other lots in the same block, which is what the
    joint MILP solve (solve_block) actually decides. A strategy marked feasible here can
    still lose to another lot's claim on the same shared capacity; see
    ai_assignment_source for the real, block-level outcome.

    Args:
        volume_tons: Lot volume in tons.
        moisture_pct: Lot moisture percentage.
        subproduct_type: Residue category label.
        capacities: Daily capacity per strategy label (the block's starting capacities).

    Returns:
        list[dict[str, Any]]: One entry per STRATEGY_ORDER member, each
        {strategy, estimated_emissions_kg, feasible}, sorted by estimated_emissions_kg.
    """

    candidates = [
        {
            "strategy": strategy,
            "estimated_emissions_kg": strategy_emissions(
                volume_tons, moisture_pct, strategy, subproduct_type
            ),
            "feasible": volume_tons <= capacities.get(strategy, 0.0),
        }
        for strategy in STRATEGY_ORDER
    ]
    candidates.sort(key=lambda c: c["estimated_emissions_kg"])
    return candidates


class ExactEmissionsOptimizer:
    """Deployed decision engine: exact per-block emissions minimization.

    Args:
        capacities: Daily capacity per strategy label.
        lots_per_day: Number of lots before capacities reset.
        fallback_strategy: Strategy used if a lot cannot be placed.
        logger_: Logger instance.
    """

    def __init__(
        self,
        capacities: dict[str, float],
        lots_per_day: int,
        fallback_strategy: str,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.capacities = dict(capacities)
        self.lots_per_day = int(lots_per_day)
        self.fallback_strategy = fallback_strategy
        self.logger = logger_ or logger
        self.capacity_fallback_count = 0

    def infer_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Assign strategies for a full dataframe, block by block.

        Args:
            dataframe: Input with the required inference columns
                (REQUIRED_LOT_COLUMNS).

        Returns:
            pd.DataFrame: Traceability columns (ai_assigned_strategy,
            ai_assignment_source, ai_is_fallback, estimated_emissions_kg,
            candidate_emissions — see candidate_breakdown()).

        Raises:
            ValueError: If required columns are missing.
        """

        missing = set(REQUIRED_LOT_COLUMNS).difference(dataframe.columns)
        if missing:
            raise ValueError(f"Missing required columns for inference: {sorted(missing)}")

        volumes = dataframe["generated_volume_tons"].astype(float).to_numpy()
        moistures = dataframe["moisture_pct"].astype(float).to_numpy()
        subproducts = dataframe["subproduct_type"].astype(str).to_numpy()

        assigned: list[str] = []
        sources: list[str] = []
        n_rows = len(dataframe)
        for start in range(0, n_rows, self.lots_per_day):
            end = min(start + self.lots_per_day, n_rows)
            block_assigned, block_sources = solve_block(
                volumes=volumes[start:end],
                moistures=moistures[start:end],
                subproducts=subproducts[start:end],
                capacities=self.capacities,
                fallback_strategy=self.fallback_strategy,
                logger_=self.logger,
            )
            assigned.extend(block_assigned)
            sources.extend(block_sources)

        self.capacity_fallback_count = sum(
            1 for src in sources if src == ASSIGNMENT_SOURCE_CAPACITY_FALLBACK
        )
        if self.capacity_fallback_count > 0:
            self.logger.warning(
                "Exact optimizer capacity fallback count=%s (strategy=%s).",
                self.capacity_fallback_count,
                self.fallback_strategy,
            )

        estimated_emissions = [
            strategy_emissions(float(volumes[i]), float(moistures[i]), assigned[i], str(subproducts[i]))
            for i in range(n_rows)
        ]
        candidates = [
            candidate_breakdown(
                float(volumes[i]), float(moistures[i]), str(subproducts[i]), self.capacities
            )
            for i in range(n_rows)
        ]

        trace_df = pd.DataFrame(
            {
                "ai_assigned_strategy": assigned,
                "ai_assignment_source": sources,
                "estimated_emissions_kg": estimated_emissions,
                "candidate_emissions": candidates,
            }
        )
        trace_df["ai_is_fallback"] = trace_df["ai_assignment_source"].isin(
            {ASSIGNMENT_SOURCE_CAPACITY_FALLBACK}
        )
        return trace_df


def summarize_strategy_distribution(assigned: pd.Series) -> dict[str, Any]:
    """Build count/percentage distribution statistics for assigned strategies.

    Args:
        assigned: Predicted strategy labels.

    Returns:
        dict[str, Any]: Count and percentage maps.
    """

    counts = assigned.value_counts(dropna=False)
    percentages = assigned.value_counts(normalize=True, dropna=False).mul(100).round(2)
    return {
        "counts": counts.to_dict(),
        "percentages": percentages.to_dict(),
    }
