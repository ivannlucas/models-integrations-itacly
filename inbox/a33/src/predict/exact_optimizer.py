"""Exact emissions-minimizing assignment optimizer (primary deployed model).

This module implements the deployed decision engine of the system. It replaces
the neuroevolutionary policy as the *primary* optimizer and answers the auditor
observation that "for this type of problem, linear programming could obtain an
exact solution".

Problem structure
-----------------
Lots are processed in operational blocks: capacities reset every
``lots_per_day`` lots (a simulated plant day). Within each block, every lot must
be assigned to exactly one reuse strategy and the total volume routed to each
strategy cannot exceed its remaining daily capacity. The objective is to
minimize the total simulated CO2 emissions of the block.

Because the per-lot emission of each strategy is a constant once the lot is
known, this is a small **generalized assignment problem** solved to optimality
as a **mixed-integer linear program** (MILP) with ``scipy.optimize.milp``:

    minimize    sum_{i,s} e[i,s] * x[i,s]
    subject to  sum_s x[i,s] = 1                (each lot assigned once)
                sum_i vol_i * x[i,s] <= cap_s    (capacity per strategy)
                x[i,s] in {0, 1}

With four strategies and ``lots_per_day`` lots per block, each sub-problem has
``4 * lots_per_day`` binary variables and is solved in milliseconds. The
biomass-combustion strategy carries a very large capacity, so every block is
feasible; a per-lot greedy fallback is retained only as a defensive guard.

Contrast with the greedy per-lot rule (``oracle`` in ``run_baselines.py``): the
greedy rule fixes each lot independently in dataset order and can waste scarce
capacity on early lots, whereas the MILP optimizes the whole block jointly and
is therefore optimal by construction.
"""

from __future__ import annotations

import contextlib
import logging
import os

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@contextlib.contextmanager
def _suppress_native_stdout():
    """Silence HiGHS' native (C++) stdout, which scipy cannot fully mute.

    The MILP backend occasionally writes solver progress directly to the OS
    file descriptor, bypassing Python's stdout. We redirect fd 1 to os.devnull
    for the duration of the solve to keep CLI output clean.
    """

    try:
        saved_fd = os.dup(1)
    except OSError:
        # Non-standard stdout (e.g. captured pipe without a real fd): skip.
        yield
        return
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(devnull_fd)
        os.close(saved_fd)

# Column and strategy conventions shared with the rest of the pipeline. The
# strategy order matches inference/evaluation so downstream reports align.
STRATEGY_ORDER: tuple[str, ...] = (
    "Biomass combustion",
    "Animal feed",
    "Composting",
    "Biochar",
)

STRATEGY_TEMPERATURE_C: dict[str, float] = {
    "Animal feed": 60.0,
    "Composting": 60.0,
    "Biochar": 450.0,
    "Biomass combustion": 900.0,
}

ASSIGNMENT_SOURCE_EXACT = "exact_min_emissions"
ASSIGNMENT_SOURCE_CAPACITY_FALLBACK = "capacity_fallback"


def strategy_emissions(
    volume_tons: float,
    moisture_pct: float,
    strategy: str,
    subproduct_type: str,
) -> float:
    """Simulated CO2 emissions for a (lot, strategy) pair.

    Identical business physics to training (``evolution.py``), inference
    (``inference.py``) and evaluation (``evaluate_inference.py``). Temperature
    is derived deterministically from the strategy, never taken as an input.

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

    Assigns each lot to its lowest-emission feasible strategy in order,
    decrementing remaining capacity. Returns assignments and per-lot sources.
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
    logger: logging.Logger | None = None,
) -> tuple[list[str], list[str]]:
    """Solve one capacity-reset block to optimality via MILP.

    Args:
        volumes: Lot volumes (tons) for the block.
        moistures: Lot moisture percentages for the block.
        subproducts: Residue categories for the block.
        capacities: Remaining capacity per strategy at block start.
        fallback_strategy: Strategy applied to any lot the MILP cannot place.
        logger: Optional logger for infeasibility diagnostics.

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
        assign_matrix[i, i * n_strategies : (i + 1) * n_strategies] = 1.0
    assignment_constraint = LinearConstraint(assign_matrix, lb=1.0, ub=1.0)

    # Capacity constraints: volume routed to each strategy <= remaining capacity.
    capacity_matrix = np.zeros((n_strategies, n_lots * n_strategies), dtype=float)
    capacity_ub = np.empty(n_strategies, dtype=float)
    for s, strategy in enumerate(STRATEGY_ORDER):
        for i in range(n_lots):
            capacity_matrix[s, i * n_strategies + s] = float(volumes[i])
        capacity_ub[s] = float(capacities.get(strategy, 0.0))
    capacity_constraint = LinearConstraint(capacity_matrix, ub=capacity_ub)

    with _suppress_native_stdout():
        result = milp(
            c=cost,
            constraints=[assignment_constraint, capacity_constraint],
            integrality=np.ones_like(cost),
            bounds=Bounds(lb=0.0, ub=1.0),
            options={"disp": False},
        )

    if not result.success or result.x is None:
        if logger is not None:
            logger.warning(
                "MILP block infeasible or unsolved (%s). Falling back to per-lot greedy.",
                getattr(result, "message", "no message"),
            )
        return _greedy_block(volumes, moistures, subproducts, capacities, fallback_strategy)

    solution = np.asarray(result.x, dtype=float).reshape(n_lots, n_strategies)
    chosen_indices = solution.argmax(axis=1)
    assigned = [STRATEGY_ORDER[int(idx)] for idx in chosen_indices]
    sources = [ASSIGNMENT_SOURCE_EXACT] * n_lots
    return assigned, sources


class ExactEmissionsOptimizer:
    """Deployed decision engine: exact per-block emissions minimization.

    Args:
        capacities: Daily capacity per strategy label.
        lots_per_day: Number of lots before capacities reset.
        fallback_strategy: Strategy used if a lot cannot be placed.
        logger: Logger instance.
    """

    def __init__(
        self,
        capacities: dict[str, float],
        lots_per_day: int,
        fallback_strategy: str,
        logger: logging.Logger,
    ) -> None:
        self.capacities = dict(capacities)
        self.lots_per_day = int(lots_per_day)
        self.fallback_strategy = fallback_strategy
        self.logger = logger
        self.capacity_fallback_count = 0

    def infer_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Assign strategies for a full dataframe, block by block.

        Args:
            dataframe: Input with the four required inference columns.

        Returns:
            pd.DataFrame: Traceability columns (ai_assigned_strategy,
            ai_assignment_source, ai_is_fallback).

        Raises:
            ValueError: If required columns are missing.
        """

        required = {"generated_volume_tons", "moisture_pct", "subproduct_type", "season"}
        missing = required.difference(dataframe.columns)
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
                logger=self.logger,
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

        trace_df = pd.DataFrame(
            {
                "ai_assigned_strategy": assigned,
                "ai_assignment_source": sources,
            }
        )
        trace_df["ai_is_fallback"] = trace_df["ai_assignment_source"].isin(
            {ASSIGNMENT_SOURCE_CAPACITY_FALLBACK}
        )
        return trace_df
