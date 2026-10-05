"""Adaptive per-instance GA (DEAP) for the ml36 plugin.

Port of the delivered ``realtime_decision_pipeline._run_adaptive_ga_for_row`` and
``ga_model_pipeline`` fitness: minimise predicted CO2 with a soft penalty
``PENALTY_FACTOR * (T_OUT_MIN - T_out)`` when T_out < T_OUT_MIN. Same operators,
parameters and per-row seeding (``random.seed`` + ``np.random.seed``) so a row run
with the same seed reproduces the AI team's decisions.

Variable names (scaler_X, T_serv, ...) intentionally mirror the original codebase;
DEAP creator/toolbox members are registered dynamically, hence the pylint disables.
"""
# pylint: disable=invalid-name,too-many-arguments,too-many-positional-arguments,no-member
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from deap import algorithms, base, creator, tools

from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.constants import (
    CO2_IDX,
    CONTEXT_COLS,
    GA_BOUNDS,
    GA_CX_ALPHA,
    GA_CXPB,
    GA_ELITE_SIZE,
    GA_MUTATION_ETA,
    GA_MUTATION_INDPB,
    GA_MUTPB,
    GA_N_GEN,
    GA_POP_SIZE,
    GA_TOURNAMENT_SIZE,
    GENE_ORDER,
    PENALTY_FACTOR,
    T_OUT_IDX,
    T_OUT_MIN,
)
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.preprocessing import (
    build_feature_frame,
    clip_individual,
)

_FITNESS_NAME = "FitnessMinMl36"
_INDIVIDUAL_NAME = "IndividualMl36"
if _FITNESS_NAME not in creator.__dict__:
    creator.create(_FITNESS_NAME, base.Fitness, weights=(-1.0,))
if _INDIVIDUAL_NAME not in creator.__dict__:
    creator.create(_INDIVIDUAL_NAME, list, fitness=creator.__dict__[_FITNESS_NAME])


def predict_scenario(model, scaler_X, scaler_Y, context: dict,
                     individual: Sequence[float]) -> tuple[float, float]:
    """Run the MLP for one context + (T_serv, Delta_P, Regeneration_perc), returning (CO2, T_out).

    As in the original GA path, the genes are clipped to the operating bounds first.
    """
    t_serv, delta_p, regen = clip_individual(individual)
    row = {c: float(context[c]) for c in CONTEXT_COLS}
    row.update({"T_serv": t_serv, "Delta_P": delta_p, "Regeneration_perc": regen})
    x_scaled = scaler_X.transform(build_feature_frame(pd.DataFrame([row])))
    with torch.no_grad():
        y_scaled = model(torch.tensor(x_scaled, dtype=torch.float32)).cpu().numpy()
    y_real = scaler_Y.inverse_transform(y_scaled)[0]
    return float(y_real[CO2_IDX]), float(y_real[T_OUT_IDX])


def fitness(model, scaler_X, scaler_Y, context: dict, individual: Sequence[float]) -> float:
    """CO2 + soft penalty for T_out below the pasteurization threshold."""
    co2, t_out = predict_scenario(model, scaler_X, scaler_Y, context, individual)
    penalty = PENALTY_FACTOR * (T_OUT_MIN - t_out) if t_out < T_OUT_MIN else 0.0
    return co2 + penalty


def _check_bounds(low: list, up: list):
    """Decorator that clamps genes to allowed bounds after crossover/mutation."""
    def decorator(func):
        def wrapper(*args, **kw):
            offspring = func(*args, **kw)
            for child in offspring:
                for i, (lo, hi) in enumerate(zip(low, up)):
                    child[i] = max(float(lo), min(float(hi), float(child[i])))
            return offspring
        return wrapper
    return decorator


def run_adaptive_ga(model, scaler_X, scaler_Y, context: dict,
                    seed_individuals: list[list[float]], row_seed: int) -> list[float]:
    """Run the adaptive GA for one context row; returns the best (T_serv, Delta_P, Regeneration_perc).

    Deterministic for a given *row_seed*.
    """
    random.seed(row_seed)
    np.random.seed(row_seed)

    low = [GA_BOUNDS[g][0] for g in GENE_ORDER]
    up = [GA_BOUNDS[g][1] for g in GENE_ORDER]

    toolbox = base.Toolbox()
    toolbox.register("attr_t_serv", random.uniform, low[0], up[0])
    toolbox.register("attr_delta_p", random.uniform, low[1], up[1])
    toolbox.register("attr_regen", random.uniform, low[2], up[2])
    toolbox.register(
        "individual", tools.initCycle, creator.__dict__[_INDIVIDUAL_NAME],
        (toolbox.attr_t_serv, toolbox.attr_delta_p, toolbox.attr_regen), n=1,
    )
    toolbox.register(
        "evaluate",
        lambda ind: (fitness(model, scaler_X, scaler_Y, context, ind),),
    )
    toolbox.register("mate", tools.cxBlend, alpha=GA_CX_ALPHA)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up,
                     eta=GA_MUTATION_ETA, indpb=GA_MUTATION_INDPB)
    toolbox.register("select", tools.selTournament, tournsize=GA_TOURNAMENT_SIZE)
    toolbox.decorate("mate", _check_bounds(low, up))
    toolbox.decorate("mutate", _check_bounds(low, up))

    population = [toolbox.individual() for _ in range(GA_POP_SIZE)]
    # Faithful to the delivered code: the seed individuals are appended and then the
    # population is truncated back to pop_size, which drops them again — so in practice
    # "hybrid" behaves like "adaptive". Kept as-is to reproduce the AI team's results.
    for seed_ind in seed_individuals:
        population.append(creator.__dict__[_INDIVIDUAL_NAME](clip_individual(seed_ind)))
    if len(population) > GA_POP_SIZE:
        population = population[:GA_POP_SIZE]

    hall = tools.HallOfFame(1)
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)
    hall.update(population)

    for _ in range(GA_N_GEN):
        elites = tools.selBest(population, GA_ELITE_SIZE)
        offspring = algorithms.varAnd(population, toolbox, cxpb=GA_CXPB, mutpb=GA_MUTPB)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        selected = toolbox.select(population + offspring, k=GA_POP_SIZE - GA_ELITE_SIZE)
        population = elites + selected
        hall.update(population)

    return [float(g) for g in hall[0]]
