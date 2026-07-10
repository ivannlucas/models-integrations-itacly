"""Single-objective GA optimizer (DEAP) for the ml34 pasteurization plugin.

Faithful port of the delivered ``src/predict/optimization.py`` (GA v4):
same operators, same parameters and same per-scenario seeding
(``random.seed(seed)`` + ``np.random.seed(seed)``) so that a scenario run
with the same seed reproduces the AI team's backtesting results.

Variable names (T_in_leche, F_flow, scaler_X...) intentionally mirror the
original codebase and dataset columns; DEAP creator/toolbox members are
registered dynamically, hence the targeted pylint disables.
"""
# pylint: disable=invalid-name,too-many-arguments,too-many-positional-arguments,no-member
from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import torch
from deap import algorithms, base, creator, tools

from app.plugins.ml34_dairy_pasteurization_energy_ga.constants import (
    GA_BOUNDS,
    GA_CXPB,
    GA_MUTPB,
    GA_N_GEN,
    GA_POP_SIZE,
    PENALTY_FACTOR,
    T_OUT_MIN,
)


def predict_scenario(model, scaler_X, scaler_y, T_in_leche, F_flow, T_servicio,
                     t_ciclo, Delta_P) -> Tuple[float, float]:
    """Run the MLP surrogate for one scenario, returning (E_consumo, T_out) in real units.

    Feature order matches model_config.json features_in_order:
    [T_in_leche, F_flow, T_servicio, t_ciclo, Delta_P].
    """
    x_raw = np.array([[T_in_leche, F_flow, T_servicio, t_ciclo, Delta_P]])
    x_scaled = scaler_X.transform(x_raw)
    with torch.no_grad():
        y_scaled = model(torch.FloatTensor(x_scaled)).numpy()
    y_real = scaler_y.inverse_transform(y_scaled)[0]
    return float(y_real[0]), float(y_real[1])


def _fitness_consumo_especifico(individual, T_in_leche, t_ciclo, Delta_P,
                                model, scaler_X, scaler_y) -> Tuple[float]:
    """Minimize E_consumo / F_flow, penalizing T_out < T_OUT_MIN (penalty wall)."""
    F_flow, T_servicio = individual
    E_consumo, T_out = predict_scenario(
        model, scaler_X, scaler_y, T_in_leche, F_flow, T_servicio, t_ciclo, Delta_P
    )
    if T_out < T_OUT_MIN:
        deficit = T_OUT_MIN - T_out
        penalty = 1.0 + PENALTY_FACTOR * deficit
        return (10000.0 + (float(E_consumo) / max(float(F_flow), 1.0)) * penalty,)
    return (float(E_consumo) / max(float(F_flow), 1.0),)


def _ensure_deap_types() -> None:
    """(Re)register the single-objective fitness and individual DEAP types."""
    for cls_name in ("FitnessMin_v4", "Individual_v4"):
        if cls_name in creator.__dict__:
            del creator.__dict__[cls_name]
    creator.create("FitnessMin_v4", base.Fitness, weights=(-1.0,))
    creator.create("Individual_v4", list, fitness=creator.FitnessMin_v4)


def _check_bounds(low: list, up: list):
    """Decorator that clamps genes to allowed bounds after crossover/mutation."""
    def decorator(func):
        def wrapper(*args, **kw):
            offspring = func(*args, **kw)
            for child in offspring:
                for i, (lo, hi) in enumerate(zip(low, up)):
                    child[i] = max(lo, min(hi, child[i]))
            return offspring
        return wrapper
    return decorator


def setup_ga_toolbox(bounds: Dict | None = None) -> base.Toolbox:
    """Configure the DEAP toolbox exactly as the original setup_ga_toolbox()."""
    bounds = bounds or GA_BOUNDS
    _ensure_deap_types()

    toolbox = base.Toolbox()
    toolbox.register("attr_F_flow", random.uniform,
                     bounds["F_flow"][0], bounds["F_flow"][1])
    toolbox.register("attr_T_servicio", random.uniform,
                     bounds["T_servicio"][0], bounds["T_servicio"][1])
    toolbox.register(
        "individual", tools.initCycle, creator.Individual_v4,
        (toolbox.attr_F_flow, toolbox.attr_T_servicio), n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    low = [bounds["F_flow"][0], bounds["T_servicio"][0]]
    up = [bounds["F_flow"][1], bounds["T_servicio"][1]]
    toolbox.decorate("mate", _check_bounds(low, up))
    toolbox.decorate("mutate", _check_bounds(low, up))
    return toolbox


def run_ga_single(toolbox, T_in_leche, t_ciclo, Delta_P, model, scaler_X,
                  scaler_y, seed: int | None = None):
    """Run the full single-objective GA for one scenario; returns the HallOfFame.

    Deterministic per scenario when *seed* is given (matches the original
    ``run_ga_single``: seed reset + varOr + tournament + HallOfFame(1)).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    toolbox.register(
        "evaluate", _fitness_consumo_especifico,
        T_in_leche=T_in_leche, t_ciclo=t_ciclo, Delta_P=Delta_P,
        model=model, scaler_X=scaler_X, scaler_y=scaler_y,
    )

    pop = toolbox.population(n=GA_POP_SIZE)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(1)
    hof.update(pop)

    for _ in range(GA_N_GEN):
        offspring = algorithms.varOr(
            pop, toolbox, lambda_=len(pop), cxpb=GA_CXPB, mutpb=GA_MUTPB,
        )
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = toolbox.select(pop + offspring, len(pop))
        hof.update(pop)

    return hof
