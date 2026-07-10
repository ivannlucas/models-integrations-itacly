"""
Single-objective GA optimization with DEAP.

Encapsulates all single-objective genetic algorithm logic:
  - DEAP toolbox configuration (types, operators, decorators).
  - Single-objective fitness function (min E_consumo / F_flow).
  - GA execution for a given scenario using HallOfFame(1).

Used in optimization_ga_v4.ipynb, ga_evaluation.ipynb and tuning_params_ga_v4.ipynb.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn
from deap import algorithms, base, creator, tools
from sklearn.preprocessing import MinMaxScaler

from src.predict.inference import predict_with_model
from src.utils.constants import BOUNDS, GA_DEFAULT_CONFIG, PENALTY_FACTOR, T_OUT_MIN


# =============================================================================
# DEAP TYPE CONFIGURATION (executed on module import)
# =============================================================================

def _ensure_deap_types():
    """Register FitnessMin_v4 and Individual_v4 types if they do not already exist."""
    for cls_name in ["FitnessMin_v4", "Individual_v4"]:
        if cls_name in creator.__dict__:
            del creator.__dict__[cls_name]
    creator.create("FitnessMin_v4", base.Fitness, weights=(-1.0,))
    creator.create("Individual_v4", list, fitness=creator.FitnessMin_v4)


# =============================================================================
# SINGLE-OBJECTIVE FITNESS FUNCTION
# =============================================================================

def fitness_consumo_especifico(
    individual: list,
    T_in_leche: float,
    t_ciclo: float,
    Delta_P: float,
    model: nn.Module = None,
    scaler_X: MinMaxScaler = None,
    scaler_y: MinMaxScaler = None,
) -> Tuple[float]:
    """
    Single-objective fitness function with temperature constraint.

    Objective:
        Minimize E_consumo / F_flow (specific energy consumption, kW/(L/h)).

    Constraint:
        - T_out_leche >= 72.3 C (food safety).
        - A proportional penalty is applied to the deficit, guiding
          the GA evolutionarily towards the feasible region.

    Parameters
    ----------
    individual : list
        [F_flow, T_servicio] — decision variables (chromosome).
    T_in_leche, t_ciclo, Delta_P : float
        Uncontrollable variables (fixed external conditions).
    model, scaler_X, scaler_y : model objects
        Loaded artifacts.

    Returns
    -------
    tuple[float]
        (specific_consumption [possibly penalized],)
    """
    F_flow, T_servicio = individual
    E_consumo, T_out = predict_with_model(
        F_flow, T_servicio, T_in_leche, t_ciclo, Delta_P,
        model=model, scaler_X=scaler_X, scaler_y=scaler_y,
    )

    # Constraint handling via penalty wall
    if T_out < T_OUT_MIN:
        deficit = T_OUT_MIN - T_out
        penalty = 1.0 + PENALTY_FACTOR * deficit
        penalized_value = 10000.0 + (float(E_consumo) / max(float(F_flow), 1.0)) * penalty
        return (penalized_value,)

    specific_consumption = float(E_consumo) / max(float(F_flow), 1.0)
    return (specific_consumption,)


# =============================================================================
# BOUNDS DECORATOR
# =============================================================================

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


# =============================================================================
# GA TOOLBOX SETUP
# =============================================================================

def setup_ga_toolbox(
    bounds: Dict = None,
    ga_config: Dict = None,
) -> Tuple[base.Toolbox, Dict]:
    """
    Configure the DEAP Toolbox for single-objective GA.

    Parameters
    ----------
    bounds : dict, optional
        Decision variable bounds.
        Default: BOUNDS from constants.py.
    ga_config : dict, optional
        GA parameters (pop_size, n_gen, cxpb, mutpb).
        Default: GA_DEFAULT_CONFIG.

    Returns
    -------
    tuple
        (configured toolbox, ga_config used)
    """
    bounds = bounds or BOUNDS
    ga_config = ga_config or dict(GA_DEFAULT_CONFIG)

    _ensure_deap_types()

    toolbox = base.Toolbox()

    # Gene generators (decision variables)
    toolbox.register("attr_F_flow", random.uniform,
                     bounds["F_flow"][0], bounds["F_flow"][1])
    toolbox.register("attr_T_servicio", random.uniform,
                     bounds["T_servicio"][0], bounds["T_servicio"][1])

    # Individual = [F_flow, T_servicio]
    toolbox.register(
        "individual", tools.initCycle, creator.Individual_v4,
        (toolbox.attr_F_flow, toolbox.attr_T_servicio), n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)

    # Tournament selection
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Decorator to enforce bounds after crossover/mutation
    low = [bounds["F_flow"][0], bounds["T_servicio"][0]]
    up = [bounds["F_flow"][1], bounds["T_servicio"][1]]
    toolbox.decorate("mate", _check_bounds(low, up))
    toolbox.decorate("mutate", _check_bounds(low, up))

    return toolbox, ga_config


# =============================================================================
# GA EXECUTION FOR A SINGLE SCENARIO
# =============================================================================

def run_ga_single(
    toolbox: base.Toolbox,
    T_in_leche: float,
    t_ciclo: float,
    Delta_P: float,
    model: nn.Module,
    scaler_X: MinMaxScaler,
    scaler_y: MinMaxScaler,
    ga_config: Dict = None,
    seed: int = None,
) -> Tuple[list, list]:
    """
    Run complete single-objective GA for a scenario (T_in, t_ciclo, Delta_P).

    Uses varOr + selTournament + HallOfFame(1) architecture.

    Parameters
    ----------
    toolbox : base.Toolbox
        Toolbox configured via setup_ga_toolbox().
    T_in_leche, t_ciclo, Delta_P : float
        External scenario conditions.
    model, scaler_X, scaler_y : objects
        Model artifacts.
    ga_config : dict, optional
        GA parameters. Default: GA_DEFAULT_CONFIG.
    seed : int, optional
        Seed for reproducibility of this scenario.

    Returns
    -------
    tuple
        (final population, HallOfFame containing the best individual)
    """
    ga_config = ga_config or dict(GA_DEFAULT_CONFIG)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Register fitness function for this scenario
    toolbox.register(
        "evaluate", fitness_consumo_especifico,
        T_in_leche=T_in_leche, t_ciclo=t_ciclo, Delta_P=Delta_P,
        model=model, scaler_X=scaler_X, scaler_y=scaler_y,
    )

    # Initialize and evaluate population
    pop = toolbox.population(n=ga_config["pop_size"])
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Hall of Fame for elitism
    hof = tools.HallOfFame(1)
    hof.update(pop)

    # Evolution
    for gen in range(ga_config["n_gen"]):
        offspring = algorithms.varOr(
            pop, toolbox, lambda_=len(pop),
            cxpb=ga_config["cxpb"], mutpb=ga_config["mutpb"],
        )
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = toolbox.select(pop + offspring, len(pop))
        hof.update(pop)

    return pop, hof
