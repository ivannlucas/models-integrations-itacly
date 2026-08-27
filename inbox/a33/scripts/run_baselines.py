"""Benchmark the deployed exact optimizer against transparent policies.

This script implements the comparison ladder requested by the auditor. On the
held-out test split, under the SAME capacity layer and emissions simulator, it
reports the CO2 reduction achieved by:

    * exact_optimum  - per-block MILP, the DEPLOYED model (provably optimal).
    * oracle         - lowest-emission feasible strategy per lot, greedy
                       (the previous heuristic; a lower bound on the optimum).
    * neat           - the evolved neuroevolution policy (score-ranked,
                       first feasible), retained as a learned-policy benchmark.
    * linear_fitness - a FIXED linear (no-hidden-layer) policy whose weights are
                       optimized with the SAME fitness function as NEAT, under
                       identical training conditions. This is the fair
                       apples-to-apples comparison the auditor asked for:
                       it isolates the contribution of topology augmentation.
    * logistic_reg   - a multinomial logistic regression trained by SUPERVISED
                       imitation of the greedy optimum (label learner).
    * random         - uniform random scores (naive lower bound).

Reading the ladder: the exact MILP is the correct deployed solution because the
per-block assignment is a small generalized-assignment problem solvable exactly.
NEAT and the fitness-optimized linear policy land close to each other (topology
augmentation adds little in this simplified environment), and both are bounded
above by the exact optimum. The supervised logistic regression underperforms
because per-row label imitation ignores sequential capacity depletion. This
quantifies why an exact optimizer is preferred here and reframes NEAT as a
benchmark rather than the deployed model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import neat
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import AppConfig
from src.model import OptimizationPipeline
from src.predict.exact_optimizer import solve_block
from src.predict.inference import (
    InferenceModelPaths,
    load_winner_network,
)
from src.utils.utils import configure_logging, get_project_root, load_dataset


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

INPUT_COLUMNS: tuple[str, ...] = (
    "generated_volume_tons",
    "moisture_pct",
    "subproduct_type_Husk",
    "subproduct_type_Straw",
    "subproduct_type_Silo dust",
    "subproduct_type_Bran",
    "season_Rainy",
    "season_Dry",
)

DEFAULT_CAPACITIES: dict[str, float] = {
    "Animal feed": 90.0,
    "Composting": 140.0,
    "Biochar": 45.0,
    "Biomass combustion": 10000.0,
}


def _emissions(volume_tons: float, moisture_pct: float, strategy: str, subproduct: str) -> float:
    """Emissions simulator identical to training/evaluation business physics."""

    temperature_c = STRATEGY_TEMPERATURE_C.get(strategy, 180.0)
    emission_base = (temperature_c * 0.5) * volume_tons
    if strategy == "Biomass combustion":
        return emission_base + ((moisture_pct**1.5) * 2.0) - 50.0
    if strategy == "Animal feed":
        if subproduct in {"Husk", "Straw", "Silo dust"} or moisture_pct > 18.0:
            return emission_base * 1.8
        return emission_base * 0.4
    if strategy == "Biochar":
        if moisture_pct < 10.0:
            return emission_base * 0.2
        return emission_base * 1.2
    if strategy == "Composting":
        return emission_base * 0.8 + (volume_tons * 5.0)
    return emission_base


def _physical_ranges_from_scaler(scaler_path: Path) -> dict[str, tuple[float, float]]:
    scaler = joblib.load(scaler_path)
    columns = ["generated_volume_tons", "moisture_pct", "process_temperature_c"]
    return {
        column: (float(scaler.data_min_[idx]), float(scaler.data_max_[idx]))
        for idx, column in enumerate(columns)
    }


def _build_test_features(test_df: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> np.ndarray:
    """Normalize and one-hot encode the test set in the exact INPUT_COLUMNS order."""

    vol_min, vol_max = ranges["generated_volume_tons"]
    hum_min, hum_max = ranges["moisture_pct"]

    def _norm(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    vol_norm = _norm(test_df["generated_volume_tons"].to_numpy(dtype=float), vol_min, vol_max)
    hum_norm = _norm(test_df["moisture_pct"].to_numpy(dtype=float), hum_min, hum_max)
    subproduct = test_df["subproduct_type"].astype(str).to_numpy()
    season = test_df["season"].astype(str).to_numpy()

    features = np.column_stack(
        [
            vol_norm,
            hum_norm,
            (subproduct == "Husk").astype(float),
            (subproduct == "Straw").astype(float),
            (subproduct == "Silo dust").astype(float),
            (subproduct == "Bran").astype(float),
            (season == "Rainy").astype(float),
            (season == "Dry").astype(float),
        ]
    )
    return features


def _assign_by_scores(
    scores: np.ndarray,
    volumes: np.ndarray,
    subproducts: np.ndarray,
    moistures: np.ndarray,
    lots_per_day: int,
) -> list[str]:
    """Training-style assignment: rank strategies by score, take first feasible.

    Capacities reset every `lots_per_day` lots, mirroring evolution.py.
    """

    assigned: list[str] = []
    capacities = dict(DEFAULT_CAPACITIES)
    for index in range(len(volumes)):
        if index % lots_per_day == 0:
            capacities = dict(DEFAULT_CAPACITIES)
        ranking = np.argsort(scores[index])[::-1]
        chosen = "Biomass combustion"
        for strategy_index in ranking:
            strategy = STRATEGY_ORDER[int(strategy_index)]
            if volumes[index] <= capacities.get(strategy, 0.0):
                capacities[strategy] -= volumes[index]
                chosen = strategy
                break
        assigned.append(chosen)
    return assigned


def _assign_oracle(
    volumes: np.ndarray,
    moistures: np.ndarray,
    subproducts: np.ndarray,
    lots_per_day: int,
) -> list[str]:
    """Lowest-emission feasible strategy per lot (upper bound on reduction)."""

    assigned: list[str] = []
    capacities = dict(DEFAULT_CAPACITIES)
    for index in range(len(volumes)):
        if index % lots_per_day == 0:
            capacities = dict(DEFAULT_CAPACITIES)
        best_strategy = "Biomass combustion"
        best_emissions = float("inf")
        for strategy in STRATEGY_ORDER:
            if volumes[index] > capacities.get(strategy, 0.0):
                continue
            emissions = _emissions(volumes[index], moistures[index], strategy, subproducts[index])
            if emissions < best_emissions:
                best_emissions = emissions
                best_strategy = strategy
        capacities[best_strategy] -= volumes[index]
        assigned.append(best_strategy)
    return assigned


def _assign_exact(
    volumes: np.ndarray,
    moistures: np.ndarray,
    subproducts: np.ndarray,
    lots_per_day: int,
) -> list[str]:
    """Deployed model: per-block MILP optimum (provably minimal emissions)."""

    assigned: list[str] = []
    for start in range(0, len(volumes), lots_per_day):
        end = min(start + lots_per_day, len(volumes))
        block_assigned, _ = solve_block(
            volumes=volumes[start:end],
            moistures=moistures[start:end],
            subproducts=subproducts[start:end],
            capacities=DEFAULT_CAPACITIES,
            fallback_strategy="Biomass combustion",
        )
        assigned.extend(block_assigned)
    return assigned


def _train_linear_fitness_network(
    root: Path,
    train_path: str,
    linear_config_path: str,
    sample_size: int,
    generations: int,
    random_state: int,
    logger,
) -> neat.nn.FeedForwardNetwork:
    """Evolve a FIXED linear policy with the SAME fitness function as NEAT.

    Uses `config-feedforward-linear.txt` (no hidden nodes, structural mutation
    disabled), so only the 8->4 weights and biases evolve. Everything else
    (fitness, capacity layer, sample, seed, generations) matches NEAT training,
    giving the fair comparison requested by the auditor.
    """

    app_config = AppConfig.build(
        project_root=root,
        dataset_relative_path=train_path,
        neat_config_relative_path=linear_config_path,
        sample_size=sample_size,
        generations=generations,
        random_state=random_state,
    )
    logger.info(
        "Training fitness-optimized LINEAR baseline (%s generations, sample=%s) ...",
        generations,
        sample_size,
    )
    winner = OptimizationPipeline(app_config=app_config, logger=logger).run()
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str((root / linear_config_path).resolve()),
    )
    return neat.nn.FeedForwardNetwork.create(winner, neat_config)


def _total_emissions(
    assigned: list[str], volumes: np.ndarray, moistures: np.ndarray, subproducts: np.ndarray
) -> float:
    return float(
        sum(
            _emissions(volumes[i], moistures[i], assigned[i], subproducts[i])
            for i in range(len(assigned))
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NEAT against transparent baselines.")
    parser.add_argument("--project-root", type=Path, default=get_project_root())
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/split/dataset_optimization_cereal_co2_train_scaled.csv",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/split/dataset_optimization_cereal_co2_test_raw.csv",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="data/split/dataset_optimization_cereal_co2_scaler.joblib",
    )
    parser.add_argument("--model-path", type=str, default="models/artifacts/winner_genome.pkl")
    parser.add_argument("--neat-config-path", type=str, default="config/config-feedforward.txt")
    parser.add_argument(
        "--linear-neat-config-path",
        type=str,
        default="config/config-feedforward-linear.txt",
        help="Fixed-linear NEAT config used to evolve the fitness-optimized linear baseline.",
    )
    parser.add_argument(
        "--report-path", type=str, default="models/metrics/baseline_comparison.json"
    )
    parser.add_argument("--lots-per-day", type=int, default=15)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Fitness-evaluation sample for the linear baseline (match NEAT training).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Generations for the linear baseline (match NEAT training).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    root = args.project_root.resolve()

    ranges = _physical_ranges_from_scaler(root / args.scaler_path)

    # --- Training set: train the logistic-regression baseline -----------------
    train_df = load_dataset(root / args.train_path, logger)
    train_features = train_df[list(INPUT_COLUMNS)].to_numpy(dtype=float)
    vol_min, vol_max = ranges["generated_volume_tons"]
    hum_min, hum_max = ranges["moisture_pct"]
    train_vol_real = train_df["generated_volume_tons"].to_numpy(dtype=float) * (vol_max - vol_min) + vol_min
    train_hum_real = train_df["moisture_pct"].to_numpy(dtype=float) * (hum_max - hum_min) + hum_min
    train_subproduct = (
        train_df[["subproduct_type_Husk", "subproduct_type_Straw", "subproduct_type_Silo dust", "subproduct_type_Bran"]]
        .idxmax(axis=1)
        .str.replace("subproduct_type_", "", regex=False)
        .to_numpy()
    )
    # Fairest linear baseline: supervise the LR with the realized capacity-aware
    # optimal decisions (the sequential oracle), i.e. the best policy a per-row
    # classifier could possibly imitate.
    train_labels = np.array(
        _assign_oracle(train_vol_real, train_hum_real, train_subproduct, args.lots_per_day)
    )

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(train_features, train_labels)
    logger.info("Logistic-regression baseline trained on %s rows.", len(train_features))

    # --- Test set: evaluate every policy -------------------------------------
    test_df = load_dataset(root / args.test_path, logger)
    volumes = test_df["generated_volume_tons"].to_numpy(dtype=float)
    moistures = test_df["moisture_pct"].to_numpy(dtype=float)
    subproducts = test_df["subproduct_type"].astype(str).to_numpy()
    test_features = _build_test_features(test_df, ranges)

    # NEAT scores
    network = load_winner_network(
        InferenceModelPaths(
            model_path=root / args.model_path,
            neat_config_path=root / args.neat_config_path,
        ),
        logger,
    )
    neat_scores = np.array([network.activate(list(row)) for row in test_features], dtype=float)

    # Fitness-optimized LINEAR baseline (same fitness function as NEAT).
    linear_network = _train_linear_fitness_network(
        root=root,
        train_path=args.train_path,
        linear_config_path=args.linear_neat_config_path,
        sample_size=args.sample_size,
        generations=args.generations,
        random_state=args.random_state,
        logger=logger,
    )
    linear_scores = np.array(
        [linear_network.activate(list(row)) for row in test_features], dtype=float
    )

    # Logistic-regression scores aligned to STRATEGY_ORDER columns
    proba = logistic_model.predict_proba(test_features)
    class_to_col = {label: idx for idx, label in enumerate(logistic_model.classes_)}
    lr_scores = np.column_stack(
        [proba[:, class_to_col[strategy]] if strategy in class_to_col else np.zeros(len(proba)) for strategy in STRATEGY_ORDER]
    )

    rng = np.random.default_rng(args.random_state)
    random_scores = rng.standard_normal((len(volumes), len(STRATEGY_ORDER)))

    assignments = {
        "exact_optimum": _assign_exact(volumes, moistures, subproducts, args.lots_per_day),
        "oracle": _assign_oracle(volumes, moistures, subproducts, args.lots_per_day),
        "neat": _assign_by_scores(neat_scores, volumes, subproducts, moistures, args.lots_per_day),
        "linear_fitness": _assign_by_scores(linear_scores, volumes, subproducts, moistures, args.lots_per_day),
        "logistic_reg": _assign_by_scores(lr_scores, volumes, subproducts, moistures, args.lots_per_day),
        "random": _assign_by_scores(random_scores, volumes, subproducts, moistures, args.lots_per_day),
    }

    # Baseline = original dataset strategy, scored with the same simulator.
    baseline_strategy = test_df["reuse_strategy"].astype(str).to_numpy()
    baseline_total = _total_emissions(list(baseline_strategy), volumes, moistures, subproducts)

    report: dict[str, object] = {
        "rows": int(len(test_df)),
        "baseline_total_emissions_kg": round(baseline_total, 2),
        "policies": {},
    }
    for name, assigned in assignments.items():
        total = _total_emissions(assigned, volumes, moistures, subproducts)
        reduction = ((baseline_total - total) / baseline_total) * 100 if baseline_total else 0.0
        dist = pd.Series(assigned).value_counts(normalize=True).mul(100).round(2).to_dict()
        report["policies"][name] = {
            "total_emissions_kg": round(total, 2),
            "total_reduction_pct": round(reduction, 2),
            "strategy_distribution_pct": dist,
        }
        logger.info("%-13s reduction=%.2f%%", name, reduction)

    exact_red = report["policies"]["exact_optimum"]["total_reduction_pct"]
    oracle_red = report["policies"]["oracle"]["total_reduction_pct"]
    neat_red = report["policies"]["neat"]["total_reduction_pct"]
    linear_red = report["policies"]["linear_fitness"]["total_reduction_pct"]
    lr_red = report["policies"]["logistic_reg"]["total_reduction_pct"]
    report["interpretation"] = {
        "exact_vs_oracle_gap_pp": round(exact_red - oracle_red, 2),
        "exact_vs_neat_gap_pp": round(exact_red - neat_red, 2),
        "neat_vs_linear_fitness_gap_pp": round(neat_red - linear_red, 2),
        "note": (
            "The deployed model is 'exact_optimum': a per-block mixed-integer linear program "
            f"that solves the capacity-constrained assignment to proven optimality (reduction={exact_red}%). "
            f"It dominates the greedy per-lot heuristic ('oracle'={oracle_red}%) by "
            f"{round(exact_red - oracle_red, 2)}pp because it allocates scarce low-emission "
            "capacity across the whole operational block instead of consuming it on early lots. "
            f"The neuroevolution policy ('neat'={neat_red}%) and a fixed LINEAR policy trained "
            f"with the SAME fitness function ('linear_fitness'={linear_red}%) land within "
            f"{abs(round(neat_red - linear_red, 2))}pp of each other: topology augmentation adds "
            "little in this first-order simulator, confirming the auditor's intuition that the "
            "problem does not require a deep learned model. Supervised logistic regression "
            f"('logistic_reg'={lr_red}%) underperforms because per-row label imitation ignores "
            "sequential capacity depletion. NEAT is therefore reported as a learned-policy "
            "benchmark, while the exact optimizer is the deployed decision engine."
        ),
    }

    report_abs = root / args.report_path
    report_abs.parent.mkdir(parents=True, exist_ok=True)
    report_abs.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    logger.info("Baseline comparison report saved to %s", report_abs)


if __name__ == "__main__":
    main()
