"""Neuroevolution execution and genome evaluation logic."""

from __future__ import annotations

import logging
import random
import unicodedata
from typing import Any

import neat
import numpy as np
import pandas as pd

from src.config import AppConfig


class EvolutionRunner:
    """Encapsulates NEAT evolution and fitness evaluation.

    Args:
        app_config: Validated app configuration.
        features_df: Encoded feature matrix used as model inputs.
        evaluation_df: Sampled dataframe with baseline strategy and emissions columns.
        logger: Logger instance.
    """

    def __init__(
        self,
        app_config: AppConfig,
        features_df: pd.DataFrame,
        evaluation_df: pd.DataFrame,
        residue_labels: pd.Series,
        physical_ranges: dict[str, tuple[float, float]],
        logger: logging.Logger,
    ) -> None:
        self.app_config = app_config
        self.features_df = features_df.reset_index(drop=True)
        self.evaluation_df = evaluation_df.reset_index(drop=True)
        self.residue_labels = residue_labels.reset_index(drop=True)
        self.physical_ranges = physical_ranges
        self.logger = logger

        self.capacity_template = {
            "Animal feed": 90.0,
            "Composting": 140.0,
            "Biochar": 45.0,
            "Biomass combustion": 10000.0,
        }
        self.strategy_temperature_c = {
            "Animal feed": 60.0,
            "Composting": 60.0,
            "Biochar": 450.0,
            "Biomass combustion": 900.0,
        }
        self.lots_per_day = 15
        self.capacity_fallback_strategy = "Biomass combustion"
        self.capacity_fallback_count = 0
        # Per-generation best/mean fitness, populated during run() for auditable
        # training curves (auditor note: "no training metrics offered").
        self.fitness_history: list[dict[str, float]] = []

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize text to ASCII lowercase for robust strategy comparisons."""

        normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
        return normalized.strip().lower()

    @classmethod
    def _canonical_strategy(cls, raw_strategy: str) -> str:
        """Map strategy labels from datasets to canonical labels used in this pipeline."""

        strategy_map = {
            "combustion biomasa": "Biomass combustion",
            "biomass combustion": "Biomass combustion",
            "alimentacion animal": "Animal feed",
            "animal feed": "Animal feed",
            "compostaje": "Composting",
            "composting": "Composting",
            "biochar": "Biochar",
        }
        normalized = cls._normalize_text(raw_strategy)
        if normalized not in strategy_map:
            raise ValueError(f"Unknown strategy label: {raw_strategy}")
        return strategy_map[normalized]

    @staticmethod
    def _denormalize_minmax(value_norm: float, min_value: float, max_value: float) -> float:
        """Convert normalized value in [0, 1] back to physical units.

        Args:
            value_norm: Normalized value, expected in [0, 1].
            min_value: Lower bound from fitted scaler.
            max_value: Upper bound from fitted scaler.

        Returns:
            float: Value in the original physical scale.
        """

        normalized = float(np.clip(value_norm, 0.0, 1.0))
        return normalized * (max_value - min_value) + min_value

    def _calculate_simulated_co2(
        self,
        volume_norm: float,
        humidity_norm: float,
        selected_strategy: str,
        residue_type: str,
    ) -> float:
        """Simulate CO2 emissions for a selected strategy.

        Temperature is not an ML decision input. It is derived from the selected
        strategy and used only inside the physical emissions simulator.

        Args:
            volume_norm: Normalized volume in [0, 1].
            humidity_norm: Normalized humidity in [0, 1].
            selected_strategy: Chosen strategy by the neural policy.
            residue_type: Residue category for business rules.

        Returns:
            float: Simulated CO2 value.
        """

        volume_min, volume_max = self.physical_ranges["generated_volume_tons"]
        humidity_min, humidity_max = self.physical_ranges["moisture_pct"]

        vol_real = self._denormalize_minmax(volume_norm, volume_min, volume_max)
        hum_real = self._denormalize_minmax(humidity_norm, humidity_min, humidity_max)
        temp_real = self.strategy_temperature_c.get(selected_strategy, 180.0)
        emission_base = (temp_real * 0.5) * vol_real

        if selected_strategy == "Biomass combustion":
            return emission_base + ((hum_real**1.5) * 2.0) - 50.0
        if selected_strategy == "Animal feed":
            if "Husk" in residue_type or "Straw" in residue_type or hum_real > 18.0:
                return emission_base * 1.8
            return emission_base * 0.4
        if selected_strategy == "Biochar":
            if hum_real < 10.0:
                return emission_base * 0.2
            return emission_base * 1.2
        if selected_strategy == "Composting":
            return emission_base * 0.8 + (vol_real * 5.0)
        return emission_base

    def _reset_capacities(self) -> dict[str, float]:
        """Return a fresh copy of the daily capacity template."""

        return dict(self.capacity_template)

    def _select_feasible_strategy(
        self,
        scores: np.ndarray,
        volume_ton: float,
        current_capacities: dict[str, float],
    ) -> str:
        """Choose highest-scoring feasible strategy, or explicit fallback.

        If no strategy can satisfy the lot volume with remaining capacity,
        the method returns `self.capacity_fallback_strategy` and increments
        `self.capacity_fallback_count` for run-level traceability.
        """

        ranking = np.argsort(scores)[::-1]
        for strategy_index in ranking:
            strategy = self.app_config.evolution.strategies[int(strategy_index)]
            if volume_ton <= current_capacities.get(strategy, 0.0):
                current_capacities[strategy] -= volume_ton
                return strategy
        self.capacity_fallback_count += 1
        return self.capacity_fallback_strategy

    def _best_feasible_strategy(
        self,
        volume_norm: float,
        humidity_norm: float,
        residue_type: str,
        volume_ton: float,
        current_capacities: dict[str, float],
    ) -> tuple[str, float]:
        """Return the lowest-emission feasible strategy for the current row."""

        best_strategy = "Biomass combustion"
        best_emissions = float("inf")

        for strategy in self.app_config.evolution.strategies:
            if volume_ton > current_capacities.get(strategy, 0.0):
                continue

            emissions = self._calculate_simulated_co2(
                volume_norm=volume_norm,
                humidity_norm=humidity_norm,
                selected_strategy=strategy,
                residue_type=residue_type,
            )
            if emissions < best_emissions:
                best_emissions = emissions
                best_strategy = strategy

        if best_emissions == float("inf"):
            best_emissions = self._calculate_simulated_co2(
                volume_norm=volume_norm,
                humidity_norm=humidity_norm,
                selected_strategy="Biomass combustion",
                residue_type=residue_type,
            )
        return best_strategy, best_emissions

    def evaluate_genomes(self, genomes: list[tuple[int, Any]], config: Any) -> None:
        """Evaluate each genome and assign fitness.

        Args:
            genomes: List of `(genome_id, genome)` pairs from NEAT.
            config: Active NEAT config object.
        """

        features_np = self.features_df.to_numpy(dtype=float, copy=False)
        baseline_strategies = [
            self._canonical_strategy(str(strategy_value))
            for strategy_value in self.evaluation_df["reuse_strategy"].to_numpy()
        ]
        residue_values = self.residue_labels.astype(str).to_numpy()
        n_rows = len(features_np)

        volume_min, volume_max = self.physical_ranges["generated_volume_tons"]
        self.capacity_fallback_count = 0

        for genome_id, genome in genomes:
            try:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                weighted_reward_accumulated = 0.0
                current_capacities = self._reset_capacities()

                for row_idx in range(n_rows):
                    inputs = features_np[row_idx]
                    output = net.activate(inputs)
                    if row_idx % self.lots_per_day == 0:
                        current_capacities = self._reset_capacities()

                    capacities_before_decision = dict(current_capacities)

                    volume_ton_real = self._denormalize_minmax(
                        value_norm=float(inputs[0]),
                        min_value=volume_min,
                        max_value=volume_max,
                    )

                    # Training policy: NEAT scores are the primary decision signal,
                    # then capacity feasibility is enforced over that ranking.
                    selected_strategy = self._select_feasible_strategy(
                        scores=np.asarray(output, dtype=float),
                        volume_ton=volume_ton_real,
                        current_capacities=current_capacities,
                    )

                    baseline_strategy = baseline_strategies[row_idx]
                    residue = residue_values[row_idx]
                    baseline_emissions = self._calculate_simulated_co2(
                        volume_norm=float(inputs[0]),
                        humidity_norm=float(inputs[1]),
                        selected_strategy=baseline_strategy,
                        residue_type=residue,
                    )

                    _, best_feasible_emissions = self._best_feasible_strategy(
                        volume_norm=float(inputs[0]),
                        humidity_norm=float(inputs[1]),
                        residue_type=residue,
                        volume_ton=volume_ton_real,
                        current_capacities=capacities_before_decision,
                    )

                    co2_emitted = self._calculate_simulated_co2(
                        volume_norm=float(inputs[0]),
                        humidity_norm=float(inputs[1]),
                        selected_strategy=selected_strategy,
                        residue_type=residue,
                    )

                    delta_vs_baseline = baseline_emissions - co2_emitted
                    delta_vs_best = best_feasible_emissions - co2_emitted

                    if delta_vs_baseline > 0:
                        weighted_reward_accumulated += delta_vs_baseline
                    else:
                        weighted_reward_accumulated += delta_vs_baseline * 2.0

                    if delta_vs_best >= 0:
                        weighted_reward_accumulated += delta_vs_best * 0.5
                    else:
                        weighted_reward_accumulated += delta_vs_best * 3.0

                genome.fitness = weighted_reward_accumulated / n_rows
            except Exception as exc:  # pragma: no cover - defensive branch
                self.logger.exception(
                    "Error while evaluating genome_id=%s. Penalizing fitness.",
                    genome_id,
                )
                genome.fitness = -1e12
                self.logger.debug("Evaluation error details: %s", exc)

        if self.capacity_fallback_count > 0:
            self.logger.warning(
                "Capacity fallback applied %s times during genome evaluation. "
                "Fallback strategy=%s.",
                self.capacity_fallback_count,
                self.capacity_fallback_strategy,
            )

    def run(self) -> Any:
        """Execute NEAT neuroevolution and return the winning genome.

        Returns:
            Any: Winning genome object.
        """

        self.logger.info(
            "Starting neuroevolution for %s generations with sample_size=%s",
            self.app_config.evolution.generations,
            self.app_config.evolution.sample_size,
        )

        random.seed(self.app_config.evolution.random_state)
        np.random.seed(self.app_config.evolution.random_state)

        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(self.app_config.paths.neat_config_path),
        )

        population = neat.Population(neat_config)
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        winner_genome = population.run(
            self.evaluate_genomes,
            self.app_config.evolution.generations,
        )

        self._record_fitness_history(stats)
        self.logger.info("Evolution finished successfully.")
        return winner_genome

    def _record_fitness_history(self, stats: "neat.StatisticsReporter") -> None:
        """Capture per-generation training curves for auditability.

        For each generation we store best/mean/std fitness plus the structural
        size (active connections and hidden nodes) of the best genome, so the
        compactness of the final architecture can be inspected over time.
        """

        try:
            mean_fitness = stats.get_fitness_mean()
            stdev_fitness = stats.get_fitness_stdev()
        except Exception:  # pragma: no cover - defensive against neat internals
            mean_fitness, stdev_fitness = [], []

        history: list[dict[str, float]] = []
        for generation, best_genome in enumerate(stats.most_fit_genomes):
            enabled_connections = sum(
                1 for conn in best_genome.connections.values() if conn.enabled
            )
            # In neat-python, output node keys are 0..num_outputs-1 and hidden
            # node keys are >= num_outputs; input nodes are implicit (negative).
            output_keys = set(range(len(self.app_config.evolution.strategies)))
            hidden_nodes = sum(
                1 for node_key in best_genome.nodes if node_key not in output_keys
            )
            history.append(
                {
                    "generation": int(generation),
                    "best_fitness": float(best_genome.fitness)
                    if best_genome.fitness is not None
                    else float("nan"),
                    "mean_fitness": float(mean_fitness[generation])
                    if generation < len(mean_fitness)
                    else float("nan"),
                    "std_fitness": float(stdev_fitness[generation])
                    if generation < len(stdev_fitness)
                    else float("nan"),
                    "best_enabled_connections": int(enabled_connections),
                    "best_hidden_nodes": int(hidden_nodes),
                }
            )
        self.fitness_history = history
