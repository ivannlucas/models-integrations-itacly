"""Offline neuroevolution for tabular regression on the synthetic CU04 environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0, ddof=0)
    if np.isscalar(std):
        return np.array(std if float(std) > 0 else 1.0, dtype=float)
    std = np.asarray(std, dtype=float)
    std[std == 0.0] = 1.0
    return std


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


@dataclass
class _PopulationMember:
    hidden_units: int
    input_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float


class NeuroevolutionRegressor:
    """Simple evolutionary search over a small one-hidden-layer regressor."""

    requires_validation_data = True

    def __init__(
        self,
        *,
        random_state: int = 42,
        population_size: int = 36,
        generations: int = 24,
        elite_size: int = 4,
        tournament_size: int = 3,
        hidden_layer_options: list[int] | None = None,
        mutation_rate: float = 0.18,
        mutation_scale: float = 0.18,
        crossover_rate: float = 0.75,
        architecture_mutation_rate: float = 0.1,
        train_fitness_weight: float = 0.7,
        validation_fitness_weight: float = 0.3,
        complexity_penalty_weight: float = 0.01,
        input_clip_sigma: float = 4.0,
        stagnation_patience: int = 8,
        log_every_generations: int = 5,
    ) -> None:
        self.random_state = int(random_state)
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.elite_size = int(elite_size)
        self.tournament_size = int(tournament_size)
        self.hidden_layer_options = sorted({int(option) for option in (hidden_layer_options or [4, 8, 12]) if int(option) > 0})
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)
        self.crossover_rate = float(crossover_rate)
        self.architecture_mutation_rate = float(architecture_mutation_rate)
        self.train_fitness_weight = float(train_fitness_weight)
        self.validation_fitness_weight = float(validation_fitness_weight)
        self.complexity_penalty_weight = float(complexity_penalty_weight)
        self.input_clip_sigma = float(input_clip_sigma)
        self.stagnation_patience = int(stagnation_patience)
        self.log_every_generations = int(log_every_generations)

        if not self.hidden_layer_options:
            raise ValueError("hidden_layer_options must contain at least one positive integer.")
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4.")
        if self.elite_size < 1 or self.elite_size >= self.population_size:
            raise ValueError("elite_size must be >= 1 and < population_size.")

    def fit(
        self,
        x_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        *,
        validation_data: tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray] | None = None,
        logger=None,
    ) -> "NeuroevolutionRegressor":
        x_train_np = self._as_2d_array(x_train)
        y_train_np = self._as_1d_array(y_train)
        if len(x_train_np) != len(y_train_np):
            raise ValueError("x_train and y_train must have the same number of rows.")
        if validation_data is None:
            x_valid_np = x_train_np.copy()
            y_valid_np = y_train_np.copy()
        else:
            x_valid_np = self._as_2d_array(validation_data[0])
            y_valid_np = self._as_1d_array(validation_data[1])
            if len(x_valid_np) != len(y_valid_np):
                raise ValueError("validation features and target must have the same number of rows.")

        self.feature_count_ = int(x_train_np.shape[1])
        self.max_hidden_units_ = max(self.hidden_layer_options)
        self.feature_mean_ = x_train_np.mean(axis=0)
        self.feature_std_ = _safe_std(x_train_np)
        self.target_mean_ = float(y_train_np.mean())
        self.target_std_ = float(_safe_std(y_train_np))

        x_train_scaled = self._transform_features(x_train_np)
        x_valid_scaled = self._transform_features(x_valid_np)
        y_train_scaled = self._transform_target(y_train_np)
        y_valid_scaled = self._transform_target(y_valid_np)

        rng = np.random.default_rng(self.random_state)
        population = [self._initialise_member(rng, self.feature_count_) for _ in range(self.population_size)]

        best_member: _PopulationMember | None = None
        best_report: dict[str, float] | None = None
        best_generation = 0
        generations_without_improvement = 0
        history: list[dict[str, float | int | bool]] = []

        for generation in range(self.generations):
            evaluated_population = [
                (
                    self._evaluate_member(
                        member,
                        x_train_scaled,
                        y_train_scaled,
                        x_valid_scaled,
                        y_valid_scaled,
                    ),
                    member,
                )
                for member in population
            ]
            evaluated_population.sort(key=lambda item: item[0]["fitness"], reverse=True)

            generation_best_report, generation_best_member = evaluated_population[0]
            average_fitness = float(np.mean([item[0]["fitness"] for item in evaluated_population]))
            improved = best_report is None or generation_best_report["fitness"] > best_report["fitness"]
            if improved:
                best_report = generation_best_report.copy()
                best_member = self._clone_member(generation_best_member)
                best_generation = generation
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            history.append(
                {
                    "generation": int(generation),
                    "best_fitness": float(generation_best_report["fitness"]),
                    "avg_fitness": average_fitness,
                    "best_train_rmse_scaled": float(generation_best_report["train_rmse_scaled"]),
                    "best_valid_rmse_scaled": float(generation_best_report["valid_rmse_scaled"]),
                    "best_hidden_units": int(generation_best_report["hidden_units"]),
                    "improved_global_best": bool(improved),
                }
            )

            if logger and (
                generation == 0
                or (generation + 1) % max(1, self.log_every_generations) == 0
                or generation == self.generations - 1
                or generations_without_improvement >= self.stagnation_patience
            ):
                logger.info(
                    "Neuroevolution generation=%s/%s best_fitness=%.6f valid_rmse_scaled=%.6f hidden_units=%s",
                    generation + 1,
                    self.generations,
                    generation_best_report["fitness"],
                    generation_best_report["valid_rmse_scaled"],
                    generation_best_report["hidden_units"],
                )

            if generations_without_improvement >= self.stagnation_patience:
                break

            elites = [self._clone_member(member) for _, member in evaluated_population[: self.elite_size]]
            breeding_pool = evaluated_population[: max(self.elite_size, self.population_size // 2)]
            next_population = elites.copy()
            while len(next_population) < self.population_size:
                parent_a = self._tournament_select(breeding_pool, rng)
                parent_b = self._tournament_select(breeding_pool, rng)
                if rng.random() < self.crossover_rate:
                    child = self._crossover(parent_a, parent_b, rng)
                else:
                    child = self._clone_member(parent_a if rng.random() < 0.5 else parent_b)
                child = self._mutate(child, rng)
                next_population.append(child)
            population = next_population[: self.population_size]

        if best_member is None or best_report is None:
            raise RuntimeError("Neuroevolution fit did not produce a valid individual.")

        self.best_member_ = best_member
        self.best_generation_ = int(best_generation)
        self.evolution_history_ = history
        self.best_genome_summary_ = {
            "hidden_units": int(best_member.hidden_units),
            "input_weight_l2": float(np.linalg.norm(best_member.input_weights[: best_member.hidden_units])),
            "output_weight_l2": float(np.linalg.norm(best_member.output_weights[: best_member.hidden_units])),
            "output_bias": float(best_member.output_bias),
        }
        self.training_metadata_ = {
            "algorithm": "evolutionary_mlp_regression_v1",
            "random_state": self.random_state,
            "population_size": self.population_size,
            "generations_requested": self.generations,
            "generations_completed": len(history),
            "elite_size": self.elite_size,
            "tournament_size": self.tournament_size,
            "hidden_layer_options": self.hidden_layer_options,
            "train_rows": int(len(x_train_np)),
            "validation_rows": int(len(x_valid_np)),
            "feature_count": self.feature_count_,
            "best_generation": self.best_generation_,
            "best_fitness": float(best_report["fitness"]),
            "best_train_rmse_scaled": float(best_report["train_rmse_scaled"]),
            "best_valid_rmse_scaled": float(best_report["valid_rmse_scaled"]),
            "best_hidden_units": int(best_report["hidden_units"]),
            "stopped_early": len(history) < self.generations,
            "train_fitness_weight": self.train_fitness_weight,
            "validation_fitness_weight": self.validation_fitness_weight,
            "complexity_penalty_weight": self.complexity_penalty_weight,
            "mutation_rate": self.mutation_rate,
            "mutation_scale": self.mutation_scale,
            "crossover_rate": self.crossover_rate,
            "architecture_mutation_rate": self.architecture_mutation_rate,
            "input_clip_sigma": self.input_clip_sigma,
        }
        return self

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not hasattr(self, "best_member_"):
            raise RuntimeError("The neuroevolution model has not been fitted yet.")
        x_np = self._as_2d_array(x)
        x_scaled = self._transform_features(x_np)
        prediction_scaled = self._forward(self.best_member_, x_scaled)
        return (prediction_scaled * self.target_std_) + self.target_mean_

    @staticmethod
    def _as_2d_array(values: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            array = values.to_numpy(dtype=float)
        else:
            array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array

    @staticmethod
    def _as_1d_array(values: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(values, pd.Series):
            array = values.to_numpy(dtype=float)
        else:
            array = np.asarray(values, dtype=float)
        return array.reshape(-1)

    def _transform_features(self, x: np.ndarray) -> np.ndarray:
        standardized = (x - self.feature_mean_) / self.feature_std_
        return np.clip(standardized, -self.input_clip_sigma, self.input_clip_sigma)

    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean_) / self.target_std_

    def _initialise_member(self, rng: np.random.Generator, input_dim: int) -> _PopulationMember:
        hidden_units = int(rng.choice(self.hidden_layer_options))
        weight_scale = 1.0 / max(1.0, np.sqrt(float(input_dim)))
        return _PopulationMember(
            hidden_units=hidden_units,
            input_weights=rng.normal(0.0, weight_scale, size=(self.max_hidden_units_, input_dim)),
            hidden_bias=rng.normal(0.0, weight_scale, size=self.max_hidden_units_),
            output_weights=rng.normal(0.0, weight_scale, size=self.max_hidden_units_),
            output_bias=float(rng.normal(0.0, weight_scale)),
        )

    def _clone_member(self, member: _PopulationMember) -> _PopulationMember:
        return _PopulationMember(
            hidden_units=int(member.hidden_units),
            input_weights=member.input_weights.copy(),
            hidden_bias=member.hidden_bias.copy(),
            output_weights=member.output_weights.copy(),
            output_bias=float(member.output_bias),
        )

    def _forward(self, member: _PopulationMember, x_scaled: np.ndarray) -> np.ndarray:
        hidden_count = member.hidden_units
        hidden_linear = (x_scaled @ member.input_weights[:hidden_count].T) + member.hidden_bias[:hidden_count]
        hidden_activation = np.tanh(hidden_linear)
        return (hidden_activation @ member.output_weights[:hidden_count]) + member.output_bias

    def _evaluate_member(
        self,
        member: _PopulationMember,
        x_train_scaled: np.ndarray,
        y_train_scaled: np.ndarray,
        x_valid_scaled: np.ndarray,
        y_valid_scaled: np.ndarray,
    ) -> dict[str, float]:
        train_pred_scaled = self._forward(member, x_train_scaled)
        valid_pred_scaled = self._forward(member, x_valid_scaled)
        train_rmse_scaled = _rmse(y_train_scaled, train_pred_scaled)
        valid_rmse_scaled = _rmse(y_valid_scaled, valid_pred_scaled)
        complexity_penalty = self.complexity_penalty_weight * (member.hidden_units / self.max_hidden_units_)
        objective = (
            self.train_fitness_weight * train_rmse_scaled
            + self.validation_fitness_weight * valid_rmse_scaled
            + complexity_penalty
        )
        return {
            "fitness": float(-objective),
            "train_rmse_scaled": float(train_rmse_scaled),
            "valid_rmse_scaled": float(valid_rmse_scaled),
            "hidden_units": float(member.hidden_units),
        }

    def _tournament_select(
        self,
        evaluated_population: list[tuple[dict[str, float], _PopulationMember]],
        rng: np.random.Generator,
    ) -> _PopulationMember:
        tournament_size = min(self.tournament_size, len(evaluated_population))
        selected_indices = rng.integers(0, len(evaluated_population), size=tournament_size)
        selected = [evaluated_population[int(index)] for index in selected_indices]
        selected.sort(key=lambda item: item[0]["fitness"], reverse=True)
        return selected[0][1]

    def _crossover(
        self,
        parent_a: _PopulationMember,
        parent_b: _PopulationMember,
        rng: np.random.Generator,
    ) -> _PopulationMember:
        input_mask = rng.random(parent_a.input_weights.shape) < 0.5
        bias_mask = rng.random(parent_a.hidden_bias.shape) < 0.5
        output_mask = rng.random(parent_a.output_weights.shape) < 0.5
        child_hidden_units = int(parent_a.hidden_units if rng.random() < 0.5 else parent_b.hidden_units)
        child_output_bias = float(parent_a.output_bias if rng.random() < 0.5 else parent_b.output_bias)
        return _PopulationMember(
            hidden_units=child_hidden_units,
            input_weights=np.where(input_mask, parent_a.input_weights, parent_b.input_weights),
            hidden_bias=np.where(bias_mask, parent_a.hidden_bias, parent_b.hidden_bias),
            output_weights=np.where(output_mask, parent_a.output_weights, parent_b.output_weights),
            output_bias=child_output_bias,
        )

    def _mutate(self, member: _PopulationMember, rng: np.random.Generator) -> _PopulationMember:
        mutated = self._clone_member(member)
        if rng.random() < self.architecture_mutation_rate and len(self.hidden_layer_options) > 1:
            current_index = self.hidden_layer_options.index(mutated.hidden_units)
            move = int(rng.choice([-1, 1]))
            next_index = int(np.clip(current_index + move, 0, len(self.hidden_layer_options) - 1))
            mutated.hidden_units = int(self.hidden_layer_options[next_index])

        input_mutation_mask = rng.random(mutated.input_weights.shape) < self.mutation_rate
        hidden_bias_mask = rng.random(mutated.hidden_bias.shape) < self.mutation_rate
        output_mutation_mask = rng.random(mutated.output_weights.shape) < self.mutation_rate

        mutated.input_weights = mutated.input_weights + (
            rng.normal(0.0, self.mutation_scale, size=mutated.input_weights.shape) * input_mutation_mask
        )
        mutated.hidden_bias = mutated.hidden_bias + (
            rng.normal(0.0, self.mutation_scale, size=mutated.hidden_bias.shape) * hidden_bias_mask
        )
        mutated.output_weights = mutated.output_weights + (
            rng.normal(0.0, self.mutation_scale, size=mutated.output_weights.shape) * output_mutation_mask
        )
        if rng.random() < self.mutation_rate:
            mutated.output_bias = float(mutated.output_bias + rng.normal(0.0, self.mutation_scale))
        return mutated
