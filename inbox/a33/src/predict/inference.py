"""Inference engine that mirrors notebook behavior with pickle-loaded NEAT genome."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import neat
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from src.utils.artifacts import validate_model_metadata_for_inference
from src.utils.utils import load_dataset


POLICY_INPUT_COLUMNS: tuple[str, ...] = (
    "generated_volume_tons",
    "moisture_pct",
    "subproduct_type_Husk",
    "subproduct_type_Straw",
    "subproduct_type_Silo dust",
    "subproduct_type_Bran",
    "season_Rainy",
    "season_Dry",
)

INFERENCE_STRATEGY_ORDER: tuple[str, ...] = (
    "Biomass combustion",
    "Animal feed",
    "Composting",
    "Biochar",
)


class InferenceInput(BaseModel):
    """Validated input schema for one inference sample.

    Attributes:
        generated_volume_tons: Batch volume in tons.
        moisture_pct: Moisture percentage.
        process_temperature_c: Optional measured temperature (not used by ML policy).
        subproduct_type: Residue type category.
        season: Seasonal category.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    generated_volume_tons: float = Field(gt=0)
    moisture_pct: float = Field(ge=0)
    process_temperature_c: float | None = None
    subproduct_type: str
    season: str


class PlantCapacities(BaseModel):
    """Daily available capacities per strategy.

    Attributes:
        animal_feed: Capacity for animal feed strategy.
        composting: Capacity for composting strategy.
        biochar: Capacity for biochar strategy.
        biomass_combustion: Capacity for biomass combustion strategy.
    """

    animal_feed: float = Field(gt=0)
    composting: float = Field(gt=0)
    biochar: float = Field(gt=0)
    biomass_combustion: float = Field(gt=0)

    def as_strategy_dict(self) -> dict[str, float]:
        """Return capacities with strategy labels used by inference.

        Returns:
            dict[str, float]: Strategy to remaining capacity mapping.
        """

        return {
            "Animal feed": float(self.animal_feed),
            "Composting": float(self.composting),
            "Biochar": float(self.biochar),
            "Biomass combustion": float(self.biomass_combustion),
        }


class InferenceRuntimeConfig(BaseModel):
    """Runtime settings for batch inference.

    Attributes:
        lots_per_day: Number of processed lots before resetting capacities.
        fallback_strategy: Strategy selected if no capacity can serve the lot.
    """

    lots_per_day: PositiveInt = 15
    fallback_strategy: str = "Biomass combustion"


@dataclass(frozen=True)
class InferenceArtifacts:
    """Artifacts produced by a full inference run.

    Attributes:
        output_path: CSV file written to disk.
        assigned_strategies: Strategy series assigned by the optimizer.
        distribution: Count and percentage summary of assigned strategies.
    """

    output_path: Path
    assigned_strategies: pd.Series
    distribution: dict[str, Any]
    runtime_metrics: dict[str, Any]


class InferenceModelPaths(BaseModel):
    """Paths required to load the winner genome and NEAT config.

    Attributes:
        model_path: Path to winner genome pickle.
        neat_config_path: Path to NEAT config file.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    model_path: Path
    neat_config_path: Path


class RealisticOptimizer:
    """Inference model that combines neural preference and capacity constraints.

    Args:
        capacities_template: Base daily capacities per strategy.
        runtime_config: Runtime parameters for batch processing.
        logger: Logger instance.
    """

    def __init__(
        self,
        capacities_template: PlantCapacities,
        runtime_config: InferenceRuntimeConfig,
        physical_ranges: dict[str, tuple[float, float]],
        network: neat.nn.FeedForwardNetwork,
        logger: logging.Logger,
    ) -> None:
        self.capacities_template = capacities_template
        self.runtime_config = runtime_config
        self.physical_ranges = physical_ranges
        self.network = network
        self.logger = logger
        self.capacity_fallback_count = 0
        self.row_error_fallback_count = 0


    ASSIGNMENT_SOURCE_OPTIMIZED = "feasible_min_emissions"
    ASSIGNMENT_SOURCE_CAPACITY_FALLBACK = "capacity_fallback"
    ASSIGNMENT_SOURCE_ERROR_FALLBACK = "error_fallback"

    @staticmethod
    def _normalize_minmax(value_real: float, min_value: float, max_value: float) -> float:
        """Normalize a physical value with min-max and clip to [0, 1].

        Args:
            value_real: Raw value in physical units.
            min_value: Lower scaler bound.
            max_value: Upper scaler bound.

        Returns:
            float: Normalized value.
        """

        denominator = max_value - min_value
        if denominator <= 0:
            raise ValueError(
                f"Invalid normalization bounds: min={min_value}, max={max_value}"
            )
        value_norm = (value_real - min_value) / denominator
        return max(0.0, min(1.0, float(value_norm)))

    def _neural_preferences(self, sample: InferenceInput) -> dict[str, float]:
        """Compute strategy preferences with the winner NEAT network.

        Args:
            sample: Validated inference input.

        Returns:
            dict[str, float]: Strategy scores sorted later by confidence.
        """

        volume_min, volume_max = self.physical_ranges["generated_volume_tons"]
        humidity_min, humidity_max = self.physical_ranges["moisture_pct"]

        vol_norm = self._normalize_minmax(sample.generated_volume_tons, volume_min, volume_max)
        hum_norm = self._normalize_minmax(sample.moisture_pct, humidity_min, humidity_max)

        is_rainy = 1 if sample.season == "Rainy" else 0
        is_dry = 1 if sample.season == "Dry" else 0
        is_husk = 1 if sample.subproduct_type == "Husk" else 0
        is_straw = 1 if sample.subproduct_type == "Straw" else 0
        is_silo_powder = 1 if sample.subproduct_type == "Silo dust" else 0
        is_bran = 1 if sample.subproduct_type == "Bran" else 0

        # Keep exact feature order expected by NEAT config.
        # Temperature is intentionally excluded from policy inputs because
        # it is determined after strategy selection.
        inputs = [
            vol_norm,
            hum_norm,
            is_husk,
            is_straw,
            is_silo_powder,
            is_bran,
            is_rainy,
            is_dry,
        ]
        output = self.network.activate(inputs)

        return {
            "Biomass combustion": float(output[0]),
            "Animal feed": float(output[1]),
            "Composting": float(output[2]),
            "Biochar": float(output[3]),
        }

    @staticmethod
    def _estimated_emissions_for_strategy(
        sample: InferenceInput,
        strategy: str,
    ) -> float:
        """Estimate emissions for a strategy using the same business physics as evaluation."""

        strategy_temperature_c = {
            "Animal feed": 60.0,
            "Composting": 60.0,
            "Biochar": 450.0,
            "Biomass combustion": 900.0,
        }
        temperature_c = strategy_temperature_c.get(strategy, 180.0)
        emission_base = (temperature_c * 0.5) * sample.generated_volume_tons

        if strategy == "Biomass combustion":
            humidity_term = (sample.moisture_pct**1.5) * 2.0
            return emission_base + humidity_term - 50.0
        if strategy == "Animal feed":
            if sample.subproduct_type in {"Husk", "Straw", "Silo dust"} or sample.moisture_pct > 18.0:
                return emission_base * 1.8
            return emission_base * 0.4
        if strategy == "Biochar":
            if sample.moisture_pct < 10.0:
                return emission_base * 0.2
            return emission_base * 1.2
        if strategy == "Composting":
            return emission_base * 0.8 + (sample.generated_volume_tons * 5.0)
        return emission_base

    def choose_strategy(
        self,
        sample: InferenceInput,
        current_capacities: dict[str, float],
        row_index: int | None = None,
    ) -> tuple[str, str]:
        """Select a feasible strategy and update remaining capacities.

        In inference, the final decision is deterministic by minimum feasible
        emissions. The NEAT network score acts as a secondary tie-breaker
        among feasible strategies. This differs from training, where neural
        scores are used directly as the primary policy signal.

        Args:
            sample: Validated sample input.
            current_capacities: Mutable capacities map for current day.
            row_index: Optional row index used for trace logs.

        Returns:
            tuple[str, str]: Chosen strategy and assignment source.
        """

        ranking = sorted(
            self._neural_preferences(sample).items(),
            key=lambda item: item[1],
            reverse=True,
        )

        feasible_candidates: list[tuple[float, float, str]] = []
        for preference_rank, (strategy, score) in enumerate(ranking):
            if sample.generated_volume_tons > current_capacities.get(strategy, 0.0):
                continue
            emissions = self._estimated_emissions_for_strategy(sample, strategy)
            feasible_candidates.append((emissions, -score, strategy))

        if feasible_candidates:
            _, _, chosen_strategy = min(feasible_candidates)
            current_capacities[chosen_strategy] -= sample.generated_volume_tons
            return chosen_strategy, self.ASSIGNMENT_SOURCE_OPTIMIZED

        self.capacity_fallback_count += 1
        self.logger.warning(
            "Capacity fallback triggered at row=%s. Using strategy=%s.",
            row_index,
            self.runtime_config.fallback_strategy,
        )
        return self.runtime_config.fallback_strategy, self.ASSIGNMENT_SOURCE_CAPACITY_FALLBACK

    def infer_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Run constrained inference for a full dataframe.

        Args:
            dataframe: Input dataframe with required columns.

        Returns:
            pd.DataFrame: Assigned strategies and traceability columns for each row.

        Raises:
            ValueError: If required columns are missing.
        """

        required_columns = {
            "generated_volume_tons",
            "moisture_pct",
            "subproduct_type",
            "season",
        }
        missing = required_columns.difference(dataframe.columns)
        if missing:
            raise ValueError(f"Missing required columns for inference: {sorted(missing)}")

        assigned_strategies: list[str] = []
        assignment_sources: list[str] = []
        capacities = self.capacities_template.as_strategy_dict()

        for index, row in enumerate(dataframe.itertuples(index=False)):
            if index % self.runtime_config.lots_per_day == 0:
                capacities = self.capacities_template.as_strategy_dict()

            try:
                sample = InferenceInput(
                    generated_volume_tons=float(getattr(row, "generated_volume_tons")),
                    moisture_pct=float(getattr(row, "moisture_pct")),
                    process_temperature_c=float(getattr(row, "process_temperature_c"))
                    if hasattr(row, "process_temperature_c")
                    else None,
                    subproduct_type=str(getattr(row, "subproduct_type")),
                    season=str(getattr(row, "season")),
                )
                strategy, assignment_source = self.choose_strategy(
                    sample=sample,
                    current_capacities=capacities,
                    row_index=index,
                )
                assigned_strategies.append(strategy)
                assignment_sources.append(assignment_source)
            except Exception as exc:  # pragma: no cover - defensive path
                self.row_error_fallback_count += 1
                self.logger.exception(
                    "Inference failed for row index=%s. Applying default strategy=%s.",
                    index,
                    self.runtime_config.fallback_strategy,
                )
                self.logger.debug("Inference row error details: %s", exc)
                assigned_strategies.append(self.runtime_config.fallback_strategy)
                assignment_sources.append(self.ASSIGNMENT_SOURCE_ERROR_FALLBACK)

        if self.capacity_fallback_count > 0:
            self.logger.warning(
                "Inference capacity fallback count=%s (strategy=%s).",
                self.capacity_fallback_count,
                self.runtime_config.fallback_strategy,
            )
        if self.row_error_fallback_count > 0:
            self.logger.warning(
                "Inference row-error fallback count=%s (strategy=%s).",
                self.row_error_fallback_count,
                self.runtime_config.fallback_strategy,
            )

        trace_df = pd.DataFrame(
            {
                "ai_assigned_strategy": assigned_strategies,
                "ai_assignment_source": assignment_sources,
            }
        )
        trace_df["ai_is_fallback"] = trace_df["ai_assignment_source"].isin(
            {
                self.ASSIGNMENT_SOURCE_CAPACITY_FALLBACK,
                self.ASSIGNMENT_SOURCE_ERROR_FALLBACK,
            }
        )
        return trace_df


def summarize_assignment_trace(trace_df: pd.DataFrame) -> dict[str, Any]:
    """Build explicit assignment-source metrics for auditability.

    Args:
        trace_df: Dataframe returned by infer_dataframe.

    Returns:
        dict[str, Any]: Counts and impact percentages for fallback sources.
    """

    total_rows = int(len(trace_df))
    source_counts = trace_df["ai_assignment_source"].value_counts(dropna=False)
    source_percentages = (
        trace_df["ai_assignment_source"].value_counts(normalize=True, dropna=False).mul(100).round(2)
    )

    biomass_mask = trace_df["ai_assigned_strategy"] == "Biomass combustion"
    biomass_total = int(biomass_mask.sum())
    biomass_from_capacity_fallback = int(
        (
            biomass_mask
            & (
                trace_df["ai_assignment_source"]
                == RealisticOptimizer.ASSIGNMENT_SOURCE_CAPACITY_FALLBACK
            )
        ).sum()
    )
    biomass_from_error_fallback = int(
        (
            biomass_mask
            & (
                trace_df["ai_assignment_source"]
                == RealisticOptimizer.ASSIGNMENT_SOURCE_ERROR_FALLBACK
            )
        ).sum()
    )
    biomass_from_optimized_decision = int(
        (
            biomass_mask
            & (
                trace_df["ai_assignment_source"]
                == RealisticOptimizer.ASSIGNMENT_SOURCE_OPTIMIZED
            )
        ).sum()
    )

    total_fallback_count = biomass_from_capacity_fallback + biomass_from_error_fallback
    total_fallback_pct = round((total_fallback_count / total_rows) * 100, 2) if total_rows else 0.0
    biomass_fallback_share_pct = (
        round((total_fallback_count / biomass_total) * 100, 2) if biomass_total else 0.0
    )

    return {
        "total_rows": total_rows,
        "assignment_source_counts": source_counts.to_dict(),
        "assignment_source_percentages": source_percentages.to_dict(),
        "capacity_fallback_count": biomass_from_capacity_fallback,
        "row_error_fallback_count": biomass_from_error_fallback,
        "total_fallback_count": total_fallback_count,
        "total_fallback_pct": total_fallback_pct,
        "biomass_combustion_total_count": biomass_total,
        "biomass_combustion_from_capacity_fallback_count": biomass_from_capacity_fallback,
        "biomass_combustion_from_error_fallback_count": biomass_from_error_fallback,
        "biomass_combustion_from_optimized_decision_count": biomass_from_optimized_decision,
        "biomass_combustion_fallback_share_pct": biomass_fallback_share_pct,
    }


def load_winner_network(paths: InferenceModelPaths, logger: logging.Logger) -> neat.nn.FeedForwardNetwork:
    """Load winner genome pickle and build NEAT feed-forward network.

    Args:
        paths: Model and config file paths.
        logger: Logger instance.

    Returns:
        neat.nn.FeedForwardNetwork: Reconstructed winner network.
    """

    if not paths.model_path.exists():
        raise FileNotFoundError(f"Winner genome pickle not found: {paths.model_path}")
    if not paths.neat_config_path.exists():
        raise FileNotFoundError(f"NEAT config not found: {paths.neat_config_path}")

    logger.info("Loading winner genome from %s", paths.model_path)
    with paths.model_path.open("rb") as file_obj:
        winner_genome = pickle.load(file_obj)

    logger.info("Loading NEAT config from %s", paths.neat_config_path)
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(paths.neat_config_path),
    )
    return neat.nn.FeedForwardNetwork.create(winner_genome, neat_config)


def _resolve_scaler_path_for_dataset(dataset_path: Path) -> Path:
    """Resolve scaler artifact path from an inference dataset path.

    Args:
        dataset_path: Inference dataset path.

    Returns:
        Path: Resolved scaler artifact path.

    Raises:
        ValueError: If dataset name does not match expected split naming pattern.
        FileNotFoundError: If scaler path cannot be found.
    """

    dataset_name = dataset_path.name
    expected_suffixes = (
        "_train_raw.csv",
        "_test_raw.csv",
        "_train_scaled.csv",
        "_test_scaled.csv",
    )

    prefix = None
    for suffix in expected_suffixes:
        if dataset_name.endswith(suffix):
            prefix = dataset_name[: -len(suffix)]
            break

    if prefix is None:
        raise ValueError(
            "Cannot infer scaler path from dataset name. "
            "Expected suffix '_train_raw.csv', '_test_raw.csv', '_train_scaled.csv' or '_test_scaled.csv'."
        )

    scaler_path = dataset_path.parent / f"{prefix}_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")
    return scaler_path


def load_physical_ranges_from_scaler(scaler_path: Path, logger: logging.Logger) -> dict[str, tuple[float, float]]:
    """Load physical min/max ranges from an explicit preprocessing scaler file.

    Args:
        scaler_path: Path to the fitted MinMaxScaler joblib artifact.
        logger: Logger instance.

    Returns:
        dict[str, tuple[float, float]]: Physical ranges for volume, humidity and temperature.
    """

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")
    logger.info("Loading preprocessing scaler from %s", scaler_path)
    scaler = joblib.load(scaler_path)
    columns = [
        "generated_volume_tons",
        "moisture_pct",
        "process_temperature_c",
    ]
    return {
        column: (float(scaler.data_min_[idx]), float(scaler.data_max_[idx]))
        for idx, column in enumerate(columns)
    }


def resolve_physical_ranges(
    *,
    metadata: dict[str, Any],
    dataset_path: Path,
    scaler_path: Path | None,
    logger: logging.Logger,
) -> dict[str, tuple[float, float]]:
    """Resolve physical min/max ranges for inference, independent of input naming.

    Resolution order (first available wins):
        1. `physical_ranges` embedded in the model metadata (schema_version >= 2).
           This makes inference self-contained and works with ANY input CSV name.
        2. An explicit `--scaler-path` provided by the caller.
        3. Legacy fallback: a `*_scaler.joblib` colocated with the input dataset,
           which requires the historical split naming convention.

    Args:
        metadata: Validated model metadata payload.
        dataset_path: Inference dataset path (only used by the legacy fallback).
        scaler_path: Optional explicit scaler artifact path.
        logger: Logger instance.

    Returns:
        dict[str, tuple[float, float]]: Physical ranges for volume, humidity and temperature.
    """

    embedded = metadata.get("physical_ranges")
    if isinstance(embedded, dict) and embedded:
        logger.info("Using physical ranges embedded in model metadata (schema_version=%s).", metadata.get("schema_version"))
        return {
            str(column): (float(bounds[0]), float(bounds[1]))
            for column, bounds in embedded.items()
        }

    if scaler_path is not None:
        return load_physical_ranges_from_scaler(scaler_path, logger)

    logger.warning(
        "Model metadata has no embedded physical_ranges and no --scaler-path was given. "
        "Falling back to legacy scaler resolution from the dataset filename."
    )
    legacy_scaler_path = _resolve_scaler_path_for_dataset(dataset_path)
    return load_physical_ranges_from_scaler(legacy_scaler_path, logger)


def run_inference_pipeline(
    dataset_path: Path,
    model_paths: InferenceModelPaths,
    capacities: PlantCapacities,
    runtime_config: InferenceRuntimeConfig,
    output_path: Path,
    logger: logging.Logger,
    scaler_path: Path | None = None,
    selector: str = "exact",
) -> InferenceArtifacts:
    """Run the full inference pipeline and persist the output CSV.

    Args:
        dataset_path: Input dataset path.
        model_paths: Winner genome and NEAT config paths.
        capacities: Daily plant capacities.
        runtime_config: Runtime settings.
        output_path: Output CSV path.
        logger: Logger instance.
        scaler_path: Optional explicit preprocessing scaler path. Only used when
            the model metadata does not embed physical ranges.
        selector: Decision engine. ``"exact"`` (default, deployed model) solves
            each capacity-reset block to optimality with a MILP and needs no
            trained artifact. ``"neat"`` uses the neuroevolutionary policy as a
            score signal and is retained as a benchmark.

    Returns:
        InferenceArtifacts: Output artifact summary.
    """

    df = load_dataset(dataset_path, logger)

    if selector == "exact":
        # Primary deployed model: self-contained exact optimizer. It requires no
        # genome, scaler or metadata, so it runs on any portable checkout.
        from src.predict.exact_optimizer import ExactEmissionsOptimizer

        logger.info("Using exact MILP optimizer (primary deployed decision engine).")
        exact_optimizer = ExactEmissionsOptimizer(
            capacities=capacities.as_strategy_dict(),
            lots_per_day=runtime_config.lots_per_day,
            fallback_strategy=runtime_config.fallback_strategy,
            logger=logger,
        )
        assignment_trace = exact_optimizer.infer_dataframe(df)
    elif selector == "neat":
        metadata = validate_model_metadata_for_inference(
            model_path=model_paths.model_path,
            neat_config_path=model_paths.neat_config_path,
            expected_input_columns=POLICY_INPUT_COLUMNS,
            expected_strategies=INFERENCE_STRATEGY_ORDER,
            logger=logger,
        )
        physical_ranges = resolve_physical_ranges(
            metadata=metadata,
            dataset_path=dataset_path,
            scaler_path=scaler_path,
            logger=logger,
        )
        network = load_winner_network(model_paths, logger)
        optimizer = RealisticOptimizer(
            capacities_template=capacities,
            runtime_config=runtime_config,
            physical_ranges=physical_ranges,
            network=network,
            logger=logger,
        )
        logger.info("Using NEAT policy optimizer (benchmark decision engine).")
        assignment_trace = optimizer.infer_dataframe(df)
    else:
        raise ValueError(f"Unknown selector '{selector}'. Expected 'exact' or 'neat'.")
    df = df.copy()
    df = pd.concat([df, assignment_trace], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    assigned = assignment_trace["ai_assigned_strategy"]
    distribution = summarize_strategy_distribution(assigned)
    runtime_metrics = summarize_assignment_trace(assignment_trace)
    logger.info("Inference finished. Output saved to %s", output_path)
    logger.info("Strategy distribution (pct): %s", distribution["percentages"])
    logger.info(
        "Inference fallback impact: total_fallback_count=%s (%.2f%% of rows), "
        "capacity_fallback_count=%s, row_error_fallback_count=%s, "
        "biomass_combustion_fallback_share_pct=%.2f%%",
        runtime_metrics["total_fallback_count"],
        runtime_metrics["total_fallback_pct"],
        runtime_metrics["capacity_fallback_count"],
        runtime_metrics["row_error_fallback_count"],
        runtime_metrics["biomass_combustion_fallback_share_pct"],
    )

    return InferenceArtifacts(
        output_path=output_path,
        assigned_strategies=assigned,
        distribution=distribution,
        runtime_metrics=runtime_metrics,
    )


def summarize_strategy_distribution(assigned: pd.Series) -> dict[str, Any]:
    """Build distribution statistics for assigned strategies.

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
