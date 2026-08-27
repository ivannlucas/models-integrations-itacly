"""Synthetic data generation pipeline based on the data_gen notebook logic."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from ctgan import CTGAN
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

try:  # pragma: no cover - optional dependency behavior depends on environment
    import torch
except Exception:  # pragma: no cover - keep generation working without torch import hooks
    torch = None


class NormalDistribution(BaseModel):
    """Normal distribution parameters.

    Attributes:
        mean: Mean value.
        std: Standard deviation.
    """

    mean: float
    std: float = Field(gt=0)


class CausalParameters(BaseModel):
    """Externalized parameters used to generate the causal seed dataset.

    Attributes:
        seasons: Season labels sampled for causal generation.
        subproducts: Subproduct labels sampled for causal generation.
        strategies: Strategy labels sampled for causal generation.
        physical_minimum: Lower clipping bound for physical variables.
        volume_by_subproduct: Volume distribution per subproduct.
        humidity_by_subproduct_and_season: Humidity distribution by subproduct and season.
        temperature_by_strategy: Process temperature distribution per strategy (realistic thermodynamic values).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    seasons: tuple[str, ...]
    subproducts: tuple[str, ...]
    strategies: tuple[str, ...]
    physical_minimum: float = Field(gt=0)
    volume_by_subproduct: dict[str, NormalDistribution]
    humidity_by_subproduct_and_season: dict[str, dict[str, NormalDistribution]]
    temperature_by_strategy: dict[str, NormalDistribution]

    @classmethod
    def default(cls) -> "CausalParameters":
        """Return default causal parameters matching notebook behavior."""

        return cls(
            seasons=("Dry", "Rainy"),
            subproducts=("Husk", "Bran", "Straw", "Silo dust"),
            strategies=("Biomass combustion", "Animal feed", "Composting", "Biochar"),
            physical_minimum=0.1,
            volume_by_subproduct={
                "Husk": NormalDistribution(mean=40, std=10),
                "Bran": NormalDistribution(mean=15, std=5),
                "Straw": NormalDistribution(mean=30, std=8),
                "Silo dust": NormalDistribution(mean=5, std=2),
            },
            humidity_by_subproduct_and_season={
                "Husk": {
                    "Dry": NormalDistribution(mean=8, std=2),
                    "Rainy": NormalDistribution(mean=14, std=3),
                },
                "Bran": {
                    "Dry": NormalDistribution(mean=15, std=3),
                    "Rainy": NormalDistribution(mean=22, std=4),
                },
                "Straw": {
                    "Dry": NormalDistribution(mean=10, std=2),
                    "Rainy": NormalDistribution(mean=18, std=5),
                },
                "Silo dust": {
                    "Dry": NormalDistribution(mean=5, std=1),
                    "Rainy": NormalDistribution(mean=9, std=2),
                },
            },
            temperature_by_strategy={
                "Animal feed": NormalDistribution(mean=60, std=10),  # Pasteurization: 50-70°C
                "Composting": NormalDistribution(mean=60, std=8),   # Thermophilic: 52-68°C
                "Biochar": NormalDistribution(mean=450, std=100),   # Pyrolysis: 350-550°C
                "Biomass combustion": NormalDistribution(mean=900, std=150),  # Combustion: 750-1050°C
            },
        )


class DataGenerationConfig(BaseModel):
    """Configuration for causal seed generation and CTGAN sampling.

    Attributes:
        random_state: Random seed for reproducibility.
        n_real_samples: Number of causal seed rows.
        ctgan_epochs: Number of CTGAN training epochs.
        n_synthetic_samples: Number of synthetic rows to generate.
        causal_parameters: Causal generator parameters loaded from external file.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    random_state: int = 42
    n_real_samples: PositiveInt = 1500
    ctgan_epochs: PositiveInt = 100
    n_synthetic_samples: PositiveInt = 50000
    causal_parameters: CausalParameters = Field(default_factory=CausalParameters.default)
    min_process_temperature_c: float = 0.0


def _seed_everything(seed: int) -> None:
    """Seed the common random number generators used by generation."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - depends on hardware
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class DataGenerationArtifacts:
    """Generated datasets from the pipeline.

    Attributes:
        seed_dataset: Causal seed dataset before CTGAN.
        synthetic_dataset: Final synthetic dataset after cleaning.
    """

    seed_dataset: pd.DataFrame
    synthetic_dataset: pd.DataFrame


def _calculate_emissions_co2(row: pd.Series) -> float:
    """Compute emissions with the same business physics used in notebooks.

    Args:
        row: Input row with process and strategy features.

    Returns:
        float: Estimated CO2 emissions.
    """

    emission_base = (row["process_temperature_c"] * 0.5) * row["generated_volume_tons"]

    if row["reuse_strategy"] == "Biomass combustion":
        moisture_penalty = (row["moisture_pct"] ** 1.5) * 2
        return emission_base + moisture_penalty - 50

    if row["reuse_strategy"] == "Animal feed":
        if row["subproduct_type"] in ["Husk", "Straw", "Silo dust"] or row["moisture_pct"] > 18:
            return emission_base * 1.8
        return emission_base * 0.4

    if row["reuse_strategy"] == "Biochar":
        if row["moisture_pct"] < 10:
            return emission_base * 0.2
        return emission_base * 1.2

    if row["reuse_strategy"] == "Composting":
        return emission_base * 0.8 + (row["generated_volume_tons"] * 5)

    return emission_base


def _build_causal_seed_dataset(config: DataGenerationConfig, logger: logging.Logger) -> pd.DataFrame:
    """Generate the causal seed dataset used to train CTGAN.

    Args:
        config: Generation configuration.
        logger: Logger instance.

    Returns:
        pd.DataFrame: Causal seed dataset.
    """

    _seed_everything(config.random_state)

    params = config.causal_parameters

    seasons = np.random.choice(list(params.seasons), config.n_real_samples)
    subproducts = np.random.choice(list(params.subproducts), config.n_real_samples)
    strategies = list(params.strategies)

    causal_rows: list[dict[str, object]] = []

    for subproduct, season in zip(subproducts, seasons):
        if subproduct not in params.volume_by_subproduct:
            raise ValueError(f"Missing volume distribution for subproduct: {subproduct}")
        if subproduct not in params.humidity_by_subproduct_and_season:
            raise ValueError(f"Missing humidity distribution for subproduct: {subproduct}")
        if season not in params.humidity_by_subproduct_and_season[subproduct]:
            raise ValueError(
                f"Missing humidity distribution for subproduct={subproduct}, season={season}"
            )

        humidity_dist = params.humidity_by_subproduct_and_season[subproduct][season]
        volume_dist = params.volume_by_subproduct[subproduct]

        moisture = np.random.normal(humidity_dist.mean, humidity_dist.std)
        volume = np.random.normal(volume_dist.mean, volume_dist.std)
        
        # Choose strategy first, then assign temperature based on strategy thermodynamics
        strategy = np.random.choice(strategies)
        if strategy not in params.temperature_by_strategy:
            raise ValueError(f"Missing temperature distribution for strategy: {strategy}")
        
        temp_dist = params.temperature_by_strategy[strategy]
        temperature = np.random.normal(temp_dist.mean, temp_dist.std)

        causal_rows.append(
            {
                "subproduct_type": subproduct,
                "season": season,
                "generated_volume_tons": max(params.physical_minimum, float(volume)),
                "moisture_pct": max(params.physical_minimum, float(moisture)),
                "process_temperature_c": float(temperature),
                "reuse_strategy": strategy,
            }
        )

    df = pd.DataFrame(causal_rows)
    df["co2_emissions_kg"] = df.apply(_calculate_emissions_co2, axis=1)
    df["co2_per_ton"] = df["co2_emissions_kg"] / df["generated_volume_tons"]

    logger.info("Causal seed dataset generated with shape=%s", df.shape)
    return df


def _reassign_temperatures_by_strategy(
    df: pd.DataFrame,
    temperature_by_strategy: dict[str, NormalDistribution],
    min_process_temperature_c: float,
) -> pd.DataFrame:
    """Reassign process temperatures to enforce strategy-thermodynamic coherence.

    CTGAN can break high-level constraints between `reuse_strategy` and
    `process_temperature_c`. This restoration mirrors notebook behavior.
    """

    if "reuse_strategy" not in df.columns or "process_temperature_c" not in df.columns:
        raise ValueError("Dataframe must contain reuse_strategy and process_temperature_c columns.")

    corrected = df.copy()
    for strategy, dist in temperature_by_strategy.items():
        mask = corrected["reuse_strategy"] == strategy
        if int(mask.sum()) == 0:
            continue

        corrected.loc[mask, "process_temperature_c"] = np.random.normal(
            loc=dist.mean,
            scale=dist.std,
            size=int(mask.sum()),
        )

    corrected["process_temperature_c"] = corrected["process_temperature_c"].clip(
        lower=min_process_temperature_c,
    )
    return corrected


def run_data_generation_pipeline(
    config: DataGenerationConfig,
    logger: logging.Logger,
) -> DataGenerationArtifacts:
    """Run the complete data generation pipeline.

    Args:
        config: Generation configuration.
        logger: Logger instance.

    Returns:
        DataGenerationArtifacts: Seed and synthetic dataframes.
    """

    seed_df = _build_causal_seed_dataset(config, logger)

    discrete_columns = ["subproduct_type", "season", "reuse_strategy"]
    logger.info("Training CTGAN with epochs=%s", config.ctgan_epochs)
    _seed_everything(config.random_state)
    model_ctgan = CTGAN(epochs=config.ctgan_epochs, verbose=True)
    model_ctgan.fit(seed_df, discrete_columns)

    logger.info("Sampling %s synthetic rows", config.n_synthetic_samples)
    _seed_everything(config.random_state)
    synthetic_df = model_ctgan.sample(config.n_synthetic_samples)

    physical_columns = ["generated_volume_tons", "moisture_pct"]
    synthetic_df[physical_columns] = synthetic_df[physical_columns].clip(lower=0.1)
    synthetic_df = _reassign_temperatures_by_strategy(
        synthetic_df,
        config.causal_parameters.temperature_by_strategy,
        config.min_process_temperature_c,
    )

    synthetic_df["co2_emissions_kg"] = synthetic_df.apply(_calculate_emissions_co2, axis=1)
    synthetic_df["co2_emissions_kg"] = synthetic_df["co2_emissions_kg"].clip(lower=0.0)
    synthetic_df["co2_per_ton"] = (
        synthetic_df["co2_emissions_kg"] / synthetic_df["generated_volume_tons"]
    )

    seed_df = seed_df.copy()
    seed_df["process_temperature_c"] = seed_df["process_temperature_c"].clip(
        lower=config.min_process_temperature_c,
    )
    seed_df["co2_emissions_kg"] = seed_df.apply(_calculate_emissions_co2, axis=1)
    seed_df["co2_emissions_kg"] = seed_df["co2_emissions_kg"].clip(lower=0.0)
    seed_df["co2_per_ton"] = seed_df["co2_emissions_kg"] / seed_df["generated_volume_tons"]

    logger.info("Synthetic dataset generated with shape=%s", synthetic_df.shape)
    return DataGenerationArtifacts(seed_dataset=seed_df, synthetic_dataset=synthetic_df)


def load_causal_parameters(params_path: Path, logger: logging.Logger) -> CausalParameters:
    """Load causal generation parameters from a JSON file.

    Args:
        params_path: Path to JSON parameter file.
        logger: Logger instance.

    Returns:
        CausalParameters: Validated parameter object.
    """

    if not params_path.exists():
        raise FileNotFoundError(f"Causal parameter file not found: {params_path}")

    logger.info("Loading causal generation parameters from %s", params_path)
    with params_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    # Backward compatibility: migrate legacy single-temperature configs.
    if "temperature_by_strategy" not in payload and "temperature" in payload:
        logger.warning(
            "Legacy causal parameter schema detected (temperature). Migrating to temperature_by_strategy."
        )
        legacy_temperature = payload.get("temperature", {"mean": 180, "std": 20})
        payload["temperature_by_strategy"] = {
            "Animal feed": {"mean": 60, "std": 10},
            "Composting": {"mean": 60, "std": 8},
            "Biochar": {"mean": 450, "std": 100},
            "Biomass combustion": {"mean": 900, "std": 150},
        }
        if not isinstance(legacy_temperature, dict):
            payload["temperature_by_strategy"] = {
                strategy: {"mean": 180, "std": 20}
                for strategy in payload.get(
                    "strategies",
                    ["Biomass combustion", "Animal feed", "Composting", "Biochar"],
                )
            }
        payload.pop("temperature", None)

    return CausalParameters(**payload)
