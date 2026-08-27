"""Pipeline orchestration for optimization experiments."""

from __future__ import annotations

import logging

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import AppConfig
from src.training.evolution import EvolutionRunner
from src.utils.utils import load_dataset


class OptimizationPipeline:
    """Run end-to-end preparation and neuroevolution.

    Args:
        app_config: Validated application configuration.
        logger: Logger instance.
    """

    RAW_REQUIRED_COLUMNS: tuple[str, ...] = (
        "generated_volume_tons",
        "moisture_pct",
        "process_temperature_c",
        "subproduct_type",
        "season",
    )

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

    PREPROCESSED_CATEGORY_COLUMNS: tuple[str, ...] = (
        "subproduct_type_Husk",
        "subproduct_type_Straw",
        "subproduct_type_Silo dust",
        "subproduct_type_Bran",
        "season_Rainy",
        "season_Dry",
    )

    def __init__(self, app_config: AppConfig, logger: logging.Logger) -> None:
        self.app_config = app_config
        self.logger = logger
        # Populated by `run()` so callers (e.g. run_optimization) can embed the
        # training scaler bounds in the model metadata and keep inference
        # decoupled from the input dataset filename.
        self.physical_ranges: dict[str, tuple[float, float]] | None = None
        # Per-generation training curves captured by the evolution runner.
        self.fitness_history: list[dict[str, float]] = []

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate expected dataset schema and null safety.

        Args:
            df: Input dataframe loaded from CSV.

        Raises:
            ValueError: If required columns are missing or contain null values.
        """

        has_raw_schema = all(col in df.columns for col in self.RAW_REQUIRED_COLUMNS)
        has_preprocessed_schema = all(col in df.columns for col in self.INPUT_COLUMNS)

        if not has_raw_schema and not has_preprocessed_schema:
            raise ValueError(
                "Dataset must contain either raw columns or preprocessed feature columns."
            )

        candidate_columns = self.RAW_REQUIRED_COLUMNS if has_raw_schema else self.INPUT_COLUMNS
        null_counts = df[list(candidate_columns)].isna().sum()
        if int(null_counts.sum()) > 0:
            raise ValueError(
                f"Null values found in required columns: {null_counts.to_dict()}"
            )

    def _is_preprocessed(self, df: pd.DataFrame) -> bool:
        """Determine whether the dataset already contains preprocessed features."""

        return all(col in df.columns for col in self.INPUT_COLUMNS) and not all(
            col in df.columns for col in self.RAW_REQUIRED_COLUMNS
        )

    def _prepare_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, tuple[float, float]]]:
        """Create encoded and normalized input features for NEAT.

        Args:
            df: Full dataset.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, tuple[float, float]]]:
            Feature matrix (without temperature as model input), sampled
            evaluation dataframe, residue labels, and physical min/max ranges
            used by the emissions simulator.
        """

        sampled_df = df.sample(
            n=self.app_config.evolution.sample_size,
            random_state=self.app_config.evolution.random_state,
        ).copy()

        if self._is_preprocessed(sampled_df):
            features_df = sampled_df[list(self.INPUT_COLUMNS)].astype(float)
            residue_labels = self._recover_residue_labels(
                sampled_df,
                list(self.PREPROCESSED_CATEGORY_COLUMNS[:4]),
            )
            scaler_path = self._resolve_scaler_path_for_scaled_dataset()
            scaler: MinMaxScaler = joblib.load(scaler_path)
            physical_columns = [
                "generated_volume_tons",
                "moisture_pct",
                "process_temperature_c",
            ]
            physical_ranges = {
                column: (float(scaler.data_min_[idx]), float(scaler.data_max_[idx]))
                for idx, column in enumerate(physical_columns)
            }
            return (
                features_df,
                sampled_df.reset_index(drop=True),
                residue_labels,
                physical_ranges,
            )

        residue_labels = sampled_df["subproduct_type"].astype(str)
        encoded_df = pd.get_dummies(
            sampled_df,
            columns=["subproduct_type", "season"],
        )

        scaler = MinMaxScaler()
        physical_columns = [
            "generated_volume_tons",
            "moisture_pct",
            "process_temperature_c",
        ]
        encoded_df[physical_columns] = scaler.fit_transform(encoded_df[physical_columns])

        for expected_column in self.INPUT_COLUMNS:
            if expected_column not in encoded_df.columns:
                encoded_df[expected_column] = 0

        features_df = encoded_df[list(self.INPUT_COLUMNS)].astype(float)
        physical_ranges = {
            column: (float(scaler.data_min_[idx]), float(scaler.data_max_[idx]))
            for idx, column in enumerate(physical_columns)
        }
        return (
            features_df,
            sampled_df.reset_index(drop=True),
            residue_labels.reset_index(drop=True),
            physical_ranges,
        )

    def _resolve_scaler_path_for_scaled_dataset(self):
        """Resolve scaler artifact path from a scaled dataset filename.

        Returns:
            Path: Scaler path matching the scaled split dataset.

        Raises:
            ValueError: If dataset name is not a recognized scaled split pattern.
            FileNotFoundError: If the scaler artifact does not exist.
        """

        dataset_path = self.app_config.paths.dataset_path
        dataset_name = dataset_path.name

        suffixes = ("_train_scaled.csv", "_test_scaled.csv")
        prefix = None
        for suffix in suffixes:
            if dataset_name.endswith(suffix):
                prefix = dataset_name[: -len(suffix)]
                break

        if prefix is None:
            raise ValueError(
                "Cannot infer scaler path from scaled dataset name. "
                "Expected suffix '_train_scaled.csv' or '_test_scaled.csv'."
            )

        scaler_path = dataset_path.parent / f"{prefix}_scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")
        return scaler_path

    def _recover_residue_labels(self, df: pd.DataFrame, residue_columns: list[str]) -> pd.Series:
        """Recover residue labels from one-hot encoded features."""

        residue_frame = df[residue_columns]
        return residue_frame.idxmax(axis=1).str.replace("subproduct_type_", "", regex=False)

    def run(self):
        """Execute the complete optimization pipeline.

        Returns:
            Any: Winning genome produced by NEAT.
        """

        try:
            df = load_dataset(self.app_config.paths.dataset_path, self.logger)
            self._validate_schema(df)
            features_df, evaluation_df, residue_labels, physical_ranges = self._prepare_features(df)
            self.physical_ranges = physical_ranges

            runner = EvolutionRunner(
                app_config=self.app_config,
                features_df=features_df,
                evaluation_df=evaluation_df,
                residue_labels=residue_labels,
                physical_ranges=physical_ranges,
                logger=self.logger,
            )
            winner = runner.run()
            self.fitness_history = runner.fitness_history
            return winner
        except Exception:
            self.logger.exception("Pipeline execution failed.")
            raise
