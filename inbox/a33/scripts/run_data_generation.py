"""CLI entrypoint to generate synthetic training data with CTGAN."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.data_processing.data_generation import (
    DataGenerationConfig,
    load_causal_parameters,
    run_data_generation_pipeline,
)
from src.utils.utils import configure_logging, get_project_root


class DataGenerationPathsConfig(BaseModel):
    """Validated path configuration for synthetic data generation.

    Attributes:
        project_root: Absolute project root path.
        output_path: Relative output CSV path.
        seed_output_path: Relative causal seed CSV path.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    project_root: Path
    output_path: str = Field(min_length=1)
    seed_output_path: str = Field(min_length=1)
    causal_params_path: str = Field(min_length=1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed argument values.
    """

    parser = argparse.ArgumentParser(
        description="Run synthetic data generation pipeline (causal seed + CTGAN).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=get_project_root(),
        help="Project root path.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/dataset_optimization_cereal_co2.csv",
        help="Synthetic dataset output path relative to project root.",
    )
    parser.add_argument(
        "--seed-output-path",
        type=str,
        default="data/raw/dataset_optimization_cereal_co2_seed.csv",
        help="Causal seed dataset output path relative to project root.",
    )
    parser.add_argument(
        "--causal-params-path",
        type=str,
        default="config/data_generation_params.json",
        help="Causal generation parameter JSON path relative to project root.",
    )
    parser.add_argument(
        "--n-real-samples",
        type=int,
        default=1500,
        help="Number of causal seed rows.",
    )
    parser.add_argument(
        "--ctgan-epochs",
        type=int,
        default=100,
        help="Number of CTGAN training epochs.",
    )
    parser.add_argument(
        "--n-synthetic-samples",
        type=int,
        default=50000,
        help="Number of synthetic rows to generate.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> None:
    """Run synthetic data generation and persist artifacts."""

    args = parse_args()
    logger = configure_logging(args.log_level)

    try:
        paths_config = DataGenerationPathsConfig(
            project_root=args.project_root.resolve(),
            output_path=args.output_path,
            seed_output_path=args.seed_output_path,
            causal_params_path=args.causal_params_path,
        )

        causal_params_abs_path = paths_config.project_root / paths_config.causal_params_path
        causal_parameters = load_causal_parameters(causal_params_abs_path, logger)

        generation_config = DataGenerationConfig(
            random_state=args.random_state,
            n_real_samples=args.n_real_samples,
            ctgan_epochs=args.ctgan_epochs,
            n_synthetic_samples=args.n_synthetic_samples,
            causal_parameters=causal_parameters,
        )

        artifacts = run_data_generation_pipeline(generation_config, logger)

        synthetic_output = paths_config.project_root / paths_config.output_path
        seed_output = paths_config.project_root / paths_config.seed_output_path

        synthetic_output.parent.mkdir(parents=True, exist_ok=True)
        seed_output.parent.mkdir(parents=True, exist_ok=True)

        artifacts.synthetic_dataset.to_csv(synthetic_output, index=False)
        artifacts.seed_dataset.to_csv(seed_output, index=False)

        logger.info("Synthetic dataset saved to %s", synthetic_output)
        logger.info("Seed dataset saved to %s", seed_output)
    except ValidationError:
        logger.exception("Invalid configuration for data generation.")
        raise
    except Exception:
        logger.exception("Data generation pipeline failed.")
        raise


if __name__ == "__main__":
    main()
