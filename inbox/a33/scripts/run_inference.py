"""CLI entrypoint to run constrained strategy inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.predict.inference import (
    InferenceModelPaths,
    InferenceRuntimeConfig,
    PlantCapacities,
    run_inference_pipeline,
)
from src.utils.utils import configure_logging, get_project_root


class InferencePathsConfig(BaseModel):
    """Validated paths for inference execution.

    Attributes:
        project_root: Absolute project root path.
        dataset_path: Relative dataset path from project root.
        output_path: Relative output path from project root.
    """

    project_root: Path
    dataset_path: str
    output_path: str
    model_path: str
    neat_config_path: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Run batch inference with capacity constraints.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=get_project_root(),
        help="Project root path.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/split/dataset_optimization_cereal_co2_test_raw.csv",
        help="Dataset path relative to project root (typically the held-out test split).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/predictions/inference_with_constraints.csv",
        help="Output CSV path relative to project root.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/artifacts/winner_genome.pkl",
        help="Winner genome pickle path relative to project root.",
    )
    parser.add_argument(
        "--neat-config-path",
        type=str,
        default="config/config-feedforward.txt",
        help="NEAT config path relative to project root.",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=None,
        help=(
            "Optional explicit preprocessing scaler path (relative to project root). "
            "Only needed for legacy models whose metadata does not embed physical ranges. "
            "Modern artifacts make inference self-contained, so any input CSV name works."
        ),
    )
    parser.add_argument(
        "--selector",
        type=str,
        default="exact",
        choices=("exact", "neat"),
        help=(
            "Decision engine. 'exact' (default, deployed model) solves each "
            "capacity-reset block to optimality with a MILP and needs no trained "
            "artifact. 'neat' uses the neuroevolutionary policy as a benchmark."
        ),
    )
    parser.add_argument(
        "--lots-per-day",
        type=int,
        default=15,
        help="Number of lots processed before resetting plant capacities.",
    )
    parser.add_argument(
        "--cap-animal-feed",
        type=float,
        default=90.0,
        help="Daily capacity for Animal feed (tons). Must match training capacity template.",
    )
    parser.add_argument(
        "--cap-composting",
        type=float,
        default=140.0,
        help="Daily capacity for Composting (tons). Must match training capacity template.",
    )
    parser.add_argument(
        "--cap-biochar",
        type=float,
        default=45.0,
        help="Daily capacity for Biochar (tons). Must match training capacity template.",
    )
    parser.add_argument(
        "--cap-biomass-combustion",
        type=float,
        default=10000.0,
        help="Daily capacity for Biomass combustion (tons). Must match training capacity template.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    """Run constrained inference and persist results."""

    args = parse_args()
    logger = configure_logging(args.log_level)

    try:
        path_config = InferencePathsConfig(
            project_root=args.project_root.resolve(),
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            model_path=args.model_path,
            neat_config_path=args.neat_config_path,
        )
        dataset_abs_path = path_config.project_root / path_config.dataset_path
        output_abs_path = path_config.project_root / path_config.output_path
        model_abs_path = path_config.project_root / path_config.model_path
        neat_config_abs_path = path_config.project_root / path_config.neat_config_path
        scaler_abs_path = (
            (path_config.project_root / args.scaler_path) if args.scaler_path else None
        )

        capacities = PlantCapacities(
            animal_feed=args.cap_animal_feed,
            composting=args.cap_composting,
            biochar=args.cap_biochar,
            biomass_combustion=args.cap_biomass_combustion,
        )
        runtime_config = InferenceRuntimeConfig(lots_per_day=args.lots_per_day)
        model_paths = InferenceModelPaths(
            model_path=model_abs_path,
            neat_config_path=neat_config_abs_path,
        )

        run_inference_pipeline(
            dataset_path=dataset_abs_path,
            model_paths=model_paths,
            capacities=capacities,
            runtime_config=runtime_config,
            output_path=output_abs_path,
            logger=logger,
            scaler_path=scaler_abs_path,
            selector=args.selector,
        )
    except ValidationError:
        logger.exception("Invalid configuration provided for inference.")
        raise
    except Exception:
        logger.exception("Inference execution failed.")
        raise


if __name__ == "__main__":
    main()
