"""CLI entrypoint to execute neuroevolution optimization."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from pydantic import ValidationError

from src.config import AppConfig
from src.model import OptimizationPipeline
from src.utils.artifacts import (
    build_training_metadata,
    compute_sha256,
    model_metadata_path,
    write_json_metadata,
)
from src.utils.utils import configure_logging, get_project_root


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed argument object.
    """

    parser = argparse.ArgumentParser(
        description="Run neuroevolution optimization with reproducible project paths.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=get_project_root(),
        help="Project root path.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/split/dataset_optimization_cereal_co2_train_scaled.csv",
        help="Dataset path relative to project root.",
    )
    parser.add_argument(
        "--neat-config-path",
        type=str,
        default="config/config-feedforward.txt",
        help="NEAT config path relative to project root.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of sampled scenarios used in fitness evaluation.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations to evolve.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--winner-output",
        type=str,
        default="models/artifacts/winner_genome.pkl",
        help="Output path for serialized winning genome (relative to project root).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> None:
    """Run optimization pipeline from the command line."""

    args = parse_args()
    logger = configure_logging(args.log_level)

    try:
        app_config = AppConfig.build(
            project_root=args.project_root,
            dataset_relative_path=args.dataset_path,
            neat_config_relative_path=args.neat_config_path,
            sample_size=args.sample_size,
            generations=args.generations,
            random_state=args.random_state,
        )

        pipeline = OptimizationPipeline(app_config=app_config, logger=logger)
        winner = pipeline.run()

        output_path = (app_config.paths.project_root / args.winner_output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file_obj:
            pickle.dump(winner, file_obj)

        model_sha256 = compute_sha256(output_path)
        dataset_sha256 = compute_sha256(app_config.paths.dataset_path)
        neat_config_sha256 = compute_sha256(app_config.paths.neat_config_path)

        metadata = build_training_metadata(
            project_root=app_config.paths.project_root,
            dataset_path=app_config.paths.dataset_path,
            neat_config_path=app_config.paths.neat_config_path,
            sample_size=app_config.evolution.sample_size,
            generations=app_config.evolution.generations,
            random_state=app_config.evolution.random_state,
            strategies=app_config.evolution.strategies,
            input_columns=OptimizationPipeline.INPUT_COLUMNS,
            uses_temperature_as_policy_input=False,
            model_sha256=model_sha256,
            dataset_sha256=dataset_sha256,
            neat_config_sha256=neat_config_sha256,
            physical_ranges=pipeline.physical_ranges,
        )
        metadata_path = model_metadata_path(output_path)
        write_json_metadata(metadata_path, metadata)

        logger.info("Winning genome saved to %s", output_path)
        logger.info("Model metadata saved to %s", metadata_path)

        if pipeline.fitness_history:
            history_path = (
                app_config.paths.project_root
                / "models"
                / "metrics"
                / "training_fitness_history.json"
            )
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(
                    {
                        "generations": app_config.evolution.generations,
                        "sample_size": app_config.evolution.sample_size,
                        "random_state": app_config.evolution.random_state,
                        "history": pipeline.fitness_history,
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("Training fitness history saved to %s", history_path)
    except ValidationError:
        logger.exception("Configuration validation failed.")
        raise
    except Exception:
        logger.exception("Optimization run failed.")
        raise


if __name__ == "__main__":
    main()
