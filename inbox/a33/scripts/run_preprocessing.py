"""CLI entrypoint to split and scale the dataset before training."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.data_processing.preprocessing import PreprocessingConfig, run_preprocessing_pipeline
from src.utils.utils import configure_logging, get_project_root


class PreprocessingPathsConfig(BaseModel):
    """Validated paths for preprocessing execution.

    Attributes:
        project_root: Absolute project root path.
        input_path: Relative raw dataset path.
        output_dir: Relative output directory for split artifacts.
        split_prefix: Prefix used for generated files.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    project_root: Path
    input_path: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    split_prefix: str = Field(min_length=1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed argument values.
    """

    parser = argparse.ArgumentParser(
        description="Split the raw dataset and scale features for training.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=get_project_root(),
        help="Project root path.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/processed/dataset_optimization_cereal_co2.csv",
        help="Raw dataset path relative to project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/split",
        help="Output directory for split artifacts.",
    )
    parser.add_argument(
        "--split-prefix",
        type=str,
        default="dataset_optimization_cereal_co2",
        help="Prefix for split file names.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the split.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    """Run preprocessing workflow and persist split artifacts."""

    args = parse_args()
    logger = configure_logging(args.log_level)

    try:
        path_config = PreprocessingPathsConfig(
            project_root=args.project_root.resolve(),
            input_path=args.input_path,
            output_dir=args.output_dir,
            split_prefix=args.split_prefix,
        )

        preprocessing_config = PreprocessingConfig(
            test_size=args.test_size,
            random_state=args.random_state,
            split_prefix=args.split_prefix,
        )

        input_abs_path = path_config.project_root / path_config.input_path
        output_dir_abs = path_config.project_root / path_config.output_dir

        run_preprocessing_pipeline(
            input_path=input_abs_path,
            output_dir=output_dir_abs,
            config=preprocessing_config,
            logger=logger,
        )
    except ValidationError:
        logger.exception("Invalid preprocessing configuration.")
        raise
    except Exception:
        logger.exception("Preprocessing pipeline failed.")
        raise


if __name__ == "__main__":
    main()
