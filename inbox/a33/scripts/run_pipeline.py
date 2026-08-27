"""Single-command pipeline runner for the full DATAGIA workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_inference import _evaluate_dataframe, _load_temperature_distributions
from src.config import AppConfig
from src.data_processing.data_generation import DataGenerationConfig, load_causal_parameters, run_data_generation_pipeline
from src.predict.inference import (
    InferenceModelPaths,
    InferenceRuntimeConfig,
    PlantCapacities,
    run_inference_pipeline,
)
from src.model import OptimizationPipeline
from src.data_processing.preprocessing import PreprocessingConfig, run_preprocessing_pipeline
from src.utils.artifacts import build_training_metadata, model_metadata_path, write_json_metadata
from src.utils.utils import configure_logging, get_project_root


def _sha256_of_file(path: Path) -> str:
    """Compute SHA-256 hash for reproducibility manifests."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_root(path: Path, project_root: Path) -> str:
    """Return a portable, root-relative POSIX path for manifest entries.

    Keeping manifest paths relative avoids leaking absolute training-machine
    paths and keeps the run manifest reproducible across environments.
    """

    resolved = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


class PipelineSection(BaseModel):
    """Generic pipeline section with relative paths and hyperparameters."""

    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)


class DataGenerationSection(PipelineSection):
    raw_seed_output_path: str = Field(min_length=1)
    processed_output_path: str = Field(min_length=1)
    n_real_samples: int = Field(gt=0)
    ctgan_epochs: int = Field(gt=0)
    n_synthetic_samples: int = Field(gt=0)
    random_state: int | None = None


class PreprocessingSection(PipelineSection):
    split_dir: str = Field(min_length=1)
    split_prefix: str = Field(min_length=1)
    test_size: float = Field(gt=0, lt=1)
    random_state: int | None = None


class TrainingSection(PipelineSection):
    dataset_path: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    sample_size: int = Field(gt=0)
    generations: int = Field(gt=0)
    random_state: int | None = None
    neat_config_path: str = Field(min_length=1)


class InferenceSection(PipelineSection):
    dataset_path: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    neat_config_path: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    lots_per_day: int = Field(gt=0)
    animal_feed_capacity: float = Field(gt=0)
    composting_capacity: float = Field(gt=0)
    biochar_capacity: float = Field(gt=0)
    biomass_combustion_capacity: float = Field(gt=0)


class EvaluationSection(PipelineSection):
    input_path: str = Field(min_length=1)
    report_path: str = Field(min_length=1)
    stochastic_runs: int = Field(default=200, gt=0)
    random_state: int | None = None
    allow_causal_params_fallback: bool = False


class PipelineConfig(BaseModel):
    """Configuration for the full end-to-end pipeline."""

    model_config = ConfigDict(str_strip_whitespace=True)

    random_state: int = 42
    causal_params_path: str = Field(min_length=1)
    data_generation: DataGenerationSection
    preprocessing: PreprocessingSection
    training: TrainingSection
    inference: InferenceSection
    evaluation: EvaluationSection


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Run the full DATAGIA pipeline end-to-end.")
    parser.add_argument("--project-root", type=Path, default=get_project_root(), help="Project root path.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/pipeline_config.json",
        help="Pipeline JSON config path relative to project root.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> None:
    """Run the complete pipeline with a single config file."""

    args = parse_args()
    logger = configure_logging(args.log_level)

    try:
        project_root = args.project_root.resolve()
        config_path = project_root / args.config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config_path}")

        pipeline_config = PipelineConfig.model_validate_json(config_path.read_text(encoding="utf-8"))

        causal_params_abs = project_root / pipeline_config.causal_params_path
        generation = pipeline_config.data_generation
        preprocessing = pipeline_config.preprocessing
        training = pipeline_config.training
        inference = pipeline_config.inference
        evaluation = pipeline_config.evaluation

        shared_random_state = pipeline_config.random_state

        generation_config = DataGenerationConfig(
            random_state=generation.random_state if generation.random_state is not None else shared_random_state,
            n_real_samples=generation.n_real_samples,
            ctgan_epochs=generation.ctgan_epochs,
            n_synthetic_samples=generation.n_synthetic_samples,
            causal_parameters=load_causal_parameters(causal_params_abs, logger),
        )
        artifacts = run_data_generation_pipeline(generation_config, logger)

        raw_seed_abs = project_root / generation.raw_seed_output_path
        processed_abs = project_root / generation.processed_output_path
        raw_seed_abs.parent.mkdir(parents=True, exist_ok=True)
        processed_abs.parent.mkdir(parents=True, exist_ok=True)
        artifacts.seed_dataset.to_csv(raw_seed_abs, index=False)
        artifacts.synthetic_dataset.to_csv(processed_abs, index=False)

        preprocessing_config = PreprocessingConfig(
            test_size=preprocessing.test_size,
            random_state=preprocessing.random_state if preprocessing.random_state is not None else shared_random_state,
            split_prefix=preprocessing.split_prefix,
        )
        run_preprocessing_pipeline(
            input_path=processed_abs,
            output_dir=project_root / preprocessing.split_dir,
            config=preprocessing_config,
            logger=logger,
        )

        app_config = AppConfig.build(
            project_root=project_root,
            dataset_relative_path=f"{preprocessing.split_dir}/{preprocessing.split_prefix}_train_scaled.csv",
            neat_config_relative_path=training.neat_config_path,
            sample_size=training.sample_size,
            generations=training.generations,
            random_state=training.random_state if training.random_state is not None else shared_random_state,
        )
        optimization_pipeline = OptimizationPipeline(app_config=app_config, logger=logger)
        winner = optimization_pipeline.run()

        model_abs = project_root / training.model_path
        model_abs.parent.mkdir(parents=True, exist_ok=True)
        with model_abs.open("wb") as file_obj:
            pickle.dump(winner, file_obj)

        model_sha256 = _sha256_of_file(model_abs)
        dataset_sha256 = _sha256_of_file(app_config.paths.dataset_path)
        neat_config_sha256 = _sha256_of_file(app_config.paths.neat_config_path)

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
            physical_ranges=optimization_pipeline.physical_ranges,
        )
        metadata_path = model_metadata_path(model_abs)
        write_json_metadata(metadata_path, metadata)
        logger.info("Model metadata saved to %s", metadata_path)

        inference_paths = InferenceModelPaths(
            model_path=model_abs,
            neat_config_path=project_root / inference.neat_config_path,
        )
        capacities = PlantCapacities(
            animal_feed=inference.animal_feed_capacity,
            composting=inference.composting_capacity,
            biochar=inference.biochar_capacity,
            biomass_combustion=inference.biomass_combustion_capacity,
        )
        # The deployed decision engine is the exact MILP optimizer. NEAT is
        # trained above only as a benchmark artifact; the pipeline's inference
        # output uses the exact selector.
        run_inference_pipeline(
            dataset_path=project_root / inference.dataset_path,
            model_paths=inference_paths,
            capacities=capacities,
            runtime_config=InferenceRuntimeConfig(lots_per_day=inference.lots_per_day),
            output_path=project_root / inference.output_path,
            logger=logger,
            selector="exact",
        )

        evaluation_df = pd.read_csv(project_root / evaluation.input_path)
        strategy_temperature_mean_c, strategy_temperature_std_c = _load_temperature_distributions(
            causal_params_abs=causal_params_abs,
            strict=not bool(evaluation.allow_causal_params_fallback),
            logger=logger,
        )
        evaluation_random_state = (
            int(evaluation.random_state)
            if evaluation.random_state is not None
            else int(shared_random_state)
        )
        report = _evaluate_dataframe(
            evaluation_df,
            stochastic_runs=max(1, int(evaluation.stochastic_runs)),
            stochastic_random_state=evaluation_random_state,
            strategy_temperature_mean_c=strategy_temperature_mean_c,
            strategy_temperature_std_c=strategy_temperature_std_c,
        ).model_dump()
        report_abs = project_root / evaluation.report_path
        report_abs.parent.mkdir(parents=True, exist_ok=True)
        report_abs.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

        manifest_path = project_root / "models/metrics/pipeline_run_manifest.json"
        manifest_payload = {
            "schema_version": 1,
            "pipeline_config": pipeline_config.model_dump(mode="json"),
            "artifacts": {
                "raw_seed": {
                    "path": _relative_to_root(raw_seed_abs, project_root),
                    "sha256": _sha256_of_file(raw_seed_abs),
                },
                "synthetic_dataset": {
                    "path": _relative_to_root(processed_abs, project_root),
                    "sha256": _sha256_of_file(processed_abs),
                },
                "winner_genome": {
                    "path": _relative_to_root(model_abs, project_root),
                    "sha256": _sha256_of_file(model_abs),
                },
                "winner_metadata": {
                    "path": _relative_to_root(metadata_path, project_root),
                    "sha256": _sha256_of_file(metadata_path),
                },
                "inference_output": {
                    "path": _relative_to_root(project_root / inference.output_path, project_root),
                    "sha256": _sha256_of_file(project_root / inference.output_path),
                },
                "evaluation_report": {
                    "path": _relative_to_root(report_abs, project_root),
                    "sha256": _sha256_of_file(report_abs),
                },
            },
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=True, indent=2), encoding="utf-8")

        logger.info("Evaluation report saved to %s", report_abs)
        logger.info("Pipeline manifest saved to %s", manifest_path)
        logger.info("Pipeline completed successfully.")
    except ValidationError:
        logger.exception("Invalid configuration provided for pipeline execution.")
        raise
    except Exception:
        logger.exception("Full pipeline execution failed.")
        raise


if __name__ == "__main__":
    main()
