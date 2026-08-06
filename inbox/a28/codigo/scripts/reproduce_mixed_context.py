from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.package_data_blob import create_data_blob
from src.data_acquisition import run_data_acquisition
from src.evaluation import run_reproducibility_get_stats, run_reproducibility_policy_simulation
from src.feature_engineering import run_feature_engineering
from src.prediction import run_reproducibility_prediction
from src.reproducibility import CACHED_MODE, FULL_MODE, SMOKE_MODE, build_reproducibility_config, build_reproducibility_manifest
from src.reproducibility.workflows import run_reproducibility_etl, run_reproducibility_make_splits
from src.training import run_reproducibility_training


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("cu28.reproduce_mixed_context")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _resolve_mode(args: argparse.Namespace) -> str:
    if args.full:
        return FULL_MODE
    if args.smoke:
        return SMOKE_MODE
    return CACHED_MODE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the official CU28 mixed_context route end-to-end.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the base YAML configuration.")
    parser.add_argument("--scope", default="mixed_context", help="Reproducibility scope. Only mixed_context is supported.")
    parser.add_argument("--skip-download", action="store_true", help="Do not attempt any raw download.")
    parser.add_argument("--use-cached-raw", action=argparse.BooleanOptionalAction, default=True, help="Prefer cached raw snapshots.")
    parser.add_argument("--smoke", action="store_true", help="Run the reduced smoke configuration.")
    parser.add_argument("--full", action="store_true", help="Run the full configuration.")
    parser.add_argument("--output-dir", default="dist", help="Directory where the packaged blob will be written.")
    parser.add_argument("--fail-on-missing-raw", action=argparse.BooleanOptionalAction, default=True, help="Fail when an active raw snapshot is missing.")
    parser.add_argument("--allow-synthetic-plant-layer", action=argparse.BooleanOptionalAction, default=True, help="Allow the declared synthetic plant layer.")
    parser.add_argument("--run-notebooks", action="store_true", help="Execute the EDA notebooks after metrics are available.")
    args = parser.parse_args(argv)

    if args.scope != "mixed_context":
        raise ValueError("Only scope=mixed_context is supported by the official reproducibility route.")

    logger = _build_logger()
    mode = _resolve_mode(args)
    config = build_reproducibility_config(
        args.config,
        mode=mode,
        allow_synthetic_plant_layer=args.allow_synthetic_plant_layer,
    )
    reference_date = str(config.get("official_release", {}).get("reference_date", "")).strip()
    if not reference_date:
        raise ValueError("official_release.reference_date must be configured for the official mixed_context run.")
    official_run_id = (
        f"mixed_context_{reference_date.replace('-', '')}_seed"
        f"{config.get('project', {}).get('seed', 42)}_{mode}"
    )
    config.setdefault("runtime", {})["official_run"] = {
        "id": official_run_id,
        "kind": "end_to_end",
        "publish_latest": True,
        "reference_date": reference_date,
        "mode": mode,
    }
    config["data_processing"]["source_refresh"]["end_date"] = reference_date

    commands_executed: list[str] = []

    logger.info("1/10 data_acquisition")
    acquisition = run_data_acquisition(
        config,
        use_cached_raw=args.use_cached_raw,
        skip_download=args.skip_download,
        fail_on_missing_raw=args.fail_on_missing_raw,
    )
    commands_executed.append("python -m src.main data_acquisition --mixed-context")

    logger.info("2/10 etl")
    etl = run_reproducibility_etl(
        config,
        logger,
        force_download=not args.use_cached_raw and not args.skip_download,
    )
    commands_executed.append("python -m src.main etl --mixed-context")

    logger.info("3/10 feature_engineering")
    feature_engineering = run_feature_engineering(config, logger)
    commands_executed.append("python -m src.main feature_engineering --mixed-context")

    logger.info("4/10 make_splits")
    splits = run_reproducibility_make_splits(config, logger)
    commands_executed.append("python -m src.main make_splits --mixed-context")

    logger.info("5/10 train")
    training = run_reproducibility_training(config, logger)
    commands_executed.append("python -m src.main train --mixed-context")

    logger.info("6/10 predict")
    prediction = run_reproducibility_prediction(config, logger)
    commands_executed.append("python -m src.main predict --mixed-context")

    logger.info("7/10 policy_simulation")
    policy = run_reproducibility_policy_simulation(config, logger)
    commands_executed.append("python -m src.main policy_simulation --mixed-context")

    logger.info("8/10 get_stats")
    stats = run_reproducibility_get_stats(config, logger)
    commands_executed.append("python -m src.main get_stats --mixed-context")

    notebook_result = None
    if args.run_notebooks:
        logger.info("9/11 run_notebooks")
        from scripts.run_notebooks import run_notebooks

        notebook_result = run_notebooks(scope=args.scope, smoke=args.smoke, output_dir="reports/notebooks")
        commands_executed.append("python scripts/run_notebooks.py --scope mixed_context")

    logger.info("9/10 reproducibility_manifest")
    reproducibility_manifest = build_reproducibility_manifest(
        config,
        commands_executed=commands_executed,
        warnings=[],
        limitations=[
            "External sources are contextual proxies rather than internal plant histories.",
            "The synthetic plant layer remains synthetic unless replaced by customer-provided plant data.",
        ],
    )

    logger.info("10/10 package_data_blob")
    blob = create_data_blob(
        output_dir=args.output_dir,
        stamp=reference_date.replace("-", ""),
    )
    commands_executed.append(f"python scripts/package_data_blob.py --output-dir {Path(args.output_dir).as_posix()}")

    payload = {
        "scope": args.scope,
        "mode": mode,
        "official_run": config["runtime"]["official_run"],
        "data_acquisition": acquisition,
        "etl": etl,
        "feature_engineering": feature_engineering,
        "make_splits": splits,
        "train": training,
        "predict": prediction,
        "policy_simulation": policy,
        "get_stats": stats,
        "run_notebooks": notebook_result,
        "reproducibility_manifest_path": "reproducibility_manifest__mixed_context.json",
        "data_blob": blob,
        "commands_executed": commands_executed,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
