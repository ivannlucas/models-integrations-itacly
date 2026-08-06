"""Command-line entry point for legacy and platform CU28 workflows."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Callable

from src.cli import run_platform_pipeline

CLI_RECOMMENDATION_COLUMNS = [
    "date",
    "raw_material_id",
    "destination_profile",
    "recommended_action",
    "purchase_trigger_proba",
    "order_quantity_tons",
    "decision_reason",
    "excess_tons",
    "stockout_tons",
]

CLI_PURCHASE_SUMMARY_COLUMNS = [
    "raw_material_id",
    "destination_profiles",
    "buy_rows",
    "total_order_quantity_tons",
    "avg_purchase_trigger_proba",
    "total_excess_tons",
    "total_stockout_tons",
]

CLI_NUMERIC_FORMATS = {
    "purchase_trigger_proba": "{:.4f}",
    "avg_purchase_trigger_proba": "{:.4f}",
    "buy_rows": "{:.0f}",
    "order_quantity_tons": "{:.3f}",
    "total_order_quantity_tons": "{:.3f}",
    "excess_tons": "{:.3f}",
    "total_excess_tons": "{:.3f}",
    "stockout_tons": "{:.3f}",
    "total_stockout_tons": "{:.3f}",
}


def _add_recipe_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recipe-profile",
        help="Run the stage against a single recipe_profile from the manufacturing registry.",
    )
    parser.add_argument(
        "--all-recipes",
        action="store_true",
        help="Execute the stage once per recipe_profile defined in the manufacturing registry.",
    )
    parser.add_argument(
        "--mixed-context",
        action="store_true",
        help="Execute the stage against the explicit mixed-context/global view of the manufacturing registry.",
    )
    parser.add_argument(
        "--recipe-config",
        help="Override the manufacturing-profile registry path for this invocation.",
    )


def _add_reproducibility_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke", action="store_true", help="Use the reduced smoke configuration for mixed_context.")
    parser.add_argument("--full", action="store_true", help="Use the full reproducibility configuration for mixed_context.")
    parser.add_argument(
        "--use-cached-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer cached raw snapshots when available.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Do not attempt any network download for raw sources.")
    parser.add_argument(
        "--fail-on-missing-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when an active raw source is missing from the local cache.",
    )
    parser.add_argument(
        "--allow-synthetic-plant-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the declared synthetic plant layer in the official mixed_context route.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CU28 / NEUROCARN-OPT command-line interface.")
    parser.add_argument("--config", help="Path to the YAML configuration file used by legacy commands.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("data_processing", "Prepare the modeling dataset and create splits."),
        ("train", "Train the configured baseline/neuroevolution comparison and persist artifacts."),
        ("predict", "Run batch prediction with a trained artifact."),
        ("get_stats", "Aggregate saved metrics across runs."),
        ("policy_simulation", "Evaluate procurement decision policies inside the synthetic simulator."),
    ]:
        subparser = subparsers.add_parser(command, help=help_text)
        _add_recipe_runtime_args(subparser)
        _add_reproducibility_args(subparser)
        if command == "predict":
            subparser.add_argument("--input", help="Optional inference CSV. Defaults to the mixed_context test split.")
    for command, help_text in [
        ("data_acquisition", "Restore or refresh the official mixed_context raw snapshots and source manifests."),
        ("etl", "Rebuild the official mixed_context ETL chain from raw snapshots to the modeling dataset."),
        ("feature_engineering", "Rebuild the official mixed_context feature-engineering outputs."),
        ("make_splits", "Create chronological mixed_context train/validation/test splits."),
    ]:
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "--mixed-context",
            action="store_true",
            help="Accepted for interface parity. These commands always operate on the official mixed_context scope.",
        )
        _add_reproducibility_args(subparser)
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove obsolete generated outputs while preserving the active state.")
    _add_recipe_runtime_args(cleanup_parser)
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which files would be removed without deleting them.",
    )
    platform_parser = subparsers.add_parser(
        "platform_run",
        help="Run the batch/offline customer platform flow for CU28.",
    )
    platform_parser.add_argument("--input", required=True, help="Path to the customer input CSV.")
    platform_parser.add_argument("--output", required=True, help="Directory where the run outputs will be written.")
    platform_parser.add_argument(
        "--platform-config",
        default="config/platform_config.yaml",
        help="Path to the platform configuration YAML.",
    )
    platform_parser.add_argument(
        "--max-cli-rows",
        type=int,
        default=20,
        help="Maximum number of recommendation rows to print in the CLI table.",
    )
    platform_parser.add_argument(
        "--show-all-recommendations",
        action="store_true",
        help="Print DO_NOT_BUY rows after BUY rows in the CLI recommendation table.",
    )
    return parser


def _resolve_reproducibility_mode(args: argparse.Namespace) -> str:
    from src.reproducibility import CACHED_MODE, FULL_MODE, SMOKE_MODE

    if args.full:
        return FULL_MODE
    if args.smoke:
        return SMOKE_MODE
    return CACHED_MODE


def _is_reproducibility_invocation(args: argparse.Namespace) -> bool:
    repro_only_commands = {"data_acquisition", "etl", "feature_engineering", "make_splits"}
    repro_shared_commands = {"train", "predict", "policy_simulation", "get_stats"}
    if args.command in repro_only_commands:
        return True
    return (
        args.command in repro_shared_commands
        and bool(getattr(args, "mixed_context", False))
        and not bool(getattr(args, "recipe_profile", None))
        and not bool(getattr(args, "all_recipes", False))
    )


def _run_reproducibility_command(args: argparse.Namespace) -> int:
    from src.data_acquisition import run_data_acquisition
    from src.evaluation import run_reproducibility_get_stats, run_reproducibility_policy_simulation
    from src.feature_engineering import run_feature_engineering
    from src.prediction import run_reproducibility_prediction
    from src.reproducibility import DEFAULT_CONFIG_PATH, build_reproducibility_config
    from src.reproducibility.workflows import run_reproducibility_etl, run_reproducibility_make_splits
    from src.training import run_reproducibility_training

    config = build_reproducibility_config(
        args.config or DEFAULT_CONFIG_PATH,
        mode=_resolve_reproducibility_mode(args),
        allow_synthetic_plant_layer=args.allow_synthetic_plant_layer,
    )
    logger, log_path = _resolve_logger(config, args.command)

    if args.command == "data_acquisition":
        result = run_data_acquisition(
            config,
            use_cached_raw=args.use_cached_raw,
            skip_download=args.skip_download,
            fail_on_missing_raw=args.fail_on_missing_raw,
        )
    elif args.command == "etl":
        result = run_reproducibility_etl(
            config,
            logger,
            force_download=not args.use_cached_raw and not args.skip_download,
        )
    elif args.command == "feature_engineering":
        result = run_feature_engineering(config, logger)
    elif args.command == "make_splits":
        result = run_reproducibility_make_splits(config, logger)
    elif args.command == "train":
        result = run_reproducibility_training(config, logger)
    elif args.command == "predict":
        result = run_reproducibility_prediction(config, logger, input_path=getattr(args, "input", None))
    elif args.command == "policy_simulation":
        result = run_reproducibility_policy_simulation(config, logger)
    elif args.command == "get_stats":
        result = run_reproducibility_get_stats(config, logger)
    else:  # pragma: no cover - parser guards this
        raise KeyError(f"Unsupported reproducibility command: {args.command}")

    payload = {
        "command": args.command,
        "scope": "mixed_context",
        "mode": _resolve_reproducibility_mode(args),
        "log_path": str(log_path),
        "result": result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _dispatch(command: str) -> Callable[[dict, object], dict]:
    if command == "data_processing":
        from src.data_processing import run_data_processing

        return run_data_processing
    if command == "train":
        from src.training import run_training

        return run_training
    if command == "predict":
        from src.predict import run_prediction

        return run_prediction
    if command == "get_stats":
        from src.get_stats import run_get_stats

        return run_get_stats
    if command == "policy_simulation":
        from src.training import run_policy_simulation

        return run_policy_simulation
    if command == "cleanup":
        from src.cleanup import run_cleanup

        return run_cleanup
    raise KeyError(f"Unsupported command: {command}")


def _resolve_logger(config: dict, stage_name: str):
    from src.utils import make_run_id, resolve_repo_path, runtime_stage_prefix, setup_stage_logger

    repo_root = Path(config["project"]["repo_root"])
    log_directory = resolve_repo_path(config["paths"]["logs_dir"], repo_root)
    run_id = make_run_id(runtime_stage_prefix(stage_name, config))
    return setup_stage_logger(stage_name, log_directory, run_id)


def _build_cleanup_runtime_config(base_config: dict) -> dict:
    cleanup_config = deepcopy(base_config)
    registry_cfg = cleanup_config.get("manufacturing_profiles", {})
    cleanup_config.setdefault("runtime", {})
    cleanup_config["runtime"]["recipe_context"] = {
        "selection_mode": "cleanup_all",
        "mode_resolution": "cleanup_all_scopes_default",
        "scope_type": "all_scopes",
        "scope_token": "all_scopes",
        "output_suffix": None,
        "recipe_profile": None,
        "manufacturing_context_profile": None,
        "recipe_slug": None,
        "recipe_registry_path": registry_cfg.get("registry_path"),
        "available_recipe_profiles": list(registry_cfg.get("available_recipe_profiles", [])),
        "available_manufacturing_profiles": list(registry_cfg.get("available_profile_names", [])),
    }
    return cleanup_config


def _format_cli_cell(value: object, column: str) -> str:
    import pandas as pd

    if pd.isna(value):
        text = ""
    elif column in CLI_NUMERIC_FORMATS:
        text = CLI_NUMERIC_FORMATS[column].format(float(value))
    else:
        text = str(value)

    if column == "decision_reason" and len(text) > 80:
        return text[:77] + "..."
    return text


def _format_cli_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "(no rows to display)"

    rendered_rows = [
        {column: _format_cli_cell(row.get(column), column) for column in columns}
        for row in rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rendered_rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    separator = "  ".join("-" * widths[column] for column in columns)
    body = [
        "  ".join(row[column].ljust(widths[column]) for column in columns)
        for row in rendered_rows
    ]
    return "\n".join([header, separator, *body])


def _build_purchase_summary_rows(recommendations) -> list[dict[str, object]]:
    import pandas as pd

    buy_rows = recommendations[recommendations["recommended_action"].eq("BUY")].copy()
    if buy_rows.empty:
        return [
            {
                "raw_material_id": "TOTAL",
                "destination_profiles": "",
                "buy_rows": 0,
                "total_order_quantity_tons": 0.0,
                "avg_purchase_trigger_proba": 0.0,
                "total_excess_tons": 0.0,
                "total_stockout_tons": 0.0,
            }
        ]

    summary_rows: list[dict[str, object]] = []
    for raw_material_id, group in buy_rows.groupby("raw_material_id", sort=True):
        destination_profiles = ", ".join(sorted(group["destination_profile"].astype(str).unique()))
        summary_rows.append(
            {
                "raw_material_id": raw_material_id,
                "destination_profiles": destination_profiles,
                "buy_rows": int(len(group)),
                "total_order_quantity_tons": float(pd.to_numeric(group["order_quantity_tons"], errors="coerce").sum()),
                "avg_purchase_trigger_proba": float(pd.to_numeric(group["purchase_trigger_proba"], errors="coerce").mean()),
                "total_excess_tons": float(pd.to_numeric(group["excess_tons"], errors="coerce").sum()),
                "total_stockout_tons": float(pd.to_numeric(group["stockout_tons"], errors="coerce").sum()),
            }
        )

    summary_rows = sorted(summary_rows, key=lambda row: row["total_order_quantity_tons"], reverse=True)
    summary_rows.append(
        {
            "raw_material_id": "TOTAL",
            "destination_profiles": "",
            "buy_rows": int(len(buy_rows)),
            "total_order_quantity_tons": float(pd.to_numeric(buy_rows["order_quantity_tons"], errors="coerce").sum()),
            "avg_purchase_trigger_proba": float(pd.to_numeric(buy_rows["purchase_trigger_proba"], errors="coerce").mean()),
            "total_excess_tons": float(pd.to_numeric(buy_rows["excess_tons"], errors="coerce").sum()),
            "total_stockout_tons": float(pd.to_numeric(buy_rows["stockout_tons"], errors="coerce").sum()),
        }
    )
    return summary_rows


def _print_recommended_purchases(
    result: dict[str, object],
    *,
    max_cli_rows: int,
    show_all_recommendations: bool,
) -> None:
    import pandas as pd

    recommendations_path = Path(result["output_paths"]["recommendations"])  # type: ignore[index]
    recommendations = pd.read_csv(recommendations_path)
    max_rows = max(1, int(max_cli_rows))

    buy_rows = recommendations[recommendations["recommended_action"].eq("BUY")].sort_values(
        ["order_quantity_tons", "purchase_trigger_proba"],
        ascending=[False, False],
    )
    blocked_rows = recommendations[recommendations["recommended_action"].eq("DO_NOT_BUY")].sort_values(
        ["purchase_trigger_proba", "raw_material_id"],
        ascending=[False, True],
    )

    print()
    print("==============================")
    print("PURCHASE SUMMARY BY RAW MATERIAL")
    print("==============================")
    print(_format_cli_table(_build_purchase_summary_rows(recommendations), CLI_PURCHASE_SUMMARY_COLUMNS))

    if show_all_recommendations:
        display_rows = pd.concat([buy_rows, blocked_rows], ignore_index=True)
    else:
        display_rows = buy_rows
        if display_rows.empty:
            display_rows = blocked_rows.head(3)

    limited_rows = display_rows.head(max_rows)

    print()
    print("==============================")
    print("RECOMMENDED PURCHASES")
    print("==============================")
    print(
        f"buy_recommendations={len(buy_rows)} "
        f"blocked_recommendations={len(blocked_rows)} "
        f"displayed_rows={len(limited_rows)}"
    )
    print(_format_cli_table(limited_rows[CLI_RECOMMENDATION_COLUMNS].to_dict(orient="records"), CLI_RECOMMENDATION_COLUMNS))
    if not show_all_recommendations and len(blocked_rows) > 0:
        print(
            "DO_NOT_BUY rows are summarized above. "
            "Use --show-all-recommendations to print blocked rows."
        )
    if len(display_rows) > len(limited_rows):
        print(f"Table limited to {max_rows} rows. Use --max-cli-rows N to change the limit.")

    print()
    print("==============================")
    print("INTERPRETATION")
    print("==============================")
    print(
        "`order_quantity_tons` es la cantidad final recomendada. "
        "Si `purchase_trigger_flag = 0`, la cantidad final se fuerza a 0.0. "
        "Las fuentes públicas/proxy aportan contexto; la decisión operativa se basa "
        "en el CSV de entrada o en supuestos sintéticos documentados."
    )

    print()
    print("==============================")
    print("DATA USE")
    print("==============================")
    print(
        "Este comando no reentrena el modelo. Ejecuta una recomendación sobre el CSV de entrada. "
        "El entrenamiento reproducible pertenece a la ruta `mixed_context`. "
        "Para entrenamiento industrial real se requieren históricos reales del cliente."
    )


def _print_platform_summary(
    result: dict[str, object],
    *,
    max_cli_rows: int = 20,
    show_all_recommendations: bool = False,
) -> None:
    print(f"platform_status={result['status']}")
    print(f"input_path={result['input_path']}")
    print(f"output_dir={result['output_dir']}")
    print(f"validation_report={result['validation_report_path']}")

    if result["status"] != "success":
        validation_report = result.get("validation_report", {})
        for error in validation_report.get("errors", []):
            print(f"validation_error={error}")
        return

    summary = result["summary"]
    print(f"row_count={summary['row_count']}")
    print(f"triggered_orders={summary['triggered_orders']}")
    print(f"total_order_quantity_tons={summary['total_order_quantity_tons']}")
    print(f"aggregate_excess_reduction_pct={summary['aggregate_excess_reduction_pct']}")
    print(f"aggregate_stockout_change_pct={summary['aggregate_stockout_change_pct']}")
    print(f"stockout_guardrail_pass={summary['stockout_guardrail_pass']}")
    for name, path in result["output_paths"].items():
        print(f"{name}={path}")
    _print_recommended_purchases(
        result,
        max_cli_rows=max_cli_rows,
        show_all_recommendations=show_all_recommendations,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "platform_run":
        result = run_platform_pipeline(
            input_path=args.input,
            output_dir=args.output,
            config_path=args.platform_config,
        )
        _print_platform_summary(
            result,
            max_cli_rows=args.max_cli_rows,
            show_all_recommendations=args.show_all_recommendations,
        )
        return 0 if result["status"] == "success" else 1

    if _is_reproducibility_invocation(args):
        return _run_reproducibility_command(args)

    if not args.config:
        parser.error("--config is required for legacy commands.")

    handler = _dispatch(args.command)
    from src.utils import build_recipe_runtime_configs, current_recipe_context, load_config

    base_config = load_config(args.config, recipe_config_path=args.recipe_config)
    if args.command == "cleanup" and not any([args.recipe_profile, args.all_recipes, args.mixed_context]):
        runtime_configs = [_build_cleanup_runtime_config(base_config)]
    else:
        runtime_configs = build_recipe_runtime_configs(
            base_config,
            recipe_profile=args.recipe_profile,
            all_recipes=args.all_recipes,
            mixed_context=args.mixed_context,
        )

    for config in runtime_configs:
        config.setdefault("runtime", {}).setdefault("cleanup", {})["dry_run"] = bool(getattr(args, "dry_run", False))

    results: list[dict[str, object]] = []
    for config in runtime_configs:
        logger, log_path = _resolve_logger(config, args.command)
        recipe_context = current_recipe_context(config)
        logger.info(
            "Resolved runtime context stage=%s selection_mode=%s mode_resolution=%s scope_token=%s recipe_profile=%s",
            args.command,
            recipe_context.get("selection_mode"),
            recipe_context.get("mode_resolution"),
            recipe_context.get("scope_token"),
            recipe_context.get("recipe_profile"),
        )

        try:
            result = handler(config, logger)
        except Exception as exc:
            logger.exception("Stage %s failed: %s", args.command, exc)
            raise

        logger.info("Stage %s completed successfully.", args.command)
        results.append(
            {
                "recipe_context": recipe_context,
                "log_path": str(log_path),
                "result": result,
            }
        )

    if len(results) == 1:
        payload = {"command": args.command, **results[0]}
    else:
        payload = {"command": args.command, "results": results}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
