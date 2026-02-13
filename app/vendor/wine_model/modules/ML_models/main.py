"""
Command-line entry point for the wine price forecasting project.

Usage examples
--------------
Run the full pipeline (train + evaluate on last 24 weeks):
    python -m modules.ML_models.main

Explicitly run the full pipeline:
    python -m modules.ML_models.main train_and_evaluate

Train models with walk-forward validation and save production artifacts:
    python -m modules.ML_models.main train

Evaluate the production model on the last 24 weeks of the main dataset:
    python -m modules.ML_models.main evaluate_test

Run predictions on a new CSV and print a preview:
    python -m modules.ML_models.main predict --input data/raw/new_wine_prices.csv

Run predictions and save the output with probabilities:
    python -m modules.ML_models.main predict \
        --input data/raw/new_wine_prices.csv \
        --output data/processed/new_wine_prices_with_preds.csv

Description
-----------
Provides three main workflows:

- train:
    * Load the main raw CSV.
    * Run walk-forward validation to compare Logistic and XGBoost.
    * Select the best model type based on validation AUC.
    * Train final production models on all train+validation data.
    * Save artifacts to models/prod/ and print validation metrics.

- evaluate_test:
    * Load production artifacts.
    * Evaluate the final model on the last 24 weeks of the historical dataset.
    * Print test metrics.

- predict:
    * Load production artifacts.
    * Run inference on a user-provided CSV file with the same schema as the
      training data.
    * Print a short summary and optionally save predictions to a CSV.

If run without arguments, the script executes the full pipeline:
train_and_evaluate = train + evaluate_test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint
import datetime
import json

from modules.common.data import load_raw_data, clean_and_index_data
from modules.common.config import RAW_DATA_DIR, DATA_CONFIG, OUTPUT_DIR, MODEL_CONFIG
from .training import train_with_walk_forward, train_final_models
from .inference import evaluate_on_last_weeks, predict_from_csv


def _cmd_train() -> None:
    """Train models with walk-forward validation and save production artifacts."""
    raw_path = RAW_DATA_DIR / DATA_CONFIG.raw_filename
    print(f"[train] Loading raw data from: {raw_path}")
    df_raw = load_raw_data(raw_path)
    df_price = clean_and_index_data(df_raw)

    print("[train] Running walk-forward validation...")
    results = train_with_walk_forward(df_price)
    best_model_type = results["best_model_type"]
    print(f"[train] Best model type by validation AUC: {best_model_type}")
    print("[train] Validation metrics (Logistic):")
    pprint(results["metrics_logreg"])
    print("[train] Validation metrics (XGBoost):")
    pprint(results["metrics_xgboost"])

    print("[train] Training final production models...")
    train_final_models(df_price, best_model_type=best_model_type)
    print("[train] Training completed. Artifacts saved under models/prod/.")


def _cmd_evaluate_test() -> None:
    """Evaluate production model and SAVE metrics/predictions to output/."""
    print(f"[evaluate_test] Evaluating production model on last {MODEL_CONFIG.test_size} weeks...")

    # Crear carpeta output si no existe
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = evaluate_on_last_weeks()

    # 1. Save metrics to JSON
    metrics_path = OUTPUT_DIR / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results["metrics"], f, indent=4)

    # 2. Save detailed predictions to CSV
    preds_path = OUTPUT_DIR / "test_predictions.csv"
    results["predictions_df"].to_csv(preds_path, index=True)

    print(f"[evaluate_test] ✅ Metrics saved to: {metrics_path}")
    print(f"[evaluate_test] ✅ Detailed predictions saved to: {preds_path}")

    print("[evaluate_test] Model type:", results["model_type"])
    print("[evaluate_test] Test metrics:")
    pprint(results["metrics"])


def _cmd_predict(input_path: str, output_path: str | None) -> None:
    """Run predictions and save to output folder if no path provided."""
    input_path = Path(input_path)
    print(f"[predict] Loading input CSV from: {input_path}")

    df_preds = predict_from_csv(input_path)

    # Automatic saving logic
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Generate timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{input_path.stem}_{timestamp}.csv"
        output_path = OUTPUT_DIR / filename
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_preds.to_csv(output_path, index=True)

    print("[predict] Predictions computed.")
    print(f"[predict] ✅ Results saved to: {output_path}")
    print("Example rows:")
    print(df_preds[["pred_proba_up"]].tail())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wine price forecasting project CLI (ML models)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # train
    subparsers.add_parser(
        "train",
        help="Run walk-forward validation and train final production models",
    )

    # evaluate_test
    subparsers.add_parser(
        "evaluate_test",
        help="Evaluate production model on last 24 weeks of the main dataset",
    )

    # predict
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run predictions on a new CSV file",
    )
    predict_parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file",
    )
    predict_parser.add_argument(
        "--output",
        required=False,
        help="Optional path to save the output CSV with predictions",
    )

    # Explicit full pipeline command
    subparsers.add_parser(
        "train_and_evaluate",
        help="Run full pipeline: train models and evaluate on last 24 weeks",
    )

    args = parser.parse_args()

    # Default behavior: if no command is given, run full pipeline
    if args.command is None:
        print("[main] No command provided. Running full train_and_evaluate pipeline.")
        _cmd_train()
        _cmd_evaluate_test()
        return

    if args.command == "train":
        _cmd_train()
    elif args.command == "evaluate_test":
        _cmd_evaluate_test()
    elif args.command == "predict":
        _cmd_predict(input_path=args.input, output_path=args.output)
    elif args.command == "train_and_evaluate":
        _cmd_train()
        _cmd_evaluate_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
