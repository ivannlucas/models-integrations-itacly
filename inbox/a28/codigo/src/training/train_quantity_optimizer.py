from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation import compute_regression_metrics
from src.reproducibility.mixed_context import (
    QUANTITY_TARGET,
    apply_feature_fill_values,
    fit_feature_fill_values,
    leakage_audit,
    quantity_feature_columns,
    validate_feature_columns_for_stage,
)
from src.reproducibility.runtime import ensure_optional_dependency, official_paths
from src.utils import write_json


def _load_split_frames(splits_dir: Path) -> dict[str, pd.DataFrame]:
    mapping = {
        "train": splits_dir / "train.csv",
        "validation": splits_dir / "validation.csv",
        "test": splits_dir / "test.csv",
    }
    return {name: pd.read_csv(path) for name, path in mapping.items()}


def _prepare_features(frame: pd.DataFrame, feature_columns: list[str], fill_values: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    resolved_fill = fill_values or fit_feature_fill_values(frame, feature_columns)
    return apply_feature_fill_values(frame, feature_columns, resolved_fill), resolved_fill


def _purchase_trigger_active_rows(frame: pd.DataFrame, *, split_name: str) -> pd.DataFrame:
    if "purchase_trigger_label" not in frame.columns:
        raise ValueError(
            f"Quantity optimizer supervised evaluation requires purchase_trigger_label in split={split_name}."
        )
    active_mask = pd.to_numeric(frame["purchase_trigger_label"], errors="coerce").fillna(0).astype(int) == 1
    return frame.loc[active_mask].copy()


def _quantity_baseline_row(metrics: dict[str, Any], *, model_name: str, split_name: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "split": split_name,
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
        "n_rows": int(metrics.get("rows", metrics.get("n_samples", 0)) or 0),
    }


def run_quantity_optimizer_training(config: dict[str, Any], logger) -> dict[str, Any]:
    repo_root = Path(config["project"]["repo_root"])
    ensure_optional_dependency("sklearn", repo_root_path=repo_root)
    from src.training.model_registry import build_model

    paths = official_paths(config)
    split_frames = _load_split_frames(paths["splits_dir"])
    feature_columns = quantity_feature_columns(split_frames["train"])
    validate_feature_columns_for_stage(feature_columns, stage="quantity_optimizer")
    if not feature_columns:
        raise ValueError("No quantity optimizer feature columns were available in the modeling splits.")

    active_split_frames = {
        split_name: _purchase_trigger_active_rows(frame, split_name=split_name)
        for split_name, frame in split_frames.items()
    }
    train_frame = active_split_frames["train"]
    if train_frame.empty:
        raise ValueError("Quantity optimizer training requires at least one train row with purchase_trigger_label == 1.")

    x_train, fill_values = _prepare_features(train_frame, feature_columns)
    y_train = pd.to_numeric(train_frame[QUANTITY_TARGET], errors="coerce").fillna(0.0)

    if len(train_frame) < 4:
        model_family = "dummy"
        hyperparameters = {"strategy": "mean"}
    else:
        model_family = "ridge"
        hyperparameters = {"alpha": 1.0}

    model = build_model(model_family, hyperparameters)
    model.fit(x_train, y_train)
    dummy_model = build_model("dummy", {"strategy": "mean"})
    dummy_model.fit(x_train, y_train)

    metrics_payload: dict[str, Any] = {
        "scope": "mixed_context",
        "selection_criterion": "validation_rmse",
        "model_family": model_family,
        "target_column": QUANTITY_TARGET,
        "baseline_comparison": {
            "name": "baseline_order_quantity_tons",
            "column": "baseline_order_quantity_tons",
        },
        "feature_columns": feature_columns,
        "feature_imputation": "train_median_with_neutral_fallback_for_all_missing_columns",
        "evaluation_population": "rows_with_purchase_trigger_label_equal_1",
        "leakage_audit": leakage_audit(split_frames["train"], feature_columns, stage="quantity_optimizer").to_dict(orient="records"),
    }
    comparison_metrics: list[dict[str, Any]] = []

    for split_name, eval_frame in active_split_frames.items():
        if eval_frame.empty:
            metrics_payload[split_name] = {}
            for comparison_model_name in ("DummyRegressor", "Ridge"):
                comparison_metrics.append(
                    {
                        "model": comparison_model_name,
                        "split": split_name,
                        "rmse": None,
                        "mae": None,
                        "r2": None,
                        "n_rows": 0,
                    }
                )
            continue
        x_split, _ = _prepare_features(eval_frame, feature_columns, fill_values)
        y_true = pd.to_numeric(eval_frame[QUANTITY_TARGET], errors="coerce").fillna(0.0)
        y_pred = pd.Series(model.predict(x_split), index=eval_frame.index).clip(lower=0.0)
        split_metrics = compute_regression_metrics(
            y_true,
            y_pred,
            split=split_name,
            model=model_family,
            target=QUANTITY_TARGET,
        )
        baseline_prediction = pd.to_numeric(
            eval_frame["baseline_order_quantity_tons"],
            errors="coerce",
        ).fillna(0.0)
        split_metrics["baseline_comparison"] = compute_regression_metrics(
            y_true,
            baseline_prediction,
            split=split_name,
            model="baseline_order_quantity_tons",
            target=QUANTITY_TARGET,
            include_distribution=False,
        )
        metrics_payload[split_name] = split_metrics
        dummy_prediction = pd.Series(dummy_model.predict(x_split), index=eval_frame.index).clip(lower=0.0)
        dummy_metrics = compute_regression_metrics(
            y_true,
            dummy_prediction,
            split=split_name,
            model="DummyRegressor",
            target=QUANTITY_TARGET,
            include_mape=False,
            include_distribution=False,
        )
        ridge_metrics = compute_regression_metrics(
            y_true,
            y_pred,
            split=split_name,
            model="Ridge",
            target=QUANTITY_TARGET,
            include_mape=False,
            include_distribution=False,
        )
        comparison_metrics.extend(
            [
                _quantity_baseline_row(dummy_metrics, model_name="DummyRegressor", split_name=split_name),
                _quantity_baseline_row(ridge_metrics, model_name="Ridge", split_name=split_name),
            ]
        )

        prediction_path = paths["quantity_optimizer_predictions"][split_name]
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "actual": y_true,
                "prediction": y_pred,
                "baseline_prediction": baseline_prediction,
                "split": split_name,
                "target": QUANTITY_TARGET,
            }
        ).to_csv(prediction_path, index=False)

    metrics_payload["prediction_paths"] = {
        split_name: str(path.relative_to(repo_root).as_posix())
        for split_name, path in paths["quantity_optimizer_predictions"].items()
    }
    comparison_payload: dict[str, Any] = {
        "scope": "mixed_context",
        "component": "quantity_optimizer",
        "target": QUANTITY_TARGET,
        "target_column": QUANTITY_TARGET,
        "evaluation_filter": "purchase_trigger_label == 1",
        "training_filter": "DummyRegressor and Ridge are fit only on train rows where purchase_trigger_label == 1.",
        "baseline": "DummyRegressor(strategy='mean')",
        "official_model": "Ridge",
        "official_model_family": model_family,
        "selection_criterion": "validation_rmse",
        "test_usage": "test split is reserved for final evaluation only; it is not used for fitting or model selection.",
        "feature_columns": feature_columns,
        "excluded_feature_columns": [
            "quantity_optimizer_target_tons",
            "purchase_trigger_label",
        ],
        "leakage_audit": metrics_payload["leakage_audit"],
        "metrics": comparison_metrics,
    }

    artifact_payload = {
        "stage": "quantity_optimizer",
        "scope": "mixed_context",
        "model_family": model_family,
        "target_column": QUANTITY_TARGET,
        "feature_columns": feature_columns,
        "fill_values": fill_values,
        "feature_imputation": "train_median_with_neutral_fallback_for_all_missing_columns",
        "model": model,
        "metrics": metrics_payload,
    }
    paths["quantity_optimizer_artifact"].parent.mkdir(parents=True, exist_ok=True)
    with paths["quantity_optimizer_artifact"].open("wb") as handle:
        pickle.dump(artifact_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(paths["quantity_optimizer_metrics_json"], metrics_payload)
    write_json(paths["quantity_optimizer_baseline_comparison_json"], comparison_payload)
    pd.DataFrame(comparison_metrics).to_csv(paths["quantity_optimizer_baseline_comparison_csv"], index=False)
    logger.info("Saved quantity optimizer artifact to %s", paths["quantity_optimizer_artifact"])
    return {
        "artifact_path": str(paths["quantity_optimizer_artifact"]),
        "metrics_path": str(paths["quantity_optimizer_metrics_json"]),
        "baseline_comparison_json_path": str(paths["quantity_optimizer_baseline_comparison_json"]),
        "baseline_comparison_csv_path": str(paths["quantity_optimizer_baseline_comparison_csv"]),
        "feature_count": len(feature_columns),
        "model_family": model_family,
    }
