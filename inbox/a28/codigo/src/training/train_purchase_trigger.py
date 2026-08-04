from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation import compute_trigger_metrics
from src.reproducibility.mixed_context import (
    TRIGGER_TARGET,
    apply_feature_fill_values,
    fit_feature_fill_values,
    leakage_audit,
    trigger_feature_columns,
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


def run_purchase_trigger_training(config: dict[str, Any], logger) -> dict[str, Any]:
    repo_root = Path(config["project"]["repo_root"])
    ensure_optional_dependency("sklearn", repo_root_path=repo_root)
    from src.training.model_registry import build_classifier

    paths = official_paths(config)
    split_frames = _load_split_frames(paths["splits_dir"])
    feature_columns = trigger_feature_columns(split_frames["train"])
    validate_feature_columns_for_stage(feature_columns, stage="trigger")
    if not feature_columns:
        raise ValueError("No trigger feature columns were available in the modeling splits.")

    x_train, fill_values = _prepare_features(split_frames["train"], feature_columns)
    y_train = pd.to_numeric(split_frames["train"][TRIGGER_TARGET], errors="coerce").fillna(0).astype(int)
    if y_train.nunique() < 2:
        model_family = "dummy"
        hyperparameters = {"strategy": "most_frequent"}
    else:
        model_family = "logistic_regression"
        hyperparameters = {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": int(config.get("project", {}).get("seed", 42)),
        }

    model = build_classifier(model_family, hyperparameters)
    model.fit(x_train, y_train)

    metrics_payload: dict[str, Any] = {
        "scope": "mixed_context",
        "selection_criterion": "validation_false_negative_rate_then_accuracy",
        "model_family": model_family,
        "target_column": TRIGGER_TARGET,
        "feature_columns": feature_columns,
        "feature_imputation": "train_median_with_neutral_fallback_for_all_missing_columns",
        "leakage_audit": leakage_audit(split_frames["train"], feature_columns, stage="trigger").to_dict(orient="records"),
    }

    for split_name, frame in split_frames.items():
        x_split, _ = _prepare_features(frame, feature_columns, fill_values)
        y_true = pd.to_numeric(frame[TRIGGER_TARGET], errors="coerce").fillna(0).astype(int)
        y_pred = pd.Series(model.predict(x_split), index=frame.index)
        if hasattr(model, "predict_proba"):
            y_proba = pd.Series(model.predict_proba(x_split)[:, 1], index=frame.index)
        else:
            y_proba = y_pred.astype(float)
        metrics_payload[split_name] = compute_trigger_metrics(
            y_true,
            y_pred,
            y_proba,
            split=split_name,
        )
        prediction_path = paths["trigger_predictions"][split_name]
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "actual": y_true,
                "prediction": y_pred,
                "probability_buy": y_proba,
                "split": split_name,
            }
        ).to_csv(prediction_path, index=False)

    artifact_payload = {
        "stage": "purchase_trigger",
        "scope": "mixed_context",
        "model_family": model_family,
        "target_column": TRIGGER_TARGET,
        "feature_columns": feature_columns,
        "fill_values": fill_values,
        "feature_imputation": "train_median_with_neutral_fallback_for_all_missing_columns",
        "model": model,
        "metrics": metrics_payload,
    }
    metrics_payload["prediction_paths"] = {
        split_name: str(path.relative_to(repo_root).as_posix())
        for split_name, path in paths["trigger_predictions"].items()
    }
    paths["purchase_trigger_artifact"].parent.mkdir(parents=True, exist_ok=True)
    with paths["purchase_trigger_artifact"].open("wb") as handle:
        pickle.dump(artifact_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(paths["trigger_metrics_json"], metrics_payload)
    logger.info("Saved purchase trigger artifact to %s", paths["purchase_trigger_artifact"])
    return {
        "artifact_path": str(paths["purchase_trigger_artifact"]),
        "metrics_path": str(paths["trigger_metrics_json"]),
        "feature_count": len(feature_columns),
        "model_family": model_family,
    }
