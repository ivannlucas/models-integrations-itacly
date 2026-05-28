import logging
import types
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import NoValidSimulationPointError
from app.plugins.wine_sulphite.model_loader import load_artifacts
from app.plugins.wine_sulphite.postprocessing import (
    apply_operational_constraints,
    compute_molecular_so2,
    decode_bound_predictions,
    select_recommendation,
)
from app.plugins.wine_sulphite.preprocessing import (
    FEATURES_QUAL,
    build_simulation_grid,
    map_request_to_wine_dict,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "wine-sulphite"
MODEL_VERSION = "1.2.0"


class WineSulphitePlugin(ModelPluginPort):
    def __init__(self) -> None:
        self._model_qual: Any = None
        self._model_bound: Any = None
        self._metadata: dict = {}
        self._loaded: bool = False
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        self._model_qual, self._model_bound, self._metadata = load_artifacts()
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def _run_inference(self, features: dict) -> dict:
        mae_quality: float = (
            self._metadata.get("metrics", {}).get("quality_cv", {}).get("mae_mean", 0.427)
        )
        mae_bound: float = (
            self._metadata.get("metrics", {}).get("bound_cv", {}).get("mae_mean", 14.5)
        )

        req_ns = types.SimpleNamespace(**features)
        base_wine = map_request_to_wine_dict(req_ns)
        free_targets, qual_rows, bound_rows = build_simulation_grid(
            base_wine, features["delta_max"]
        )

        raw_bound_pred = self._model_bound.predict(bound_rows)
        pred_bounds = decode_bound_predictions(raw_bound_pred, free_targets)

        pred_totals = np.maximum(free_targets + pred_bounds, free_targets)
        qual_rows = qual_rows.copy()
        qual_rows["total sulfur dioxide"] = pred_totals

        pred_qualities = self._model_qual.predict(qual_rows[FEATURES_QUAL])
        molecular_so2 = compute_molecular_so2(free_targets, base_wine["pH"])

        baseline_idx = int(np.argmin(np.abs(free_targets - base_wine["free sulfur dioxide"])))
        baseline_quality = float(pred_qualities[baseline_idx])

        try:
            valid_free, valid_bounds, valid_totals, valid_moleculars, valid_qualities = (
                apply_operational_constraints(
                    free_targets,
                    pred_bounds,
                    pred_totals,
                    molecular_so2,
                    pred_qualities,
                    features["min_molecular"],
                    features["max_total"],
                )
            )
        except ValueError as exc:
            raise NoValidSimulationPointError(str(exc)) from exc

        rec_idx, reason, intervention = select_recommendation(
            valid_free,
            valid_bounds,
            valid_totals,
            valid_moleculars,
            valid_qualities,
            baseline_quality,
            mae_quality,
        )

        return {
            "mae_quality": mae_quality,
            "mae_bound": mae_bound,
            "valid_free": valid_free,
            "valid_bounds": valid_bounds,
            "valid_totals": valid_totals,
            "valid_moleculars": valid_moleculars,
            "valid_qualities": valid_qualities,
            "baseline_quality": baseline_quality,
            "rec_idx": rec_idx,
            "reason": reason,
            "intervention": intervention,
        }

    def predict_batch(self, *, data_path: str) -> dict:
        df = pd.read_csv(data_path, sep=None, engine="python")
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]

        predictions = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict.setdefault("min_molecular", 0.6)
            row_dict.setdefault("max_total", 200.0)
            row_dict.setdefault("delta_max", 40.0)
            try:
                res = self._run_inference(row_dict)
                i = res["rec_idx"]
                predictions.append(
                    {
                        "row": int(idx),
                        "intervention_recommended": res["intervention"],
                        "recommended_free_so2": float(res["valid_free"][i]),
                        "predicted_quality": float(res["valid_qualities"][i]),
                        "recommendation_reason": res["reason"],
                    }
                )
            except Exception as exc:
                predictions.append({"row": int(idx), "error": str(exc)})

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info(
            "predict_batch done — %d predictions count=%d", len(predictions), self._predict_count
        )

        return {
            "model_id": MODEL_NAME,
            "predictions": predictions,
            "output_path": None,
        }

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> dict:
        res = self._run_inference(features)
        i = res["rec_idx"]

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info(
            "predict_inline done — intervention=%s rec_free=%.1f quality=%.3f count=%d",
            res["intervention"],
            float(res["valid_free"][i]),
            float(res["valid_qualities"][i]),
            self._predict_count,
        )

        return {
            "model_id": MODEL_NAME,
            "threshold": threshold,
            "prediction": res["intervention"],
            "confidence": float(res["valid_qualities"][i]),
            "features_used": list(FEATURES_QUAL),
            "recommended_free_so2": float(res["valid_free"][i]),
            "recommended_bound_so2": float(res["valid_bounds"][i]),
            "recommended_total_so2": float(res["valid_totals"][i]),
            "recommended_molecular_so2": float(res["valid_moleculars"][i]),
            "predicted_quality": float(res["valid_qualities"][i]),
            "baseline_predicted_quality": res["baseline_quality"],
            "recommendation_reason": res["reason"],
            "intervention_recommended": res["intervention"],
            "mae_quality": res["mae_quality"],
            "mae_bound": res["mae_bound"],
        }

    def stats(self) -> StatsResponse:
        return StatsResponse(
            model_name=MODEL_NAME,
            model_type="RandomForestRegressor (dual: quality + bound SO2)",
            framework="sklearn",
            artifact_path="model-runtime-wine_sulphite/artifacts/",
            input_schema={
                "mode=inline": {
                    "fixed_acidity": "float (g/dm³)",
                    "volatile_acidity": "float (g/dm³)",
                    "citric_acid": "float (g/dm³)",
                    "residual_sugar": "float (g/dm³)",
                    "chlorides": "float (g/dm³)",
                    "density": "float (g/cm³)",
                    "pH": "float",
                    "sulphates": "float (g/dm³)",
                    "alcohol": "float (% vol.)",
                    "free_sulfur_dioxide": "float (mg/L)",
                    "total_sulfur_dioxide": "float (mg/L)",
                    "min_molecular": "float (mg/L, default 0.6)",
                    "max_total": "float (mg/L, default 200.0)",
                    "delta_max": "float (mg/L, default 40.0)",
                },
                "mode=batch": {
                    "data_path": "str — path to CSV with wine physicochemical properties"
                },
            },
            output_schema={
                "batch": {"predictions": "list[dict] — per-wine recommendations"},
                "inline": {
                    "recommended_free_so2": "float (mg/L)",
                    "recommended_bound_so2": "float (mg/L)",
                    "recommended_total_so2": "float (mg/L)",
                    "recommended_molecular_so2": "float (mg/L)",
                    "predicted_quality": "float (0–10)",
                    "baseline_predicted_quality": "float (0–10)",
                    "recommendation_reason": "str",
                    "intervention_recommended": "bool",
                    "mae_quality": "float (~0.427)",
                    "mae_bound": "float (~14.5 mg/L)",
                },
            },
            predict_count=self._predict_count,
            last_predict_at=self._last_predict_at,
        )
