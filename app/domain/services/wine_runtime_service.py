# app/domain/services/wine_runtime_service.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Union

import numpy as np


class ModelRepo(Protocol):
    def load(self, model_id: str | None = None) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class InlinePrediction:
    probability_up: float
    prediction: int


class RuntimeService:
    """
    Runtime de inferencia para el modelo de fluctuación del vino.

    Espera que model_repo.load() devuelva un bundle con, al menos:
      - "model": modelo sklearn-like con predict_proba(X)
      - "features": lista[str] con el orden exacto de columnas
    Opcional:
      - "scaler": scaler sklearn-like con transform(X)
      - "model_id": id del modelo
      - "threshold": float (si quieres guardarlo en el bundle)
    """

    def __init__(self, model_repo: ModelRepo, default_threshold: float = 0.5):
        self.model_repo = model_repo
        self.default_threshold = float(default_threshold)

    def _prepare_matrix(
        self, features_rows: List[Mapping[str, Any]], feature_schema: List[str]
    ) -> np.ndarray:
        # Validación de keys (faltantes / extras)
        required = set(feature_schema)

        for i, row in enumerate(features_rows):
            keys = set(row.keys())
            missing = required - keys
            if missing:
                raise ValueError(
                    f"Faltan features en la fila {i}: {sorted(missing)}. "
                    f"Esperadas: {feature_schema}"
                )

        x = np.array(
            [[float(row[f]) for f in feature_schema] for row in features_rows],
            dtype=float,
        )
        return x

    def predict_inline(
        self,
        features: Union[Mapping[str, Any], List[Mapping[str, Any]]],
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> Dict[str, Any]:
        """
        Predicción inline:
          - features: dict (1 fila) o list[dict] (muchas filas)
          - model_key: lo que tú llamas "model_key" en la API (se mapea a model_id del repo)
          - threshold: override del umbral (por defecto 0.5)

        Devuelve:
          {
            "model_id": "...",
            "predictions": {...}  # si 1 fila
          }
        o
          {
            "model_id": "...",
            "predictions": [ {...}, {...} ]  # si lista
          }
        """
        bundle = self.model_repo.load(model_id=model_key)

        model = bundle["model"]
        scaler = bundle.get("scaler")
        feature_schema = bundle["features"]

        # threshold: request > bundle > default
        thr = float(
            threshold
            if threshold is not None
            else bundle.get("threshold", self.default_threshold)
        )

        # Normaliza entrada
        multi = isinstance(features, list)
        rows: List[Mapping[str, Any]] = features if multi else [features]  # type: ignore[arg-type]

        x = self._prepare_matrix(rows, feature_schema)

        if scaler is not None:
            x = scaler.transform(x)

        # Probabilidad clase positiva (sube)
        # sklearn: predict_proba -> [P(class0), P(class1)]
        proba_up = model.predict_proba(x)[:, 1].astype(float)

        preds = (proba_up >= thr).astype(int)

        # formatea salida
        if not multi:
            pred_obj = InlinePrediction(probability_up=float(proba_up[0]), prediction=int(preds[0]))
            return {
                "model_id": bundle.get("model_id", model_key or "default"),
                "predictions": {
                    "probability_up": pred_obj.probability_up,
                    "prediction": pred_obj.prediction,
                },
            }

        pred_list = [
            {"probability_up": float(p), "prediction": int(y)}
            for p, y in zip(proba_up, preds)
        ]
        return {
            "model_id": bundle.get("model_id", model_key or "default"),
            "predictions": pred_list,
        }