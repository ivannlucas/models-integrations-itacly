from __future__ import annotations

import json
import subprocess
import joblib
from pathlib import Path
from typing import Optional

from app.domain.ports.ml_plugin_port import MLPluginPort, PredictResult, TrainResult, StatsResult
from app.domain.services.wine_runtime_service import RuntimeService


class WinePricePlugin(MLPluginPort):
    def __init__(self) -> None:
        self.base = Path(__file__).resolve().parents[3] / "vendor" / "wine_model"
        self.output_dir = self.base / "output"
        self.models_dir = self.base / "models" / "prod"


        class _InlineRepo:
            def __init__(self, models_dir: Path):
                self.models_dir = models_dir
                self._bundle = None

            def load(self, model_id: str | None = None):
                if self._bundle is None:
                    model = joblib.load(self.models_dir / "ml_model.pkl")

                    scaler_path = self.models_dir / "scaler.pkl"
                    schema_path = self.models_dir / "feature_schema.json"

                    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
                    schema = json.loads(schema_path.read_text("utf-8"))

                    if isinstance(schema, dict) and "feature_columns" in schema:
                        features = schema["feature_columns"]
                    else:
                        features = schema

                    self._bundle = {
                        "model_id": model_id or "wine-prod",
                        "model": model,
                        "scaler": scaler,
                        "features": features,
                    }

                return self._bundle

        repo = _InlineRepo(self.models_dir)
        self._inline_service = RuntimeService(repo)

    def _run(self, args: list[str]) -> None:
        subprocess.run(["python", *args], cwd=str(self.base), check=True)

    def _normalize_to_wine_path(self, p: str) -> str:
        """
        Convierte rutas del request a una ruta que exista.
        Admite:
          - "data/raw/xxx.csv" (relativa a wine_model)
          - "app/vendor/wine_model/data/raw/xxx.csv" (relativa al root del repo)
          - "/app/vendor/wine_model/data/raw/xxx.csv" (estilo Docker)
        Devuelve siempre una ruta relativa a wine_model (para pasar al CLI).
        """
        raw = Path(p)

        # Caso Docker (/app/...)
        if str(raw).startswith("/app/"):
            raw = Path(str(raw).replace("/app/", "", 1))

        # Caso "app/vendor/wine_model/..."
        prefix = Path("app/vendor/wine_model")
        try:
            rel = raw.relative_to(prefix)
            return rel.as_posix()  # e.g. data/raw/file.csv
        except ValueError:
            pass

        # Si ya es relativa tipo data/raw/..., la devolvemos tal cual
        return raw.as_posix()

    def predict(self, *, data_path: str, output_path: Optional[str] = None) -> PredictResult:
        in_path = self._normalize_to_wine_path(data_path)

        out = Path(output_path) if output_path else (self.output_dir / "preds.csv")
        out_path = self._normalize_to_wine_path(str(out))

        scaler = self.base / "models" / "prod" / "scaler.pkl"
        if not scaler.exists():
            # Entrena automáticamente si falta el modelo
            self._run(["-m", "modules.ML_models.main", "train_and_evaluate"])

        self._run([
            "-m", "modules.ML_models.main",
            "predict",
            "--input", in_path,
            "--output", out_path,
        ])

        # devolvemos output_path 
        return PredictResult(
            model_id="wine-prod",
            predictions=[],
            output_path=str(self.base / out_path).replace(str(Path.cwd()) + "/", ""),
        )
    

    def predict_inline(self, features: dict, model_key: str | None = None):
        return self._inline_service.predict_inline(features, model_key=model_key)

    def train(self, *, data_path: str) -> TrainResult:
        self._run(["-m", "modules.ML_models.main", "train_and_evaluate"])

        metrics_path = self.output_dir / "test_metrics.json"
        metrics = json.loads(metrics_path.read_text("utf-8")) if metrics_path.exists() else {}
        return TrainResult(model_id="wine-prod", metrics=metrics, artifacts_path=str(self.models_dir))

    def get_stats(self) -> StatsResult:
        metrics_path = self.output_dir / "test_metrics.json"
        metrics = json.loads(metrics_path.read_text("utf-8")) if metrics_path.exists() else {}
        return StatsResult(model_id="wine-prod", metrics=metrics, artifacts_path=str(self.models_dir))