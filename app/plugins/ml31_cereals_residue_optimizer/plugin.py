"""Ml31CerealsResidueOptimizer — surrogate MLPRegressor estimating available soil residue."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError
from app.plugins.ml31_cereals_residue_optimizer.constants import (
    FRAMEWORK,
    MODEL_ID,
    REQUIRED_COLS,
    TRAIN_TARGET_COL,
    VERSION,
)
from app.plugins.ml31_cereals_residue_optimizer.model_loader import load_pipeline
from app.plugins.ml31_cereals_residue_optimizer.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml31_cereals_residue_optimizer.train_dto import TrainResponse

logger = logging.getLogger(__name__)


class Ml31CerealsResidueOptimizerPlugin(ModelPluginPort):
    """Regression plugin estimating available cereal-crop soil residue (tons)."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._pipe = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load the surrogate pipeline."""
        self._pipe = load_pipeline()
        logger.info("Ml31CerealsResidueOptimizerPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True if the pipeline is loaded."""
        return self._pipe is not None

    def _require_loaded(self) -> None:
        """Raise ModelNotLoadedError if the model is not loaded."""
        if self._pipe is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self) -> None:
        """Update runtime counters after a prediction."""
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> PredictInlineResponse:
        """Predict available residue for a single cereal scenario."""
        self._require_loaded()
        df = pd.DataFrame([{c: features[c] for c in REQUIRED_COLS}])
        prediction = float(self._pipe.predict(df)[0])
        self._record()
        return PredictInlineResponse(
            model_id=MODEL_ID,
            prediction=prediction,
            confidence=None,
            xai_feature_values={c: features[c] for c in REQUIRED_COLS if c != "Cultivo"},
        )

    def predict_batch(self, *, data_path: str) -> PredictBatchResponse:
        """Predict available residue for every row of a CSV."""
        self._require_loaded()
        df = pd.read_csv(data_path)
        predictions: list[dict] = []
        for idx, row in df.iterrows():
            try:
                row_df = pd.DataFrame([{c: row[c] for c in REQUIRED_COLS}])
                predictions.append({
                    "row": int(idx),
                    "prediction": float(self._pipe.predict(row_df)[0]),
                    "Cultivo": str(row["Cultivo"]),
                })
            except Exception as exc:
                logger.warning("Error en fila %s: %s", idx, exc)
                predictions.append({"row": int(idx), "error": str(exc)})
        self._record()
        return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)

    def train(self, *, data_path: str) -> TrainResponse:  # pylint: disable=too-many-locals
        """Re-fit the surrogate pipeline on a CSV and persist the artifact."""
        from sklearn.compose import ColumnTransformer  # pylint: disable=import-outside-toplevel
        from sklearn.metrics import r2_score  # pylint: disable=import-outside-toplevel
        from sklearn.model_selection import train_test_split  # pylint: disable=import-outside-toplevel
        from sklearn.neural_network import MLPRegressor  # pylint: disable=import-outside-toplevel
        from sklearn.pipeline import Pipeline  # pylint: disable=import-outside-toplevel
        from sklearn.preprocessing import OneHotEncoder, StandardScaler  # pylint: disable=import-outside-toplevel
        import joblib  # pylint: disable=import-outside-toplevel

        from app.plugins.ml31_cereals_residue_optimizer.constants import (  # pylint: disable=import-outside-toplevel
            CATEGORICAL_FEATURES, MODEL_FILENAME, NUMERIC_FEATURES,
        )
        from app.plugins.ml31_cereals_residue_optimizer.model_loader import _store  # pylint: disable=import-outside-toplevel

        df = pd.read_csv(data_path)
        missing = [c for c in REQUIRED_COLS + [TRAIN_TARGET_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV falta columnas requeridas: {missing}")

        x = df[REQUIRED_COLS].copy()
        y = df[TRAIN_TARGET_COL].copy()
        if "Año" in df.columns:
            years = sorted(df["Año"].unique())
            split_year = years[int(len(years) * 0.8)]
            mask = df["Año"] < split_year
            x_train, x_test, y_train, y_test = x[mask], x[~mask], y[mask], y[~mask]
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )

        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=500, random_state=42,
                early_stopping=True, validation_fraction=0.1,
            )),
        ])
        pipe.fit(x_train, y_train)
        r2 = float(r2_score(y_test, pipe.predict(x_test)))

        _store.local_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, _store.local_dir / MODEL_FILENAME)
        upload_warning: str | None = None
        try:
            _store.upload(MODEL_FILENAME)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            upload_warning = f"Artefacto guardado en local; fallo en S3: {exc}"
            logger.warning(upload_warning)

        self.load()
        logger.info("train() done — r2=%.4f", r2)
        return TrainResponse(
            detail="Entrenamiento completado",
            r2_test=round(r2, 4),
            n_train=int(len(x_train)),
            n_test=int(len(x_test)),
            upload_warning=upload_warning,
        )

    def stats(self) -> StatsResponse:
        """Return model metadata and runtime statistics."""
        avg = None  # latency not tracked
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Estimación del residuo vegetal disponible en suelo para cultivos cerealistas "
                "mediante surrogate MLPRegressor."
            ),
            task_type="regression",
            framework=FRAMEWORK,
            inputs=[
                InputField(name="Sup_Secano_ha", type="float",
                           description="Superficie en secano (ha)"),
                InputField(name="Sup_Regadio_ha", type="float",
                           description="Superficie en regadío (ha)"),
                InputField(name="Lluvia_Primavera_mm", type="float",
                           description="Precipitación primaveral (mm)"),
                InputField(name="Sequia_Primavera", type="int", default=0,
                           description="1 si lluvia < 200mm (sequía), 0 si no"),
                InputField(name="Cultivo", type="str",
                           description="Tipo de cultivo: Trigo, Cebada, Maíz, Girasol, etc."),
            ],
            outputs=[
                OutputField(name="prediction", type="float",
                            description="Residuo disponible predicho (toneladas)"),
            ],
            metrics={},
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=avg),
        )
