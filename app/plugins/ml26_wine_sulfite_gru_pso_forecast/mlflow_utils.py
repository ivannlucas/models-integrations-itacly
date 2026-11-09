"""MLflow helper for ml26 — download a user fine-tuned GRU-PSO model from MLflow."""

from __future__ import annotations

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import require_user_model
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.constants import MLFLOW_ARTIFACT_PATH
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.model_loader import LoadedModel, load_user_model


# None or an artifact load error, with run_id → UserModelUnavailableError (422)
@require_user_model
def download_user_model_from_mlflow(run_id: str) -> tuple[LoadedModel, str] | None:
    """Download a user fine-tuned model (model_state.pt + bundle_meta.json) from MLflow.

    Returns (model, temp_dir) or None if the run has no artifacts.
    Caller MUST remove temp_dir after inference (``removing_dir``). A load error (e.g. a
    missing bundle_meta.json) never leaves the temp dir behind.
    """

    def _load(local_path: str) -> LoadedModel:
        return load_user_model(local_path, source=f"mlflow:{run_id}")

    return download_run_artifacts(
        run_id, _load, prefix="mlflow_ml26_", artifact_path=MLFLOW_ARTIFACT_PATH
    )
