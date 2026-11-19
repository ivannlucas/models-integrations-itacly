"""Pydantic request/response DTOs for the ml37 /train endpoint — retrains the LSTM multitask
model, faithful to src/training/trainer.py (random_stratified_cooler split, same hyperparams)."""
from pydantic import Field

from app.application.dto.plugin_dto import PluginDTO, TrainRequestBase
from app.plugins.ml37_wine_press_predictive_maintenance.constants import PHYSICAL_SENSORS


class TrainRequest(TrainRequestBase):
    """Retrain the LSTM multitask model on a client-supplied long-format sensor table."""

    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// a un CSV/parquet en formato long: una fila por lectura, "
            "columnas cycle_id, t, " + ", ".join(PHYSICAL_SENSORS) + ", y_cooler, y_valve, "
            "y_pump, y_accumulator[, y_stable opcional]. Se divide 70/15/15 con "
            "random_stratified_cooler (seed=42), igual que el procedimiento original."
        ),
    )


class TrainResponse(PluginDTO):
    """Test-split metrics per head, same protocol as trainer.py (balanced_accuracy, f1_macro)."""

    detail: str
    accuracy: dict[str, float] = Field(..., description="accuracy de test por head")
    balanced_accuracy: dict[str, float] = Field(
        ..., description="balanced_accuracy de test por head"
    )
    f1_macro: dict[str, float] = Field(..., description="f1_macro de test por head")
    n_train: int
    n_val: int
    n_test: int
    epochs_run: int
    upload_warning: str | None = None
