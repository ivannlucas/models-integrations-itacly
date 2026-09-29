"""Pydantic DTOs for the ml26 (wine sulfite GRU-PSO) /train endpoint."""
# pylint: disable=too-few-public-methods  # Pydantic DTOs carry no behaviour

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Fine-tune request: CSV in the AI team's sequential format (data/raw/sequential.csv.gz)."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "CSV con una fila por paso de 2 h y lote: lot_id, timestamp_index, wine_type, "
            "volume_l, ambient_temp_c, "
            "stage, temperature_c, dissolved_oxygen_mg_l, density_g_ml, co2_g_l, stage_progress "
            "(obligatorio, sin vacíos en filas operativas) y free_sulfite_mg_l (SO2 libre "
            "continuo, del que se derivan los targets a 72 h). Recomendadas: ph_lab, "
            "free_sulfite_lab, total_sulfite_lab, actual_dose_mg_l, hours_since_last_lab, "
            "lab_sample, target_only. "
            "Se necesitan al menos 7 lotes (partición 70/15/15 por lot_id)."
        ),
    )
    mlflow_run_id: str = ""


class TrainResponse(BaseModel):
    """Fine-tune response: validation metrics comparable to memoria Tabla 10 (+ test if the CSV
    allows it).
    """

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    n_lots_train: int
    n_lots_val: int
    n_lots_test: int
    n_windows_train: int
    n_windows_val: int
    epochs_run: int
    best_epoch: int
    val_rmse_future_free_sulfite_72h: float
    val_mae_future_free_sulfite_72h: float
    val_rmse_underprotection_risk_72h: float
    val_mae_underprotection_risk_72h: float
    val_overall_rmse: float
    val_overall_mae: float
    test_rmse_future_free_sulfite_72h: Optional[float] = None
    test_mae_future_free_sulfite_72h: Optional[float] = None
    test_rmse_underprotection_risk_72h: Optional[float] = None
    test_mae_underprotection_risk_72h: Optional[float] = None
    local_artifact_dir: str = Field(
        ...,
        description="Carpeta local del modelo reentrenado (nunca sobrescribe el artefacto fijo)",
    )
    upload_warning: Optional[str] = Field(
        default=None, description="Informativo si falló la subida a MLflow"
    )
