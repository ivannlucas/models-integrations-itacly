from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "CSV de ciclos a 10 Hz etiquetados: Cycle_ID, Time_Segundos, PS1, PS3, EPS1, FS1, TS1, "
            "TS2, VS1 y Target_Fouling, Target_Valvula, Target_Bomba, Target_Acumulador "
            "(0=Sano, 1=Warning, 2=Crítico). Se separa 70/15/15 por Cycle_ID; las métricas son "
            "las del 15% de test."
        ),
    )
    mlflow_run_id: MlflowRunId
    mode: Literal["fine_tune", "full"] = Field(
        default="fine_tune",
        description=(
            "fine_tune (por defecto): calibra las 4 cabezas del modelo servido con datos de planta, "
            "backbone CNN congelado (fine_tuner.py del equipo de IA; bastan unos cientos de ciclos). "
            "full: reentrena la CNN desde cero con los hiperparámetros de Optuna (trainer.py; "
            "pensado para miles de ciclos — con pocos ciclos da un modelo peor que el servido)."
        ),
    )


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
    mode: str
    exact_match: float
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    n_train: int
    n_val: int
    n_test: int
    epochs_run: int
    training_time_s: float
    mlflow_run_id: str = ""
    upload_warning: str | None = None
