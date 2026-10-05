from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "CSV a 10 Hz con Cycle_ID, Time_Segundos, PS1, PS3, EPS1, FS1, TS1, TS2, VS1 y las 4 etiquetas "
            "Target_Fouling, Target_Valvula, Target_Bomba, Target_Acumulador (0=Sano, 1=Warning, 2=Crítico)"
        ),
    )
    mlflow_run_id: str


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
    exact_match: float
    accuracy: float
    f1_macro: float
    recall_macro: float
    n_train: int
    n_test: int
    training_time_s: float
    upload_warning: str | None = None
