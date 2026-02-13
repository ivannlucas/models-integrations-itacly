from pydantic import BaseModel, ConfigDict

class TrainRequest(BaseModel):
    data_path: str  

class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    metrics: dict
    artifacts_path: str