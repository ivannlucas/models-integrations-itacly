from pydantic import BaseModel, ConfigDict

class StatsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    metrics: dict
    artifacts_path: str