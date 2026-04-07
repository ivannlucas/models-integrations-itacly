from pydantic import BaseModel


class StatsResponse(BaseModel):
    model_name: str
    model_type: str
    framework: str
    artifact_path: str
    input_schema: dict
    output_schema: dict
    predict_count: int
    last_predict_at: str | None
