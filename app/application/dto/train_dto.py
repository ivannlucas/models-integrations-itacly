from pydantic import BaseModel


class TrainRequest(BaseModel):
    data_path: str = ""


class TrainResponse(BaseModel):
    detail: str
    metrics: dict = {}
