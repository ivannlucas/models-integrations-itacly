from pydantic import BaseModel


class TrainRequest(BaseModel):
    data_path: str
