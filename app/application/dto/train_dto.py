from pydantic import BaseModel


class TrainRequest(BaseModel):
    """Default training request body used when a plugin does not supply its own."""

    data_path: str
