from pydantic import BaseModel, ConfigDict, Field

class PredictInlineRequest(BaseModel):
    model_key: str | None = Field(default=None)
    logret: float
    distsma12: float
    rsi14: float
    bollingerpos: float
    weeksin: float
    weekcos: float

class PredictInlineResponse(BaseModel):
    model_key: str
    predictions: dict