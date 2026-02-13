from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any

class PredictRequest(BaseModel):
    data_path: str
    output_path: Optional[str] = None

class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    output_path: Optional[str] = None
    predictions: List[Dict[str, Any]] = []