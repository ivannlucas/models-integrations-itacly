from fastapi import APIRouter, Depends
from app.application.dto.train import TrainRequest, TrainResponse
from app.infrastructure.http.dependencies.container import get_train_uc

router = APIRouter()

@router.post("/train", response_model=TrainResponse)
def train(req: TrainRequest, uc=Depends(get_train_uc)):
    result = uc.execute(req.data_path)
    return TrainResponse(model_id=result.model_id, metrics=result.metrics, artifacts_path=result.artifacts_path)