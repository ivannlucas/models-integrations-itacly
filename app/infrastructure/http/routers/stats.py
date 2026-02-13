from fastapi import APIRouter, Depends
from app.application.dto.stats import StatsResponse
from app.infrastructure.http.dependencies.container import get_stats_uc

router = APIRouter()

@router.get("/stats", response_model=StatsResponse)
def stats(uc=Depends(get_stats_uc)):
    result = uc.execute()
    return StatsResponse(model_id=result.model_id, metrics=result.metrics, artifacts_path=result.artifacts_path)