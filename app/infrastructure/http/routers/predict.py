from fastapi import APIRouter, Depends
from app.application.dto.predict import PredictRequest, PredictResponse
from app.infrastructure.http.dependencies.container import get_predict_uc
from app.application.dto.predict_request import PredictInlineRequest, PredictInlineResponse

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, uc=Depends(get_predict_uc)):
    result = uc.execute(req.data_path, req.output_path)
    return PredictResponse(model_id=result.model_id, output_path=result.output_path, predictions=result.predictions)


@router.post("/predict/inline", response_model=PredictInlineResponse)
def predict_inline(req: PredictInlineRequest, uc=Depends(get_predict_uc)):
    payload = req.model_dump(exclude_none=True)
    model_key = payload.pop("model_key", None)

    result = uc.execute_inline(payload, model_key=model_key)  

    return PredictInlineResponse(
        model_key=result.get("model_id", model_key or "prod"),
        predictions=result["predictions"],
    )