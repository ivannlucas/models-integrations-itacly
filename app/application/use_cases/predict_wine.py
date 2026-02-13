# app/application/use_cases/predict_wine.py

from app.domain.services.wine_runtime_service import WineRuntimeService

service = WineRuntimeService(
    model_path="models/prod/ml_model.pkl"
)

def predict_wine(data: dict):
    return service.predict(data)