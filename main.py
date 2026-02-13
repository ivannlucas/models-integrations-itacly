from fastapi import FastAPI
from app.infrastructure.http.routers.health import router as health_router
from app.infrastructure.http.routers.train import router as train_router
from app.infrastructure.http.routers.predict import router as predict_router
from app.infrastructure.http.routers.stats import router as stats_router

app = FastAPI(title="Vitivinicola price fluctuation model API")

app.include_router(health_router)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(stats_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )