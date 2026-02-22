import os
from functools import lru_cache

from fastapi import FastAPI

from app.adapters.http.routes import create_router
from app.adapters.mlflow_model import MlflowTensorflowModelAdapter
from app.application.use_cases import PredictConditionUseCase

app = FastAPI(title="BrainSight Public API")


@lru_cache(maxsize=1)
def get_predict_use_case() -> PredictConditionUseCase:
    model_uri = os.getenv("MODEL_URI", "runs:/b347c773a181434fae3e122921c1d937/model")
    model_adapter = MlflowTensorflowModelAdapter(model_uri=model_uri)
    return PredictConditionUseCase(model_port=model_adapter)

app.include_router(create_router(get_predict_use_case))


@app.get("/")
async def read_root() -> dict[str, str]:
    return {"status": "ok", "service": "brainsight-api"}
