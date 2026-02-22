from collections.abc import Callable

from fastapi import APIRouter, File, UploadFile

from app.application.use_cases import PredictConditionUseCase


def create_router(get_use_case: Callable[[], PredictConditionUseCase]) -> APIRouter:
    router = APIRouter()

    @router.post("/predictionalz", tags=["Deep Learning classification: Alzheimer"])
    async def predict_alzheimer(file: UploadFile = File(...)) -> dict[str, object]:
        image_bytes = await file.read()
        result = get_use_case().execute(image_bytes=image_bytes)
        return result.to_dict()

    @router.post("/predictionbt", tags=["Deep Learning classification: Brain Tumors"])
    async def predict_brain_tumor(file: UploadFile = File(...)) -> dict[str, object]:
        image_bytes = await file.read()
        result = get_use_case().execute(image_bytes=image_bytes)
        return result.to_dict()

    return router
