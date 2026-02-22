from app.application.ports import BrainConditionModelPort
from app.domain.prediction import PredictionResult, prediction_to_label


class PredictConditionUseCase:
    def __init__(self, model_port: BrainConditionModelPort, image_size: tuple[int, int] = (176, 208)):
        self._model_port = model_port
        self._image_size = image_size

    def execute(self, image_bytes: bytes) -> PredictionResult:
        prediction = self._model_port.predict(image_bytes=image_bytes, image_size=self._image_size)
        return PredictionResult(prediction=prediction, predicted_label=prediction_to_label(prediction))
