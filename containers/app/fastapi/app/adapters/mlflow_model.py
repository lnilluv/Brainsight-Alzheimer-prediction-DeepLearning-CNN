import mlflow
import tensorflow as tf

from app.application.ports import BrainConditionModelPort


class MlflowTensorflowModelAdapter(BrainConditionModelPort):
    def __init__(self, model_uri: str):
        self._model = mlflow.tensorflow.load_model(model_uri, keras_model_kwargs={"compile": False})

    def predict(self, image_bytes: bytes, image_size: tuple[int, int]) -> list[float]:
        image = tf.io.decode_image(image_bytes, channels=3)
        image = tf.image.resize(image, image_size)
        image = image / 255.0
        image = tf.expand_dims(image, 0)
        prediction = self._model.predict(image)[0]
        return prediction.tolist()
