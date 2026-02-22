from typing import Protocol


class BrainConditionModelPort(Protocol):
    def predict(self, image_bytes: bytes, image_size: tuple[int, int]) -> list[float]:
        ...
