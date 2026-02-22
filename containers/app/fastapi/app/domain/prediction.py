from dataclasses import dataclass


LABELS = {
    0: "NonDemented",
    1: "VeryMildDemented",
    2: "MildDemented",
    3: "ModerateDemented",
}


def prediction_to_label(prediction: list[float]) -> str:
    max_index = max(range(len(prediction)), key=prediction.__getitem__)
    return LABELS[max_index]


@dataclass(frozen=True)
class PredictionResult:
    prediction: list[float]
    predicted_label: str

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction": self.prediction,
            "predicted_label": self.predicted_label,
            "labels": LABELS,
        }
