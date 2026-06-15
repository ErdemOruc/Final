import numpy as np
from services.base_keras_service import BaseKerasService


class FruitDamageService(BaseKerasService):

    CLASS_NAMES = [
        "Damaged",
        "Healthy",
        "Old",
    ]

    PROBLEM_CLASSES = {"Damaged", "Old"}
    CONFIDENCE_THRESHOLD = 0.50

    def predict(self, crop: np.ndarray) -> dict:
        class_name, confidence = self._run_inference(crop)

        if confidence < self.CONFIDENCE_THRESHOLD:
            return {
                "status":       "Healthy", 
                "damage_level": "Uncertain",
                "confidence":   confidence,
            }

        is_healthy = class_name not in self.PROBLEM_CLASSES

        return {
            "status":       "Healthy" if is_healthy else class_name, 
            "damage_level": "None" if is_healthy else class_name,
            "confidence":   confidence,
        }

    def _demo_predict(self):
        import random
        return random.choice(self.CLASS_NAMES), round(random.uniform(0.72, 0.97), 3)