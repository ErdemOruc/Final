import numpy as np
from services.base_keras_service import BaseKerasService


class FruitRipenessService(BaseKerasService):

    CLASS_NAMES = [
        "Ripe",
        "Unripe",
    ]
    
    CONFIDENCE_THRESHOLD = 0.50

    def predict(self, crop: np.ndarray) -> dict:
        class_name, confidence = self._run_inference(crop)

        if confidence < self.CONFIDENCE_THRESHOLD:
            return {
                "ripeness":   "Uncertain",         
                "is_ideal":   False,
                "confidence": confidence,
            }

        return {
            "ripeness":   class_name,         
            "is_ideal":   class_name == "Ripe",
            "confidence": confidence,
        }

    def _demo_predict(self):
        import random
        return random.choice(self.CLASS_NAMES), round(random.uniform(0.70, 0.96), 3)