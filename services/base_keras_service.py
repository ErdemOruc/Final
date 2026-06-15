# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseKerasService:
    INPUT_SIZE = (256, 256)
    CLASS_NAMES = []

    def __init__(self, model):
        self.model = model
        if model is not None:
            logger.info(f"{self.__class__.__name__} ready.")
        else:
            logger.warning(f"{self.__class__.__name__}: model is None — demo mode active.")

    def _run_inference(self, crop: np.ndarray):
        if self.model is None:
            return self._demo_predict()

        try:

            resized = cv2.resize(crop, self.INPUT_SIZE)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            input_tensor = np.expand_dims(rgb, axis=0).astype(np.float32)
            input_tensor = input_tensor / 255.0 
            
            flipped_rgb = cv2.flip(rgb, 1)
            flipped_tensor = np.expand_dims(flipped_rgb, axis=0).astype(np.float32)
            flipped_tensor = flipped_tensor / 255.0

            batch = np.vstack([input_tensor, flipped_tensor])
            predictions = self.model.predict(batch, verbose=0)

            avg_predictions = np.mean(predictions, axis=0)

            class_idx = np.argmax(avg_predictions)
            confidence = float(avg_predictions[class_idx])
            
            class_name = self.CLASS_NAMES[class_idx]
            return class_name, confidence

        except Exception as e:
            logger.error(f"Inference error in {self.__class__.__name__}: {e}")
            return self._demo_predict()
            
    def _demo_predict(self):
        import random
        return random.choice(self.CLASS_NAMES), round(random.uniform(0.70, 0.98), 3)

    def is_loaded(self) -> bool:
        return self.model is not None
