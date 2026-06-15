import numpy as np
from services.base_keras_service import BaseKerasService


class LeafDiseaseService(BaseKerasService):

    CLASS_NAMES = [
        "Tomato__Bacterial_spot",               
        "Tomato__Early_blight",                 
        "Tomato__Late_blight",                 
        "Tomato__Leaf_Mold",                    
        "Tomato__Septoria_leaf_spot",           
        "Tomato___Spider_mites Two-spotted_spider_mite",  
        "Tomato__Target_Spot",                 
        "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato__Tomato_mosaic_virus",         
        "Tomato__healthy",                      
    ]

    HEALTHY_CLASSES = {"Tomato__healthy"}
    CONFIDENCE_THRESHOLD = 0.50

    def predict(self, crop: np.ndarray) -> dict:
        class_name, confidence = self._run_inference(crop)
        
        if confidence < self.CONFIDENCE_THRESHOLD:
            return {
                "status":       "Healthy",
                "disease_type": "Uncertain",
                "confidence":   confidence,
            }

        is_healthy = class_name in self.HEALTHY_CLASSES

        return {
            "status":       "Healthy" if is_healthy else "Diseased",
            "disease_type": class_name,
            "confidence":   confidence,
        }

    def _demo_predict(self):
        import random
        return random.choice(self.CLASS_NAMES), round(random.uniform(0.75, 0.98), 3)