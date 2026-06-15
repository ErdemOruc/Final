import numpy as np
import logging

logger = logging.getLogger(__name__)


class YOLOService:

    CLASS_MAP = {
        0: "leaf",
        1: "fruit",
    }

    CONFIDENCE_THRESHOLD = 0.40
    IOU_THRESHOLD        = 0.45

    def __init__(self, model):

        self.model = model
        if model is not None:
            logger.info("YOLOService ready.")
        else:
            logger.warning("YOLOService: model is None — demo mode active.")

    def detect(self, frame: np.ndarray) -> list:

        if self.model is None:
            return self._demo_detect(frame)

        results = self.model(
            frame,
            conf=self.CONFIDENCE_THRESHOLD,
            iou=self.IOU_THRESHOLD,
            augment=True,
            verbose=False
        )

        detections = []
        leaf_count  = 0
        fruit_count = 0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf   = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                obj_class = self.CLASS_MAP.get(cls_id)
                if obj_class is None or x2 <= x1 or y2 <= y1:
                    continue

                if obj_class == "leaf":
                    leaf_count += 1
                    det_id = leaf_count
                else:
                    fruit_count += 1
                    det_id = fruit_count

                detections.append({
                    "id":    det_id,
                    "class": obj_class,
                    "bbox":  [x1, y1, x2, y2],
                    "conf":  conf,
                })

        logger.info(f"Detected: {leaf_count} leaf(ves), {fruit_count} fruit(s)")
        return detections

    def _demo_detect(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        return [
            {"id": 1, "class": "leaf",  "bbox": [10,     10, w//2-10, h-10], "conf": 0.91},
            {"id": 1, "class": "fruit", "bbox": [w//2+10, 10, w-10,   h-10], "conf": 0.85},
        ]

    def is_loaded(self) -> bool:
        return self.model is not None
