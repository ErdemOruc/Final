import numpy as np
import logging

logger = logging.getLogger(__name__)


class YOLOService:

    CLASS_MAP = {
        0: "leaf",
        1: "fruit",
    }

    CONFIDENCE_THRESHOLD = 0.01
    IOU_THRESHOLD        = 0.15

    def __init__(self, model):

        self.model = model
        if model is not None:
            logger.info("YOLOService ready.")
        else:
            logger.warning("YOLOService: model is None — demo mode active.")

    def detect(self, frame: np.ndarray) -> list:

        if self.model is None:
            return self._demo_detect(frame)

        results = self.model.predict(
            source=frame,
            conf=self.CONFIDENCE_THRESHOLD,
            iou=self.IOU_THRESHOLD,
            augment=False,
            agnostic_nms=True,
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

                detections.append({
                    "class": obj_class,
                    "bbox":  [x1, y1, x2, y2],
                    "conf":  conf,
                })

        detections.sort(key=lambda x: x["conf"], reverse=True)
        
        final_detections = []
        for det in detections:
            keep = True
            box1 = det["bbox"]
            for f_det in final_detections:
                box2 = f_det["bbox"]
                
                xi1 = max(box1[0], box2[0])
                yi1 = max(box1[1], box2[1])
                xi2 = min(box1[2], box2[2])
                yi2 = min(box1[3], box2[3])
                
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                if inter_area > 0:
                    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    union_area = box1_area + box2_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > self.IOU_THRESHOLD:
                        keep = False
                        break
            
            if keep:
                if det["class"] == "leaf":
                    leaf_count += 1
                    det["id"] = leaf_count
                else:
                    fruit_count += 1
                    det["id"] = fruit_count
                final_detections.append(det)

        logger.info(f"Detected: {leaf_count} leaf(ves), {fruit_count} fruit(s)")
        return final_detections

    def _demo_detect(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        return [
            {"id": 1, "class": "leaf",  "bbox": [10,     10, w//2-10, h-10], "conf": 0.91},
            {"id": 1, "class": "fruit", "bbox": [w//2+10, 10, w-10,   h-10], "conf": 0.85},
        ]

    def is_loaded(self) -> bool:
        return self.model is not None
