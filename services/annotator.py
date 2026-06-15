# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from typing import List
from models.schemas import DetectionItem


class Annotator:

    COLORS = {
        "healthy":  (0, 200, 0),
        "warning":  (0, 200, 255),
        "problem":  (0, 0, 220),
        "default":  (200, 200, 200),
    }

    BOX_THICKNESS  = 2
    FONT           = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE     = 0.6
    FONT_THICKNESS = 2
    PAD            = 5

    def draw_annotations(
        self,
        frame: np.ndarray,
        items: List[DetectionItem],
        detections: list,
    ) -> np.ndarray:
        item_map = {item.id: item for item in items}

        for det in detections:
            item_id = f"{det['class']}{det['id']}"
            item    = item_map.get(item_id)
            if item is None:
                continue

            x1, y1, x2, y2 = det["bbox"]
            color  = self._color(item)
            label  = self._label(item)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.BOX_THICKNESS)

            (tw, th), _ = cv2.getTextSize(label, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
            ly = max(y1 - self.PAD - th, 0)
            cv2.rectangle(
                frame,
                (x1, ly - self.PAD),
                (x1 + tw + self.PAD * 2, y1),
                color, -1,
            )
            cv2.putText(
                frame, label,
                (x1 + self.PAD, y1 - self.PAD),
                self.FONT, self.FONT_SCALE,
                (255, 255, 255), self.FONT_THICKNESS,
            )
            cv2.putText(
                frame, f"{item.confidence:.0%}",
                (x2 - 48, y2 - 8),
                self.FONT, 0.45, color, 1,
            )

        return frame

    def _color(self, item: DetectionItem) -> tuple:
        if item.status == "Healthy":
            if item.ripeness and item.ripeness != "Ripe":
                return self.COLORS["warning"]   
            return self.COLORS["healthy"]       
        return self.COLORS["problem"]           

    def _label(self, item: DetectionItem) -> str:
        if item.type == "leaf":
            if item.disease_type and item.disease_type not in ("Healthy", "healthy"):
                return item.disease_type.split("__")[-1]
            return "Healthy_Leaf"

        parts = []
        if item.status in ("Damaged", "Old") and item.damage_level:
            parts.append(item.damage_level)   
        if item.ripeness and item.ripeness != "Ripe":
            parts.append(item.ripeness)        
        return "_".join(parts) if parts else "Healthy_Fruit"

