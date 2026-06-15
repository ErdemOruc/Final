from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class ItemType(str, Enum):
    LEAF  = "leaf"
    FRUIT = "fruit"


class DetectionItem(BaseModel):
    id:           str
    type:         ItemType
    bbox:         List[int]     
    confidence:   float

    status:       str           

    disease_type: Optional[str] = None  

    damage_level: Optional[str] = None   
    ripeness:     Optional[str] = None   

    llm_advice:   Optional[str] = None

    class Config:
        use_enum_values = True

    def to_api_dict(self) -> dict:
        base = {
            "type":       self.type,
            "status":     self.status,
            "confidence": f"{self.confidence:.1%}",
        }

        if self.type == "leaf":
            base["disease_type"] = self.disease_type or "None"
        else:
            base["damage_level"] = self.damage_level or "None"
            base["ripeness"]     = self.ripeness     or "Unknown"

        if self.llm_advice:
            base["llm_recommendation"] = self.llm_advice

        return base


class PipelineResponse(BaseModel):
    overall_status:      str
    message:             str
    items:               List[DetectionItem]
    annotated_image_url: Optional[str] = None
    processing_time_ms:  int
