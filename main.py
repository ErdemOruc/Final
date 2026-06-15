from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import numpy as np
# pyrefly: ignore [missing-import]
import cv2
import time
import uuid
import logging
import asyncio
import os
from pathlib import Path

LLM_IS_OPEN = False

from LoadModels import LoadModel

yolo_model, keras_LeafDisease, keras_OldDamaged, keras_RipeUnripe = LoadModel()

from services.yolo_service              import YOLOService
from services.leaf_disease_service      import LeafDiseaseService
from services.fruit_damage_service      import FruitDamageService
from services.fruit_ripeness_service    import FruitRipenessService
from services.llm_service               import LLMService
from services.annotator                 import Annotator
from models.schemas                     import PipelineResponse, DetectionItem

yolo_service          = YOLOService(model=yolo_model)
leaf_disease_service  = LeafDiseaseService(model=keras_LeafDisease)
fruit_damage_service  = FruitDamageService(model=keras_OldDamaged)
fruit_ripeness_service = FruitRipenessService(model=keras_RipeUnripe)
llm_service           = LLMService(ollama_url="http://localhost:11434")
annotator             = Annotator()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Plant Health Pipeline",
    description="YOLO + Keras + Ollama — leaf & fruit health analysis",
    version="1.0.0",
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def process_image(frame: np.ndarray, start: float) -> dict:
    detections = yolo_service.detect(frame)

    if not detections:
        return {
            "overall_status":    "No Detection",
            "summary":           "No leaves or fruits found in the image.",
            "processing_time_ms": int((time.time() - start) * 1000),
        }

    items       = []
    has_problem = False

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        
        h, w = frame.shape[:2]
        pad = 20
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(w, x2 + pad)
        py2 = min(h, y2 + pad)
        
        crop      = frame[py1:py2, px1:px2]
        obj_class = det["class"]         
        item_id   = f"{obj_class}{det['id']}"

        if obj_class == "leaf":
            res        = leaf_disease_service.predict(crop)
            status     = res["status"]
            disease    = res["disease_type"]
            confidence = res["confidence"]

            if status != "Healthy":
                has_problem = True

            items.append(DetectionItem(
                id           = item_id,
                type         = "leaf",
                bbox         = det["bbox"],
                confidence   = confidence,
                status       = status,
                disease_type = disease if status != "Healthy" else None,
            ))

        elif obj_class == "fruit":
            dmg        = fruit_damage_service.predict(crop)
            ripe       = fruit_ripeness_service.predict(crop)

            dmg_status   = dmg["status"]
            dmg_level    = dmg["damage_level"]
            ripeness     = ripe["ripeness"]
            confidence   = (dmg["confidence"] + ripe["confidence"]) / 2

            fruit_problem = dmg_status != "Healthy" or ripeness not in ("Ripe",)

            if fruit_problem:
                has_problem = True

            items.append(DetectionItem(
                id           = item_id,
                type         = "fruit",
                bbox         = det["bbox"],
                confidence   = confidence,
                status       = dmg_status,
                damage_level = dmg_level if dmg_status != "Healthy" else None,
                ripeness     = ripeness,
            ))

    # Her zaman resmi çiz ve kaydet
    annotated_frame = annotator.draw_annotations(frame.copy(), items, detections)
    fname           = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(OUTPUT_DIR / fname), annotated_frame)
    annotated_image_path = f"/results/{fname}"

    leaves = {}
    fruits = {}
    for item in items:
        if item.type == "leaf":
            leaves[item.id] = item.to_api_dict()
        else:
            fruits[item.id] = item.to_api_dict()

    problem_count = sum(
        1 for i in items
        if i.status != "Healthy" or (i.ripeness and i.ripeness not in ("Ripe", None))
    )

    summary_text = f"Analyzed a plant image with {len(items)} detected parts. "
    if not has_problem:
        summary_text += "All parts appear healthy."
    else:
        summary_text += f"{problem_count} part(s) show issues: "
        issues = []
        for i in items:
            if i.type == "leaf" and i.status != "Healthy":
                issues.append(f"Leaf with {i.disease_type}")
            elif i.type == "fruit" and (i.status != "Healthy" or i.ripeness != "Ripe"):
                issues.append(f"Fruit that is {i.status}/{i.ripeness}")
        summary_text += ", ".join(issues) + "."

    overall_llm_advice = None
    if LLM_IS_OPEN:
        overall_llm_advice = llm_service.get_overall_advice(summary_text)

    response = {
        "overall_status": "Problem Detected" if has_problem else "Healthy",
        "summary": summary_text,
        "llm": overall_llm_advice,
        "processing_time_ms": int((time.time() - start) * 1000),
    }

    if leaves:
        response["leaves"] = leaves
    if fruits:
        response["fruits"] = fruits
    if annotated_image_path:
        response["annotated_image"] = annotated_image_path

    return response

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Full pipeline:
      1. YOLO  → detect leaves and fruits, get bounding boxes
      2. Keras → classify each crop
      3. LLM   → generate recommendation
      4. If problem → draw annotated image
      5. Return JSON
    """
    start = time.time()

    raw   = await file.read()
    arr   = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    global latest_simulated_result
    response_dict = process_image(frame, start)
    latest_simulated_result = response_dict
    
    return JSONResponse(response_dict)

latest_simulated_result = {"status": "Waiting for data..."}

@app.get("/latest")
async def get_latest():
    return JSONResponse(latest_simulated_result)

@app.get("/latest/image")
async def get_latest_image():
    if "annotated_image" not in latest_simulated_result:
        raise HTTPException(status_code=404, detail="No image processed yet.")
    
    image_url = latest_simulated_result["annotated_image"]
    filename = image_url.split("/")[-1]
    path = OUTPUT_DIR / filename
    
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk.")
        
    return FileResponse(str(path), media_type="image/jpeg")

@app.get("/results/{filename}")
async def get_result(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(str(path), media_type="image/jpeg")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "yolo":           yolo_service.is_loaded(),
            "leaf_disease":   leaf_disease_service.is_loaded(),
            "fruit_damage":   fruit_damage_service.is_loaded(),
            "fruit_ripeness": fruit_ripeness_service.is_loaded(),
            "llm_ollama":     llm_service.is_available() if LLM_IS_OPEN else False,
        },
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
