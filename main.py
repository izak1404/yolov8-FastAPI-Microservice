import io
import time
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI(
    title="Real-Time Computer Vision API (YOLOv8)",
    description="Microsserviço de detecção de objetos e inferência visual com FastAPI e Ultralytics YOLOv8.",
    version="1.0.0"
)

# Carrega o modelo Nano (6MB) otimizado para CPU. 
# Na primeira execução, o Ultralytics fará o download automático do arquivo yolov8n.pt.
model = YOLO("yolov8n.pt")

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

class InferenceResponse(BaseModel):
    filename: str
    inference_time_ms: float
    total_objects: int
    summary_counts: Dict[str, int]
    detections: List[DetectionItem]

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "model": "yolov8n", "device": "cpu"}

@app.post("/api/v1/detect", response_model=InferenceResponse, tags=["Inference"])
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, ge=0.05, le=1.0, description="Grau mínimo de confiança (0.05 a 1.0)")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado deve ser uma imagem válida.")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    start_time = time.perf_counter()
    results = model.predict(source=image, conf=conf_threshold, verbose=False)
    inference_time = round((time.perf_counter() - start_time) * 1000, 2)

    result = results[0]
    detections = []
    summary_counts: Dict[str, int] = {}

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = result.names[cls_id]
        conf = round(float(box.conf[0].item()), 4)
        x1, y1, x2, y2 = [round(coord, 2) for coord in box.xyxy[0].tolist()]

        summary_counts[cls_name] = summary_counts.get(cls_name, 0) + 1

        detections.append(
            DetectionItem(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            )
        )

    return InferenceResponse(
        filename=file.filename,
        inference_time_ms=inference_time,
        total_objects=len(detections),
        summary_counts=summary_counts,
        detections=detections
    )

@app.post("/api/v1/annotate", tags=["Inference"])
async def annotate_image(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, ge=0.05, le=1.0)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado deve ser uma imagem válida.")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(source=image, conf=conf_threshold, verbose=False)
    annotated_bgr = results[0].plot()

    success, encoded_img = cv2.imencode(".jpg", annotated_bgr)
    if not success:
        raise HTTPException(status_code=500, detail="Erro ao processar a imagem anotada.")

    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")