# Real-Time Computer Vision API (YOLOv8 + FastAPI + Docker)

A production-ready REST API microservice for real-time object detection and visual inference powered by Ultralytics YOLOv8 and FastAPI. Designed for low-latency CPU execution and containerized deployment.

## Architecture & Tech Stack

* **API Framework:** FastAPI (Asynchronous REST endpoints + automated OpenAPI/Swagger documentation)
* **Computer Vision Engine:** Ultralytics YOLOv8 Nano, OpenCV Headless, Pillow
* **Data Validation:** Pydantic schemas for strict JSON output typing
* **MLOps & Deployment:** Docker containerization with CPU-optimized PyTorch

## API Endpoints

* **GET /health:** System healthcheck endpoint for cloud load balancers.
* **POST /api/v1/detect:** Returns structured JSON with bounding box coordinates, class labels, confidence scores, object counts, and inference latency.
* **POST /api/v1/annotate:** Streams back the annotated JPEG image with drawn bounding boxes and confidence labels.
