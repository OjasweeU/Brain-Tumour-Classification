from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.brain_tumor_ui.inference import (
    format_probabilities,
    load_trained_model,
    predict_image,
    read_image,
    resolve_model_path,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FRONTEND_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app = FastAPI(
    title="Brain Tumor MRI Classifier API",
    version="1.0.0",
    description="Inference API for EfficientNet-based brain tumor classification with Grad-CAM.",
)
app.state.model = None
app.state.model_path = ""
app.state.model_error = "Model has not been loaded yet."


def _allowed_origins() -> list[str]:
    raw_value = os.getenv("FRONTEND_ORIGINS", ",".join(DEFAULT_FRONTEND_ORIGINS))
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _image_to_data_url(image: np.ndarray) -> str:
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.on_event("startup")
def load_model_on_startup() -> None:
    model_path = resolve_model_path(os.getenv("BRAIN_TUMOR_MODEL_PATH"))
    try:
        app.state.model = load_trained_model(ROOT_DIR / model_path)
        app.state.model_path = str(ROOT_DIR / model_path)
    except FileNotFoundError as exc:
        app.state.model = None
        app.state.model_path = str(ROOT_DIR / model_path)
        app.state.model_error = str(exc)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    if app.state.model is None:
        raise HTTPException(status_code=503, detail=getattr(app.state, "model_error", "Model is unavailable."))
    return {"status": "ok", "model_path": app.state.model_path}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if app.state.model is None:
        raise HTTPException(status_code=503, detail=getattr(app.state, "model_error", "Model is unavailable."))

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = read_image(image_bytes)
        result = predict_image(app.state.model, image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "predictedLabel": result.predicted_label,
        "confidence": result.confidence,
        "probabilities": format_probabilities(result.probabilities),
        "images": {
            "original": _image_to_data_url(result.original_image),
            "cropped": _image_to_data_url(result.cropped_image),
            "gradcam": _image_to_data_url(result.overlay_image),
        },
    }
