"""FastAPI service exposing the steel surface defect classifier.

Run locally:
    uvicorn api.main:app --reload

Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_report import InspectionReport, generate_report  # noqa: E402
from model import CLASS_NAMES, build_transform, load_model  # noqa: E402

MODEL_PATH = ROOT / "models" / "defect_classifier.pt"
METRICS_PATH = ROOT / "models" / "metrics.json"

app = FastAPI(
    title="Steel Defect Detection API",
    description=(
        "Classifies steel surface defects from an image and generates an "
        "LLM-backed inspection report (defect type, confidence, likely "
        "cause, recommended action)."
    ),
    version="1.0.0",
)

_model = None
_transform = build_transform()


def get_model():
    """Load the model once and reuse it across requests."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model artifact not found. Run 'python src/train_model.py' first.",
            )
        _model = load_model(MODEL_PATH)
    return _model


class ClassificationResult(BaseModel):
    defect_type: str
    confidence: float
    class_probabilities: dict[str, float]


class PredictResponse(BaseModel):
    classification: ClassificationResult
    inspection_report: InspectionReport


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    trained_at: str | None = None
    test_accuracy: float | None = None


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe that also surfaces which model version is loaded."""
    if not MODEL_PATH.exists():
        return HealthResponse(status="degraded", model_loaded=False)
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    return HealthResponse(
        status="ok",
        model_loaded=True,
        trained_at=metrics.get("trained_at"),
        test_accuracy=metrics.get("accuracy"),
    )


@app.post("/predict", response_model=PredictResponse, tags=["scoring"])
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """Classify a steel surface image and generate its inspection report."""
    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    model = get_model()
    tensor = _transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    class_probabilities = {
        name: round(float(probabilities[i]), 4) for i, name in enumerate(CLASS_NAMES)
    }
    best_idx = int(probabilities.argmax())
    defect_type = CLASS_NAMES[best_idx]
    confidence = round(float(probabilities[best_idx]), 4)

    report = generate_report(defect_type, confidence)

    return PredictResponse(
        classification=ClassificationResult(
            defect_type=defect_type,
            confidence=confidence,
            class_probabilities=class_probabilities,
        ),
        inspection_report=report,
    )


@app.get("/model/metrics", tags=["ops"])
def model_metrics() -> dict:
    """Return the held-out evaluation metrics for the loaded model."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    return json.loads(METRICS_PATH.read_text())
