"""Tests for the steel defect detection API.

Exercises the real API with real sample images (tests/fixtures/), not
mocked calls -- including one corrupted-upload case, matching the pattern
from the Olist API tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import app  # noqa: E402

client = TestClient(app)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def test_health_reports_loaded_model():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert 0.0 < body["test_accuracy"] <= 1.0


def test_metrics_endpoint_exposes_accuracy():
    response = client.get("/model/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "accuracy" in body
    assert body["class_names"] == CLASS_NAMES


def test_predict_classifies_real_sample_correctly():
    """The classifier should get its own training-domain samples right --
    a weak but real correctness check, not just a shape check."""
    for class_name in CLASS_NAMES:
        image_path = FIXTURES_DIR / f"{class_name}_sample.jpg"
        with open(image_path, "rb") as f:
            response = client.post(
                "/predict", files={"file": (image_path.name, f, "image/jpeg")}
            )
        assert response.status_code == 200, response.text
        body = response.json()

        classification = body["classification"]
        assert classification["defect_type"] == class_name
        assert 0.0 <= classification["confidence"] <= 1.0
        assert set(classification["class_probabilities"]) == set(CLASS_NAMES)

        report = body["inspection_report"]
        assert report["defect_type"] == class_name
        assert report["likely_cause"]
        assert report["recommended_action"]
        assert report["report_source"] in {"llm", "template_fallback"}


def test_predict_rejects_corrupted_image():
    """An unusual or corrupted upload must return a clean 400, not crash
    the endpoint -- the CV equivalent of Olist's unseen-category test."""
    corrupted = b"this is not a real image file, just garbage bytes"
    response = client.post(
        "/predict", files={"file": ("broken.jpg", corrupted, "image/jpeg")}
    )
    assert response.status_code == 400


def test_predict_rejects_missing_file():
    response = client.post("/predict")
    assert response.status_code == 422
