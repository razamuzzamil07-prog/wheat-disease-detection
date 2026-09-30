"""
API tests for the wheat disease detection backend.

These tests exercise validation and error handling without requiring a
trained model artifact, so they run anywhere (CI included).
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app

client = TestClient(app)


def _make_image_bytes(fmt: str = "PNG", size=(64, 64)) -> bytes:
    img = Image.new("RGB", size, color=(60, 140, 60))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_rejects_unsupported_extension():
    resp = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 415


def test_predict_rejects_non_image_content():
    resp = client.post(
        "/predict",
        files={"file": ("fake.png", b"not an image at all", "image/png")},
    )
    assert resp.status_code == 400


def test_predict_rejects_empty_file():
    resp = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert resp.status_code in (400, 415)


def test_predict_without_trained_model_returns_503():
    """With no model artifact present, a valid image yields a clear 503."""
    if client.get("/health").json()["model_loaded"]:
        pytest.skip("A trained model is present; 503 path not applicable.")
    resp = client.post(
        "/predict",
        files={"file": ("leaf.png", _make_image_bytes(), "image/png")},
    )
    assert resp.status_code == 503
    assert "Train the model first" in resp.json()["detail"]
