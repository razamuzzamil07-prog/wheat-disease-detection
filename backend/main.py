"""
FastAPI backend for the Wheat Leaf Disease Detection system.

Endpoints:
    GET  /health     - service and model status
    GET  /classes    - list of supported disease classes
    POST /predict    - upload a leaf image, receive disease prediction

Run locally:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .predictor import ModelNotLoadedError, predictor

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Wheat Leaf Disease Detection API",
    description=(
        "AI-assisted wheat leaf disease classification. "
        "Predictions are dataset-dependent and are NOT a substitute for "
        "professional agricultural diagnosis."
    ),
    version="1.0.0",
)

# Allow the static frontend (and local dev servers) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Input validation constants
# ---------------------------------------------------------------------------

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

DISCLAIMER = (
    "This prediction is AI-assisted and dataset-dependent. "
    "It is not a replacement for professional agricultural diagnosis."
)


async def _validate_image(file: UploadFile) -> bytes:
    """Validate the uploaded file and return its raw bytes.

    Rejects unsupported content types/extensions, oversized files and
    content that cannot be decoded as an image.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type '{file.content_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 10 MB.",
        )

    # Verify the bytes really are a decodable image.
    try:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        )

    return data


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Service and model status."""
    return {
        "status": "ok",
        "model_loaded": predictor.is_loaded,
        "supported_classes": predictor.classes if predictor.is_loaded else None,
    }


@app.get("/classes")
def classes():
    """Return the disease classes supported by the model."""
    try:
        if not predictor.is_loaded:
            predictor.load()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {
        "classes": predictor.classes,
        "disease_info": predictor._disease_info,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify an uploaded wheat leaf image."""
    data = await _validate_image(file)

    try:
        result = predictor.predict(data)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(
            status_code=422, detail=f"Could not process image: {exc}"
        )

    result["disclaimer"] = DISCLAIMER
    return result


# ---------------------------------------------------------------------------
# Static frontend (optional: serves ../frontend at the API root)
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)
if os.path.isdir(FRONTEND_DIR):
    app.mount(
        "/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend"
    )

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
