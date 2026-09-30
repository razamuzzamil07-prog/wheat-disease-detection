"""
Predictor service: loads the trained model once and exposes a predict()
function used by the FastAPI application.
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

# Artifacts are resolved relative to the repository root so the backend
# works both locally and inside Docker.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.getenv(
    "MODEL_ARTIFACTS_DIR", os.path.join(REPO_ROOT, "model", "artifacts")
)
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "wheat_disease_model.keras")
CLASS_INDICES_PATH = os.path.join(ARTIFACTS_DIR, "class_indices.json")
DISEASE_INFO_PATH = os.path.join(REPO_ROOT, "model", "disease_info.json")


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but no model file is present."""


class Predictor:
    """Thread-safe singleton wrapper around the Keras model."""

    def __init__(self) -> None:
        self._model = None
        self._idx_to_class: dict[int, str] = {}
        self._disease_info: dict = {}
        self._load_disease_info()

    # -- loading ----------------------------------------------------------

    def _load_disease_info(self) -> None:
        if os.path.exists(DISEASE_INFO_PATH):
            with open(DISEASE_INFO_PATH, encoding="utf-8") as f:
                self._disease_info = json.load(f)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def classes(self) -> list[str]:
        return [self._idx_to_class[i] for i in sorted(self._idx_to_class)]

    def load(self) -> None:
        """Load the model and class mapping from disk (idempotent)."""
        if self._model is not None:
            return
        if not os.path.exists(MODEL_PATH):
            raise ModelNotLoadedError(
                f"Model file not found at {MODEL_PATH}. "
                "Train the model first: python model/train.py --data_dir data/raw"
            )
        # Imported lazily so the API can still boot (and report a clear
        # 503 error) even if TensorFlow/model artifacts are missing.
        import tensorflow as tf

        self._model = tf.keras.models.load_model(MODEL_PATH)

        if os.path.exists(CLASS_INDICES_PATH):
            with open(CLASS_INDICES_PATH, encoding="utf-8") as f:
                class_indices = json.load(f)
            self._idx_to_class = {v: k for k, v in class_indices.items()}
        else:
            # Fall back to the canonical class order used during training.
            from model.train import CLASSES  # noqa: WPS433 (lazy import)

            self._idx_to_class = {i: c for i, c in enumerate(CLASSES)}

    # -- inference --------------------------------------------------------

    @staticmethod
    def preprocess(image_bytes: bytes) -> np.ndarray:
        """Decode, normalize and batch a raw image upload."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32)
        return np.expand_dims(arr, axis=0)

    def predict(self, image_bytes: bytes) -> dict:
        """Run inference and return a serializable prediction payload."""
        if not self.is_loaded:
            self.load()

        batch = self.preprocess(image_bytes)
        probs = self._model.predict(batch, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        label = self._idx_to_class[top_idx]
        info = self._disease_info.get(label, {})

        return {
            "label": label,
            "display_name": info.get("display_name", label.replace("_", " ").title()),
            "confidence": round(float(probs[top_idx]), 4),
            "severity": info.get("severity"),
            "description": info.get("description"),
            "symptoms": info.get("symptoms"),
            "treatment": info.get("treatment"),
            "all_probabilities": {
                self._idx_to_class[i]: round(float(probs[i]), 4)
                for i in range(len(probs))
            },
        }


# Module-level singleton shared across requests.
predictor = Predictor()
