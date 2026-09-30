"""
Standalone inference helper for the wheat disease model.

Can be used directly from the command line:

    python model/predict.py path/to/leaf.jpg

The FastAPI backend (backend/) uses its own lightweight predictor module,
but this script shares the same preprocessing logic and is convenient for
quick local checks.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "wheat_disease_model.keras")
CLASS_INDICES_PATH = os.path.join(ARTIFACTS_DIR, "class_indices.json")
DISEASE_INFO_PATH = os.path.join(os.path.dirname(__file__), "disease_info.json")


def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    disease_info = {}
    if os.path.exists(DISEASE_INFO_PATH):
        with open(DISEASE_INFO_PATH) as f:
            disease_info = json.load(f)
    return model, idx_to_class, disease_info


def preprocess(image_path: str) -> np.ndarray:
    """Load and preprocess an image exactly as during training."""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)  # 0..255; model rescales internally
    return np.expand_dims(arr, axis=0)


def predict(image_path: str) -> dict:
    model, idx_to_class, disease_info = load_artifacts()
    probs = model.predict(preprocess(image_path), verbose=0)[0]
    top_idx = int(np.argmax(probs))
    label = idx_to_class[top_idx]
    return {
        "label": label,
        "confidence": float(probs[top_idx]),
        "all_probabilities": {
            idx_to_class[i]: float(probs[i]) for i in range(len(probs))
        },
        "info": disease_info.get(label, {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict wheat leaf disease.")
    parser.add_argument("image", help="Path to a wheat leaf image.")
    args = parser.parse_args()

    result = predict(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
