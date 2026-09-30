# Wheat Leaf Disease Detection (AI)

An AI-assisted web system that classifies a **wheat leaf image** into one of **5 categories** using a deep learning model (MobileNetV2 transfer learning, TensorFlow/Keras):

| Class | Label |
|---|---|
| Healthy | `healthy` |
| Leaf Rust (Brown Rust) | `leaf_rust` |
| Stripe Rust (Yellow Rust) | `stripe_rust` |
| Powdery Mildew | `powdery_mildew` |
| Septoria Leaf Blotch | `septoria` |

The project implements the functional and non-functional requirements from the course project proposal: image upload, validation & preprocessing, deep-learning classification, prediction + confidence display, invalid-input handling, test-set evaluation, and a clear AI-assistance disclaimer.

> **Disclaimer:** This is an educational AI prototype. Predictions are AI-assisted and dataset-dependent — not a replacement for professional agricultural diagnosis.

---

## Project structure

```
wheat-disease-detection/
├── backend/                  # FastAPI backend
│   ├── main.py               # API endpoints (upload validation, prediction)
│   └── predictor.py          # Model loading + preprocessing + inference
├── frontend/                 # Static web frontend (HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── model/
│   ├── train.py              # Training pipeline (split, augment, train, evaluate)
│   ├── predict.py            # Standalone CLI inference
│   ├── disease_info.json     # Per-disease descriptions, symptoms, treatments
│   └── artifacts/            # (generated) trained model + evaluation reports
├── notebooks/
│   └── wheat_disease_training.ipynb   # Google Colab training notebook
├── tests/
│   └── test_api.py           # API validation tests (no model required)
├── requirements.txt
├── Dockerfile
└── README.md
```

## 1. Dataset

Use any public wheat leaf disease image dataset (e.g. search Kaggle for *"wheat leaf disease dataset"*, or the CGIAR / public wheat rust image collections) and organize it in image-folder format:

```
data/raw/
├── healthy/          # healthy wheat leaf images
├── leaf_rust/        # leaf rust images
├── stripe_rust/      # stripe/yellow rust images
├── powdery_mildew/   # powdery mildew images
└── septoria/         # septoria leaf blotch images
```

The training script automatically splits each class into **train / validation / test (70/15/15)**, so evaluation is always performed on a separate test set.

## 2. Train the model

```bash
pip install -r requirements.txt
python model/train.py --data_dir data/raw --epochs 20 --fine_tune_epochs 10
```

Artifacts are written to `model/artifacts/`:

- `wheat_disease_model.keras` — trained model (loaded by the backend)
- `class_indices.json` — class mapping
- `training_history.png`, `confusion_matrix.png`, `classification_report.txt` — evaluation outputs

Prefer GPU? Open **`notebooks/wheat_disease_training.ipynb`** in Google Colab — it runs the same pipeline and downloads the artifacts at the end.

The model uses **two-phase transfer learning** on MobileNetV2 (ImageNet weights): (1) train a new classification head, (2) fine-tune the top convolutional layers, with data augmentation and class-weight balancing.

## 3. Run the backend + frontend

```bash
uvicorn backend.main:app --reload --port 8000
```

Then open:

- **Web app:** <http://localhost:8000> (the backend serves `frontend/`, also at `/app`)
- **API docs (Swagger):** <http://localhost:8000/docs>

### API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service + model status |
| GET | `/classes` | Supported classes and disease info |
| POST | `/predict` | Upload an image (`multipart/form-data`, field `file`) → disease prediction |

Example `POST /predict` response:

```json
{
  "label": "leaf_rust",
  "display_name": "Leaf Rust (Brown Rust)",
  "confidence": 0.9642,
  "severity": "high",
  "description": "Fungal disease caused by Puccinia triticina...",
  "symptoms": "Small, round, orange-brown pustules...",
  "treatment": "Apply triazole or strobilurin fungicides...",
  "all_probabilities": {"leaf_rust": 0.9642, "...": 0.01},
  "disclaimer": "This prediction is AI-assisted and dataset-dependent..."
}
```

If the frontend is hosted separately from the API, point it at the backend by adding before `app.js`:

```html
<script>window.API_BASE = "http://localhost:8000";</script>
```

## 4. Run with Docker

```bash
docker build -t wheat-disease-app .
docker run -p 8000:8000 -v $(pwd)/model/artifacts:/app/model/artifacts wheat-disease-app
```

## 5. Tests

```bash
pytest tests/
```

The tests verify upload validation (unsupported types, invalid images, empty files) and the 503 behavior when no trained model is present — they run without any model artifact, so they work in CI.

## 6. Extend with new classes

1. Add a new folder under `data/raw/` and add its name to `CLASSES` in `model/train.py`.
2. Add an entry in `model/disease_info.json`.
3. Retrain — the backend picks up the new `class_indices.json` automatically.

## Tech stack

Python · TensorFlow/Keras (MobileNetV2) · FastAPI · Uvicorn · HTML/CSS/JavaScript · Google Colab · Docker · GitHub

## Team

- Matee Ur Rehman (Team Lead)
- Abdul Qadir
- Abdul Ahad
- Muzammil Raza
