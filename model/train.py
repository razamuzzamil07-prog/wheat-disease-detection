"""
Wheat Leaf Disease Detection - Model Training Script
====================================================

Trains an image classification model (MobileNetV2 transfer learning) that
classifies a wheat leaf image into one of 5 classes:

    1. healthy          - Healthy leaf
    2. leaf_rust        - Leaf Rust (Brown Rust)
    3. stripe_rust      - Stripe Rust (Yellow Rust)
    4. powdery_mildew   - Powdery Mildew
    5. septoria         - Septoria Leaf Blotch

Expected dataset layout (image-folder format):

    data/raw/
        healthy/          *.jpg / *.png ...
        leaf_rust/        ...
        stripe_rust/      ...
        powdery_mildew/   ...
        septoria/         ...

The script automatically splits each class folder into train / validation /
test sets (70 / 15 / 15) so the final model is evaluated on a separate test
set, as required by the project proposal's non-functional requirements.

Usage:
    python model/train.py --data_dir data/raw --epochs 20 --fine_tune_epochs 10

Outputs (written to --output_dir, default model/artifacts/):
    wheat_disease_model.keras   - trained model file (used by the backend)
    class_indices.json          - class name <-> index mapping
    training_history.png        - accuracy / loss curves
    confusion_matrix.png        - test-set confusion matrix
    classification_report.txt   - per-class precision / recall / F1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLASSES = ["healthy", "leaf_rust", "stripe_rust", "powdery_mildew", "septoria"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test
SEED = 42


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def split_dataset(data_dir: str, work_dir: str) -> str:
    """Split an image-folder dataset into train/val/test subsets.

    Returns the path of the split dataset root containing
    train/, val/ and test/ subdirectories.
    """
    split_root = os.path.join(work_dir, "split")
    random.seed(SEED)

    for cls in CLASSES:
        src = os.path.join(data_dir, cls)
        if not os.path.isdir(src):
            raise FileNotFoundError(
                f"Class folder '{cls}' not found under {data_dir}. "
                f"Expected folders: {', '.join(CLASSES)}"
            )
        files = [
            f for f in os.listdir(src)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        if len(files) < 10:
            raise ValueError(
                f"Class '{cls}' has only {len(files)} images. "
                "Provide at least 10 images per class."
            )
        random.shuffle(files)

        n = len(files)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])
        subsets = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for subset, subset_files in subsets.items():
            dst_dir = os.path.join(split_root, subset, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in subset_files:
                shutil.copy2(os.path.join(src, fname), os.path.join(dst_dir, fname))

        print(f"  {cls:15s} total={n:5d}  "
              f"train={len(subsets['train'])}  "
              f"val={len(subsets['val'])}  "
              f"test={len(subsets['test'])}")

    return split_root


def build_datasets(split_root: str):
    """Create tf.data datasets for train / val / test with augmentation."""
    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(split_root, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(split_root, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        os.path.join(split_root, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"Class order: {class_names}")

    # Data augmentation applied only to the training pipeline.
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.15),
        ],
        name="augmentation",
    )

    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(split_root: str, class_names: list[str]) -> dict:
    """Compute class weights to handle class imbalance."""
    counts = []
    for cls in class_names:
        cls_dir = os.path.join(split_root, "train", cls)
        counts.append(len(os.listdir(cls_dir)))
    total = sum(counts)
    weights = {
        i: total / (len(class_names) * c)
        for i, c in enumerate(counts)
    }
    print(f"Class weights: {weights}")
    return weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> keras.Model:
    """Build a MobileNetV2 transfer-learning classifier."""
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # phase 1: feature extraction

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="wheat_disease_classifier")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, test_ds, class_names, output_dir):
    """Evaluate on the held-out test set and save reports/plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print("\n===== Test-set classification report =====")
    print(report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    test_acc = float(np.mean(y_true == y_pred))
    print(f"Test accuracy: {test_acc:.4f}")
    return test_acc


def plot_history(history_phase1, history_phase2, output_dir):
    """Save accuracy/loss curves for both training phases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = history_phase1.history.get("accuracy", [])
    val_acc = history_phase1.history.get("val_accuracy", [])
    loss = history_phase1.history.get("loss", [])
    val_loss = history_phase1.history.get("val_loss", [])

    if history_phase2 is not None:
        acc += history_phase2.history.get("accuracy", [])
        val_acc += history_phase2.history.get("val_accuracy", [])
        loss += history_phase2.history.get("loss", [])
        val_loss += history_phase2.history.get("val_loss", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(acc, label="train")
    ax1.plot(val_acc, label="val")
    ax1.set_title("Accuracy")
    ax1.legend()
    ax2.plot(loss, label="train")
    ax2.plot(val_loss, label="val")
    ax2.set_title("Loss")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_history.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Step 1/5: Splitting dataset into train/val/test ...")
    with tempfile.TemporaryDirectory() as work_dir:
        split_root = split_dataset(args.data_dir, work_dir)

        print("Step 2/5: Building tf.data pipelines ...")
        train_ds, val_ds, test_ds, class_names = build_datasets(split_root)
        class_weights = compute_class_weights(split_root, class_names)

        print("Step 3/5: Building model (MobileNetV2 transfer learning) ...")
        model = build_model(num_classes=len(class_names))

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(args.output_dir, "wheat_disease_model.keras"),
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ]

        print("Step 4/5: Phase 1 - training classifier head ...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=callbacks,
        )

        history2 = None
        if args.fine_tune_epochs > 0:
            print("Phase 2 - fine-tuning top layers of MobileNetV2 ...")
            base_model = None
            for layer in model.layers:
                if isinstance(layer, keras.Model):
                    base_model = layer
                    break
            if base_model is not None:
                base_model.trainable = True
                # Freeze all layers except the last N.
                for layer in base_model.layers[:-args.fine_tune_layers]:
                    layer.trainable = False

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            history2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.fine_tune_epochs,
                class_weight=class_weights,
                callbacks=callbacks,
            )

        print("Step 5/5: Evaluating on held-out test set ...")
        # Load the best checkpoint before final evaluation.
        model = keras.models.load_model(
            os.path.join(args.output_dir, "wheat_disease_model.keras")
        )
        evaluate_model(model, test_ds, class_names, args.output_dir)
        plot_history(history1, history2, args.output_dir)

    # Persist the class mapping used by the backend.
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    with open(os.path.join(args.output_dir, "class_indices.json"), "w") as f:
        json.dump(class_indices, f, indent=2)

    print(f"\nDone. Artifacts saved to: {os.path.abspath(args.output_dir)}")
    print("  - wheat_disease_model.keras")
    print("  - class_indices.json")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("  - classification_report.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Train the wheat leaf disease classification model."
    )
    parser.add_argument(
        "--data_dir", default="data/raw",
        help="Root folder containing one subfolder per class.",
    )
    parser.add_argument(
        "--output_dir", default="model/artifacts",
        help="Where to save the trained model and reports.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Epochs for phase 1 (classifier head training).",
    )
    parser.add_argument(
        "--fine_tune_epochs", type=int, default=10,
        help="Epochs for phase 2 (fine-tuning). Set 0 to disable.",
    )
    parser.add_argument(
        "--fine_tune_layers", type=int, default=40,
        help="Number of top MobileNetV2 layers to unfreeze in phase 2.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
