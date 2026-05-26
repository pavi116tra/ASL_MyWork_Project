"""
train_advanced.py
=================
Full advanced training pipeline for ASL fusion model.

Steps:
  1. Load images + pre-extracted feature .npy files from data/
  2. Data augmentation (15x per sample)
  3. Train dual-input fusion model (landmark + image CNN)
  4. Hard negative mining on A/E/M/N/S/T confused classes
  5. Fine-tune with class weights on hard classes
  6. Save best model to model/asl_fusion_model.h5
  7. Plot accuracy/loss curves

Usage:
  python train_advanced.py [--data data] [--epochs 100] [--batch 32]
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from fusion_model  import build_fusion_model, build_landmark_only_model, get_callbacks
from augmentation  import augment_dataset

IMG_SIZE   = 128
HARD_CLASS = ["A", "E", "M", "N", "S", "T"]     # hard negative mining targets


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset(data_root: str, img_size: int = IMG_SIZE):
    """
    Load images and feature vectors from:
      data_root/images/{LETTER}/*.jpg
      data_root/features/{LETTER}/*.npy

    Returns (images, features, labels_str) all as lists.
    """
    img_root  = Path(data_root) / "images"
    feat_root = Path(data_root) / "features"

    images, features, labels = [], [], []

    if not img_root.exists() or not feat_root.exists():
        print(f"[ERROR] data/images/ or data/features/ not found at {data_root}.")
        print("  Run collect_data_advanced.py first to gather training data.")
        sys.exit(1)

    letter_dirs = sorted(d.name for d in img_root.iterdir() if d.is_dir())
    print(f"\nFound {len(letter_dirs)} classes: {letter_dirs}")

    import cv2
    for letter in letter_dirs:
        img_dir  = img_root  / letter
        feat_dir = feat_root / letter

        img_files  = sorted(img_dir.glob("*.jpg"))
        feat_files = sorted(feat_dir.glob("*.npy"))

        # Match by stem (timestamp)
        feat_map = {f.stem.replace("feat_", ""): f for f in feat_files}

        loaded = 0
        for img_path in img_files:
            stem = img_path.stem.replace("img_", "")
            feat_path = feat_map.get(stem)
            if feat_path is None:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))

            feat = np.load(str(feat_path)).astype(np.float32)
            if feat.shape[0] != 41:
                continue

            images.append(img)
            features.append(feat)
            labels.append(letter)
            loaded += 1

        print(f"  {letter:8s}: {loaded} samples")

    print(f"\nTotal loaded: {len(labels)} samples across {len(letter_dirs)} classes")
    return images, features, labels


# ══════════════════════════════════════════════════════════════════════════════
# HARD NEGATIVE MINING
# ══════════════════════════════════════════════════════════════════════════════
def compute_class_weights(labels_enc: np.ndarray,
                          le: LabelEncoder,
                          hard_classes: list,
                          hard_weight: float = 2.0) -> dict:
    """
    Build class_weight dict with higher weight for hard-to-distinguish classes.
    """
    weights = {i: 1.0 for i in range(len(le.classes_))}
    for cls in hard_classes:
        if cls in le.classes_:
            idx = int(le.transform([cls])[0])
            weights[idx] = hard_weight
    return weights

def mine_hard_negatives(model, X_feat, X_img, y_true, le,
                        confused_classes=HARD_CLASS, factor=3):
    """
    Find misclassified samples → oversample them factor× for retraining.
    Returns (extra_feat, extra_img, extra_labels).
    """
    print("\n[Hard Negative Mining] Running predictions on training set...")
    preds      = model.predict([X_feat, X_img], batch_size=64, verbose=0)
    pred_class = np.argmax(preds, axis=1)
    true_class = np.argmax(y_true, axis=1)
    wrong      = pred_class != true_class

    # Also include samples from hard classes even if correct
    hard_idx_set = set()
    for cls in confused_classes:
        if cls in le.classes_:
            hard_idx_set.add(int(le.transform([cls])[0]))

    mask = wrong.copy()
    for i, tc in enumerate(true_class):
        if tc in hard_idx_set:
            mask[i] = True

    hard_feat = X_feat[mask]
    hard_img  = X_img[mask]
    hard_lbl  = y_true[mask]

    extra_feat = np.tile(hard_feat, (factor, 1))
    extra_img  = np.tile(hard_img,  (factor, 1, 1, 1))
    extra_lbl  = np.tile(hard_lbl,  (factor, 1))

    print(f"  Hard samples: {mask.sum()} × {factor}× = {len(extra_feat)} extra")
    return extra_feat, extra_img, extra_lbl


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
def plot_history(history, save_path="model/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train acc", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val acc",   linewidth=2)
    axes[0].set_title("Accuracy", fontsize=14)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val loss",   linewidth=2)
    axes[1].set_title("Loss", fontsize=14)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Training curves → {save_path}")

def plot_confusion_matrix(y_true, y_pred, classes, save_path="model/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Validation Confusion Matrix", fontsize=14)
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def train(data_root="data", epochs=100, batch_size=32, augment_factor=8,
          model_out="model/asl_fusion_model.h5"):

    Path("model").mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("  ASL Advanced Training Pipeline")
    print("="*60)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n[Step 1] Loading dataset...")
    images, features, labels_str = load_dataset(data_root)

    le       = LabelEncoder()
    labels   = le.fit_transform(labels_str)
    n_classes = len(le.classes_)
    print(f"  Classes ({n_classes}): {list(le.classes_)}")

    np.save("model/label_classes_advanced.npy", le.classes_)

    # ── Step 2: Augmentation ──────────────────────────────────────────────────
    print(f"\n[Step 2] Augmenting data ({augment_factor}×)...")
    aug_images, aug_features, aug_labels = augment_dataset(
        images, features, list(labels), n_augments=augment_factor
    )
    print(f"  After augmentation: {len(aug_labels)} samples")

    # ── Step 3: Prepare arrays ────────────────────────────────────────────────
    print("\n[Step 3] Preparing tensors...")
    X_img  = aug_images.astype(np.float32) / 255.0
    X_feat = aug_features.astype(np.float32)
    y_cat  = tf.keras.utils.to_categorical(aug_labels, n_classes)

    X_img_tr, X_img_val, X_feat_tr, X_feat_val, y_tr, y_val = train_test_split(
        X_img, X_feat, y_cat,
        test_size=0.2, random_state=42,
        stratify=aug_labels
    )
    print(f"  Train: {len(y_tr)}   Val: {len(y_val)}")

    # ── Step 4: Build & train model ───────────────────────────────────────────
    print("\n[Step 4] Building fusion model...")
    model = build_fusion_model(n_classes=n_classes, img_size=IMG_SIZE)
    model.summary(line_length=90)

    class_wts = compute_class_weights(aug_labels, le, HARD_CLASS, hard_weight=2.0)

    print("\n[Step 5] Training phase 1...")
    callbacks = get_callbacks(checkpoint_path=model_out)

    history1 = model.fit(
        [X_feat_tr, X_img_tr], y_tr,
        validation_data=([X_feat_val, X_img_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_wts,
        callbacks=callbacks,
        verbose=1
    )

    # ── Step 5: Hard negative mining ──────────────────────────────────────────
    print("\n[Step 6] Hard negative mining...")
    extra_feat, extra_img, extra_lbl = mine_hard_negatives(
        model, X_feat_tr, X_img_tr, y_tr, le
    )

    X_feat_tr2 = np.concatenate([X_feat_tr, extra_feat])
    X_img_tr2  = np.concatenate([X_img_tr,  extra_img])
    y_tr2      = np.concatenate([y_tr,       extra_lbl])

    print("\n[Step 7] Training phase 2 (with hard negatives)...")
    callbacks2 = get_callbacks(checkpoint_path=model_out.replace(".h5", "_v2.h5"))

    hard_class_wts = compute_class_weights(
        aug_labels, le, HARD_CLASS, hard_weight=3.0
    )

    history2 = model.fit(
        [X_feat_tr2, X_img_tr2], y_tr2,
        validation_data=([X_feat_val, X_img_val], y_val),
        epochs=max(30, epochs // 3),
        batch_size=batch_size,
        class_weight=hard_class_wts,
        callbacks=callbacks2,
        verbose=1
    )

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    print("\n[Step 8] Final evaluation...")
    val_loss, val_acc = model.evaluate(
        [X_feat_val, X_img_val], y_val, verbose=0
    )
    print(f"  Validation accuracy : {val_acc*100:.2f}%")
    print(f"  Validation loss     : {val_loss:.4f}")

    # ── Step 7: Plots ─────────────────────────────────────────────────────────
    print("\n[Step 9] Generating plots...")
    plot_history(history1, "model/training_curves_phase1.png")
    plot_history(history2, "model/training_curves_phase2.png")

    y_pred = np.argmax(model.predict([X_feat_val, X_img_val], verbose=0), axis=1)
    y_true = np.argmax(y_val, axis=1)
    plot_confusion_matrix(y_true, y_pred, le.classes_,
                          "model/confusion_matrix_advanced.png")

    print("\n" + classification_report(
        y_true, y_pred, target_names=le.classes_, zero_division=0
    ))

    # ── Log ──────────────────────────────────────────────────────────────────
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open("model/model_log.txt", "a") as f:
        f.write(
            f"{ts} | acc: {val_acc*100:.2f}% | loss: {val_loss:.4f} | "
            f"samples: {len(y_tr)} | classes: {n_classes} | "
            f"file: asl_fusion_model.h5 | type: fusion\n"
        )

    print(f"\n  Best model saved → {model_out}")
    print("  Done!\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL fusion model")
    parser.add_argument("--data",    default="data",                      help="Data root directory")
    parser.add_argument("--epochs",  default=100, type=int,               help="Max training epochs")
    parser.add_argument("--batch",   default=32,  type=int,               help="Batch size")
    parser.add_argument("--augment", default=8,   type=int,               help="Augmentations per sample")
    parser.add_argument("--out",     default="model/asl_fusion_model.h5", help="Output model path")
    args = parser.parse_args()

    train(data_root=args.data, epochs=args.epochs,
          batch_size=args.batch, augment_factor=args.augment,
          model_out=args.out)
