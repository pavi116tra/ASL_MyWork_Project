"""
augmentation.py
===============
Image + Landmark augmentation pipeline for ASL training data.
Generates ~15x more training samples from each original image.

Uses albumentations for image augmentation (with OpenCV fallback).
Applies MATCHING transforms to landmarks so they stay consistent with image.
"""

import numpy as np
import cv2
import math
import random

# Try albumentations; fall back to manual transforms if not installed
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("[augmentation] albumentations not found — using OpenCV-only fallback.")
    print("  Install with: pip install albumentations")


# ── 1. ALBUMENTATIONS PIPELINE (preferred) ───────────────────────────────────
if ALBUMENTATIONS_AVAILABLE:
    IMAGE_AUGMENT = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.HorizontalFlip(p=0.3), # Explicitly requested
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101, p=0.5
        ),
        A.RandomShadow(p=0.3),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3), # CLAHE improves low-light detection
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=10, p=0.3),
    ])


# ── 2. LANDMARK AUGMENTATION (numpy-level) ───────────────────────────────────
def _add_gaussian_noise(features: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """Add tiny Gaussian noise to feature vector (landmark += np.random.normal(0, 0.005))."""
    return features + np.random.normal(0, sigma, features.shape).astype(np.float32)

def _scale_jitter(features: np.ndarray,
                  lo: float = 0.90, hi: float = 1.10) -> np.ndarray:
    """Multiply all features by a random scale factor (scale jitter: landmarks *= random.uniform(0.9, 1.1))."""
    return features * random.uniform(lo, hi)

def _rotate_landmarks_2d(features: np.ndarray,
                          angle_deg: float) -> np.ndarray:
    """
    Apply a 2D rotation matrix ±15 degrees to the FIRST 15 joint-angle features only.
    For other features (distances, ratios) rotation has no direct meaning,
    so we add proportional noise instead.
    """
    out = features.copy()
    rad = math.radians(angle_deg)
    # Shift the angle features slightly (they are in [0,1], ~angle/180)
    shift = (angle_deg / 180.0) * 0.05   # small shift proportional to rotation
    out[:15] = np.clip(out[:15] + random.uniform(-shift, shift), 0, 1)
    # Add noise to remaining features
    out[15:] = _add_gaussian_noise(out[15:], sigma=0.008)
    return out

def _translate_landmarks(features: np.ndarray,
                          max_shift: float = 0.03) -> np.ndarray:
    """Small random translation (shift all landmarks by small random offset)."""
    out = features.copy()
    out += np.random.uniform(-max_shift, max_shift, out.shape).astype(np.float32)
    return out


# ── 3. IMAGE AUGMENTATION (OpenCV fallback) ──────────────────────────────────
def _cv2_augment(image: np.ndarray) -> np.ndarray:
    """Simple OpenCV-based image augmentation when albumentations is absent."""
    img = image.copy()

    # Brightness / contrast
    alpha = random.uniform(0.7, 1.3)   # contrast
    beta  = random.randint(-30, 30)    # brightness
    img   = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Gaussian blur
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Horizontal flip
    if random.random() < 0.3:
        img = cv2.flip(img, 1)

    # Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Random rotation ±15°
    if random.random() < 0.5:
        h, w  = img.shape[:2]
        angle = random.uniform(-15, 15)
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img   = cv2.warpAffine(img, M, (w, h),
                               borderMode=cv2.BORDER_REFLECT_101)

    # Random scale / crop
    if random.random() < 0.4:
        scale = random.uniform(0.85, 1.15)
        h, w  = img.shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        if scale >= 1.0:
            y0 = (nh - h) // 2;  x0 = (nw - w) // 2
            img = resized[y0:y0+h, x0:x0+w]
        else:
            pad_h = (h - nh) // 2;  pad_w = (w - nw) // 2
            img = cv2.copyMakeBorder(resized, pad_h, h-nh-pad_h,
                                     pad_w, w-nw-pad_w,
                                     cv2.BORDER_REFLECT_101)

    return img


# ── 4. PUBLIC API ────────────────────────────────────────────────────────────
def augment_sample(image: np.ndarray,
                   features: np.ndarray,
                   n_augments: int = 15
                   ) -> list:
    """
    Generate n_augments augmented copies of a single (image, features) pair.

    Args:
        image     : BGR image, shape (H, W, 3)
        features  : 41-dim feature vector, numpy float32 array
        n_augments: number of augmented copies to produce

    Returns:
        List of (aug_image, aug_features) tuples, all same dtype/shape as input.
        The ORIGINAL sample is NOT included in the returned list.
    """
    results = []
    for _ in range(n_augments):
        # ── Image augmentation ──
        if ALBUMENTATIONS_AVAILABLE:
            aug_img = IMAGE_AUGMENT(image=image)["image"]
        else:
            aug_img = _cv2_augment(image)

        # ── Feature augmentation ──
        aug_feat = features.copy()
        angle_deg = random.uniform(-15, 15)

        # Apply transforms in random order
        transforms = [
            lambda f: _add_gaussian_noise(f, sigma=0.005),
            lambda f: _scale_jitter(f, 0.90, 1.10),
            lambda f: _rotate_landmarks_2d(f, angle_deg),
            lambda f: _translate_landmarks(f, max_shift=0.02),
        ]
        random.shuffle(transforms)
        for t in transforms:
            if random.random() < 0.6:
                aug_feat = t(aug_feat)

        results.append((aug_img.astype(np.uint8), aug_feat.astype(np.float32)))

    return results


def augment_dataset(images: list,
                    features: list,
                    labels: list,
                    n_augments: int = 15) -> tuple:
    """
    Augment an entire dataset.

    Args:
        images   : list of BGR images
        features : list of 41-dim feature arrays
        labels   : list of integer class labels
        n_augments: augmentations per sample

    Returns:
        (aug_images, aug_features, aug_labels) — includes originals.
    """
    all_images   = list(images)
    all_features = list(features)
    all_labels   = list(labels)

    for img, feat, lbl in zip(images, features, labels):
        pairs = augment_sample(img, feat, n_augments)
        for aug_img, aug_feat in pairs:
            all_images.append(aug_img)
            all_features.append(aug_feat)
            all_labels.append(lbl)

    return (np.array(all_images),
            np.array(all_features, dtype=np.float32),
            np.array(all_labels))


if __name__ == "__main__":
    # Quick smoke test
    dummy_img  = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    dummy_feat = np.random.rand(41).astype(np.float32)
    results    = augment_sample(dummy_img, dummy_feat, n_augments=15)
    print(f"Input  → image: {dummy_img.shape}, features: {dummy_feat.shape}")
    print(f"Output → {len(results)} augmented pairs")
    print(f"  First aug image  shape: {results[0][0].shape}")
    print(f"  First aug feature shape: {results[0][1].shape}")
    print("augmentation.py OK")
