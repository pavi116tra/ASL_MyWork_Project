# 🤟 ASL Gesture Recognition — Project Overview
### `ASL_about.md`

---

## 📌 What Is This Project?

This is an **American Sign Language (ASL) Hand Gesture Recognition** system that uses a webcam to detect hand signs in real time. The user shows a hand sign, clicks **CAPTURE**, and the AI model detects which ASL letter it is — then the user can confirm it to build words and sentences.

> Built with **Python**, **MediaPipe**, **TensorFlow/Keras**, and **OpenCV**.

---

## 🎯 Project Goal

| Goal | Description |
|---|---|
| Detect | Recognize 26 ASL alphabet letters (A–Z) + `space`, `delete` |
| Accuracy | Achieved **88.51%** validation accuracy (original model) |
| Interface | Live split-screen webcam app with CAPTURE → CONFIRM → sentence builder flow |
| Advanced | Fusion model pipeline with 41-dim features + image CNN (optional upgrade) |

---

## 🗂️ Project Structure

```
Gesture_ASL_Regconization-main/
│
├── ⭐ asl_app.py                  ← MAIN APP (advanced fusion, split-screen UI)
├── feature_extractor.py           ← 41-dim advanced feature extractor
├── fusion_model.py                ← Dual-input fusion model architecture
├── augmentation.py                ← Image + landmark augmentation pipeline
├── collect_data_advanced.py       ← Collect training images + landmarks
├── train_advanced.py              ← Advanced training pipeline (fusion model)
├── setup_and_run_asl.ps1          ← PowerShell auto-setup script
├── kaggle.json                    ← Kaggle API credentials (dataset download)
├── Untitled23.ipynb               ← Jupyter notebook (training experiments)
│
└── files/
    ├── setup_data.py              ← Download/organise Kaggle dataset
    │
    └── model/
        ├── asl_model_20260509_182234.h5  ← Best trained model (88.51% acc)
        ├── asl_dense_model_BEST.h5       ← Copy always pointing to best
        ├── label_classes.npy             ← 28 class labels (A-Z + space + delete)
        ├── labels_20260509_182234.npy    ← Versioned labels backup
        └── model_log.txt                 ← Training log with accuracy records
```

---

## ⭐ Most Important Files

| File | Importance | What It Does |
|---|---|---|
| `asl_app.py` | 🔴 CRITICAL | **Main application** — advanced fusion app, split-screen UI, CAPTURE/CONFIRM/DELETE, temporal smoothing, hand quality scoring |
| `feature_extractor.py` | 🔴 CRITICAL | Extracts **41-dim** advanced geometric features (joint angles, curl, distances, wrist angle) |
| `fusion_model.py` | 🔴 CRITICAL | Builds the dual-input fusion model (41-dim landmarks + 128×128 cropped hand image CNN) |
| `train_advanced.py` | 🔴 CRITICAL | Full training pipeline: load, augment, train fusion model, hard negative mining |
| `augmentation.py` | 🔴 CRITICAL | 15× data augmentation for both images and landmarks |
| `collect_data_advanced.py` | 🟠 IMPORTANT | Collects training images + pre-extracts landmark features per frame |
| `files/model/asl_model_*.h5` | 🟠 IMPORTANT | Trained model weights — needed to run the app |
| `files/model/label_classes.npy` | 🟠 IMPORTANT | Maps model output indices to letter labels (A, B, C…) |
| `files/setup_data.py` | 🟡 USEFUL | Utility script to download/organize Kaggle dataset |
| `setup_and_run_asl.ps1` | 🟢 HELPER | Auto-installs dependencies and launches the app |

---

## 📁 Each File — What It Does

### 🔵 Core Application (Root Level)

| File | Description |
|---|---|
| **`asl_app.py`** | The upgraded main app. Uses the **advanced fusion model** (41-dim features). Split-screen layout: LEFT = live webcam with hand skeleton, RIGHT = captured photo. Temporal smoothing (majority vote over 7 frames), hand quality scoring (GOOD/PARTIAL/POOR), top-3 predictions with confidence bar graphs, CAPTURE/CONFIRM/DELETE flow. Falls back to 93-dim model if no fusion model is found. |

### 🔵 Advanced Feature & Model Pipeline (Root Level)

| File | Description |
|---|---|
| **`feature_extractor.py`** | Extracts a **41-dimensional** advanced geometric feature vector from MediaPipe hand landmarks. Feature groups: joint angles (15), fingertip distances (5), finger curl ratios (5), inter-fingertip distances (10), thumb opposition (4), wrist angle (2). Replaces raw (x,y,z) coordinates with scale-invariant geometric features. |
| **`fusion_model.py`** | Builds the dual-input fusion model architecture: one branch processes 41-dim landmark features, the other processes a 128×128 cropped hand image via a lightweight CNN. Outputs fused classification over all classes. |
| **`augmentation.py`** | Generates up to 15× augmented training samples from each image+landmark pair. Image augmentation uses albumentations (or OpenCV fallback). Landmark augmentation applies Gaussian noise, scale jitter, 2D rotation, and translation at feature level. |
| **`collect_data_advanced.py`** | Webcam-based data collection tool. Records hand images AND simultaneously extracts and saves the 41-dim landmark feature vectors for each captured frame. Organises into `data/images/{LETTER}/` and `data/landmarks/{LETTER}/`. |
| **`train_advanced.py`** | Full training pipeline: loads paired images + landmark .npy files, augments 8× per sample, trains the dual-input fusion model (Phase 1), performs hard negative mining on confused classes (A/E/M/N/S/T/G/H), and retrains with oversampled hard negatives (Phase 2). Saves best model to `files/model/asl_fusion_model.h5`. |

### 🔵 Utility & Data Setup (files/)

| File | Description |
|---|---|
| **`files/setup_data.py`** | Downloads ASL dataset from Kaggle. Organises into `data/train/` and `data/val/`. |

---

## 🧠 Techniques Used

### 1. 🖐️ Hand Landmark Extraction — MediaPipe
- **MediaPipe Hands** detects **21 keypoints** (landmarks) on the hand
- Each landmark has (x, y, z) coordinates
- Used in both live video and static image (snapshot) mode

### 2. 📐 41-Dimensional Advanced Feature Vector
The current production model uses a hand-crafted **geometric, scale-invariant** feature vector:

| Feature Group | Dimensions | What It Captures |
|---|---|---|
| Joint angles (normalized) | 15-dim | Bend at each of 3 joints × 5 fingers |
| Fingertip distances | 5-dim | Distance from each fingertip to palm centre (scale-normalized) |
| Finger curl ratios | 5-dim | 0 = curled, 1 = extended (straight/chain length ratio) |
| Inter-fingertip distances | 10-dim | All C(5,2)=10 pairwise fingertip gaps |
| Thumb opposition | 4-dim | Thumb tip distance to each finger's PIP joint |
| Wrist angle | 2-dim | Roll and pitch of hand orientation |
| **Total** | **41-dim** | |

> **Legacy model** (fallback): uses 93-dim raw landmark + angle + distance + thumb + pinch features.

### 3. 🤖 Dual-Input Fusion Model
```
Input A: 41-dim features → Dense branch
Input B: 128×128 image  → CNN branch (Conv2D + pooling layers)
Both branches → Concatenated → Dense → Softmax (n_classes)
```
- Combines geometric landmark features + raw image appearance
- Trained with hard negative mining on difficult class pairs

### 4. 🧪 Training Enhancements
- **Data augmentation**: 8–15× per sample (albumentations + landmark jitter)
- **Hard negative mining**: Oversample misclassified A/E/M/N/S/T/G/H samples ×3
- **Class weights**: Higher penalty for confusable classes
- **Two-phase training**: Initial fit → mine hard negatives → fine-tune

### 5. ⏱️ Temporal Smoothing
- Majority vote over last **7 frames** (need ≥5/7 agreement to show stable prediction)
- Only runs inference when hand quality is **GOOD**
- Prevents flickering and false positives

### 6. 🏆 Hand Quality Scoring
- **GOOD**: Hand fully in frame, good stereoscopic depth
- **PARTIAL**: Hand at edge or flat angle to camera
- **POOR**: Hand out of frame or not detected
- Capture is only allowed when quality is **GOOD**

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Name** | ASL Alphabet Dataset |
| **Source** | [Kaggle — grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| **Classes** | 26 letters (A–Z) + `space` + `delete` = **28 classes** |
| **Images** | ~87,000 training images (3,000 per class) |
| **Format** | JPG images, 200×200 pixels |
| **Download** | Run `python files/setup_data.py` (needs `kaggle.json`) |

### Dataset Folder Structure (after setup)
```
data/
├── train/
│   ├── A/   (3000 images)
│   ├── B/
│   └── ...
├── val/
│   ├── A/   (300 images)
│   └── ...
├── images/{LETTER}/    ← collected by collect_data_advanced.py
└── landmarks/{LETTER}/ ← 41-dim .npy files per image
```

---

## 📓 Notebooks

| File | Description |
|---|---|
| **`Untitled23.ipynb`** | Jupyter notebook used for training experiments, testing feature extraction, and visualising results during development |

---

## 🏃 How to Run

### Option A — Quick Start (Auto-Setup)
```powershell
.\setup_and_run_asl.ps1
```

### Option B — Manual Setup

#### Step 1 — Install dependencies
```powershell
pip install tensorflow==2.12 mediapipe opencv-python scikit-learn numpy albumentations seaborn matplotlib
```

#### Step 2 — Download Dataset (optional, for training)
```powershell
python files/setup_data.py
```

#### Step 3 — Collect Own Training Data (optional)
```powershell
python collect_data_advanced.py
```

#### Step 4 — Train Advanced Fusion Model (optional)
```powershell
python train_advanced.py --data data --epochs 100 --augment 8
```

#### Step 5 — ⭐ Run the Main App
```powershell
python asl_app.py
```

> The app auto-detects and loads the best available model (fusion → landmark-only → legacy 93-dim).

---

## 🎮 App Controls

| Key / Button | Action |
|---|---|
| `C` / CAPTURE | Take photo of hand sign (only when hand quality = GOOD) |
| `SPACE` / CONFIRM | Add detected letter to word (only when confidence ≥ 80%) |
| `BACKSPACE` / DELETE | Discard snapshot, try again |
| `ENTER` | Finalise word → add to sentence |
| `ESC` | Clear everything |
| `Q` | Quit |
| STOP CAM button | Pause/resume camera |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Validation Accuracy | **88.51%** |
| Validation Loss | 0.4410 |
| Training Samples | 4,096 |
| Classes | 28 |
| Model File | `files/model/asl_model_20260509_182234.h5` |
| Trained On | 2026-05-09 18:22 |

> Advanced fusion model performance depends on your own collected data via `collect_data_advanced.py`.

---

## 🔗 GitHub Repositories

| Repo | URL |
|---|---|
| Main project | https://github.com/pavi116tra/ASL_MyWork_Project |
| Original repo | https://github.com/pavi116tra/Gesture_ASL_Regconization |

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10 | Main language |
| TensorFlow / Keras | 2.12 | Model training & inference |
| MediaPipe | Latest | Hand landmark detection |
| OpenCV | Latest | Webcam capture & drawing |
| NumPy | Latest | Feature vector math |
| scikit-learn | Latest | Label encoding, train/val split |
| albumentations | Latest | Advanced image augmentation |
| seaborn / matplotlib | Latest | Training plots & confusion matrix |
| h5py | Latest | Model file compatibility |

---

## 🚀 Future Roadmap & Advanced Upgrades

To push the real-world accuracy from **88% to 95%+**, the system can be upgraded using the following multi-phased roadmap. These upgrades target dataset diversity, temporal dynamics, and pipeline optimizations.

### 1. 📊 Better & More Diverse Datasets
The current Kaggle dataset is limited by single-user, single-background environments. Augmenting or transitioning to the following datasets will dramatically improve real-world robustness:

| Dataset | Volume | Key Benefit & Description |
|---|---|---|
| **WLASL (World Level ASL)** | 21,000+ video clips | Overcomes environmental bias with hundreds of signers, adding **5–10% real-world accuracy**. |
| **MS-ASL (Microsoft)** | 25,000 video clips | Ethnically diverse signers across multiple environments and lighting settings. |
| **Custom Multi-Signer** | ~50 samples / letter | Collect 50 samples per letter from 3–4 different people under varied lighting using `collect_data_advanced.py`. |

---

### 2. 🧠 Advanced Algorithms (Motion & Graph Networks)
ASL gestures are inherently dynamic (especially letters like **J** and **Z**). Replacing static snapshot models with temporal or skeletal graph models yields the highest performance gains:

*   **MediaPipe Holistic Tracking (Drop-in Upgrade)**
    *   *What:* Tracks face (468 pts), pose (33 pts), and hand landmarks (21 pts).
    *   *Why:* Disambiguates subtle wrist tilt and body postures in highly similar signs (e.g., **A** vs **S** vs **E**).
*   **Temporal Sequence Models (LSTM / Transformers)**
    *   *What:* Processes sequential sequences of 10–15 landmark frames instead of single frames.
    *   *Why:* Natively handles dynamic signing motion, replacing heuristic temporal smoothing.
*   **Spatial-Temporal Graph Convolutional Networks (ST-GCN)**
    *   *What:* Models hand joints as graph nodes and bones as edges over space and time.
    *   *Why:* The state-of-the-art approach for skeletal gesture recognition; consistently outperforms plain Dense/CNN networks.

---

### 3. ⚡ Quick Wins & Production Tweaks (Current Pipeline)
Targeted micro-optimizations that can be implemented directly on the current setup:

> [!TIP]
> **Targeted Hard-Pair Expansion:**
> Use the confusion matrix to identify the top 5 confused pairs (e.g., **M/N**, **A/E**, **G/H**). Collect 500+ targeted extra samples specifically for these pairs. Targeted data is extremely powerful for removing classification bottlenecks.

> [!NOTE]
> **Weighted Temporal Voting:**
> Enhance the current majority voting window (7 frames) by introducing a decay factor where more recent frames carry higher decision weight.

> [!IMPORTANT]
> **Per-User Calibration:**
> Have the user sign 3 calibration letters upon first launching the app. Fine-tune the final dense classification layer on these 3 samples to adapt the model to their specific hand proportions, boosting local accuracy by **8–12%**.

---

### 🎯 Recommended Upgrade Path
For the fastest and most efficient accuracy jump, follow this path:
```
WLASL Data Expansion ──> MediaPipe Holistic ──> Targeted Confusion Collection ──> 95%+ Real-World Accuracy
```

---

*Project by pavi116tra — ASL Gesture Recognition with Advanced Fusion Model & Split-Screen Capture UX*

