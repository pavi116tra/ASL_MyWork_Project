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
| Accuracy | Achieved **88.51%** validation accuracy |
| Interface | Live split-screen webcam app with CAPTURE → CONFIRM → sentence builder flow |

---

## 🗂️ Project Structure

```
Gesture_ASL_Regconization-main/
├── Untitled23.ipynb              ← Jupyter notebook (training experiments)
├── asl_webcam.py                 ← Original basic webcam script
├── setup_and_run_asl.ps1         ← PowerShell auto-setup script
├── kaggle.json                   ← Kaggle API credentials (for dataset download)
│
└── files/
    ├── ⭐ asl_app_pro.py          ← MAIN APP (split-screen, CAPTURE/CONFIRM)
    ├── asl_sentence_builder.py   ← Older full-featured sentence builder app
    ├── asl_app_mediapipe.py      ← Earlier MediaPipe-only version
    ├── asl_app_mobilenetv2.py    ← MobileNetV2 image classifier version
    ├── asl_app_mobilenetv2_FIXED.py ← Fixed MobileNetV2 version
    │
    ├── extract_landmarks.py      ← Step 1: Extract 93-dim features from dataset
    ├── train_mediapipe.py        ← Step 2: Train the Dense Neural Network model
    ├── evaluate_accuracy.py      ← Step 3: Evaluate and test the trained model
    ├── setup_data.py             ← Download/organize Kaggle dataset
    │
    ├── asl_mediapipe_dense.h5    ← Saved model (root-level backup)
    │
    ├── model/
    │   ├── asl_model_20260509_182234.h5  ← Best trained model (88.51% acc)
    │   ├── asl_dense_model_BEST.h5       ← Copy always pointing to best
    │   ├── label_classes.npy             ← 28 class labels (A-Z + space + delete)
    │   ├── labels_20260509_182234.npy    ← Versioned labels backup
    │   └── model_log.txt                 ← Training log with accuracy records
    │
    └── test_*.py                 ← Debug/test scripts (mp, upb, crop tests)
```

---

## ⭐ Most Important Files

| File | Importance | What It Does |
|---|---|---|
| `files/asl_app_pro.py` | 🔴 CRITICAL | **Main application** — split-screen UI, CAPTURE/CONFIRM/DELETE flow |
| `files/extract_landmarks.py` | 🔴 CRITICAL | Converts dataset images → 93-dim feature vectors (.npy files) |
| `files/train_mediapipe.py` | 🔴 CRITICAL | Trains the Dense Neural Network on extracted landmarks |
| `files/model/asl_model_*.h5` | 🔴 CRITICAL | Trained model weights — needed to run the app |
| `files/model/label_classes.npy` | 🟠 IMPORTANT | Maps model output indices to letter labels (A, B, C…) |
| `files/asl_sentence_builder.py` | 🟡 USEFUL | Older version of the app with hold-timer UX |
| `files/evaluate_accuracy.py` | 🟡 USEFUL | Run to check model performance on test data |
| `setup_and_run_asl.ps1` | 🟢 HELPER | Auto-installs dependencies and launches the app |

---

## 📁 Each File — What It Does

### 🔵 Core Application
| File | Description |
|---|---|
| **`asl_app_pro.py`** | The final, polished app. Split-screen layout: LEFT = live webcam with hand skeleton, RIGHT = captured photo. When user presses **C**, snapshot is taken and model detects the letter. Result shown below both panels as `A — 99% accuracy`. CONFIRM adds it to word, DELETE discards. |
| **`asl_sentence_builder.py`** | Earlier version. Uses a hold-timer approach (hold sign steady for 1.5s → auto-detects → SPACE to confirm). Fully functional but replaced by `asl_app_pro.py`. |
| **`asl_webcam.py`** | The original very basic webcam script. No sentence builder, just raw detection. |

### 🔵 Data Pipeline
| File | Description |
|---|---|
| **`setup_data.py`** | Downloads the ASL dataset from Kaggle using the API. Organises into `data/train/` and `data/val/` folders. |
| **`extract_landmarks.py`** | Reads each image from the dataset, runs MediaPipe to detect the hand, then extracts a **93-dimensional feature vector** (landmarks + angles + distances + thumb position + pinch gaps). Saves one `.npy` file per class in `data/features/`. |
| **`train_mediapipe.py`** | Loads all `.npy` feature files, trains a Dense Neural Network (256→128→64→n_classes), uses oversampling on difficult classes (A/S/E/T/M/N/C/O), saves the model and logs accuracy. |
| **`evaluate_accuracy.py`** | Loads the best model and runs it on validation/test images to measure accuracy per class. |

### 🔵 Alternative Models (Experimental)
| File | Description |
|---|---|
| **`asl_app_mobilenetv2.py`** | Attempted to use MobileNetV2 (image-based CNN) instead of MediaPipe landmarks. |
| **`asl_app_mobilenetv2_FIXED.py`** | Fixed version of MobileNetV2 approach with better preprocessing. |
| **`train_mobilenetv2.py`** | Training script for MobileNetV2 — fine-tunes the pre-trained CNN on ASL images. |

### 🔵 Test / Debug Scripts
| File | Description |
|---|---|
| `test_mp.py` | Tests if MediaPipe imports correctly |
| `test_mp2.py` | Tests MediaPipe hand detection on a sample image |
| `test_crop.py` | Tests image cropping and preprocessing |
| `test_upb.py` | Tests the protobuf/UPB compatibility fix |

---

## 🧠 Techniques Used

### 1. 🖐️ Hand Landmark Extraction — MediaPipe
- **MediaPipe Hands** detects **21 keypoints** (landmarks) on the hand
- Each landmark has (x, y, z) coordinates
- Used in both live video and static image mode

### 2. 📐 93-Dimensional Feature Vector
The model does NOT see pixel images — it sees a hand-crafted feature vector:

| Feature Group | Dimensions | What It Captures |
|---|---|---|
| Landmark positions (normalized) | 63-dim | All 21 joints relative to wrist, scale-normalized |
| Joint angles | 10-dim | Finger bend angles (rotation-invariant) |
| Fingertip distances | 8-dim | How far each fingertip is from palm center |
| Thumb position | 5-dim | Where thumb sits relative to fist (separates A/S/T/I) |
| Pinch gaps | 7-dim | Thumb-to-fingertip gaps (separates C/O/D/G/F) |
| **Total** | **93-dim** | |

### 3. 🤖 Dense Neural Network (DNN)
```
Input (93) → Dense(256, ReLU) → BN → Dropout(0.4)
           → Dense(128, ReLU) → BN → Dropout(0.3)
           → Dense(64, ReLU)
           → Dense(28, Softmax)   ← 28 classes
```
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau
- Oversampling on hard classes: A, S, E, T, M, N, C, O (×3)

### 4. 📏 Geometric Override Rules
Hard-coded geometry rules to fix common confusions:
- `A + pinky extended → I`
- `I + pinky NOT extended → A`
- `O + large thumb-index gap → C`
- `C + small thumb-index gap → O`

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Name** | ASL Alphabet Dataset |
| **Source** | [Kaggle — grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| **Classes** | 26 letters (A–Z) + `space` + `delete` = **28 classes** |
| **Images** | ~87,000 training images (3,000 per class) |
| **Format** | JPG images, 200×200 pixels |
| **Download** | Run `python setup_data.py` (needs `kaggle.json`) |

### Dataset Folder Structure (after setup)
```
data/
├── train/
│   ├── A/   (3000 images)
│   ├── B/   (3000 images)
│   ├── ...
│   └── Z/   (3000 images)
└── val/
    ├── A/   (300 images)
    ├── B/   (300 images)
    └── ...
```

### Extracted Features Path
```
data/features/
├── A.npy    ← 93-dim vectors for letter A
├── B.npy
├── ...
└── Z.npy
```

---

## 📓 Notebooks

| File | Description |
|---|---|
| **`Untitled23.ipynb`** | Jupyter notebook used for training experiments, testing feature extraction, and visualizing results during development |

---

## 🏃 How to Run

### Step 1 — Install dependencies
```powershell
pip install tensorflow==2.12 mediapipe opencv-python scikit-learn numpy
```

### Step 2 — Download Dataset (optional, for training)
```powershell
python files/setup_data.py
```

### Step 3 — Extract Landmarks (optional, for training)
```powershell
python files/extract_landmarks.py --data data --out data/features
```

### Step 4 — Train Model (optional, if no model exists)
```powershell
python files/train_mediapipe.py
```

### Step 5 — ⭐ Run the App
```powershell
& "C:\Users\pavit\AppData\Local\Programs\Python\Python310\python.exe" files/asl_app_pro.py
```

---

## 🎮 App Controls

| Key / Button | Action |
|---|---|
| `C` / CAPTURE | Take photo of hand sign |
| `SPACE` / CONFIRM | Add detected letter to word |
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
| Model File | `model/asl_model_20260509_182234.h5` |
| Trained On | 2026-05-09 18:22 |

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
| h5py | Latest | Model file compatibility |

---

*Project by pavi116tra — ASL Gesture Recognition with Split-Screen Capture UX*
