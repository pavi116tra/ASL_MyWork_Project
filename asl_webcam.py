"""
ASL Gesture Recognition - Live Webcam Inference
Rebuilds the exact CNN architecture and loads weights from asl_model.h5.
Run with:  python asl_webcam.py
"""

import os
import sys
import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"   # force CPU, skip GPU detection delay

print("[INFO] Importing TensorFlow (this may take ~30s on first run) ...")
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import h5py

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH    = r"C:\Users\pavit\Downloads\asl_model.h5"
IMG_SIZE      = (64, 64)
ROI           = (100, 400, 100, 400)   # y1, y2, x1, x2
CONF_THRESH   = 0.50
STABLE_FRAMES = 8

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "delete", "nothing"]

# ─── Build model ──────────────────────────────────────────────────────────────
print("[INFO] Building model architecture ...")
model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32,  (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64,  (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASSES), activation='softmax'),
])

# ─── Load weights ─────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    sys.exit(f"[ERROR] Model not found: {MODEL_PATH}")

print("[INFO] Loading weights ...")
# Path structure: model_weights/<name>/sequential/<name>/{kernel,bias}
with h5py.File(MODEL_PATH, 'r') as f:
    wg = f['model_weights']

    def load_layer_weights(layer, saved_name):
        g = wg[saved_name]['sequential'][saved_name]
        kernel = g['kernel'][()]
        bias   = g['bias'][()]
        layer.set_weights([kernel, bias])
        print(f"  + {saved_name}: kernel={kernel.shape}")

    load_layer_weights(model.layers[0], 'conv2d')    # Conv2D 32
    load_layer_weights(model.layers[2], 'conv2d_1')  # Conv2D 64
    load_layer_weights(model.layers[4], 'conv2d_2')  # Conv2D 128
    load_layer_weights(model.layers[7], 'dense')     # Dense 256
    load_layer_weights(model.layers[9], 'dense_1')   # Dense 29 (output)

print("[INFO] Weights loaded. Starting webcam ...")
print("       >> Press 'q' to quit, 'c' to clear text, hand in the BLUE BOX")

# ─── Helpers ──────────────────────────────────────────────────────────────────
history_buf  = []
built_text   = ""
add_cooldown = 0

def predict_frame(roi_bgr):
    img = cv2.resize(roi_bgr, IMG_SIZE).astype("float32") / 255.0
    preds = model(np.expand_dims(img, 0), training=False).numpy()[0]
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx])

def stable_prediction(label):
    history_buf.append(label)
    if len(history_buf) > STABLE_FRAMES:
        history_buf.pop(0)
    if len(history_buf) == STABLE_FRAMES and len(set(history_buf)) == 1:
        return label
    return None

def draw_overlay(frame, label, conf, text):
    y1, y2, x1, x2 = ROI
    # Blue ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 200, 255), 2)
    cv2.putText(frame, "Hand here", (x1 + 4, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

    # Prediction banner (top-left dark strip)
    banner = f"{label}  {conf*100:.0f}%" if label else "---"
    cv2.rectangle(frame, (0, 0), (370, 62), (15, 15, 15), -1)
    cv2.putText(frame, banner, (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 220, 255), 2)

    # Bottom text strip
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 58), (w, h), (15, 15, 15), -1)
    display = text[-42:] if len(text) > 42 else text
    cv2.putText(frame, f"Text: {display}_", (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 255, 160), 2)
    cv2.putText(frame, "q=quit   c=clear",
                (10, h - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

# ─── Main loop ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("[ERROR] Cannot open webcam (index 0). Check camera connection.")

y1, y2, x1, x2 = ROI

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to grab frame — exiting.")
        break

    frame = cv2.flip(frame, 1)
    roi   = frame[y1:y2, x1:x2]

    label, conf = predict_frame(roi)
    stable = stable_prediction(label) if conf >= CONF_THRESH else None

    add_cooldown = max(0, add_cooldown - 1)
    if stable and stable != "nothing" and add_cooldown == 0:
        if stable == "space":
            built_text += " "
        elif stable == "delete":
            built_text = built_text[:-1]
        else:
            built_text += stable
        add_cooldown = STABLE_FRAMES * 2

    draw_overlay(frame, label if conf >= CONF_THRESH else None, conf, built_text)
    cv2.imshow("ASL Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        built_text = ""
        history_buf.clear()

cap.release()
cv2.destroyAllWindows()
print("[INFO] Goodbye!")
