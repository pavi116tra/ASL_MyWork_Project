"""
asl_app.py
==========
Upgraded ASL real-time app using the advanced fusion model.

Features:
  ✓ Uses 41-dim advanced feature extractor
  ✓ Temporal smoothing: majority vote over last 7 frames (need >=5/7 agreement)
  ✓ Confidence calibration with color coding
  ✓ Top-3 predictions always shown with bar graphs
  ✓ Hand quality score (GOOD / PARTIAL / POOR)
  ✓ ONLY predicts live when quality is GOOD
  ✓ ONLY allows CONFIRM when confidence > 80%
"""

import os
import sys
import time
import collections
import numpy as np
import cv2
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"

import tensorflow as tf

# MediaPipe protobuf fix
try:
    import google._upb._message
    import google.protobuf.symbol_database as sym_db
    import google.protobuf.message_factory as msg_factory
    if not hasattr(sym_db.SymbolDatabase, "GetPrototype"):
        sym_db.SymbolDatabase.GetPrototype = (
            lambda self, d: msg_factory.GetMessageClass(d)
        )
except Exception:
    pass

import mediapipe as mp
from feature_extractor import extract_advanced_features

# ══════════════════════════════════════════════════════════════════════════════
# LEGACY FEATURE EXTRACTION (93-dim) - fallback compat
# ══════════════════════════════════════════════════════════════════════════════
def _landmarks_vec(h):
    wrist = h.landmark[0]
    v = []
    for lm in h.landmark:
        v.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    mid = h.landmark[9]
    s = np.sqrt((mid.x-wrist.x)**2 + (mid.y-wrist.y)**2 + (mid.z-wrist.z)**2)
    if s > 0:
        v = [x / s for x in v]
    return np.array(v, dtype=np.float32)

def _angle_vec(h):
    lm = h.landmark
    def ang(a, b, c):
        ba = np.array([a.x-b.x, a.y-b.y, a.z-b.z])
        bc = np.array([c.x-b.x, c.y-b.y, c.z-b.z])
        return np.degrees(np.arccos(np.clip(
            np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6), -1, 1)))
    joints = [(1,2,3),(2,3,4),(5,6,7),(6,7,8),(9,10,11),
              (10,11,12),(13,14,15),(14,15,16),(17,18,19),(18,19,20)]
    return np.array([ang(lm[a],lm[b],lm[c]) for a,b,c in joints], dtype=np.float32)

def _dist_vec(h):
    lm   = h.landmark
    palm = [lm[i] for i in [0,1,5,9,13,17]]
    cx   = np.mean([p.x for p in palm])
    cy   = np.mean([p.y for p in palm])
    cz   = np.mean([p.z for p in palm])
    tips  = [4,8,12,16,20]
    dists = [np.sqrt((lm[t].x-cx)**2+(lm[t].y-cy)**2+(lm[t].z-cz)**2) for t in tips]
    cross = [np.sqrt((lm[4].x-lm[t].x)**2+(lm[4].y-lm[t].y)**2+(lm[4].z-lm[t].z)**2) for t in [8,12,16]]
    return np.array(dists+cross, dtype=np.float32)

def _thumb_vec(h):
    lm = h.landmark
    tt,im,pt,pm,w = lm[4],lm[5],lm[20],lm[17],lm[0]
    return np.array([tt.x-im.x,tt.y-im.y,tt.z-im.z,pm.y-pt.y,
                     np.sqrt((tt.x-w.x)**2+(tt.y-w.y)**2+(tt.z-w.z)**2)], dtype=np.float32)

def _pinch_vec(h):
    lm    = h.landmark
    gaps  = [np.sqrt((lm[4].x-lm[t].x)**2+(lm[4].y-lm[t].y)**2+(lm[4].z-lm[t].z)**2) for t in [8,12,16,20]]
    curv  = [lm[6].y-lm[8].y, lm[10].y-lm[12].y, lm[14].y-lm[16].y]
    return np.array(gaps+curv, dtype=np.float32)

def get_legacy_feature_vector(h):
    return np.concatenate([_landmarks_vec(h), _angle_vec(h),
                           _dist_vec(h), _thumb_vec(h), _pinch_vec(h)])


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
WIN_W, WIN_H = 1280, 720

TOP_H    = 60
CAM_H    = 510
RESULT_H = 90
BTN_H    = 60
CAM_W    = WIN_W // 2

TOP_Y1, TOP_Y2  = 0,           TOP_H
CAM_Y1, CAM_Y2  = TOP_H,       TOP_H + CAM_H
RES_Y1, RES_Y2  = CAM_Y2,      CAM_Y2 + RESULT_H
BTN_Y1, BTN_Y2  = RES_Y2,      WIN_H

B1X1, B1X2 =    0,  320
B2X1, B2X2 =  320,  640
B3X1, B3X2 =  640,  960
B4X1, B4X2 =  960, 1280
BPD         =    8

CONF_GOOD    = 0.80    # green threshold, allow confirm
CONF_MEDIUM  = 0.60    # yellow threshold, warn if below
SMOOTH_N     = 7       # frames in voting buffer
VOTE_THRESH  = 5       # need 5 out of 7 frames to agree
IMG_SIZE     = 128
HAND_MARGIN  = 30

FONT = cv2.FONT_HERSHEY_DUPLEX


class State:
    IDLE     = "idle"
    CAPTURED = "captured"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_model_and_classes():
    """
    Try to load the advanced fusion model first, fall back to old model.
    """
    model_dir = Path("model")

    # 1. Try fusion model
    fusion_path = model_dir / "asl_fusion_model.h5"
    if fusion_path.exists():
        try:
            model = tf.keras.models.load_model(str(fusion_path), compile=False)
            classes_path = model_dir / "label_classes_advanced.npy"
            if not classes_path.exists():
                classes_path = model_dir / "label_classes.npy"
            classes = np.load(str(classes_path)).tolist()
            print(f"  [OK] Loaded FUSION model: {fusion_path}")
            return model, classes, True, 41
        except Exception as e:
            print(f"  [WARN] Fusion model load failed: {e}")

    # 2. Fall back to landmark-only advanced model
    lm_path = model_dir / "asl_landmark_advanced.h5"
    if lm_path.exists():
        try:
            model = tf.keras.models.load_model(str(lm_path), compile=False)
            classes = np.load(str(model_dir / "label_classes_advanced.npy")).tolist()
            print(f"  [OK] Loaded landmark-only advanced model")
            return model, classes, False, 41
        except Exception as e:
            print(f"  [WARN]: {e}")

    # 3. Fall back to original model (rebuild architecture)
    print("  [COMPAT] No advanced model found — loading original model...")
    log = model_dir / "model_log.txt"
    best_acc, best_file = 0, None
    if log.exists():
        for line in open(log):
            parts = line.strip().split("|")
            if len(parts) < 6: continue
            try:
                acc = float(parts[1].split(":")[1].strip().replace("%", ""))
                fname = parts[5].split(":")[1].strip().split()[0]
                if acc > best_acc:
                    best_acc, best_file = acc, fname
            except Exception:
                continue

    mp_ = model_dir / best_file if best_file else None
    if not mp_ or not mp_.exists():
        mp_ = model_dir / "asl_dense_model_BEST.h5"
    if not mp_.exists():
        mp_ = Path("asl_mediapipe_dense.h5")
    if not mp_.exists():
        print("[ERROR] No model file found!"); sys.exit(1)

    # Rebuild the original architecture
    nc = 28
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(93,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(nc, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    try:
        model.load_weights(str(mp_), by_name=True, skip_mismatch=True)
    except Exception:
        pass

    cp = model_dir / "label_classes.npy"
    classes = np.load(str(cp)).tolist() if cp.exists() else \
              list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["delete", "space"]
    print(f"  [COMPAT] Using original model with 93-dim features")
    return model, classes, False, 93


# ══════════════════════════════════════════════════════════════════════════════
# HAND QUALITY SCORE
# ══════════════════════════════════════════════════════════════════════════════
def hand_quality(hand_lm, frame_shape):
    """
    Returns ("GOOD"/"PARTIAL"/"POOR", description string).
    Checks:
      - Hand fully within frame (at least 2% away from boundary)
      - Knuckles show stereoscopic distribution
    """
    H, W = frame_shape[:2]
    xs = [lm.x for lm in hand_lm.landmark]
    ys = [lm.y for lm in hand_lm.landmark]
    zs = [lm.z for lm in hand_lm.landmark]

    out_of_frame = any(x < 0.02 or x > 0.98 or y < 0.02 or y > 0.98
                       for x, y in zip(xs, ys))
    z_range = max(zs) - min(zs)

    if out_of_frame:
        return "POOR", "Hand out of frame boundary"
    if z_range < 0.015:
        return "PARTIAL", "Keep hand facing camera angle"
    return "GOOD", "Clear view"


# ══════════════════════════════════════════════════════════════════════════════
# CROP HAND FROM FRAME
# ══════════════════════════════════════════════════════════════════════════════
def crop_hand(frame, hand_lm, margin=HAND_MARGIN, size=IMG_SIZE):
    H, W = frame.shape[:2]
    xs = [lm.x * W for lm in hand_lm.landmark]
    ys = [lm.y * H for lm in hand_lm.landmark]
    x1 = max(0, int(min(xs)) - margin)
    y1 = max(0, int(min(ys)) - margin)
    x2 = min(W, int(max(xs)) + margin)
    y2 = min(H, int(max(ys)) + margin)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size))


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def rect_alpha(img, x1, y1, x2, y2, color=(0,0,0), alpha=0.55):
    x1,y1,x2,y2 = max(0,x1),max(0,y1),min(img.shape[1],x2),min(img.shape[0],y2)
    sub = img[y1:y2,x1:x2]
    if sub.size == 0: return
    ov = np.full(sub.shape, color, dtype=np.uint8)
    cv2.addWeighted(ov, alpha, sub, 1-alpha, 0, sub)
    img[y1:y2,x1:x2] = sub

def put_text(img, text, x, y, scale=0.7, color=(255,255,255), thickness=1):
    cv2.putText(img, text, (x,y), FONT, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x,y), FONT, scale, color,   thickness,   cv2.LINE_AA)

def centered_text(img, text, cx, y, scale=0.7, color=(255,255,255), thickness=1):
    (tw,_),_ = cv2.getTextSize(text, FONT, scale, thickness)
    put_text(img, text, cx-tw//2, y, scale, color, thickness)


def draw_top_bar(c, word_buf, sentence, state):
    rect_alpha(c, 0, TOP_Y1, WIN_W, TOP_Y2, color=(10,10,20), alpha=0.92)
    wd = "  ".join(list(word_buf)) if word_buf else ""
    put_text(c, f"Word:  [ {wd} ]", 18, 38, scale=0.82, color=(255,230,50))
    put_text(c, f"Sentence:  {sentence[-80:]}", 18, 57, scale=0.52, color=(60,220,60))
    badge_col = {State.IDLE:(70,70,70), State.CAPTURED:(0,180,70)}.get(state,(70,70,70))
    badge_txt = state.upper()
    (tw,th),_ = cv2.getTextSize(badge_txt, FONT, 0.52, 1)
    bx = WIN_W - tw - 22
    rect_alpha(c, bx-6, 8, bx+tw+6, 8+th+10, color=badge_col, alpha=0.85)
    put_text(c, badge_txt, bx, 8+th+2, scale=0.52)


def draw_left_panel(c, live_disp, quality, quality_msg, vote_label, camera_active):
    """Left 640×510 — live camera + hand quality + temporal smoothing result."""
    if live_disp is not None and camera_active:
        resized = cv2.resize(live_disp, (CAM_W, CAM_H))
        c[CAM_Y1:CAM_Y2, 0:CAM_W] = resized
    else:
        c[CAM_Y1:CAM_Y2, 0:CAM_W] = (20,20,30)
        if not camera_active:
            centered_text(c, "CAMERA STOPPED", CAM_W//2, (CAM_Y1+CAM_Y2)//2,
                          scale=1.0, color=(80,80,220))

    # Header
    rect_alpha(c, 0, CAM_Y1, CAM_W, CAM_Y1+36, color=(0,0,0), alpha=0.6)
    put_text(c, "  LIVE CAMERA", 8, CAM_Y1+25, scale=0.62, color=(100,210,255))

    # Hand quality badge
    q_col = {"GOOD":(0,220,80), "PARTIAL":(220,180,0), "POOR":(0,60,220)}.get(quality,(80,80,80))
    rect_alpha(c, 0, CAM_Y2-38, CAM_W, CAM_Y2, color=(0,0,0), alpha=0.55)
    put_text(c, f"Hand Quality: {quality}  ({quality_msg})", 10, CAM_Y2-12,
             scale=0.55, color=q_col)

    # Temporal smoothing vote display
    if vote_label:
        put_text(c, f"Seeing: {vote_label} (STABLE)", 10, CAM_Y2-30,
                 scale=0.55, color=(180,255,180))
    else:
        put_text(c, "Seeing: --- (Hold sign stable)", 10, CAM_Y2-30,
                 scale=0.55, color=(160,160,160))

    cv2.line(c, (CAM_W, CAM_Y1), (CAM_W, CAM_Y2), (60,60,90), 2)


def draw_right_panel(c, snap_disp, snap_label, top3, state):
    """Right 640×510 — captured photo + top-3 predictions."""
    xo = CAM_W
    if snap_disp is not None:
        resized = cv2.resize(snap_disp, (CAM_W, CAM_H))
        c[CAM_Y1:CAM_Y2, xo:xo+CAM_W] = resized
        rect_alpha(c, xo, CAM_Y1, xo+CAM_W, CAM_Y1+36, color=(0,0,0), alpha=0.6)
        put_text(c, "  CAPTURED PHOTO", xo+8, CAM_Y1+25, scale=0.62,
                 color=(255,200,60))

        # Always Show Top-3 predictions on right panel with color coding
        if top3:
            rect_alpha(c, xo, CAM_Y2-130, xo+CAM_W, CAM_Y2, color=(0,0,0), alpha=0.7)
            # Rank colors matching calibration: green (>=80%), yellow (60-80%), gray (<60%)
            for rank, (lbl, conf) in enumerate(top3):
                pct  = int(conf*100)
                if conf >= CONF_GOOD:
                    col = (80, 255, 80)     # green
                elif conf >= CONF_MEDIUM:
                    col = (80, 220, 255)    # yellow
                else:
                    col = (160, 160, 160)   # gray

                tx   = xo + 14
                ty   = CAM_Y2 - 105 + rank * 38
                put_text(c, f"Line {rank+1}: {rank+1}st: {lbl}  {pct}%" if rank==0 else
                            f"Line {rank+1}: {rank+1}nd: {lbl}  {pct}%" if rank==1 else
                            f"Line {rank+1}: {rank+1}rd: {lbl}  {pct}%",
                         tx, ty, scale=0.58, color=col, thickness=1 if rank>0 else 2)
                # Bar graph representation
                bw = int((pct / 100) * 320)
                cv2.rectangle(c, (tx+220, ty-18), (tx+220+bw, ty-4), col, -1)
    else:
        c[CAM_Y1:CAM_Y2, xo:xo+CAM_W] = (15,15,25)
        rect_alpha(c, xo, CAM_Y1, xo+CAM_W, CAM_Y1+36, color=(0,0,0), alpha=0.6)
        put_text(c, "  CAPTURED PHOTO", xo+8, CAM_Y1+25, scale=0.62,
                 color=(110,110,130))
        lines = ["Show your hand on the LEFT camera,",
                 "ensure Hand Quality is GOOD,",
                 "then press  C  or click CAPTURE"]
        cy = (CAM_Y1+CAM_Y2)//2 - 30
        for ln in lines:
            centered_text(c, ln, xo+CAM_W//2, cy, scale=0.58, color=(85,85,105))
            cy += 36


def draw_result_bar(c, state, top3, snap_conf):
    bar_color = (8,30,8) if state==State.CAPTURED else (12,12,20)
    rect_alpha(c, 0, RES_Y1, WIN_W, RES_Y2, color=bar_color, alpha=0.96)
    cv2.line(c, (0,RES_Y1), (WIN_W,RES_Y1), (60,60,80), 1)
    cv2.line(c, (0,RES_Y2), (WIN_W,RES_Y2), (60,60,80), 1)
    mid_y = (RES_Y1+RES_Y2)//2

    if state==State.CAPTURED and top3:
        lbl, conf = top3[0]
        pct       = int(conf*100)
        is_good   = conf >= CONF_GOOD
        is_medium = conf >= CONF_MEDIUM

        # Confidence Calibration color-coded labels
        if is_good:
            col = (80, 255, 80)     # green
            hint = "✓ Press SPACE / Click CONFIRM to write."
            hint_col = (180,255,180)
        elif is_medium:
            col = (80, 220, 255)    # yellow
            hint = "⚠ Uncertain (60-80%) - try repositioning hand. [Confirm disabled]"
            hint_col = (80, 220, 255)
        else:
            col = (160, 160, 160)   # gray
            hint = "unclear - reposition hand (below 60%). [Confirm disabled]"
            hint_col = (160, 160, 160)

        # Big letter display
        (lw,lh),_ = cv2.getTextSize(lbl, FONT, 2.4, 3)
        put_text(c, lbl, 60, mid_y+lh//2, scale=2.4, color=col, thickness=3)
        put_text(c, "—", 60+lw+18, mid_y+lh//2-8, scale=1.2, color=(150,150,150))
        put_text(c, f"{pct}% confidence", 60+lw+70, mid_y+lh//2-6,
                 scale=1.0, color=col)

        centered_text(c, hint, WIN_W-360, mid_y+15, scale=0.55, color=hint_col)

    else:
        centered_text(c, "Align hand (GOOD quality) and press  C  to analyze sign.",
                      WIN_W//2, mid_y+10, scale=0.68, color=(90,90,110))


def draw_button_bar(c, state, camera_active, top3):
    rect_alpha(c, 0, BTN_Y1, WIN_W, BTN_Y2, color=(8,8,14), alpha=0.95)
    confirm_ready = (state==State.CAPTURED and top3 and
                     top3[0][0] not in ("-","nothing") and
                     top3[0][1] >= CONF_GOOD)

    def btn(x1, x2, bg, label, lbl_col=(255,255,255)):
        iy1,iy2 = BTN_Y1+BPD, BTN_Y2-BPD
        cv2.rectangle(c, (x1+BPD,iy1), (x2-BPD,iy2), bg, -1)
        cv2.rectangle(c, (x1+BPD,iy1), (x2-BPD,iy2), (200,200,200), 1)
        mid = (x1+x2)//2
        (tw,th),_ = cv2.getTextSize(label, FONT, 0.72, 2)
        ty = iy1 + (iy2-iy1+th)//2
        put_text(c, label, mid-tw//2, ty, scale=0.72, color=lbl_col, thickness=2)

    btn(B1X1, B1X2, (20,130,170), "CAPTURE  [C]")
    btn(B2X1, B2X2,
        (0,160,70) if confirm_ready else (20,55,30),
        "CONFIRM  [SPACE]",
        (255,255,255) if confirm_ready else (80,110,80))
    btn(B3X1, B3X2,
        (180,20,20) if state==State.CAPTURED else (55,20,20),
        "DELETE  [BKSP]",
        (255,255,255) if state==State.CAPTURED else (110,80,80))
    btn(B4X1, B4X2,
        (30,30,190) if camera_active else (0,150,60),
        "STOP CAM" if camera_active else "START CAM")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_inference_on_frame(frame, hand_lm, model, classes, is_fusion, feature_dim=41):
    """
    Run model prediction on one frame.
    """
    if feature_dim == 93:
        feat = get_legacy_feature_vector(hand_lm)
    else:
        feat = extract_advanced_features(hand_lm.landmark)

    if feat.shape[0] != feature_dim:
        return []

    if is_fusion:
        img_crop = crop_hand(frame, hand_lm).astype(np.float32) / 255.0
        inp_feat = np.expand_dims(feat, 0)
        inp_img  = np.expand_dims(img_crop, 0)
        preds    = model.predict([inp_feat, inp_img], verbose=0)[0]
    else:
        preds = model.predict(np.expand_dims(feat, 0), verbose=0)[0]

    top_idx = np.argsort(preds)[::-1][:3]
    return [(classes[min(i, len(classes)-1)], float(preds[i])) for i in top_idx]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("  ASL Advanced App — Upgraded Fusion Core")
    print("="*60)

    print("\nLoading model...")
    model, classes, is_fusion, feature_dim = load_model_and_classes()
    print(f"  Architecture: {'DUAL-INPUT FUSION (128x128 crop + 41-dim)' if is_fusion else 'Landmark-only'}")
    print(f"  Feature dimensions: {feature_dim}")
    print(f"  Classes: {classes}")

    print("\nInitialising MediaPipe...")
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    hands_live = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                min_detection_confidence=0.6,
                                min_tracking_confidence=0.6)
    hands_snap = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                                min_detection_confidence=0.5)

    # Open camera
    cap = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(1)
        for idx in [0,1,2]:
            for backend in [cv2.CAP_DSHOW, None]:
                try:
                    tc = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
                    if tc.isOpened():
                        ret, fr = tc.read()
                        if ret and fr is not None:
                            cap = tc
                            print(f"  [OK] Camera opened successfully at index {idx}")
                            break
                        tc.release()
                except Exception:
                    pass
            if cap: break
        if cap: break

    if cap is None:
        print("[ERROR] No webcam found!"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # State variables
    state         = State.IDLE
    camera_active = True
    last_clean    = None
    last_display  = None

    snap_display  = None
    snap_top3     = None
    snap_conf     = 0.0
    snap_label    = None

    word_buf    = []
    sentence    = ""

    # Temporal smoothing buffer
    pred_buffer  = collections.deque(maxlen=SMOOTH_N)
    quality      = "POOR"
    quality_msg  = "No hand"
    vote_label   = None

    # Callbacks
    def do_capture():
        nonlocal snap_display, snap_top3, snap_conf, snap_label, state
        if last_clean is None:
            return
        if quality != "GOOD":
            print(f"[CAPTURE REJECTED] Cannot capture. Hand Quality is: {quality}")
            return

        print("[CAPTURE] Running inference on snapshot...")
        ann = last_clean.copy()
        res = hands_snap.process(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            print("[RESULT] No hand detected in snapshot")
            snap_display = ann
            snap_top3    = None
            snap_conf    = 0.0
            snap_label   = None
            state        = State.CAPTURED
            return

        hlm = res.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(ann, hlm, mp_hands.HAND_CONNECTIONS,
                                  mp_styles.get_default_hand_landmarks_style(),
                                  mp_styles.get_default_hand_connections_style())
        top3 = run_inference_on_frame(last_clean, hlm, model, classes, is_fusion, feature_dim)
        snap_display = ann
        snap_top3    = top3
        snap_label   = top3[0][0] if top3 else None
        snap_conf    = top3[0][1] if top3 else 0.0
        state        = State.CAPTURED
        if top3:
            print(f"[RESULT] {top3[0][0]} ({int(top3[0][1]*100)}%) | "
                  f"2nd: {top3[1][0]} ({int(top3[1][1]*100)}%) | "
                  f"3rd: {top3[2][0]} ({int(top3[2][1]*100)}%)")

    def do_confirm():
        nonlocal state, snap_display, snap_top3, snap_label, snap_conf
        nonlocal word_buf, sentence
        # ONLY allow confirm when confidence > 80%
        if not snap_label or snap_label in ("nothing","-") or snap_conf < CONF_GOOD:
            return
        letter = snap_label
        if letter == "space":
            sentence += "".join(word_buf) + " "; word_buf.clear()
        elif letter == "delete":
            if word_buf: word_buf.pop()
        else:
            word_buf.append(letter)
        print(f"[CONFIRM] '{letter}' | Built text: {''.join(word_buf)}")
        snap_display = snap_top3 = snap_label = None
        snap_conf = 0.0; state = State.IDLE

    def do_delete():
        nonlocal state, snap_display, snap_top3, snap_label, snap_conf
        snap_display = snap_top3 = snap_label = None
        snap_conf = 0.0; state = State.IDLE
        print("[DELETE] Snapshot discarded")

    def mouse_cb(event, x, y, flags, param):
        nonlocal camera_active
        if event != cv2.EVENT_LBUTTONDOWN: return
        if BTN_Y1+BPD <= y <= BTN_Y2-BPD:
            if   B1X1+BPD <= x <= B1X2-BPD:
                if camera_active: do_capture()
            elif B2X1+BPD <= x <= B2X2-BPD:
                if state==State.CAPTURED: do_confirm()
            elif B3X1+BPD <= x <= B3X2-BPD:
                if state==State.CAPTURED: do_delete()
            elif B4X1+BPD <= x <= B4X2-BPD:
                camera_active = not camera_active

    cv2.namedWindow("ASL Advanced", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ASL Advanced", WIN_W, WIN_H)
    cv2.setMouseCallback("ASL Advanced", mouse_cb)

    # Main inference loop
    while True:
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

        if camera_active:
            ret, raw = cap.read()
            if not ret: break
            raw       = cv2.flip(raw, 1)
            last_clean = raw.copy()  # clean snapshot frame

            rgb  = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            res  = hands_live.process(rgb)

            quality, quality_msg = "POOR", "No hand detected"

            if res.multi_hand_landmarks:
                hlm = res.multi_hand_landmarks[0]
                quality, quality_msg = hand_quality(hlm, raw.shape)

                # Draw landmarks skeleton on live feed
                mp_drawing.draw_landmarks(
                    raw, hlm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                # ONLY run inference & temporal smoothing when hand quality is GOOD
                if quality == "GOOD":
                    top3_live = run_inference_on_frame(raw, hlm, model, classes, is_fusion, feature_dim)
                    if top3_live:
                        pred_buffer.append(top3_live[0][0])
                else:
                    pred_buffer.clear()

                # Calculate majority vote
                if len(pred_buffer) >= SMOOTH_N:
                    counter = collections.Counter(pred_buffer)
                    top, cnt = counter.most_common(1)[0]
                    # Stabilized prediction needs 5/7 frames
                    vote_label = top if cnt >= VOTE_THRESH else None
                else:
                    vote_label = None
            else:
                pred_buffer.clear()
                vote_label = None

            last_display = raw.copy()

        # Draw interface panels
        draw_top_bar(canvas, word_buf, sentence, state)
        draw_left_panel(canvas, last_display if camera_active else None,
                        quality, quality_msg, vote_label, camera_active)
        draw_right_panel(canvas, snap_display, snap_label, snap_top3, state)
        draw_result_bar(canvas, state, snap_top3, snap_conf)
        draw_button_bar(canvas, state, camera_active, snap_top3)

        cv2.imshow("ASL Advanced", canvas)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord("q") or key == ord("Q"): break
        elif key in (ord("c"),ord("C")):
            if camera_active: do_capture()
        elif key == ord(" "):
            if state==State.CAPTURED: do_confirm()
        elif key == 8:   # BACKSPACE
            if state==State.CAPTURED: do_delete()
            elif word_buf: word_buf.pop()
        elif key == 13:  # ENTER
            if word_buf:
                sentence += "".join(word_buf) + " "; word_buf.clear()
            snap_display = snap_top3 = snap_label = None
            snap_conf = 0.0; state = State.IDLE
        elif key == 27:  # ESC
            sentence = ""; word_buf.clear()
            snap_display = snap_top3 = snap_label = None
            snap_conf = 0.0; state = State.IDLE

    cap.release()
    cv2.destroyAllWindows()
    hands_live.close(); hands_snap.close()

    final = (sentence + "".join(word_buf)).strip()
    if final: print(f"\nFinal sentence: {final}")
    print("Goodbye!")


if __name__ == "__main__":
    main()
