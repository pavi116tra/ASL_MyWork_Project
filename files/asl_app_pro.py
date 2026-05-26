"""
ASL Sentence Builder - Split Screen Edition
============================================
Layout (1280 x 720):
  TOP BAR   (60px)  : Word buffer + Sentence display
  PANELS   (530px)  : LEFT = live camera  |  RIGHT = captured snapshot
  RESULT   ( 70px)  : Detected letter + accuracy  →  "A — 99%  Confirm?"
  BUTTONS  ( 60px)  : [CAPTURE C] [CONFIRM ✓] [DELETE ✗] [STOP CAM]

UX Flow:
  1. Show hand on the LEFT live panel.
  2. Press C / click CAPTURE  →  photo taken and shown on RIGHT.
  3. Model detects the letter  →  result shown in the RESULT BAR below.
     e.g.  "Detected: A  —  99% accuracy  |  ✓ CONFIRM   ✗ DELETE"
  4. Click CONFIRM or press SPACE  →  letter added to the word.
  5. Click DELETE  or press BKSP   →  snapshot discarded, try again.
  6. Press ENTER to finalise the word into the sentence.
"""

import os, sys, time, collections
import cv2
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"

import tensorflow as tf

# ── MediaPipe protobuf fix ────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
WIN_W, WIN_H = 1280, 720

TOP_H    = 60          # word / sentence info bar
CAM_H    = 530         # height of both camera panels
RESULT_H = 70          # result display bar
BTN_H    = 60          # bottom button row
CAM_W    = WIN_W // 2  # each panel = 640 px wide

# y-ranges
TOP_Y1, TOP_Y2    = 0,             TOP_H
CAM_Y1, CAM_Y2   = TOP_H,         TOP_H + CAM_H
RES_Y1, RES_Y2   = CAM_Y2,        CAM_Y2 + RESULT_H
BTN_Y1, BTN_Y2   = RES_Y2,        WIN_H

# Button x-ranges (4 equal sections of 320 px each)
B1X1, B1X2 =    0,  320   # CAPTURE
B2X1, B2X2 =  320,  640   # CONFIRM
B3X1, B3X2 =  640,  960   # DELETE
B4X1, B4X2 =  960, 1280   # STOP CAM
BPD         =    8         # inner padding for buttons

CONF_THRESHOLD = 0.70
MODEL_DIR      = Path("model")

FONT = cv2.FONT_HERSHEY_DUPLEX


class State:
    IDLE     = "idle"
    CAPTURED = "captured"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (93-dim)
# ══════════════════════════════════════════════════════════════════════════════
def _landmarks_vec(h, _):
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

def get_feature_vector(h, shape):
    return np.concatenate([_landmarks_vec(h,shape), _angle_vec(h),
                           _dist_vec(h), _thumb_vec(h), _pinch_vec(h)])

def apply_rules(label, h):
    lm  = h.landmark
    pex = (lm[18].y - lm[20].y) > 0.04
    tig = np.sqrt((lm[4].x-lm[8].x)**2+(lm[4].y-lm[8].y)**2+(lm[4].z-lm[8].z)**2)
    if label == "I" and not pex: label = "A"
    if label == "A" and pex:     label = "I"
    if label == "O" and tig > 0.12: label = "C"
    if label == "C" and tig < 0.07: label = "O"
    return label


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def safe_load_model(path):
    try:
        return tf.keras.models.load_model(str(path), compile=False)
    except Exception:
        pass
    print("  [compat] Rebuilding architecture...")
    import h5py
    with h5py.File(str(path), "r") as f:
        kernels = []
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and "kernel" in name:
                kernels.append((name, obj.shape))
        f.visititems(visit)
        kd = [(n,s) for n,s in kernels if len(s)==2]
        nc = min(kd, key=lambda x: x[1][-1])[1][-1] if kd else 28
    print(f"  [compat] Output classes: {nc}")
    m = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(93,)),
        tf.keras.layers.Dense(256, activation="relu", name="dense"),
        tf.keras.layers.BatchNormalization(name="batch_normalization"),
        tf.keras.layers.Dropout(0.4, name="dropout"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.BatchNormalization(name="batch_normalization_1"),
        tf.keras.layers.Dropout(0.3, name="dropout_1"),
        tf.keras.layers.Dense(64,  activation="relu", name="dense_2"),
        tf.keras.layers.Dense(nc,  activation="softmax", name="dense_3"),
    ], name="sequential")
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    try:
        m.load_weights(str(path), by_name=True, skip_mismatch=True)
        print("  [compat] Weights loaded OK")
    except Exception as e:
        print(f"  [compat] Warning: {e}")
    return m

def load_model_and_classes():
    log = MODEL_DIR / "model_log.txt"
    best_acc, best_file = 0, None
    if log.exists():
        for line in open(log):
            parts = line.strip().split("|")
            if len(parts) < 6: continue
            try:
                acc   = float(parts[1].split(":")[1].strip().replace("%",""))
                fname = parts[5].split(":")[1].strip()
                if acc > best_acc: best_acc, best_file = acc, fname
            except Exception: continue
    mp_ = MODEL_DIR / best_file if best_file else None
    if not mp_ or not mp_.exists(): mp_ = MODEL_DIR / "asl_dense_model_BEST.h5"
    if not mp_.exists():            mp_ = Path("asl_mediapipe_dense.h5")
    if not mp_.exists():
        print("[ERROR] No model found!"); sys.exit(1)
    print(f"  Loading: {mp_} ({best_acc:.2f}% acc)")
    model   = safe_load_model(mp_)
    cp      = MODEL_DIR / "label_classes.npy"
    classes = np.load(str(cp)).tolist() if cp.exists() else \
              list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["delete","nothing","space"]
    return model, classes


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def rect_alpha(img, x1, y1, x2, y2, color=(0,0,0), alpha=0.55):
    x1,y1,x2,y2 = max(0,x1),max(0,y1),min(img.shape[1],x2),min(img.shape[0],y2)
    sub = img[y1:y2, x1:x2]
    if sub.size == 0: return
    ov = np.full(sub.shape, color, dtype=np.uint8)
    cv2.addWeighted(ov, alpha, sub, 1-alpha, 0, sub)
    img[y1:y2, x1:x2] = sub

def put_text(img, text, x, y, scale=0.7, color=(255,255,255), thickness=1):
    cv2.putText(img, text, (x, y), FONT, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color,   thickness,   cv2.LINE_AA)

def centered_text(img, text, cx, y, scale=0.7, color=(255,255,255), thickness=1):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    put_text(img, text, cx - tw//2, y, scale, color, thickness)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL DRAWING
# ══════════════════════════════════════════════════════════════════════════════
def draw_top_bar(c, word_buf, sentence, state):
    """60px top bar: Word + Sentence + state badge."""
    rect_alpha(c, 0, TOP_Y1, WIN_W, TOP_Y2, color=(10,10,20), alpha=0.92)
    wd = "  ".join(list(word_buf)) if word_buf else ""
    put_text(c, f"Word:  [ {wd} ]", 18, 38, scale=0.82, color=(255,230,50))
    sd = sentence[-100:] if len(sentence) > 100 else sentence
    put_text(c, f"Sentence:  {sd}", 18, 57, scale=0.55, color=(60,220,60))
    # Badge
    badge_col = {State.IDLE:(70,70,70), State.CAPTURED:(0,180,70)}.get(state,(70,70,70))
    badge_txt = state.upper()
    (tw,th),_ = cv2.getTextSize(badge_txt, FONT, 0.52, 1)
    bx = WIN_W - tw - 22
    rect_alpha(c, bx-6, 8, bx+tw+6, 8+th+10, color=badge_col, alpha=0.85)
    put_text(c, badge_txt, bx, 8+th+2, scale=0.52, color=(255,255,255))

def draw_left_panel(c, live_disp, camera_active):
    """LEFT 640x530 — live camera with skeleton only. No accuracy shown."""
    if live_disp is not None and camera_active:
        resized = cv2.resize(live_disp, (CAM_W, CAM_H))
        c[CAM_Y1:CAM_Y2, 0:CAM_W] = resized
    else:
        c[CAM_Y1:CAM_Y2, 0:CAM_W] = (20,20,30)
        if not camera_active:
            centered_text(c, "CAMERA STOPPED", CAM_W//2, (CAM_Y1+CAM_Y2)//2,
                          scale=1.0, color=(80,80,220))

    # Header label
    rect_alpha(c, 0, CAM_Y1, CAM_W, CAM_Y1+36, color=(0,0,0), alpha=0.6)
    put_text(c, "  LIVE CAMERA", 8, CAM_Y1+25, scale=0.62, color=(100,210,255))

    # Bottom hint: prompt user to press C
    rect_alpha(c, 0, CAM_Y2-36, CAM_W, CAM_Y2, color=(0,0,0), alpha=0.55)
    put_text(c, "Show your hand sign, then press  C  to capture",
             10, CAM_Y2-11, scale=0.52, color=(180,180,180))

    # Divider
    cv2.line(c, (CAM_W, CAM_Y1), (CAM_W, CAM_Y2), (60,60,90), 2)

def draw_right_panel(c, snap_disp, state):
    """RIGHT 640×530 — captured snapshot."""
    xo = CAM_W
    if snap_disp is not None:
        resized = cv2.resize(snap_disp, (CAM_W, CAM_H))
        c[CAM_Y1:CAM_Y2, xo:xo+CAM_W] = resized
        rect_alpha(c, xo, CAM_Y1, xo+CAM_W, CAM_Y1+36, color=(0,0,0), alpha=0.6)
        put_text(c, "  CAPTURED PHOTO", xo+8, CAM_Y1+25, scale=0.62, color=(255,200,60))
    else:
        c[CAM_Y1:CAM_Y2, xo:xo+CAM_W] = (15,15,25)
        rect_alpha(c, xo, CAM_Y1, xo+CAM_W, CAM_Y1+36, color=(0,0,0), alpha=0.6)
        put_text(c, "  CAPTURED PHOTO", xo+8, CAM_Y1+25, scale=0.62, color=(110,110,130))
        # Placeholder
        lines = ["Show your hand on the LEFT camera,",
                 "then press  C  or click  CAPTURE",
                 "Your snapshot will appear here"]
        cy = (CAM_Y1+CAM_Y2)//2 - 30
        for ln in lines:
            centered_text(c, ln, xo+CAM_W//2, cy, scale=0.58, color=(85,85,105))
            cy += 36
        cv2.circle(c, (xo+CAM_W//2, CAM_Y1+80), 38, (45,45,65), -1)
        cv2.circle(c, (xo+CAM_W//2, CAM_Y1+80), 38, (75,75,95), 2)
        cv2.circle(c, (xo+CAM_W//2, CAM_Y1+80), 18, (75,75,95), 2)

def draw_result_bar(c, state, snap_label, snap_conf):
    """70px result bar spanning full width below both panels."""
    # Background
    bar_color = (8,30,8) if state == State.CAPTURED else (12,12,20)
    rect_alpha(c, 0, RES_Y1, WIN_W, RES_Y2, color=bar_color, alpha=0.96)
    cv2.line(c, (0, RES_Y1), (WIN_W, RES_Y1), (60,60,80), 1)
    cv2.line(c, (0, RES_Y2), (WIN_W, RES_Y2), (60,60,80), 1)

    mid_y = (RES_Y1 + RES_Y2) // 2

    if state == State.CAPTURED and snap_label and snap_label not in ("-","nothing"):
        pct     = int(snap_conf * 100)
        is_good = snap_conf >= CONF_THRESHOLD

        # Letter — big and bold
        letter_txt = snap_label
        (lw, lh), _ = cv2.getTextSize(letter_txt, FONT, 2.2, 3)
        lx = 60
        put_text(c, letter_txt, lx, mid_y + lh//2,
                 scale=2.2, color=(255,255,80), thickness=3)

        # Separator dash
        dash_x = lx + lw + 20
        put_text(c, "—", dash_x, mid_y + lh//2 - 6,
                 scale=1.2, color=(150,150,150))

        # Accuracy
        acc_col = (80,255,80) if is_good else (255,160,40)
        acc_txt = f"{pct}%  accuracy"
        acc_x   = dash_x + 55
        put_text(c, acc_txt, acc_x, mid_y + lh//2 - 4,
                 scale=1.0, color=acc_col, thickness=1)

        # Hint text
        if is_good:
            hint     = "Click  CONFIRM  to add  |  Click  DELETE  to discard"
            hint_col = (180, 255, 180)
        else:
            hint     = "Low confidence — try a clearer sign, then CAPTURE again"
            hint_col = (255, 200, 100)

        (hw, _), _ = cv2.getTextSize(hint, FONT, 0.52, 1)
        centered_text(c, hint, WIN_W - 340, mid_y + lh//2 - 4,
                      scale=0.52, color=hint_col)

    elif state == State.IDLE:
        centered_text(c, "Capture your hand sign  —  result will appear here",
                      WIN_W//2, mid_y + 10,
                      scale=0.68, color=(90,90,110))

def draw_button_bar(c, state, camera_active, snap_label, snap_conf):
    """60px button bar: CAPTURE | CONFIRM | DELETE | STOP CAM"""
    rect_alpha(c, 0, BTN_Y1, WIN_W, BTN_Y2, color=(8,8,14), alpha=0.95)

    confirm_ready = (state == State.CAPTURED
                     and snap_label
                     and snap_label not in ("-","nothing","delete","space")
                     and snap_conf >= CONF_THRESHOLD)

    # Inner button rect helper
    def btn(x1, x2, bg, label, label_col=(255,255,255)):
        inner_y1 = BTN_Y1 + BPD
        inner_y2 = BTN_Y2 - BPD
        cv2.rectangle(c, (x1+BPD, inner_y1), (x2-BPD, inner_y2), bg, -1)
        cv2.rectangle(c, (x1+BPD, inner_y1), (x2-BPD, inner_y2), (200,200,200), 1)
        mid = (x1+x2)//2
        (tw,th),_ = cv2.getTextSize(label, FONT, 0.72, 2)
        ty = inner_y1 + (inner_y2-inner_y1+th)//2
        put_text(c, label, mid-tw//2, ty, scale=0.72, color=label_col, thickness=2)

    # CAPTURE
    btn(B1X1, B1X2, (20,130,170),   "CAPTURE  [C]")
    # CONFIRM (bright green when ready, dim when not)
    btn(B2X1, B2X2,
        (0,160,70) if confirm_ready else (20,55,30),
        "CONFIRM  [SPACE]",
        (255,255,255) if confirm_ready else (80,110,80))
    # DELETE (red)
    btn(B3X1, B3X2,
        (180,20,20) if state==State.CAPTURED else (55,20,20),
        "DELETE  [BKSP]",
        (255,255,255) if state==State.CAPTURED else (110,80,80))
    # STOP / START CAM
    cam_bg    = (30,30,190) if camera_active else (0,150,60)
    cam_label = "STOP CAM" if camera_active else "START CAM"
    btn(B4X1, B4X2, cam_bg, cam_label)


# ══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(frame, hands_proc, model, classes, mp_hands, mp_drawing, mp_styles):
    """Return (annotated_frame, label_or_None, confidence)."""
    ann = frame.copy()
    res = hands_proc.process(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return ann, None, 0.0
    hlm = res.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(ann, hlm, mp_hands.HAND_CONNECTIONS,
                              mp_styles.get_default_hand_landmarks_style(),
                              mp_styles.get_default_hand_connections_style())
    feat = get_feature_vector(hlm, frame.shape)
    if feat.shape[0] != 93:
        return ann, None, 0.0
    preds = model.predict(np.expand_dims(feat,0), verbose=0)[0]
    idx   = min(int(np.argmax(preds)), len(classes)-1)
    label = apply_rules(classes[idx], hlm)
    return ann, label, float(preds[idx])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("  ASL Sentence Builder — Split Screen")
    print("="*60)

    print("\nLoading model...")
    model, classes = load_model_and_classes()
    print(f"  Classes: {classes}")

    print("\nInitialising MediaPipe...")
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    hands_live = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands_snap = mp_hands.Hands(static_image_mode=True,  max_num_hands=1,
                                min_detection_confidence=0.5)

    # ── Open webcam (robust: retry up to 3 times) ─────────────────────────────
    cap = None
    for attempt in range(3):
        if attempt > 0:
            print(f"  Retry {attempt}/2 — waiting 1s for camera to release...")
            time.sleep(1)
        for idx in [0, 1, 2]:
            # Try DSHOW first (most reliable on Windows), then default backend
            for backend in [cv2.CAP_DSHOW, None]:
                try:
                    tc = cv2.VideoCapture(idx, backend) if backend is not None \
                         else cv2.VideoCapture(idx)
                    if not tc.isOpened():
                        tc.release()
                        continue
                    # Warm-up: read 2 frames to confirm it actually works
                    ok = False
                    for _ in range(2):
                        ret, fr = tc.read()
                        if ret and fr is not None:
                            ok = True
                            break
                    if ok:
                        cap = tc
                        bname = "DSHOW" if backend else "default"
                        print(f"  [OK] Camera index {idx} ({bname})")
                        break
                    tc.release()
                except Exception as e:
                    print(f"  Camera {idx} error: {e}")
            if cap:
                break
        if cap:
            break

    if cap is None:
        print("[ERROR] No webcam found!"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ── App state ─────────────────────────────────────────────────────────────
    state         = State.IDLE
    camera_active = True

    # Live feed (skeleton only, no live inference)
    last_clean   = None   # clean frame (no skeleton) → used for snapshots
    last_display = None   # frame with skeleton drawn → shown on left panel

    # Snapshot
    snap_display = None   # annotated snapshot shown on right
    snap_label   = None
    snap_conf    = 0.0

    # Text buffers
    word_buf  = []
    sentence  = ""
    flash_end = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────
    def do_capture():
        nonlocal snap_display, snap_label, snap_conf, state
        if last_clean is None:
            print("[CAPTURE] No frame yet.")
            return
        print("[CAPTURE] Snapshot taken — running inference...")
        ann, lbl, conf = run_inference(last_clean, hands_snap, model, classes,
                                       mp_hands, mp_drawing, mp_styles)
        snap_display = ann
        snap_label   = lbl
        snap_conf    = conf
        state        = State.CAPTURED
        if lbl:
            print(f"[RESULT]  {lbl}  ({int(conf*100)}%)")
        else:
            print("[RESULT]  No hand detected — try again")

    def do_confirm():
        nonlocal state, snap_display, snap_label, snap_conf
        nonlocal word_buf, sentence, flash_end
        letter = snap_label
        if not letter or letter in ("nothing","-"):
            return
        if letter == "space":
            sentence += "".join(word_buf) + " "
            word_buf.clear()
        elif letter == "delete":
            if word_buf: word_buf.pop()
        else:
            word_buf.append(letter)
        print(f"[CONFIRM] Added '{letter}'  |  Word: {''.join(word_buf)}")
        snap_display = snap_label = None
        snap_conf    = 0.0
        state        = State.IDLE
        flash_end    = time.time() + 0.5

    def do_delete():
        nonlocal state, snap_display, snap_label, snap_conf
        print("[DELETE] Snapshot discarded")
        snap_display = snap_label = None
        snap_conf    = 0.0
        state        = State.IDLE

    # ── Mouse callback ────────────────────────────────────────────────────────
    def mouse_cb(event, x, y, flags, param):
        nonlocal camera_active
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if BTN_Y1+BPD <= y <= BTN_Y2-BPD:
            if B1X1+BPD <= x <= B1X2-BPD:        # CAPTURE
                if camera_active: do_capture()
            elif B2X1+BPD <= x <= B2X2-BPD:      # CONFIRM
                if state == State.CAPTURED and snap_label:
                    do_confirm()
            elif B3X1+BPD <= x <= B3X2-BPD:      # DELETE
                if state == State.CAPTURED:
                    do_delete()
            elif B4X1+BPD <= x <= B4X2-BPD:      # STOP/START CAM
                camera_active = not camera_active
                print("[CAM]", "Stopped" if not camera_active else "Started")

    cv2.namedWindow("ASL Sentence Builder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ASL Sentence Builder", WIN_W, WIN_H)
    cv2.setMouseCallback("ASL Sentence Builder", mouse_cb)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

        # Read + process live frame
        if camera_active:
            ret, raw = cap.read()
            if not ret: break
            raw        = cv2.flip(raw, 1)
            last_clean = raw.copy()   # CLEAN frame saved for snapshot inference

            # Draw hand skeleton on live feed (no model inference here)
            rgb_live = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            res_live = hands_live.process(rgb_live)
            if res_live.multi_hand_landmarks:
                hlm = res_live.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    raw, hlm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())
            last_display = raw.copy()   # WITH skeleton → shown on left panel

        # Draw all sections
        draw_top_bar(canvas, word_buf, sentence, state)
        draw_left_panel(canvas, last_display if camera_active else None, camera_active)
        draw_right_panel(canvas, snap_display, state)
        draw_result_bar(canvas, state, snap_label, snap_conf)
        draw_button_bar(canvas, state, camera_active, snap_label, snap_conf)

        cv2.imshow("ASL Sentence Builder", canvas)

        # Keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key in (ord("c"), ord("C")):
            if camera_active: do_capture()
        elif key == ord(" "):
            if state == State.CAPTURED and snap_label: do_confirm()
        elif key == 8:    # BACKSPACE → delete snapshot or last letter
            if state == State.CAPTURED:
                do_delete()
            elif word_buf:
                word_buf.pop()
        elif key == 13:   # ENTER → finalise word
            if word_buf:
                sentence += "".join(word_buf) + " "
                word_buf.clear()
            snap_display = snap_label = None
            snap_conf = 0.0
            state = State.IDLE
        elif key == 27:   # ESC → clear all
            sentence = ""; word_buf.clear()
            snap_display = snap_label = None
            snap_conf = 0.0; state = State.IDLE

    cap.release()
    cv2.destroyAllWindows()
    hands_live.close()
    hands_snap.close()

    final = (sentence + "".join(word_buf)).strip()
    if final: print(f"\nFinal: {final}")
    print("Goodbye!")


if __name__ == "__main__":
    main()
