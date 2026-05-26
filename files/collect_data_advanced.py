"""
collect_data_advanced.py
========================
Live data collection tool for ASL alphabet training data.

Features:
  - Live webcam with MediaPipe hand skeleton
  - Cycle through A-Z automatically or on demand
  - Auto-capture every 0.5s when hand confidence >= 0.8
  - Saves BOTH image (JPG) AND feature vector (NPY)
  - Progress bar: "A: 47/500 images"
  - Live preview of extracted features
  - Press SKIP (S) to move to next letter
  - Press SPACE to force capture
  - Session summary at end

Output structure:
  data/
  ├── images/{LETTER}/img_{timestamp}.jpg
  └── features/{LETTER}/feat_{timestamp}.npy
"""

import os
import sys
import time
import datetime
import collections
import cv2
import numpy as np
from pathlib import Path

# ── MediaPipe protobuf fix ────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
LETTERS          = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "delete"]
TARGET_PER_CLASS = 500          # images to collect per letter
CAPTURE_INTERVAL = 0.5          # seconds between auto-captures
MIN_CONFIDENCE   = 0.8          # minimum MediaPipe detection confidence
IMG_SIZE         = 128          # saved image size (square)
HAND_MARGIN      = 30           # pixels of padding around hand bounding box

WIN_W, WIN_H     = 1280, 720
FONT             = cv2.FONT_HERSHEY_DUPLEX

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

def draw_progress_bar(img, x, y, w, h, fraction,
                      bg=(40,40,40), fg=(0,200,100)):
    cv2.rectangle(img, (x,y), (x+w, y+h), bg, -1)
    filled = int(w * min(max(fraction, 0), 1))
    if filled > 0:
        cv2.rectangle(img, (x,y), (x+filled, y+h), fg, -1)
    cv2.rectangle(img, (x,y), (x+w, y+h), (150,150,150), 1)

def crop_hand(frame, hand_lm, margin=HAND_MARGIN):
    """Crop hand region from frame, return square 128×128 BGR image."""
    H, W = frame.shape[:2]
    xs = [lm.x * W for lm in hand_lm.landmark]
    ys = [lm.y * H for lm in hand_lm.landmark]
    x1 = max(0, int(min(xs)) - margin)
    y1 = max(0, int(min(ys)) - margin)
    x2 = min(W, int(max(xs)) + margin)
    y2 = min(H, int(max(ys)) + margin)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN COLLECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════
def collect():
    # ── Setup directories ─────────────────────────────────────────────────────
    data_root = Path("data")
    img_root  = data_root / "images"
    feat_root = data_root / "features"
    for letter in LETTERS:
        (img_root  / letter).mkdir(parents=True, exist_ok=True)
        (feat_root / letter).mkdir(parents=True, exist_ok=True)

    # ── Count existing samples ────────────────────────────────────────────────
    counts = {}
    for letter in LETTERS:
        counts[letter] = len(list((img_root / letter).glob("*.jpg")))

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = None
    for idx in [0, 1, 2]:
        for backend in [cv2.CAP_DSHOW, None]:
            tc = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
            if tc.isOpened():
                ret, fr = tc.read()
                if ret and fr is not None:
                    cap = tc
                    break
                tc.release()
        if cap: break

    if cap is None:
        print("[ERROR] No webcam found!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FPS, 30)

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    hands      = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=0.7
    )

    # ── State ─────────────────────────────────────────────────────────────────
    letter_idx    = 0
    last_capture  = 0.0
    session_start = time.time()
    paused        = False
    recent_feats  = collections.deque(maxlen=10)  # for live feature preview

    cv2.namedWindow("ASL Data Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ASL Data Collector", WIN_W, WIN_H)

    print("\n" + "="*55)
    print("  ASL Advanced Data Collector")
    print("="*55)
    print("  S / → : Skip to next letter")
    print("  ← B  : Go back to previous letter")
    print("  SPACE : Force capture now")
    print("  P     : Pause / Resume")
    print("  Q     : Quit and show summary")
    print("="*55 + "\n")

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        raw = cv2.flip(raw, 1)
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

        now         = time.time()
        letter      = LETTERS[letter_idx]
        count       = counts[letter]
        fraction    = count / TARGET_PER_CLASS
        hand_found  = False
        feat_vec    = None
        confidence  = 0.0

        # ── MediaPipe processing ──────────────────────────────────────────────
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks and res.multi_handedness:
            hand_lm = res.multi_hand_landmarks[0]
            handedness = res.multi_handedness[0]
            confidence = handedness.classification[0].score
            hand_found = (confidence >= MIN_CONFIDENCE)

            # Draw skeleton
            mp_drawing.draw_landmarks(
                raw, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            if hand_found:
                feat_vec = extract_advanced_features(hand_lm.landmark)
                recent_feats.append(feat_vec)

                # ── Auto-capture ──────────────────────────────────────────────
                if (not paused
                        and count < TARGET_PER_CLASS
                        and (now - last_capture) >= CAPTURE_INTERVAL):
                    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    hand_crop = crop_hand(raw, hand_lm)

                    # Save image
                    img_path = img_root / letter / f"img_{ts}.jpg"
                    cv2.imwrite(str(img_path), hand_crop)

                    # Save features
                    feat_path = feat_root / letter / f"feat_{ts}.npy"
                    np.save(str(feat_path), feat_vec)

                    counts[letter] += 1
                    count = counts[letter]
                    last_capture = now

        # ── Draw camera on left ───────────────────────────────────────────────
        cam = cv2.resize(raw, (640, 520))
        canvas[80:600, 0:640] = cam

        # ── Draw right panel ──────────────────────────────────────────────────
        rect_alpha(canvas, 640, 80, WIN_W, 600, color=(12,12,20), alpha=0.9)

        # Letter being collected — BIG
        put_text(canvas, letter, 690, 200, scale=4.0,
                 color=(255, 230, 50) if not paused else (150,150,150), thickness=4)

        # Progress bar
        draw_progress_bar(canvas, 660, 230, 580, 28, fraction,
                          fg=(0,200,80) if fraction < 1.0 else (0,100,255))
        status = f"{count} / {TARGET_PER_CLASS}"
        put_text(canvas, status, 660, 285, scale=0.85,
                 color=(255,255,255) if count < TARGET_PER_CLASS else (0,180,255))

        # Hand quality
        if hand_found:
            q_col = (80, 255, 80)
            q_txt = f"Hand: GOOD  ({int(confidence*100)}%)"
        else:
            q_col = (100, 100, 255)
            q_txt = f"Hand: NOT DETECTED  (need >= {int(MIN_CONFIDENCE*100)}%)"
        put_text(canvas, q_txt, 660, 325, scale=0.65, color=q_col)

        # Live feature preview (first 10 features as bar chart)
        put_text(canvas, "Live Features (41-dim):", 660, 360, scale=0.55,
                 color=(180,180,180))
        if feat_vec is not None:
            for fi in range(min(15, len(feat_vec))):
                bx = 660 + fi * 37
                bh = int(abs(feat_vec[fi]) * 80)
                col_val = int(min(255, abs(feat_vec[fi]) * 400))
                col = (0, col_val, 255 - col_val)
                cv2.rectangle(canvas, (bx, 445), (bx+32, 445-bh), col, -1)
            put_text(canvas, "joint angles", 660, 460, scale=0.40,
                     color=(120,120,120))

        # Cooldown bar
        time_since = now - last_capture
        cool_frac  = min(time_since / CAPTURE_INTERVAL, 1.0)
        draw_progress_bar(canvas, 660, 475, 580, 14, cool_frac,
                          fg=(200,150,0), bg=(30,30,30))
        put_text(canvas, "Capture cooldown", 660, 503, scale=0.45,
                 color=(150,150,150))

        # Paused indicator
        if paused:
            rect_alpha(canvas, 640, 80, WIN_W, 600, color=(80,0,0), alpha=0.4)
            put_text(canvas, "PAUSED — press P", 660, 550, scale=0.9,
                     color=(255, 100, 100))

        # ── Top bar ───────────────────────────────────────────────────────────
        rect_alpha(canvas, 0, 0, WIN_W, 78, color=(8,8,12), alpha=0.92)
        put_text(canvas, "ASL Advanced Data Collector", 18, 32,
                 scale=0.85, color=(100,210,255))
        elapsed = int(now - session_start)
        m, s    = divmod(elapsed, 60)
        total   = sum(counts.values())
        put_text(canvas,
                 f"Total: {total} | Time: {m:02d}:{s:02d} | "
                 f"[S]=Skip  [B]=Back  [SPACE]=Capture  [P]=Pause  [Q]=Quit",
                 18, 60, scale=0.45, color=(160,160,160))

        # Letter progress strip at top
        strip_w = WIN_W // len(LETTERS)
        for i, ltr in enumerate(LETTERS):
            frac = min(counts[ltr] / TARGET_PER_CLASS, 1.0)
            col  = (0, int(200*frac), int(80 + 175*(1-frac)))
            cv2.rectangle(canvas,
                          (i*strip_w, WIN_H-12),
                          ((i+1)*strip_w, WIN_H),
                          col, -1)
            if i == letter_idx:
                cv2.rectangle(canvas,
                              (i*strip_w, WIN_H-12),
                              ((i+1)*strip_w, WIN_H),
                              (255,255,255), 1)

        # ── Bottom bar ────────────────────────────────────────────────────────
        rect_alpha(canvas, 0, 600, WIN_W, WIN_H-12, color=(8,8,12), alpha=0.9)
        x = 20
        for ltr in LETTERS[:26]:
            c  = counts[ltr]
            fc = min(c / TARGET_PER_CLASS, 1.0)
            col = (0, int(200*fc), 80) if fc < 1 else (0, 100, 255)
            cv2.rectangle(canvas, (x, 608), (x+34, 630), col, -1)
            put_text(canvas, ltr,   x+8,  624, scale=0.4, color=(255,255,255))
            put_text(canvas, str(c), x+2, 640, scale=0.35,color=(200,200,200))
            x += 38

        cv2.imshow("ASL Data Collector", canvas)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("s") or key == ord("S") or key == 83:  # S or →
            letter_idx = (letter_idx + 1) % len(LETTERS)
            print(f"  Skipped → {LETTERS[letter_idx]}")
        elif key == ord("b") or key == ord("B") or key == 81:  # B or ←
            letter_idx = (letter_idx - 1) % len(LETTERS)
            print(f"  Back → {LETTERS[letter_idx]}")
        elif key == ord(" "):
            # Force capture now
            if hand_found and feat_vec is not None:
                ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                hcrop = crop_hand(raw, res.multi_hand_landmarks[0])
                cv2.imwrite(str(img_root / letter / f"img_{ts}.jpg"), hcrop)
                np.save(str(feat_root / letter / f"feat_{ts}.npy"), feat_vec)
                counts[letter] += 1
                last_capture = now
                print(f"  [FORCE] {letter}: {counts[letter]}")
        elif key == ord("p") or key == ord("P"):
            paused = not paused
            print(f"  {'PAUSED' if paused else 'RESUMED'}")

        # Auto-advance when target reached
        if counts[letter] >= TARGET_PER_CLASS:
            next_idx = (letter_idx + 1) % len(LETTERS)
            if next_idx != 0 or letter_idx != len(LETTERS) - 1:
                print(f"  ✓ {letter} complete! → {LETTERS[next_idx]}")
                letter_idx = next_idx
                time.sleep(0.5)

    # ── Cleanup & summary ─────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print("\n" + "="*55)
    print("  SESSION SUMMARY")
    print("="*55)
    total = 0
    for ltr in LETTERS:
        c   = counts[ltr]
        bar = "█" * int(c / TARGET_PER_CLASS * 20)
        pct = int(c / TARGET_PER_CLASS * 100)
        print(f"  {ltr:6s}  {bar:<20s}  {c:4d}/{TARGET_PER_CLASS}  ({pct:3d}%)")
        total += c
    print(f"\n  Total images collected: {total}")
    print(f"  Data saved to: data/images/  and  data/features/")
    print("="*55)


if __name__ == "__main__":
    collect()
