import os
import sys
import time
import collections
import cv2
import numpy as np
from pathlib import Path

# Fix for protobuf compatibility issue with mediapipe
import google._upb._message
import google.protobuf.symbol_database as sym_db
import google.protobuf.message_factory as msg_factory

if not hasattr(google._upb._message.FieldDescriptor, 'label'):
    google._upb._message.FieldDescriptor.label = property(lambda self: getattr(self, '_label', None))
if not hasattr(sym_db.SymbolDatabase, 'GetPrototype'):
    sym_db.SymbolDatabase.GetPrototype = lambda self, descriptor: msg_factory.GetMessageClass(descriptor)

import mediapipe as mp
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# --- FEATURE EXTRACTORS (93-dim) ---

def extract_landmarks(hand_landmarks, frame_shape):
    h, w = frame_shape[:2]
    landmarks = []
    wrist = hand_landmarks.landmark[0]

    for lm in hand_landmarks.landmark:
        landmarks.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])

    mid_mcp = hand_landmarks.landmark[9]
    scale = np.sqrt(
        (mid_mcp.x - wrist.x)**2 +
        (mid_mcp.y - wrist.y)**2 +
        (mid_mcp.z - wrist.z)**2
    )
    if scale > 0:
        landmarks = [v / scale for v in landmarks]

    return np.array(landmarks, dtype=np.float32)

def extract_angles(hand_landmarks):
    lm = hand_landmarks.landmark

    def angle(a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    finger_joints = [
        (1, 2, 3),   (2, 3, 4),    
        (5, 6, 7),   (6, 7, 8),    
        (9, 10, 11), (10, 11, 12), 
        (13, 14, 15),(14, 15, 16), 
        (17, 18, 19),(18, 19, 20), 
    ]
    angles = [angle(lm[a], lm[b], lm[c]) for a, b, c in finger_joints]
    return np.array(angles, dtype=np.float32)

def extract_fingertip_distances(hand_lm):
    lm = hand_lm.landmark
    palm_pts = [0, 1, 5, 9, 13, 17]
    palm_x = np.mean([lm[i].x for i in palm_pts])
    palm_y = np.mean([lm[i].y for i in palm_pts])
    palm_z = np.mean([lm[i].z for i in palm_pts])

    tips = [4, 8, 12, 16, 20]
    dists = []
    for t in tips:
        d = np.sqrt(
            (lm[t].x - palm_x)**2 +
            (lm[t].y - palm_y)**2 +
            (lm[t].z - palm_z)**2
        )
        dists.append(d)

    thumb = lm[4]
    cross = []
    for t in [8, 12, 16]:
        cross.append(np.sqrt(
            (thumb.x - lm[t].x)**2 +
            (thumb.y - lm[t].y)**2 +
            (thumb.z - lm[t].z)**2
        ))

    return np.array(dists + cross, dtype=np.float32)

def extract_thumb_position(hand_lm):
    lm = hand_lm.landmark
    thumb_tip   = lm[4]
    index_mcp   = lm[5]   
    pinky_tip   = lm[20]
    pinky_mcp   = lm[17]  
    wrist       = lm[0]

    thumb_x_offset = thumb_tip.x - index_mcp.x   
    thumb_z_offset = thumb_tip.z - index_mcp.z   
    thumb_y_offset = thumb_tip.y - index_mcp.y   
    pinky_extension = pinky_mcp.y - pinky_tip.y  

    thumb_wrist_dist = np.sqrt(
        (thumb_tip.x - wrist.x)**2 +
        (thumb_tip.y - wrist.y)**2 +
        (thumb_tip.z - wrist.z)**2
    )
    return np.array([
        thumb_x_offset, thumb_y_offset, thumb_z_offset, pinky_extension, thumb_wrist_dist
    ], dtype=np.float32)

def extract_pinch_gaps(hand_lm):
    lm = hand_lm.landmark
    thumb = lm[4]
    tips  = [8, 12, 16, 20]  
    gaps = []
    for t in tips:
        gaps.append(np.sqrt(
            (thumb.x - lm[t].x)**2 +
            (thumb.y - lm[t].y)**2 +
            (thumb.z - lm[t].z)**2
        ))
    index_curve  = lm[6].y - lm[8].y   
    middle_curve = lm[10].y - lm[12].y
    ring_curve   = lm[14].y - lm[16].y
    return np.array(gaps + [index_curve, middle_curve, ring_curve], dtype=np.float32)

def get_feature_vector(hand_landmarks, frame_shape):
    lm_vec    = extract_landmarks(hand_landmarks, frame_shape)
    angle_vec = extract_angles(hand_landmarks)
    dist_vec  = extract_fingertip_distances(hand_landmarks)
    thumb_vec = extract_thumb_position(hand_landmarks)
    pinch_vec = extract_pinch_gaps(hand_landmarks)
    return np.concatenate([lm_vec, angle_vec, dist_vec, thumb_vec, pinch_vec])

# --- OVERRIDE RULES ---
def apply_disambiguation_rules(label, hand_lm):
    lm = hand_lm.landmark
    pinky_tip = lm[20]
    pinky_pip = lm[18]
    thumb_tip = lm[4]
    index_mcp = lm[5]

    pinky_extended = pinky_pip.y - pinky_tip.y > 0.04
    
    if label == 'I' and not pinky_extended:
        return 'A'
    if label == 'A' and pinky_extended:
        return 'I'
    return label

def fix_c_vs_o(label, hand_lm):
    lm = hand_lm.landmark
    thumb_tip  = lm[4]
    index_tip  = lm[8]
    
    thumb_index_gap = np.sqrt(
        (thumb_tip.x - index_tip.x)**2 +
        (thumb_tip.y - index_tip.y)**2 +
        (thumb_tip.z - index_tip.z)**2
    )

    if label == 'O' and thumb_index_gap > 0.12: return 'C'
    if label == 'C' and thumb_index_gap < 0.07: return 'O'
    return label

def check_hand_orientation(hand_lm):
    lm = hand_lm.landmark
    wrist   = lm[0]
    mid_mcp = lm[9]   

    dy = wrist.y - mid_mcp.y   
    dx = abs(wrist.x - mid_mcp.x)
    angle_deg = np.degrees(np.arctan2(dx, dy))

    if angle_deg > 40:
        return f"Tilt warning: {angle_deg:.0f} deg - face palm toward camera"
    return None

# --- MODEL LOADING ---
def safe_load_model(path, n_classes=None):
    try:
        return tf.keras.models.load_model(str(path))
    except Exception as e:
        print(f"  [compat] Direct load failed: {e}. Rebuilding model architecture configuration...")
        pass

    import h5py
    if n_classes is None:
        try:
            with h5py.File(str(path), "r") as f:
                kernels = []
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset) and "kernel" in name:
                        kernels.append((name, obj.shape))
                f.visititems(visit)
                kernels_2d = [(n, s) for n, s in kernels if len(s) == 2]
                n_classes = min(kernels_2d, key=lambda x: x[1][-1])[1][-1] if kernels_2d else 29
        except Exception:
            n_classes = 29
    print(f"  [compat] Rebuilding Sequential architecture for output classes: {n_classes}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu", name="dense", input_shape=(93,)),
        tf.keras.layers.BatchNormalization(               name="batch_normalization"),
        tf.keras.layers.Dropout(0.4,                      name="dropout"),
        tf.keras.layers.Dense(128, activation="relu",     name="dense_1"),
        tf.keras.layers.BatchNormalization(               name="batch_normalization_1"),
        tf.keras.layers.Dropout(0.3,                      name="dropout_1"),
        tf.keras.layers.Dense(64,  activation="relu",     name="dense_2"),
        tf.keras.layers.Dense(n_classes, activation="softmax", name="dense_3"),
    ], name="sequential")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    try:
        model.load_weights(str(path), by_name=True, skip_mismatch=True)
        print("  [compat] Weights loaded successfully.")
    except Exception as e:
        print(f"  [compat] Weight loading failed: {e}")
    return model

def load_best_model(model_dir="model"):
    log_path = Path(model_dir) / "model_log.txt"
    if log_path.exists():
        best_acc   = 0
        best_file  = None
        with open(log_path) as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 5: continue
                try:
                    acc  = float(parts[1].split(":")[1].strip().replace("%",""))
                    # Find the part that starts with "file:" or use the last part
                    file_part = parts[-1]
                    for p in parts:
                        if "file:" in p:
                            file_part = p
                            break
                    fname = file_part.split(":")[1].strip()
                    if acc > best_acc:
                        best_acc  = acc
                        best_file = fname
                except:
                    continue
        if best_file:
            model_path = Path(model_dir) / best_file
            print(f"Loading best model: {best_file} ({best_acc:.2f}% acc)")
            return safe_load_model(model_path)

    fallback = Path(model_dir) / "asl_dense_model_BEST.h5"
    if fallback.exists():
        print(f"Loading fallback: {fallback}")
        return safe_load_model(fallback)
    
    # Backward compatibility
    old = Path("asl_mediapipe_dense.h5")
    if old.exists():
        print(f"Loading old model: {old}")
        return safe_load_model(old)
        
    print("No model found!")
    sys.exit(1)

def get_classes():
    p1 = Path("model/label_classes.npy")
    p2 = Path("data/label_classes.npy")
    if p1.exists(): return np.load(p1).tolist()
    if p2.exists(): return np.load(p2).tolist()
    return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["delete", "nothing", "space"]

# --- STATE MACHINE AND UI STYLING ---

class State:
    IDLE      = "idle"       # waiting for a hand / low confidence prediction
    HOLDING   = "holding"    # hand sign is held steady, countdown active
    PENDING   = "pending"    # hold finished, waiting for SPACE/BACKSPACE confirmation
    COOLDOWN  = "cooldown"   # 2-second cooldown after a confirmed letter

# Visual helpers
FONT = cv2.FONT_HERSHEY_DUPLEX

def draw_semi_transparent_rect(img, x1, y1, x2, y2, color=(0, 0, 0), alpha=0.6):
    """Draw a semi-transparent filled rectangle for HUD background."""
    sub = img[y1:y2, x1:x2]
    overlay = np.full(sub.shape, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub

def draw_text_with_shadow(img, text, x, y, scale=0.7, color=(255, 255, 255), thickness=1):
    """Draw text with a drop shadow to ensure readability on any background."""
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)

def draw_progress_bar(img, x, y, w, h, fraction, bg=(50, 50, 50), fg=(0, 200, 100), border=(180, 180, 180)):
    """Draw a progress bar with custom fill fraction."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    filled_w = int(w * min(max(fraction, 0.0), 1.0))
    if filled_w > 0:
        cv2.rectangle(img, (x, y), (x + filled_w, y + h), fg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), border, 1)

def draw_ui(frame, state, cur_label, conf, hold_fraction, pending_letter, current_word, sentence, cooldown_left):
    H, W = frame.shape[:2]
    PAD = 10

    # 1. Top HUD Background (semi-transparent black)
    rect_h = 160
    draw_semi_transparent_rect(frame, 0, 0, W, rect_h, color=(15, 15, 15), alpha=0.75)

    # Row 1 (White): Live Prediction
    conf_pct = int(conf * 100)
    if cur_label and cur_label not in ("nothing", "-"):
        row1_text = f"Detected Sign: {cur_label} ({conf_pct}%)"
        row1_color = (255, 255, 255)
    else:
        row1_text = "Detected Sign: -"
        row1_color = (180, 180, 180)
    draw_text_with_shadow(frame, row1_text, PAD + 10, 38, scale=0.85, color=row1_color, thickness=1)

    # Row 2 (Yellow): Current Word
    # Format word like: [ H E L L O ]
    word_display = " ".join(list(current_word)).upper() if current_word else ""
    row2_text = f"Current Word: [ {word_display} ]"
    draw_text_with_shadow(frame, row2_text, PAD + 10, 80, scale=0.85, color=(0, 255, 255), thickness=1)

    # Row 3 (Green): Sentence
    # Wrap sentence display to avoid going off-screen
    max_len = 50
    sentence_display = sentence[-max_len:] if len(sentence) > max_len else sentence
    row3_text = f"Sentence: {sentence_display}"
    draw_text_with_shadow(frame, row3_text, PAD + 10, 118, scale=0.85, color=(0, 255, 0), thickness=1)

    # 2. Progress / Hold / Cooldown bar
    bar_y = 132
    bar_w = W - (PAD * 2)
    bar_h = 16
    
    if state == State.HOLDING:
        bar_fg = (0, 165, 255)  # Orange
        bar_lbl = f"Holding steady... {hold_fraction * 1.5:.1f}s / 1.5s"
        draw_progress_bar(frame, PAD, bar_y, bar_w, bar_h, hold_fraction, fg=bar_fg)
        draw_text_with_shadow(frame, bar_lbl, PAD + 6, bar_y + 12, scale=0.45, color=(220, 220, 220), thickness=1)
    elif state == State.COOLDOWN:
        bar_fg = (100, 100, 255)  # Purple-ish blue
        bar_lbl = f"Cooldown... {cooldown_left:.1f}s / 2.0s"
        draw_progress_bar(frame, PAD, bar_y, bar_w, bar_h, hold_fraction, fg=bar_fg)
        draw_text_with_shadow(frame, bar_lbl, PAD + 6, bar_y + 12, scale=0.45, color=(220, 220, 220), thickness=1)
    elif state == State.PENDING:
        bar_fg = (0, 255, 140)  # Green-blue
        letter_str = pending_letter.upper() if pending_letter else ""
        bar_lbl = f">> '{letter_str}' Ready - SPACE to confirm / BACKSPACE to reject"
        draw_progress_bar(frame, PAD, bar_y, bar_w, bar_h, 1.0, fg=bar_fg)
        draw_text_with_shadow(frame, bar_lbl, PAD + 6, bar_y + 12, scale=0.45, color=(20, 20, 20), thickness=1)

    # 3. Confirmation Dialog Box (during PENDING)
    if state == State.PENDING:
        bx1, by1 = W // 2 - 220, H // 2 - 60
        bx2, by2 = W // 2 + 220, H // 2 + 60
        # Draw green-accented box
        draw_semi_transparent_rect(frame, bx1, by1, bx2, by2, color=(0, 45, 0), alpha=0.85)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 220, 80), 2)
        letter_str = pending_letter.upper() if pending_letter else ""
        draw_text_with_shadow(frame, f"Pending Sign: '{letter_str}'", bx1 + 25, by1 + 45, scale=1.1, color=(0, 255, 120), thickness=2)
        draw_text_with_shadow(frame, "SPACE = Confirm  |  BACKSPACE = Reject", bx1 + 25, by1 + 85, scale=0.6, color=(200, 255, 200), thickness=1)

    # 4. State badge (top-right overlay)
    badge_colors = {
        State.IDLE:     (80, 80, 80),
        State.HOLDING:  (0, 160, 255),
        State.PENDING:  (0, 200, 80),
        State.COOLDOWN: (140, 100, 255),
    }
    badge_text = state.upper()
    badge_col  = badge_colors.get(state, (80, 80, 80))
    (tw, th), _ = cv2.getTextSize(badge_text, FONT, 0.5, 1)
    bx = W - tw - 20
    draw_semi_transparent_rect(frame, bx - 6, 6, bx + tw + 6, 6 + th + 10, color=badge_col, alpha=0.85)
    draw_text_with_shadow(frame, badge_text, bx, 6 + th + 2, scale=0.5, color=(255, 255, 255), thickness=1)

    # 5. Bottom Shortcut Bar (semi-transparent black)
    draw_semi_transparent_rect(frame, 0, H - 34, W, H, color=(10, 10, 10), alpha=0.8)
    shortcuts = "[SPACE] Confirm  [BKSP] Delete Letter  [ENTER] Add Word  [ESC] Clear  [Q] Quit"
    draw_text_with_shadow(frame, shortcuts, PAD + 10, H - 10, scale=0.45, color=(180, 180, 180), thickness=1)

# --- MAIN APP ---
def main():
    print("\n" + "=" * 60)
    print("  ASL Sentence Builder - Enhanced UX Edition")
    print("=" * 60)

    print("\nLoading model...")
    model = load_best_model()
    classes = get_classes()
    print(f"  Classes loaded: {classes}")

    print("\nInitializing MediaPipe hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Robust camera scanner
    cap = None
    print("\nScanning for available webcam devices...")
    for index in [0, 1, 2]:
        for backend in [None, cv2.CAP_DSHOW]:
            print(f"  Trying webcam index {index} with backend {backend}...")
            if backend is not None:
                try_cap = cv2.VideoCapture(index, backend)
            else:
                try_cap = cv2.VideoCapture(index)
                
            if try_cap.isOpened():
                ret, frame = try_cap.read()
                if ret and frame is not None:
                    cap = try_cap
                    print(f"  [SUCCESS] Webcam opened successfully at index {index}!")
                    break
                else:
                    try_cap.release()
        if cap is not None:
            break
            
    if cap is None:
        print("[ERROR] Cannot open any webcam! Please check connection.")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 860)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 560)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("  Webcam configurations applied successfully.")
    print("  Press 'Q' inside the window to exit.")

    # State variables
    state = State.IDLE
    hold_start = None
    hold_buf = collections.deque(maxlen=20)  # recent labels during hold to verify steadiness
    pending_letter = None
    cooldown_end = None
    last_confirmed = None  # Prevent immediate repeat of same letter in a row

    current_word = ""
    sentence = ""

    # Live detection values
    cur_label = "-"
    conf = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        now = time.time()
        
        hand_detected = results.multi_hand_landmarks is not None
        
        if hand_detected:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Orientation check warning
            warn = check_hand_orientation(hand_landmarks)
            if warn:
                draw_text_with_shadow(frame, warn, 10, H - 45, scale=0.6, color=(0, 0, 255), thickness=1)
            
            features = get_feature_vector(hand_landmarks, frame.shape)
            if features.shape[0] == 93:
                input_arr = np.expand_dims(features, axis=0)
                preds = model.predict(input_arr, verbose=0)[0]
                
                top_idx = np.argmax(preds)
                predicted_label = classes[top_idx]
                conf = float(preds[top_idx])
                
                # Apply override rules
                predicted_label = apply_disambiguation_rules(predicted_label, hand_landmarks)
                predicted_label = fix_c_vs_o(predicted_label, hand_landmarks)
                cur_label = predicted_label
            else:
                cur_label, conf = "-", 0.0
        else:
            cur_label, conf = "-", 0.0
            # If hand is completely removed, clear the repetition prevention buffer
            last_confirmed = None

        # ── State Machine Processing ──────────────────────────────────────────
        hold_fraction = 0.0
        cooldown_left = 0.0

        if state == State.COOLDOWN:
            cooldown_left = cooldown_end - now
            if cooldown_left <= 0:
                state = State.IDLE
                cooldown_left = 0.0
                last_confirmed = None  # Allow repeating after a full cooldown and release cycle
            hold_fraction = cooldown_left / 2.0
            draw_ui(frame, state, cur_label, conf, hold_fraction, pending_letter, current_word, sentence, cooldown_left)

        elif state == State.PENDING:
            # In PENDING state, wait for keyboard confirmation (SPACE / BACKSPACE)
            draw_ui(frame, state, cur_label, conf, 1.0, pending_letter, current_word, sentence, 0.0)

        elif state == State.HOLDING:
            elapsed = now - hold_start
            hold_fraction = elapsed / 1.5

            # Steadiness check: ensure the label sequence matches the target
            hold_buf.append(cur_label)
            dominant = collections.Counter(hold_buf).most_common(1)[0][0]
            stability = hold_buf.count(dominant) / len(hold_buf)

            if not hand_detected or conf < 0.70 or stability < 0.7 or cur_label == "nothing":
                # Hand moved, was lost, or dropped confidence -> reset
                state = State.IDLE
                hold_start = None
                hold_buf.clear()
            elif elapsed >= 1.5:
                # Hold duration met -> proceed to confirmation prompt
                pending_letter = dominant
                state = State.PENDING
            
            draw_ui(frame, state, cur_label, conf, hold_fraction, pending_letter, current_word, sentence, 0.0)

        else:  # IDLE
            if (hand_detected 
                    and conf >= 0.70 
                    and cur_label not in ("nothing", "-") 
                    and cur_label != last_confirmed):
                state = State.HOLDING
                hold_start = now
                hold_buf.clear()
                hold_buf.append(cur_label)
                
            draw_ui(frame, state, cur_label, conf, 0.0, None, current_word, sentence, 0.0)

        # ── Keyboard Event Handlers ──────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # [Q] to quit
            print("\nExiting app cleanly...")
            break

        if state == State.PENDING:
            if key == 32:  # SPACE to confirm
                action = pending_letter
                if action == "space":
                    sentence += current_word + " "
                    current_word = ""
                elif action == "delete":
                    if current_word:
                        current_word = current_word[:-1]
                elif action != "nothing":
                    current_word += action
                    last_confirmed = action
                
                pending_letter = None
                cooldown_end = time.time() + 2.0
                state = State.COOLDOWN
                hold_buf.clear()

            elif key == 8:  # BACKSPACE to reject pending
                print(f"Rejected pending sign '{pending_letter}'")
                pending_letter = None
                state = State.IDLE
                hold_buf.clear()
                last_confirmed = None

        else:  # IDLE or COOLDOWN or HOLDING
            if key == 8:  # BACKSPACE to delete last letter in current word
                if current_word:
                    current_word = current_word[:-1]
            elif key == 13:  # ENTER to finalize word and add space to sentence
                if current_word:
                    sentence += current_word + " "
                    current_word = ""
            elif key == 255 or key == 0 or key == 127:  # DELETE key to clear current word
                current_word = ""
            elif key == 27:  # ESC key to clear entire sentence
                sentence = ""
                current_word = ""
                state = State.IDLE
                hold_buf.clear()
                last_confirmed = None

        cv2.imshow('ASL Sentence Builder', frame)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    final_output = (sentence + current_word).strip()
    if final_output:
        print(f"\nFinal Sentence Built: {final_output}")
    print("Goodbye!")

if __name__ == "__main__":
    main()
