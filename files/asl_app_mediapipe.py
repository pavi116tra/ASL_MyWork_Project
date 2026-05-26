import os
import sys
import time
import collections
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

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

# --- Constants & Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "asl_mediapipe_dense.h5")
# Will be loaded from data/label_classes.npy later

# --- Visual Style ---
BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"
ACCENT_COLOR = "#7c6af7"
CONFIRMED_COLOR = "#4af0c4"
TEXT_COLOR = "#ffffff"
MUTED_TEXT = "#a0a0a0"

def extract_landmarks(hand_landmarks, frame_shape):
    h, w = frame_shape[:2]
    landmarks = []
    wrist = hand_landmarks.landmark[0]  # anchor point

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
    """
    Distances from each fingertip to the palm center.
    This is the KEY feature that separates A / S / E / T / M / N.
    """
    lm = hand_lm.landmark

    # Palm center = average of wrist + 5 MCP joints
    palm_pts = [0, 1, 5, 9, 13, 17]
    palm_x = np.mean([lm[i].x for i in palm_pts])
    palm_y = np.mean([lm[i].y for i in palm_pts])
    palm_z = np.mean([lm[i].z for i in palm_pts])

    # Distance of each fingertip to palm center
    tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    dists = []
    for t in tips:
        d = np.sqrt(
            (lm[t].x - palm_x)**2 +
            (lm[t].y - palm_y)**2 +
            (lm[t].z - palm_z)**2   # Z is critical for A/S/E/T
        )
        dists.append(d)

    # Also: thumb tip to index/middle/ring tip distances (separates A vs S vs T)
    thumb = lm[4]
    cross = []
    for t in [8, 12, 16]:
        cross.append(np.sqrt(
            (thumb.x - lm[t].x)**2 +
            (thumb.y - lm[t].y)**2 +
            (thumb.z - lm[t].z)**2
        ))

    return np.array(dists + cross, dtype=np.float32)  # 8-dim

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
        thumb_x_offset,
        thumb_y_offset,
        thumb_z_offset,
        pinky_extension,
        thumb_wrist_dist,
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

    return np.array(
        gaps + [index_curve, middle_curve, ring_curve],
        dtype=np.float32
    )  

def get_feature_vector(hand_landmarks, frame_shape):
    lm_vec    = extract_landmarks(hand_landmarks, frame_shape)
    angle_vec = extract_angles(hand_landmarks)
    dist_vec  = extract_fingertip_distances(hand_landmarks)
    thumb_vec = extract_thumb_position(hand_landmarks)
    pinch_vec = extract_pinch_gaps(hand_landmarks)
    return np.concatenate([lm_vec, angle_vec, dist_vec, thumb_vec, pinch_vec])


class ASLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Sign Detector - MediaPipe Edition")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        
        # State variables
        self.cap = None
        self.is_running = False
        self.history_buf = collections.deque(maxlen=15)
        self.sentence = ""
        self.last_confirm_time = 0
        self.last_detection_time = time.time()
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load Model & Classes
        self.classes = []
        self.model = self.load_model()
        
        # Build UI
        self.build_ui()
        self.update_status("Camera Off")

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", f"Model file not found: {MODEL_PATH}\nPlease run train_mediapipe.py first.")
            sys.exit(1)
            
        class_path = os.path.join(BASE_DIR, "data", "label_classes.npy")
        if os.path.exists(class_path):
            self.classes = np.load(class_path).tolist()
        else:
            self.classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["delete", "nothing", "space"]
            
        print(f"[INFO] Loading MediaPipe Dense Model: {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)

    def build_ui(self):
        # Top Bar
        top_frame = tk.Frame(self.root, bg=BG_COLOR, pady=10)
        top_frame.pack(fill=tk.X)
        
        tk.Label(top_frame, text="ASL Sign Detector", font=("Helvetica", 28, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR).pack()
        tk.Label(top_frame, text="Powered by MediaPipe Hand Tracking", font=("Helvetica", 12), bg=BG_COLOR, fg=MUTED_TEXT).pack()

        # Main Content
        main_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left Panel - Camera Feed
        left_panel = tk.Frame(main_frame, bg=PANEL_COLOR, bd=2, relief=tk.FLAT)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self.video_label = tk.Label(left_panel, bg="black")
        self.video_label.pack(expand=True)
        
        # Right Panel - Predictions
        right_panel = tk.Frame(main_frame, bg=PANEL_COLOR, width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        tk.Label(right_panel, text="PREDICTION", font=("Helvetica", 14, "bold"), bg=PANEL_COLOR, fg=MUTED_TEXT).pack(pady=(20, 0))
        self.pred_letter_label = tk.Label(right_panel, text="-", font=("Helvetica", 96, "bold"), bg=PANEL_COLOR, fg=TEXT_COLOR)
        self.pred_letter_label.pack(pady=10)
        
        self.pred_conf_label = tk.Label(right_panel, text="Confidence: 0%", font=("Helvetica", 14), bg=PANEL_COLOR, fg=MUTED_TEXT)
        self.pred_conf_label.pack()
        
        # Confidence Bar
        self.conf_canvas = tk.Canvas(right_panel, width=250, height=15, bg=BG_COLOR, highlightthickness=0)
        self.conf_canvas.pack(pady=10)
        self.conf_bar = self.conf_canvas.create_rectangle(0, 0, 0, 15, fill="red")
        
        # Top 3 List
        tk.Label(right_panel, text="Top 3 Predictions:", font=("Helvetica", 12, "bold"), bg=PANEL_COLOR, fg=MUTED_TEXT).pack(pady=(20, 5))
        self.top3_labels = []
        for i in range(3):
            lbl = tk.Label(right_panel, text=f"{i+1}st: -", font=("Helvetica", 12), bg=PANEL_COLOR, fg=TEXT_COLOR)
            lbl.pack(anchor=tk.W, padx=50)
            self.top3_labels.append(lbl)
            
        # Built Sentence Text Area
        tk.Label(right_panel, text="Sentence Built:", font=("Helvetica", 12, "bold"), bg=PANEL_COLOR, fg=MUTED_TEXT).pack(pady=(30, 5), anchor=tk.W, padx=20)
        self.sentence_text = tk.Text(right_panel, height=4, width=30, font=("Helvetica", 14), bg=BG_COLOR, fg=CONFIRMED_COLOR, wrap=tk.WORD, bd=0, padx=10, pady=10)
        self.sentence_text.pack(padx=20, fill=tk.X)
        self.sentence_text.configure(state='disabled')
        
        # Clear Button
        clear_btn = tk.Button(right_panel, text="Clear Text", font=("Helvetica", 12, "bold"), bg="#555555", fg="white", activebackground="#666666", activeforeground="white", relief=tk.FLAT, command=self.clear_text)
        clear_btn.pack(pady=10, ipadx=10, ipady=5)

        # Bottom Bar
        bottom_frame = tk.Frame(self.root, bg=BG_COLOR, pady=15)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20)
        
        self.start_btn = tk.Button(bottom_frame, text="Start Camera", font=("Helvetica", 12, "bold"), bg="#28a745", fg="white", activebackground="#218838", activeforeground="white", relief=tk.FLAT, command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, ipadx=15, ipady=5)
        
        self.stop_btn = tk.Button(bottom_frame, text="Stop Camera", font=("Helvetica", 12, "bold"), bg="#dc3545", fg="white", activebackground="#c82333", activeforeground="white", relief=tk.FLAT, command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10, ipadx=15, ipady=5)
        
        self.status_label = tk.Label(bottom_frame, text="Status: Ready", font=("Helvetica", 12, "italic"), bg=BG_COLOR, fg=MUTED_TEXT)
        self.status_label.pack(side=tk.RIGHT)

    def update_status(self, text, color=MUTED_TEXT):
        self.status_label.config(text=f"Status: {text}", fg=color)

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.update_status("No camera found", "red")
                messagebox.showerror("Camera Error", "Could not access the webcam.")
                return
            
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.update_status("Camera Running", CONFIRMED_COLOR)
            self.history_buf.clear()
            self.update_frame()

    def stop_camera(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        self.video_label.config(image='')
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status("Camera Off")
        self.clear_prediction_panel()

    def clear_prediction_panel(self):
        self.pred_letter_label.config(text="-")
        self.pred_conf_label.config(text="Confidence: 0%")
        self.conf_canvas.coords(self.conf_bar, 0, 0, 0, 15)
        for i in range(3):
            self.top3_labels[i].config(text=f"{i+1}{['st','nd','rd'][i]}: -")
        self.history_buf.clear()

    def confirm_letter(self, letter):
        current_time = time.time()
        if current_time - self.last_confirm_time > 1.5:
            if letter == "space":
                self.sentence += " "
            elif letter == "delete":
                self.sentence = self.sentence[:-1]
            elif letter != "nothing":
                self.sentence += letter
                
            self.sentence_text.configure(state='normal')
            self.sentence_text.delete(1.0, tk.END)
            self.sentence_text.insert(tk.END, self.sentence)
            self.sentence_text.see(tk.END)
            self.sentence_text.configure(state='disabled')
            
            self.last_confirm_time = current_time

    def clear_text(self):
        self.sentence = ""
        self.sentence_text.configure(state='normal')
        self.sentence_text.delete(1.0, tk.END)
        self.sentence_text.configure(state='disabled')
        self.history_buf.clear()

    def update_frame(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(15, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process MediaPipe
        results = self.hands.process(rgb_frame)
        
        preds = None

        if results.multi_hand_landmarks:
            self.last_detection_time = time.time()
            self.update_status("Hand Detected", CONFIRMED_COLOR)
            
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            if self.model is not None:
                # Extract 73-dim feature vector
                features = get_feature_vector(hand_landmarks, frame.shape)
                input_arr = np.expand_dims(features, axis=0)
                preds = self.model.predict(input_arr, verbose=0)[0]
                
        else:
            cv2.putText(rgb_frame, "No hand detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            self.update_status("Camera Running", ACCENT_COLOR)
            if time.time() - self.last_detection_time > 2.0:
                self.clear_prediction_panel()

        # Update right panel if we have predictions
        if preds is not None:
            top_indices = np.argsort(preds)[::-1]
            top_conf = preds[top_indices[0]]
            top_class = self.classes[top_indices[0]]
            
            self.pred_letter_label.config(text=top_class)
            self.pred_conf_label.config(text=f"Confidence: {top_conf*100:.0f}%")
            
            bar_len = int(250 * top_conf)
            if top_conf > 0.70: color = "#28a745"
            elif top_conf > 0.40: color = "#ffc107"
            else: color = "#dc3545"
            self.conf_canvas.coords(self.conf_bar, 0, 0, bar_len, 15)
            self.conf_canvas.itemconfig(self.conf_bar, fill=color)
            
            suffixes = ['st', 'nd', 'rd']
            for i in range(3):
                idx = top_indices[i]
                self.top3_labels[i].config(text=f"{i+1}{suffixes[i]}: {self.classes[idx]} — {preds[idx]*100:.0f}%")
                
            if top_conf > 0.65:
                self.history_buf.append(top_class)
                
            if len(self.history_buf) == 15:
                counts = collections.Counter(self.history_buf)
                most_common, count = counts.most_common(1)[0]
                if count >= 12:
                    self.confirm_letter(most_common)
                    self.history_buf.clear()

        # Resize video to fit left panel while maintaining aspect ratio
        panel_w = self.video_label.master.winfo_width()
        panel_h = self.video_label.master.winfo_height()
        if panel_w > 10 and panel_h > 10:
            img_h, img_w, _ = rgb_frame.shape
            scale = min(panel_w / img_w, panel_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            rgb_frame = cv2.resize(rgb_frame, (new_w, new_h))
            
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(15, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLApp(root)
    root.mainloop()
