import os
import cv2
import numpy as np
import sys
import glob

# --- Protobuf Monkey Patches for MediaPipe ---
import google._upb._message
import google.protobuf.symbol_database as sym_db
import google.protobuf.message_factory as msg_factory

if not hasattr(google._upb._message.FieldDescriptor, 'label'):
    google._upb._message.FieldDescriptor.label = property(lambda self: getattr(self, '_label', None))
if not hasattr(sym_db.SymbolDatabase, 'GetPrototype'):
    sym_db.SymbolDatabase.GetPrototype = lambda self, descriptor: msg_factory.GetMessageClass(descriptor)

import mediapipe as mp

# ─────────────────────────────────────────
# 1. MEDIAPIPE SETUP
# ─────────────────────────────────────────
mp_hands = mp.solutions.hands
# Using static_image_mode=True for processing dataset images
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
)

# ─────────────────────────────────────────
# 2. LANDMARK EXTRACTION
# ─────────────────────────────────────────
def extract_landmarks(hand_landmarks, frame_shape):
    """
    Returns a normalized 63-dim vector (21 points × x,y,z).
    Normalized relative to wrist so position-invariant.
    """
    h, w = frame_shape[:2]
    landmarks = []
    wrist = hand_landmarks.landmark[0]  # anchor point

    for lm in hand_landmarks.landmark:
        # Subtract wrist to make translation-invariant
        landmarks.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])

    # Normalize scale by the distance wrist→middle_finger_mcp
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
    """
    Returns 15 joint angles (thumb + 4 fingers × 3 joints each).
    Angles are rotation-invariant features.
    """
    lm = hand_landmarks.landmark

    def angle(a, b, c):
        """Angle at joint b, given points a-b-c."""
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    # Finger joint triplets: (base, mid, tip) indices
    finger_joints = [
        (1, 2, 3),   (2, 3, 4),    # Thumb
        (5, 6, 7),   (6, 7, 8),    # Index
        (9, 10, 11), (10, 11, 12), # Middle
        (13, 14, 15),(14, 15, 16), # Ring
        (17, 18, 19),(18, 19, 20), # Pinky
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
    """
    Captures WHERE the thumb sits relative to the fist.
    A = thumb beside fist (x-axis difference large)
    I = thumb tucked down (y-axis difference + pinky up)
    S = thumb OVER fingers (z-axis difference)
    """
    lm = hand_lm.landmark

    thumb_tip   = lm[4]
    index_mcp   = lm[5]   # index knuckle
    pinky_tip   = lm[20]
    pinky_mcp   = lm[17]  # pinky knuckle
    wrist       = lm[0]

    # Key separator: thumb tip vs index knuckle (x and z)
    thumb_x_offset = thumb_tip.x - index_mcp.x   # negative = thumb left of fist = A
    thumb_z_offset = thumb_tip.z - index_mcp.z   # positive = thumb in front = S
    thumb_y_offset = thumb_tip.y - index_mcp.y   # A vs T separation

    # Pinky extension = I detector
    pinky_extension = pinky_mcp.y - pinky_tip.y  # positive = pinky up = I

    # Thumb tip to wrist distance (A has thumb close to side)
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
    ], dtype=np.float32)  # 5-dim

def extract_pinch_gaps(hand_lm):
    """
    Measures gaps between thumb and each fingertip.
    This is the KEY feature for C vs O vs D vs G vs F.
    """
    lm = hand_lm.landmark
    thumb = lm[4]
    tips  = [8, 12, 16, 20]  # index, middle, ring, pinky

    gaps = []
    for t in tips:
        gaps.append(np.sqrt(
            (thumb.x - lm[t].x)**2 +
            (thumb.y - lm[t].y)**2 +
            (thumb.z - lm[t].z)**2
        ))

    # Also: curvature — how curved are the fingers overall
    # High curvature = C, full closure = O
    index_curve  = lm[6].y - lm[8].y   # pip above tip = curved
    middle_curve = lm[10].y - lm[12].y
    ring_curve   = lm[14].y - lm[16].y

    return np.array(
        gaps + [index_curve, middle_curve, ring_curve],
        dtype=np.float32
    )  # 7-dim

def get_feature_vector(hand_landmarks, frame_shape):
    """Combine all extractors → 93-dim feature vector."""
    lm_vec    = extract_landmarks(hand_landmarks, frame_shape)   # 63-dim
    angle_vec = extract_angles(hand_landmarks)                   # 10-dim
    dist_vec  = extract_fingertip_distances(hand_landmarks)      # 8-dim
    thumb_vec = extract_thumb_position(hand_landmarks)           # 5-dim
    pinch_vec = extract_pinch_gaps(hand_landmarks)               # 7-dim
    return np.concatenate([lm_vec, angle_vec, dist_vec, thumb_vec, pinch_vec]) # 93-dim

def process_dataset(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # We will look at data/train and data/val combined to maximize MediaPipe extraction
    splits = ['train', 'val']
    
    classes = sorted(os.listdir(os.path.join(data_dir, 'train')))
    
    total_samples = 0
    
    for cls in classes:
        print(f"Processing class: {cls}...")
        samples = []
        
        # Collect images from both train and val for this class
        image_paths = []
        for split in splits:
            cls_dir = os.path.join(data_dir, split, cls)
            if os.path.exists(cls_dir):
                image_paths.extend(glob.glob(os.path.join(cls_dir, "*.jpg")) + 
                                   glob.glob(os.path.join(cls_dir, "*.jpeg")) + 
                                   glob.glob(os.path.join(cls_dir, "*.png")))
        
        extracted_count = 0
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None: continue
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:
                feat = get_feature_vector(results.multi_hand_landmarks[0], img.shape)
                samples.append(feat)
                extracted_count += 1
                
                if extracted_count >= 100:  # We just need 100 good samples per class
                    break
        
        print(f"  Extracted {len(samples)} valid landmarks out of {len(image_paths)} images.")
        
        if len(samples) > 0:
            np.save(os.path.join(output_dir, f"{cls}.npy"), np.array(samples))
            total_samples += len(samples)
        else:
            print(f"  [WARNING] Could not extract ANY landmarks for class {cls}!")
            
    print(f"\nFinished! Extracted {total_samples} total landmarks. Saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="Path to data/ folder containing train/ and val/")
    parser.add_argument("--out", default="data/features", help="Path to save .npy files")
    args = parser.parse_args()
    
    process_dataset(args.data, args.out)
