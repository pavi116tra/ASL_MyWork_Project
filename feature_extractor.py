"""
feature_extractor.py
====================
Advanced 41-dimensional feature extractor for ASL hand signs.
Replaces raw (x,y,z) coordinates with geometric, scale-invariant features.

Feature groups:
  1. Finger joint angles        (15) — bend at each joint
  2. Fingertip distances         (5) — distance from palm center
  3. Finger curl ratio           (5) — 0=curled, 1=extended
  4. Inter-fingertip distances  (10) — pairwise fingertip gaps
  5. Thumb opposition            (4) — thumb tip vs PIP joints
  6. Wrist angle                 (2) — roll + pitch
  ─────────────────────────────────
  Total                         41
"""

import numpy as np

# ── MediaPipe landmark indices ───────────────────────────────────────────────
WRIST         = 0
THUMB_CMC     = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP     = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP    = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP      = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP     = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20


# ── Low-level helpers ────────────────────────────────────────────────────────
def _p(lm, idx):
    """Return (x, y, z) of landmark at index."""
    l = lm[idx]
    return np.array([l.x, l.y, l.z], dtype=np.float32)

def _dist(lm, i, j):
    """Euclidean distance between landmarks i and j."""
    d = _p(lm, i) - _p(lm, j)
    return float(np.linalg.norm(d))

def _hand_size(lm):
    """Reference scale: wrist → middle-finger MCP distance."""
    return _dist(lm, WRIST, MIDDLE_MCP) + 1e-6

def _angle_at(lm, a, b, c):
    """
    Angle (degrees) at joint b formed by a→b→c.
    Returns value in [0, 1] (normalised by 180°).
    """
    v1 = _p(lm, a) - _p(lm, b)
    v2 = _p(lm, c) - _p(lm, b)
    n1 = np.linalg.norm(v1);  n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a))) / 180.0


# ── Feature group 1: Joint angles (15) ──────────────────────────────────────
def _joint_angles(lm):
    """
    15 angles — 3 joints per finger × 5 fingers.
    Normalised to [0, 1].  Straight finger ≈ 1.0, fully bent ≈ 0.0.
    """
    triplets = [
        # Thumb
        (WRIST,      THUMB_CMC, THUMB_MCP),
        (THUMB_CMC,  THUMB_MCP, THUMB_IP),
        (THUMB_MCP,  THUMB_IP,  THUMB_TIP),
        # Index
        (WRIST,      INDEX_MCP, INDEX_PIP),
        (INDEX_MCP,  INDEX_PIP, INDEX_DIP),
        (INDEX_PIP,  INDEX_DIP, INDEX_TIP),
        # Middle
        (WRIST,      MIDDLE_MCP, MIDDLE_PIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
        (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        # Ring
        (WRIST,      RING_MCP, RING_PIP),
        (RING_MCP,   RING_PIP, RING_DIP),
        (RING_PIP,   RING_DIP, RING_TIP),
        # Pinky
        (WRIST,      PINKY_MCP, PINKY_PIP),
        (PINKY_MCP,  PINKY_PIP, PINKY_DIP),
        (PINKY_PIP,  PINKY_DIP, PINKY_TIP),
    ]
    return np.array([_angle_at(lm, a, b, c) for a, b, c in triplets],
                    dtype=np.float32)


# ── Feature group 2: Fingertip distances (5) ────────────────────────────────
def _fingertip_distances(lm):
    """
    Distance from each fingertip to palm centre, normalised by hand size.
    Palm centre = mean of wrist + 5 MCP joints.
    """
    palm_idx = [WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    pts      = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in palm_idx])
    centre   = pts.mean(axis=0)
    scale    = _hand_size(lm)

    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    dists = []
    for t in tips:
        pt = _p(lm, t)
        dists.append(float(np.linalg.norm(pt - centre)) / scale)
    return np.array(dists, dtype=np.float32)


# ── Feature group 3: Finger curl ratio (5) ──────────────────────────────────
def _finger_curl_ratios(lm):
    """
    How curled each finger is.
    curl = tip-to-base straight distance / chain length of joints.
    0.0 = fully curled, 1.0 = fully extended.
    Key for: A/E/S/M/N/T group.
    """
    fingers = [
        [THUMB_CMC,  THUMB_MCP,  THUMB_IP,   THUMB_TIP],
        [INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP],
        [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
        [RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP],
        [PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP],
    ]
    curls = []
    for joints in fingers:
        straight = _dist(lm, joints[0], joints[-1])
        chain    = sum(_dist(lm, joints[i], joints[i+1])
                       for i in range(len(joints) - 1))
        curls.append(straight / (chain + 1e-6))
    return np.array(curls, dtype=np.float32)


# ── Feature group 4: Inter-fingertip distances (10) ─────────────────────────
def _inter_fingertip_distances(lm):
    """
    All C(5,2)=10 pairwise distances between fingertips.
    Normalised by hand size.
    Order: thumb-index, thumb-middle, thumb-ring, thumb-pinky,
           index-middle, index-ring, index-pinky,
           middle-ring, middle-pinky, ring-pinky.
    """
    tips  = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    scale = _hand_size(lm)
    dists = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dists.append(_dist(lm, tips[i], tips[j]) / scale)
    return np.array(dists, dtype=np.float32)


# ── Feature group 5: Thumb opposition (4) ───────────────────────────────────
def _thumb_opposition(lm):
    """
    Distance from thumb tip to each finger's PIP joint, normalised.
    Critical for A vs E vs S disambiguation:
      A → thumb beside fist, large distances to all PIPs
      S → thumb over index/middle, small distances
      E → fingers curled, thumb tucked under
    """
    scale = _hand_size(lm)
    pips  = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    return np.array([_dist(lm, THUMB_TIP, p) / scale for p in pips],
                    dtype=np.float32)


# ── Feature group 6: Wrist angle (2) ────────────────────────────────────────
def _wrist_angle(lm):
    """
    Roll and pitch of the hand from wrist orientation.
    Helps distinguish R vs U, K vs V, and hand tilt variants.
    Both values are normalised to [-1, 1].
    """
    dx = lm[MIDDLE_MCP].x - lm[WRIST].x
    dy = lm[MIDDLE_MCP].y - lm[WRIST].y
    dz = lm[MIDDLE_MCP].z - lm[WRIST].z
    roll  = float(np.arctan2(dy, dx)) / np.pi
    pitch = float(np.arctan2(dz, np.sqrt(dx**2 + dy**2))) / np.pi
    return np.array([roll, pitch], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def extract_advanced_features(landmarks):
    """
    Extract 41-dimensional advanced feature vector from MediaPipe landmarks.

    Args:
        landmarks: list/sequence of 21 MediaPipe NormalizedLandmark objects
                   (result.multi_hand_landmarks[0].landmark)

    Returns:
        numpy array of shape (41,), dtype float32

    Feature layout:
        [0:15]  — joint angles          (15)
        [15:20] — fingertip distances   ( 5)
        [20:25] — finger curl ratios    ( 5)
        [25:35] — inter-fingertip dists (10)
        [35:39] — thumb opposition      ( 4)
        [39:41] — wrist angle           ( 2)
    """
    lm = landmarks
    return np.concatenate([
        _joint_angles(lm),               # 15
        _fingertip_distances(lm),        #  5
        _finger_curl_ratios(lm),         #  5
        _inter_fingertip_distances(lm),  # 10
        _thumb_opposition(lm),           #  4
        _wrist_angle(lm),                #  2
    ]).astype(np.float32)                # 41 total


def feature_names():
    """Return human-readable names for each of the 41 features."""
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    joints  = ["mcp", "pip", "dip"]
    names = []
    # joint angles
    for f in fingers:
        for j in joints:
            names.append(f"angle_{f}_{j}")
    # fingertip distances
    for f in fingers:
        names.append(f"tip_dist_{f}")
    # curl ratios
    for f in fingers:
        names.append(f"curl_{f}")
    # inter-fingertip
    pairs = [(i, j) for i in range(5) for j in range(i+1, 5)]
    for i, j in pairs:
        names.append(f"ift_{fingers[i]}_{fingers[j]}")
    # thumb opposition
    for f in fingers[1:]:
        names.append(f"thumb_opp_{f}")
    # wrist
    names += ["wrist_roll", "wrist_pitch"]
    return names


if __name__ == "__main__":
    print(f"Feature count : {len(feature_names())}")
    print(f"Feature names :")
    for i, n in enumerate(feature_names()):
        print(f"  [{i:2d}] {n}")
