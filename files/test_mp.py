import cv2
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

img_path = r'c:\Users\pavit\Downloads\Gesture\files\data\train\A\A1.jpg'
if not os.path.exists(img_path):
    print("File not found")
else:
    img = cv2.imread(img_path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            print("Hand detected!")
        else:
            print("No hand detected.")
