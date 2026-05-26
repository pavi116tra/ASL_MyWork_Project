"""
ASL Sign Detector Application - MobileNetV2 Version
Real-time hand sign detection with MediaPipe + TensorFlow
"""

import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np
import os
from collections import deque
from datetime import datetime

class ASLDetector:
    """
    Real-time ASL sign detector using MobileNetV2
    """
    
    def __init__(self, model_path='asl_detector_mobilenetv2', confidence_threshold=0.7):
        """
        Initialize the ASL detector
        
        Args:
            model_path: Path to trained model (SavedModel or .h5)
            confidence_threshold: Minimum confidence to display prediction
        """
        print("🚀 Initializing ASL Sign Detector...")
        
        # Load model
        print(f"📦 Loading model from: {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"   Make sure model exists at: {model_path}")
            raise
        
        # Model parameters
        self.img_size = 224
        self.num_classes = 29
        self.confidence_threshold = confidence_threshold
        self.class_names = [chr(65 + i) for i in range(26)] + ["delete", "nothing", "space"]
        
        # MediaPipe hand detection
        print("🤚 Initializing MediaPipe Hands...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        print("✓ MediaPipe initialized")
        
        # Prediction smoothing (reduce flickering)
        self.prediction_buffer = deque(maxlen=5)
        self.sentence_buffer = deque(maxlen=10)
        
        print("\n✓ ASL Detector ready!")
        print(f"  Classes: {self.num_classes} (A-Z)")
        print(f"  Input size: {self.img_size}×{self.img_size}")
        print(f"  Confidence threshold: {self.confidence_threshold:.0%}")
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for MobileNetV2 input
        
        CRITICAL: Must match training preprocessing exactly!
        """
        # Resize to model input size
        img = cv2.resize(frame, (self.img_size, self.img_size))
        
        # Convert BGR to RGB (OpenCV uses BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        img = img.astype('float32') / 255.0
        
        # Apply MobileNetV2-specific preprocessing
        # (centering and scaling)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict_sign(self, frame):
        """
        Predict ASL sign from frame
        
        Returns:
            dict: {
                'sign': str,
                'confidence': float (0-1),
                'top_3': list of (sign, confidence) tuples,
                'all_confidences': dict of all predictions
            }
        """
        # Preprocess
        preprocessed = self.preprocess_frame(frame)
        
        # Predict (disable verbose to avoid spam)
        predictions = self.model.predict(preprocessed, verbose=0)[0]
        
        # Get top prediction
        pred_idx = np.argmax(predictions)
        confidence = predictions[pred_idx]
        pred_sign = self.class_names[pred_idx]
        
        # Get top 3
        top_3_idx = np.argsort(predictions)[-3:][::-1]
        top_3 = [(self.class_names[i], float(predictions[i])) for i in top_3_idx]
        
        return {
            'sign': pred_sign,
            'confidence': float(confidence),
            'top_3': top_3,
            'all_predictions': {self.class_names[i]: float(predictions[i]) 
                               for i in range(len(self.class_names))}
        }
    
    def smooth_prediction(self, prediction):
        """
        Smooth predictions using buffer to reduce flickering
        """
        self.prediction_buffer.append(prediction['sign'])
        
        if len(self.prediction_buffer) > 0:
            # Return most common sign in buffer
            from collections import Counter
            smoothed_sign = Counter(self.prediction_buffer).most_common(1)[0][0]
            return smoothed_sign
        
        return prediction['sign']
    
    def draw_prediction(self, frame, prediction, hand_landmarks=None):
        """
        Draw prediction and hand landmarks on frame
        """
        h, w, c = frame.shape
        
        # Draw hand landmarks if detected
        if hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Draw prediction box
        confidence = prediction['confidence']
        sign = prediction['sign']
        
        # Color based on confidence
        if confidence > 0.95:
            color = (0, 255, 0)  # Green - very confident
        elif confidence > 0.80:
            color = (0, 165, 255)  # Orange - confident
        else:
            color = (0, 0, 255)  # Red - less confident
        
        # Draw background box for text
        cv2.rectangle(frame, (10, 30), (250, 150), (0, 0, 0), -1)
        
        # Draw prediction text
        cv2.putText(frame, f"Sign: {sign}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Draw top 3 predictions
        cv2.putText(frame, "Top 3:", (20, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        top_3_text = ", ".join([f"{s}({c:.0%})" for s, c in prediction['top_3']])
        cv2.putText(frame, top_3_text, (80, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def draw_sentence(self, frame, sentence):
        """
        Draw built sentence at bottom of frame
        """
        h, w, c = frame.shape
        
        # Draw background
        cv2.rectangle(frame, (10, h-50), (w-10, h-10), (0, 0, 0), -1)
        
        # Draw sentence
        cv2.putText(frame, f"Sentence: {sentence}", (20, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def add_sign_to_sentence(self, sign):
        """
        Add sign to sentence buffer (for continuous text building)
        """
        if len(self.sentence_buffer) == 0 or self.sentence_buffer[-1] != sign:
            self.sentence_buffer.append(sign)
    
    def get_sentence(self):
        """
        Get current sentence from buffer
        """
        return ''.join(self.sentence_buffer)
    
    def clear_sentence(self):
        """
        Clear sentence buffer (for space/delete in real app)
        """
        self.sentence_buffer.clear()
    
    def run_webcam(self):
        """
        Run real-time ASL detection from webcam
        
        Controls:
        - SPACE: Add space to sentence
        - 'C': Clear sentence
        - 'Q': Quit
        """
        print("\n" + "="*70)
        print("STARTING WEBCAM - ASL SIGN DETECTION")
        print("="*70)
        print("\nControls:")
        print("  SPACE: Add space to sentence")
        print("  C: Clear sentence")
        print("  Q: Quit")
        print("\n" + "="*70)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Webcam opened")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Predict sign
            prediction = self.predict_sign(frame)
            
            # Only add to sentence if confident enough
            if prediction['confidence'] > self.confidence_threshold:
                smoothed = self.smooth_prediction(prediction)
                self.add_sign_to_sentence(smoothed)
            
            # Draw
            frame = self.draw_prediction(frame, prediction, results.multi_hand_landmarks)
            frame = self.draw_sentence(frame, self.get_sentence())
            
            # Display
            cv2.imshow('ASL Sign Detector', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✓ Exiting...")
                break
            elif key == ord('c'):
                self.clear_sentence()
                print("✓ Sentence cleared")
            elif key == ord(' '):
                self.sentence_buffer.append(' ')
                print("✓ Space added")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Webcam closed")
    
    def test_image(self, image_path):
        """
        Test on a single image file
        
        Args:
            image_path: Path to image file
        """
        print(f"\n📸 Testing on image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        # Load image
        frame = cv2.imread(image_path)
        
        # Predict
        prediction = self.predict_sign(frame)
        
        print(f"\n✓ Prediction Results:")
        print(f"  Sign: {prediction['sign']}")
        print(f"  Confidence: {prediction['confidence']:.2%}")
        print(f"\n  Top 3:")
        for sign, conf in prediction['top_3']:
            print(f"    {sign}: {conf:.2%}")
        
        # Draw and show
        frame = self.draw_prediction(frame, prediction)
        cv2.imshow('ASL Sign Test', frame)
        print("\n  Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='ASL Sign Detector')
    parser.add_argument('--model', type=str, default='asl_detector_mobilenetv2',
                       help='Path to model')
    parser.add_argument('--test', type=str, default=None,
                       help='Test image path (optional)')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ASLDetector(model_path=args.model, 
                          confidence_threshold=args.confidence)
    
    # Test or run webcam
    if args.test:
        detector.test_image(args.test)
    else:
        detector.run_webcam()

if __name__ == '__main__':
    main()
