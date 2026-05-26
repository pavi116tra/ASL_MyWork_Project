import tensorflow as tf
import os
import numpy as np

def evaluate_accuracy():
    model_path = 'asl_detector_mobilenetv2'
    val_dir = 'data/val'
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model {model_path} not found. Wait for training to finish.")
        return
        
    print(f"🚀 Loading ASL MobileNetV2 Model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    classes = [chr(65 + i) for i in range(26)] + ["delete", "nothing", "space"]
    
    total_correct = 0
    total_images = 0
    sign_stats = {c: {'correct': 0, 'total': 0, 'high_conf': 0} for c in classes}
    
    print(f"\n📊 Starting Per-Sign Accuracy Evaluation on Validation Set...")
    print("="*60)
    
    # Process each class
    for cls in classes:
        cls_dir = os.path.join(val_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.png'))]
        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            
            # Preprocess matching the app exactly
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = img_array.astype('float32') / 255.0
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            pred = model.predict(img_array, verbose=0)[0]
            pred_idx = np.argmax(pred)
            pred_conf = pred[pred_idx]
            pred_class = classes[pred_idx]
            
            sign_stats[cls]['total'] += 1
            total_images += 1
            
            if pred_class == cls:
                sign_stats[cls]['correct'] += 1
                total_correct += 1
                if pred_conf > 0.85:
                    sign_stats[cls]['high_conf'] += 1
                    
        # Print progress per class
        total = sign_stats[cls]['total']
        if total > 0:
            acc = sign_stats[cls]['correct'] / total
            print(f"Sign {cls:>7}: {acc:7.1%} ({sign_stats[cls]['correct']}/{total})")

    # Final Report Output matching the user's Prompt Guide
    overall_acc = total_correct / total_images if total_images > 0 else 0
    
    high_conf_total = sum(stats['high_conf'] for stats in sign_stats.values())
    high_conf_pct = high_conf_total / total_images if total_images > 0 else 0
    
    error_rate = 1.0 - overall_acc
    
    # Sort classes by accuracy
    class_accs = [(c, stats['correct']/stats['total'] if stats['total'] > 0 else 0) for c, stats in sign_stats.items()]
    class_accs.sort(key=lambda x: x[1], reverse=True)
    
    best_signs = [c for c, acc in class_accs if acc >= 0.95]
    worst_signs = [c for c, acc in class_accs if acc <= 0.75 and acc > 0]
    
    print("\n" + "="*60)
    print("🎯 ACCURACY REPORTING:")
    print(f"My ASL detector achieved:")
    print(f"- Overall Accuracy: {overall_acc:.0%}")
    print(f"- High Confidence (>85%): {high_conf_pct:.0%} of predictions")
    print(f"- Low Error Rate (<15%): {error_rate < 0.15} ({error_rate:.0%})")
    print(f"- Best Signs: {', '.join(best_signs[:5])} (>95%)")
    print(f"- Worst Signs: {', '.join(worst_signs[:5])} (<=75%)")
    print("="*60)

if __name__ == "__main__":
    evaluate_accuracy()
