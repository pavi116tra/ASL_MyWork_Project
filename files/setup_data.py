import os
import zipfile
import shutil
import random
from pathlib import Path

def setup_data():
    base_dir = Path(r"c:\Users\pavit\Downloads\Gesture\files")
    zip_path = base_dir / "asl-alphabet.zip"
    data_dir = base_dir / "data"
    
    if not zip_path.exists():
        print(f"Error: {zip_path} not found.")
        return

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)
        
    print("Extraction complete. Organizing files...")
    
    # Path where Kaggle extracts the train images
    extracted_train_dir = base_dir / "asl_alphabet_train" / "asl_alphabet_train"
    
    train_dest = data_dir / "train"
    val_dest = data_dir / "val"
    
    # Create target directories
    for path in [train_dest, val_dest]:
        path.mkdir(parents=True, exist_ok=True)
        
    if not extracted_train_dir.exists():
        print(f"Error: Could not find extracted directory {extracted_train_dir}")
        return

    classes = [d for d in os.listdir(extracted_train_dir) if os.path.isdir(extracted_train_dir / d)]
    print(f"Found {len(classes)} classes: {classes}")

    # For each class, split images into train and val
    split_ratio = 0.8
    for cls in classes:
        print(f"Processing {cls}...")
        cls_dir = extracted_train_dir / cls
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Limit to 150 images as requested by user ("Need 150+ images per sign")
        # to make training fast (20-25 mins) and avoid waiting for 87k images.
        # But wait, MobileNetV2 with 87k images takes hours. 
        # So sampling 200 images per class (160 train, 40 val) is perfect!
        random.seed(42)
        random.shuffle(images)
        
        # Take 200 images maximum
        images = images[:200]
        
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        # Create class folders in train and val
        (train_dest / cls).mkdir(parents=True, exist_ok=True)
        (val_dest / cls).mkdir(parents=True, exist_ok=True)
        
        # Move files
        for img in train_imgs:
            shutil.move(str(cls_dir / img), str(train_dest / cls / img))
            
        for img in val_imgs:
            shutil.move(str(cls_dir / img), str(val_dest / cls / img))
            
    print("Cleaning up extracted directories...")
    # Clean up the large unneeded extracted folders to save space
    shutil.rmtree(base_dir / "asl_alphabet_train", ignore_errors=True)
    shutil.rmtree(base_dir / "asl_alphabet_test", ignore_errors=True)
    
    print("Data setup complete!")

if __name__ == "__main__":
    setup_data()
