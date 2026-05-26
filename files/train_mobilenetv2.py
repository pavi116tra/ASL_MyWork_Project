"""
ASL Sign Detector - MobileNetV2 Transfer Learning Training Script
Complete implementation with frozen phase + fine-tuning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

class ASLMobileNetV2Trainer:
    def __init__(self, data_path="data", model_name="asl_detector_mobilenetv2"):
        self.data_path = data_path
        self.model_name = model_name
        self.img_size = 224
        self.num_classes = 29  # A-Z + space, delete, nothing
        self.batch_size = 32
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def validate_dataset(self):
        """
        CRITICAL: Verify dataset is properly balanced before training
        """
        print("\n" + "="*70)
        print("STEP 0: DATASET VALIDATION")
        print("="*70)
        
        # Check train folder
        train_path = os.path.join(self.data_path, 'train')
        val_path = os.path.join(self.data_path, 'val')
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data folder not found: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data folder not found: {val_path}")
        
        # Count images per class
        train_counts = {}
        val_counts = {}
        
        print("\nTraining Set Distribution:")
        print(f"{'Class':<10} {'Count':<15} {'Status':<20}")
        print("-"*45)
        
        for folder in sorted(os.listdir(train_path)):
            if os.path.isdir(os.path.join(train_path, folder)):
                count = len([f for f in os.listdir(os.path.join(train_path, folder)) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))])
                train_counts[folder] = count
                
                status = "✓ GOOD" if count >= 120 else "⚠️ LOW" if count >= 80 else "❌ CRITICAL"
                print(f"{folder:<10} {count:<15} {status:<20}")
        
        print("\nValidation Set Distribution:")
        print(f"{'Class':<10} {'Count':<15} {'Status':<20}")
        print("-"*45)
        
        for folder in sorted(os.listdir(val_path)):
            if os.path.isdir(os.path.join(val_path, folder)):
                count = len([f for f in os.listdir(os.path.join(val_path, folder)) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))])
                val_counts[folder] = count
                
                status = "✓ GOOD" if count >= 25 else "⚠️ LOW" if count >= 15 else "❌ CRITICAL"
                print(f"{folder:<10} {count:<15} {status:<20}")
        
        # Check balance
        if train_counts:
            train_min = min(train_counts.values())
            train_max = max(train_counts.values())
            train_ratio = train_max / train_min if train_min > 0 else float('inf')
            
            print(f"\n📊 Balance Analysis (Training):")
            print(f"   Min: {train_min}, Max: {train_max}")
            print(f"   Imbalance ratio: {train_ratio:.2f}x")
            
            if train_ratio > 2.0:
                print("   ⚠️  WARNING: Dataset is imbalanced!")
                print("   Recommendation: Collect more data for underrepresented classes")
            else:
                print("   ✓ Dataset is well-balanced")
        
        print(f"\n✓ Total training samples: {sum(train_counts.values())}")
        print(f"✓ Total validation samples: {sum(val_counts.values())}")
        print(f"✓ Classes found: {len(train_counts)}")
        
        if len(train_counts) < 29:
            print(f"⚠️  WARNING: Expected 29 classes, found {len(train_counts)}")
        
        return train_counts, val_counts
    
    def prepare_data_generators(self):
        """
        Create optimized data generators for ASL hand signs
        """
        print("\n" + "="*70)
        print("STEP 1: DATA AUGMENTATION & GENERATORS")
        print("="*70)
        
        # Training augmentation - optimized for hand signs
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            
            # Hand position variations
            rotation_range=30,           # Different hand orientations
            width_shift_range=0.2,       # Hand moving left/right
            height_shift_range=0.2,      # Hand moving up/down
            zoom_range=0.2,              # Hand closer/further from camera
            
            # Environmental variations
            brightness_range=[0.7, 1.3], # Different lighting conditions
            
            # Geometric transformations
            shear_range=0.1,             # Slight perspective shifts
            horizontal_flip=True,        # Mirror hand (train both left/right)
            
            # Filling strategy
            fill_mode='nearest'
        )
        
        # Validation - no augmentation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        print("\n✓ Training augmentation enabled:")
        print("   • Rotation: ±30°")
        print("   • Shift: ±20% horizontal & vertical")
        print("   • Zoom: ±20%")
        print("   • Brightness: 0.7x to 1.3x")
        print("   • Horizontal flip: Yes")
        
        # Load generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_path, 'train'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        val_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_path, 'val'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"\n✓ Data loaded:")
        print(f"   Training batches: {len(train_generator)}")
        print(f"   Validation batches: {len(val_generator)}")
        
        return train_generator, val_generator
    
    def compute_class_weights(self, train_generator):
        """
        Calculate weights to handle any remaining class imbalance
        """
        print("\n" + "="*70)
        print("STEP 2: COMPUTING CLASS WEIGHTS")
        print("="*70)
        
        class_indices = train_generator.class_indices
        num_classes = len(class_indices)
        
        # Count samples per class
        samples_per_class = np.zeros(num_classes)
        
        # Iterate through batches to count
        steps = 0
        for batch_x, batch_y in train_generator:
            for i in range(num_classes):
                samples_per_class[i] += np.sum(batch_y[:, i])
            
            steps += 1
            if steps >= len(train_generator):
                break
        
        # Compute weights
        total_samples = np.sum(samples_per_class)
        class_weights = {}
        
        print(f"\nClass Weights (to balance training):")
        print(f"{'Class':<10} {'Samples':<15} {'Weight':<12}")
        print("-"*37)
        
        for class_idx, (class_name, idx) in enumerate(class_indices.items()):
            if samples_per_class[idx] > 0:
                weight = total_samples / (num_classes * samples_per_class[idx])
            else:
                weight = 1.0
            
            class_weights[idx] = weight
            print(f"{class_name:<10} {int(samples_per_class[idx]):<15} {weight:>10.3f}")
        
        print(f"\n✓ Total training samples: {int(total_samples)}")
        print("✓ Class weights will prevent majority class bias")
        
        return class_weights
    
    def build_mobilenetv2_model(self):
        """
        Build MobileNetV2 with transfer learning
        """
        print("\n" + "="*70)
        print("STEP 3: BUILDING MOBILENETV2 MODEL")
        print("="*70)
        
        # Load pre-trained MobileNetV2 (ImageNet weights)
        print("\n⏳ Loading MobileNetV2 with ImageNet weights...")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,           # Remove ImageNet classification head
            weights='imagenet'           # Pre-trained weights
        )
        
        # Freeze base model (don't retrain on ImageNet weights)
        base_model.trainable = False
        print("✓ Base model frozen (will not update ImageNet weights)")
        
        # Build custom classification head
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.img_size, self.img_size, 3)),
            
            # MobileNetV2 feature extractor
            base_model,
            
            # Global average pooling (converts spatial dimensions to single vector)
            layers.GlobalAveragePooling2D(),
            
            # Dense layers for ASL classification
            layers.Dense(512, 
                        kernel_regularizer=keras.regularizers.l2(0.001),
                        bias_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(256,
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(128,
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Output layer (29 ASL signs)
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        print("\n✓ Model architecture:")
        print(f"   • Base: MobileNetV2 (3.5M parameters)")
        print(f"   • Head: Custom dense layers (512→256→128→29)")
        print(f"   • Regularization: L2 + Dropout + BatchNorm")
        print(f"   • Output: 29 classes")
        
        # Compile for frozen phase
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_acc')]
        )
        
        print("\n✓ Model compiled")
        print(f"   • Optimizer: Adam (lr=0.0001)")
        print(f"   • Loss: Categorical Crossentropy")
        print(f"   • Metrics: Accuracy, Top-3 Accuracy")
        
        return model, base_model
    
    def train_phase1_frozen(self, model, train_gen, val_gen, class_weights):
        """
        Phase 1: Train with frozen base model
        Only custom head learns
        """
        print("\n" + "="*70)
        print("STEP 4A: PHASE 1 - FROZEN BASE TRAINING")
        print("="*70)
        print("\nTraining custom classification head while MobileNetV2 base is frozen")
        print("This learns ASL-specific patterns on top of ImageNet features\n")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                f'{self.model_name}_phase1_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        history1 = model.fit(
            train_gen,
            steps_per_epoch=len(train_gen),
            epochs=10,
            validation_data=val_gen,
            validation_steps=len(val_gen),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Phase 1 complete")
        print(f"   Final accuracy: {history1.history['accuracy'][-1]:.2%}")
        print(f"   Final val accuracy: {history1.history['val_accuracy'][-1]:.2%}")
        
        return history1
    
    def train_phase2_finetune(self, model, base_model, train_gen, val_gen, class_weights):
        """
        Phase 2: Fine-tune base model
        Unfreeze last layers and train with very low learning rate
        """
        print("\n" + "="*70)
        print("STEP 4B: PHASE 2 - BASE MODEL FINE-TUNING")
        print("="*70)
        
        # Unfreeze last N layers of base model
        num_unfreeze = 10
        base_model.trainable = True
        for layer in base_model.layers[:-num_unfreeze]:
            layer.trainable = False
        
        print(f"\nUnfreezing last {num_unfreeze} layers of MobileNetV2")
        print(f"Total trainable parameters: {model.count_params():,}")
        
        # Recompile with much lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),  # 10x lower
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_acc')]
        )
        
        print("Recompiled with lower learning rate (0.00001)")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                f'{self.model_name}_phase2_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=0.000001,
                verbose=1
            )
        ]
        
        print("\nTraining with unfrozen base model...")
        history2 = model.fit(
            train_gen,
            steps_per_epoch=len(train_gen),
            epochs=20,
            initial_epoch=0,
            validation_data=val_gen,
            validation_steps=len(val_gen),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Phase 2 complete")
        print(f"   Final accuracy: {history2.history['accuracy'][-1]:.2%}")
        print(f"   Final val accuracy: {history2.history['val_accuracy'][-1]:.2%}")
        
        return history2
    
    def evaluate_model(self, model, val_gen):
        """
        Comprehensive model evaluation
        """
        print("\n" + "="*70)
        print("STEP 5: MODEL EVALUATION")
        print("="*70)
        
        val_loss, val_acc, val_top3 = model.evaluate(val_gen, verbose=0)
        
        print(f"\nValidation Results:")
        print(f"   Loss: {val_loss:.4f}")
        print(f"   Accuracy: {val_acc:.2%}")
        print(f"   Top-3 Accuracy: {val_top3:.2%}")
        
        if val_acc > 0.90:
            print("\n✓ EXCELLENT! Model is ready for deployment")
        elif val_acc > 0.85:
            print("\n✓ GOOD! Model is working well")
        elif val_acc > 0.75:
            print("\n⚠️  FAIR! Model works but could be better")
            print("   Recommendation: Collect more training data")
        else:
            print("\n❌ POOR! Model is not learning well")
            print("   Possible causes:")
            print("   • Dataset too small or imbalanced")
            print("   • Labels incorrect")
            print("   • Data preprocessing issue")
    
    def save_model(self, model):
        """
        Save model in multiple formats
        """
        print("\n" + "="*70)
        print("STEP 6: SAVING MODEL")
        print("="*70)
        
        # SavedModel format (recommended)
        print(f"\nSaving as SavedModel: {self.model_name}/")
        model.save(self.model_name)
        
        # H5 format (compatibility)
        print(f"Saving as H5: {self.model_name}.h5")
        model.save(f'{self.model_name}.h5')
        
        print("\n✓ Model saved successfully!")
        print(f"\nTo load model later:")
        print(f"   model = tf.keras.models.load_model('{self.model_name}')")
        print(f"   # or")
        print(f"   model = tf.keras.models.load_model('{self.model_name}.h5')")
    
    def run_complete_pipeline(self):
        """
        Execute complete training pipeline
        """
        print("\n")
        print("+" + "="*68 + "+")
        print("|" + " "*10 + "ASL SIGN DETECTOR - MobileNetV2 TRAINING PIPELINE" + " "*9 + "|")
        print("+" + "="*68 + "+")
        print(f"\nTimestamp: {self.timestamp}")
        
        try:
            # Step 0: Validate dataset
            train_counts, val_counts = self.validate_dataset()
            
            # Step 1: Prepare data
            train_gen, val_gen = self.prepare_data_generators()
            
            # Step 2: Compute class weights
            class_weights = self.compute_class_weights(train_gen)
            
            # Step 3: Build model
            model, base_model = self.build_mobilenetv2_model()
            
            # Step 4: Two-phase training
            print("\n" + "="*70)
            print("TRAINING PHASES")
            print("="*70)
            
            history1 = self.train_phase1_frozen(model, train_gen, val_gen, class_weights)
            history2 = self.train_phase2_finetune(model, base_model, train_gen, val_gen, class_weights)
            
            # Step 5: Evaluate
            self.evaluate_model(model, val_gen)
            
            # Step 6: Save
            self.save_model(model)
            
            # Summary
            print("\n" + "="*70)
            print("TRAINING COMPLETE!")
            print("="*70)
            print("\n✓ Model saved and ready for deployment")
            print(f"\nNext steps:")
            print(f"1. Test the model: python test_asl_model.py")
            print(f"2. Run the app: python asl_app.py")
            print(f"3. Show G sign - should predict G, not B!")
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            print("\nTroubleshooting:")
            print("• Verify data folder structure: data/train/A/, data/train/B/, etc.")
            print("• Ensure all images are valid (not corrupted)")
            print("• Check that you have 80-200 images per class")
            raise

if __name__ == "__main__":
    # Initialize trainer
    trainer = ASLMobileNetV2Trainer(
        data_path="data",  # Update if different
        model_name="asl_detector_mobilenetv2"
    )
    
    # Run complete pipeline
    trainer.run_complete_pipeline()
