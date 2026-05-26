import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from pathlib import Path
import datetime

def oversample_confused_classes(X, y, confused_labels, le, factor=3):
    try:
        hard_indices = np.where(
            np.isin(y, le.transform(confused_labels))
        )[0]
    except ValueError:
        return X, y

    if len(hard_indices) == 0:
        return X, y

    X_hard = np.tile(X[hard_indices], (factor, 1))
    y_hard = np.tile(y[hard_indices], factor)

    noise = np.random.normal(0, 0.01, X_hard.shape).astype(np.float32)
    X_hard += noise

    X_out = np.concatenate([X, X_hard])
    y_out = np.concatenate([y, y_hard])
    return X_out, y_out

def train_model(data_path="data/features", out_path_str="model"):
    out_path = Path(out_path_str)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading MediaPipe Landmark Dataset from {data_path}...")
    X, y = [], []
    for fname in os.listdir(data_path):
        if fname.endswith(".npy"):
            label = fname.replace(".npy", "")
            samples = np.load(os.path.join(data_path, fname))
            X.extend(samples)
            y.extend([label] * len(samples))

    if len(X) == 0:
        print("No data found! Run extract_landmarks.py first.")
        return

    X = np.array(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"Dataset Size: {len(X)} samples across {len(le.classes_)} classes.")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    X_train, y_train = oversample_confused_classes(
        X_train, y_train,
        confused_labels=['A', 'S', 'E', 'T', 'M', 'N', 'C', 'O'],
        le=le,
        factor=3
    )

    n_classes = len(le.classes_)
    
    print("Building Lightweight Dense Network (93-dim)...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(93,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    class ConfusedClassWeight(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch == 20: 
                hard = ['A','S','E','T','M','N', 'C', 'O']
                valid_hard = [h for h in hard if h in le.classes_]
                if not valid_hard: return
                
                hard_idx = le.transform(valid_hard)
                weights = {i: 1.0 for i in range(len(le.classes_))}
                for i in hard_idx:
                    weights[i] = 2.5 
                self.model.class_weight = weights 

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
        ConfusedClassWeight()
    ]

    print("\nTraining Model (this will be very fast!)...")
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=100, batch_size=32,
              callbacks=callbacks, verbose=1)

    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = out_path / f"asl_model_{timestamp}.h5"
    best_path  = out_path / "asl_dense_model_BEST.h5"
    
    model.save(str(model_path))       
    model.save(str(best_path))        

    np.save(out_path / f"labels_{timestamp}.npy", le.classes_)
    np.save(out_path / "label_classes.npy", le.classes_)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    log_path = out_path / "model_log.txt"
    with open(log_path, "a") as f:
        f.write(
            f"{timestamp} | "
            f"acc: {val_acc*100:.2f}% | "
            f"loss: {val_loss:.4f} | "
            f"samples: {len(X_train)} | "
            f"classes: {n_classes} | "
            f"file: {model_path.name}\n"
        )

    print("Generating Confusion Matrix...")
    y_pred = np.argmax(model.predict(X_val), axis=-1)
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Validation Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved versioned model -> {model_path}")
    print(f"Saved best pointer   -> {best_path}")
    print(f"Log updated -> {log_path}")

if __name__ == "__main__":
    train_model()
