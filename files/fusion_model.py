"""
fusion_model.py
===============
Dual-Input Fusion Model for ASL Recognition.

Architecture:
  Branch A (Landmark branch):
    Input: 41 engineered features from feature_extractor.py
    Dense(256)→BN→Drop(0.3) → Dense(128)→BN→Drop(0.3) → Dense(64)

  Branch B (Image CNN branch):
    Input: 128x128x3 cropped hand image
    Conv2D(32)→Pool → Conv2D(64)→Pool → Conv2D(128)→Pool →
    Conv2D(256) → GlobalAveragePooling → Dense(128)→Drop(0.4)

  Fusion:
    Concatenate(A, B) → Dense(256)→BN→Drop(0.4) → Dense(128) → Dense(28, softmax)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_fusion_model(n_classes: int = 28,
                       feature_dim: int = 41,
                       img_size: int = 128) -> keras.Model:
    """
    Build the dual-input ASL fusion model.

    Args:
        n_classes   : number of output classes (default 28: A-Z + space + delete)
        feature_dim : size of engineered feature vector (default 41)
        img_size    : side length of square input image (default 128)

    Returns:
        Compiled keras.Model with two inputs:
          - 'landmark_input' : shape (feature_dim,)
          - 'image_input'    : shape (img_size, img_size, 3)
    """

    # ── Branch A: Landmark Feature Branch ────────────────────────────────────
    landmark_input = keras.Input(shape=(feature_dim,), name="landmark_input")

    x = layers.Dense(256, activation="relu", name="lm_dense1")(landmark_input)
    x = layers.BatchNormalization(name="lm_bn1")(x)
    x = layers.Dropout(0.3, name="lm_drop1")(x)

    x = layers.Dense(128, activation="relu", name="lm_dense2")(x)
    x = layers.BatchNormalization(name="lm_bn2")(x)
    x = layers.Dropout(0.3, name="lm_drop2")(x)

    x = layers.Dense(64, activation="relu", name="lm_dense3")(x)
    landmark_out = x  # shape: (64,)

    # ── Branch B: Image CNN Branch ────────────────────────────────────────────
    image_input = keras.Input(shape=(img_size, img_size, 3), name="image_input")

    y = layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      name="cnn_conv1")(image_input)
    y = layers.MaxPooling2D((2, 2), name="cnn_pool1")(y)

    y = layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                      name="cnn_conv2")(y)
    y = layers.MaxPooling2D((2, 2), name="cnn_pool2")(y)

    y = layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                      name="cnn_conv3")(y)
    y = layers.MaxPooling2D((2, 2), name="cnn_pool3")(y)

    y = layers.Conv2D(256, (3, 3), activation="relu", padding="same",
                      name="cnn_conv4")(y)
    y = layers.GlobalAveragePooling2D(name="cnn_gap")(y)

    y = layers.Dense(128, activation="relu", name="cnn_dense1")(y)
    y = layers.Dropout(0.4, name="cnn_drop1")(y)
    image_out = y  # shape: (128,)

    # ── Fusion Layer ──────────────────────────────────────────────────────────
    fused = layers.Concatenate(name="fusion_concat")([landmark_out, image_out])
    # shape: (64 + 128,) = (192,)

    z = layers.Dense(256, activation="relu", name="fusion_dense1")(fused)
    z = layers.BatchNormalization(name="fusion_bn1")(z)
    z = layers.Dropout(0.4, name="fusion_drop1")(z)

    z = layers.Dense(128, activation="relu", name="fusion_dense2")(z)

    output = layers.Dense(n_classes, activation="softmax", name="output")(z)

    # ── Build & Compile ───────────────────────────────────────────────────────
    model = keras.Model(
        inputs=[landmark_input, image_input],
        outputs=output,
        name="ASL_Fusion_Model"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_callbacks(checkpoint_path: str = "model/asl_fusion_model.h5"):
    """Return standard training callbacks."""
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=5,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]


def build_landmark_only_model(n_classes: int = 28,
                               feature_dim: int = 41) -> keras.Model:
    """
    Lightweight landmark-only model (same Branch A as fusion).
    Use this for fast inference when image crop is unavailable.
    """
    inp = keras.Input(shape=(feature_dim,), name="landmark_input")

    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=x, name="ASL_Landmark_Model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    model = build_fusion_model()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
