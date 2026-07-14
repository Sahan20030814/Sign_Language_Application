"""
train_lstm.py
----------------
Trains a free, open-source LSTM neural network (TensorFlow/Keras) to
classify a 30-frame sequence of MediaPipe hand-landmarks into one of
your vocabulary words.

Run this AFTER collect_data.py has produced samples in gesture_data/.

    python train_lstm.py
    python train_lstm.py --epochs 100
"""

import argparse
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from gesture_utils import DATA_DIR, MODEL_PATH, LABELS_PATH, SEQUENCE_LENGTH, FEATURE_SIZE


def load_dataset():
    words = sorted([w for w in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, w))])
    if not words:
        raise SystemExit(f"No data found in '{DATA_DIR}'. Run collect_data.py first.")

    X, y = [], []
    for label_idx, word in enumerate(words):
        folder = os.path.join(DATA_DIR, word)
        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        for fname in files:
            seq = np.load(os.path.join(folder, fname))
            if seq.shape == (SEQUENCE_LENGTH, FEATURE_SIZE):
                X.append(seq)
                y.append(label_idx)
        print(f"  {word}: {len(files)} samples")

    return np.array(X, dtype=np.float32), np.array(y), words


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(128, return_sequences=True, activation="tanh"),
        layers.Dropout(0.3),
        layers.LSTM(64, activation="tanh"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    print("Loading recorded gesture data...")
    X, y, words = load_dataset()
    print(f"\nTotal samples: {len(X)}  |  Classes: {words}")

    y_cat = to_categorical(y, num_classes=len(words))
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.15, random_state=42, stratify=y
    )

    model = build_model(len(words))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({str(i): w for i, w in enumerate(words)}, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved '{MODEL_PATH}' and '{LABELS_PATH}'.")
    print("You can now run: python app.py")


if __name__ == "__main__":
    main()
