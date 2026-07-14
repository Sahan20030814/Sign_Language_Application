"""
make_demo_model.py
---------------------
Creates a PLACEHOLDER 'word_model.h5' + 'word_labels.json' using
synthetically generated (fake) landmark sequences, so you can test the
whole app - camera, power switch, mute, speak, copy, animations, etc -
BEFORE you invest time recording your own real gestures.

⚠️ This demo model does NOT actually recognise real hand signs. It was
trained on made-up numbers, not your hands. Signing in front of the
camera will still trigger word recognition (so you can test the full
pipeline end-to-end), but which word appears will be close to random. For
REAL accuracy, run collect_data.py + train_lstm.py with your own
recorded gestures.

Run:
    python make_demo_model.py
"""

import json
import numpy as np

from tensorflow.keras.utils import to_categorical
from gesture_utils import SEQUENCE_LENGTH, FEATURE_SIZE, MODEL_PATH, LABELS_PATH, DEFAULT_WORDS
from train_lstm import build_model


def make_synthetic_dataset(words, samples_per_word=40):
    """
    Generates fake-but-structured landmark sequences. To make the demo feel
    more responsive when you wave a REAL hand at the camera, each sample
    mimics the shape of real MediaPipe output: one 63-length hand block has
    realistic 0..1-ish coordinates, and the other hand's block is exactly
    zero (like a hand that isn't visible) - the same pattern a real single
    hand produces.
    """
    rng = np.random.default_rng(42)
    half = FEATURE_SIZE // 2  # 63
    X, y = [], []
    for idx, _ in enumerate(words):
        active_slot = idx % 2  # alternate which hand-block classes "live" in
        centroid = rng.uniform(0.25, 0.75, size=half)
        drift = rng.uniform(-0.03, 0.03, size=half)
        for _ in range(samples_per_word):
            active = np.stack([
                np.clip(centroid + drift * t + rng.normal(0, 0.05, size=half), 0.0, 1.0)
                for t in range(SEQUENCE_LENGTH)
            ]).astype(np.float32)
            zeros = np.zeros((SEQUENCE_LENGTH, half), dtype=np.float32)
            seq = np.concatenate([active, zeros], axis=1) if active_slot == 0 else \
                  np.concatenate([zeros, active], axis=1)
            X.append(seq)
            y.append(idx)
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    words = DEFAULT_WORDS
    print(f"Generating synthetic data for: {words}")
    X, y = make_synthetic_dataset(words)
    y_cat = to_categorical(y, num_classes=len(words))

    model = build_model(len(words))
    model.fit(X, y_cat, epochs=15, batch_size=16, verbose=1)
    model.save(MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({str(i): w for i, w in enumerate(words)}, f, ensure_ascii=False, indent=2)

    print(f"\nDemo model saved: {MODEL_PATH}, {LABELS_PATH}")
    print("You can now run: python app.py")
    print("Remember: this model is for testing the APP only, not real sign accuracy.")


if __name__ == "__main__":
    main()
