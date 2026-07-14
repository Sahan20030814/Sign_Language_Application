"""
gesture_utils.py
------------------
Shared constants + MediaPipe hand-landmark extraction helpers.

We use MediaPipe ONLY as a hand-keypoint detector (21 points per hand).
The actual "intelligence" - recognising WHICH WORD a sequence of hand
movements means - is a separate LSTM neural network that WE train
ourselves (see train_lstm.py), on data WE record ourselves
(see collect_data.py). MediaPipe never decides what word it is; it only
finds where the hand/fingers are in each frame.
"""

import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ----------------------------------------------------------------------------
SEQUENCE_LENGTH = 30              # frames captured per gesture (~1 second at 30fps)
NUM_LANDMARKS = 21                # MediaPipe hand landmark points
COORDS_PER_LANDMARK = 3           # x, y, z
FEATURES_PER_HAND = NUM_LANDMARKS * COORDS_PER_LANDMARK   # 63
FEATURE_SIZE = FEATURES_PER_HAND * 2                      # 126 (two hands)

DATA_DIR = "gesture_data"
MODEL_PATH = "word_model.h5"
LABELS_PATH = "word_labels.json"

# Starter vocabulary. Add/remove words here, then re-run collect_data.py
# and train_lstm.py to include your changes.
DEFAULT_WORDS = [
    "hello", "thanks", "yes", "no", "please",
    "help", "water", "food", "stop", "love",
]


def create_hands_detector(static_image_mode=False):
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


def extract_landmarks(results):
    """
    Converts a MediaPipe Hands result into a fixed-length (126,) feature
    vector: [left_hand(63), right_hand(63)]. Any hand not visible is
    zero-filled. Also returns whether at least one hand was detected,
    which the app uses to make sure NOTHING is recognised when the
    person's hands are not in frame.
    """
    left = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    right = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    hand_seen = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32).flatten()
            side = handed.classification[0].label  # 'Left' or 'Right'
            hand_seen = True
            if side == "Left":
                left = coords
            else:
                right = coords

    return np.concatenate([left, right]), hand_seen


def draw_landmarks(frame, results):
    """Draws the hand skeleton onto the frame (nice visual feedback for the user)."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
