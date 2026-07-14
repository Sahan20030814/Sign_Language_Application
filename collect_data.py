"""
collect_data.py
------------------
Records YOUR OWN webcam samples for each word in the vocabulary. This is
how we get training data without needing to download any external
sign-language video dataset.

For every word, you will sign it in front of the webcam several times.
Each repetition is one "sequence" of SEQUENCE_LENGTH frames of hand
landmarks (extracted with MediaPipe), saved as a .npy file under:

    gesture_data/<word>/<sequence_number>.npy      shape = (30, 126)

HOW TO RUN
    python collect_data.py
    python collect_data.py --words hello thanks yes no       (custom subset)
    python collect_data.py --sequences 40                    (more samples per word)

CONTROLS WHILE RUNNING
    SPACE  -> start recording the next sequence for the current word
    N      -> skip to the next word early
    Q      -> quit
"""

import argparse
import os

import cv2
import numpy as np

from gesture_utils import (
    create_hands_detector,
    extract_landmarks,
    draw_landmarks,
    SEQUENCE_LENGTH,
    DATA_DIR,
    DEFAULT_WORDS,
)


def ensure_dirs(words):
    for w in words:
        os.makedirs(os.path.join(DATA_DIR, w), exist_ok=True)


def next_sequence_index(word):
    folder = os.path.join(DATA_DIR, word)
    existing = [int(f.split(".")[0]) for f in os.listdir(folder) if f.endswith(".npy")]
    return (max(existing) + 1) if existing else 0


def main():
    parser = argparse.ArgumentParser(description="Record webcam samples for word-level sign gestures.")
    parser.add_argument("--words", nargs="+", default=DEFAULT_WORDS,
                         help="Words to record. Defaults to the starter vocabulary.")
    parser.add_argument("--sequences", type=int, default=30,
                         help="How many repetitions to record per word.")
    parser.add_argument("--cam_index", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs(args.words)
    cap = cv2.VideoCapture(args.cam_index)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    with create_hands_detector() as hands:
        for word in args.words:
            start_idx = next_sequence_index(word)
            seq_num = 0
            print(f"\n=== Word: '{word}' ===  (have {start_idx} samples already)")

            while seq_num < args.sequences:
                # --- Wait for SPACE to begin this repetition ---
                waiting = True
                while waiting:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"Word: {word}  ({seq_num + 1}/{args.sequences})",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
                    cv2.putText(frame, "Press SPACE to record, N = next word, Q = quit",
                                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow("Collecting gesture data", frame)
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord(" "):
                        waiting = False
                    elif key == ord("n"):
                        seq_num = args.sequences  # break out of this word
                        waiting = False
                    elif key == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                if seq_num >= args.sequences:
                    break

                # --- Small countdown so you have time to get into position ---
                for count in (3, 2, 1):
                    ok, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, str(count), (frame.shape[1] // 2 - 20, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 200), 4)
                    cv2.imshow("Collecting gesture data", frame)
                    cv2.waitKey(500)

                # --- Record SEQUENCE_LENGTH frames ---
                sequence = []
                for _ in range(SEQUENCE_LENGTH):
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    draw_landmarks(frame, results)
                    landmarks, _ = extract_landmarks(results)
                    sequence.append(landmarks)

                    cv2.putText(frame, "RECORDING...", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.imshow("Collecting gesture data", frame)
                    cv2.waitKey(1)

                out_path = os.path.join(DATA_DIR, word, f"{start_idx + seq_num}.npy")
                np.save(out_path, np.array(sequence, dtype=np.float32))
                print(f"  saved {out_path}")
                seq_num += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone collecting. Now run: python train_lstm.py")


if __name__ == "__main__":
    main()
