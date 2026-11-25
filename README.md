# SignBridge v2.0 — Two-Way Sign Language ↔ Speech Communication

SignBridge is a free, open-source Windows desktop application that helps people who cannot speak or hear communicate with each other in real time. It works in **two directions**:

| Mode | Direction | What it does |
|---|---|---|
| **Mode 1** | Sign Language → Speech | Watches a person's hand signs through a webcam, recognises whole **words**, builds them into a sentence, and speaks the sentence out loud. |
| **Mode 2** *(new in v2.0)* | Speech → Sign Language | Listens to a microphone **or** plays an audio file, converts the speech to text, and shows each recognised word as a **hand gesture performed by an on-screen human avatar**. |

Everything runs on your own PC. Only the speech-to-text step in Mode 2 uses the internet (see [Privacy](#16-privacy)).

---

## Table of Contents

1. [What's new in v2.0](#1-whats-new-in-v20)
2. [How it works](#2-how-it-works)
3. [Download & run (no Python needed)](#3-download--run-no-python-needed)
4. [System requirements](#4-system-requirements)
5. [Developer setup](#5-developer-setup)
6. [Quick start: run with the demo model](#6-quick-start-run-with-the-demo-model)
7. [Training your own real model](#7-training-your-own-real-model)
8. [Using the app](#8-using-the-app)
9. [Project structure](#9-project-structure)
10. [Configuration & tuning](#10-configuration--tuning)
11. [Building the standalone Windows release](#11-building-the-standalone-windows-release)
12. [Publishing to GitHub](#12-publishing-to-github)
13. [Troubleshooting](#13-troubleshooting)
14. [Honest limitations](#14-honest-limitations)
15. [Contributing](#15-contributing)
16. [Privacy](#16-privacy)
17. [Changelog](#17-changelog)
18. [License](#18-license)

---

## 1. What's new in v2.0

**New: Speech → Sign Language mode**

- A **🔁 Mode** button in the header switches between the two modes.
- **Microphone input** — pick any working microphone from an automatically populated dropdown. The mic starts *muted*; unmute it (with **⏻ Power ON**) to start listening.
- **Audio-file input** — browse for a `.wav` file, then play, pause, seek on a timeline, skip ±5 seconds, and change playback speed (0.5x – 2x).
- **Live transcript box** with a **Copy** button. Transcription of audio files streams in progressively (roughly 8-second chunks) with a progress bar, so the avatar can start signing before the whole file is processed.
- **Human avatar** (head, hair, ears, eyes, nose, mouth, neck, formal long-sleeve shirt, arms). The body stays still; only the **hands** move. At rest the hands are folded together. While signing, each hand is driven by a **real recorded MediaPipe hand-landmark sequence** for that word, not a synthetic animation.
- **▶ Play Signs / ⏸ Pause Signs** to step through the transcript word by word at your own pace.
- While an audio file is playing, the avatar **signs in sync with playback** automatically, including after seeking, skipping, or changing speed.
- The webcam is **released while in Speech → Sign mode** and reconnects automatically when you switch back.
- Switching mode or input source cleanly resets the path, timeline, speed and transcript, and discards any transcription still running in the background.

**Carried over from v1.0:** word-level recognition with a custom-trained LSTM, automatic speech output, power/mute/speak/copy/space/backspace/clear controls, responsive layout and fullscreen (F11).

---

## 2. How it works

### Mode 1 — Sign → Speech

Recognising a signed **word** requires tracking motion over about a second, not just one photo. SignBridge therefore splits the problem into two free, open-source pieces:

1. **[MediaPipe Hands](https://github.com/google/mediapipe)** finds 21 landmark points per hand in every camera frame (up to two hands, giving 126 numbers per frame). It does **not** know which word is being signed; it only locates the hand's shape.
2. A **30-frame sequence** of those landmarks (~1 second) is fed into an **LSTM neural network that this project trains itself** (`train_lstm.py`). That network is the actual "intelligence" that predicts the word.
3. When the prediction confidence is high enough, the word is committed to the sentence and, unless muted, spoken with the offline text-to-speech engine (**pyttsx3**).

Training data comes from **your own webcam recordings** (`collect_data.py`). There is no pretrained word-level model to download.

### Mode 2 — Speech → Sign

1. Speech is captured from a microphone, or read from a `.wav` file.
2. It is converted to text with the **SpeechRecognition** library using Google's free web speech API (**internet required**).
3. Each recognised word that exists in the model's vocabulary (`word_labels.json`) is looked up in `gesture_data/<word>/*.npy` — the same recordings used for training.
4. The avatar's hands are drawn from the recorded landmark sequence for that word.

---

## 3. Download & run (no Python needed)

1. Open the **[Releases](../../releases)** page of this repository.
2. Download the latest **`SignBridge_v2.0_windows.zip`** (under *Assets*).
3. **Extract the whole zip** to a folder of your choice (right-click → *Extract All…*). Do not run anything from inside the zip.
4. Open the extracted folder and double-click **`SignBridge.exe`**.
   - The first launch can take a few extra seconds. That is normal.
   - Windows may show **"Windows protected your PC"** (SmartScreen) because this is an unsigned hobby/student project. Click **More info → Run anyway**.
5. Allow the **camera** permission (Mode 1) and the **microphone** permission (Mode 2) if Windows asks.

> ⚠️ **Keep the whole extracted folder together.** `SignBridge.exe` needs the `python_embed/`, `gesture_data/`, `resources/`, `word_model.h5`, `word_labels.json` and the `.py` files next to it. Do not move the `.exe` on its own and do not delete anything else in the folder.

Notes for end users:

- **Mode 2 needs an internet connection** (speech-to-text). Mode 1 works fully offline.
- The audio-file picker accepts **`.wav`** files.
- If the app does not start on a very clean PC, install the **[Microsoft Visual C++ Redistributable (x64)](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)** and try again.

---

## 4. System requirements

**To run the release build**

- Windows 10 or 11 (64-bit)
- A working **webcam** (Mode 1)
- A working **microphone** and **internet** (Mode 2 live listening / audio transcription)
- Several GB of free disk space (TensorFlow and MediaPipe are large)

**To run from source (developers)**

- Windows 10/11 (the setup notes are Windows-focused; the app itself is Python/Tkinter)
- **Python 3.11** — *not* 3.12 or 3.13. `mediapipe==0.10.14` (pinned in `requirements.txt`) has no Windows wheels for Python 3.12+, and newer MediaPipe versions removed the `mp.solutions` API this project uses.
- ~2–3 GB of free disk space

---

## 5. Developer setup

### 5.1 Get the code

```powershell
git clone https://github.com/Sahan20030814/Sign_Language_Application.git
cd Sign_Language_Application
```

### 5.2 Create a virtual environment (Python 3.11)

```powershell
py -3.11 -m venv venv
```

Activate it. **Do this in every new terminal** you open for this project:

```powershell
.\venv\Scripts\activate
```

- In **PowerShell** the leading `.\` is required. Without it PowerShell tries to import a module called `venv`.
- In **Command Prompt (cmd.exe)** use `venv\Scripts\activate`.
- Your prompt should now start with `(venv)`.

If PowerShell says *"running scripts is disabled on this system"*, run this once and try again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Check that the right interpreter is active:

```powershell
python --version        # must print Python 3.11.x
pip show mediapipe      # must show Version: 0.10.14
```

### 5.3 Install dependencies

```powershell
pip install -r requirements.txt
```

`requirements.txt`:

| Package | Used for |
|---|---|
| `opencv-python` | Camera capture and drawing |
| `mediapipe==0.10.14` | Hand landmark detection |
| `tensorflow>=2.16` | LSTM word model |
| `scikit-learn` | Train/validation split |
| `numpy`, `Pillow` | Arrays and image handling |
| `pyttsx3` | Offline text-to-speech |
| `SpeechRecognition` | Speech-to-text (Mode 2) |
| `pydub` | Audio conversion helper |
| `pyaudio` | Microphone access (Mode 2) |
| `pygame` | Audio-file playback (Mode 2) |
| `pypiwin32` (Windows only) | Windows speech/COM support |

If a package fails with *"Microsoft Visual C++ 14.0 or greater is required"*, install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and retry. If `pyaudio` fails to install, see [Troubleshooting](#13-troubleshooting).

---

## 6. Quick start: run with the demo model

The app needs two files in the project root: `word_model.h5` and `word_labels.json`. To try the app immediately, generate a placeholder model from synthetic data:

```powershell
python make_demo_model.py
```

This creates `word_model.h5` + `word_labels.json` in seconds, and also writes sample recordings into `gesture_data/` so the Mode 2 avatar has something to animate.

> ⚠️ **The demo model does not recognise real signs.** It is trained on made-up numbers, not real hands. Whatever you show the camera, it will pick one of the 10 words more or less at random. That is expected. It exists so you can test the camera, power switch, mute, speak, copy, sentence box, microphone, audio player and avatar mechanics end to end. Train a real model (next section) for real accuracy.

Run the app:

```powershell
python app.py
```

---

## 7. Training your own real model

### Step 1 — Record your signs

```powershell
python collect_data.py
```

- **Default vocabulary (10 words):** `hello, thanks, yes, no, please, help, water, food, stop, love`.
- Custom words or a subset:
  ```powershell
  python collect_data.py --words hello thanks yes no please goodbye
  ```
- More samples per word (default is 30):
  ```powershell
  python collect_data.py --sequences 40
  ```
- Use a different camera:
  ```powershell
  python collect_data.py --cam_index 1
  ```

While recording:

| Key | Action |
|---|---|
| **SPACE** | Start recording the next sample (3-2-1 countdown, then ~1 second of "RECORDING…") |
| **N** | Skip to the next word |
| **Q** | Quit (samples already recorded are kept) |

Each sample is a 30-frame sequence of hand landmarks (not video) saved as `gesture_data/<word>/<n>.npy` with shape `(30, 126)`. Files are tiny and no images are stored.

**Tips for good data**

- Keep your hand fully inside the frame for the whole second.
- Aim for **30–40+ samples per word**.
- Vary position, lighting and background slightly between takes, and record in several sessions rather than all at once.

### Step 2 — Train the model

```powershell
python train_lstm.py
python train_lstm.py --epochs 100 --batch_size 16     # optional tuning
```

- Trains an LSTM classifier (LSTM 128 → Dropout → LSTM 64 → Dropout → Dense 64 → Softmax) on everything in `gesture_data/`.
- Uses an 85/15 train/validation split, early stopping and keeps the best checkpoint.
- Writes **`word_model.h5`** and **`word_labels.json`**, the two files the app loads at runtime.
- Takes a few minutes on a normal CPU. It always retrains from scratch on the full folder, so re-run it whenever you add recordings or words.

> **Important for Mode 2:** the avatar can only sign a word if `gesture_data/<word>/` contains at least one `.npy` recording for it. Keep `gesture_data/` alongside the model (and include it in any release you build).

---

## 8. Using the app

### 8.1 Header controls (both modes)

| Control | Function |
|---|---|
| **🔁 Mode** | Switch between *Sign → Speech* and *Speech → Sign* |
| **⏻ Power** | Master switch. **Starts OFF.** Recognition and listening only work when it is ON. |
| **⛶ Fullscreen / F11** | Toggle fullscreen. **Esc** exits fullscreen. |
| **Language** | English only for now. Changing it asks for confirmation and clears the current text. |

The window is freely resizable and the layout is fully responsive.

### 8.2 Mode 1 — Sign → Speech

1. Turn **⏻ Power** ON.
2. Show a hand sign anywhere in the camera view. There is no ROI box; MediaPipe finds the hand automatically. **Nothing is recognised while no hand is visible.**
3. The **BUILDING WORD** bar shows the model's live, in-progress guess with a confidence percentage.
4. When the confidence is high enough (default ≥ 45%) and the cooldown has passed (default 1.4 s), the word is:
   - added to the sentence automatically (with a trailing space), **and**
   - spoken immediately (unless muted).

| Button | Function |
|---|---|
| **🔔 / 🔕 Mute** | Starts *unmuted*. When muted, new words appear as text but are not spoken automatically. The manual Speak button still works. |
| **🔊 Speak** | Reads the whole sentence aloud. If a new sign is recognised meanwhile, it stops at once and speaks only the new word. |
| **⧉ Copy** | Copies the sentence to the clipboard. |
| **␣ Add Space** | Inserts a space manually. |
| **⌫ Backspace** | Removes the last recognised **word** (signs are whole words, not letters). |
| **🗑 Clear All** | Clears everything. |

### 8.3 Mode 2 — Speech → Sign

Click **🔁 Mode** to switch. Pick an input source from the dropdown.

#### 🎙 Microphone

1. Choose your microphone from the dropdown. It lists the working input devices that Windows Sound Settings shows as enabled (duplicates and disabled devices are filtered out).
2. Make sure **⏻ Power** is ON, then **unmute** the mic. It starts muted.
3. Speak. Words that match the model's vocabulary make the avatar perform the corresponding recorded hand gesture. Everything you say is added to the **TRANSCRIPT** box.
4. Use **⧉ Copy** to copy the transcript.

#### 🔊 Audio file

1. Click **Browse…** and select a **`.wav`** file.
2. The file is transcribed in the background in short chunks (internet required). A progress bar and percentage show the status, and results stream into the transcript as each chunk completes.
3. Controls:
   - **▶ Play Audio** — plays the sound. The avatar signs in step with playback automatically.
   - **Timeline** — click or drag to seek.
   - **⏪ 5s / 5s ⏩** — skip backward or forward.
   - **Speed** — 0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 2.0x. It is disabled until a file is loaded, resets to 1.0x when you switch mode/source, and changing it while playing does not interrupt the audio.
4. Selecting a new file resets the path, timeline and transcript.

#### ▶ Play Signs / ⏸ Pause Signs

Below the transcript box. **Play Signs** performs the gestures for the words **from the cursor position onward**, in order, and moves the cursor along. **Pause** stops on the current gesture, and **Play** resumes from there. This works independently of the audio, so you can review the transcript at your own pace.

#### What the avatar does

- Idle: arms rest inward, hands folded together.
- Signing: arms spread, and each hand follows the recorded landmark sequence for the word. The non-signing hand stays in its resting shape and is always visible.
- If no recording exists for a word, the status line under the avatar says **"No recorded gesture for '<word>'"**.

---

## 9. Project structure

```
Sign_Language_Application/
├── app.py               # Main application: GUI, camera, recognition, TTS, both modes
├── gesture_utils.py     # MediaPipe landmark helpers, constants, get_word_sample()
├── audio_player.py      # .wav playback with play/pause/seek/speed (pygame)
├── stt_utils.py         # Speech-to-text, microphone listing, audio chunking
├── avatar.py            # Human avatar drawn on a Tkinter canvas (hands from real landmarks)
├── renderer3d.py        # Experimental pure-Python 3D renderer (not used by app.py yet)
├── collect_data.py      # Record your own webcam samples per word
├── train_lstm.py        # Train the LSTM word classifier
├── make_demo_model.py   # Generate an instant placeholder model + sample data
├── launcher.py          # Entry point compiled to SignBridge.exe (see section 11)
├── requirements.txt
├── resources/
│   ├── SignBridge.png   # App icon (source image)
│   └── SignBridge.ico   # Windows icon (taskbar / .exe)
├── gesture_data/        # Recorded samples: gesture_data/<word>/<n>.npy
├── word_model.h5        # Trained model (from train_lstm.py / make_demo_model.py)
├── word_labels.json     # Class index → word mapping
└── LICENSE
```

---

## 10. Configuration & tuning

Edit these constants directly in the source:

| File | Constant | Default | Meaning |
|---|---|---|---|
| `app.py` | `CONFIDENCE_THRESHOLD` | `0.45` | Minimum model confidence to commit a word. Raise it (e.g. 0.8) for a *real, well-trained* model to reduce false positives. |
| `app.py` | `COMMIT_COOLDOWN` | `1.4` | Seconds before another word can be committed. |
| `app.py` | `CAM_INDEX` | `0` | Which camera to open. |
| `gesture_utils.py` | `SEQUENCE_LENGTH` | `30` | Frames per gesture (~1 s). Changing it requires re-recording and retraining. |
| `gesture_utils.py` | `DEFAULT_WORDS` | 10 words | Starter vocabulary. |
| `stt_utils.py` | `chunk_seconds` | `8` | Length of each audio chunk sent to speech-to-text. |

**Adding a new word:** add it with `collect_data.py --words ...` (any word not yet in `gesture_data/` is treated as new), then re-run `train_lstm.py`.

---

## 11. Building the standalone Windows release

### Why not a plain `pyinstaller app.py`?

Freezing `app.py` directly does **not** work reliably. TensorFlow's native runtime fails inside a PyInstaller-frozen app on Windows with:

```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal
```

This is a long-standing upstream issue. Instead, SignBridge bundles a **portable (embeddable) Python** next to the app and uses a tiny launcher `.exe` that simply starts `app.py` through it, exactly like running `python app.py`. TensorFlow is never frozen, so the problem never occurs.

### Step 1 — Prepare the project

- Make sure `launcher.py` exists (see below) and your real `word_model.h5`, `word_labels.json` and `gesture_data/` are in place.
- Make sure `.gitignore` excludes `python_embed/`, `build/`, `dist/`, `*.spec`, `SignBridge_Release/` and `*.zip`.

`launcher.py`:

```python
import subprocess
import sys
import os

def get_base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def main():
    base_dir = get_base_dir()
    python_dir = os.path.join(base_dir, "python_embed")
    python_exe = os.path.join(python_dir, "pythonw.exe")
    app_script = os.path.join(base_dir, "app.py")

    env = os.environ.copy()
    env["TCL_LIBRARY"] = os.path.join(python_dir, "tcl", "tcl8.6")
    env["TK_LIBRARY"] = os.path.join(python_dir, "tcl", "tk8.6")

    subprocess.Popen([python_exe, app_script], cwd=base_dir, env=env)

if __name__ == "__main__":
    main()
```

### Step 2 — Get the Python embeddable package

Download **Python 3.11's Windows embeddable package** and extract it into the project root as `python_embed/`:

<https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip>

### Step 3 — Enable site-packages

Edit `python_embed\python311._pth` so it contains exactly:

```
python311.zip
.
Lib\site-packages
..

import site
```

(`Lib\site-packages` lets installed packages be found. `..` puts the app folder on the path so `app.py`, `gesture_utils.py` and the other modules can be imported.)

### Step 4 — Install pip

```powershell
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "get-pip.py"
.\python_embed\python.exe get-pip.py
```

### Step 5 — Install dependencies into the embedded Python

```powershell
.\python_embed\python.exe -m pip install -r requirements.txt
```

### Step 6 — Add Tkinter (the embeddable package omits it)

Copy it from a regular Python 3.11 install:

```powershell
$py311 = (py -3.11 -c "import sys; print(sys.prefix)")

Copy-Item -Recurse "$py311\Lib\tkinter" ".\python_embed\Lib\site-packages\tkinter" -Force
Copy-Item "$py311\DLLs\_tkinter.pyd" ".\python_embed\" -Force
Copy-Item "$py311\DLLs\tcl86t.dll" ".\python_embed\" -Force
Copy-Item "$py311\DLLs\tk86t.dll" ".\python_embed\" -Force
Copy-Item "$py311\DLLs\zlib1.dll" ".\python_embed\" -Force -ErrorAction SilentlyContinue
Copy-Item -Recurse "$py311\tcl" ".\python_embed\tcl" -Force
```

### Step 7 — Test the portable Python directly

```powershell
$env:TCL_LIBRARY = "$PWD\python_embed\tcl\tcl8.6"
$env:TK_LIBRARY  = "$PWD\python_embed\tcl\tk8.6"
.\python_embed\python.exe app.py
```

Test **both modes**: camera + recognition in Mode 1; microphone list, `.wav` playback, transcription (internet on) and the avatar in Mode 2. Use `python.exe` (not `pythonw.exe`) while testing so error messages are visible.

### Step 8 — Compile the launcher

```powershell
pip install pyinstaller
pyinstaller --name "SignBridge" --onefile --windowed --icon "resources\SignBridge.ico" launcher.py
```

This takes seconds because the launcher has no heavy dependencies. Output: `dist\SignBridge.exe`.

### Step 9 — Assemble the release folder

```powershell
Remove-Item -Recurse -Force SignBridge_Release -ErrorAction SilentlyContinue
mkdir SignBridge_Release

Copy-Item dist\SignBridge.exe SignBridge_Release\
Copy-Item -Recurse python_embed SignBridge_Release\
Copy-Item app.py, audio_player.py, avatar.py, renderer3d.py, stt_utils.py, gesture_utils.py SignBridge_Release\
Copy-Item word_model.h5, word_labels.json SignBridge_Release\
Copy-Item -Recurse resources SignBridge_Release\
Copy-Item -Recurse gesture_data SignBridge_Release\
```

> `gesture_data/` **must** be included in v2.0 — the avatar animates from those recordings. Delete any `__pycache__` folders from the release folder.

Final layout:

```
SignBridge_Release/
├── SignBridge.exe
├── python_embed/                 # includes Lib\site-packages
├── app.py
├── audio_player.py
├── avatar.py
├── renderer3d.py
├── stt_utils.py
├── gesture_utils.py
├── word_model.h5
├── word_labels.json
├── gesture_data/
└── resources/
    ├── SignBridge.png
    └── SignBridge.ico
```

### Step 10 — Test it as an end user would

```powershell
cd SignBridge_Release
.\SignBridge.exe
```

Ideally test on a PC (or clean VM) **without Python installed**.

### Step 11 — Package

```powershell
Compress-Archive -Path "SignBridge_Release\*" -DestinationPath "SignBridge_v2.0_windows.zip"
```

The zip will be large because TensorFlow lives in `python_embed\Lib`. PowerShell's `Compress-Archive` fails on archives over **2 GB**; if you hit that, use **7-Zip** instead. GitHub release assets are limited to 2 GB each. Extract the zip somewhere else and run it from there to verify it before publishing.

---

## 12. Publishing to GitHub

1. Check what Git is about to commit:
   ```powershell
   git status
   ```
   `python_embed/`, `dist/`, `build/`, `SignBridge_Release/` and `*.zip` must **not** appear. Git (and GitHub) reject files over 100 MB.
2. Commit and push the source:
   ```powershell
   git add .
   git commit -m "Release v2.0: Speech-to-Sign mode, avatar, audio player"
   git push origin main
   ```
3. On GitHub: **Releases → Draft a new release**.
4. Create the tag `v2.0.0`, set the title to `SignBridge v2.0`.
5. Attach **`SignBridge_v2.0_windows.zip`** as a release asset.
6. In the release notes, summarise what's new and repeat the [download & run](#3-download--run-no-python-needed) steps.

Never commit `venv/`, `python_embed/`, build output or zip files. Only the source goes in the repository; the packaged build goes in the Release.

---

## 13. Troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
You are almost certainly running the wrong Python (often 3.13) or the wrong MediaPipe. Activate the virtual environment (`.\venv\Scripts\activate`), confirm `python --version` prints 3.11.x, and run `pip install mediapipe==0.10.14`. If NumPy conflicts appear: `pip install "numpy<2"`.

**`pip install` says "no matching distribution found" for TensorFlow or MediaPipe**
Your Python is 3.12/3.13. Recreate the venv with `py -3.11 -m venv venv`.

**`venv\Scripts\activate` fails with "The module 'venv' could not be loaded"**
You are in PowerShell and left off the leading `.\` — use `.\venv\Scripts\activate`.

**"running scripts is disabled on this system"**
Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once.

**`oneDNN custom operations are on…` / `absl::InitializeLog()` messages**
Harmless TensorFlow log lines. Ignore them (or set `TF_ENABLE_ONEDNN_OPTS=0` to silence the first).

**`pip install pyaudio` fails**
Make sure you are on Python 3.11 (prebuilt Windows wheels exist for it). Otherwise install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or download a matching prebuilt `PyAudio` wheel and install it with `pip install <file>.whl`.

**Microphone dropdown is empty / wrong device**
Open Windows *Settings → System → Sound → Input* and make sure the device is **Enabled**. Also check *Settings → Privacy & security → Microphone* and allow desktop apps to use the microphone.

**Speech-to-text errors ("Speech service error", nothing transcribed)**
Mode 2 transcription needs a working internet connection. Free Google endpoints can also fail on very noisy audio or music; this is speech recognition, not song/lyric recognition.

**The avatar never signs, and the status says "No recorded gesture for '<word>'"**
`gesture_data/<word>/` has no `.npy` files for that word. Run `python make_demo_model.py` (demo data) or `python collect_data.py` + `python train_lstm.py` (real data). In a release build, make sure `gesture_data/` was copied next to `SignBridge.exe`.

**Camera doesn't open / "Could not open the webcam"**
Close other apps that use the camera (Zoom, Teams, another SignBridge window) and check *Settings → Privacy & security → Camera*. Change `CAM_INDEX` in `app.py` if you have several cameras.

**"Model not found" popup on launch**
`word_model.h5` and `word_labels.json` are missing from the app folder. Run `python make_demo_model.py` or train a real model.

**Avatar feels slow**
Rendering runs on the CPU in plain Python. Close other heavy apps, lower the avatar frame rate in `avatar.py`, or reduce the window size.

**`ImportError: DLL load failed while importing _pywrap_tensorflow_internal` in a packaged `.exe`**
This is the TensorFlow + PyInstaller freezing bug. Use the portable-Python launcher approach in [section 11](#11-building-the-standalone-windows-release) instead of freezing `app.py`.

**`ModuleNotFoundError: No module named 'tkinter'` with `python_embed\python.exe`**
The embeddable package omits Tkinter. See Step 6 of [section 11](#11-building-the-standalone-windows-release).

**`ModuleNotFoundError: No module named 'gesture_utils'` (or other project modules) with `python_embed\python.exe`**
`python311._pth` must contain the `..` line. See Step 3 of [section 11](#11-building-the-standalone-windows-release).

**The release `.exe` opens nothing and shows no error**
The launcher uses `pythonw.exe`, which hides errors. Run `.\python_embed\python.exe app.py` from the release folder (after setting `TCL_LIBRARY` / `TK_LIBRARY` as in Step 7) to see the actual message.

**Windows SmartScreen blocks the app**
Click *More info → Run anyway*. The app is unsigned because it has no paid code-signing certificate.

---

## 14. Honest limitations

- Accuracy depends on how many consistent samples you record per word (30–40+, with varied lighting and background). This is a small student/hobby-scale project, not a production-grade sign language system.
- The vocabulary is a small fixed set (10–20 words). It does **not** understand natural sign language grammar.
- If your hand leaves the frame mid-sign, the recognition buffer resets. Keep the hand visible for the whole sign.
- The bundled `word_model.h5` may be a **demo model** trained on synthetic data. Train on your own recordings for real accuracy.
- **Mode 2 speech-to-text needs internet** and is tuned for spoken speech, not music or singing.
- The audio-file picker accepts **`.wav`** only.
- The free speech API does not return per-word timestamps. For audio files, each chunk's words are spread **evenly** across that chunk's time slot, so avatar timing is an approximation, not frame-perfect. Early in playback, before the first chunk returns, the avatar stays idle for a few seconds (that is real network latency).
- The timeline is a simple seek bar, not a waveform.
- The avatar is a stylised 2D figure drawn with Tkinter. Only its hands move, and they reflect the *recorded* landmark shapes, so gesture quality is only as good as your recordings.
- Tkinter has no true blur or shadow effects, so the UI uses gradients, rounded cards and colour-transition animations instead.

---

## 15. Contributing

Contributions are welcome — new vocabulary, better training data, UI polish, or bug fixes.

- If you propose a **new vocabulary word**, mention in your pull request how many samples you recorded, and ideally include the `gesture_data/<word>/` folder so others can retrain and verify.
- Prefer committing **`gesture_data/` recordings** over a large retrained `word_model.h5`. The `.h5` is a binary artifact and bloats the repository history; maintainers can regenerate it from the data.
- Record several sessions at different times of day and lighting, with your hand fully in frame.
- Please do not commit `venv/`, `python_embed/`, `dist/`, `build/` or `*.zip` files.

---

## 16. Privacy

- **Camera video never leaves your PC.** Only hand-landmark numbers are computed locally, and recordings you make are stored as small `.npy` files (no video or images).
- **Mode 1 and text-to-speech are fully offline.**
- **Mode 2 sends audio to Google's free web speech API** to turn speech into text (microphone speech in short phrases, or audio-file chunks). Do not use Mode 2 with audio you are not comfortable sending to an online service.

---

## 17. Changelog

### v2.0
- **New:** Speech → Sign Language mode with a 🔁 Mode switch.
- **New:** Microphone input with automatic device list and mute control.
- **New:** `.wav` audio-file input with play/pause, seek timeline, ±5 s skip and 0.5x–2x speed.
- **New:** Streaming transcription (~8 s chunks) with progress overlay and a copyable transcript box.
- **New:** Human avatar whose hands are driven by real recorded landmark sequences; **Play/Pause Signs**; automatic sync with audio playback.
- **New:** Webcam released in Speech → Sign mode and reconnected on return.
- **New:** `audio_player.py`, `stt_utils.py`, `avatar.py`, `renderer3d.py`; `get_word_sample()` in `gesture_utils.py`.
- **Changed:** new dependencies — `SpeechRecognition`, `pydub`, `pyaudio`, `pygame`.
- **Fixed:** header layout, language dropdown rendering on Windows, mode/source switching race conditions with background transcription.

### v1.0
- Sign Language → Speech mode with word-level LSTM recognition, automatic speech, sentence building, and portable-Python Windows release.

---

## 18. License

See [`LICENSE`](./LICENSE) for details.
