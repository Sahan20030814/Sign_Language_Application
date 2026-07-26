# SignBridge — Sign Language → Speech (v1.0)

SignBridge watches a person's hand signs through a webcam, recognises
whole **words** (not just letters) using a custom-trained AI model, builds
them into a sentence, and speaks the sentence out loud automatically —
helping a person who cannot speak communicate with a hearing person in
real time.

This is **v1.0**: the **Sign Language → Speech** mode only. (A future v2.0
adds a reverse **Speech → Sign Language** mode with an on-screen avatar.)

---

## Table of Contents

1. [How it works](#1-how-it-works)
2. [Download & run (no Python needed)](#2-download--run-no-python-needed)
3. [Developer setup — requirements](#3-developer-setup--requirements)
4. [Getting the code](#4-getting-the-code)
5. [Set up a virtual environment](#5-set-up-a-virtual-environment)
6. [Install dependencies](#6-install-dependencies)
7. [Quick start: run it with the dummy demo model](#7-quick-start-run-it-with-the-dummy-demo-model)
8. [Run the app from source](#8-run-the-app-from-source)
9. [Using the app](#9-using-the-app)
10. [Project structure](#10-project-structure)
11. [Building a standalone .exe](#11-building-a-standalone-exe)
12. [Troubleshooting](#12-troubleshooting)
13. [Honest limitations](#13-honest-limitations)
14. [For Contributors: training a REAL AI model](#14-for-contributors-training-a-real-ai-model)
15. [License](#15-license)

---

## 1. How it works

Recognising a signed **word** requires tracking motion over roughly a
second, not just a single photo — so this project splits the problem into
two free, open-source pieces instead of relying on one big pretrained
model (no such free "download-and-go" word-level sign model reliably
exists):

- **[MediaPipe Hands](https://github.com/google/mediapipe)** (free,
  open-source) finds 21 hand landmark points per frame. It does **not**
  know what word is being signed — it only locates the hand's shape.
- A **30-frame sequence of those landmarks (~1 second)** is fed into an
  **LSTM neural network that this project trains itself**
  (`train_lstm.py`). That trained network is the actual "intelligence"
  that predicts which word was signed.
- Training data comes from **your own webcam recordings**
  (`collect_data.py`) — there is no pretrained word-model to download.

The result is a fully free, fully open-source pipeline, trained entirely
on your own data.

---

## 2. Download & run (no Python needed)

If you just want to **use** the app — no coding, no Python install — grab
a pre-built release instead of following the developer setup below.

1. Go to the **[Releases](../../releases)** page of this repository.
2. Download the latest **`SignBridge_v1.0_windows.zip`** file (under
   "Assets" on the release).
3. **Extract the whole zip** to a folder of your choice (right-click →
   *Extract All...*). Don't try to run anything from inside the zip
   without extracting first.
4. Open the extracted folder and double-click **`SignBridge.exe`**.
   - The first launch can take a few extra seconds while things warm up
     — that's normal.
   - Windows may show a **"Windows protected your PC"** SmartScreen
     warning, because this is an unsigned hobby/student project (no paid
     code-signing certificate). Click **"More info" → "Run anyway"**.
5. Allow the **camera** permission prompt if Windows asks for it.

> ⚠️ **Keep the whole extracted folder together.** `SignBridge.exe` needs
> the `python_embed/`, `resources/`, `word_model.h5`, and other files that
> sit next to it — don't move just the `.exe` on its own, and don't
> delete anything else in that folder.

That's it — no Python, no pip installs, no virtual environment needed for
this route. If you *do* want to modify the code, retrain the model, or
build your own release, continue to the developer setup below.

---

## 3. Developer setup — requirements

- **Windows 10/11** (the setup notes below are Windows-focused; the app
  itself is cross-platform Python/Tkinter)
- **Python 3.11** — **not 3.12/3.13**. `mediapipe==0.10.14` (the version
  pinned in `requirements.txt`, needed for a feature this project relies
  on — see [Troubleshooting](#12-troubleshooting)) does not have Windows
  wheels for Python 3.12+.
- A working **webcam**
- ~2–3 GB free disk space (TensorFlow + MediaPipe are large libraries)

---

## 4. Getting the code

```powershell
git clone https://github.com/<your-username>/Sign_Language_Application.git
cd Sign_Language_Application
```

---

## 5. Set up a virtual environment

Using a virtual environment keeps this project's dependencies separate
from anything else on your machine.

```powershell
py -3.11 -m venv venv
```

Activate it — **every time you open a new terminal** to work on this
project, you need to activate it again:

```powershell
.\venv\Scripts\activate
```

> **PowerShell users:** the leading `.\` is required — running
> `venv\Scripts\activate` without it makes PowerShell try to import a
> module called `venv` instead of running the script, which fails with
> `The module 'venv' could not be loaded`.
>
> If you're using Command Prompt (`cmd.exe`) instead of PowerShell, drop
> the `.\`: just `venv\Scripts\activate`.

You'll know it worked when your prompt shows `(venv)` at the start.

If PowerShell refuses to run the activation script at all with a message
like *"running scripts is disabled on this system"*, run this once
(in an Administrator PowerShell window), then try activating again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 6. Install dependencies

With the virtual environment activated:

```powershell
pip install -r requirements.txt
```

If `pyaudio`-style native-build errors show up for any dependency, or a
package fails with *"Microsoft Visual C++ 14.0 or greater is required"*,
install the
[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
and try again.

---

## 7. Quick start: run it with the dummy demo model

The app needs two files in the project root before it can recognise
anything: `word_model.h5` and `word_labels.json`. For a quick look at the
app with **zero setup**, generate a placeholder ("dummy") model from
synthetic data:

```powershell
python make_demo_model.py
```

This creates `word_model.h5` + `word_labels.json` in seconds.

⚠️ **Important — read this before you judge the accuracy:** this dummy
model is trained on made-up, synthetic numbers, **not real hand shapes**.
It exists purely so you (or anyone trying the repo out) can test the
camera, power switch, mute, speak, copy, sentence box, and overall UI
end-to-end, immediately, without recording anything first. **It will not
correctly recognise real signs** — whatever hand shape you show it, it
will confidently call it one of the 10 vocabulary words essentially at
random. That is expected, not a bug.

If you want the app to actually work correctly, you (or a contributor)
need to train it on real recorded data — see
[section 14, For Contributors](#14-for-contributors-training-a-real-ai-model).

---

## 8. Run the app from source

```powershell
python app.py
```

---

## 9. Using the app

1. **⏻ Power** (top-right) — **starts OFF** by default. Turn it ON before
   recognition begins.
2. **⛶ Fullscreen** / **F11** — toggles fullscreen. **Esc** exits it. The
   window is also freely resizable — the layout is fully responsive.
3. There's no ROI box — show your hand anywhere in frame; MediaPipe finds
   it automatically. Nothing is recognised while no hand is visible.
4. **BUILDING WORD** bar — shows the model's live, in-progress guess
   before it's confident enough to commit. Once confirmed, the word:
   - is added to the sentence automatically (with a space), **and**
   - (unless muted) is spoken immediately.
5. **🔔/🔕 Mute** — starts **unmuted**. When muted, new words still appear
   as text but aren't spoken automatically (the manual Speak button still
   works either way).
6. **🔊 Speak** — reads the whole sentence aloud. If a new sign is
   recognised while it's speaking, it stops immediately and speaks just
   the new word (it doesn't restart the full sentence).
7. **⧉ Copy** — copies the sentence to the clipboard.
8. **␣ Add Space** — manually inserts a space.
9. **⌫ Backspace** — removes the last recognised *word* (signs produce
   whole words, so this removes a word, not a single character).
10. **🗑 Clear All** — clears everything.
11. **Language dropdown** (top-right) — English only for now; switching
    languages asks for confirmation and clears the current sentence.

---

## 10. Project structure

```
Sign_Language_Application/
├── app.py                # Main application (GUI, camera, recognition, TTS)
├── gesture_utils.py       # MediaPipe landmark extraction helpers/constants
├── collect_data.py        # Records your own webcam samples per word
├── train_lstm.py          # Trains the LSTM word-classifier on your recordings
├── make_demo_model.py     # Generates an instant placeholder model for testing
├── launcher.py            # Entry point used for the packaged .exe (see section 11)
├── requirements.txt
├── resources/
│   ├── SignBridge.png     # App icon (source image)
│   └── SignBridge.ico     # App icon (Windows .ico, for the taskbar/.exe)
├── gesture_data/           # Your recorded samples (created by collect_data.py)
├── word_model.h5           # Trained model (created by make_demo_model.py / train_lstm.py)
├── word_labels.json         # Class index → word mapping (same as above)
└── LICENSE
```

---

## 11. Building a standalone .exe

### Why this isn't a plain `pyinstaller app.py`

The obvious approach — freezing `app.py` directly with
[PyInstaller](https://pyinstaller.org/) — **does not work reliably** for
this project. TensorFlow's native runtime consistently fails inside a
PyInstaller-frozen app on Windows with:

```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal:
A dynamic link library (DLL) initialization routine failed.
```

This is a long-standing, unresolved upstream issue (see
`pyinstaller/pyinstaller` issues #3754, #8300, #8656 and
`tensorflow/tensorflow` issues #90244, #96760) — `--noupx`,
`--collect-all`, manually relocating the DLL, and using the latest
PyInstaller version did not fix it in testing. Nuitka (an alternative
freezer) has similar reports. Both `--onedir` and `--onefile` are
affected identically, so switching between them doesn't help either.

### The approach that actually works: a portable-Python launcher

Instead of freezing TensorFlow at all, we bundle a small, portable Python
runtime alongside the app and use a **tiny, dependency-free launcher**
(compiled with PyInstaller, since it has nothing heavy to freeze) that
simply starts the real app through that portable runtime — exactly like
running `python app.py` normally, just packaged up. This sidesteps the
DLL-freezing bug entirely because TensorFlow never gets frozen; it runs
through a completely normal Python interpreter.

#### Step 1 — Get the Python Embeddable Package

Download **Python 3.11's Windows embeddable package**:
👉 https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip

Extract it into the project root as `python_embed/`, so you have:
```
Sign_Language_Application/python_embed/
```

#### Step 2 — Enable site-packages in the embeddable Python

Open `python_embed\python311._pth` in a text editor. It looks like this
by default:
```
python311.zip
.

# Uncomment to run site.main() automatically
#import site
```

Edit it to:
```
python311.zip
.
Lib\site-packages
..

import site
```
(uncomment `import site`, add a `Lib\site-packages` line, and add a `..`
line so the app's own folder — where `app.py`, `gesture_utils.py`, the
model files, etc. live — is on the path too.)

#### Step 3 — Install pip into the embeddable Python

```powershell
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "get-pip.py"
.\python_embed\python.exe get-pip.py
```

#### Step 4 — Install this project's dependencies into it

```powershell
.\python_embed\python.exe -m pip install -r requirements.txt
```

#### Step 5 — Add `tkinter` (the embeddable package omits it)

The embeddable package deliberately ships without Tkinter. Copy it in
from a **regular** Python 3.11 install (install one from python.org if
you don't have one — it can coexist with the embeddable copy):

```powershell
$py311 = "C:\Path\To\Your\Regular\Python311"   # find yours with: py -3.11 -c "import sys; print(sys.prefix)"

Copy-Item -Recurse "$py311\Lib\tkinter" ".\python_embed\Lib\site-packages\tkinter" -Force
Copy-Item "$py311\DLLs\_tkinter.pyd" ".\python_embed\" -Force
Copy-Item "$py311\DLLs\tcl86t.dll" ".\python_embed\" -Force
Copy-Item "$py311\DLLs\tk86t.dll" ".\python_embed\" -Force
Copy-Item "$py311\DLLs\zlib1.dll" ".\python_embed\" -Force -ErrorAction SilentlyContinue
Copy-Item -Recurse "$py311\tcl" ".\python_embed\tcl"
```

(If `tcl86t.dll`/`tk86t.dll` don't exist in your install, check
`dir "$py311\DLLs\tcl*.dll", "$py311\DLLs\tk*.dll"` for the actual names
and adjust the commands above to match.)

#### Step 6 — Test the portable Python directly

```powershell
$env:TCL_LIBRARY = "$PWD\python_embed\tcl\tcl8.6"
$env:TK_LIBRARY = "$PWD\python_embed\tcl\tk8.6"
.\python_embed\python.exe app.py
```

The app should open normally. If it does, the portable runtime is fully
working and you're ready for the launcher.

#### Step 7 — Create `launcher.py`

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

This sets the Tcl/Tk environment variables automatically, so end users
never have to touch them.

#### Step 8 — Compile the launcher (not `app.py`!) with PyInstaller

```powershell
pip install pyinstaller
pyinstaller --name "SignBridge" --onefile --windowed --icon "resources\SignBridge.ico" launcher.py
```

This build is fast (seconds, not minutes) and produces a small
`.exe`, because `launcher.py` has no heavy dependencies to freeze — it
just calls `subprocess`.

#### Step 9 — Assemble the release folder

```powershell
mkdir SignBridge_Release
Copy-Item dist\SignBridge.exe SignBridge_Release\
Copy-Item -Recurse python_embed SignBridge_Release\
Copy-Item app.py, gesture_utils.py, word_model.h5, word_labels.json SignBridge_Release\
Copy-Item -Recurse resources SignBridge_Release\
```

Final layout:
```
SignBridge_Release/
├── SignBridge.exe
├── python_embed/
├── app.py
├── gesture_utils.py
├── word_model.h5
├── word_labels.json
└── resources/
    ├── SignBridge.png
    └── SignBridge.ico
```

#### Step 10 — Test it as an end user would

```powershell
cd SignBridge_Release
.\SignBridge.exe
```

Ideally also copy this folder to a machine without Python installed (or a
clean VM) and confirm it runs there too.

#### Step 11 — Package and publish

```powershell
Compress-Archive -Path "SignBridge_Release\*" -DestinationPath "SignBridge_v1.0_windows.zip"
```

Verify the zip itself works by extracting it somewhere else and running
it from there, *then* publish it:

1. Go to this repo's **Releases** page → **"Draft a new release"**.
2. Create a tag (e.g. `v1.0.0`).
3. Attach `SignBridge_v1.0_windows.zip` as a release asset.
4. In the release notes, link back to [section 2](#2-download--run-no-python-needed)
   of this README (or just repeat the extract-and-run steps) so end users
   know what to do with it.

Don't commit `python_embed/`, `build/`, `dist/`, `SignBridge_Release/`, or
`*.zip` to git — see the `.gitignore` in this repo, which already
excludes them.

---

## 12. Troubleshooting

**`pip install -r requirements.txt` fails on `tensorflow==...` with "no
matching distribution found"**
Your Python version is likely 3.12/3.13. Rebuild the virtual environment
with Python 3.11 (see [section 5](#5-set-up-a-virtual-environment)).

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
Newer MediaPipe releases (0.10.30+) removed the legacy `solutions` API
this project uses. Make sure `requirements.txt` still pins
`mediapipe==0.10.14`, and that you're installing into an **active virtual
environment** (check your terminal prompt shows `(venv)` — a fresh
terminal window does *not* auto-activate it).

**`venv\Scripts\activate` fails with "The module 'venv' could not be
loaded"**
You're in PowerShell and left off the leading `.\` — see
[section 5](#5-set-up-a-virtual-environment).

**Camera doesn't open / `Could not open the webcam`**
Check no other application (Zoom, Teams, another instance of this app)
is already using it, and that Windows camera privacy settings allow
desktop apps to access it.

**Model not found popup on launch**
Run `python make_demo_model.py` for an instant test model — see
[section 7](#7-quick-start-run-it-with-the-dummy-demo-model) — or train a
real one, see [section 14](#14-for-contributors-training-a-real-ai-model).

**`ImportError: DLL load failed while importing
_pywrap_tensorflow_internal` when running a packaged `.exe`**
This is the TensorFlow+PyInstaller freezing bug described in
[section 11](#11-building-a-standalone-exe) — build using the
portable-Python launcher approach there instead of freezing `app.py`
directly.

**`ModuleNotFoundError: No module named 'tkinter'` while testing
`python_embed\python.exe app.py`**
The embeddable package doesn't ship Tkinter — see Step 5 in
[section 11](#11-building-a-standalone-exe).

**`ModuleNotFoundError: No module named 'gesture_utils'` (or the app's own
files aren't found) with `python_embed\python.exe app.py`**
The embeddable Python's `python311._pth` file restricts the import path
to only what's listed in it — make sure it includes a `..` line pointing
back to the project folder (Step 2 in [section 11](#11-building-a-standalone-exe)).

---

## 13. Honest limitations

- Accuracy depends on recording enough consistent samples per word
  (30–40+, varying lighting/background helps) — this is a small student/
  hobby-scale project, not a production-grade sign language system.
- If your hand leaves the frame mid-sign, the recognition buffer resets —
  keep your hand visible for the whole sign.
- This recognises a small fixed vocabulary (10–20 words), not full,
  natural sign language grammar.

---

## 14. For Contributors: training a REAL AI model

The dummy model from [section 7](#7-quick-start-run-it-with-the-dummy-demo-model)
is only for kicking the tires. If you're contributing to this project —
improving accuracy, adding new vocabulary words, or just want the app to
actually work correctly for you — here's how to train it on **real,
your-own-recorded** hand signs.

### Step 1 — Record your signs

```powershell
python collect_data.py
```

- Default vocabulary: `hello, thanks, yes, no, please, help, water, food,
  stop, love` (10 words). To record a custom subset, or add a brand-new
  word to the vocabulary, pass it explicitly:
  ```powershell
  python collect_data.py --words hello thanks yes no please goodbye
  ```
  (Any word not already in `gesture_data/` is treated as new.)
- A camera window opens. For each word: press **SPACE**, wait for the
  3-2-1 countdown, then perform the sign while it records "RECORDING..."
  for ~1 second (30 frames of hand landmarks, not video).
- Records 30 samples per word by default — pass `--sequences 40` (or
  more) for a larger set. **More samples = better accuracy.** Aim for at
  least 30–40 per word, varying your hand position, lighting, and
  background slightly between takes so the model generalises instead of
  memorising one exact setup.
- **N** skips to the next word early, **Q** quits at any point (samples
  already recorded are kept).
- Recordings are saved under `gesture_data/<word>/*.npy` — plain NumPy
  arrays of landmark coordinates (tiny files, no video/images are stored).

### Step 2 — Train the model

```powershell
python train_lstm.py
```

- Trains an LSTM classifier on everything currently in `gesture_data/`.
- Produces/overwrites `word_model.h5` + `word_labels.json` in the project
  root — these are the two files the app actually loads at runtime.
- Takes a few minutes on a normal CPU; the dataset is intentionally small.
- Re-run this any time you add more recordings or new words to
  `gesture_data/` — it always retrains from scratch on the full folder.

### Tips for contributing good training data

- Keep your hand fully inside the frame for the whole ~1 second recording
  — if it leaves frame, that sample is effectively noise.
- Record a few sessions at different times of day / lighting rather than
  30 samples back-to-back in identical conditions.
- If you're proposing a new vocabulary word in a pull request, please
  mention in the PR description how many samples you recorded for it, and
  ideally include the `gesture_data/<word>/` folder so others can retrain
  and verify.
- `word_model.h5` is a **binary artifact** — if you're contributing
  improved training data rather than a pre-trained model file, prefer
  committing the `gesture_data/` recordings over committing a large
  retrained `.h5`, and let maintainers regenerate the model from that
  data (keeps the repo's history from bloating with binary diffs).

---

## 15. License

See [`LICENSE`](./LICENSE) for details.
