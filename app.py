"""
app.py
--------
Sign Language -> Sentence -> Speech  (word-level, v2.0)
===========================================================

- Live webcam feed that fills its panel completely (no black bars, no ROI box).
- MediaPipe extracts hand landmarks every frame (free & open-source; it does
  NOT decide the word - it only finds the hand's shape/position).
- A 30-frame landmark sequence is fed into OUR OWN trained LSTM model
  (word_model.h5 / word_labels.json) which predicts a whole WORD.
- "Building word" bar shows the live, in-progress guess before it's committed.
- Recognised words are appended to the sentence with an automatic space,
  and (unless muted) spoken immediately.
- Power switch (default OFF) gates recognition entirely.
- Mute switch (default UNMUTED) controls automatic per-word speech.
- "Speak" reads the whole sentence; a new recognised gesture interrupts it
  and speaks only the new word.
- Copy button copies the sentence to the clipboard.
- Language dropdown (English only for now) clears the sentence on change.
- Fully responsive layout: resizable window + real fullscreen (F11).

REQUIREMENTS: word_model.h5 + word_labels.json must exist in this folder.
Run make_demo_model.py for an instant placeholder to test the app, or
collect_data.py + train_lstm.py for a real trained model.
"""

import os
import json
import time
import math
import threading
import collections

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pyttsx3
from tensorflow.keras.models import load_model  # type: ignore

from gesture_utils import (
    create_hands_detector, extract_landmarks, draw_landmarks, get_word_sample,
    SEQUENCE_LENGTH, MODEL_PATH, LABELS_PATH,
)
from avatar import HumanAvatar
from audio_player import AudioPlayer
import stt_utils

# ----------------------------------------------------------------------------
# Behaviour tuning
# ----------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.45
COMMIT_COOLDOWN = 1.4          # seconds before a new word can commit again
CAM_INDEX = 0

# ----------------------------------------------------------------------------
# Palette (deep-space navy -> vivid blue gradient theme)
# ----------------------------------------------------------------------------
GRAD_TOP = "#0a0f2b"
GRAD_BOTTOM = "#1c3fae"
CONTENT_BG = "#0e1436"
CARD_BG = "#141a33"
CARD_BG_ALT = "#1b2246"
ACCENT = "#4f8cff"
ACCENT_BRIGHT = "#7fb0ff"
ACCENT_SOFT = "#2a3a70"
TEXT_LIGHT = "#f2f5ff"
TEXT_MUTED = "#8b93b8"
GREEN_ON = "#37d67a"
RED_OFF = "#e2555c"
AMBER = "#ffb454"
TIMELINE_TRACK = "#0e1330"
TIMELINE_FILL = "#4f8cff"

DEFAULT_W, DEFAULT_H = 1240, 760
HEADER_MIN_H = 88


# ============================================================== small utils
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lerp_color(c1, c2, t):
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def make_vertical_gradient(w, h, c_top, c_bottom):
    w, h = max(1, w), max(1, h)
    col = Image.new("RGB", (1, h))
    r1, g1, b1 = hex_to_rgb(c_top)
    r2, g2, b2 = hex_to_rgb(c_bottom)
    for y in range(h):
        t = y / max(h - 1, 1)
        col.putpixel((0, y), (int(r1 + (r2 - r1) * t),
                               int(g1 + (g2 - g1) * t),
                               int(b1 + (b2 - b1) * t)))
    return col.resize((w, h))


def rounded_points(x1, y1, x2, y2, r):
    r = max(2, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    return [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]


def cover_resize(img, target_w, target_h):
    """Resize + center-crop so the image FILLS target_w x target_h with no bars."""
    target_w, target_h = max(2, target_w), max(2, target_h)
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, int(src_w * scale) + 1), max(1, int(src_h * scale) + 1)
    img = img.resize((new_w, new_h))
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


# ============================================================ responsive card
class Card(tk.Frame):
    """A panel with true rounded corners that resizes responsively with the window."""

    def __init__(self, parent, bg_color, radius=18, **kwargs):
        super().__init__(parent, bg=parent["bg"], **kwargs)
        self.bg_color = bg_color
        self.radius = radius
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0, bg=parent["bg"])
        self.canvas.pack(fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg_color)
        self._win = self.canvas.create_window(radius, radius, anchor="nw", window=self.inner)
        self.canvas.bind("<Configure>", self._redraw)

    def _redraw(self, event):
        w, h = max(event.width, 2), max(event.height, 2)
        self.canvas.delete("shape")
        self.canvas.create_polygon(rounded_points(1, 1, w - 1, h - 1, self.radius),
                                    smooth=True, fill=self.bg_color, outline="", tags="shape")
        self.canvas.tag_lower("shape")
        r = self.radius
        self.canvas.coords(self._win, r, r)
        self.canvas.itemconfig(self._win, width=max(w - 2 * r, 10), height=max(h - 2 * r, 10))


class GradientBackground(tk.Canvas):
    """Root-level canvas: full-window blue gradient with a content frame inset by `margin`."""

    def __init__(self, parent, top, bottom, content_bg, margin=14, **kwargs):
        super().__init__(parent, highlightthickness=0, bd=0, **kwargs)
        self.top, self.bottom, self.margin = top, bottom, margin
        self.content = tk.Frame(self, bg=content_bg)
        self._win = self.create_window(margin, margin, anchor="nw", window=self.content)
        self._img_ref = None
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        w, h = max(event.width, 2), max(event.height, 2)
        self._img_ref = ImageTk.PhotoImage(make_vertical_gradient(w, h, self.top, self.bottom))
        self.delete("bg")
        self.create_image(0, 0, anchor="nw", image=self._img_ref, tags="bg")
        self.tag_lower("bg")
        m = self.margin
        self.coords(self._win, m, m)
        self.itemconfig(self._win, width=max(w - 2 * m, 10), height=max(h - 2 * m, 10))


# ============================================================== video thread
class VideoStream:
    """Captures frames + runs MediaPipe hand detection on a background thread."""

    def __init__(self, cam_index=CAM_INDEX):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open the webcam. Check the camera connection / index.")
        self.hands = create_hands_detector()
        self.lock = threading.Lock()
        self.display_frame = None
        self.landmarks = None
        self.hand_seen = False
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            draw_landmarks(frame, results)
            landmarks, hand_seen = extract_landmarks(results)
            with self.lock:
                self.display_frame = frame
                self.landmarks = landmarks
                self.hand_seen = hand_seen

    def read(self):
        with self.lock:
            if self.display_frame is None:
                return None, None, False
            return self.display_frame.copy(), self.landmarks, self.hand_seen

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()
        self.hands.close()


# ============================================================== speech
class SpeechEngine:
    """Speaks text on a background thread. speak() interrupts anything in progress."""

    def __init__(self, on_state_change=None):
        self._lock = threading.Lock()
        self.active_engine = None
        self.on_state_change = on_state_change or (lambda speaking: None)

    def _run(self, text):
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        with self._lock:
            self.active_engine = engine
        self.on_state_change(True)
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
        finally:
            with self._lock:
                if self.active_engine is engine:
                    self.active_engine = None
            try:
                engine.stop()
            except Exception:
                pass
            self.on_state_change(False)

    def speak(self, text):
        text = (text or "").strip()
        if not text:
            return
        self.interrupt()
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def interrupt(self):
        with self._lock:
            eng = self.active_engine
        if eng is not None:
            try:
                eng.stop()
            except Exception:
                pass


# ============================================================== pulsing dot
class PulsingDot(tk.Canvas):
    """A small glowing/breathing status dot used as a live 'recognising...' indicator."""

    def __init__(self, parent, color=GREEN_ON, size=14, **kwargs):
        super().__init__(parent, width=size, height=size, bg=parent["bg"],
                          highlightthickness=0, **kwargs)
        self.size = size
        self.color = color
        self.phase = 0.0
        self.oval = self.create_oval(2, 2, size - 2, size - 2, fill=color, outline="")
        self._animate()

    def set_color(self, color):
        self.color = color

    def _animate(self):
        if not self.winfo_exists():
            return
        self.phase = (self.phase + 0.10) % (2 * math.pi)
        t = (1 + math.sin(self.phase)) / 2
        shade = lerp_color(CARD_BG, self.color, 0.55 + 0.45 * t)
        try:
            self.itemconfig(self.oval, fill=shade)
        except tk.TclError:
            return
        self.after(60, self._animate)


# ============================================================== main app
class SignLanguageApp:
    LANGUAGES = {"English": "en"}

    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language → Speech")
        self.root.configure(bg=GRAD_TOP)
        self.root.geometry(f"{DEFAULT_W}x{DEFAULT_H}")
        self.root.minsize(1000, 620)
        self.root.resizable(True, True)

        # ---- state ----
        self.power_on = False
        self.muted = False
        self.is_speaking = False
        self.is_fullscreen = False
        self.sentence = ""
        self.history = collections.deque(maxlen=SEQUENCE_LENGTH)
        self.last_commit_time = 0.0
        self.current_language = "English"

        # ---- mode-2 (speech -> sign) state ----
        self.mode = "sign_to_speech"          # or "speech_to_sign"
        self.input_source = "microphone"       # or "audio_file"
        self.mic_devices = []                  # [(index, name), ...]
        self.selected_mic_index = None
        self.mic_muted = True                  # default MUTED, per spec
        self.live_stt = None
        self.transcript = ""
        self.avatar_queue = collections.deque()
        self.avatar_busy = False
        self.replay_active = False
        self.replay_words = []
        self.replay_index = 0
        self.audio_player = None
        self.audio_path = None
        self.audio_transcribing = False
        self.audio_transcribe_token = 0
        self.audio_sync_slots = []       # [(start_sec, end_sec, word), ...] built incrementally per chunk
        self.audio_sync_last_slot = None
        self._timeline_dragging = False

        self._load_model()
        self.speech = SpeechEngine(on_state_change=self._on_speech_state_change)

        try:
            self.stream = VideoStream(CAM_INDEX)
        except RuntimeError as e:
            messagebox.showerror("Camera error", str(e))
            self.stream = None

        self._build_ui()
        self._bind_shortcuts()
        self._tick()

    # ------------------------------------------------------------ model
    def _load_model(self):
        if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
            self.model = None
            self.labels = {}
            self.root.after(300, lambda: messagebox.showwarning(
                "Model not found",
                f"Could not find '{MODEL_PATH}' / '{LABELS_PATH}'.\n\n"
                "Quick test option:\n  python make_demo_model.py\n\n"
                "For a REAL trained model:\n"
                "  python collect_data.py\n  python train_lstm.py\n\n"
                "The app will open, but recognition is disabled until then."))
            return
        self.model = load_model(MODEL_PATH)
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

    # ================================================================ UI
    def _build_ui(self):
        self._setup_styles()

        self.bg = GradientBackground(self.root, GRAD_TOP, GRAD_BOTTOM, CONTENT_BG, margin=14)
        self.bg.pack(fill="both", expand=True)
        content = self.bg.content
        content.grid_rowconfigure(0, weight=0, minsize=HEADER_MIN_H)
        content.grid_rowconfigure(1, weight=1)
        content.grid_columnconfigure(0, weight=1)

        self._build_header(content)

        self.body_container = tk.Frame(content, bg=content["bg"])
        self.body_container.grid(row=1, column=0, sticky="nsew")

        self.mode1_frame = tk.Frame(self.body_container, bg=content["bg"])
        self.mode2_frame = tk.Frame(self.body_container, bg=content["bg"])
        for f in (self.mode1_frame, self.mode2_frame):
            f.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._build_body(self.mode1_frame)
        self._build_mode2_body(self.mode2_frame)
        self.mode1_frame.tkraise()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._apply_power_visuals()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

    def _build_header(self, content):
        header_wrap = tk.Frame(content, bg=content["bg"], height=HEADER_MIN_H)
        header_wrap.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_wrap.pack_propagate(False)   # header Card inside uses pack(), so this is the flag that matters
        header_wrap.grid_propagate(False)   # belt-and-braces: also stop the grid resizing it

        header = Card(header_wrap, CARD_BG, radius=18)
        header.pack(fill="both", expand=True)

        left = tk.Frame(header.inner, bg=CARD_BG)
        left.pack(side="left", fill="y", padx=16, pady=4)
        tk.Label(left, text="🖐", bg=CARD_BG, fg=ACCENT_BRIGHT, font=("Segoe UI", 16)).pack(side="left", padx=(0, 8))
        title_box = tk.Frame(left, bg=CARD_BG)
        title_box.pack(side="left")
        self.title_label = tk.Label(title_box, text="Sign Language → Speech", bg=CARD_BG, fg=TEXT_LIGHT,
                                     font=("Segoe UI", 13, "bold"))
        self.title_label.pack(anchor="w")
        self.subtitle_label = tk.Label(title_box, text="live webcam  •  word-level AI  •  offline voice",
                                        bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 8))
        self.subtitle_label.pack(anchor="w")

        right = tk.Frame(header.inner, bg=CARD_BG)
        right.pack(side="right", fill="y", padx=16, pady=4)

        self.power_btn = tk.Button(right, text="⏻  Power: OFF", bg=RED_OFF, fg="#2a0505",
                                    activebackground=RED_OFF, bd=0, font=("Segoe UI", 10, "bold"),
                                    padx=14, pady=10, relief="flat", cursor="hand2",
                                    command=self.toggle_power)
        self.power_btn.pack(side="right", padx=(10, 0))

        self.fullscreen_btn = tk.Button(right, text="⛶", bg=CARD_BG_ALT, fg=TEXT_LIGHT, bd=0,
                                         font=("Segoe UI", 12, "bold"), padx=10, pady=10,
                                         cursor="hand2", command=self.toggle_fullscreen)
        self.fullscreen_btn.pack(side="right", padx=(10, 0))

        self.mode_btn = tk.Button(right, text="🔁 Mode: Sign → Speech", bg=CARD_BG_ALT, fg=TEXT_LIGHT, bd=0,
                                   font=("Segoe UI", 10, "bold"), padx=14, pady=10, relief="flat",
                                   cursor="hand2", command=self.toggle_mode)
        self.mode_btn.pack(side="right", padx=(10, 0))

        self.lang_var = tk.StringVar(value="English")
        self.lang_btn = tk.Menubutton(right, text=f"Language: {self.lang_var.get()}",
                                       bg=CARD_BG_ALT, fg=TEXT_LIGHT, activebackground=ACCENT,
                                       activeforeground="#03102b", bd=0, relief="flat",
                                       cursor="hand2", font=("Segoe UI", 10, "bold"),
                                       padx=14, pady=10, highlightthickness=0)
        lang_menu = tk.Menu(self.lang_btn, tearoff=0, bg=CARD_BG_ALT, fg=TEXT_LIGHT,
                             activebackground=ACCENT, activeforeground="#03102b",
                             font=("Segoe UI", 10), bd=0)
        for lang in self.LANGUAGES:
            lang_menu.add_command(label=lang, command=lambda l=lang: self._on_language_change(l))
        self.lang_btn.config(menu=lang_menu)
        self.lang_btn.pack(side="right", padx=(10, 0))

    def _build_body(self, content):
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        body = tk.Frame(content, bg=content["bg"])
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)

        self._build_video_card(body)
        self._build_sidebar_card(body)

    # -------------------------------------------------------- video card
    def _build_video_card(self, body):
        video_card = Card(body, "#000000", radius=20)
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))

        self.video_label = tk.Label(video_card.inner, bg="#000000", bd=0)
        self.video_label.pack(fill="both", expand=True)

        pill = tk.Frame(video_card.inner, bg=CARD_BG)
        pill.place(x=14, y=14)
        self.status_dot = PulsingDot(pill, color=RED_OFF)
        self.status_dot.pack(side="left", padx=(10, 6), pady=6)
        self.status_label = tk.Label(pill, text="Power off", bg=CARD_BG, fg=TEXT_LIGHT,
                                      font=("Segoe UI", 10, "bold"))
        self.status_label.pack(side="left", padx=(0, 12), pady=6)

        self.flash_label = tk.Label(video_card.inner, text="", bg=ACCENT, fg="#03102b",
                                     font=("Segoe UI", 20, "bold"), padx=22, pady=8)
        # shown on demand via _flash_word()

    # ------------------------------------------------------------ sidebar
    def _build_sidebar_card(self, body):
        sidebar_card = Card(body, CARD_BG, radius=20)
        sidebar_card.grid(row=0, column=1, sticky="nsew")
        sidebar = tk.Frame(sidebar_card.inner, bg=CARD_BG)
        sidebar.pack(fill="both", expand=True, padx=18, pady=16)

        # --- Building word (live, in-progress guess) ---
        tk.Label(sidebar, text="BUILDING WORD", bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.building_label = tk.Label(sidebar, text="—", bg=CARD_BG_ALT, fg=ACCENT_BRIGHT,
                                        anchor="w", font=("Consolas", 15, "bold"),
                                        padx=12, pady=10)
        self.building_label.pack(fill="x", pady=(4, 14))

        # --- Sentence + copy ---
        row = tk.Frame(sidebar, bg=CARD_BG)
        row.pack(fill="x", pady=(0, 6))
        tk.Label(row, text="SENTENCE", bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 9, "bold")).pack(side="left")
        self.copy_btn = tk.Button(row, text="⧉ Copy", bg=CARD_BG_ALT, fg=TEXT_LIGHT, bd=0,
                                   font=("Segoe UI", 9, "bold"), padx=10, pady=4,
                                   cursor="hand2", command=self.on_copy)
        self.copy_btn.pack(side="right")

        text_wrap = tk.Frame(sidebar, bg=CARD_BG_ALT)
        text_wrap.pack(fill="both", expand=True, pady=(4, 12))
        self.sentence_box = tk.Text(text_wrap, wrap="word", bg=CARD_BG_ALT, fg=TEXT_LIGHT,
                                     font=("Segoe UI", 14), bd=0, padx=12, pady=12,
                                     insertbackground=TEXT_LIGHT, height=1)
        self.sentence_box.pack(fill="both", expand=True)

        # --- Controls ---
        btn_row1 = tk.Frame(sidebar, bg=CARD_BG)
        btn_row1.pack(fill="x", pady=(0, 8))
        self.speak_btn = tk.Button(btn_row1, text="🔊 Speak", bg=ACCENT, fg="#03102b", bd=0,
                                    font=("Segoe UI", 11, "bold"), padx=10, pady=10,
                                    cursor="hand2", command=self.on_speak)
        self.speak_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.mute_btn = tk.Button(btn_row1, text="🔔 Unmuted", bg=CARD_BG_ALT, fg=TEXT_LIGHT, bd=0,
                                   font=("Segoe UI", 11, "bold"), padx=10, pady=10,
                                   cursor="hand2", command=self.toggle_mute)
        self.mute_btn.pack(side="left", expand=True, fill="x", padx=6)

        btn_row2 = tk.Frame(sidebar, bg=CARD_BG)
        btn_row2.pack(fill="x", pady=(0, 8))
        tk.Button(btn_row2, text="␣ Add Space", bg=CARD_BG_ALT, fg=TEXT_LIGHT, bd=0,
                  font=("Segoe UI", 10, "bold"), padx=10, pady=9, cursor="hand2",
                  command=self.on_space).pack(side="left", expand=True, fill="x", padx=(0, 6))
        tk.Button(btn_row2, text="⌫ Backspace", bg=CARD_BG_ALT, fg=TEXT_LIGHT, bd=0,
                  font=("Segoe UI", 10, "bold"), padx=10, pady=9, cursor="hand2",
                  command=self.on_backspace).pack(side="left", expand=True, fill="x", padx=6)

        tk.Button(sidebar, text="🗑 Clear All", bg=RED_OFF, fg="#2a0505", bd=0,
                  font=("Segoe UI", 10, "bold"), padx=10, pady=9, cursor="hand2",
                  command=self.on_clear).pack(fill="x", pady=(0, 8))

        self.footer_label = tk.Label(sidebar, text=self._model_status_text(),
                                      bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 8))
        self.footer_label.pack(anchor="w", side="bottom")

    def _model_status_text(self):
        return "Model: ready" if self.model is not None else "Model: NOT LOADED (see popup)"

    @staticmethod
    def _set_enabled(btn, enabled, enabled_bg, enabled_fg):
        """Toggles a button between a clearly-visible enabled look and a
        clearly-visible (but obviously muted) disabled look. Tk renders a
        disabled button's text using `disabledforeground`, not `fg` - both
        must be set explicitly or the text can end up unreadable."""
        if enabled:
            btn.config(state="normal", bg=enabled_bg, fg=enabled_fg, cursor="hand2")
        else:
            btn.config(state="disabled", bg=CARD_BG_ALT, fg=TEXT_MUTED,
                       disabledforeground=TEXT_MUTED, cursor="arrow")

    # =========================================================== MODE 2 UI
    def _build_mode2_body(self, parent):
        body = tk.Frame(parent, bg=parent["bg"])
        body.pack(fill="both", expand=True)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)

        # --- avatar card (left) ---
        avatar_card = Card(body, "#000000", radius=20)
        avatar_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.avatar = HumanAvatar(avatar_card.inner, bg="#000000")
        self.avatar.pack(fill="both", expand=True)

        pill = tk.Frame(avatar_card.inner, bg=CARD_BG)
        pill.place(x=14, y=14)
        self.mode2_status_dot = PulsingDot(pill, color=RED_OFF)
        self.mode2_status_dot.pack(side="left", padx=(10, 6), pady=6)
        self.mode2_status_label = tk.Label(pill, text="Power off", bg=CARD_BG, fg=TEXT_LIGHT,
                                            font=("Segoe UI", 10, "bold"))
        self.mode2_status_label.pack(side="left", padx=(0, 12), pady=6)

        # --- right control panel ---
        panel_card = Card(body, CARD_BG, radius=20)
        panel_card.grid(row=0, column=1, sticky="nsew")
        panel = tk.Frame(panel_card.inner, bg=CARD_BG)
        panel.pack(fill="both", expand=True, padx=18, pady=16)

        # input source dropdown
        tk.Label(panel, text="INPUT SOURCE", bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.source_var = tk.StringVar(value="🎙 Microphone")
        self.source_btn = tk.Menubutton(panel, textvariable=self.source_var, bg=CARD_BG_ALT,
                                         fg=TEXT_LIGHT, activebackground=ACCENT, activeforeground="#03102b",
                                         bd=0, relief="flat", cursor="hand2", anchor="w",
                                         font=("Segoe UI", 10, "bold"), padx=12, pady=9)
        source_menu = tk.Menu(self.source_btn, tearoff=0, bg=CARD_BG_ALT, fg=TEXT_LIGHT,
                               activebackground=ACCENT, activeforeground="#03102b", font=("Segoe UI", 10))
        source_menu.add_command(label="🎙 Microphone", command=lambda: self._select_source("microphone"))
        source_menu.add_command(label="🔊 Audio File", command=lambda: self._select_source("audio_file"))
        self.source_btn.config(menu=source_menu)
        self.source_btn.pack(fill="x", pady=(4, 14))

        self.source_panel_holder = tk.Frame(panel, bg=CARD_BG)
        self.source_panel_holder.pack(fill="x")

        self._build_mic_panel(self.source_panel_holder)
        self._build_audio_panel(self.source_panel_holder)
        self.mic_panel.pack(fill="x")  # microphone is the default source

        # transcript
        row = tk.Frame(panel, bg=CARD_BG)
        row.pack(fill="x", pady=(14, 6))
        tk.Label(row, text="TRANSCRIPT", bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 9, "bold")).pack(side="left")
        self.transcript_copy_btn = tk.Button(row, text="⧉ Copy", bg=CARD_BG_ALT, fg=TEXT_LIGHT, bd=0,
                                              font=("Segoe UI", 9, "bold"), padx=10, pady=4,
                                              cursor="hand2", command=self.on_copy_transcript)
        self.transcript_copy_btn.pack(side="right")

        text_wrap = tk.Frame(panel, bg=CARD_BG_ALT)
        text_wrap.pack(fill="both", expand=True, pady=(4, 12))
        self.transcript_box = tk.Text(text_wrap, wrap="word", bg=CARD_BG_ALT, fg=TEXT_LIGHT,
                                       font=("Segoe UI", 13), bd=0, padx=12, pady=12,
                                       insertbackground=TEXT_LIGHT, height=1)
        self.transcript_box.pack(fill="both", expand=True)

        # overlay shown ONLY while an audio file is being transcribed
        self.transcribe_overlay = tk.Frame(text_wrap, bg=CARD_BG_ALT)
        tk.Label(self.transcribe_overlay, text="Transcribing audio…", bg=CARD_BG_ALT, fg=TEXT_LIGHT,
                 font=("Segoe UI", 11, "bold")).pack(pady=(0, 10))
        self.transcribe_progress_canvas = tk.Canvas(self.transcribe_overlay, width=220, height=10,
                                                     bg=ACCENT_SOFT, highlightthickness=0)
        self.transcribe_progress_canvas.pack()
        self.transcribe_progress_label = tk.Label(self.transcribe_overlay, text="0%", bg=CARD_BG_ALT,
                                                   fg=TEXT_MUTED, font=("Segoe UI", 9))
        self.transcribe_progress_label.pack(pady=(6, 0))

        self.signs_btn = tk.Button(panel, text="▶ Play Signs", bg=ACCENT, fg="#03102b", bd=0,
                                    font=("Segoe UI", 11, "bold"), padx=10, pady=10, cursor="hand2",
                                    disabledforeground=TEXT_MUTED, command=self.on_toggle_signs_playback)
        self.signs_btn.pack(fill="x")
        self._set_enabled(self.signs_btn, False, ACCENT, "#03102b")

        self.root.after(200, self._update_timeline)

    # ------------------------------------------------------------ mic panel
    def _build_mic_panel(self, holder):
        self.mic_panel = tk.Frame(holder, bg=CARD_BG)

        tk.Label(self.mic_panel, text="MICROPHONE", bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.mic_var = tk.StringVar(value="No microphone found")
        self.mic_btn = tk.Menubutton(self.mic_panel, textvariable=self.mic_var, bg=CARD_BG_ALT,
                                      fg=TEXT_LIGHT, activebackground=ACCENT, activeforeground="#03102b",
                                      bd=0, relief="flat", cursor="hand2", anchor="w",
                                      font=("Segoe UI", 10, "bold"), padx=12, pady=9)
        self.mic_menu = tk.Menu(self.mic_btn, tearoff=0, bg=CARD_BG_ALT, fg=TEXT_LIGHT,
                                 activebackground=ACCENT, activeforeground="#03102b", font=("Segoe UI", 10))
        self.mic_btn.config(menu=self.mic_menu)
        self.mic_btn.pack(fill="x", pady=(4, 10))

        self.mic_mute_btn = tk.Button(self.mic_panel, text="🔇 Mic Muted", bg=RED_OFF, fg="#2a0505", bd=0,
                                      font=("Segoe UI", 11, "bold"), padx=10, pady=10, cursor="hand2",
                                      command=self.toggle_mic_mute)
        self.mic_mute_btn.pack(fill="x", pady=(0, 4))

        self._refresh_mic_devices()

    def _refresh_mic_devices(self):
        try:
            self.mic_devices = stt_utils.list_microphones()
        except Exception:
            self.mic_devices = []
        self.mic_menu.delete(0, "end")
        if not self.mic_devices:
            self.mic_menu.add_command(label="No microphones found", state="disabled")
            self.mic_var.set("No microphone found")
            return
        for idx, name in self.mic_devices:
            label = name[:42] if name else f"Device {idx}"
            self.mic_menu.add_command(label=label, command=lambda i=idx, n=label: self._select_mic(i, n))
        if self.selected_mic_index is None:
            first_idx, first_name = self.mic_devices[0]
            self._select_mic(first_idx, (first_name or f"Device {first_idx}")[:42])

    def _select_mic(self, index, name):
        self.selected_mic_index = index
        self.mic_var.set(f"🎙 {name}")
        self._sync_mic_state()
        self._apply_mode2_status()

    def toggle_mic_mute(self):
        self.mic_muted = not self.mic_muted
        if self.mic_muted:
            self.mic_mute_btn.config(text="🔇 Mic Muted", bg=RED_OFF, fg="#2a0505")
        else:
            self.mic_mute_btn.config(text="🎙 Mic Unmuted", bg=GREEN_ON, fg="#04240f")
        self._sync_mic_state()
        self._apply_mode2_status()

    def _sync_mic_state(self):
        should_listen = (self.mode == "speech_to_sign" and self.input_source == "microphone"
                         and self.power_on and not self.mic_muted and self.selected_mic_index is not None)
        if should_listen and self.live_stt is None:
            self.live_stt = stt_utils.LiveSTT(self.selected_mic_index, self._on_stt_words, self._on_stt_error)
            self.live_stt.start()
        elif not should_listen and self.live_stt is not None:
            self.live_stt.stop()
            self.live_stt = None

    def _on_stt_words(self, words):
        self.root.after(0, lambda: self._handle_stt_words(words))

    def _handle_stt_words(self, words):
        vocab = set(self.labels.values())
        for w in words:
            self.transcript += w + " "
            if w in vocab:
                self.avatar_queue.append(w)
        self._refresh_transcript_box()
        self._set_enabled(self.signs_btn, bool(self.transcript.strip()), ACCENT, "#03102b")
        self._drive_avatar_queue()

    def _on_stt_error(self, msg):
        self.root.after(0, lambda: self.mode2_status_label.config(text="Mic error - check connection"))

    def _drive_avatar_queue(self):
        if self.avatar_busy or not self.avatar_queue:
            return
        word = self.avatar_queue.popleft()
        sample = get_word_sample(word)
        self.avatar_busy = True

        def done():
            self.avatar_busy = False
            if self.avatar_queue:
                self._drive_avatar_queue()
            else:
                self.avatar.go_idle()
                self._apply_mode2_status()

        if sample is not None:
            self.avatar.play_sequence(sample, on_complete=done)
        else:
            self.mode2_status_label.config(
                text=f"No recorded gesture for '{word}' - re-run make_demo_model.py / collect_data.py")
            self.avatar.go_idle()
            self.root.after(1500, done)

    # ---------------------------------------------------------- audio panel
    def _build_audio_panel(self, holder):
        self.audio_panel = tk.Frame(holder, bg=CARD_BG)

        tk.Label(self.audio_panel, text="AUDIO FILE", bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")

        path_row = tk.Frame(self.audio_panel, bg=CARD_BG)
        path_row.pack(fill="x", pady=(4, 10))
        self.audio_path_var = tk.StringVar(value="No file selected")
        path_entry = tk.Entry(path_row, textvariable=self.audio_path_var, state="readonly",
                               readonlybackground=CARD_BG_ALT, fg=TEXT_LIGHT, bd=0,
                               font=("Segoe UI", 9), insertbackground=TEXT_LIGHT)
        path_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

        # Built from two separately-centred Labels (instead of one emoji+text
        # string) so the icon and text share the same baseline reliably -
        # mixing an emoji glyph into a single Label's text was rendering the
        # icon at a different vertical position than the text next to it.
        self.browse_btn = tk.Frame(path_row, bg=CARD_BG_ALT, cursor="hand2")
        self.browse_btn.pack(side="right", ipady=6, ipadx=10)
        browse_icon = tk.Label(self.browse_btn, text="📁", bg=CARD_BG_ALT, fg=TEXT_LIGHT,
                                font=("Segoe UI Emoji", 10))
        browse_icon.pack(side="left", padx=(0, 6))
        browse_text = tk.Label(self.browse_btn, text="Browse...", bg=CARD_BG_ALT, fg=TEXT_LIGHT,
                                font=("Segoe UI", 9, "bold"))
        browse_text.pack(side="left")
        for w in (self.browse_btn, browse_icon, browse_text):
            w.bind("<Button-1>", lambda e: self.on_browse_audio())

        self.timeline = tk.Canvas(self.audio_panel, height=22, bg=TIMELINE_TRACK, highlightthickness=0)
        self.timeline.pack(fill="x", pady=(0, 4))
        self.timeline.bind("<Button-1>", self._on_timeline_click)
        self.timeline.bind("<B1-Motion>", self._on_timeline_click)

        self.time_label = tk.Label(self.audio_panel, text="0:00 / 0:00", bg=CARD_BG, fg=TEXT_MUTED,
                                    font=("Segoe UI", 8))
        self.time_label.pack(anchor="e", pady=(0, 10))

        controls = tk.Frame(self.audio_panel, bg=CARD_BG)
        controls.pack(fill="x", pady=(0, 4))
        self.audio_back_btn = tk.Button(controls, text="⏪ 5s", bd=0,
                                         font=("Segoe UI", 10, "bold"), padx=10, pady=9,
                                         command=lambda: self.skip_audio(-5))
        self.audio_back_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.audio_play_btn = tk.Button(controls, text="▶ Play Audio", bd=0,
                                         font=("Segoe UI", 10, "bold"), padx=10, pady=9,
                                         command=self.toggle_audio_playback)
        self.audio_play_btn.pack(side="left", expand=True, fill="x", padx=6)
        self.audio_fwd_btn = tk.Button(controls, text="5s ⏩", bd=0,
                                        font=("Segoe UI", 10, "bold"), padx=10, pady=9,
                                        command=lambda: self.skip_audio(5))
        self.audio_fwd_btn.pack(side="left", expand=True, fill="x", padx=(6, 0))

        speed_row = tk.Frame(self.audio_panel, bg=CARD_BG)
        speed_row.pack(fill="x", pady=(8, 4))
        tk.Label(speed_row, text="Speed", bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 8)).pack(side="left")
        self.speed_var = tk.StringVar(value="1.0x")
        self.speed_btn = tk.Menubutton(speed_row, textvariable=self.speed_var, bg=CARD_BG_ALT,
                                        fg=TEXT_LIGHT, activebackground=ACCENT, activeforeground="#03102b",
                                        bd=0, relief="flat", font=("Segoe UI", 9, "bold"), padx=10, pady=5)
        speed_menu = tk.Menu(self.speed_btn, tearoff=0, bg=CARD_BG_ALT, fg=TEXT_LIGHT,
                              activebackground=ACCENT, activeforeground="#03102b", font=("Segoe UI", 9))
        for factor in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            label = f"{factor}x"
            speed_menu.add_command(label=label, command=lambda f=factor, l=label: self._select_speed(f, l))
        self.speed_btn.config(menu=speed_menu)
        self.speed_btn.pack(side="right")

        self._set_audio_controls_enabled(False)

        self.audio_status_label = tk.Label(self.audio_panel, text="Select an audio file to begin",
                                            bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 8))
        self.audio_status_label.pack(anchor="w", pady=(8, 0))

    def _select_speed(self, factor, label):
        self.speed_var.set(label)
        if self.audio_player is not None:
            self.audio_player.set_speed(factor)   # keeps playing (if already playing), no restart needed

    def _set_audio_controls_enabled(self, enabled):
        self._set_enabled(self.audio_back_btn, enabled, CARD_BG_ALT, TEXT_LIGHT)
        self._set_enabled(self.audio_play_btn, enabled, ACCENT, "#03102b")
        self._set_enabled(self.audio_fwd_btn, enabled, CARD_BG_ALT, TEXT_LIGHT)
        if enabled:
            self.speed_btn.config(state="normal", bg=CARD_BG_ALT, fg=TEXT_LIGHT, cursor="hand2")
        else:
            self.speed_btn.config(state="disabled", bg=CARD_BG_ALT, fg=TEXT_MUTED,
                                   disabledforeground=TEXT_MUTED, cursor="arrow")

    def _select_source(self, source):
        if source == self.input_source:
            return
        # reset the source we're LEAVING before switching
        if self.input_source == "microphone":
            self._reset_mode2_mic_fields()
        else:
            self._reset_mode2_audio_fields()
        self._reset_mode2_transcript()

        self.input_source = source
        self.source_var.set("🎙 Microphone" if source == "microphone" else "🔊 Audio File")
        if source == "microphone":
            self.audio_panel.pack_forget()
            self.mic_panel.pack(fill="x")
        else:
            self.mic_panel.pack_forget()
            self.audio_panel.pack(fill="x")
        self._sync_mic_state()
        self._apply_mode2_status()

    def on_browse_audio(self):
        path = filedialog.askopenfilename(
            title="Select a WAV audio file",
            filetypes=[("WAV audio files", "*.wav"), ("All files", "*.*")])
        if not path:
            return
        if not path.lower().endswith(".wav"):
            messagebox.showinfo(
                "WAV files only",
                "Please select a .wav audio file only.\n\n"
                "Other formats (mp3, m4a, etc.) aren't supported here.")
            return
        self._reset_mode2_audio_fields()   # fresh start for the newly-selected file
        self._reset_mode2_transcript()
        if self.audio_player is None:
            self.audio_player = AudioPlayer()
        self.audio_path = path
        self.audio_path_var.set(path)
        try:
            self.audio_player.load(path)
        except Exception as e:
            messagebox.showerror("Audio error", f"Could not load audio file:\n{e}")
            self.audio_path = None
            self.audio_path_var.set("No file selected")
            return
        self._set_audio_controls_enabled(True)
        self.audio_status_label.config(text="Transcribing… (needs internet) - you can press Play already")

        self.audio_transcribe_token += 1
        token = self.audio_transcribe_token
        self.audio_transcribing = True
        self._show_transcribe_overlay()
        stt_utils.transcribe_file_async(
            path,
            lambda text: self._on_file_transcribed(text, token),
            lambda msg: self._on_file_transcribe_error(msg, token),
            on_progress=lambda done, total: self._on_transcribe_progress(done, total, token),
            on_chunk=lambda start, end, text: self._on_transcribe_chunk(start, end, text, token),
        )

    def _on_transcribe_chunk(self, start, end, text, token):
        self.root.after(0, lambda: self._apply_transcribe_chunk(start, end, text, token))

    def _apply_transcribe_chunk(self, start, end, text, token):
        if token != self.audio_transcribe_token:
            return  # a newer/cancelled request - ignore this stale chunk
        words = [w.strip(".,!?") for w in text.lower().split() if w.strip()]
        if words:
            span = (end - start) / len(words)
            for i, w in enumerate(words):
                self.audio_sync_slots.append((start + i * span, start + (i + 1) * span, w))
            # progressively fill the transcript box too - "live captioning" as chunks arrive
            self.transcript += text.lower() + " "
            self._refresh_transcript_box()
            self._set_enabled(self.signs_btn, True, ACCENT, "#03102b")

    def _on_transcribe_progress(self, done, total, token):
        self.root.after(0, lambda: self._apply_transcribe_progress(done, total, token))

    def _apply_transcribe_progress(self, done, total, token):
        if token != self.audio_transcribe_token:
            return  # a newer/cancelled request - ignore this stale update
        pct = int(100 * done / total) if total else 0
        self._update_transcribe_progress(pct)

    def _show_transcribe_overlay(self):
        self._update_transcribe_progress(0)
        self.transcribe_overlay.place(relx=0.5, rely=0.5, anchor="c")

    def _hide_transcribe_overlay(self):
        self.transcribe_overlay.place_forget()

    def _update_transcribe_progress(self, pct):
        pct = max(0, min(100, pct))
        c = self.transcribe_progress_canvas
        c.delete("all")
        w = int(c["width"])
        h = int(c["height"])
        c.create_rectangle(0, 0, w, h, fill=ACCENT_SOFT, outline="")
        c.create_rectangle(0, 0, w * pct / 100, h, fill=ACCENT, outline="")
        self.transcribe_progress_label.config(text=f"{pct}%")

    def _on_file_transcribed(self, text, token):
        self.root.after(0, lambda: self._apply_file_transcript(text, token))

    def _apply_file_transcript(self, text, token):
        if token != self.audio_transcribe_token:
            return  # user switched mode/source mid-transcription - discard this stale result
        # self.transcript / audio_sync_slots were already built incrementally,
        # chunk by chunk, as transcription progressed - this just finalises status.
        self.audio_transcribing = False
        self._hide_transcribe_overlay()
        self._set_enabled(self.signs_btn, bool(self.transcript.strip()), ACCENT, "#03102b")
        self.audio_status_label.config(text="Ready" if self.transcript.strip() else "Could not understand the audio")

    def _on_file_transcribe_error(self, msg, token):
        self.root.after(0, lambda: self._apply_file_transcribe_error(msg, token))

    def _apply_file_transcribe_error(self, msg, token):
        if token != self.audio_transcribe_token:
            return
        self.audio_transcribing = False
        self._hide_transcribe_overlay()
        self.audio_status_label.config(text="Transcription failed - check internet")

    def toggle_audio_playback(self):
        if self.audio_player is None or self.audio_path is None:
            return
        if self.audio_player.is_playing():
            self.audio_player.pause()
            self.audio_play_btn.config(text="▶ Play Audio")
        else:
            if self.replay_active:
                self._pause_signs_playback()
            self.audio_player.play()
            self.audio_play_btn.config(text="⏸ Pause Audio")

    def skip_audio(self, delta):
        if self.audio_player is not None:
            self.audio_player.skip(delta)

    def _on_timeline_click(self, event):
        if self.audio_player is None or not self.audio_player.duration:
            return
        w = max(self.timeline.winfo_width(), 1)
        frac = max(0.0, min(1.0, event.x / w))
        self.audio_player.seek(frac * self.audio_player.duration)

    def _update_timeline(self):
        if self.audio_player is not None and self.audio_path is not None:
            dur = self.audio_player.duration or 0.0
            pos = self.audio_player.position() if dur else 0.0
            self._draw_timeline((pos / dur) if dur else 0.0)
            self.time_label.config(text=f"{self._fmt_time(pos)} / {self._fmt_time(dur)}")
            self._sync_avatar_to_playback(pos)
            if self.audio_player.is_finished():
                self.audio_player.stop()
                self.audio_play_btn.config(text="▶ Play Audio")
                self.audio_sync_last_slot = None
                self.avatar.go_idle()
        self.root.after(200, self._update_timeline)

    def _sync_avatar_to_playback(self, pos):
        """
        Shows whichever word's actual (approximate) time slot the current
        playback position falls into. Slots are built incrementally as
        each ~8-second chunk gets transcribed in the background, so this
        starts working within moments of pressing Play - it doesn't wait
        for the whole file to finish transcribing first. Because `pos`
        already accounts for the current speed (see AudioPlayer.position),
        pausing/seeking/changing speed all just naturally shift which word
        is shown. Word start/end times within each chunk are evenly spread
        across that chunk's known duration (an approximation - real speech
        isn't perfectly evenly spaced - since the free speech API doesn't
        give true per-word timestamps).
        """
        if not self.audio_sync_slots or not self.audio_player.is_playing():
            return
        match = None
        for slot in self.audio_sync_slots:
            if slot[0] <= pos < slot[1]:
                match = slot
                break
        if match is None or match == self.audio_sync_last_slot:
            return
        self.audio_sync_last_slot = match
        sample = get_word_sample(match[2])
        if sample is not None:
            self.avatar.play_sequence(sample)
        else:
            self.avatar.go_idle()

    @staticmethod
    def _fmt_time(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 60}:{seconds % 60:02d}"

    def _draw_timeline(self, frac):
        if not hasattr(self, "timeline"):
            return
        self.timeline.delete("all")
        w = max(self.timeline.winfo_width(), 10)
        h = max(self.timeline.winfo_height(), 10)
        self.timeline.create_rectangle(0, 0, w, h, fill=TIMELINE_TRACK, outline="")
        self.timeline.create_rectangle(0, 0, w * frac, h, fill=TIMELINE_FILL, outline="")
        cx = w * frac
        self.timeline.create_oval(cx - 6, h / 2 - 6, cx + 6, h / 2 + 6, fill=TEXT_LIGHT, outline="")

    # ---------------------------------------------------------- transcript
    def _refresh_transcript_box(self):
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("1.0", self.transcript)

    def on_copy_transcript(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.transcript.strip())
        original = self.transcript_copy_btn.cget("text")
        self.transcript_copy_btn.config(text="✓ Copied!")
        self.root.after(1200, lambda: self.transcript_copy_btn.config(text=original))

    # ------------------------------------------------------- signs playback
    def on_toggle_signs_playback(self):
        if self.replay_active:
            self._pause_signs_playback()
        else:
            self._start_signs_playback()

    def _start_signs_playback(self):
        if self.audio_player is not None and self.audio_player.is_playing():
            self.audio_player.pause()
            self.audio_play_btn.config(text="▶ Play Audio")
        import re
        text = self.transcript_box.get("1.0", "end-1c")
        words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        if not words:
            return
        cursor_offset = len(self.transcript_box.get("1.0", tk.INSERT))
        start_i = 0
        for i, (_, s, e) in enumerate(words):
            if e > cursor_offset:
                start_i = i
                break
        else:
            start_i = 0
        self.replay_words = words
        self.replay_index = start_i
        self.replay_active = True
        self.signs_btn.config(text="⏸ Pause Signs")
        self._play_next_replay_word()

    def _play_next_replay_word(self):
        if not self.replay_active or self.replay_index >= len(self.replay_words):
            finished_naturally = self.replay_active
            self._pause_signs_playback(reset=True)
            if finished_naturally:
                self.avatar.go_idle()
            return
        word, s, e = self.replay_words[self.replay_index]
        self.transcript_box.tag_remove("current_word", "1.0", "end")
        start_idx, end_idx = f"1.0+{s}c", f"1.0+{e}c"
        self.transcript_box.tag_add("current_word", start_idx, end_idx)
        self.transcript_box.tag_config("current_word", background=ACCENT, foreground="#03102b")
        self.transcript_box.mark_set(tk.INSERT, end_idx)
        self.transcript_box.see(end_idx)

        sample = get_word_sample(word.lower().strip(".,!?"))

        def after_word():
            if not self.replay_active:
                return
            self.replay_index += 1
            self.root.after(150, self._play_next_replay_word)

        if sample is not None:
            self.avatar.play_sequence(sample, on_complete=after_word)
        else:
            self.avatar.go_idle()
            self.root.after(250, after_word)

    def _pause_signs_playback(self, reset=False):
        self.replay_active = False
        self.avatar.stop()
        self.signs_btn.config(text="▶ Play Signs")
        if reset:
            self.transcript_box.tag_remove("current_word", "1.0", "end")

    # ------------------------------------------------------------ resets
    def _reset_mode1_fields(self):
        self.speech.interrupt()
        self.sentence = ""
        self.building_label.config(text="—")
        self._refresh_sentence_box()
        self.history.clear()

    def _reset_mode2_mic_fields(self):
        if self.live_stt is not None:
            self.live_stt.stop()
            self.live_stt = None
        self.mic_muted = True
        self.mic_mute_btn.config(text="🔇 Mic Muted", bg=RED_OFF, fg="#2a0505")
        self.avatar_queue.clear()

    def _reset_mode2_audio_fields(self):
        if self.audio_player is not None:
            self.audio_player.stop()
        self.audio_transcribe_token += 1   # invalidate any transcription still running in the background
        self.audio_transcribing = False
        self._hide_transcribe_overlay()
        self.audio_path = None
        self.audio_path_var.set("No file selected")
        self.audio_play_btn.config(text="▶ Play Audio")
        self.audio_sync_slots = []
        self.audio_sync_last_slot = None
        self.speed_var.set("1.0x")
        self._set_audio_controls_enabled(False)
        self._draw_timeline(0.0)
        self.time_label.config(text="0:00 / 0:00")
        self.audio_status_label.config(text="Select an audio file to begin")

    def _reset_mode2_transcript(self):
        self._pause_signs_playback(reset=True)
        self.avatar.go_idle()
        self.transcript = ""
        self._refresh_transcript_box()
        self._set_enabled(self.signs_btn, False, ACCENT, "#03102b")

    # ------------------------------------------------------------ mode swap
    def toggle_mode(self):
        if self.mode == "sign_to_speech":
            self._reset_mode1_fields()   # reset the mode we're LEAVING
            self.mode = "speech_to_sign"
            self.mode_btn.config(text="🔁 Mode: Speech → Sign", bg=ACCENT, fg="#03102b")
            self.title_label.config(text="Speech → Sign Language")
            self.subtitle_label.config(text="microphone or audio file  •  AI-matched signs  •  avatar playback")
            self.mode2_frame.tkraise()
            if self.stream is not None:   # camera isn't needed in this mode - switch it off
                self.stream.stop()
                self.stream = None
        else:
            self._reset_mode2_mic_fields()   # reset the mode we're LEAVING
            self._reset_mode2_audio_fields()
            self._reset_mode2_transcript()
            self.mode = "sign_to_speech"
            self.mode_btn.config(text="🔁 Mode: Sign → Speech", bg=CARD_BG_ALT, fg=TEXT_LIGHT)
            self.title_label.config(text="Sign Language → Speech")
            self.subtitle_label.config(text="live webcam  •  word-level AI  •  offline voice")
            self.mode1_frame.tkraise()
            if self.stream is None:   # camera is needed again - switch it back on
                try:
                    self.stream = VideoStream(CAM_INDEX)
                except RuntimeError as e:
                    messagebox.showerror("Camera error", str(e))
                    self.stream = None
        self._sync_mic_state()
        self._apply_mode2_status()

    def _apply_mode2_status(self):
        if not hasattr(self, "mode2_status_label"):
            return
        if not self.power_on:
            self.mode2_status_dot.set_color(RED_OFF)
            self.mode2_status_label.config(text="Power off")
        elif self.input_source == "microphone":
            if self.mic_muted:
                self.mode2_status_dot.set_color(AMBER)
                self.mode2_status_label.config(text="Mic muted")
            elif self.live_stt is not None:
                self.mode2_status_dot.set_color(GREEN_ON)
                self.mode2_status_label.config(text="Listening…")
            else:
                self.mode2_status_dot.set_color(AMBER)
                self.mode2_status_label.config(text="No microphone selected")
        else:
            self.mode2_status_dot.set_color(GREEN_ON)
            self.mode2_status_label.config(text="Audio file mode")

    # ------------------------------------------------------------ shortcuts
    def _bind_shortcuts(self):
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda e: self._exit_fullscreen())

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
        self.fullscreen_btn.config(bg=ACCENT if self.is_fullscreen else CARD_BG_ALT)

    def _exit_fullscreen(self):
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.root.attributes("-fullscreen", False)
            self.fullscreen_btn.config(bg=CARD_BG_ALT)

    # ============================================================ main loop
    def _tick(self):
        if self.stream is not None:
            frame, landmarks, hand_seen = self.stream.read()
            if frame is not None:
                self._process_frame(frame, landmarks, hand_seen)
        self.root.after(15, self._tick)

    def _process_frame(self, frame, landmarks, hand_seen):
        if self.mode == "sign_to_speech" and self.power_on and self.model is not None:
            if hand_seen:
                self.history.append(landmarks)
                self.status_label.config(text="Reading gesture…")
                self.status_dot.set_color(AMBER)
            else:
                self.history.clear()
                self.status_label.config(text="Show a hand sign")
                self.status_dot.set_color(GREEN_ON)
                self.building_label.config(text="—")

            if len(self.history) == SEQUENCE_LENGTH:
                self._predict_and_maybe_commit()
        self._render(frame)

    def _predict_and_maybe_commit(self):
        seq = np.expand_dims(np.array(self.history, dtype=np.float32), axis=0)
        preds = self.model.predict(seq, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        word = self.labels.get(str(idx), "?")

        # live preview, shown even before it's confident/committed
        self.building_label.config(text=f"{word}   ({conf * 100:.0f}%)")

        now = time.time()
        if conf >= CONFIDENCE_THRESHOLD and (now - self.last_commit_time) >= COMMIT_COOLDOWN:
            self.last_commit_time = now
            self.history.clear()
            self._commit_word(word, conf)

    def _commit_word(self, word, conf):
        self.speech.interrupt()  # a brand-new gesture always interrupts current speech
        self.sentence += word + " "
        self._refresh_sentence_box(flash=True)
        self._flash_word(word, conf)
        self.building_label.config(text="—")
        if not self.muted:
            self.speech.speak(word)

    # ============================================================ rendering
    def _render(self, frame):
        target_w = self.video_label.winfo_width()
        target_h = self.video_label.winfo_height()
        if target_w < 4 or target_h < 4:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = cover_resize(img, target_w, target_h)
        photo = ImageTk.PhotoImage(img)
        self.video_label.imgtk = photo
        self.video_label.configure(image=photo)

    def _flash_word(self, word, conf):
        self.flash_label.config(text=f"{word}   {conf * 100:.0f}%")
        self.flash_label.place(relx=0.5, rely=0.92, anchor="s")
        self.root.after(1400, lambda: self.flash_label.place_forget())

    def _refresh_sentence_box(self, flash=False):
        self.sentence_box.delete("1.0", "end")
        self.sentence_box.insert("1.0", self.sentence)
        if flash:
            self._flash_widget_bg(self.sentence_box, CARD_BG_ALT, ACCENT_SOFT)

    def _flash_widget_bg(self, widget, base_color, flash_color, steps=6):
        def step(i):
            if not widget.winfo_exists():
                return
            t = 1 - (i / steps)
            widget.configure(bg=lerp_color(base_color, flash_color, t))
            if i < steps:
                self.root.after(45, lambda: step(i + 1))
        step(0)

    # ============================================================ controls
    def toggle_power(self):
        self.power_on = not self.power_on
        self.history.clear()
        self.building_label.config(text="—")
        self._apply_power_visuals()
        self._sync_mic_state()
        self._apply_mode2_status()

    def _apply_power_visuals(self):
        if self.power_on:
            self.power_btn.config(text="⏻  Power: ON", bg=GREEN_ON, fg="#04240f")
            self.status_dot.set_color(GREEN_ON)
            self.status_label.config(text="Show a hand sign")
        else:
            self.power_btn.config(text="⏻  Power: OFF", bg=RED_OFF, fg="#2a0505")
            self.status_dot.set_color(RED_OFF)
            self.status_label.config(text="Power off")

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self.mute_btn.config(text="🔕 Muted", bg=RED_OFF, fg="#2a0505")
        else:
            self.mute_btn.config(text="🔔 Unmuted", bg=CARD_BG_ALT, fg=TEXT_LIGHT)

    def on_speak(self):
        self.speech.speak(self.sentence)

    def _on_speech_state_change(self, speaking):
        self.is_speaking = speaking
        self.root.after(0, self._update_speak_button)

    def _update_speak_button(self):
        self.speak_btn.config(text="🔊 Speaking…" if self.is_speaking else "🔊 Speak")

    def on_copy(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.sentence.strip())
        original = self.copy_btn.cget("text")
        self.copy_btn.config(text="✓ Copied!")
        self.root.after(1200, lambda: self.copy_btn.config(text=original))

    def on_space(self):
        self.sentence = (self.sentence.rstrip(" ") + " ") if self.sentence.strip() else self.sentence
        self.sentence += " " if not self.sentence.endswith("  ") else ""
        self._refresh_sentence_box()

    def on_backspace(self):
        """Removes the last recognised WORD (gestures produce whole words, not letters)."""
        trimmed = self.sentence.strip()
        if not trimmed:
            return
        words = trimmed.split(" ")
        words.pop()
        self.sentence = (" ".join(words) + " ") if words else ""
        self._refresh_sentence_box()

    def on_clear(self):
        self.speech.interrupt()
        self.sentence = ""
        self.building_label.config(text="—")
        self._refresh_sentence_box()

    def _on_language_change(self, selected_value=None):
        new_lang = selected_value or self.lang_var.get()
        if new_lang == self.current_language:
            return
        confirmed = messagebox.askyesno(
            "Change language",
            f"Switching to '{new_lang}' will CLEAR the current sentence and start fresh.\n\nContinue?")
        if confirmed:
            self.current_language = new_lang
            self.lang_var.set(new_lang)
            self.lang_btn.config(text=f"Language: {new_lang}")
            self.speech.interrupt()
            self.sentence = ""
            self._refresh_sentence_box()
        # if not confirmed, simply leave everything as-is (menu never changed the active language)

    def on_close(self):
        self.speech.interrupt()
        if self.live_stt is not None:
            self.live_stt.stop()
        if self.audio_player is not None:
            self.audio_player.stop()
        if self.stream is not None:
            self.stream.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    SignLanguageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
