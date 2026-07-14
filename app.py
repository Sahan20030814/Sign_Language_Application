"""
app.py
--------
Sign Language -> Sentence -> Speech  (word-level, v2.1)
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
from tkinter import ttk, messagebox

import pyttsx3
from tensorflow.keras.models import load_model  # type: ignore

from gesture_utils import (
    create_hands_detector, extract_landmarks, draw_landmarks,
    SEQUENCE_LENGTH, MODEL_PATH, LABELS_PATH,
)

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
        self._build_body(content)

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
        tk.Label(title_box, text="Sign Language → Speech", bg=CARD_BG, fg=TEXT_LIGHT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(title_box, text="live webcam  •  word-level AI  •  offline voice",
                 bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 8)).pack(anchor="w")

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
        body = tk.Frame(content, bg=content["bg"])
        body.grid(row=1, column=0, sticky="nsew")
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
        if self.power_on and self.model is not None:
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
        if self.stream is not None:
            self.stream.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    SignLanguageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
