"""
avatar.py
-----------
A stylized human-figure avatar drawn with plain Tkinter Canvas shapes
(head, hair, ears, eyes, nose, mouth, neck, shoulders, a formal long-
sleeve shirt, and two arms/hands). No 3D engine or external art assets -
everything is primitive shapes, so it works anywhere Tkinter does.

The body is always static. Only the HANDS move:
- IDLE (no word being signed): arms rest inward, hands folded together
  (one resting on top of the other), clearly visible.
- ACTIVE (a word is being signed): arms spread out and each hand is
  driven by a REAL recorded MediaPipe hand-landmark sequence (the same
  .npy files collect_data.py saves) - so the shape you see the avatar's
  hand make is an actual recording of a real hand doing that sign, just
  re-drawn on the avatar instead of the camera feed.
"""

import tkinter as tk

from gesture_utils import FEATURES_PER_HAND

SKIN = "#e0ad82"
SKIN_SHADOW = "#c99567"
HAIR = "#2b1c14"
SHIRT = "#f0a83c"
SHIRT_DARK = "#c9822a"
COLLAR = "#f2f2f2"
LINE = "#12162c"

# MediaPipe hand connections (pairs of landmark indices) - used to draw the
# hand skeleton as connected bones rather than a cloud of dots.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]


class HumanAvatar(tk.Canvas):
    def __init__(self, parent, bg="#000000", **kwargs):
        super().__init__(parent, highlightthickness=0, bd=0, bg=bg, **kwargs)
        self._anim_after_id = None
        self.bind("<Configure>", lambda e: self._draw_idle())
        self._draw_idle()

    # ------------------------------------------------------------ geometry
    def _dims(self):
        w = max(self.winfo_width(), 100)
        h = max(self.winfo_height(), 100)
        return w, h

    # ------------------------------------------------------------ body
    def _draw_body(self, converge=False):
        """
        Draws the static male figure (head/hair/face/shirt/arms).
        converge=True bends the arms inward to a resting point in front of
        the body (idle pose); converge=False spreads the wrists apart so
        two-handed signs render correctly during playback.
        Returns the (lx, ly), (rx, ry) wrist anchor points.
        """
        self.delete("all")
        w, h = self._dims()
        cx = w / 2
        top = h * 0.08

        head_r = min(w, h) * 0.125
        head_cy = top + head_r
        neck_w = head_r * 0.55

        shoulder_y = head_cy + head_r * 1.5
        shoulder_w = min(w, h) * 0.28
        torso_bottom = h * 0.99

        # --- shirt / torso ---
        self.create_polygon(
            cx - shoulder_w, shoulder_y,
            cx + shoulder_w, shoulder_y,
            cx + shoulder_w * 0.86, torso_bottom,
            cx - shoulder_w * 0.86, torso_bottom,
            fill=SHIRT, outline=LINE, width=2, smooth=False,
        )
        # collar
        self.create_polygon(
            cx - neck_w, shoulder_y - 4,
            cx, shoulder_y + head_r * 0.55,
            cx + neck_w, shoulder_y - 4,
            fill=COLLAR, outline=LINE,
        )
        # shirt center seam + buttons for a "formal shirt" feel
        self.create_line(cx, shoulder_y + head_r * 0.6, cx, torso_bottom, fill=SHIRT_DARK, width=2)
        for i in range(3):
            by = shoulder_y + head_r * 1.1 + i * head_r * 0.7
            self.create_oval(cx - 3, by - 3, cx + 3, by + 3, fill=SHIRT_DARK, outline="")

        # --- neck ---
        self.create_rectangle(cx - neck_w * 0.5, head_cy + head_r * 0.7,
                               cx + neck_w * 0.5, shoulder_y,
                               fill=SKIN, outline="")

        # --- arms (sleeves) down to wrist anchor points ---
        arm_len = h * 0.30
        if converge:
            rest_y = torso_bottom - h * 0.06
            l_wrist = (cx - head_r * 0.35, rest_y)
            r_wrist = (cx + head_r * 0.35, rest_y)
        else:
            l_wrist = (cx - shoulder_w * 1.15, shoulder_y + arm_len)
            r_wrist = (cx + shoulder_w * 1.15, shoulder_y + arm_len)

        for (wx, wy), side in ((l_wrist, -1), (r_wrist, 1)):
            elbow_x = cx + side * shoulder_w * (0.98 if not converge else 1.02)
            elbow_y = shoulder_y + arm_len * 0.55
            self.create_line(cx + side * shoulder_w * 0.85, shoulder_y + 6, elbow_x, elbow_y, wx, wy - 10,
                              fill=SHIRT, width=int(shoulder_w * 0.30), capstyle="round",
                              joinstyle="round", smooth=True)
            self.create_line(cx + side * shoulder_w * 0.85, shoulder_y + 6, elbow_x, elbow_y, wx, wy - 10,
                              fill=SHIRT_DARK, width=2, smooth=True)

        # --- head + hair + face ---
        self.create_oval(cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r,
                          fill=SKIN, outline=LINE, width=2, tags="head")
        # ears
        for side in (-1, 1):
            ex = cx + side * head_r * 0.98
            self.create_oval(ex - head_r * 0.14, head_cy - head_r * 0.12,
                              ex + head_r * 0.14, head_cy + head_r * 0.28,
                              fill=SKIN, outline=LINE)
        # hair (top cap, simple "male" style)
        self.create_arc(cx - head_r * 1.05, head_cy - head_r * 1.3,
                         cx + head_r * 1.05, head_cy + head_r * 0.2,
                         start=0, extent=180, fill=HAIR, outline=HAIR)
        self.create_arc(cx - head_r * 1.05, head_cy - head_r * 1.05,
                         cx + head_r * 1.05, head_cy + head_r * 0.35,
                         start=15, extent=150, fill=HAIR, outline=HAIR)
        # eyes
        eye_y = head_cy - head_r * 0.05
        for side in (-1, 1):
            ex = cx + side * head_r * 0.38
            self.create_oval(ex - head_r * 0.11, eye_y - head_r * 0.08,
                              ex + head_r * 0.11, eye_y + head_r * 0.08, fill="white", outline=LINE)
            self.create_oval(ex - head_r * 0.05, eye_y - head_r * 0.05,
                              ex + head_r * 0.05, eye_y + head_r * 0.05, fill=LINE, outline="")
        # eyebrows
        for side in (-1, 1):
            ex = cx + side * head_r * 0.38
            self.create_line(ex - head_r * 0.17, eye_y - head_r * 0.24,
                              ex + head_r * 0.17, eye_y - head_r * 0.28, fill=HAIR, width=3)
        # nose
        self.create_line(cx, eye_y + head_r * 0.05, cx - head_r * 0.06, eye_y + head_r * 0.42,
                          fill=LINE, width=2)
        self.create_line(cx - head_r * 0.06, eye_y + head_r * 0.42, cx + head_r * 0.09, eye_y + head_r * 0.42,
                          fill=LINE, width=2)
        # mouth
        self.create_arc(cx - head_r * 0.32, eye_y + head_r * 0.5,
                         cx + head_r * 0.32, eye_y + head_r * 0.85,
                         start=200, extent=140, style="arc", outline=LINE, width=2)

        return l_wrist, r_wrist

    # ------------------------------------------------------------ idle pose
    def _draw_idle(self):
        l_wrist, r_wrist = self._draw_body(converge=True)
        self._draw_folded_hands(l_wrist, r_wrist)

    def _draw_folded_hands(self, l_wrist, r_wrist):
        """Draws two clearly-visible hands resting together, one on top of
        the other - the avatar's default 'nothing recognised' pose."""
        cx = (l_wrist[0] + r_wrist[0]) / 2
        cy = (l_wrist[1] + r_wrist[1]) / 2
        r = max(14, min(self._dims()) * 0.05)

        self._draw_simple_hand(cx - r * 0.5, cy, r, shade=SKIN_SHADOW)
        self._draw_simple_hand(cx + r * 0.35, cy - r * 0.15, r, shade=SKIN)

    def _draw_simple_hand(self, x, y, r, shade=SKIN):
        """A simplified relaxed hand shape: a rounded palm plus four soft
        finger bumps, clearly readable at small sizes."""
        self.create_oval(x - r, y - r * 0.75, x + r, y + r * 0.85, fill=shade, outline=LINE, width=2)
        for fx in (-0.62, -0.2, 0.22, 0.62):
            fw = r * 0.34
            fh = r * 0.55
            cx_f = x + fx * r
            cy_f = y - r * 0.7
            self.create_oval(cx_f - fw / 2, cy_f - fh, cx_f + fw / 2, cy_f + fh * 0.3,
                              fill=shade, outline=LINE, width=1)
        self.create_oval(x - r * 1.05, y - r * 0.1, x - r * 0.35, y + r * 0.5,
                          fill=shade, outline=LINE, width=1)

    # ------------------------------------------------------------ hand render
    def _render_hand_block(self, block63, wrist_anchor, scale):
        """block63: flat array of 21*(x,y,z). Draws it anchored at wrist_anchor."""
        pts = []
        for i in range(21):
            x, y = block63[i * 3], block63[i * 3 + 1]
            pts.append((x, y))
        wx0, wy0 = pts[0]
        ax, ay = wrist_anchor
        screen_pts = [(ax + (x - wx0) * scale, ay + (y - wy0) * scale) for x, y in pts]

        for a, b in HAND_CONNECTIONS:
            x1, y1 = screen_pts[a]
            x2, y2 = screen_pts[b]
            self.create_line(x1, y1, x2, y2, fill=SKIN, width=6, capstyle="round")
            self.create_line(x1, y1, x2, y2, fill=LINE, width=1)
        for x, y in screen_pts:
            self.create_oval(x - 4, y - 4, x + 4, y + 4, fill=SKIN, outline=LINE)

    def show_frame(self, feature_vec_126, scale=None):
        """Renders one frame of hand-landmark data on top of the static body
        (arms spread, so two-handed signs display correctly)."""
        l_wrist, r_wrist = self._draw_body(converge=False)
        w, h = self._dims()
        if scale is None:
            scale = min(w, h) * 0.9

        left_block = feature_vec_126[:FEATURES_PER_HAND]
        right_block = feature_vec_126[FEATURES_PER_HAND:]

        left_active = any(left_block)
        right_active = any(right_block)
        default_r = max(14, min(w, h) * 0.045)

        if left_active:
            self._render_hand_block(left_block, l_wrist, scale)
        else:
            self._draw_simple_hand(*l_wrist, default_r)

        if right_active:
            self._render_hand_block(right_block, r_wrist, scale)
        else:
            self._draw_simple_hand(*r_wrist, default_r)

    # ------------------------------------------------------------ playback
    def play_sequence(self, sequence, fps=15, on_complete=None):
        """
        Plays a (SEQUENCE_LENGTH, 126) landmark sequence as an animation.
        Cancels any animation already in progress first.
        """
        self.stop()
        if sequence is None or len(sequence) == 0:
            self.go_idle()
            if on_complete:
                on_complete()
            return

        delay_ms = int(1000 / fps)

        def step(i):
            if i >= len(sequence):
                self._anim_after_id = None
                if on_complete:
                    on_complete()
                return
            self.show_frame(sequence[i])
            self._anim_after_id = self.after(delay_ms, lambda: step(i + 1))

        step(0)

    def stop(self):
        if self._anim_after_id is not None:
            try:
                self.after_cancel(self._anim_after_id)
            except Exception:
                pass
            self._anim_after_id = None

    def go_idle(self):
        self.stop()
        self._draw_idle()
