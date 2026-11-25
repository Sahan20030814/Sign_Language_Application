"""
audio_player.py
------------------
Plays a selected .wav file with real play/pause/seek support AND variable
playback speed.

IMPORTANT: pygame's `pygame.mixer.music.play(start=seconds)` is documented
to only work for MP3/OGG/MOD files - it silently doesn't seek for WAV, so
every play()/skip() call was restarting from 0:00. This reads the WAV's
raw PCM frames with the standard `wave` module and plays byte-sliced
chunks through `pygame.mixer.Sound` + a dedicated `Channel`, which
supports arbitrary start offsets and real pause/resume. Duration is also
computed directly from the WAV header (no `mutagen` dependency needed).

SPEED CONTROL: changing speed re-initialises the mixer at a scaled sample
rate and re-plays the SAME audio bytes through it - the classic simple
"faster = higher pitch too" trick (no separate time-stretch library
needed). Position tracking accounts for the current speed so the
timeline and progress stay accurate at any speed.
"""

import time
import wave

import pygame


class AudioPlayer:
    def __init__(self):
        self.channel = None
        self.path = None
        self.duration = 0.0
        self._frames = b""
        self._framerate = 0
        self._sampwidth = 0
        self._nchannels = 0
        self._offset = 0.0        # position in the ORIGINAL (1.0x) timeline, seconds
        self._speed = 1.0
        self._playing = False
        self._play_started_at = None
        self._current_sound = None  # keep a reference alive while playing

    def load(self, path):
        self.stop()
        with wave.open(path, "rb") as wf:
            self._nchannels = wf.getnchannels()
            self._sampwidth = wf.getsampwidth()
            self._framerate = wf.getframerate()
            nframes = wf.getnframes()
            self._frames = wf.readframes(nframes)

        self.duration = (nframes / float(self._framerate)) if self._framerate else 0.0
        self._speed = 1.0
        self.path = path
        self._offset = 0.0
        self._playing = False
        self._play_started_at = None
        self._reinit_mixer(self._speed)

    def _reinit_mixer(self, speed):
        size = -16 if self._sampwidth == 2 else (8 if self._sampwidth == 1 else -16)
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        pygame.mixer.init(frequency=max(1, int((self._framerate or 44100) * speed)),
                           size=size, channels=max(1, self._nchannels))
        self.channel = pygame.mixer.Channel(0)

    def _slice_from(self, seconds):
        bytes_per_frame = self._sampwidth * self._nchannels
        start_byte = int(seconds * self._framerate) * bytes_per_frame
        start_byte = max(0, min(start_byte, len(self._frames)))
        return self._frames[start_byte:]

    def play(self):
        if not self.path or not self._frames or self.channel is None:
            return
        data = self._slice_from(self._offset)
        if not data:
            self._playing = False
            return
        self._current_sound = pygame.mixer.Sound(buffer=data)
        self.channel.play(self._current_sound)
        self._playing = True
        self._play_started_at = time.time()

    def pause(self):
        if self._playing:
            self._offset = self.position()
            if self.channel is not None:
                self.channel.stop()
            self._playing = False

    def stop(self):
        if self.channel is not None:
            self.channel.stop()
        self._playing = False
        self._offset = 0.0
        self._play_started_at = None

    def seek(self, seconds):
        seconds = max(0.0, min(seconds, self.duration if self.duration else max(seconds, 0.0)))
        was_playing = self._playing
        if self.channel is not None:
            self.channel.stop()
        self._offset = seconds
        self._playing = False
        if was_playing:
            self.play()

    def skip(self, delta_seconds):
        self.seek(self.position() + delta_seconds)

    def set_speed(self, factor):
        """Changes playback speed WITHOUT stopping - it keeps playing (if it
        already was) from the same position, just at the new rate."""
        factor = max(0.25, min(4.0, factor))
        if abs(factor - self._speed) < 1e-6:
            return
        was_playing = self._playing
        current_pos = self.position()
        if self.channel is not None:
            self.channel.stop()
        self._speed = factor
        self._offset = current_pos
        self._playing = False
        self._reinit_mixer(self._speed)
        if was_playing:
            self.play()

    def position(self):
        """Current position in the ORIGINAL (1.0x) timeline, in seconds."""
        if self._playing and self._play_started_at is not None:
            wall_elapsed = time.time() - self._play_started_at
            pos = self._offset + wall_elapsed * self._speed
            return min(pos, self.duration) if self.duration else pos
        return self._offset

    def is_playing(self):
        return self._playing and self.channel is not None and self.channel.get_busy()

    def is_finished(self):
        """True once playback has run to the natural end of the file (as
        opposed to being manually paused/stopped)."""
        if self._playing and self.channel is not None and not self.channel.get_busy():
            return True
        return False
