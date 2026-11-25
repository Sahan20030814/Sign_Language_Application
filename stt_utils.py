"""
stt_utils.py
--------------
Speech-to-text helpers for the "Speech -> Sign Language" mode.

Uses the free `SpeechRecognition` library. Live microphone listening and
audio-file transcription both use Google's free web speech API via
`recognize_google()` - this is the one part of the whole project that
needs an internet connection (there is no good fully-offline free
alternative with comparable accuracy that installs painlessly on
Windows). Everything else in the app (TTS, hand-sign recognition) stays
fully offline.

`speech_recognition.AudioFile` only natively reads WAV/AIFF/FLAC. Any
other format (mp3, m4a, etc) is converted to a temporary WAV file first
using `pydub` (which itself needs ffmpeg installed and on PATH). Long
files are also split into short chunks, since Google's free endpoint is
far more reliable on short clips than on multi-minute audio - and note
that this is speech recognition, not music/lyrics recognition, so a
sung audio track will transcribe poorly even with all of this in place.
"""

import os
import tempfile
import threading

import speech_recognition as sr


def list_microphones():
    """
    Returns a list of (index, name) for microphones matching what Windows
    Sound Settings shows as input devices. Enumerates PyAudio devices
    directly, filtered to the system's default host API and to devices
    that actually have input channels, and de-duplicated by name - this
    avoids the noisy/duplicate entries you get from testing every host API
    (MME/DirectSound/WASAPI all exposing the same physical device).
    """
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        try:
            default_host_api = pa.get_default_host_api_info()["index"]
            seen = set()
            devices = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0 and info.get("hostApi") == default_host_api:
                    name = info.get("name", f"Device {i}")
                    if name not in seen:
                        seen.add(name)
                        devices.append((i, name))
            if devices:
                return devices
        finally:
            pa.terminate()
    except Exception:
        pass

    # fall back to speech_recognition's own listing if direct PyAudio access fails
    try:
        working = sr.Microphone.list_working_microphones()  # {index: name}, tests each device
        if working:
            return sorted(working.items())
    except Exception:
        pass
    try:
        names = sr.Microphone.list_microphone_names()
        return list(enumerate(names))
    except Exception:
        return []


class LiveSTT:
    """Listens on a background thread and calls on_words(list_of_words) for each
    recognised utterance. Call start() to begin, stop() to end."""

    def __init__(self, device_index, on_words, on_error=None):
        self.device_index = device_index
        self.on_words = on_words
        self.on_error = on_error or (lambda msg: None)
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.6
        self._mic = None
        self._stopper = None

    def start(self):
        # Explicitly requesting 16kHz mono is the standard fix for "works on
        # audio files but never recognises live microphone speech" - Google's
        # speech API is tuned for this rate, and letting PyAudio fall back to
        # a device's own native rate (commonly 44.1/48kHz, sometimes unusual
        # rates on USB mics) can produce audio the API can't parse well.
        try:
            self._mic = sr.Microphone(device_index=self.device_index, sample_rate=16000)
        except Exception:
            self._mic = sr.Microphone(device_index=self.device_index)  # fall back to device default
        try:
            with self._mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
        except Exception as e:
            self.on_error(f"Could not open microphone: {e}")
            return

        # adjust_for_ambient_noise can occasionally calibrate to an unusably
        # high threshold in a noisy room, which then means normal speech
        # never registers as "loud enough to be speech" at all - cap it.
        if self.recognizer.energy_threshold > 2000:
            print(f"[Mic] Calibrated threshold was {self.recognizer.energy_threshold:.0f} - "
                  f"capping to 2000 so normal speech isn't ignored.")
            self.recognizer.energy_threshold = 2000

        print(f"[Mic] Listening started on device index {self.device_index} "
              f"(energy_threshold={self.recognizer.energy_threshold:.0f})")
        self._stopper = self.recognizer.listen_in_background(
            self._mic, self._on_audio, phrase_time_limit=4
        )

    def _on_audio(self, recognizer, audio):
        try:
            print("[Mic] Heard something - sending to speech recognition...")
            text = recognizer.recognize_google(audio)
            print(f"[Mic] Recognised: '{text}'")
            words = [w.lower().strip(".,!?") for w in text.split() if w.strip()]
            if words:
                self.on_words(words)
        except sr.UnknownValueError:
            print("[Mic] Heard audio but couldn't make out any words in it.")
        except sr.RequestError as e:
            print(f"[Mic] Speech service error: {e}")
            self.on_error(f"Speech service error: {e}")
        except Exception as e:
            # IMPORTANT: catch-all. If something unexpected throws here and we
            # only caught the two specific exceptions above, this callback
            # could silently stop contributing without any visible sign of it.
            print(f"[Mic] Unexpected error while recognising speech: {e}")
            self.on_error(f"Unexpected error: {e}")

    def stop(self):
        if self._stopper is not None:
            try:
                self._stopper(wait_for_stop=False)
            except Exception:
                pass
            self._stopper = None
        print("[Mic] Listening stopped.")


def transcribe_file_async(path, on_done, on_error=None, on_progress=None, on_chunk=None):
    """
    Transcribes an audio file on a background thread, chunk by chunk.
    on_chunk(chunk_start_seconds, chunk_end_seconds, chunk_text) fires
    progressively as each ~8-second chunk finishes - this is what lets the
    app start syncing avatar gestures to playback WHILE the rest of the
    file is still being transcribed, instead of waiting for the whole
    thing. on_done(full_text) fires once at the very end with everything
    stitched together.
    """
    on_error = on_error or (lambda msg: None)

    def _run():
        wav_path, is_temp = None, False
        try:
            wav_path, is_temp = _ensure_wav(path)
            text = _transcribe_wav_chunked(wav_path, on_progress, on_chunk)
            on_done(text)
        except Exception as e:
            on_error(str(e))
        finally:
            if is_temp and wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    threading.Thread(target=_run, daemon=True).start()


def _ensure_wav(path):
    """Returns (wav_path, is_temporary). Converts to WAV via pydub if the
    file isn't already one (needs ffmpeg installed and on PATH)."""
    if path.lower().endswith(".wav"):
        return path, False
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError(
            "This file format needs conversion, which needs the 'pydub' "
            "package plus ffmpeg installed and on PATH. Install ffmpeg, or "
            "use a .wav file instead."
        )
    try:
        audio = AudioSegment.from_file(path)
    except Exception as e:
        raise RuntimeError(
            f"Could not read this audio file (is ffmpeg installed and on PATH?): {e}"
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    audio.export(tmp.name, format="wav")
    return tmp.name, True


def _transcribe_wav_chunked(wav_path, on_progress=None, on_chunk=None, chunk_seconds=8):
    """Transcribes a WAV file in short chunks (more reliable with Google's
    free endpoint than one long request, and lets results stream in
    progressively) and stitches the text together."""
    recognizer = sr.Recognizer()
    texts = []
    with sr.AudioFile(wav_path) as source:
        total = getattr(source, "DURATION", None) or 0
        offset = 0.0
        while offset < total or (offset == 0 and total == 0):
            chunk_start = offset
            audio = recognizer.record(source, duration=chunk_seconds)
            if len(audio.frame_data) == 0:
                break
            chunk_end = min(offset + chunk_seconds, total) if total else offset + chunk_seconds
            chunk_text = ""
            try:
                chunk_text = recognizer.recognize_google(audio)
                if chunk_text:
                    texts.append(chunk_text)
            except sr.UnknownValueError:
                pass  # this chunk had no understandable speech - keep going
            except sr.RequestError as e:
                raise RuntimeError(f"Speech service error: {e}")
            if on_chunk:
                on_chunk(chunk_start, chunk_end, chunk_text)
            offset += chunk_seconds
            if on_progress and total:
                on_progress(min(offset, total), total)
            if total == 0:
                break  # couldn't determine duration - only do a single pass
    return " ".join(texts)
