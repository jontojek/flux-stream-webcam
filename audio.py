# ---------------------------------------------------------------------------
# audio.py  –  background mic → faster-whisper → live prompt string
# ---------------------------------------------------------------------------
#
# Rolling-buffer design (ported from the proven webcam-voice-i2i project):
#
#   _reader callback  -- sounddevice pushes mic chunks into a rolling buffer,
#                        never blocked by Whisper
#   _transcribe_loop  -- wakes every TRANSCRIBE_INTERVAL s, snapshots the
#                        buffer, gates on RMS energy (skips silence so Whisper
#                        can't hallucinate on dead air), then transcribes
#
# Exposes the latest text via get_prompt(); main.py applies a stability
# debounce before committing a new prompt.
# ---------------------------------------------------------------------------

import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

import config


class AudioThread(threading.Thread):
    """Continuously listens and exposes the latest transcription via get_prompt()."""

    def __init__(self, initial_prompt: str = "a vivid painting"):
        super().__init__(daemon=True)
        self._prompt   = initial_prompt
        self._lock     = threading.Lock()
        self._stop_evt = threading.Event()

        self._sr            = config.AUDIO_SAMPLE_RATE
        self._buf_samples   = int(self._sr * config.AUDIO_BUFFER_SECONDS)
        self._interval      = config.AUDIO_TRANSCRIBE_INTERVAL
        self._silence_rms   = config.AUDIO_SILENCE_THRESHOLD
        self._min_words     = getattr(config, "AUDIO_MIN_WORDS", 1)
        self._use_vad       = getattr(config, "AUDIO_USE_VAD", True)
        self._debug         = getattr(config, "AUDIO_DEBUG", True)
        self._buffer        = deque(maxlen=self._buf_samples)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_prompt(self) -> str:
        with self._lock:
            return self._prompt

    def stop(self) -> None:
        self._stop_evt.set()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _update_prompt(self, text: str) -> None:
        text = text.strip()
        if text:
            with self._lock:
                self._prompt = text

    @staticmethod
    def _peak_rms(seg: np.ndarray, win: int) -> float:
        """Loudest RMS over any `win`-sample sub-window — short phrases aren't
        diluted by surrounding silence the way a whole-window average would be."""
        if len(seg) < win:
            return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
        best = 0.0
        for i in range(0, len(seg) - win + 1, win):
            r = float(np.sqrt(np.mean(seg[i:i + win] ** 2)))
            if r > best:
                best = r
        return best

    def _transcribe_loop(self, model: WhisperModel) -> None:
        win = self._sr // 2  # 0.5 s
        while not self._stop_evt.is_set():
            time.sleep(self._interval)

            with self._lock:
                if len(self._buffer) < self._sr // 2:   # need ≥0.5 s of audio
                    continue
                seg = np.array(self._buffer, dtype=np.float32)

            # Energy gate on the LOUDEST half-second (not the diluted average).
            level = self._peak_rms(seg, win)
            if self._silence_rms > 0 and level < self._silence_rms:
                # Only log when there's *some* sound, so pure silence stays quiet.
                if self._debug and level > 0.0015:
                    print(f"[audio] heard sound (level {level:.4f}) below "
                          f"threshold {self._silence_rms:.4f} — skipped "
                          f"(lower AUDIO_SILENCE_THRESHOLD to catch this)")
                continue

            segments, _ = model.transcribe(
                seg,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                language="en",
                condition_on_previous_text=False,
                vad_filter=self._use_vad,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            n_words = len(text.split())
            if text and n_words >= self._min_words:
                self._update_prompt(text)
                print(f"[audio] prompt → {text!r}   (level {level:.3f})")
            elif self._debug and level >= self._silence_rms:
                if not text:
                    print(f"[audio] level {level:.3f} ok but no speech detected "
                          f"(VAD filtered it — set AUDIO_USE_VAD=False if it keeps "
                          f"dropping you)")
                else:
                    print(f"[audio] ignored {text!r} (< {self._min_words} words)")

    def run(self) -> None:
        print(f"[audio] Loading faster-whisper '{config.WHISPER_MODEL}' …")
        model = WhisperModel(config.WHISPER_MODEL, device="cuda", compute_type="int8")
        print("[audio] Whisper ready.")

        try:
            dev = sd.query_devices(kind="input")
            print(f"[audio] Microphone: {dev['name']}")
        except Exception:
            pass

        transcribe_thread = threading.Thread(
            target=self._transcribe_loop, args=(model,), daemon=True
        )
        transcribe_thread.start()

        # ~0.5 s mic blocks keep the buffer fresh without much overhead
        block = max(1, int(self._sr * 0.5))

        def _cb(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                print(f"[audio] sounddevice status: {status}")
            mono = indata[:, 0]
            with self._lock:
                self._buffer.extend(mono)

        with sd.InputStream(
            device=config.AUDIO_DEVICE,
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=block,
            callback=_cb,
        ):
            self._stop_evt.wait()

        transcribe_thread.join(timeout=5)
        print("[audio] Thread exiting.")
