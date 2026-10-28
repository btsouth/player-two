"""Audio out: an interruptible speech lane plus mixed sound cues, in stereo."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

RATE = 24_000  # Kokoro's native rate; cues are generated at the same rate.


class Mixer:
    """Pulls speech and cue samples from buffers inside the sound card callback.

    Speech is one lane: clear_speech() drops whatever is queued, which is how a new,
    more urgent line cuts off a stale one. Cues are short sounds mixed on top and
    can be panned left or right to say where something is.
    """

    def __init__(self, sink: str = "device"):
        self._lock = threading.Lock()
        self._speech: deque[np.ndarray] = deque()
        self._speech_pos = 0
        self._cues: list[list] = []  # [stereo samples, position]
        self.idle = threading.Event()
        self.idle.set()
        self.sink = sink
        self.recorded: list[np.ndarray] = []
        self._stream = None
        if sink == "device":
            self._open()

    def _open(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=RATE, channels=2, dtype="float32", blocksize=480,
            callback=self._callback, latency="low",
        )
        self._stream.start()

    def check(self) -> bool:
        """Reopen the sound device if it went away (a headset unplugged, a device change).

        While there's no device, queued speech is dropped so nothing waits on it forever.
        Returns whether sound is playing again.
        """
        if self.sink != "device":
            return True
        if self._stream is not None and self._stream.active:
            return True
        self.clear_speech()
        try:
            if self._stream is not None:
                self._stream.close(ignore_errors=True)
            import sounddevice as sd

            sd._terminate()  # PortAudio only sees device changes after re-initializing
            sd._initialize()
            self._open()
        except Exception:  # noqa: BLE001  try again on the next check
            self._stream = None
            return False
        return True

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def push_speech(self, mono: np.ndarray) -> None:
        with self._lock:
            self._speech.append(np.asarray(mono, dtype=np.float32))
            self.idle.clear()
        if self.sink == "file":
            self.recorded.append(np.asarray(mono, dtype=np.float32))
            self.drain_offline()

    def clear_speech(self) -> None:
        with self._lock:
            self._speech.clear()
            self._speech_pos = 0
            self.idle.set()

    def cue(self, mono: np.ndarray, pan: float = 0.0, gain: float = 0.6) -> None:
        """Play a short sound. pan is -1 (left) to 1 (right)."""
        pan = max(-1.0, min(1.0, pan))
        left = np.cos((pan + 1) * np.pi / 4)
        right = np.sin((pan + 1) * np.pi / 4)
        st = np.stack([mono * left * gain, mono * right * gain], axis=1).astype(np.float32)
        with self._lock:
            self._cues.append([st, 0])

    def drain_offline(self) -> None:
        with self._lock:
            self._speech.clear()
            self._speech_pos = 0
            self.idle.set()

    def _callback(self, out, frames, _time, _status) -> None:
        out.fill(0)
        with self._lock:
            filled = 0
            while filled < frames and self._speech:
                buf = self._speech[0]
                take = min(frames - filled, len(buf) - self._speech_pos)
                chunk = buf[self._speech_pos : self._speech_pos + take]
                out[filled : filled + take, 0] += chunk
                out[filled : filled + take, 1] += chunk
                filled += take
                self._speech_pos += take
                if self._speech_pos >= len(buf):
                    self._speech.popleft()
                    self._speech_pos = 0
            if not self._speech:
                self.idle.set()
            keep = []
            for c in self._cues:
                st, pos = c
                take = min(frames, len(st) - pos)
                out[:take] += st[pos : pos + take]
                c[1] = pos + take
                if c[1] < len(st):
                    keep.append(c)
            self._cues = keep
        np.clip(out, -1.0, 1.0, out=out)


def tone(freq: float, ms: int, shape: str = "soft", volume: float = 0.5) -> np.ndarray:
    n = int(RATE * ms / 1000)
    t = np.arange(n) / RATE
    wave = np.sin(2 * np.pi * freq * t)
    if shape == "soft":
        wave += 0.25 * np.sin(4 * np.pi * freq * t)
    attack = min(n, int(RATE * 0.008))
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    env *= np.exp(-3.0 * t / max(t[-1], 1e-3))
    return (wave * env * volume).astype(np.float32)


CUES = {
    # Short, distinct sounds a player learns fast.
    "listening": lambda: np.concatenate([tone(660, 70), tone(880, 90)]),
    "heard": lambda: np.concatenate([tone(880, 60), tone(660, 80)]),
    "working": lambda: tone(520, 60, volume=0.25),
    "done": lambda: np.concatenate([tone(784, 70), tone(1046, 120)]),
    "error": lambda: np.concatenate([tone(330, 120), tone(262, 160)]),
    "low": lambda: np.concatenate([tone(440, 90), np.zeros(1200, np.float32), tone(440, 90)]),
    "critical": lambda: np.concatenate([tone(392, 80), tone(392, 80), tone(392, 80)]),
    "ping": lambda: tone(1200, 50, volume=0.35),
}
