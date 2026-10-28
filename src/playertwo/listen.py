"""Ears: speech recognition with Muse Voice Transcribe over a WebSocket.

Push to talk: the socket opens the moment the talk key goes down and the mic is buffered
while it connects, so the first word is never lost.

Hands-free: the mic stays open and a small speech detector (Silero VAD) runs on the PC.
Only stretches of speech go to Muse Voice Transcribe, starting a moment before the
speech did, and the session acts only on what starts with its name ("Player Two, open
the menu"), so silence costs nothing and talk in the room isn't taken as a command.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
import websockets

from .config import CACHE_DIR, api_key

RATE = 24_000
FRAME = RATE * 80 // 1000  # 80 ms of 16-bit mono
URI = "wss://api.meta.ai/v1/asr/realtime?sessionId={sid}"
MODEL = "muse-voice-transcribe-1.0"


class Mic:
    """Microphone frames as bytes on an asyncio queue.

    A `background` mic (hands-free listening, always open) doesn't stop the speaker from
    being found again after it's unplugged: the restart closes it, and its owner opens it
    again (see audio.Mixer.check).
    """

    def __init__(self, background: bool = False):
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.background = background
        self.retired = False  # closed for good: a start still under way opens nothing
        self._stream = None

    @property
    def open(self) -> bool:
        return self._stream is not None

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Open the mic. Pass the event loop when calling from another thread."""
        import sounddevice as sd

        loop = loop or asyncio.get_running_loop()

        def cb(indata, _frames, _t, _status):
            loop.call_soon_threadsafe(self.queue.put_nowait, bytes(indata))

        from . import audio

        with audio.PA_LOCK:
            if self.retired or self._stream is not None:
                return
            stream = sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16",
                                       blocksize=FRAME, callback=cb)
            try:
                stream.start()
            except Exception:
                stream.close(ignore_errors=True)
                raise
            self._stream = stream
            if self.background:
                audio.background_inputs.add(self)
            else:
                audio.open_inputs += 1

    def stop(self) -> None:
        from . import audio

        with audio.PA_LOCK:
            self.close_locked()

    def retire(self) -> None:
        """Close the mic for good."""
        from . import audio

        with audio.PA_LOCK:
            self.retired = True
            self.close_locked()

    def close_locked(self) -> None:
        """Close the stream; the caller holds audio.PA_LOCK."""
        from . import audio

        if self._stream is None:
            return
        with contextlib.suppress(Exception):
            self._stream.stop()
        with contextlib.suppress(Exception):
            self._stream.close()
        self._stream = None
        if self.background:
            audio.background_inputs.discard(self)
        else:
            audio.open_inputs -= 1


class Utterance:
    """One press of the talk key."""

    def __init__(self, keywords: list[str] | None = None, on_partial=None):
        self.keywords = [k for k in (keywords or []) if k][:50]
        self.on_partial = on_partial
        self.audio: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.result: asyncio.Future[str] | None = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self.result = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._run())

    def feed(self, frame: bytes) -> None:
        self.audio.put_nowait(frame)

    def finish(self) -> None:
        self.audio.put_nowait(None)

    async def text(self, timeout: float = 8.0) -> str:
        assert self.result is not None
        return await asyncio.wait_for(self.result, timeout)

    async def _run(self) -> None:
        try:
            self.result.set_result(await self._session())
        except Exception as exc:  # noqa: BLE001  surfaced to the caller
            if not self.result.done():
                self.result.set_exception(exc)

    async def _session(self) -> str:
        hello = {
            "authorization": {"accessToken": f"Bearer {api_key()}"},
            "audioEncoding": "PCM_24KHZ",
            "model": MODEL,
            "mode": "PUSH_TO_TALK",
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
        }
        if self.keywords:
            hello["keywords"] = self.keywords
        async with websockets.connect(URI.format(sid=f"p2-{uuid.uuid4()}"),
                                      open_timeout=10) as ws:
            await ws.send(json.dumps(hello))
            ack = json.loads(await ws.recv())
            if "sessionId" not in ack:
                raise RuntimeError(f"speech handshake failed: {ack}")
            final: list[str] = []

            async def receive() -> None:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        continue
                    ev = json.loads(msg)
                    if ev.get("type") == "error":
                        raise RuntimeError(ev.get("message", "speech error"))
                    if ev.get("type") == "transcript":
                        if ev.get("final"):
                            final.append(ev["transcript"])
                        elif self.on_partial:
                            self.on_partial(ev["transcript"])

            rx = asyncio.create_task(receive())
            while True:
                frame = await self.audio.get()
                if frame is None:
                    break
                await ws.send(frame)
            await ws.send(json.dumps({"type": "endStream"}))
            await asyncio.wait_for(rx, 10)
            return " ".join(final).strip()


async def transcribe_file(path: str, keywords: list[str] | None = None) -> str:
    """Run a WAV file through the same live path. Used by tests and the CLI."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != RATE:
        idx = np.round(np.arange(0, len(audio), sr / RATE)).astype(int)
        audio = audio[idx[idx < len(audio)]]
    u = Utterance(keywords)
    u.start()
    data = audio.astype("<i2").tobytes()
    for i in range(0, len(data), FRAME * 2):
        u.feed(data[i : i + FRAME * 2])
    u.finish()
    return await u.text(20)


def mic_ok() -> bool:
    """Whether there's a microphone to listen with."""
    try:
        import sounddevice as sd

        sd.query_devices(kind="input")
    except Exception:  # noqa: BLE001  PortAudio raises plain errors for "no device"
        return False
    return True


# Hands-free listening

VAD_URL = ("https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/"
           "silero_vad.onnx")
VAD_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
VAD_RATE = 16_000
VAD_CHUNK = 512  # 32 ms at 16 kHz, what Silero VAD v5 takes
VAD_CONTEXT = 64  # samples of the previous chunk it sees too

# "Player Two", however speech recognition spells it ("PlayerPlayer Two" when it stutters).
# "Player to" only counts on its own: "pass it to the player to the left" isn't a call.
WAKE = re.compile(r"(?:\b(?:hey|ok|okay)[\s,]+)?(?:player|layer|play a)s?[\s,-]*"
                  r"(?:(?:two|2|ii)\b|(?:to|too)(?=[\s]*[,.!?]|\s*$))[\s,.!?:-]*",
                  re.IGNORECASE)


def addressed(text: str) -> str | None:
    """What was said to Player Two: the words after its name, or None without the name.

    The last time the name comes up counts, so "Yosuke attacks. Player Two, stop" (the
    narrator's own voice picked up before the player spoke) is just "stop". Only the
    name, nothing after it, is "".
    """
    found = list(WAKE.finditer(text))
    if not found:
        return None
    return text[found[-1].end():].strip(" ,.!?")


def ensure_vad() -> Path:
    """The speech detector's model file, downloaded once (2 MB) and checked."""
    p = CACHE_DIR / "models" / "silero_vad.onnx"
    if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == VAD_SHA256:
        return p
    import urllib.request

    p.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(VAD_URL, timeout=60) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != VAD_SHA256:
        raise OSError("the speech detector download didn't match; try again later")
    tmp = p.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(p)
    return p


class SpeechDetector:
    """Silero VAD on the CPU: how likely each 80 ms microphone frame is speech."""

    def __init__(self, path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(VAD_CONTEXT, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)

    def __call__(self, frame: bytes) -> float:
        """The highest speech probability in this frame (24 kHz, 16-bit mono)."""
        x = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        # 24 kHz to 16 kHz: a light low-pass, then every 1.5th sample.
        if len(x) > 2:
            x = np.convolve(x, (0.25, 0.5, 0.25), mode="same")
        x = np.interp(np.arange(0, len(x), RATE / VAD_RATE), np.arange(len(x)), x)
        self.pending = np.concatenate([self.pending, x.astype(np.float32)])
        best = 0.0
        sr = np.array(VAD_RATE, dtype=np.int64)
        while len(self.pending) >= VAD_CHUNK:
            chunk, self.pending = self.pending[:VAD_CHUNK], self.pending[VAD_CHUNK:]
            inp = np.concatenate([self.context, chunk])[None, :]
            out, self.state = self.session.run(None, {"input": inp, "state": self.state,
                                                      "sr": sr})
            self.context = chunk[-VAD_CONTEXT:]
            best = max(best, float(out[0][0]))
        return best


class Segmenter:
    """Turns per-frame speech probabilities into stretches of speech.

    Speech starts after START_FRAMES frames above ON (with PREROLL frames of lead-in, so
    the first syllable isn't clipped) and ends after END_S of frames below OFF, or at
    MAX_S. Frames are 80 ms.
    """

    ON, OFF = 0.5, 0.35
    START_FRAMES = 2
    PREROLL = 5
    FRAME_S = FRAME / RATE
    END_S = 0.8
    MAX_S = 15.0
    END_FRAMES = round(END_S / FRAME_S)
    MAX_FRAMES = round(MAX_S / FRAME_S)

    def __init__(self):
        self.lead: deque[bytes] = deque(maxlen=self.PREROLL)
        self.voiced = 0
        self.active = False
        self.quiet = 0
        self.length = 0

    def push(self, frame: bytes, p: float) -> tuple[str, list[bytes]]:
        """("start", lead-in and this frame), ("speech", [frame]), ("end", [frame]) or
        ("", []) between stretches."""
        if not self.active:
            self.lead.append(frame)
            self.voiced = self.voiced + 1 if p >= self.ON else 0
            if self.voiced < self.START_FRAMES:
                return "", []
            frames = list(self.lead)
            self.lead.clear()
            self.active, self.quiet, self.length = True, 0, len(frames)
            return "start", frames
        self.length += 1
        self.quiet = self.quiet + 1 if p < self.OFF else 0
        if self.quiet >= self.END_FRAMES or self.length >= self.MAX_FRAMES:
            self.active, self.voiced = False, 0
            return "end", [frame]
        return "speech", [frame]

    def drop(self) -> None:
        self.active, self.voiced = False, 0
        self.lead.clear()


class HandsFree:
    """Always listening: sends each stretch of speech to Muse Voice Transcribe.

    `on_partial(text, started)` gets the words so far of the stretch that began at
    `started`, and `on_text(text, error, started, ended)` its final words (or the error
    that stopped it) and when the speech in it began and ended; the session decides what
    was said to it. `on_drop()` says a stretch was cut off unheard. While `paused()` (the
    talk key is held), it doesn't listen.
    """

    def __init__(self, on_partial: Callable[[str, float], None],
                 on_text: Callable[[str, BaseException | None, float, float],
                                   Awaitable[None]],
                 on_drop: Callable[[], None],
                 keywords: Callable[[], list[str]], paused: Callable[[], bool],
                 detector: SpeechDetector | None = None,
                 refresh: Callable[[], None] | None = None):
        self.on_partial = on_partial
        self.refresh = refresh  # restarts the sound system, to find a new mic
        self.on_text = on_text
        self.on_drop = on_drop
        self.keywords = keywords
        self.paused = paused
        self.detector = detector or SpeechDetector(ensure_vad())
        self.segmenter = Segmenter()
        self.mic = Mic(background=True)
        self.utt: Utterance | None = None
        self.tasks: set[asyncio.Task] = set()
        self.heard_at = time.monotonic()  # the last frame from the microphone
        self.started = 0.0  # when the stretch being heard began

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        refreshed = 0.0
        lost = False  # the mic failed or went quiet: the device list may be stale
        try:
            while True:
                if not self.mic.open:
                    self._end(cancel=True)
                    if lost and self.refresh and time.monotonic() - refreshed > 10.0:
                        refreshed = time.monotonic()
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(self.refresh)
                    try:
                        # Off the event loop: it waits while the sound system restarts.
                        await asyncio.to_thread(self.mic.start, loop)
                        self.heard_at = time.monotonic()
                        lost = False
                    except Exception:  # noqa: BLE001  unplugged: try again shortly
                        lost = True
                        await asyncio.sleep(2.0)
                        continue
                try:
                    frame = await asyncio.wait_for(self.mic.queue.get(), 1.0)
                except TimeoutError:
                    if self.mic.open and time.monotonic() - self.heard_at > 5.0:
                        lost = True
                        await asyncio.to_thread(self.mic.stop)  # gone: open it again
                    continue
                self.heard_at = time.monotonic()
                self.feed(frame)
        finally:
            self._end(cancel=True)
            self.mic.retire()

    def feed(self, frame: bytes) -> None:
        if self.paused():
            self._end(cancel=True)
            self.detector.reset()
            return
        what, frames = self.segmenter.push(frame, self.detector(frame))
        if what == "start":
            started = self.started = time.monotonic() - len(frames) * Segmenter.FRAME_S
            self.utt = Utterance(self.keywords(),
                                 on_partial=lambda text: self.on_partial(text, started))
            self.utt.start()
        if self.utt is not None:
            for f in frames:
                self.utt.feed(f)
        if what == "end":
            self._end()

    def _end(self, cancel: bool = False) -> None:
        """Finish the stretch being heard: transcribe it, or with `cancel` drop it."""
        utt, self.utt = self.utt, None
        started = self.started
        # The speech itself ended before the quiet that ended the stretch.
        ended = time.monotonic() - self.segmenter.quiet * Segmenter.FRAME_S
        if cancel:
            self.segmenter.drop()
        if utt is None:
            return
        utt.finish()
        if cancel:
            if utt._task is not None:
                utt._task.cancel()
            if utt.result is not None and utt.result.done() and not utt.result.cancelled():
                utt.result.exception()  # retrieved, so asyncio doesn't log it
            self.on_drop()
            return

        async def result() -> None:
            try:
                text = await utt.text(10)
            except Exception as exc:  # noqa: BLE001  the session says what went wrong
                await self.on_text("", exc, started, ended)
                return
            await self.on_text(text, None, started, ended)

        task = asyncio.create_task(result())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
