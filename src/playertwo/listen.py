"""Ears: push-to-talk speech recognition with Muse Voice Transcribe over a WebSocket.

The socket opens the moment the talk key goes down and the mic is buffered while it
connects, so the first word is never lost.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import websockets

from .config import api_key

RATE = 24_000
FRAME = RATE * 80 // 1000  # 80 ms of 16-bit mono
URI = "wss://api.meta.ai/v1/asr/realtime?sessionId={sid}"
MODEL = "muse-voice-transcribe-1.0"


class Mic:
    """Microphone frames as bytes on an asyncio queue."""

    def __init__(self):
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        loop = asyncio.get_running_loop()

        def cb(indata, _frames, _t, _status):
            loop.call_soon_threadsafe(self.queue.put_nowait, bytes(indata))

        self._stream = sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16",
                                         blocksize=FRAME, callback=cb)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


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
