"""Speak through the player's own screen reader instead of the built-in voice.

Screen reader users set their voice, speed and braille display up exactly how they like
them, often at speeds no narrator voice matches. This sends Player Two's words to that
screen reader: Orca on Linux (its D-Bus remote controller, which also flashes the text on
a braille display), speech-dispatcher when Orca isn't running, and NVDA or JAWS on Windows.
Our cues and direction pings still play through Player Two's own audio.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from dataclasses import dataclass

from .audio import CUES, Mixer
from .config import Settings
from .voice import _SENTENCE, clean_for_speech, speakable


@dataclass
class Bus:
    """One Orca D-Bus API generation."""

    name: str
    path: str
    speech_path: str = ""

    def call(self, path: str, method: str, *args: str) -> list[str]:
        return ["gdbus", "call", "--session", "--dest", self.name, "--object-path", path,
                "--method", method, *args]


ORCA = [
    # Orca 51 and later.
    Bus("org.gnome.Orca1.Service", "/org/gnome/Orca1/Service",
        "/org/gnome/Orca1/Service/SpeechManager"),
    # Orca 48 to 50. Speaking works the same; we don't interrupt it.
    Bus("org.gnome.Orca.Service", "/org/gnome/Orca/Service"),
]


class Backend:
    name = ""

    async def speak(self, text: str) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        pass


async def _run(cmd: list[str]) -> bool:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    return await proc.wait() == 0


class Orca(Backend):
    name = "Orca"

    def __init__(self, bus: Bus):
        self.bus = bus

    async def speak(self, text: str) -> None:
        # Orca queues messages at speech-dispatcher's "message" priority, so sentences
        # sent one after another play in order instead of cutting each other off.
        await _run(self.bus.call(self.bus.path, f"{self.bus.name}.PresentMessage", text))

    async def stop(self) -> None:
        if self.bus.speech_path:
            await _run(self.bus.call(self.bus.speech_path,
                                     "org.gnome.Orca1.SpeechManager.InterruptSpeech", "false"))


class SpeechDispatcher(Backend):
    name = "speech-dispatcher"

    async def speak(self, text: str) -> None:
        await _run(["spd-say", "--application-name", "player-two", "-P", "message", "--", text])

    async def stop(self) -> None:
        await _run(["spd-say", "-C"])


class Windows(Backend):
    """NVDA, JAWS and the other Windows screen readers, through accessible_output2."""

    def __init__(self, output):
        self.out = output
        self.name = type(output).__name__

    async def speak(self, text: str) -> None:
        await asyncio.to_thread(self.out.output, text, interrupt=False)

    async def stop(self) -> None:
        await asyncio.to_thread(self.out.silence)


def find_backend() -> Backend | None:
    """The screen reader running now, or None."""
    if sys.platform == "win32":
        try:
            from accessible_output2.outputs.auto import Auto
        except ImportError:
            return None
        for out in Auto().outputs:
            # SAPI is Windows' own voice, not a screen reader; our voice is better.
            if type(out).__name__.startswith("Sapi"):
                continue
            try:
                if out.is_active():
                    return Windows(out)
            except Exception:  # noqa: BLE001, S112  a missing reader can fail any way
                continue
        return None
    if shutil.which("gdbus"):
        for bus in ORCA:
            r = subprocess.run(bus.call(bus.path, f"{bus.name}.GetVersion"),
                               capture_output=True, timeout=3, check=False)
            if r.returncode == 0:
                return Orca(bus)
    if shutil.which("spd-say"):
        return SpeechDispatcher()
    return None


class ScreenReaderVoice:
    """Same interface as voice.Voice, speaking through the screen reader."""

    device = "screen reader"
    # Rough speaking time per word, for the few places that wait for speech to end.
    # Screen reader users often listen fast; this errs a little long.
    SECONDS_PER_WORD = 0.3

    def __init__(self, settings: Settings, mixer: Mixer, backend: Backend):
        self.s = settings
        self.mixer = mixer
        self.backend = backend
        self.voice_name = backend.name
        self.last_spoken = ""
        self.first_audio_ms: list[float] = []
        self._gen = 0
        self._busy_until = 0.0

    def warm_up(self) -> None:
        pass

    def cue(self, name: str, pan: float = 0.0) -> None:
        self.mixer.cue(CUES[name](), pan=pan)

    def stop(self) -> None:
        self._gen += 1
        self._busy_until = 0.0
        asyncio.get_running_loop().create_task(self.backend.stop())

    async def _send(self, text: str) -> None:
        loop = asyncio.get_running_loop()
        start = max(loop.time(), self._busy_until)
        self._busy_until = start + max(0.5, len(text.split()) * self.SECONDS_PER_WORD)
        await self.backend.speak(text)

    async def say(self, text: str, interrupt: bool = True) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        if interrupt:
            self.stop()
        self.last_spoken = text
        await self._send(text)

    async def say_stream(self, deltas, interrupt: bool = True) -> str:
        if interrupt:
            self.stop()
        gen = self._gen
        buf = ""
        spoken: list[str] = []
        async for delta in deltas:
            if gen != self._gen:
                return " ".join(spoken)
            buf += delta
            parts = _SENTENCE.split(buf)
            for done in parts[:-1]:
                done = clean_for_speech(done)
                if speakable(done):
                    spoken.append(done)
                    await self._send(done)
            buf = parts[-1]
        tail = clean_for_speech(buf)
        if speakable(tail) and gen == self._gen:
            spoken.append(tail)
            await self._send(tail)
        self.last_spoken = " ".join(spoken)
        return self.last_spoken

    async def wait_idle(self, timeout: float = 30.0) -> None:
        loop = asyncio.get_running_loop()
        await asyncio.sleep(min(timeout, max(0.0, self._busy_until - loop.time())))


def open_voice(settings: Settings, mixer: Mixer):
    """The built-in narrator voice, or the screen reader when output is "screen-reader"."""
    if settings.output == "screen-reader":
        backend = find_backend()
        if backend is not None:
            return ScreenReaderVoice(settings, mixer, backend)
        print("No screen reader found (Orca, speech-dispatcher, NVDA or JAWS); "
              "using Player Two's own voice.", file=sys.stderr)
    from .voice import Voice

    return Voice(settings, mixer)
