"""Speak through the player's own screen reader instead of the built-in voice.

Screen reader users set their voice, speed and braille display up exactly how they like
them, often at speeds no narrator voice matches. This sends Player Two's words to that
screen reader: Orca on Linux (its D-Bus remote controller, which also flashes the text on
a braille display), speech-dispatcher when Orca isn't running, and NVDA or JAWS on Windows.
Our cues and direction pings still play through Player Two's own audio.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import platform
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .audio import CUES, Mixer
from .config import CACHE_DIR, Settings
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

    async def speaking(self) -> bool | None:
        """Whether the screen reader is still talking, or None if it can't say."""
        return None


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


# NVDA's own controller client, current as of NVDA 2026.3. The copy bundled with
# accessible_output2 is from 2022 and its calls fail inside current NVDA.
NVDA_CLIENT = ("https://download.nvaccess.org/releases/2026.3/nvda_2026.3_controllerClient.zip",
               "e38a00689eab422f13aaed9aaaacdd3ae96411643b6f2f5fbf86222660ac7938")


def nvda_client() -> Path:
    """Fetch NVDA's controller client DLL once (LGPL 2.1, from NV Access)."""
    arch = {"AMD64": "x64", "ARM64": "arm64", "x86": "x86"}.get(platform.machine(), "x64")
    d = CACHE_DIR / "nvda-client"
    dll = d / arch / "nvdaControllerClient.dll"
    if dll.exists():
        return dll
    import urllib.request

    d.mkdir(parents=True, exist_ok=True)
    zpath = d / "client.zip"
    urllib.request.urlretrieve(NVDA_CLIENT[0], zpath)
    if hashlib.sha256(zpath.read_bytes()).hexdigest() != NVDA_CLIENT[1]:
        zpath.unlink()
        raise RuntimeError("NVDA controller client download didn't match its checksum")
    with zipfile.ZipFile(zpath) as z:
        z.extract(f"{arch}/nvdaControllerClient.dll", d)
        z.extract("license.txt", d)
    zpath.unlink()
    return dll


class NVDA(Backend):
    name = "NVDA"

    def __init__(self, lib):
        self.lib = lib

    @classmethod
    def connect(cls) -> NVDA | None:
        import ctypes

        try:
            lib = ctypes.windll.LoadLibrary(str(nvda_client()))
        except (OSError, RuntimeError) as exc:
            print(f"Can't load NVDA's controller client: {exc}", file=sys.stderr)
            return None
        return cls(lib) if lib.nvdaController_testIfRunning() == 0 else None

    async def speak(self, text: str) -> None:
        # NVDA queues these behind whatever it's saying, and shows them on a braille display.
        await asyncio.to_thread(self.lib.nvdaController_speakText, text)
        await asyncio.to_thread(self.lib.nvdaController_brailleMessage, text)

    async def stop(self) -> None:
        await asyncio.to_thread(self.lib.nvdaController_cancelSpeech)

    async def speaking(self) -> bool | None:
        import ctypes

        flag = ctypes.c_ubyte(0)
        if self.lib.nvdaController_isSpeaking(ctypes.byref(flag)) != 0:
            return None  # NVDA older than 2024.1
        return bool(flag.value)


class OtherWindows(Backend):
    """JAWS and the other Windows screen readers, through accessible_output2."""

    def __init__(self, output):
        self.out = output
        self.name = type(output).__name__

    async def speak(self, text: str) -> None:
        await asyncio.to_thread(self.out.speak, text, interrupt=False)

    async def stop(self) -> None:
        await asyncio.to_thread(self.out.silence)


def find_backend() -> Backend | None:
    """The screen reader running now, or None."""
    if sys.platform == "win32":
        nvda = NVDA.connect()
        if nvda is not None:
            return nvda
        try:
            from accessible_output2.outputs.auto import Auto
        except ImportError:
            return None
        for out in Auto().outputs:
            # SAPI is Windows' own voice, not a screen reader; our voice is better.
            name = type(out).__name__.lower()
            if name.startswith("sapi") or name == "nvda" or out.is_system_output():
                continue
            try:
                if out.is_active():
                    return OtherWindows(out)
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
        self._stopping: asyncio.Task | None = None

    def warm_up(self) -> None:
        pass

    def idle(self) -> bool:
        """Whether it's probably done talking (screen readers can't always say)."""
        return asyncio.get_running_loop().time() >= self._busy_until

    def cue(self, name: str, pan: float = 0.0) -> None:
        self.mixer.cue(CUES[name](), pan=pan)

    def stop(self) -> None:
        self._gen += 1
        self._busy_until = 0.0
        self._stopping = asyncio.get_running_loop().create_task(self.backend.stop())

    async def _send(self, text: str) -> None:
        # The interrupt has to reach the screen reader before the new line does, or it
        # cancels the line it was meant to make room for.
        if self._stopping is not None:
            stopping, self._stopping = self._stopping, None
            with contextlib.suppress(Exception):  # a failed interrupt still lets us speak
                await stopping
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
        end = loop.time() + timeout
        if await self.backend.speaking() is None:
            await asyncio.sleep(min(timeout, max(0.0, self._busy_until - loop.time())))
            return
        await asyncio.sleep(0.15)  # let the last message reach the synthesizer
        while loop.time() < end and await self.backend.speaking():
            await asyncio.sleep(0.1)


def open_voice(settings: Settings, mixer: Mixer):
    """The built-in narrator voice, or the player's screen reader.

    "screen-reader" uses whatever screen reader or speech service there is; "auto" (the
    default) uses a screen reader only when one is running (Orca, NVDA, JAWS), since
    people who use one want Player Two in the voice and speed they already know.
    """
    if settings.output in ("screen-reader", "auto"):
        backend = find_backend()
        if settings.output == "auto" and backend is not None and backend.name == "speech-dispatcher":
            backend = None  # a speech service, not a screen reader in use
        if backend is not None:
            return ScreenReaderVoice(settings, mixer, backend)
    if settings.output == "screen-reader":
        print("No screen reader found (Orca, speech-dispatcher, NVDA or JAWS); "
              "using Player Two's own voice.", file=sys.stderr)
    from .voice import Voice

    return Voice(settings, mixer)
