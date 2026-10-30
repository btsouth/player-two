"""The narrator's voice: Kokoro, a free local neural voice, streamed sentence by sentence."""

from __future__ import annotations

import asyncio
import platform
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .audio import CUES, Mixer
from .config import CACHE_DIR, Settings

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")

# Kokoro's languages: (phonemizer code, default voice).
LANGUAGES = {
    "english": ("en-us", "af_heart"),
    "british english": ("en-gb", "bf_emma"),
    "spanish": ("es", "ef_dora"),
    "french": ("fr-fr", "ff_siwis"),
    "italian": ("it", "if_sara"),
    "portuguese": ("pt-br", "pf_dora"),
    "hindi": ("hi", "hf_alpha"),
}
# Japanese and Chinese voices exist, but kokoro-onnx's espeak front end garbles them, so they
# aren't offered. Games in those languages can still be narrated in any language above.


def voice_for(language: str, voice: str = "") -> tuple[str, str]:
    code, default = LANGUAGES.get(language.strip().lower(), LANGUAGES["english"])
    return code, voice or default


_SENTENCE = re.compile(r"(?<=[.!?…])\s+|\n+")


def model_dir() -> Path:
    return CACHE_DIR / "models"


def ensure_models() -> Path:
    d = model_dir()
    d.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        p = d / name
        if not p.exists() or p.stat().st_size == 0:
            import urllib.request

            print(f"Downloading voice file {name} (one time)...", flush=True)
            tmp = p.with_suffix(".part")
            said = [0]

            # A read timeout, so a stalled connection fails instead of hanging setup.
            with urllib.request.urlopen(MODEL_URL + name, timeout=60) as r, tmp.open("wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    pct = done * 100 // total if total else 0
                    if pct >= said[0] + 25 and pct < 100:
                        said[0] = pct - pct % 25
                        print(f"{said[0]} percent", flush=True)
            if total and tmp.stat().st_size != total:
                raise OSError(f"the download of {name} was cut short")
            tmp.replace(p)
    return d


def load_kokoro():
    """Kokoro on the GPU when onnxruntime-gpu can use it, otherwise on the CPU."""
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    d = ensure_models()
    onnx, voices = str(d / MODEL_FILES[0]), str(d / MODEL_FILES[1])
    ort.set_default_logger_severity(3)
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            sess = ort.InferenceSession(
                onnx, opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            if "CUDAExecutionProvider" in sess.get_providers():
                return Kokoro.from_session(sess, voices), "gpu"
        except Exception as exc:  # noqa: BLE001  any CUDA setup problem means CPU
            print(f"Voice: GPU unavailable ({exc}), using the CPU.", file=sys.stderr)
    return Kokoro(onnx, voices), "cpu"


GPU_ADDON = 'uv tool install --reinstall ".[cuda]" --excludes uv-excludes.txt'


def nvidia_card() -> bool:
    return Path("/proc/driver/nvidia/version").exists() or shutil.which("nvidia-smi") is not None


def gpu_hint(device: str, spoken: bool = False) -> str:
    """How to make the voice quicker when an NVIDIA card sits unused (Linux only: the
    graphics card build of the voice model is a Linux x86_64 download)."""
    if device != "cpu" or sys.platform != "linux" or platform.machine() != "x86_64" \
            or not nvidia_card():
        return ""
    if spoken:
        return ("My voice runs on the processor, not your NVIDIA card; player-two setup "
                "says how to make it start each line about half a second sooner.")
    return ("You have an NVIDIA graphics card, but my voice runs on the processor. To start "
            "each line about half a second sooner, add the graphics card download (about "
            f"2 GB). In the player-two folder, run: {GPU_ADDON}")


def first_chunk_split(sentence: str, limit: int = 60) -> list[str]:
    """Split a long opening sentence at its first comma so speech starts sooner."""
    if len(sentence) <= limit:
        return [sentence]
    i = sentence.find(", ", 12)
    if 0 < i < limit:
        return [sentence[: i + 1], sentence[i + 2 :]]
    return [sentence]


def tidy(text: str) -> str:
    """House style for anything shown or spoken: no dashes as punctuation."""
    return re.sub(r"\s*[\u2014\u2013]\s*", ", ", text)


def clean_for_speech(text: str) -> str:
    text = tidy(re.sub(r"[*_#`>]+", "", text))
    text = re.sub(r"\b([HSM]P)\b\.?", lambda m: ".".join(m.group(1)) + ".", text)
    return re.sub(r"\s+", " ", text).strip()


def speakable(text: str) -> bool:
    """False for empty text and the narrator's <silent> marker (cleaning eats its ">")."""
    return bool(text) and text.strip("<>/ ").lower() != "silent"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


class Voice:
    def __init__(self, settings: Settings, mixer: Mixer):
        self.kokoro, self.device = load_kokoro()
        self.s = settings
        self.mixer = mixer
        self.lang, self.voice_name = voice_for(settings.language, settings.voice)
        self._gen = 0
        self._lock = asyncio.Lock()
        self.last_spoken = ""
        self.first_audio_ms: list[float] = []

    def warm_up(self) -> None:
        """The first synthesis is slow (graph setup), so do it before anyone is waiting."""
        self._synth("Ready.")

    def cue(self, name: str, pan: float = 0.0) -> None:
        self.mixer.cue(CUES[name](), pan=pan)

    def idle(self) -> bool:
        return self.mixer.idle.is_set()

    def stop(self) -> None:
        """Cut off whatever is being said."""
        self._gen += 1
        self.mixer.clear_speech()

    def _synth(self, sentence: str) -> np.ndarray:
        samples, _sr = self.kokoro.create(
            sentence, voice=self.voice_name, speed=self.s.speech_rate, lang=self.lang
        )
        return samples

    async def say(self, text: str, interrupt: bool = True) -> None:
        """Speak text. With interrupt, anything still playing is cut off first."""
        text = clean_for_speech(text)
        if not text:
            return
        if interrupt:
            self.stop()
        queued = self._gen  # a stop() or a newer line while it waits drops it
        # One line at a time: two said together would otherwise alternate sentences.
        async with self._lock:
            await self._say(text, queued)

    async def _say(self, text: str, gen: int) -> None:
        if gen != self._gen:
            return  # cut off by a newer line while it waited its turn
        gen = self._gen
        self.last_spoken = text
        t0 = time.monotonic()
        first = True
        sentences = split_sentences(text)
        if sentences:
            sentences = first_chunk_split(sentences[0]) + sentences[1:]
        for sentence in sentences:
            samples = await asyncio.to_thread(self._synth, sentence)
            if gen != self._gen:
                return
            if first:
                self.first_audio_ms.append((time.monotonic() - t0) * 1000)
                first = False
            self.mixer.push_speech(samples)

    async def say_stream(self, deltas, interrupt: bool = True) -> str:
        """Speak streamed model text, starting as soon as the first sentence is complete."""
        if interrupt:
            self.stop()
        queued = self._gen
        async with self._lock:
            return await self._say_stream(deltas, queued)

    async def _say_stream(self, deltas, mine: int) -> str:
        if mine != self._gen:
            return ""
        gen = self._gen
        buf = ""
        spoken: list[str] = []
        pending: asyncio.Task | None = None

        async def speak_chunk(chunk: str, prev: asyncio.Task | None) -> None:
            samples = await asyncio.to_thread(self._synth, chunk)
            if prev is not None:
                await prev
            if gen == self._gen:
                self.mixer.push_speech(samples)

        chunks_left: list[asyncio.Task] = []

        def queue(chunk: str) -> None:
            nonlocal pending
            pending = asyncio.create_task(speak_chunk(chunk, pending))
            chunks_left.append(pending)

        try:
            async for delta in deltas:
                if gen != self._gen:
                    return " ".join(spoken)
                buf += delta
                parts = _SENTENCE.split(buf)
                for done in parts[:-1]:
                    done = clean_for_speech(done)
                    if speakable(done):
                        chunks = first_chunk_split(done) if not spoken else [done]
                        spoken.append(done)
                        for chunk in chunks:
                            queue(chunk)
                buf = parts[-1]
            tail = clean_for_speech(buf)
            if speakable(tail) and gen == self._gen:
                spoken.append(tail)
                queue(tail)
            if pending is not None:
                await pending
        except BaseException:
            # Cancelled (or the model stream broke): nothing more of this is said.
            for task in chunks_left:
                task.cancel()
            raise
        self.last_spoken = " ".join(spoken)
        return self.last_spoken

    async def wait_idle(self, timeout: float = 30.0) -> None:
        await asyncio.to_thread(self.mixer.idle.wait, timeout)
