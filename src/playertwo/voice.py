"""The narrator's voice: Kokoro, a free local neural voice, streamed sentence by sentence."""

from __future__ import annotations

import asyncio
import re
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
    "japanese": ("ja", "jf_alpha"),
    "chinese": ("cmn", "zf_xiaobei"),
}


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

            print(f"Downloading voice model {name} (one time)...")
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL + name, tmp)
            tmp.rename(p)
    return d


def clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_#`>]+", "", text)
    text = text.replace("HP", "H.P.").replace("SP", "S.P.")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


class Voice:
    def __init__(self, settings: Settings, mixer: Mixer):
        from kokoro_onnx import Kokoro

        d = ensure_models()
        self.kokoro = Kokoro(str(d / MODEL_FILES[0]), str(d / MODEL_FILES[1]))
        self.s = settings
        self.mixer = mixer
        self.lang, self.voice_name = voice_for(settings.language, settings.voice)
        self._gen = 0
        self._lock = asyncio.Lock()
        self.last_spoken = ""
        self.first_audio_ms: list[float] = []

    def cue(self, name: str, pan: float = 0.0) -> None:
        self.mixer.cue(CUES[name](), pan=pan)

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
        gen = self._gen
        self.last_spoken = text
        t0 = time.monotonic()
        first = True
        for sentence in split_sentences(text):
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

        async for delta in deltas:
            if gen != self._gen:
                return " ".join(spoken)
            buf += delta
            parts = _SENTENCE.split(buf)
            for done in parts[:-1]:
                done = clean_for_speech(done)
                if done and done != "<silent>":
                    spoken.append(done)
                    pending = asyncio.create_task(speak_chunk(done, pending))
            buf = parts[-1]
        tail = clean_for_speech(buf)
        if tail and tail != "<silent>" and gen == self._gen:
            spoken.append(tail)
            pending = asyncio.create_task(speak_chunk(tail, pending))
        if pending is not None:
            await pending
        self.last_spoken = " ".join(spoken)
        return self.last_spoken

    async def wait_idle(self, timeout: float = 30.0) -> None:
        await asyncio.to_thread(self.mixer.idle.wait, timeout)
