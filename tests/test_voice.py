"""Player Two's own voice, without loading a speech model or a sound device."""

import asyncio
import time
from types import SimpleNamespace

from playertwo.voice import Voice


def fake_voice():
    v = Voice.__new__(Voice)
    v._gen = 0
    v._lock = asyncio.Lock()
    v.last_spoken = ""
    pushed = []
    v.mixer = SimpleNamespace(push_speech=pushed.append, clear_speech=lambda: None)

    def synth(sentence):
        time.sleep(0.15)  # the speech model takes a moment
        return sentence

    v._synth = synth
    return v, pushed


async def test_cancelled_narration_says_nothing_more():
    v, pushed = fake_voice()
    more = asyncio.Event()

    async def deltas():
        yield "There's a chest by the door. "
        await more.wait()  # the model is still writing
        yield "And a key."

    task = asyncio.create_task(v.say_stream(deltas()))
    await asyncio.sleep(0.05)  # the first sentence is being synthesized
    task.cancel()
    await asyncio.sleep(0.4)
    assert pushed == []


async def test_streamed_narration_is_spoken_in_order():
    v, pushed = fake_voice()

    async def deltas():
        for d in ("One thing. ", "Two things. ", "Three."):
            yield d

    out = await v.say_stream(deltas())
    assert " ".join(pushed) == out and pushed[-1] == "Three."
