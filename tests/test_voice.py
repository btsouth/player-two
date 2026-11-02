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
    v.mixer = SimpleNamespace(push_speech=lambda x, text="", tag="": pushed.append(x),
                              clear_speech=lambda: None)

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


def test_complete_thoughts_keep_their_commas_and_share_prosody():
    from playertwo.voice import first_chunk_split, speech_chunks
    line = "We made it to Guardia Castle, but the guard stopped you and Marle wants the fair."
    assert first_chunk_split(line) == [line]
    assert speech_chunks("Frog's down to 1 HP. Crono and Marle are fine.") == [
        "Frog's down to 1 HP. Crono and Marle are fine."]


def test_quiet_boundary_is_measured_without_cutting_a_soft_final_sound():
    import numpy as np

    from playertwo.voice import sentence_tail
    samples = np.r_[np.ones(2400)*.1, np.ones(120)*.0002, np.zeros(12000)]
    out = sentence_tail(samples)
    assert np.array_equal(out[:2520], samples[:2520])
    assert len(out) == 2520 + 2880


async def test_simultaneous_status_events_share_one_synthesis_but_keep_order():
    v, pushed = fake_voice()
    v.first_audio_ms = []
    await asyncio.gather(v.say("Frog's at 1 HP.", interrupt=False),
                         v.say("Crono and Marle are fine.", interrupt=False))
    assert pushed == ["Frog's at 1 H.P. Crono and Marle are fine."]


async def test_stop_discards_a_status_batch_before_synthesis():
    v, pushed = fake_voice()
    v.first_audio_ms = []
    task = asyncio.create_task(v.say("Frog's using Heal.", interrupt=False))
    await asyncio.sleep(.01)
    v.stop()
    await task
    assert pushed == []
