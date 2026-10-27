import asyncio
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from playertwo.agent import Agent
from playertwo.controls import DryRun
from playertwo.narrator import Narrator
from playertwo.screen import ChangeWatcher, Frame
from playertwo.voice import clean_for_speech, first_chunk_split, split_sentences, voice_for


def frame(arr: np.ndarray, t: float) -> Frame:
    return Frame(Image.fromarray(arr.astype(np.uint8)), t)


def blank(value: int = 20) -> np.ndarray:
    return np.full((270, 480, 3), value, np.uint8)


class TestWatcher:
    def run(self, frames):
        w = ChangeWatcher()
        return [w.update(f) for f in frames]

    def test_static_screen_fires_once(self):
        frames = [frame(blank(), i / 6) for i in range(30)]
        fired = self.run(frames)
        assert sum(fired) == 1

    def test_new_screen_fires_after_it_settles(self):
        w = ChangeWatcher()
        t = 0.0
        for _ in range(6):
            w.update(frame(blank(), t))
            t += 1 / 6
        new = blank()
        new[180:260, 20:460] = 230  # a dialogue box appears
        events = []
        for _ in range(8):
            events.append(w.update(frame(new, t)))
            t += 1 / 6
        assert events.count(True) == 1
        assert not events[0]  # not on the very first changed frame

    def test_constant_animation_is_ignored(self):
        w = ChangeWatcher()
        rng = np.random.default_rng(0)
        fired = 0
        for i in range(120):
            arr = blank()
            arr[10:60, 10:80] = rng.integers(0, 255, (50, 70, 3))  # a flickering TV
            fired += w.update(frame(arr, i / 6))
        # It may fire while it learns the flicker, but not every couple of seconds forever.
        assert fired <= 3


class TestSpeechText:
    def test_split_sentences(self):
        assert split_sentences("Your turn. Attack? Guard!") == ["Your turn.", "Attack?", "Guard!"]

    def test_first_chunk_split_long(self):
        s = "Mr. Morooka says, I'm Kinshiro Morooka, your homeroom teacher from today forward!"
        parts = first_chunk_split(s)
        assert parts[0] == "Mr. Morooka says,"
        assert " ".join(parts) == s

    def test_first_chunk_split_short(self):
        assert first_chunk_split("Your turn, go.") == ["Your turn, go."]

    def test_clean(self):
        assert clean_for_speech("**HP** is low") == "H.P. is low"

    def test_voice_for_language(self):
        assert voice_for("Spanish") == ("es", "ef_dora")
        assert voice_for("Klingon") == ("en-us", "af_heart")
        assert voice_for("English", "am_michael") == ("en-us", "am_michael")


class FakeMuse:
    """Scripted model: each call returns the next canned reply."""

    def __init__(self, replies=None, stream_text=""):
        self.replies = list(replies or [])
        self.stream_text = stream_text
        self.s = SimpleNamespace(image_max_side=640)
        self.seen: list[list[dict]] = []

    async def tools(self, messages, tools, effort=None, max_tokens=3000):
        self.seen.append(messages)
        calls = self.replies.pop(0)
        return SimpleNamespace(content="", tool_calls=[
            SimpleNamespace(id=f"c{i}", function=SimpleNamespace(
                name=name, arguments=json.dumps(args)),
                model_dump=lambda n=name, a=args, i=i: {
                    "id": f"c{i}", "type": "function",
                    "function": {"name": n, "arguments": json.dumps(a)}})
            for i, (name, args) in enumerate(calls)])

    async def stream(self, messages, effort=None, max_tokens=1500):
        for i in range(0, len(self.stream_text), 5):
            yield self.stream_text[i : i + 5]


def run(coro):
    return asyncio.run(coro)


class TestAgent:
    def make(self, replies, answer="yes"):
        muse = FakeMuse(replies)
        pad = DryRun()
        said: list[str] = []
        asked: list[str] = []

        async def grab():
            return frame(blank(), 0)

        async def speak(text):
            said.append(text)

        async def ask(q):
            asked.append(q)
            return answer

        agent = Agent(muse, pad, grab, speak, ask=ask, settle_s=0)
        return agent, pad, said, asked, muse

    def test_presses_then_finishes(self):
        agent, pad, said, _, _ = self.make([
            [("press", {"buttons": ["down"]})],
            [("press", {"buttons": ["a"]})],
            [("finish", {"text": "Done."})],
        ])
        result = run(agent.run("use skill"))
        assert pad.log == ["down", "a"]
        assert said == ["Done."]
        assert result.outcome == "Done."

    def test_ask_player_waits_for_answer(self):
        agent, pad, _, asked, muse = self.make([
            [("ask_player", {"question": "Shadow B. Go?"})],
            [("press", {"buttons": ["a"]})],
            [("finish", {"text": "Hit it."})],
        ], answer="yes")
        run(agent.run("attack"))
        assert asked == ["Shadow B. Go?"]
        assert pad.log == ["a"]
        tool_results = [m["content"] for m in muse.seen[1] if m["role"] == "tool"]
        assert tool_results[0] == 'The player said: "yes"'

    def test_unknown_buttons_are_dropped(self):
        agent, pad, _, _, _ = self.make([
            [("press", {"buttons": ["a", "selfdestruct"]})],
            [("finish", {"text": "ok"})],
        ])
        run(agent.run("x"))
        assert pad.log == ["a"]

    def test_gives_up_after_max_turns(self):
        agent, _, said, _, _ = self.make([[("press", {"buttons": ["a"]})]] * 10)
        agent.max_turns = 3
        result = run(agent.run("loop forever"))
        assert result.outcome == "gave up"
        assert "couldn't finish" in said[-1]


def test_never_confirms_without_looking():
    from playertwo.agent import split_before_confirm

    assert split_before_confirm(["a", "a", "a"]) == (["a", "a", "a"], "")
    assert split_before_confirm(["down", "down", "a"]) == (["down", "down"], "a")
    assert split_before_confirm(["a", "right", "a", "a"]) == (["a", "right"], "a")
    assert split_before_confirm(["down"]) == (["down"], "")


class TestNarrator:
    def collect(self, text):
        nar = Narrator(FakeMuse(stream_text=text))
        out = run(self._gather(nar))
        return nar, out

    async def _gather(self, nar):
        return "".join([d async for d in nar.narrate(frame(blank(), 0))])

    def test_silent_says_nothing(self):
        nar, out = self.collect("<silent>")
        assert out == ""
        assert not nar.recent

    def test_speech_is_streamed_and_remembered(self):
        nar, out = self.collect("Your turn. Shadow B is hurt.")
        assert out == "Your turn. Shadow B is hurt."
        assert nar.recent[-1] == "Your turn. Shadow B is hurt."

    def test_language_and_team_up_in_prompt(self):
        nar = Narrator(FakeMuse(), language="Spanish", team_up=True)
        system = nar.system()
        assert "Speak Spanish" in system
        assert "own screen narrator" in system
