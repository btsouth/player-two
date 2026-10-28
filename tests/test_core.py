import asyncio
import json
import time
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

        agent = Agent(muse, pad, grab, speak, ask=ask, max_settle_s=0.2)
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

    def test_same_opening_twice_is_silent(self):
        nar = Narrator(FakeMuse(stream_text='Teddie says: "Sensei, are you leaving?" Press A.'))
        first = run(self._gather(nar))
        nar.muse.stream_text = 'Teddie says: "Sensei, are you leaving?" Still the stage.'
        second = run(self._gather(nar))
        assert first.startswith("Teddie") and second == ""
        assert len(nar.recent) == 1

    def test_reworded_opening_is_a_repeat_but_a_new_turn_is_not(self):
        nar = Narrator(FakeMuse())
        nar.recent.append("You're at the Entrance on the TV World stage. Teddie is here.")
        assert nar.repeats("You're at the Entrance of the TV World stage.")
        nar.recent.append("Your turn. Shadow A is hurt.")
        assert nar.repeats("Your turn. Shadow A is hurt.")
        assert not nar.repeats("Your turn. Shadow B is down, one more.")
        assert not nar.repeats("Your turn. Shadow B is hurt.")
        assert not nar.repeats("Yosuke's turn. Attack is selected.")
        assert not nar.repeats("Chie says: I'm starving.")

    def test_language_and_team_up_in_prompt(self):
        nar = Narrator(FakeMuse(), language="Spanish", team_up=True)
        system = nar.system()
        assert "Speak Spanish" in system
        assert "own screen narrator" in system


def test_hud_reads_bar_and_warns():
    from playertwo.hud import HudWatch, Meter

    def bar(fill: float) -> Image.Image:
        arr = np.full((200, 400, 3), 200, np.uint8)
        arr[20:30, 40:240] = (60, 60, 60)  # empty track
        arr[20:30, 40 : 40 + int(200 * fill)] = (170, 40, 40)  # red fill
        return Image.fromarray(arr)

    m = Meter("health", (0.09, 0.09, 0.61, 0.16), (170, 40, 40), estimate=1.0)
    m.locate(bar(1.0))
    watch = HudWatch([m])
    levels = []
    for i, f in enumerate([1.0, 0.6, 0.25, 0.1, 0.1, 0.8]):
        levels += [a.level for a in watch.update(bar(f), float(i))]
        assert abs(m.history[-1] - f) < 0.05
    assert levels == ["low", "critical", "recovered"]


def test_where_questions_and_wording():
    from playertwo.locate import Match, describe, parse_boxes, where_request

    assert where_request("Where's the door?") == "door"
    assert where_request("how many enemies on screen") == "enemies"
    assert where_request("use Zio on the left one") is None
    boxes = parse_boxes("<0f>0<|box;x1=0;y1=0;x2=100;y2=100;w=1000;h=1000|><|mask;x=0|>,"
                        "1<|box;x1=800;y1=400;x2=900;y2=600;w=1000;h=1000|>")
    assert [round(b.cx, 2) for b in boxes] == [0.05, 0.85]
    assert describe("doors", []) == "I can't find any doors on screen."
    assert describe("door", [Match(0.1, 0.5, 0.05)]) == "Found it: on the far left."
    two = [Match(0.7, 0.5, 0.01), Match(0.3, 0.5, 0.01)]
    assert describe("shadows", two) == "2 shadows: one on the left, one on the right."
    many = [Match(x, 0.5, 0.01) for x in (0.1, 0.2, 0.5, 0.9)]
    assert describe("desks", many) == "4 desks: 2 on the left, 1 in the middle, 1 on the right."


async def test_screen_reader_streams_sentences_in_order():
    from playertwo.audio import Mixer
    from playertwo.config import Settings
    from playertwo.screenreader import Backend, ScreenReaderVoice

    class Fake(Backend):
        name = "Fake"

        def __init__(self):
            self.said, self.stops = [], 0

        async def speak(self, text):
            self.said.append(text)

        async def stop(self):
            self.stops += 1

    async def deltas():
        for d in ["Yosuke: Dude, ", "that was close. Your HP", " is low."]:
            yield d

    fake = Fake()
    v = ScreenReaderVoice(Settings(), Mixer("file"), fake)
    out = await v.say_stream(deltas())
    await asyncio.sleep(0)
    assert fake.said == ["Yosuke: Dude, that was close.", "Your H.P. is low."]
    assert out == v.last_spoken and fake.stops == 1

    fake.said.clear()

    async def silent():
        yield "<silent>"

    assert await v.say_stream(silent(), interrupt=False) == ""
    assert fake.said == []


def test_local_commands_only_match_whole_utterances():
    from playertwo.app import local_action

    assert local_action("Stop.") == "stop"
    assert local_action("hey, repeat that please") == "repeat"
    assert local_action("What's on screen?") == "describe"
    assert local_action("resume") == "resume"
    assert local_action("keep going, but don't leave, I want to stay here") is None
    assert local_action("stop attacking and heal Yosuke") is None
    assert local_action("tell me more about Teddie") is None
    assert local_action("describe the enemy on the left") is None


def test_trim_images_keeps_only_the_newest_screens():
    from playertwo.agent import trim_images

    def shot(n):
        return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": n}},
                                            {"type": "text", "text": n}]}

    msgs = [{"role": "system", "content": "rules"}, shot("1"), {"role": "assistant",
            "content": ""}, shot("2"), shot("3")]
    out = trim_images(msgs, keep=2)
    urls = [c["image_url"]["url"] for m in out if isinstance(m["content"], list)
            for c in m["content"] if c["type"] == "image_url"]
    assert urls == ["2", "3"]
    assert out[1]["content"][0] == {"type": "text", "text": "(an earlier screen)"}
    assert msgs[1]["content"][0]["type"] == "image_url"  # the original is untouched


def test_screen_change_and_settle_return_quickly_on_a_still_screen():
    from playertwo.agent import screen_change

    still = frame(blank(), 0)
    assert screen_change(still, frame(blank(), 1)) == 0.0
    agent = Agent(FakeMuse([]), DryRun(), None, None, max_settle_s=2.0)

    async def grab():
        return still

    agent.grab = grab
    t = time.monotonic()
    run(agent._settled(0.0))
    assert time.monotonic() - t < 0.5
