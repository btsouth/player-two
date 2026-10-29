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

    def test_numbers_sent_as_text_or_decimals_still_work(self):
        agent, pad, _, _, _ = self.make([
            [("hold", {"input": "left_stick", "direction": "up", "ms": "400.0"})],
            [("press", {"buttons": ["down", "down"], "gap_ms": 90.5})],
            [("finish", {"text": "Done."})],
        ])
        run(agent.run("walk up"))
        assert pad.log == ["left:up:400", "down", "down"]

    def test_an_answer_isnt_followed_by_done(self):
        agent, _, said, _, _ = self.make([
            [("say", {"text": "You're on a stage, with Yosuke next to you."})],
            [("finish", {})],
        ])
        run(agent.run("where am I?"))
        assert said == ["You're on a stage, with Yosuke next to you."]

    def test_running_out_of_steps_asks_for_where_things_stand(self):
        agent, _, said, _, muse = self.make(
            [[("press", {"buttons": ["up"]})]] * 2 + [[("finish", {"text": "Got halfway."})]])
        agent.max_turns = 3
        result = run(agent.run("walk to the stairs"))
        assert said == ["Got halfway."] and result.outcome == "Got halfway."
        last = muse.seen[-1][-1]["content"][-1]["text"]
        first = muse.seen[1][-1]["content"][-1]["text"]
        assert "out of steps" in last and "out of steps" not in first

    def test_finish_doesnt_repeat_what_say_just_said(self):
        agent, _, said, _, _ = self.make([
            [("say", {"text": "Your HP is 42 out of 50."}),
             ("finish", {"text": "That's 42 out of 50."})],
        ])
        run(agent.run("what's my HP?"))
        assert said == ["Your HP is 42 out of 50."]
        agent, _, said, _, _ = self.make([
            [("say", {"text": "Your HP is 42 out of 50."}),
             ("finish", {"text": "You also have 3 potions."})],
        ])
        run(agent.run("what's my HP?"))
        assert said == ["Your HP is 42 out of 50.", "You also have 3 potions."]
        agent, _, said, _, _ = self.make([
            [("say", {"text": "I'll open the map."}),
             ("finish", {"text": "I couldn't open the map."})],
        ])
        run(agent.run("open the map"))
        assert said == ["I'll open the map.", "I couldn't open the map."]

    def test_holding_the_talk_key_pauses_presses_until_released(self):
        agent, pad, said, _, _ = self.make([
            [("press", {"buttons": ["down", "down", "right"]})],
            [("finish", {"text": "Done."})],
        ])

        async def go():
            agent.hold()
            task = asyncio.create_task(agent.run("buy two potions"))
            await asyncio.sleep(0.3)
            assert pad.log == [] and said == []  # nothing while the player talks
            agent.resume()  # they asked "what?", not stop
            return await task

        result = run(go())
        assert pad.log == ["down", "down", "right"] and result.outcome == "Done."

    def test_cancel_while_held_stops_without_pressing(self):
        agent, pad, said, _, _ = self.make([
            [("press", {"buttons": ["down", "a"]})],
            [("finish", {"text": "Done."})],
        ])

        async def go():
            agent.hold()
            task = asyncio.create_task(agent.run("attack"))
            await asyncio.sleep(0.2)
            agent.cancel()
            return await task

        result = run(go())
        assert pad.log == [] and said == [] and result.outcome == "cancelled"

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

    # A second a skips a line the player never heard; start then a picks from a menu
    # unseen; b then a answers a Yes/No nobody read.
    assert split_before_confirm(["a", "a", "a"]) == (["a"], "a")
    assert split_before_confirm(["down", "down", "a"]) == (["down", "down"], "a")
    assert split_before_confirm(["a", "right", "a", "a"]) == (["a", "right"], "a")
    assert split_before_confirm(["start", "a"]) == (["start"], "a")
    assert split_before_confirm(["b", "b", "a"]) == (["b", "b"], "a")
    assert split_before_confirm(["down"]) == (["down"], "")
    assert split_before_confirm(["a"]) == (["a"], "")


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
        # New dialogue after a repeated opening is still said; a changed number is news.
        nar.muse.stream_text = ('Teddie says: "Sensei, are you leaving?" Still the stage. '
                                'Yosuke: "Wait up!"')
        assert run(self._gather(nar)) == 'Yosuke: "Wait up!"'
        nar.muse.stream_text = "Your turn. Chie has 60 HP."
        run(self._gather(nar))
        nar.muse.stream_text = "Your turn. Chie has 40 HP."
        assert run(self._gather(nar)) == "Your turn. Chie has 40 HP."

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
    assert where_request("where are we") is None
    assert where_request("Where are we going?") is None
    assert where_request("how many enemies on screen") == "enemies"
    assert where_request("how many enemies are there") == "enemies"
    assert where_request("how many enemies are on screen") == "enemies"
    assert where_request("use Zio on the left one") is None
    assert where_request("find a healing item and use it") is None
    assert where_request("how many potions do I have") is None
    assert where_request("show me my items") is None
    boxes = parse_boxes("<0f>0<|box;x1=0;y1=0;x2=100;y2=100;w=1000;h=1000|><|mask;x=0|>,"
                        "1<|box;x1=800;y1=400;x2=900;y2=600;w=1000;h=1000|>")
    assert [round(b.cx, 2) for b in boxes] == [0.05, 0.85]
    assert describe("doors", []) == "I can't find any doors on screen."
    assert describe("door", [Match(0.1, 0.5, 0.05)]) == "Found it: on the far left."
    # From where the player's character stands, nearest first, in its own steps.
    from playertwo.locate import describe_from, pan_from

    me, step = (0.5, 0.5), (0.05, 0.1)
    assert describe_from("door", [Match(0.35, 0.2, 0.01)], me, step) == \
        "Found it: 3 steps up and 3 to your left."
    assert describe_from("chest", [Match(0.52, 0.53, 0.01)], me, step) == \
        "Found it: right next to you."
    two = describe_from("guards", [Match(0.9, 0.5, 0.01), Match(0.5, 0.8, 0.01)], me, step)
    assert two == "2 guards. The nearest is 3 steps down; then 8 steps to your right."
    assert pan_from(Match(0.3, 0.5, 0.01), me) < 0 < pan_from(Match(0.6, 0.5, 0.01), me)
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
            return True

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
    assert local_action("resume narration") == "resume"
    assert local_action("pause") is None  # the game's pause: the agent's job
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


async def test_screen_reader_that_refuses_lines_falls_back_to_our_voice(monkeypatch):
    from playertwo import voice as voice_mod
    from playertwo.audio import Mixer
    from playertwo.config import Settings
    from playertwo.screenreader import Backend, ScreenReaderVoice

    class Asleep(Backend):
        name = "NVDA"

        async def speak(self, text):
            return False

    class Own:
        def __init__(self, s, mixer):
            self.said = []

        async def say(self, text, interrupt=True):
            self.said.append(text)

        def idle(self):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(voice_mod, "Voice", Own)
    v = ScreenReaderVoice(Settings(), Mixer("file"), Asleep())
    await v.say("Shadow ahead.")
    await v.say("It's weak to fire.", interrupt=False)
    assert v._own.said[0].startswith("NVDA isn't taking my lines")
    assert v._own.said[1:] == ["Shadow ahead.", "It's weak to fire."]


def test_modal_token_only_goes_to_a_modal_address(tmp_path, monkeypatch):
    from playertwo import config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    for name in ("MODAL_PROXY_KEY", "MODAL_PROXY_SECRET", "PLAYER_TWO_OPEN_API_KEY",
                 "BASETEN_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "modal.env").write_text("MODAL_PROXY_KEY=wk-1\nMODAL_PROXY_SECRET=ws-1\n")
    (tmp_path / "open-api-key").write_text("bt-key\n")
    key, headers = config.open_credentials("https://ws--ep-server.us-west.modal.direct/v1")
    assert key == "modal" and headers["Modal-Key"] == "wk-1"
    assert config.open_credentials("https://inference.baseten.co/v1") == ("bt-key", {})


async def test_two_lines_at_once_are_said_one_after_the_other():
    import asyncio
    import time

    import numpy as np

    from playertwo.voice import Voice

    v = Voice.__new__(Voice)
    v._gen, v._lock, v.first_audio_ms, v.last_spoken = 0, asyncio.Lock(), [], ""
    heard: list[str] = []

    def synth(sentence):
        time.sleep(0.01)
        return np.array([len(heard)], np.float32), sentence

    v._synth = lambda s: synth(s)
    v.mixer = type("M", (), {"push_speech": lambda self, x: heard.append(x[1]),
                             "clear_speech": lambda self: None})()
    await asyncio.gather(v.say("One. Two. Three.", interrupt=False),
                         v.say("Four. Five. Six.", interrupt=False))
    assert heard == ["One.", "Two.", "Three.", "Four.", "Five.", "Six."], heard
    # A line still waiting its turn is dropped by a stop.
    heard.clear()
    first = asyncio.create_task(v.say("Seven. Eight.", interrupt=False))
    waiting = asyncio.create_task(v.say("Nine.", interrupt=False))
    await asyncio.sleep(0)
    v.stop()
    await asyncio.gather(first, waiting)
    assert "Nine." not in heard


class TestAgentHonesty:
    make = TestAgent.make

    def test_a_bare_button_name_is_one_press_and_nonsense_presses_nothing(self):
        agent, pad, _, _, muse = self.make([
            [("press", {"buttons": "start"})],
            [("press", {"buttons": ["jump"]})],
            [("finish", {"text": "ok"})],
        ])
        run(agent.run("open the menu"))
        assert pad.log == ["start"]  # not s, t, a, r, t
        results = [m["content"] for m in muse.seen[-1] if m["role"] == "tool"]
        assert results[1].startswith("nothing pressed")

    def test_the_line_is_said_before_the_press_that_moves_it_on(self):
        agent, pad, said, _, _ = self.make([
            [("press", {"buttons": ["a"], "say": "Mom: Crono! Wake up!"})],
            [("finish", {"text": "Mom's waking you up."})],
        ])
        run(agent.run("read it to me"))
        assert said[0] == "Mom: Crono! Wake up!" and pad.log == ["a"]

    def test_no_finishing_before_looking_and_no_saying_it_twice(self):
        agent, _pad, said, _, muse = self.make([
            [("press", {"buttons": ["b"]}), ("finish", {"text": "Closed it."})],
            [("say", {"text": "The menu is closed."})],
            [("say", {"text": "The menu is closed."})],
            [("finish", {"text": ""})],
        ])
        result = run(agent.run("close the menu"))
        results = [m["content"] for m in muse.seen[-1] if m["role"] == "tool"]
        assert results[1].startswith("not finished")
        assert said == ["The menu is closed."]
        assert "don't say it again" in results[3]
        assert result.outcome == "The menu is closed."  # kept for the next request

    def test_two_failed_walks_in_a_row_say_that_way_is_cut_off(self):
        agent, _, _, _, muse = self.make([
            [("walk_to", {"target": "the fair"})],
            [("walk_to", {"target": "the colorful tents"})],
            [("finish", {"text": "I can't get there from here."})],
        ])

        async def blocked(name, a):
            return "blocked: something is in the way, about 6 steps up from it (20.0 s)"

        agent._act = blocked
        run(agent.run("take me to the fair"))
        results = [m["content"] for m in muse.seen[-1] if m["role"] == "tool"]
        assert "cut off" not in results[0] and "cut off" in results[1]

    def test_skipping_text_the_player_asked_to_skip_is_allowed(self):
        agent, pad, _, _, _ = self.make([
            [("press", {"buttons": ["a", "a", "a"], "skipping": True})],
            [("press", {"buttons": ["down", "a"], "skipping": True})],  # not a skip
            [("finish", {"text": "Skipped the intro."})],
        ])
        run(agent.run("skip the intro"))
        assert pad.log == ["a", "a", "a", "down"]

    def test_holding_a_doesnt_get_around_looking(self):
        agent, pad, _, _, muse = self.make([
            [("hold", {"input": "a", "ms": 3000})],
            [("finish", {"text": "ok"})],
        ])
        run(agent.run("go"))
        assert pad.log == []
        assert next(m["content"] for m in muse.seen[-1] if m["role"] == "tool").startswith(
            "not held")

    def test_done_is_said_after_saying_what_it_was_about_to_do(self):
        agent, _, said, _, _ = self.make([
            [("say", {"text": "I'll save the game in slot 1."}),
             ("press", {"buttons": ["a"]})],
            [("finish", {"text": "Saved the game in slot 1."})],
        ])
        run(agent.run("save in slot 1"))
        assert said == ["I'll save the game in slot 1.", "Saved the game in slot 1."]

    def test_nothing_is_said_about_an_action_before_looking(self):
        agent, _, said, _, muse = self.make([
            [("press", {"buttons": ["start"]}), ("say", {"text": "The menu is open now."})],
            [("finish", {"text": "The menu is open."})],
        ])
        run(agent.run("open the menu"))
        assert said == ["The menu is open."]
        assert [m["content"] for m in muse.seen[-1] if m["role"] == "tool"][1].startswith(
            "not said")

    def test_never_ends_in_silence(self):
        agent, _, said, _, muse = self.make([
            [],  # nothing at all: its thinking ran out of room
            [("press", {"buttons": ["b"]})],
            [("finish", {})],  # finished without saying anything
            [("finish", {"text": "Closed the menu."})],
        ])
        run(agent.run("close the menu"))
        assert said == ["Closed the menu."]
        results = [m["content"] for m in muse.seen[-1] if m["role"] == "tool"]
        assert results[1].startswith("not finished: give the text")

    def test_tools_not_offered_dont_run(self):
        agent, pad, _, _, muse = self.make([
            [("teleport", {"to": "the fair"})],
            [("press", {"buttons": ["down", "down"]})],
            [("finish", {"text": "ok"})],
        ])
        agent.max_turns = 2
        run(agent.run("go"))
        results = [m["content"] for m in muse.seen[-1] if m["role"] == "tool"]
        assert results[0].startswith("no tool called teleport")
        assert pad.log == []  # the last turn could only finish

    def test_a_hold_cut_short_says_so(self):
        agent, _, _, _, muse = self.make([
            [("hold", {"input": "down", "ms": 2000})],
            [("finish", {"text": "ok"})],
        ])

        async def go():
            task = asyncio.create_task(agent.run("walk down"))
            await asyncio.sleep(0.3)
            agent.go.clear()  # the player starts talking
            await asyncio.sleep(0.2)
            agent.go.set()
            await task

        run(go())
        result = next(m["content"] for m in muse.seen[-1] if m["role"] == "tool")
        assert result.startswith("held down for only") and "the player spoke" in result

    def test_the_last_turn_can_only_finish(self):
        agent, _, _, _, muse = self.make([[("wait", {"seconds": 0.5})]] * 2
                                         + [[("finish", {"text": "Still loading."})]])
        agent.max_turns = 3
        tools_seen = []
        real = muse.tools

        async def tools(messages, tools, **kw):
            tools_seen.append([t["function"]["name"] for t in tools])
            return await real(messages, tools, **kw)

        muse.tools = tools
        run(agent.run("wait for it"))
        assert tools_seen[-1] == ["finish"] and len(tools_seen[0]) > 1
