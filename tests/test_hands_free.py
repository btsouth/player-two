"""Hands-free listening: what counts as said to Player Two, and when speech starts and ends."""

import asyncio
import time
from collections import deque
from types import SimpleNamespace

from playertwo.app import Events, Session
from playertwo.listen import FRAME, Segmenter, addressed


def test_its_name_starts_a_command_however_it_is_spelled():
    assert addressed("Player Two, open the menu.") == "open the menu"
    assert addressed("hey player 2 who's hurt?") == "who's hurt"
    assert addressed("PlayerPlayer Two.") == ""
    assert addressed("Player too, stop") == "stop"
    assert addressed("Yosuke attacks. Player Two, stop") == "stop"
    assert addressed("I think it's going to rain later") is None
    assert addressed("pass it to the player to the left") is None
    assert addressed("multiplayer is fun") is None


def test_speech_starts_with_its_lead_in_and_ends_after_a_pause():
    seg = Segmenter()
    frame = b"\0\0" * FRAME
    out = [seg.push(frame, p)[0] for p in [0.1, 0.9, 0.1, 0.9, 0.9, 0.9]]
    assert out == ["", "", "", "", "start", "speech"]  # one loud frame (a click) isn't speech
    out = [seg.push(frame, 0.1)[0] for _ in range(Segmenter.END_FRAMES)]
    assert out[-1] == "end" and "end" not in out[:-1]
    seg.push(frame, 0.9)
    what, frames = seg.push(frame, 0.9)
    assert what == "start" and len(frames) == 2


def test_long_speech_is_cut_into_pieces():
    seg = Segmenter()
    frame = b"\0\0" * FRAME
    whats = [seg.push(frame, 0.9)[0] for _ in range(Segmenter.MAX_FRAMES + 3)]
    assert "end" in whats


class FakeVoice:
    def __init__(self):
        self.speaking = False
        self.stopped = 0
        self.cues: list[str] = []

    def idle(self):
        return not self.speaking

    def stop(self):
        self.stopped += 1
        self.speaking = False

    def cue(self, name, pan=0.0):
        self.cues.append(name)


class FakeAgent:
    def __init__(self):
        self.held = False

    def hold(self):
        self.held = True

    def resume(self):
        self.held = False


def session(monkeypatch):
    s = Session.__new__(Session)
    s.events = Events()
    s.voice = FakeVoice()
    s.agent = FakeAgent()
    s.agent_task = None
    s.answer = None
    s.utt = None
    s.hands_active = s.transcribing = False
    s.woke_at = s.follow_until = s.spoke_at = 0.0
    s.spans = deque()
    s.stop_seen = (0.0, 0)
    s.hands = SimpleNamespace(utt=object(), started=0.0)
    s.narrating_text = ""
    s.handled = []

    async def handle(text):
        s.handled.append(text)

    s.handle = handle

    async def local(action):
        s.handled.append(f"local:{action}")

    s.local = local
    s._tasks = set()
    s.trouble = lambda *a, **k: s.handled.append("<trouble>")
    return s


async def heard(s, text, ago=2.0, length=1.5):
    """The final words of speech that began `ago` seconds ago and lasted `length`."""
    now = time.monotonic()
    await s.hands_heard(text, None, now - ago, now - ago + length)


def spoke(s, ago, length):
    """Player Two was speaking from `ago` seconds ago, for `length` seconds."""
    now = time.monotonic()
    s.spans.append([now - ago, now - ago + length])
    s.spoke_at = max(s.spoke_at, now - ago + length)


def partial(s, text, ago=1.0):
    s.hands.started = started = time.monotonic() - ago
    s.hands_partial(text, started)
    return started


async def test_only_what_is_said_to_it_is_acted_on(monkeypatch):
    s = session(monkeypatch)
    await heard(s, "Player Two, open the menu.")
    await heard(s, "I think it's going to rain later.")
    assert s.handled == ["open the menu"]
    assert [e["kind"] for e in s.events.recent if e["kind"] == "heard"] == ["heard"]


async def test_its_name_alone_waits_for_the_next_sentence(monkeypatch):
    s = session(monkeypatch)
    await heard(s, "Player Two.", ago=1.0, length=0.6)
    assert s.handled == [] and s.voice.cues[-1] == "listening"
    # Started inside the window, finished after a long sentence: still the follow-up.
    await heard(s, "Use Garu on the left one.", ago=0.5)
    assert s.handled == ["Use Garu on the left one."]
    await heard(s, "And then heal.", ago=0.5)
    assert s.handled == ["Use Garu on the left one."]  # the window closed after one


async def test_its_own_voice_is_not_a_command(monkeypatch):
    s = session(monkeypatch)
    s.events.emit("say", text="Hi, I'm Player Two. I'll tell you what's happening in "
                              "your game, and I can play it for you.", source="hello")
    spoke(s, 4, 4)
    await heard(s, "I'm Player Two. I'll tell you what's happening in your game", ago=4,
                length=3.5)
    assert s.handled == []
    # The player talking over it, name first, still counts.
    await heard(s, "Player Two, stop", ago=1, length=0.8)
    await asyncio.sleep(0)
    assert s.handled == ["local:stop"]


async def test_a_command_using_its_own_words_counts_once_it_has_finished(monkeypatch):
    s = session(monkeypatch)
    s.events.emit("say", text="I'm listening hands-free. Start with my name, like: Player "
                              "Two, open the menu.", source="hello")
    spoke(s, 8, 4)  # done four seconds ago
    await heard(s, "Player Two, open the menu.", ago=2)
    assert s.handled == ["open the menu"]


async def test_answers_need_no_name_but_its_own_question_is_not_one(monkeypatch):
    s = session(monkeypatch)
    s.answer = asyncio.get_running_loop().create_future()
    s.events.emit("ask", text="Agi on Shadow B. Go?")
    spoke(s, 3, 2)
    await heard(s, "Agi on Shadow B. Go?", ago=3, length=2)
    assert s.handled == []
    await heard(s, "Go.", ago=0.6, length=0.4)  # right after it finished asking
    assert s.handled == ["Go."]


async def test_stop_needs_no_name_while_it_is_busy(monkeypatch):
    s = session(monkeypatch)
    await heard(s, "Stop.")
    assert s.handled == []  # nothing to stop: someone else in the room
    s.voice.speaking = True
    await heard(s, "Stop.")
    assert s.handled == ["Stop."]


async def test_its_name_stops_its_voice_and_holds_an_action_until_it_hears_why(monkeypatch):
    s = session(monkeypatch)
    loop = asyncio.get_running_loop()
    s.agent_task = loop.create_future()  # an action in progress
    s.voice.speaking = True
    started = partial(s, "Player Two")
    assert s.hands_active and s.agent.held and s.voice.stopped == 1
    assert s.listening()
    now = time.monotonic()
    await s.hands_heard("Player Two, what's my HP?", None, started, now)
    assert s.handled == ["what's my HP"] and not s.agent.held and not s.hands_active
    s.agent_task.cancel()


async def test_speech_cut_off_unheard_lets_go_of_the_action(monkeypatch):
    s = session(monkeypatch)
    s.agent_task = asyncio.get_running_loop().create_future()
    partial(s, "Player Two, open")
    assert s.agent.held
    s.hands_dropped()  # the mic went away mid-sentence
    assert not s.agent.held and not s.hands_active and not s.listening()
    s.hands.utt = None
    partial(s, "Player Two, open the")  # a late partial of the dropped speech
    assert not s.hands_active
    s.agent_task.cancel()


async def test_partial_echo_of_its_own_name_does_not_cut_it_off(monkeypatch):
    s = session(monkeypatch)
    s.events.emit("say", text="Player Two here. I can see Persona 4 Golden.", source="hello")
    s.voice.speaking = True
    s.spoke_at = time.monotonic()
    partial(s, "Player Two here I can see")
    assert not s.hands_active and s.voice.stopped == 0


async def test_its_greeting_heard_back_is_not_a_command(monkeypatch):
    s = session(monkeypatch)
    greeting = ("Hi, I'm Player Two. I'll tell you what's happening in your game, and I can "
                "play it for you. I'm listening hands-free. Start with my name, like: Player "
                "Two, open the menu.")
    s.events.emit("say", text=greeting, source="hello")
    s.voice.speaking = True
    spoke(s, 6, 6)
    partial(s, "Hi, I'm Player Two, I'll")
    assert not s.hands_active and s.voice.stopped == 0
    await heard(s, "Hi, I'm Player Two. I'll tell you what's", ago=6, length=2)
    await heard(s, "Start with my name like Player Two open the menu", ago=3, length=2.5)
    assert s.handled == []


async def test_player_two_stop_works_mixed_into_its_own_voice(monkeypatch):
    s = session(monkeypatch)
    s.spawn = lambda coro: asyncio.ensure_future(coro)
    s.events.emit("say", text="Start with my name, Player Two, and ask me anything about "
                              "the game. You can also say: describe, repeat, stop, help.")
    s.voice.speaking = True
    spoke(s, 9, 9)
    started = partial(s, "start with my name. Player Two and ask me anything", ago=8)
    s.hands_partial("start with my name. Player Two and ask me anything about the game like."
                    " Player Two stop. Open the", started)
    s.hands_partial("start with my name. Player Two and ask me anything about the game like."
                    " Player Two stop. Open the menu or", started)  # the same stop again
    await asyncio.sleep(0)
    assert s.handled == ["local:stop"]
    await s.hands_heard("start with my name. Player Two and ask me anything about the game "
                        "like. Player Two stop. Open the menu", None, started,
                        time.monotonic())
    assert s.handled == ["local:stop"]


async def test_stop_then_a_new_request_in_one_breath(monkeypatch):
    s = session(monkeypatch)
    s.voice.speaking = True
    now = time.monotonic()
    await s.hands_heard("Player Two, stop. Player Two, open the map.", None, now - 3, now - 1)
    assert s.handled == ["local:stop", "open the map"]
