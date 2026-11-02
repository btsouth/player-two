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
    s.woke_at = s.woke_for = s.follow_until = s.spoke_at = 0.0
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


async def test_game_speech_cannot_cancel_with_a_bare_stop(monkeypatch):
    s = session(monkeypatch)
    await heard(s, "Stop.")
    assert s.handled == []  # nothing to stop: someone else in the room
    s.voice.speaking = True
    await heard(s, "Stop.")
    assert s.handled == []
    s.answer = asyncio.get_running_loop().create_future()
    await heard(s, "Stop.")
    assert s.handled == []
    await heard(s, "Player Two, stop.")
    assert s.handled == ["local:stop"]


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


async def test_commands_that_start_with_stop_are_not_a_plain_stop(monkeypatch):
    s = session(monkeypatch)
    s.spawn = lambda coro: asyncio.ensure_future(coro)
    s.voice.speaking = True
    for text in ("Player Two, stop narrating.", "Player Two, quiet mode.",
                 "Player Two, stop watching my health."):
        started = partial(s, text)
        await s.hands_heard(text, None, started, time.monotonic())
    await asyncio.sleep(0)
    assert s.handled == ["stop narrating", "quiet mode", "stop watching my health"]


async def test_stop_without_its_name_from_the_room_is_ignored(monkeypatch):
    s = session(monkeypatch)
    s.spawn = lambda coro: asyncio.ensure_future(coro)
    s.agent_task = asyncio.get_running_loop().create_future()
    started = partial(s, "tell that player to stop camping.")
    await s.hands_heard("tell that player to stop camping.", None, started, time.monotonic())
    await asyncio.sleep(0)
    assert s.handled == []
    s.agent_task.cancel()


async def test_a_new_sentence_keeps_its_hold_when_the_last_one_finishes(monkeypatch):
    s = session(monkeypatch)
    s.agent_task = asyncio.get_running_loop().create_future()
    first = time.monotonic() - 3
    second = partial(s, "Player Two, wait", ago=0.5)  # the next sentence, already going
    assert s.agent.held
    await s.hands_heard("Player Two, open the map.", None, first, first + 1.5)
    assert s.handled == ["open the map"]
    assert s.hands_active and s.agent.held  # still listening to the second one
    await s.hands_heard("Player Two, wait.", None, second, time.monotonic())
    assert not s.hands_active and not s.agent.held
    s.agent_task.cancel()


async def test_okay_stopped_is_not_cut_off_by_the_rest_of_the_same_stop(monkeypatch):
    s = session(monkeypatch)
    s.spawn = lambda coro: asyncio.ensure_future(coro)
    s.voice.speaking = True
    started = partial(s, "Player Two stop.")
    s.voice.speaking = True  # now saying "Okay, stopped."
    s.hands_partial("Player Two stop. Um", started)
    await asyncio.sleep(0)
    assert s.handled == ["local:stop"] and s.voice.stopped == 0 and not s.hands_active


async def test_completed_transcription_timing_is_logged_without_guessing(monkeypatch):
    s = session(monkeypatch)
    timing = {"transcribe_s": 2.34, "transcribe_finalize_s": 0.12, "transcribe_audio_s": 2.08}
    now = time.monotonic()
    await s.hands_heard("Player Two, open the menu", None, now - 2, now, timing)
    event = next(e for e in s.events.recent if e["kind"] == "heard")
    assert event["text"] == "open the menu"
    assert event["raw_transcript"] == "Player Two, open the menu"
    assert {k: event[k] for k in timing} == timing


async def test_partial_stop_does_not_invent_a_completed_asr_time(monkeypatch):
    s = session(monkeypatch)
    now = time.monotonic()
    s.hands.started = now
    s.hands_partial("Player Two, stop.", now)
    await asyncio.sleep(0)
    first = next(e for e in s.events.recent if e["kind"] == "heard")
    assert "transcribe_s" not in first
    assert "utterance_started_at" in first
    timing = {"transcribe_s": 1.5, "transcribe_finalize_s": 0.1, "transcribe_audio_s": 1.4}
    stages = []
    emit = s.events.emit
    def capture(kind, **data):
        if kind == "model_stage":
            stages.append(data)
        emit(kind, **data)
    s.events.emit = capture
    await s.hands_heard("Player Two, stop.", None, now, now + 1, timing)
    assert len(stages) == 1 and stages[0]["seconds"] == 1.5
    assert s.handled.count("local:stop") == 1


async def test_utterance_timing_measures_stream_not_text(monkeypatch):
    from playertwo.listen import RATE, Utterance
    clock = [10.0]
    monkeypatch.setattr("playertwo.listen.time.monotonic", lambda: clock[0])
    u = Utterance()
    async def session():
        clock[0] = 12.6
        return "Open the menu."
    monkeypatch.setattr(u, "_session", session)
    u.start()
    u.feed(b"\0\0" * RATE)
    clock[0] = 12.5
    u.finish()
    assert u.timing() == {}
    assert await u.text() == "Open the menu."
    assert u.timing() == {"transcribe_s": 2.6, "transcribe_finalize_s": 0.1,
                          "transcribe_audio_s": 1.0}


async def test_a_different_target_is_not_its_own_greeting_echo(monkeypatch):
    s = session(monkeypatch)
    s.events.emit("say", text="I'm Player Two. Start with my name, like Player Two, open the menu.")
    spoke(s, ago=2, length=3)
    s.voice.speaking = True
    await heard(s, "Player Two, open the journal.", ago=1.5, length=1)
    assert s.handled == ["open the journal"]
    assert s.voice.stopped == 1  # final genuine request cuts off its greeting


async def test_its_actual_named_greeting_echo_still_does_not_act(monkeypatch):
    s = session(monkeypatch)
    s.events.emit("say", text="I'm Player Two. Start with my name, like Player Two, open the menu.")
    spoke(s, ago=2, length=3)
    s.voice.speaking = True
    await heard(s, "Player Two, open the menu.", ago=1.5, length=1)
    assert s.handled == []


async def test_a_bare_answer_keeps_its_unaddressed_warning_for_the_agent(monkeypatch):
    s = session(monkeypatch)
    s.paused = False
    s.agent.cancelled = False
    s.handle = Session.handle.__get__(s, Session)
    async def say(*args, **kwargs):
        return None
    async def wait_idle(*args, **kwargs):
        return None
    s.voice.say = say
    s.voice.wait_idle = wait_idle
    ask = asyncio.create_task(s.agent_ask("How many?", timeout=1))
    await asyncio.sleep(0)
    await heard(s, "Three.")
    answer = await ask
    assert answer.startswith("Three.")
    assert 'heard hands-free without "Player Two"' in answer
    assert "for anything hard to undo, ask again" in answer


async def test_stop_after_an_explicit_name_only_call_is_still_addressed(monkeypatch):
    s = session(monkeypatch)
    await heard(s, "Player Two.")
    assert s.follow_until > time.monotonic()
    await heard(s, "Stop.")
    assert s.handled == ["Stop."]


async def test_an_obviously_truncated_action_is_not_sent_to_the_agent(monkeypatch):
    s = session(monkeypatch)
    replies = []
    async def announce(text, **kwargs):
        replies.append(text)
    s.announce = announce
    await heard(s, "Player Two, attack the.")
    assert s.handled == []
    assert replies and "whole request" in replies[0]
    # A concrete target is not a grammatical truncation, including enemy A.
    await heard(s, "Player Two, attack the one on the left.")
    await heard(s, "Player Two, attack A.")
    assert s.handled == ["attack the one on the left", "attack A"]


def test_damaged_names_need_a_command_at_the_start():
    from playertwo.listen import wake_match
    for name in ('Two', 'Too', '2', 'There too', 'Layer two', 'Prayer two', 'Play two', 'Player,'):
        match = wake_match(f'{name} read this menu.')
        assert match and match[0] == 'read this menu' and match[2]
    assert addressed('two paused the game.') == 'paused the game'
    for text in ('Two shadows block the gate', 'Two of them left', 'There too', 'Two',
                 'The game says two read this menu', 'A prayer two read this menu',
                 'Player read this menu', 'There too, shadows wait'):
        assert addressed(text) is None, text


async def test_damaged_names_are_logged_and_still_subject_to_echo(monkeypatch):
    s = session(monkeypatch)
    await heard(s, 'Two, read this menu.')
    event = next(e for e in s.events.recent if e['kind'] == 'heard')
    assert s.handled == ['read this menu']
    assert event['wake_form'] == 'Two' and event['wake_damaged'] is True
    s.handled.clear()
    s.events.emit('say', text='Two, read this menu.', source='narrator')
    spoke(s, ago=2, length=2)
    await heard(s, 'Two, read this menu.', ago=1.5, length=1)
    assert s.handled == []
    s.events.emit('say', text='Two, stop.', source='narrator')
    await heard(s, 'Two, stop.', ago=1.5, length=1)
    assert s.handled == []


def test_soft_onset_keeps_twelve_frames_without_moving_vad_onset(monkeypatch):
    from playertwo import listen
    clock = [10.0]
    monkeypatch.setattr(listen.time, 'monotonic', lambda: clock[0])
    class Buffered:
        def __init__(self, *args, **kwargs):
            self.frames = []
        def start(self):
            pass
        def feed(self, frame):
            self.frames.append(frame)
    monkeypatch.setattr(listen, 'Utterance', Buffered)
    h = listen.HandsFree.__new__(listen.HandsFree)
    h.paused = lambda: False
    h.segmenter = Segmenter()
    h.utt = None
    h.keywords = list
    h.on_partial = lambda *args: None
    probabilities = iter([0.1] * 18 + [0.9, 0.9])
    h.detector = lambda frame: next(probabilities)
    for i in range(20):
        clock[0] = 10.0 + (i + 1) * Segmenter.FRAME_S
        h.feed(bytes([i]) * FRAME * 2)
    assert len(h.utt.frames) == 12
    assert h.utt.frames[0][0] == 8
    assert h.started == clock[0] - 0.16
    assert h.audio_started == clock[0] - 0.96
    assert sum(map(len, h.utt.frames)) == 46_080


async def test_preroll_tail_does_not_steal_a_new_named_request(monkeypatch):
    s = session(monkeypatch)
    now = time.monotonic()
    s.spans.append([now - 3, now - 2])
    s.events.emit('say', text='Player Two, open the menu.', source='hello')
    timing = {'utterance_audio_started_at': time.time() - 3}
    await s.hands_heard('Player Two, open the menu. Player Two, read this menu.', None,
                        now - 2, now, timing)
    assert s.handled == ['read this menu']
    s.handled.clear()
    await s.hands_heard('Player Two, open the menu.', None, now - 2, now, timing)
    assert s.handled == []
