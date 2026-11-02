"""Hearing the game's own voices: speech stretches, finding the game's sound stream, and
narration that waits for gaps instead of talking over the game or reading its subtitles."""

import asyncio
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
from PIL import Image

from playertwo import gameaudio
from playertwo.app import Session, local_action
from playertwo.audio import Mixer
from playertwo.gameaudio import (
    GameEars,
    SpeechTrack,
    game_streams,
    known_voicing,
    voiced_hint,
    window_identity,
)
from playertwo.narrator import Narrator
from playertwo.prompts import VOICE_NOW, VOICED_GAME
from playertwo.screen import ChangeWatcher, Frame
from playertwo.voice import Voice


def feed(track, seconds, p, t0):
    """Push frames of probability p for `seconds`, from t0; returns the end time."""
    t = t0
    for _ in range(round(seconds / track.frame_s)):
        t += track.frame_s
        track.push(t, p)
    return t


def test_a_stretch_of_speech_starts_where_it_began_and_ends_at_its_last_word():
    starts = []
    tr = SpeechTrack(on_start=starts.append)
    t = feed(tr, 0.96, 0.0, 0.0)
    assert not tr.speaking() and not tr.ever()
    t = feed(tr, 1.2, 0.9, t)  # someone talks from 0.96 s to 2.16 s
    assert tr.speaking()
    assert len(starts) == 1 and abs(starts[0] - 0.96) < 1e-6
    t = feed(tr, 0.32, 0.0, t)  # a breath: still the same line
    assert tr.speaking()
    t = feed(tr, 1.0, 0.0, t)
    assert not tr.speaking()
    (start, end), = tr.segments
    assert abs(start - 0.96) < 1e-6 and abs(end - 2.16) < 1e-6
    assert abs(tr.quiet_for(t) - (t - 2.16)) < 1e-6
    assert tr.heard_between(2.0, 5.0) and not tr.heard_between(2.3, 5.0)


def test_a_short_burst_isnt_speech():
    tr = SpeechTrack()
    t = feed(tr, 0.16, 0.95, 0.0)  # two frames: a clang, a shout in the music
    feed(tr, 1.0, 0.1, t)
    assert not tr.ever()


def test_the_games_streams_are_found_and_never_player_twos_own():
    streams = [
        {"index": 1, "properties": {"application.process.id": "500"}},  # the game
        {"index": 2, "properties": {"application.process.id": "600"}},  # started by it
        {"index": 3, "properties": {"application.process.id": "700"}},  # a browser
        {"index": 4, "properties": {"application.process.id": "999"}},  # Player Two
        {"index": 5, "properties": {"application.process.id": "800"}},  # Steam, same game
        {"index": 6, "properties": {"pipewire.sec.pid": "500",
                                    "application.process.id": "7"}},  # from a container
        {"index": 7, "properties": {}},
    ]
    parents = {600: 500, 500: 10, 700: 10, 800: 20, 999: 10}
    env = {800: {"SteamAppId": "268500"}}
    got = game_streams(streams, 500, "268500", own_pid=999,
                       environ=lambda pid: env.get(pid, {}), parent=parents.get,
                       comm=lambda pid: "")
    assert got == [1, 2, 5, 6]
    # A game in a container reports its own process ids: its program's name still fits.
    boxed = [{"index": 8, "properties": {"application.process.id": "12",
                                         "application.process.binary": "XCom2.exe"}},
             {"index": 9, "properties": {"application.process.id": "13",
                                         "application.process.binary": "wine64-preloader"}}]
    assert game_streams(boxed, 500, "", own_pid=999, environ=lambda pid: {},
                        parent=parents.get, comm=lambda pid: "XCom2.exe") == [8]
    assert game_streams(boxed, 500, "", own_pid=999, environ=lambda pid: {},
                        parent=parents.get, comm=lambda pid: "wine64-preloader") == []
    # Even if the "game" were Player Two's own process, its voice isn't listened to.
    assert game_streams(streams, 999, "", own_pid=999, environ=lambda pid: {},
                        parent=parents.get, comm=lambda pid: "") == []
    assert game_streams(streams, None, "", own_pid=999) == []


def test_window_identity_from_hyprland():
    assert window_identity({"pid": 4321, "class": "steam_app_268500"}) == (4321, "268500")
    assert window_identity({"pid": 77, "class": "retroarch"}) == (77, "")
    assert window_identity(None) == (None, "")


def test_a_stream_that_stops_mid_line_isnt_left_speaking():
    tr = SpeechTrack()
    t = feed(tr, 1.0, 0.9, 0.0)
    assert tr.speaking() and tr.speaking(t + 0.5)
    assert not tr.speaking(t + 2.0)  # no frames since: paused, or its listener is gone
    tr.close()  # the stream ended
    assert not tr.speaking() and tr.segments[0][1] is not None
    assert not tr.heard_between(t + 1.0, t + 5.0)


def test_game_packs_say_whether_lines_are_voiced():
    assert known_voicing("CHRONO TRIGGER") == "none"
    assert known_voicing("Chrono Trigger (SNES)") == "none"
    assert known_voicing("XCOM 2") == ""
    assert voiced_hint("- Battles: ...\n- Voiced: none (text boxes only)\n") == "none"
    assert voiced_hint("Voiced: all") == "all"
    assert voiced_hint("- voiced: yes") == "all"
    assert voiced_hint("- Voiced: some") == ""
    assert voiced_hint("No word on it.") == ""


def test_held_speech_waits_and_starts_its_sentence_over():
    m = Mixer(sink="test")
    m.push_speech(np.linspace(0.1, 0.2, 100, dtype=np.float32), tag="narration")
    out = np.zeros((40, 2), np.float32)
    m._callback(out, 40, None, None)
    assert out[0, 0] > 0 and m.current_tag() == "narration"
    m.hold_speech(True)
    out = np.zeros((40, 2), np.float32)
    m._callback(out, 40, None, None)
    assert not out.any() and not m.idle.is_set()  # silent, and still has it to say
    m.hold_speech(False)
    out = np.zeros((40, 2), np.float32)
    m._callback(out, 40, None, None)
    assert abs(out[0, 0] - 0.1) < 1e-6  # from the sentence's start again
    m.clear_speech()
    assert m.current_tag() is None and not m.held


async def test_a_gate_that_says_no_drops_the_rest_and_reports_only_what_was_said():
    v = Voice.__new__(Voice)
    v.mixer = Mixer(sink="test")
    v._gen, v._lock, v.last_spoken, v.first_audio_ms = 0, asyncio.Lock(), "", []
    v._synth = lambda s: np.zeros(10, np.float32)
    calls = []

    async def gate():
        calls.append(1)
        return len(calls) < 2  # a gap for the first sentence, none for the second

    async def deltas():
        for d in ("A figure steps out of the fog. ", "Another sentence here. ", "More."):
            yield d

    said = await v.say_stream(deltas(), gate=gate, tag="narration")
    assert said.startswith("A figure") and "Another" not in said
    assert list(v.mixer._tags) and set(v.mixer._tags) == {"narration"}
    # Each buffer also carries its words, for the "spoken" event captions are timed by.
    assert " ".join(v.mixer._texts) == said


def test_narration_held_for_the_game_and_restarted_is_captioned_as_heard():
    """A sentence paused for the game's voice and started over is told to on_played twice:
    what was heard before the pause, as cut, and then the whole of it, timed from when it
    started again (not from its first start, seconds before)."""
    m = Mixer(sink="test")
    heard = []
    m.on_played = lambda text, began, ended, cut: heard.append((text, began, ended, cut))
    line = "A soldier walks in."
    m.push_speech(np.full(4800, 0.1, np.float32), line, tag="narration")
    out = np.zeros((480, 2), np.float32)
    for _ in range(3):
        m._callback(out, 480, None, None)
    assert m.current_tag() == "narration" and heard == []
    m.hold_speech(True)  # the game starts to speak
    assert [(t, c) for t, _b, _e, c in heard] == [(line, True)]
    assert heard[0][2] >= heard[0][1]
    m._callback(out, 480, None, None)
    assert len(heard) == 1 and not out.any()  # held: silent, nothing more told
    time.sleep(0.01)
    again = time.time()
    m.hold_speech(False)
    for _ in range(12):
        m._callback(out, 480, None, None)
    assert len(heard) == 2
    text, began, ended, cut = heard[1]
    assert (text, cut) == (line, False) and began >= again and ended >= began
    assert m.current_tag() is None
    # Dropped while held (a stale description): nothing more is told, it wasn't heard.
    m.push_speech(np.full(4800, 0.1, np.float32), "A door opens.", tag="narration")
    m._callback(out, 480, None, None)
    m.hold_speech(True)
    m.clear_speech()
    assert [(t, c) for t, _b, _e, c in heard[2:]] == [("A door opens.", True)]


def test_the_watcher_says_when_a_settled_screen_started_changing():
    w = ChangeWatcher()
    black, white = Image.new("RGB", (320, 180)), Image.new("RGB", (320, 180), "white")
    w.update(Frame(black, 0.0))
    w.accept(Frame(black, 0.1))
    settled = None
    for i in range(2, 12):
        f = Frame(white, i * 0.1)
        if w.update(f):
            settled = f
            break
    assert settled is not None and abs(settled.changed_at - 0.2) < 1e-9


def test_the_narrator_is_told_the_game_is_speaking():
    n = Narrator(SimpleNamespace(s=SimpleNamespace(image_max_side=320)), game="XCOM 2")
    f = Frame(Image.new("RGB", (64, 36)), 0.0)
    text = n.messages(f, voice=VOICE_NOW)[1]["content"][1]["text"]
    assert "voice actors" in text and "You said recently" in text
    assert "voice actors" not in n.messages(f)[1]["content"][1]["text"]


class Ears:
    def __init__(self, speaking=False, heard=False, ever=True, quiet=1e9):
        self._speaking, self.heard, self._ever, self.quiet = speaking, heard, ever, quiet

    def quiet_for(self):
        return 0.0 if self._speaking else self.quiet

    def speaking(self):
        return self._speaking

    def heard_between(self, a, b):
        return self.heard

    def ever(self):
        return self._ever


def ears_session(ears, pack=""):
    s = Session.__new__(Session)
    s.ears = ears
    s.narrator = SimpleNamespace(game="Some Game", recent=deque(maxlen=4))
    s.agent = SimpleNamespace(notes=SimpleNamespace(pack=lambda game: pack),
                              _ct_steam=lambda f: False)
    s.ad_said_at = 0.0
    s.narration_ad = False
    return s


def frame(t=None):
    t = time.monotonic() if t is None else t
    return Frame(Image.new("RGB", (64, 36)), t, changed_at=t - 0.4)


def test_subtitles_the_game_is_saying_arent_read_and_unvoiced_text_is():
    f = frame()
    assert ears_session(Ears(heard=True))._voice_note(f) == VOICE_NOW
    assert ears_session(Ears(heard=False))._voice_note(f) == ""  # a text box no one says
    assert ears_session(Ears(heard=False), "- Voiced: all")._voice_note(f) == VOICED_GAME
    assert ears_session(Ears(heard=True), "- Voiced: none")._voice_note(f) == ""
    assert ears_session(None, "- Voiced: all")._voice_note(f) == VOICED_GAME  # can't hear
    # A voice a few seconds ago: still a voiced scene, so a subtitle isn't read out, whether
    # or not a voice came with this shot. Long after it, text is read again.
    assert ears_session(Ears(quiet=5.0))._voice_note(f) == VOICE_NOW
    assert ears_session(Ears(quiet=30.0))._voice_note(f) == ""
    assert ears_session(Ears(quiet=5.0), "- Voiced: none")._voice_note(f) == ""


def test_narration_waits_while_the_game_talks_and_for_a_voice_to_start():
    f = frame(time.monotonic() - 2.0)
    assert ears_session(Ears(speaking=True))._hold_for_game(f, cut=True)
    # A still screen waits only so long: singing in the music mustn't hide a menu.
    assert not ears_session(Ears(speaking=True))._hold_for_game(
        frame(time.monotonic() - 3.5), cut=False)
    assert not ears_session(Ears())._hold_for_game(f, cut=False)
    fresh = frame()
    assert ears_session(Ears(ever=True))._hold_for_game(fresh, cut=False)  # a voice may come
    assert not ears_session(Ears(ever=False))._hold_for_game(fresh, cut=False)  # never spoke
    # A game whose pack says it never speaks (Chrono Trigger) is never held.
    assert not ears_session(Ears(speaking=True), "Voiced: none")._hold_for_game(f, cut=True)
    assert not ears_session(None)._hold_for_game(f, cut=True)  # can't hear: as before
    # Audio description: one line every so often, unless the scene cuts, and then only a
    # look for a menu or a choice (VOICE_SPACED).
    s = ears_session(Ears(heard=True))
    s.ad_said_at = time.monotonic()
    assert s._hold_for_game(f, cut=False) and not s._hold_for_game(f, cut=True)


def test_maybe_narrate_keeps_the_screen_until_the_game_stops_talking():
    ears = Ears(speaking=True)
    s = ears_session(ears)
    s.pending = (frame(time.monotonic() - 1.0), 0.5)
    s.narrating = s.describing = None
    s.utt, s.transcribing, s.hands_active = None, False, False
    s.error_until = s.soft_until = s.quiet_until = 0.0
    s._reads = lambda f: False
    s.voice = SimpleNamespace(idle=lambda: True)
    started = []
    s._narrate = lambda f, interrupt=False: started.append(f) or asyncio.sleep(0)

    async def go():
        s._maybe_narrate()
        assert not started and s.pending is not None  # held, not dropped
        ears._speaking = False
        s._maybe_narrate()
        await asyncio.sleep(0)
        assert started and s.pending is None

    asyncio.run(go())


def test_a_new_scene_doesnt_cut_off_an_answer_still_being_said():
    s = ears_session(Ears(ever=False))
    s.pending = (frame(time.monotonic() - 1.0), 1.0)  # a whole new scene
    s.utt, s.transcribing, s.hands_active = None, False, False
    s.error_until = s.soft_until = s.quiet_until = 0.0
    s.SCENE_CUT = 0.5
    s._reads = lambda f: False
    s._ears_on = lambda: False
    idle = [False]
    s.voice = SimpleNamespace(idle=lambda: idle[0])
    started = []
    s._narrate = lambda f, interrupt=False: started.append(interrupt) or asyncio.sleep(0)

    async def go():
        s.narrating = s.describing = asyncio.create_task(asyncio.sleep(0))
        await s.narrating  # the model is done, its answer still playing
        s._maybe_narrate()
        assert not started and s.pending is not None
        idle[0] = True  # heard out: the new scene gets its turn
        s._maybe_narrate()
        await asyncio.sleep(0)
        assert started

    asyncio.run(go())


def test_ears_listen_to_the_games_stream_as_it_plays(tmp_path, monkeypatch):
    """The whole path with a recording standing in for the game's stream: found by the
    game's process, read frame by frame in its own thread, speech heard and reported."""
    rate = gameaudio.RATE
    quiet = np.zeros(rate, np.int16)
    talk = (0.3 * 32767 * np.sin(np.arange(rate) * 2 * np.pi * 220 / rate)).astype(np.int16)
    raw = tmp_path / "game.raw"
    raw.write_bytes(np.concatenate([quiet, talk, quiet]).tobytes())
    monkeypatch.setattr(gameaudio, "can_listen", lambda: True)
    monkeypatch.setattr(gameaudio, "sink_inputs", lambda: [
        {"index": 42, "properties": {"application.process.id": "4242"}},
        {"index": 43, "properties": {"application.process.id": "1"}}])
    monkeypatch.setattr(gameaudio, "record_command", lambda index: ["cat", str(raw)])

    class Loud:  # stands in for Silero: anything loud is speech
        def __call__(self, frame):
            x = np.frombuffer(frame, "<i2").astype(np.float32) / 32768
            return 0.9 if np.sqrt(np.mean(x * x)) > 0.1 else 0.0

        def reset(self):
            pass

    heard = []
    clock = iter(i * gameaudio.FRAME_S for i in range(10_000))
    ears = GameEars(lambda: (4242, ""), on_speech=heard.append,
                    detector_factory=Loud, clock=lambda: next(clock))
    assert ears.start()
    try:
        deadline = time.monotonic() + 5
        while not (ears.ever() and not ears.speaking()) and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        ears.stop()
    assert list(ears.streams) == [] and heard  # stopped, and it heard the game speak
    (start, end), = ears.tracks[0].segments
    assert 0.9 < start < 1.3 and 1.9 < end < 2.3


def held_session(ad: bool):
    s = ears_session(Ears(speaking=True))
    s.GAP_MAX_S = 0.2
    s.mixer = Mixer(sink="test")
    s.mixer.push_speech(np.zeros(100, np.float32), tag="narration")
    stopped = []
    s.voice = SimpleNamespace(stop=lambda: (stopped.append(1), s.mixer.clear_speech()))
    s.events = SimpleNamespace(emit=lambda *a, **k: None)
    s.holding, s.holds, s.narration_ad = None, 0, ad
    return s, stopped


def test_description_is_dropped_when_a_voice_starts_never_said_again():
    s, stopped = held_session(ad=True)
    s._game_spoke()
    assert stopped and not s.mixer.held and s.holding is None


def test_description_waits_for_a_real_pause_and_is_dropped_without_one():
    async def go():
        ears = Ears(quiet=0.2)
        s = ears_session(ears)
        s.GAP_MAX_S, s.narration_ad = 0.1, True
        assert not await s._narration_gap()  # only a breath between lines, for too long
        ears.quiet = 2.0
        assert await s._narration_gap()
        ears.quiet = 0.2
        s.narration_ad = False  # a menu doesn't wait for a pause, only for the voice to stop
        assert await s._narration_gap()

    asyncio.run(go())


def test_answers_wait_a_moment_for_a_gap_but_are_always_said():
    async def go():
        ears = Ears(speaking=True)
        s = ears_session(ears)
        s.GAP_MAX_S = 0.1
        t = time.monotonic()
        assert await s._answer_gap()
        assert time.monotonic() - t >= 0.1

    asyncio.run(go())


def test_what_just_happened_and_whos_talking_are_commands_of_their_own():
    for said in ("what just happened", "What happened?", "what did I miss"):
        assert local_action(said) == "just_happened", said
    for said in ("who's talking", "Who is speaking now?", "who said that"):
        assert local_action(said) == "who_talks", said


def test_the_last_few_looks_are_kept_some_seconds_apart():
    s = ears_session(Ears())
    s.looks = deque(maxlen=4)
    for t in (0.0, 1.0, 5.0, 6.0, 10.5, 20.0, 30.0):
        s._keep_look(frame(t))
    assert [f.t for f in s.looks] == [5.0, 10.5, 20.0, 30.0]


def test_held_narration_restarts_a_few_times_at_most_and_goes_on_past_the_limit():
    async def go():
        s, stopped = held_session(ad=False)
        s._game_spoke()
        assert s.mixer.held
        await s.holding  # the game talks past the limit: said over it
        assert not stopped and not s.mixer.held
        s, stopped = held_session(ad=False)
        for _ in range(s.MAX_HOLDS):
            s._game_spoke()
            s.holding.cancel()
            await asyncio.sleep(0)
            s.holding = None
        s._game_spoke()  # one hold too many: short gaps, it goes on rather than restart
        assert not stopped and not s.holding

    asyncio.run(go())


def test_an_answer_that_replaced_held_narration_is_never_cut_off():
    async def go():
        s, stopped = held_session(ad=False)
        s._game_spoke()
        s.mixer.clear_speech()  # the player asked something: its answer takes the lane
        s.mixer.push_speech(np.zeros(100, np.float32))
        await s.holding
        assert not stopped and s.mixer.current_tag() == ""

    asyncio.run(go())
