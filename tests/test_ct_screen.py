"""Chrono Trigger (Steam) screens read for narration without the model (ctscreen)."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import pytest

from playertwo import ctscreen
from playertwo.ctscreen import Look, Reader

LIVE = Path(os.environ.get("PLAYER_TWO_CT_LIVE", "~/.cache/player-two/live")).expanduser()
PARTY = ["Crono", "Frog", "Marle", "Ayla", "Lucca", "Robo"]


def look(kind="menu", typing=(), **parts):
    lk = Look(kind, parts=dict(parts))
    lk.texts = {p for p in parts if p in ("top", "desc", "question", "text", "title")}
    lk.typing = set(typing)
    return lk


def test_the_cursor_is_said_at_once_and_only_when_it_moves():
    r = Reader()
    assert r.update(look(focus="Equipment"), now=0) == "Equipment."
    assert r.update(look(focus="Equipment"), now=0.2) == ""
    assert r.update(look(focus="Inventory"), now=0.4) == "Inventory."


def test_text_still_typing_out_is_never_read():
    r = Reader()
    lk = look(typing=("desc",), focus="Load Game", desc="Please select a fi")
    assert r.update(lk, now=0) == "Load Game."
    # Still typing: it changed since the last look.
    assert r.update(look(typing=("desc",), focus="Load Game", desc="Please select a file to l"),
                    now=0.2) == ""
    whole = look(typing=("desc",), focus="Load Game", desc="Please select a file to load.")
    assert r.update(whole, now=0.4) == ""  # once: it may still be typing
    assert r.update(whole, now=0.6) == "Please select a file to load."
    assert r.update(whole, now=0.8) == ""  # never twice


def test_text_without_an_end_is_read_once_it_has_stood_still():
    r = Reader()
    lk = look(typing=("desc",), focus="Techs", desc="Select a tech")
    assert r.update(lk, now=0) == "Techs."
    assert r.update(lk, now=0.2) == ""
    assert r.update(lk, now=0.4) == "Select a tech."


def test_the_same_text_isnt_read_twice_on_coming_back_but_new_text_is():
    r = Reader()
    box = look("text", typing=("text",), text="Marle: Hey, wait up!")
    r.update(box, now=0)
    assert r.update(box, now=0.3) == "Marle: Hey, wait up!"
    r.update(look("none"), now=1)
    r.update(box, now=2)
    assert r.update(box, now=2.3) == ""  # the same box back again within a while
    nxt = look("text", typing=("text",), text="Marle: Let's go to the fair.")
    r.update(nxt, now=3)
    assert r.update(nxt, now=3.3) == "Marle: Let's go to the fair."


def test_a_battle_message_is_said_first_and_danger_on_change():
    r = Reader()
    lk = look("battle", top="Combo Attack: Reduces HP to 1", turn="Crono's turn",
              focus="Attack", danger="Frog 1 HP")
    lk.order = ("pause", "top", "turn", "focus", "danger")
    assert r.update(lk, now=0) == ("Combo Attack: Reduces HP to 1. Crono's turn. Attack. "
                                   "Frog 1 HP.")
    lk2 = look("battle", turn="Marle's turn", focus="Attack", danger="Frog 1 HP")
    assert r.update(lk2, now=1) == "Marle's turn."
    lk3 = look("battle", turn="Marle's turn", focus="Attack", danger="Frog is down")
    assert r.update(lk3, now=2) == "Frog is down."


def test_reopening_the_menu_says_where_the_cursor_is_again():
    r = Reader()
    r.update(look(focus="Equipment"), now=0)
    r.update(look("none"), now=1)
    assert r.update(look(focus="Equipment"), now=2) == "Equipment."


def test_the_party_order_is_learned_from_what_screens_show():
    r = Reader()
    r.learn_party(["Crono", "Frog", "Marle"])
    r.learn_party(["Marle", "Ayla", "Lucca"])
    r.learn_party(["Lucca", "Robo"])
    assert r.party == ["Crono", "Frog", "Marle", "Ayla", "Lucca", "Robo"]


def test_plausible_text_and_texture():
    from playertwo.glyphread import Run

    def ok(t, worst=10.0):
        return ctscreen.plausible(Run(t, 0, 0, 0, "white", worst))

    assert ok("Restore a small amount of HP to an ally.")
    assert ok("17662 G") and ok("Yes") and ok("No")
    assert not ok("'' ' '''' rrr") and not ok("t2-b2") and not ok("--,- P")
    assert not ok("Restore a small amount", worst=45.0)


FRAMES = LIVE / "steam-2026-11-01" / "frames"
needs_frames = pytest.mark.skipif(not FRAMES.is_dir(), reason="no live Chrono Trigger frames")


def first_line(path: Path) -> tuple[str, str]:
    lk = ctscreen.look(cv2.imread(str(path)), PARTY)
    return lk.kind, Reader().update(lk, first=True)


@needs_frames
@pytest.mark.parametrize("name,kind,line", [
    ("134639-00", "battle", "Combo Attack: Reduces HP to 1. Crono's turn. Attack. Frog 1 HP."),
    ("135007-19", "battle", "Athenian Water. Revives a KO'd ally."),
    ("140314-00", "battle", "Gargoyle. Crono 1 HP, Marle 1 HP."),
    ("180436-01", "battle", "Paused. Crono 1 HP, Frog 1 HP."),
    ("135431-01", "menu", "Menu. Techs."),
    ("135629-31", "menu", "Menu. Techs. View or use known techs."),
    ("135445-06", "menu", "Shelter, 21. Fully restores HP/MP at a save point."),
    ("140413-03", "menu", "Frog."),  # the highlighted block, counted from its neighbours
    ("140421-06", "menu", "Crono, Frog and Marle. Restore moderate HP to all allies."),
])
def test_live_frames_read_as_a_screen_reader_would(name, kind, line):
    assert first_line(FRAMES / f"{name}.jpg") == (kind, line)


@needs_frames
def test_half_typed_menu_descriptions_are_left_unsaid():
    for name in ("135428-00", "140409-01", "175403-01"):  # "Change characte", ...
        kind, line = first_line(FRAMES / f"{name}.jpg")
        assert kind == "menu" and line.count(".") == 2, line  # "Menu. Equipment." only


@needs_frames
def test_choices_say_the_question_then_the_cursor():
    assert first_line(LIVE / "steam-2026-10-31" / "s3.png") == (
        "choice", "Resume bookmarked game? No selected, or Yes.")
    assert first_line(LIVE / "steam-2026-11-01" / "q4.png") == (
        "choice", "Exit? No selected, or Yes.")


@needs_frames
def test_fields_and_world_maps_are_left_to_the_model():
    names = ["135215-15", "135853-04", "172345-00", "185131-09"]
    paths = [FRAMES / f"{n}.jpg" for n in names]
    paths += [LIVE / "steam-2026-10-31" / f"{n}.png" for n in ("a4", "a6", "p1", "a3", "p2")]
    for p in paths:
        assert ctscreen.look(cv2.imread(str(p)), PARTY).kind == "none", p.name


async def test_the_session_says_what_the_reader_reads_and_leaves_the_rest_to_the_model(
        monkeypatch):
    """Owned screens are said by the reader as they change, with no model narration;
    anything else goes to the model as before."""
    import asyncio
    from collections import deque
    from types import SimpleNamespace

    from PIL import Image

    from playertwo.app import Session
    from playertwo.screen import Frame

    s = Session.__new__(Session)
    s.reader, s.reading, s.read_t, s.read_owned = Reader(), None, 0.0, False
    s.read_thumb, s.read_prime, s.read_last_parts = None, False, set()
    s.read_model = None
    s.paused, s.agent_task, s.narrating = False, None, None
    s.watcher = SimpleNamespace(settle_s=0.3)
    s.utt, s.transcribing, s.hands_active = None, False, False
    s.narrator = SimpleNamespace(recent=deque(maxlen=4))
    said, events = [], []
    s.events = SimpleNamespace(emit=lambda kind, **k: events.append((kind, k)))

    async def say(text, interrupt=True):
        said.append((text, interrupt))

    s.voice = SimpleNamespace(say=say)
    s._tasks = set()
    s.agent = SimpleNamespace(_ct_steam=lambda f: True)
    looks = iter([look(focus="Equipment"), look(focus="Inventory"), look("none")])
    monkeypatch.setattr(ctscreen, "look", lambda image, party=(): next(looks))
    img = Image.new("RGB", (1920, 1080))  # big enough to read the font from
    for t in (1.0, 2.0):
        await s._read(Frame(img, t))
        await asyncio.sleep(0)
    assert said == [("Equipment.", True), ("Inventory.", True)]
    assert s.read_owned and s.read_t == 2.0
    # A settled menu screen waits for the reader's verdict, then gets no model line.
    s.pending, s.error_until = (Frame(img, 2.0), 0.0), 0.0
    s._maybe_narrate()
    assert s.pending is None
    await s._read(Frame(img, 3.0))  # the field again: the model's turn
    assert not s.read_owned
    s.pending, s.narrating = (Frame(img, 4.0), 0.0), None
    s.listening = lambda: False
    s._maybe_narrate()  # not seen by the reader yet: wait
    assert s.pending is not None and s.narrating is None


def test_only_questions_are_answered_from_the_screen():
    assert ctscreen.asks("Player Two, what's my HP?") and ctscreen.asks("how's everyone doing")
    assert ctscreen.asks("what does that say") and ctscreen.asks("read that again")
    assert ctscreen.asks("who's in my party") and ctscreen.asks("what items do I have")
    assert not ctscreen.asks("heal Frog's HP") and not ctscreen.asks("use a potion")
    assert not ctscreen.asks("what is the HP of the boss")  # not shown: the agent looks


@needs_frames
def test_questions_the_screen_answers_at_once():
    def ask(path, q):
        return ctscreen.answer(q, ctscreen.look(cv2.imread(str(path)), PARTY), PARTY)

    battle = FRAMES / "134639-00.jpg"
    assert ask(battle, "what's my HP") == ("Crono 780 of 780 HP, Frog 1 of 494 HP, "
                                           "Marle 430 of 430 HP.")
    assert ask(battle, "what's Frog's HP?") == "Frog 1 of 494 HP."
    assert ask(battle, "who's in my party") == "Crono, Frog and Marle."
    assert ask(battle, "what does it say") == "Combo Attack: Reduces HP to 1."
    assert ask(FRAMES / "135428-00.jpg", "how's everyone doing") == (
        "Crono 227 of 811 HP, Frog 241 of 494 HP, Marle 188 of 430 HP.")
    assert ask(FRAMES / "140314-00.jpg", "what are we fighting") == "The target now is Gargoyle."
    assert ask(FRAMES / "134854-00.jpg", "what items do we have").startswith("On screen: Potion")
    # The field shows none of it: the agent answers (and may open the menu).
    assert ask(FRAMES / "185021-00.jpg", "what's my HP") is None


def test_a_text_box_that_reads_badly_goes_to_the_model_once_it_is_whole():
    r = Reader()
    box = look("text", typing=("text",), text="Lucca: Hey, Cr0no! Over here")
    box.unsure = True
    assert r.update(box, now=0) == ""
    assert not r.model_read
    r.update(box, now=0.3)
    r.update(box, now=0.6)  # three looks alike: whole, though it has no end mark
    assert r.model_read  # the session hands this frame to the model narrator
    r.model_read = False
    r.update(box, now=0.9)
    assert not r.model_read  # once


def test_misread_words_are_caught():
    assert ctscreen.misread("Com6o Attackl Red0ces HP to 1") and ctscreen.misread("SlQp 5ash")
    assert not ctscreen.misread("Restores 200 HP to all party members.")
    assert not ctscreen.misread("Reduces physical damage by 1/3.")
    assert not ctscreen.misread("Earned 648 EXP.") and not ctscreen.misread("Lightning II")


def test_the_same_fighter_gets_each_new_turn_said_but_not_a_flicker():
    r = Reader()

    def turn():
        return look("battle", turn="Crono's turn", focus="Attack")

    assert r.update(turn(), now=0) == "Crono's turn. Attack."
    r.update(look("battle"), now=0.3)  # one look without the window: a flicker
    assert r.update(turn(), now=0.6) == ""
    r.update(look("battle"), now=1.0)
    r.update(look("battle"), now=1.3)  # gone for two looks: the turn was taken
    assert r.update(turn(), now=3.0) == "Crono's turn. Attack."


async def test_a_box_the_reader_hands_to_the_model_isnt_cut_off_by_the_next_look(monkeypatch):
    import asyncio
    from collections import deque
    from types import SimpleNamespace

    from PIL import Image

    from playertwo.app import Session
    from playertwo.screen import Frame

    s = Session.__new__(Session)
    s.reader, s.reading, s.read_t, s.read_owned = Reader(), None, 0.0, False
    s.read_thumb, s.read_prime, s.read_last_parts, s.read_model = None, False, set(), None
    s.paused, s.agent_task, s.narrating = False, None, None
    s.utt, s.transcribing, s.hands_active = None, False, False
    s.narrator = SimpleNamespace(recent=deque(maxlen=4))
    s.events = SimpleNamespace(emit=lambda kind, **k: None)
    s._tasks = set()
    started = []

    def start(frame, describe=False):
        started.append(frame.t)
        s.narrating = asyncio.get_running_loop().create_task(asyncio.sleep(5))

    s._start_narration = start
    box = look("text", typing=("text",), text="Lucca: Hey, Cr0no! Over here")
    box.unsure = True
    monkeypatch.setattr(ctscreen, "look", lambda image, party=(): box)
    img = Image.new("RGB", (1920, 1080))
    for t in (1.0, 1.3, 1.6, 1.9):
        await s._read(Frame(img, t))
    assert started == [1.6]  # once, when the box had stood still
    assert not s.narrating.cancelled() and not s.narrating.done()
    s.narrating.cancel()
