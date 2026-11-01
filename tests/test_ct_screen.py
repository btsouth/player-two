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
    s.paused, s.agent_task, s.narrating = False, None, None
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
    img = Image.new("RGB", (64, 36))
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
