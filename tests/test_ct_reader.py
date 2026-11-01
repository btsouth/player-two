"""The Chrono Trigger (Steam) battle reader on live frames (2026-11-01, Gargoyle fights
on the Mountain of Woe). Skipped where the frames aren't on disk."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from playertwo import ctsteam

LIVE = Path(os.environ.get("PLAYER_TWO_CT_FRAMES",
                           "~/.cache/player-two/live/steam-2026-11-01")).expanduser()
FRAMES = LIVE / "frames"
pytestmark = pytest.mark.skipif(not FRAMES.is_dir(), reason="no live Chrono Trigger frames")

# Checked by eye: kind, what the hand is on, the actor, and HP / MP for Crono, Frog, Marle.
GT = {
    "134639-00": ("command", "Attack", "Crono", [(780,780,64,64), (1,494,65,65), (430,430,66,66)]),
    "134707-00": ("list", "Ice", None, None),
    "134712-01": ("list", "Aura", None, None),
    "134716-02": ("target", "", None, [(0,780,64,64), (0,494,65,65), (1,430,66,66)]),
    "134832-08": ("command", "Attack", "Marle", [(0,780,64,64), (0,494,65,65), (188,430,65,66)]),
    "134854-00": ("list", "Panacea", None, None),
    "134857-01": ("list", "Lapis", None, None),
    "134901-02": ("list", "Ether", None, None),
    "134905-03": ("list", "Lapis", None, None),
    "134909-04": ("list", "Panacea", None, None),
    "134913-05": ("list", "Hi-Potion", None, None),
    "134916-06": ("list", "Potion", None, None),
    "134923-08": ("list", "Hi-Potion", None, None),
    "134933-11": ("list", "Barrier Sphere", None, None),
    "134936-12": ("list", "Shield Sphere", None, None),
    "135001-18": ("list", "Mid-Potion", None, None),
    "135007-19": ("list", "Athenian Water", None, None),
    "135011-20": ("target", "", None, [(0,780,64,64), (0,494,65,65), (188,430,65,66)]),
    "135019-22": ("target", "", None, [(0,780,64,64), (0,494,65,65), (188,430,65,66)]),
    "135105-01": ("command", "Attack", "Crono", [(50,780,64,64), (0,494,65,65), (188,430,64,66)]),
    "135112-02": ("list", "Athenian Water", None, None),
    "135117-03": ("target", "", None, [(227,780,64,64), (0,494,65,65), (188,430,64,66)]),
    "135122-04": ("command", "Attack", "Marle", [(227,780,64,64), (0,494,65,65), (188,430,64,66)]),
    "135146-08": ("command", "Attack", "Crono", [(227,780,64,64), (50,494,65,65), (188,430,63,66)]),
    "135153-09": ("command", "Attack", "Frog", [(227,780,64,64), (241,494,65,65), (188,430,63,66)]),
    "135158-10": ("command", "Attack", "Marle", [(227,780,64,64), (241,494,65,65), (188,430,63,66)]),
    "140154-00": ("list", "Lightning II", None, None),
    "140201-01": ("list", "Cyclone", None, None),
    "140210-03": ("command", "Attack", "Marle", [(811,811,64,66), (481,494,55,65), (324,430,63,66)]),
    "140238-01": ("target", "", None, [(766,811,62,66), (481,494,55,65), (209,430,63,66)]),
    "140244-02": ("command", "Attack", "Crono", [(766,811,62,66), (481,494,55,65), (209,430,62,66)]),
    "140314-00": ("target", "", None, [(1,811,62,66), (481,494,55,65), (1,430,62,66)]),
    "164519-00": ("command", "Attack", "Frog", [(780,780,64,64), (494,494,65,65), (1,430,66,66)]),
    "164526-01": ("list", "Water II", None, None),
    "164538-03": ("command", "Attack", "Marle", [(780,780,64,64), (1,494,63,65), (1,430,66,66)]),
    "164545-04": ("list", "Ice", None, None),
    "164547-05": ("list", "Cure", None, None),
    "164619-12": ("command", "Attack", "Crono", [(780,780,64,64), (494,494,63,65), (136,430,64,66)]),
    "164625-13": ("command", "Attack", "Frog", [(704,780,64,64), (494,494,63,65), (136,430,64,66)]),
    "164630-14": ("list", "Heal", None, None),
    "164645-17": ("command", "Attack", "Marle", [(780,780,64,64), (494,494,61,65), (257,430,64,66)]),
    "164658-20": ("command", "Attack", "Crono", [(1,780,64,64), (494,494,61,65), (257,430,64,66)]),
    "164702-21": ("command", "Attack", "Frog", [(1,780,64,64), (494,494,61,65), (257,430,64,66)]),
    "164726-26": ("command", "Tech", "Frog", [(0,780,64,64), (494,494,61,65), (257,430,64,66)]),
    "164730-27": ("command", "Item", "Frog", [(0,780,64,64), (494,494,61,65), (257,430,64,66)]),
    "164736-29": ("list", "Athenian Water", None, None),
    "164843-00": ("command", "Attack", "Marle", [(0,780,64,64), (494,494,61,65), (1,430,64,66)]),
    "164912-06": ("list", "Hi-Ether", None, None),
    "164915-07": ("list", "Athenian Water", None, None),
    "164920-09": ("list", "Panacea", None, None),
    "165039-19": ("command", "Item", "Marle", [(50,780,64,64), (494,494,61,65), (1,430,64,66)]),
}
NAMES = ("Crono", "Frog", "Marle")


def load(name: str) -> np.ndarray:
    return cv2.imread(str(FRAMES / f"{name}.jpg"))


def test_kind_focus_actor_and_party_numbers():
    bad, times = [], []
    for name, (kind, focus, actor, party) in GT.items():
        img = load(name)
        t = time.perf_counter()
        st = ctsteam.read(img)
        times.append(time.perf_counter() - t)
        if st.kind != kind:
            bad.append((name, "kind", st.kind))
        if focus and (st.command if kind == "command" else st.entry) != focus:
            bad.append((name, "focus", st.command or st.entry))
        if actor and st.actor != actor:
            bad.append((name, "actor", st.actor))
        if party:
            got = {m.name: (m.hp, m.hp_max, m.mp, m.mp_max) for m in st.party}
            for who, want in zip(NAMES, party, strict=True):
                if got.get(who) != want:
                    bad.append((name, who, got.get(who)))
    assert not bad, bad
    assert np.median(times) < 0.4  # about 0.1 s here


def test_ready_columns_and_where_the_actor_stands():
    st = ctsteam.read(load("134639-00"))
    assert st.columns == [0, 1, 2] and st.actor_col == 0 and st.ready() == list(NAMES)
    assert st.bracket is not None and abs(st.bracket[0] - 1884) < 30
    st = ctsteam.read(load("135153-09"))  # Frog behind the windows: no bracket, still read
    assert (st.actor, st.columns, st.actor_col) == ("Frog", [1, 2], 1)


def test_critical_names_under_the_pause_and_whose_turn_it_is():
    # Paused with two fighters at 1 HP: their grey lines are drawn at about 83 and used to
    # misread, Crono dropped out of the party and the slots below him moved up a place.
    for name, actor, hps in (("020449-00", "Frog", [(1, 811), (101, 494), (1, 430)]),
                             ("015255-00", "Marle", [(780, 780), (1, 494), (1, 430)]),
                             ("180328-00", "Frog", [(1, 780), (1, 494), (430, 430)])):
        st = ctsteam.read(load(name))
        assert st.paused and st.actor == actor, name
        assert [(m.name, m.hp, m.hp_max) for m in st.party] == [
            (who, hp, most) for who, (hp, most) in zip(NAMES, hps, strict=True)], name
        assert st.slots == list(NAMES)
    st = ctsteam.read(load("180121-00"))  # Crono's own turn, at 1 HP
    assert st.actor == "Crono" and st.member("Crono").hp == 1


def test_an_unread_row_keeps_its_slot():
    st = ctsteam.State(kind="command", columns=[1, 2], actor_col=1, slots=[None, "Frog", "Marle"],
                       party=[ctsteam.Member("Frog", 101, 494, 60, 65, "yellow"),
                              ctsteam.Member("Marle", 1, 430, 62, 66, "grey")])
    assert st.who_in(0) is None and st.who_in(1) == "Frog" and st.ready() == ["Frog", "Marle"]


def test_list_rows_and_greyed_entries():
    st = ctsteam.read(load("134933-11"))
    rows = [r for r in st.entries if r]
    assert rows == [["Lapis", "Hi-Ether"], ["Ether", "Elixir"],
                    ["Barrier Sphere", "Shield Sphere"]]
    assert st.usable is True


@pytest.mark.skipif(not (LIVE / "btrace" / "004751-1494.jpg").exists(),
                    reason="no Giga Gaia trace frame")
def test_the_hand_on_a_used_up_items_empty_slot():
    st = ctsteam.read(cv2.imread(str(LIVE / "btrace" / "004751-1494.jpg")))
    assert st.kind == "list"
    rows = [r for r in st.entries if r]
    assert rows == [["Potion", "Mid-Potion"], ["", "Mid-Ether"], ["Panacea", "Athenian Water"]]
    assert st.entry == "" and st.entry_rc == (2, 0)


@pytest.mark.skipif(not (LIVE / "btrace" / "012116-2913.jpg").exists(),
                    reason="no empty tech list frame")
def test_an_empty_tech_list_is_an_open_list_with_nothing_in_it():
    st = ctsteam.read(cv2.imread(str(LIVE / "btrace" / "012116-2913.jpg")))
    assert (st.kind, st.entries, st.entry, st.entry_rc) == ("list", [], "", None)


def test_target_glove_and_the_enemy_name_in_the_bar():
    st = ctsteam.read(load("140314-00"))
    assert st.kind == "target" and st.top == "Gargoyle" and len(st.gloves) == 1
    st = ctsteam.read(load("164634-15"))  # a heal for everyone: a glove on each
    assert len(st.gloves) == 3


@pytest.mark.skipif(not (LIVE / "p1.png").exists(), reason="no pause shots")
def test_the_start_pause_and_nothing_else_reads_as_paused():
    for shot in ("p1", "p2", "p3"):
        assert ctsteam.read(cv2.imread(str(LIVE / f"{shot}.png"))).paused, shot
    assert not ctsteam.read(cv2.imread(str(LIVE / "p4.png"))).paused
    for name in [*list(GT)[:20], "135528-23", "170857-07"]:  # battles and field menus
        if (FRAMES / f"{name}.jpg").exists():
            assert not ctsteam.read(load(name)).paused, name


def test_frame_shapes():
    assert ctsteam.fits(np.zeros((1440, 2560, 3), np.uint8))
    assert ctsteam.fits(np.zeros((540, 960, 3), np.uint8))
    assert not ctsteam.fits(np.zeros((224, 256, 3), np.uint8))


ITEMS = ["Potion", "Mid-Potion", "Hi-Potion", "Mid-Ether", "Panacea", "Athenian Water",
         "Lapis", "Hi-Ether", "Ether", "Elixir", "Barrier Sphere", "Shield Sphere"]


def step(i: int, button: str, n: int) -> int:
    """The two-column list: down on the last row's left entry moves right."""
    if button == "down":
        if i + 2 < n:
            return i + 2
        return i + 1 if i % 2 == 0 and i + 1 < n else i
    if button == "up":
        return i - 2 if i >= 2 else i
    if button == "right":
        return i + 1 if i % 2 == 0 and i + 1 < n else i
    if button == "left":
        return i - 1 if i % 2 == 1 else i
    return i


@pytest.mark.skipif(not (LIVE / "events.jsonl").exists(), reason="no event log")
def test_list_moves_replayed_from_the_live_run():
    """The model's own d-pad presses in the item list, frame before and after each: the
    two-column model predicts where the hand lands (the executor's planning model)."""
    events = [json.loads(line) for line in (LIVE / "events.jsonl").open()]
    ok = total = 0
    for e in events:
        if e.get("kind") != "agent" or e["request"] not in (
                "Use a revive item on Crono", "Use Athenian Water on Crono"):
            continue
        t0 = e["t"] - e["seconds"]
        stamp = lambda t: time.strftime("%H%M%S", time.localtime(t))
        frames = sorted(f for f in FRAMES.glob("*.jpg")
                        if stamp(t0 - 1) <= f.name[:6] <= stamp(e["t"]))
        states = [ctsteam.read(cv2.imread(str(f))) for f in frames]
        for k, s in enumerate(e["steps"]):
            if k == 0 or k >= len(states):
                continue
            a, b = states[k - 1], states[k]
            buttons = (s.get("args") or {}).get("buttons") or []
            if a.kind != "list" or b.kind != "list" or a.entry not in ITEMS or not buttons \
                    or any(x not in ("up", "down", "left", "right") for x in buttons):
                continue
            i = ITEMS.index(a.entry)
            for x in buttons:
                i = step(i, x, len(ITEMS))
            ok += ITEMS[i] == b.entry
            total += 1
    if total == 0:
        pytest.skip("the frames of those runs aren't here (times are local)")
    assert ok == total and total >= 20, (ok, total)


def test_the_actor_name_with_forest_leaves_run_into_it():
    """Guardia Forest, live 2026-11-01: yellow leaves between the windows ran into Marle's
    yellow name ("t -.2u t C' -i'-i' Marle"), so no one was choosing and "win this fight"
    gave up after 0 commands."""
    img = load("forest-041307-04")
    if img is None:
        pytest.skip("no forest frame")
    st = ctsteam.read(img)
    assert (st.kind, st.actor, st.actor_col, st.columns) == ("command", "Marle", 1, [0, 1])
    assert [(m.name, m.hp, m.hp_max, m.mp) for m in st.party] == [("Crono", 22, 70, 8),
                                                                  ("Marle", 40, 65, 12)]
