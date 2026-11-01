"""Finding the person the player means from their own words (people.resolve), asking when
two fit alike, and "who's here?" (people.around), on a made-up square and on the room
survey when it's there."""

import json
import os
from pathlib import Path

import pytest

from playertwo import people
from playertwo.people import Who, answer, directions, parse, resolve, wanted

# A made-up market square, 16-pixel tiles: who stands where, what they're called and what
# they said when talked to (none of it the game's).
SQUARE = [
    Who("Barnaby the sword seller in the middle (in here)",
        "Barnaby the sword seller in the middle", 1, (440.0, 464.0),
        aka=("Barnaby", "the bearded old man at the sword stall"),
        said="Barnaby: Fine steel, fresh from the forge!"),
    Who("the old man at the lower stalls (in here)", "the old man at the lower stalls", 1,
        (376.0, 624.0), said="Old man: The harvest moon is out tonight."),
    Who("the item seller at the purple tent (in here)", "the item seller at the purple tent",
        1, (240.0, 596.0), aka=("the man who sells potions",)),
    Who("the armor seller at the blue tent on the left (in here)",
        "the armor seller at the blue tent on the left", 1, (304.0, 500.0)),
    Who("the girl in the green dress (in here)", "the girl in the green dress", 1,
        (472.0, 480.0)),
    Who("the guard at the big tents (in here)", "the guard at the big tents", 1,
        (136.0, 608.0)),
    Who("the man walking by the west stalls (in here)", "the man walking by the west stalls",
        1, (114.0, 464.0), moves=True, said="Mind the puddles, the rain came early."),
    Who("the man by the bell game (in here)", "the man by the bell game", 1, (56.0, 576.0)),
    Who("the bell (in here)", "the bell", 1, (88.0, 560.0), kind="thing"),
    Who("the cat (in here)", "the cat", 1, (344.0, 530.0), kind="animal", moves=True),
    Who("the shopkeeper at the north stall (the north fair)",
        "the shopkeeper at the north stall", 2, (424.0, 96.0), here=False,
        where="the north fair"),
]
MARKS = [({people.stem(t) for t in people.tokens(w.name)}, w.at, w.name) for w in SQUARE
         if w.here]
GATE = (392.0, 700.0)


def pick(said, me=GATE):
    return resolve(said, SQUARE, me, MARKS, mid=392.0)


def name(p):
    return next(w.name for w in SQUARE if w.key == p.key) if p.sure else None


def asks(p):
    return {o.name for o in p.options} if not p.sure else set()


def test_the_players_words_find_the_person():
    sword = "Barnaby the sword seller in the middle"
    assert name(pick("talk to Barnaby")) == sword  # a name they go by
    assert name(pick("the guy selling swords")) == sword  # what he is and sells
    assert name(pick("the swordsmith")) == sword
    assert name(pick("the weapon shop")) == sword
    assert name(pick("talk to the merchant with the blades")) == sword
    assert name(pick("the old man selling swords")) == sword
    assert name(pick("talk to the shopkeeper in the middle")) == sword  # the place's middle
    assert name(pick("the girl")) == "the girl in the green dress"
    assert name(pick("the lady")) == "the girl in the green dress"
    assert name(pick("talk to the woman next to the swords")) == "the girl in the green dress"
    assert name(pick("that soldier")) == "the guard at the big tents"
    assert name(pick("talk to the knight")) == "the guard at the big tents"
    assert name(pick("the potion seller")) == "the item seller at the purple tent"
    assert name(pick("the armor shop")) == "the armor seller at the blue tent on the left"
    assert name(pick("the man by the bell")) == "the man by the bell game"  # by something
    assert name(pick("the bell")) == "the bell"  # its name word for word
    assert name(pick("the cat")) == "the cat"


def test_what_they_said_tells_them_apart():
    # Words from their line count for more than "guy" does.
    assert name(pick("the guy who talked about the rain")) == \
        "the man walking by the west stalls"
    assert name(pick("the old man who talked about the harvest")) == \
        "the old man at the lower stalls"
    # The name a line is shown under is a name they go by.
    assert "barnaby" in SQUARE[0].proper and "old" in SQUARE[1].groups


def test_where_they_stand():
    assert name(pick("talk to the old man at the lower stalls")) == \
        "the old man at the lower stalls"
    assert name(pick("the old man at the bottom")) == "the old man at the lower stalls"
    # The furthest that way of those that fit, by a clear step: "on the left" for the
    # sellers is the item seller's spot, but the armor seller is named "on the left":
    # asked, not guessed.
    assert asks(pick("the shopkeeper on the left")) == {
        "the item seller at the purple tent", "the armor seller at the blue tent on the left"}


def test_two_or_three_alike_are_asked_about():
    p = pick("the old man by the stalls")
    assert not p.sure and asks(p) == {"Barnaby the sword seller in the middle",
                                      "the old man at the lower stalls"}
    assert p.ask.startswith("Do you mean ") and p.ask.endswith("?") and " or " in p.ask
    # In here: who's next door isn't asked about when someone in here fits.
    assert asks(pick("talk to the shopkeeper")) == {
        "Barnaby the sword seller in the middle", "the item seller at the purple tent",
        "the armor seller at the blue tent on the left"}
    # The answer picks among them, by its words or which one.
    assert answer("the sword seller", p.options) == SQUARE[0].key
    assert answer("the second one", p.options) == p.options[1].key
    assert answer("Barnaby", p.options) == SQUARE[0].key
    assert answer("hmm", p.options) is None
    assert answer("that one", p.options) is None  # not "the first"


def test_no_one_fits():
    for said in ("the purple monster", "talk to Lucca", "talk to the innkeeper"):
        p = pick(said)
        assert not p.sure and not p.options, (said, p)


def test_next_door_by_its_name():
    p = pick("talk to the shopkeeper at the north stall")
    assert name(p) == "the shopkeeper at the north stall"
    assert SQUARE[-1].said_as == "the shopkeeper at the north stall, in the north fair"


def test_nearest_of_those_that_fit():
    two = [Who("a (in here)", "the guard on the left", 1, (100.0, 100.0)),
           Who("b (in here)", "the guard on the right", 1, (300.0, 100.0))]
    assert resolve("that guard", two, (110.0, 120.0)).key == "a (in here)"
    assert resolve("that guard", two, (290.0, 120.0)).key == "b (in here)"
    assert resolve("the guard on the right", two, (110.0, 120.0)).key == "b (in here)"


def test_parsing():
    a = parse("talk to the guy selling swords")
    assert a.who == ["guy", "seller"] and a.what == ["swords"] and a.head == "guy" and a.talk
    a = parse("the old man by the stalls")
    assert a.who == ["old", "man"] and a.near == ["stalls"] and a.head == "man"
    a = parse("the shopkeeper on the left")
    assert a.who == ["shopkeeper"] and a.ways == ["left"] and not a.near
    a = parse("that soldier")
    assert a.nearest and a.who == ["soldier"]
    assert wanted("please talk to the old man and buy a potion") == "the old man"
    assert wanted("take me to the fountain") == "the fountain"
    assert wanted("I need to buy some potions") == ""
    assert wanted("talk to the guard, then go upstairs") == "the guard"
    assert wanted("take me to the save point and save") == "the save point"
    assert wanted("Melchior") == ""


def test_directions_on_the_d_pad():
    assert directions(0, -64) == "4 steps up"
    assert directions(-48, -48) == "4 steps up and to your left"
    assert directions(80, 4) == "5 steps to your right"
    assert directions(8, 4) == "right next to you"


# -- the room survey --------------------------------------------------------------------

def _square_rooms(tmp_path):
    from test_rooms import make, with_people

    from playertwo.rooms import Rooms

    walk = [[r, c] for r in range(1, 13) for c in range(1, 15)]
    make(tmp_path, 3, (256, 224), walk,
         [{"from": [12, 7], "dir": [0, 1], "to": 496, "at": [100, 100]}], 4)
    with_people(tmp_path, 3, [
        {"obj": 8, "at": [56.0, 40.0], "name": "the seller on the left", "kind": "person",
         "aka": ["Barnaby"], "said": "Barnaby: Fine steel!"},
        {"obj": 9, "at": [200.0, 40.0], "name": "the old man on the right", "kind": "person"},
        {"obj": 10, "at": [120.0, 120.0], "name": "the well", "kind": "thing"},
        {"obj": 11, "at": [40.0, 180.0], "name": "the save point", "kind": "save point"},
        {"obj": 12, "at": [80.0, 80.0], "name": "the wall", "kind": "none"},
    ])
    (tmp_path / "names.json").write_text(json.dumps({
        "names": {"3": "the square", "496": "outside (the world map)"}, "worlds": [496]}))
    return Rooms(tmp_path)


def test_whos_here_from_the_survey(tmp_path):
    from playertwo.rooms import Here

    rooms = _square_rooms(tmp_path)
    here = Here(3, (0, 0), 120.0, 173.0, 1.0)
    text, n = people.around(rooms, here)
    assert n == 2  # people only (not the well, the save point or the wall)
    assert text.startswith("In the square. Nearest: ")
    assert "the old man on the right, 10 steps up and to your right" in text
    assert "the seller on the left, 9 steps up and to your left" in text
    assert "Things: the well, 3 steps up; the save point, 5 steps to your left." in text
    assert "Ways out: the way out, 2 steps down." in text
    assert "wall" not in text
    more, _ = people.around(rooms, here, 2, 6)
    assert more == "That's everyone here."
    # The survey's aka and said are known to the resolver.
    choices, targets, _ = rooms.options(here)
    cands = people.candidates(rooms, here, {k: choices[k] for k in targets}, targets)
    seller = next(c for c in cands if c.name == "the seller on the left")
    assert "barnaby" in seller.proper
    assert resolve("talk to Barnaby", cands, (here.x, here.y)).key == seller.key
    save = next(c for c in cands if c.name == "the save point")
    assert save.kind == "save point" and targets[save.key].kind == "save point"


# The fair and Truce on the real survey (scripts/people_bench.py): with the people's other
# names and lines (aka, said) staged for it.
BENCH = Path(os.environ.get("P2_PEOPLE_ROOMS") or
             Path.home() / ".cache/player-two/day-2026-11-01/rooms")


def _has_aka(folder: Path) -> bool:
    try:
        d = json.loads((folder / "loc-005.json").read_text())
    except (OSError, ValueError):
        return False
    return any(p.get("aka") for p in d.get("people", []))


@pytest.mark.skipif(not _has_aka(BENCH), reason=f"no survey with aka in {BENCH}")
def test_people_bench_on_the_survey():
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from people_bench import run

    ok, n, misses = run(BENCH)
    assert ok == n, "\n".join(misses)
