"""What a place is like: asked for ("what does it look like?"), and said once, the first
time the player walks into a place."""

import asyncio
import json
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
from PIL import Image

from playertwo.app import Session, local_action, palette, same_look
from playertwo.narrator import Narrator
from playertwo.prompts import ATMOSPHERE, PLACE_FEEL, PLACE_SAID
from playertwo.screen import Frame

LEENE = "A bright, festive fairground: striped stalls, round tents and a fountain."
INN = "A warm, lamplit wooden inn: a long counter, a laid dining table, stairs."


def test_asking_what_it_looks_like_is_a_command_of_its_own():
    for said in ("describe the atmosphere", "What does it look like?", "what's it like here",
                 "paint me a picture", "what's the mood", "describe the scene",
                 "hey player two, what does this place look like", "set the scene",
                 "could you paint me a picture please", "what's the vibe in here",
                 "tell me what it looks like"):
        assert local_action(said) == "atmosphere", said
    # The tour stays the tour, and questions about one thing are the agent's.
    assert local_action("describe") == "describe"
    assert local_action("describe the screen") == "describe"
    assert local_action("what do you see") == "describe"
    for said in ("describe the enemy on the left", "what does the enemy look like",
                 "what does Frog look like", "what do I look like"):
        assert local_action(said) is None, said


def session(locs, names, feel, verbosity="normal"):
    """A session on Chrono Trigger (Steam), its room survey finding `locs` in turn."""
    s = Session.__new__(Session)
    s.agent = SimpleNamespace(_ct_steam=lambda f: True, game="Chrono Trigger")
    s.narrator = SimpleNamespace(recent=deque(maxlen=4), verbosity=verbosity)
    where = iter(locs)

    def locate(image):
        w = next(where)
        return w and SimpleNamespace(loc=w)

    s.rooms = SimpleNamespace(worlds={0x1F0}, name=lambda loc: names[loc], locate=locate,
                              feel=feel, dark=False)
    s._reads = lambda frame: True
    s.read_owned = False
    return s


def frame(colour=(0, 0, 0)):
    return Frame(Image.new("RGB", (1920, 1080), colour), time.monotonic())


async def test_a_place_is_painted_once_after_its_name_the_first_time():
    names = {0x05: "Leene Square", 0x0C: "Truce Inn"}
    s = session([0x05, 0x05, 0x0C, 0x05], names, {0x05: LEENE, 0x0C: INN})
    f = frame()
    assert await s._arrival(f, s._feel_ok(f)) == ("Leene Square", LEENE, False)
    assert await s._arrival(f, s._feel_ok(f)) == ("", "", False)  # still there
    assert await s._arrival(f, s._feel_ok(f)) == ("Truce Inn", INN, False)
    s.place_said = ("", 0.0)  # long enough ago to name it again
    assert await s._arrival(f, s._feel_ok(f)) == ("Leene Square", "", False)  # never again


async def test_no_place_line_in_terse_or_over_a_text_box():
    names = {0x05: "Leene Square"}
    s = session([0x05, 0x05, 0x05], names, {0x05: LEENE}, verbosity="terse")
    f = frame()
    assert not s._feel_ok(f)
    assert await s._arrival(f, s._feel_ok(f)) == ("Leene Square", "", False)
    s.narrator.verbosity = "normal"
    s.read_owned = True  # the reader is reading a text box on this screen
    assert not s._feel_ok(f)
    s.read_owned = False
    s.place_said = ("", 0.0)
    # Not heard yet (terse then), so it's still the first time.
    assert await s._arrival(f, s._feel_ok(f)) == ("Leene Square", LEENE, False)


async def test_the_line_still_comes_when_player_two_just_named_the_place():
    s = session([0x05], {0x05: "Leene Square"}, {0x05: LEENE})
    s.narrator.recent.append("We're at Leene Square.")
    f = frame()
    assert await s._arrival(f, s._feel_ok(f)) == ("", LEENE, False)


async def test_a_place_not_surveyed_gets_the_models_line_once_by_its_colours():
    s = session([None, None, None, None], {}, {})
    forest, castle = frame((20, 120, 30)), frame((90, 80, 90))
    assert await s._arrival(forest, True) == ("", "", True)
    assert await s._arrival(forest, True) == ("", "", False)  # the same place
    assert await s._arrival(castle, True) == ("", "", True)
    assert await s._arrival(castle, False) == ("", "", False)  # terse: nothing to ask
    s.rooms.dark = True
    s.rooms.locate = lambda image: None
    assert await s._arrival(frame((200, 40, 40)), True) == ("", "", False)  # a fade


async def test_a_world_map_the_atlas_knows_is_said_by_where_on_it():
    """Live 3: slot 4 loaded on the world map with "1000 A.D." or a line about the woods,
    and "take me back outside" ended on "World map.": never where."""
    s = session([None, None, None], {}, {})
    wheres = iter(["World map, 1000 A.D., at Leene Square",
                   "World map, 1000 A.D., at Leene Square", ""])

    async def world_where(image):
        return next(wheres)

    s.agent.world_where = world_where
    f = frame((20, 120, 30))
    assert await s._arrival(f, True) == ("World map, 1000 A.D., at Leene Square", "", False)
    assert await s._arrival(f, True) == ("", "", False)  # still there: said once
    assert await s._arrival(frame((90, 80, 90)), True) == ("", "", True)  # not a map: as before


async def test_a_stretch_not_surveyed_of_a_place_just_painted_gets_no_second_line():
    s = session([0x189, None, None], {0x189: "Mountain of Woe"}, {0x189: "High ledges."})
    mist, cave = frame((230, 230, 200)), frame((60, 40, 30))
    assert await s._arrival(mist, True) == ("Mountain of Woe", "High ledges.", False)
    assert await s._arrival(mist, True) == ("", "", False)  # more of the same mist
    assert await s._arrival(cave, True) == ("", "", True)  # somewhere else


def test_screens_of_one_place_share_their_colours():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    moved = np.roll(base, 12, axis=1)  # scrolled
    other = (base // 3).astype(np.uint8)  # a darker place
    a, b, c = (palette(Image.fromarray(x)) for x in (base, moved, other))
    assert same_look(a, b) > 0.9
    assert same_look(a, c) < 0.5


class FakeNarrator:
    def __init__(self, replies):
        self.replies = replies
        self.recent: deque[str] = deque(maxlen=4)
        self.calls: list[tuple[str, list[str]]] = []
        self.verbosity = "normal"

    def narrate(self, frame, extra=""):
        """A model call, made when asked for (its reply comes a moment later)."""
        self.calls.append((extra, list(self.recent)))
        reply = self.replies.pop(0)

        async def stream():
            await asyncio.sleep(0.05)
            for d in reply:
                yield d
        return stream()


def narrating(s, narrator):
    events = []
    s.narrator = narrator
    s.events = SimpleNamespace(emit=lambda kind, **k: events.append((kind, k)))
    s.recovered = lambda: None

    async def say_stream(deltas, interrupt=False):
        return "".join([d async for d in deltas])

    s.voice = SimpleNamespace(say_stream=say_stream)
    return events


async def test_after_the_places_line_the_model_says_only_whats_new():
    s = Session.__new__(Session)
    narrator = FakeNarrator([["A market with stalls."], []])  # then <silent>
    events = narrating(s, narrator)

    async def arrival(frame, feel=True, asked=False):
        return "Leene Square", LEENE, False

    s._arrival = arrival
    s._feel_ok = lambda frame: True
    await s._narrate(frame())
    said = [k["text"].strip() for kind, k in events if kind == "say"]
    assert said == [f"Leene Square. {LEENE}"]  # not "A market with stalls." as well
    # Asked again, knowing what was said.
    assert narrator.calls[-1] == (PLACE_SAID, [f"Leene Square. {LEENE}"])


async def test_a_new_place_not_surveyed_asks_the_model_what_its_like():
    s = Session.__new__(Session)
    narrator = FakeNarrator([["A rocky ledge."],
                             ["A rocky ledge.", " Grey crags under a pale sky."]])
    events = narrating(s, narrator)

    async def arrival(frame, feel=True, asked=False):
        return "", "", True

    s._arrival = arrival
    s._feel_ok = lambda frame: True
    await s._narrate(frame())
    said = [k["text"] for kind, k in events if kind == "say"]
    assert said == ["A rocky ledge. Grey crags under a pale sky."]
    assert narrator.calls[-1][0] == PLACE_FEEL


async def test_what_it_looks_like_starts_with_the_place_and_counts_as_its_first_time():
    s = session([0x05, 0x05], {0x05: "Leene Square"}, {0x05: LEENE})

    async def atmosphere(frame):
        await asyncio.sleep(0.05)
        for d in ("Bright cobbles", " and striped stalls."):
            yield d

    s.narrator.atmosphere = atmosphere
    said = "".join([d async for d in s._atmosphere(frame())])
    assert said == "Leene Square. Bright cobbles and striped stalls."
    f = frame()
    assert await s._arrival(f, s._feel_ok(f)) == ("", "", False)  # no line after that


async def test_the_atmosphere_answer_asks_for_the_scene_at_the_verbosity():
    asked = []

    class Muse:
        s = SimpleNamespace(image_max_side=640)

        async def stream(self, messages):
            asked.append(messages)
            yield "A sunlit plaza."

    n = Narrator(Muse(), "terse", "Chrono Trigger")
    out = "".join([d async for d in n.atmosphere(frame())])
    assert out == "A sunlit plaza."
    text = asked[0][1]["content"][1]["text"]
    assert ATMOSPHERE.split(":")[0] in text and "two short sentences" in text
    assert list(n.recent) == ["A sunlit plaza."]


async def test_the_tour_is_told_where_this_is_when_the_survey_knows():
    asked = []

    class Muse:
        s = SimpleNamespace(image_max_side=640)

        async def stream(self, messages):
            asked.append(messages[1]["content"][1]["text"])
            yield "A warm inn dining room. You're in Truce Inn."

    s = session([0x0C], {0x0C: "Truce Inn"}, {0x0C: INN})
    s.narrator = Narrator(Muse(), "normal", "Chrono Trigger")
    said = "".join([d async for d in s._describe(frame())])
    assert said == "A warm inn dining room. You're in Truce Inn."
    assert "it's Truce Inn" in asked[0] and "Open with one short sentence" in asked[0]


def test_rooms_read_each_places_line(tmp_path):
    from playertwo.rooms import Rooms

    img = np.zeros((224, 256, 3), np.uint8)
    img[::16] = 200
    Image.fromarray(img).save(tmp_path / "loc-005.png")
    (tmp_path / "loc-005.json").write_text(json.dumps({
        "loc": 5, "name": "Leene Square", "tile": 16, "walk": [[2, 2]], "wall": [],
        "exits": [], "cams": [[0, 0]]}))
    (tmp_path / "atmos.json").write_text(json.dumps({"lines": {"5": f"  {LEENE} "}}))
    assert Rooms(tmp_path).feel == {5: LEENE}
