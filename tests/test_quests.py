"""The quest list, where the player is in the story, and the agent tools on it: no
network, a scripted model."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from playertwo import atlas as atlas_mod
from playertwo import guide as guide_mod
from playertwo.agent import Agent
from playertwo.controls import DryRun
from playertwo.guide import Guide, chapters
from playertwo.notes import GameNotes
from playertwo.quests import (
    Progress,
    Quests,
    QuestStep,
    build,
    era_key,
    judge,
    link_places,
    pick_place,
    place_in_story,
    where_text,
)
from playertwo.screen import Frame

GUIDE = """1. The Fair

Wake up at home. Go to Leene Square, north of town, and meet a girl.

Blue Imp, RolyTreasures: Tonic x2

Level HP Attack Speed Magic Power 13 960 16 8 7 Defense Evade 127 128

Yakra

Some numbered tips:

1. Rename characters at Laruba Village whenever you like, it's a long tip line.

2. The Castle

Head to Guardia Forest, then north to Guardia Castle.
"""


def run(coro):
    return asyncio.run(coro)


class JsonMuse:
    """Scripted model for json calls: each call returns the next reply (a dict, or a
    function of the messages)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen: list = []
        self.s = SimpleNamespace(image_max_side=640)

    async def json(self, messages, effort=None, max_tokens=3000):
        self.seen.append(messages)
        r = self.replies.pop(0)
        return r(messages) if callable(r) else r


def step(n, era="1000 A.D.", place="Leene Square", **kw):
    return QuestStep(n=n, chapter=kw.pop("chapter", "1. The Fair"), era=era, place=place,
                     objective=kw.pop("objective", f"do thing {n}"), **kw)


def world(places=None):
    return SimpleNamespace(name="1000 A.D.", places=places if places is not None else {
        "Leene Square": (424, 214), "Truce Inn": (380, 280), "Crono's house": (440, 270),
        "Market": (330, 250)})


# --- the guide --------------------------------------------------------------------------

def test_chapters_count_up_so_a_numbered_tip_isnt_one():
    cs = chapters(GUIDE)
    assert [t for t, _ in cs] == ["1. The Fair", "2. The Castle"]
    assert "Rename characters" in cs[0][1]


def test_search_skips_stat_tables_and_enemy_lists():
    g = Guide(GUIDE)
    texts = [p.text for p in g.passages]
    assert not any("Level HP" in t or "Treasures:" in t or t == "Yakra" for t in texts)
    assert any("Leene Square" in t for t in texts)


def test_eras_compare_loosely():
    assert era_key("1000 A.D.") == era_key("1000 AD") == era_key("AD 1000")
    assert era_key("65 million B.C.") == era_key("65,000,000 B.C.")
    assert era_key("12,000 B.C.") == era_key("12000 BC") != era_key("12000 AD")
    assert era_key("The End of Time") == era_key("End of Time")


# --- building -----------------------------------------------------------------------------

def test_build_keeps_story_order_and_fills_unclear_eras():
    def reply(messages):
        title = messages[1]["content"].splitlines()[0]
        if "The Fair" in title:
            return {"steps": [
                {"era": "1000 A.D.", "place": "Crono's house", "objective": "Wake up"},
                {"era": "", "place": "Leene Square", "objective": "Meet a girl",
                 "items": "Pendant"},
                {"era": "", "place": "  ", "objective": "no place: dropped"}]}
        return {"steps": [{"era": "600 A.D.", "place": "Guardia Castle",
                           "objective": "Go in", "how_to_get_there": "north"}]}

    async def go():
        return await build(JsonMuse([reply, reply]), "Test Game", GUIDE, ["1000 A.D."],
                           log=lambda _t: None)

    q = run(go())
    assert [(s.n, s.place, s.era) for s in q.steps] == [
        (1, "Crono's house", "1000 A.D."), (2, "Leene Square", "1000 A.D."),
        (3, "Guardia Castle", "600 A.D.")]
    assert q.steps[1].items == ["Pendant"] and q.steps[2].how == "north"


def test_links_only_take_names_that_are_on_the_map():
    q = Quests([step(1, place="the Millennial Fair"), step(2, place="Truce"),
                step(3, place="Truce Inn"), step(4, era="600 A.D.", place="Truce Inn")])
    muse = JsonMuse([{"links": {"1": "Leene Square", "2": "Truce Village"}}])
    n = run(link_places(muse, q, {"1000 A.D.": ["Leene Square", "Truce Inn"]}, "G",
                        log=lambda _t: None))
    # Made up names are dropped, an unanswered exact name still links, other eras don't.
    assert [s.atlas_place for s in q.steps] == ["Leene Square", "", "Truce Inn", ""]
    assert n == 2
    assert q.aliases() == {"Leene Square": {"the Millennial Fair"}}


def test_quests_save_and_load(tmp_path):
    (tmp_path / "test-game.txt").write_text(GUIDE)
    q = Quests([step(1, items=["Pendant"], atlas_place="Leene Square")], "Test Game")
    q.save(tmp_path / "test-game.quests.json")
    back = Quests.for_game("Test Game", tmp_path)
    assert back is not None and back.steps == q.steps


# --- where in the story ------------------------------------------------------------------

def test_a_guess_is_only_fact_after_the_player_confirmed_near_it():
    q = Quests([step(i) for i in range(1, 11)] + [step(11, era="600 A.D.")])
    never = Progress()
    told = Progress(step=4, confirmed=True)
    guessed = Progress(step=4, confirmed=False)
    assert not judge(5, "high", never, "1000 A.D.", q)  # nothing confirmed yet
    assert judge(5, "high", told, "1000 A.D.", q)
    assert judge(4, "medium", told, "", q)  # still where they said
    assert not judge(5, "medium", told, "", q)
    assert not judge(9, "high", told, "", q)  # a jump ahead
    assert not judge(3, "high", told, "", q)  # back before what they said
    assert not judge(5, "high", guessed, "", q)
    assert not judge(5, "low", told, "", q)
    q.steps[4].era = "600 A.D."
    assert not judge(5, "high", told, "1000 A.D.", q)  # on another era's map


def test_unsure_place_in_story_says_to_ask():
    q = Quests([step(1, summary="Wake up at home"), step(2, summary="Meet a girl at the fair"),
                step(3, era="600 A.D.", place="Truce Canyon", summary="Cross a canyon"),
                step(4, era="600 A.D.", place="Guardia Castle", summary="Find the queen")])
    muse = JsonMuse([{"step": 2, "sure": "high", "other": 1, "why": "the fair on screen"}])
    g = run(place_in_story(muse, q, Progress(), Image.new("RGB", (64, 64)), "1000 A.D.",
                           "Leene Square"))
    assert g.ask and g.step == 2 and g.other == 1
    assert "ask_player" in g.text and "Meet a girl at the fair" in g.text
    prompt = muse.seen[0][0]["content"][-1]["text"]
    assert "era 1000 A.D., near Leene Square" in prompt
    # Steps in other eras aren't offered when the era is known, but for the one next
    # to them (where the story changes era).
    assert "Cross a canyon" in prompt and "Find the queen" not in prompt


def test_sure_place_in_story_says_now_and_after():
    q = Quests([step(1), step(2, place="Truce Inn", objective="Rest", how="south of the "
                                 "fair", atlas_place="Truce Inn"),
                step(3, optional=True), step(4, place="Guardia Forest", objective="Go north")])
    muse = JsonMuse([{"step": 2, "sure": "high", "why": "inn sign"}])
    g = run(place_in_story(muse, q, Progress(step=1, confirmed=True)))
    assert not g.ask
    assert g.text.startswith("Now (step 2 of 4")
    assert "south of the fair" in g.text and "'Truce Inn' (go_to_place" in g.text
    assert "After that (step 4)" in g.text  # the optional step is skipped


def test_a_made_up_step_number_falls_back_to_the_last_settled():
    q = Quests([step(i) for i in range(1, 6)])
    muse = JsonMuse([{"step": 99, "sure": "high"}])
    g = run(place_in_story(muse, q, Progress(step=3, confirmed=True)))
    assert g.step == 3


# --- where on the map, and which place --------------------------------------------------

def test_where_text_says_the_players_side_of_the_nearest_place():
    text, at = where_text(world(), 424, 214 + 3 * 8 + 12, rise=12)
    assert text.startswith("On the 1000 A.D. map, just south of Leene Square (3 steps)")
    assert at == "Leene Square"
    text, at = where_text(world(), 424, 214)
    assert "at Leene Square" in text and at == "Leene Square"
    text, at = where_text(world(), 700, 600)
    assert "just" not in text and at == ""
    text, _ = where_text(world({}), 1, 1)
    assert "no places are named" in text


def test_where_said_is_the_same_plain_line_from_the_atlas():
    """Live 3: "where am I?" was reworded by the model every time, once "at the fairgrounds
    ... with Leene Square just to the north" (the same place)."""
    from playertwo.quests import where_said

    aliases = {"Leene Square": {"the fair", "Millennial Fair"}}
    line, at = where_said(world(), 424, 214, aliases=aliases)
    assert line == ("On the world map, 1000 A.D., right at Leene Square, the fair. "
                    "Crono's house is south and Truce Inn is southwest.")
    assert at == "Leene Square"
    line, at = where_said(world(), 424, 214 + 3 * 8 + 12, rise=12)
    assert line.startswith("On the world map, 1000 A.D., just south of Leene Square.")
    line, at = where_said(world(), 700, 600)
    assert "a little way off" in line and at == ""
    other = SimpleNamespace(name="Overworld", places={"Truce Residence (west)": (0, 0)})
    assert where_said(other, 0, 0)[0] == "On the Overworld map, right at the west Truce Residence."


def test_where_short_for_a_line_on_arriving():
    from playertwo.quests import where_short

    assert where_short(world(), 424, 214) == "World map, 1000 A.D., at Leene Square"
    assert where_short(world(), 424, 214 - 4 * 8) == "World map, 1000 A.D., north of Leene Square"
    assert where_short(world(), 900, 900) == "World map, 1000 A.D."


def test_pick_place_exact_names_and_aliases_need_no_model():
    muse = JsonMuse([])
    m = world()
    assert run(pick_place(muse, "G", "truce inn", [m], {})) == (m, "Truce Inn")
    assert run(pick_place(muse, "G", "the Millennial Fair", [m],
                          {"Leene Square": {"the Millennial Fair"}})) == (m, "Leene Square")
    assert muse.seen == []


def test_walkthrough_names_are_hints_and_never_a_whole_town():
    m = world({"Truce Inn": (1, 1), "Truce Market": (2, 2), "Leene Square": (3, 3)})
    muse = JsonMuse([{"place": "", "why": "a town"}])
    hints = {"Truce Market": {"Truce", "the shop in Truce"}, "Leene Square": {"the fair"}}
    assert run(pick_place(muse, "G", "Truce", [m], {}, hints=hints)) == "a town"
    prompt = muse.seen[0][0]["content"]
    assert "Truce Market (the shop in Truce)" in prompt and "Leene Square (the fair)" in prompt


def test_pick_place_asks_muse_and_checks_its_answer():
    m = world()
    muse = JsonMuse([{"map": "1000 A.D.", "place": "Leene Square", "why": "the fair"},
                     {"place": "", "why": "Truce is a town with several places"},
                     {"place": "Porre", "why": "made up"}])
    aliases = {"Leene Square": {"the Telepod"}}
    assert run(pick_place(muse, "G", "take me to the fair", [m], aliases)) == (
        m, "Leene Square")
    assert "Leene Square (the Telepod)" in muse.seen[0][0]["content"]
    assert run(pick_place(muse, "G", "Truce", [m], {})) == (
        "Truce is a town with several places")
    assert isinstance(run(pick_place(muse, "G", "Porre", [m], {})), str)


# --- the agent tools --------------------------------------------------------------------

class FakeAtlas:
    def __init__(self, spot):
        self.spot = spot
        self.maps = [atlas_mod.WorldMap("1000 A.D.", np.zeros((64, 64, 3), np.uint8),
                                        dict(world().places))]

    def locate(self, _image, near=None):
        return self.spot

    def map(self, name):
        return next(m for m in self.maps if m.name == name)


def agent_with(monkeypatch, tmp_path, spot, replies=()):
    fake = FakeAtlas(spot)
    monkeypatch.setattr(atlas_mod.Atlas, "for_game", classmethod(lambda cls, g: fake))
    monkeypatch.setattr(atlas_mod, "ATLASES", tmp_path / "atlas")
    monkeypatch.setattr(guide_mod, "GUIDES", tmp_path / "guides")
    muse = JsonMuse(list(replies))

    async def grab():
        return Frame(Image.new("RGB", (256, 224)), 0)

    async def speak(_t):
        pass

    agent = Agent(muse, DryRun(), grab, speak, game="Test Game",
                  notes=GameNotes(tmp_path / "games"))
    return agent, muse


def write_quests(tmp_path, steps):
    (tmp_path / "guides").mkdir(exist_ok=True)
    (tmp_path / "guides" / "test-game.txt").write_text(GUIDE)
    Quests(steps, "Test Game").save(tmp_path / "guides" / "test-game.quests.json")


def test_where_am_i_on_the_map_and_off_it(monkeypatch, tmp_path):
    spot = atlas_mod.Spot("1000 A.D.", 424, 214 + 2 * 8 + atlas_mod.RISE, 0.9, 0.3)
    agent, _ = agent_with(monkeypatch, tmp_path, spot)
    out = run(agent._act("where_am_i", {}))
    assert out.startswith("On the 1000 A.D. map, just south of Leene Square (2 steps)")
    assert "Places on this map for go_to_place: Leene Square, Truce Inn" in out
    assert Progress.load(tmp_path / "games", "Test Game").visited == ["Leene Square"]
    agent, _ = agent_with(monkeypatch, tmp_path, None)
    out = run(agent._act("where_am_i", {}))
    assert out.startswith("not on a world map") and "near Leene Square" in out


def test_where_am_i_is_said_from_the_atlas_with_no_model_turn(monkeypatch, tmp_path):
    spot = atlas_mod.Spot("1000 A.D.", 424, 214 + atlas_mod.RISE, 0.9, 0.3)
    agent, muse = agent_with(monkeypatch, tmp_path, spot)
    said = []

    async def speak(t):
        said.append(t)

    agent.speak = speak
    agent._ct_steam = lambda frame: True
    result = run(agent.run("Where am I?"))
    assert said == [result.outcome] and result.outcome.startswith(
        "On the world map, 1000 A.D., right at Leene Square")
    assert muse.seen == []  # no model turn
    # One path for every way of asking (fast.py had a second, world map only, with other
    # words): the same line, logged as a where_am_i step.
    for words in ("Player Two, where are we now?", "tell me where we are please",
                  "so, where am I exactly"):
        again = run(agent.run(words))
        assert again.outcome == result.outcome, words
        assert [s.tool for s in again.steps] == ["where_am_i"]
    assert muse.seen == []


def test_next_goal_asks_then_takes_the_players_word(monkeypatch, tmp_path):
    write_quests(tmp_path, [step(1, summary="Wake up"), step(2, summary="Meet a girl"),
                            step(3, place="Truce Inn", objective="Rest",
                                 atlas_place="Truce Inn")])
    agent, muse = agent_with(monkeypatch, tmp_path, None, [
        {"step": 2, "sure": "high", "why": "fair music"},
        {"step": 3, "sure": "high", "why": "inn sign"}])
    out = run(agent._act("next_goal", {}))
    assert "not sure" in out and "ask_player" in out
    assert Progress.load(tmp_path / "games", "Test Game").step == 0  # nothing settled
    out = run(agent._act("next_goal", {"step": "2"}))
    assert out.startswith("Now (step 2 of 3") and len(muse.seen) == 1  # no model call
    p = Progress.load(tmp_path / "games", "Test Game")
    assert (p.step, p.confirmed) == (2, True)
    out = run(agent._act("next_goal", {"player_says": "I met her"}))
    assert out.startswith("Now (step 3 of 3") and "'Truce Inn'" in out
    assert 'The player says: "I met her"' in muse.seen[-1][0]["content"][-1]["text"]
    assert Progress.load(tmp_path / "games", "Test Game").step == 3


def test_a_match_no_better_than_elsewhere_isnt_on_the_map(monkeypatch, tmp_path):
    flat = atlas_mod.Spot("1000 A.D.", 424, 240, 0.74, 0.0)  # a cutscene of sky
    agent, _ = agent_with(monkeypatch, tmp_path, flat)
    assert run(agent._act("where_am_i", {})).startswith("not on a world map")


def test_the_players_word_opens_the_whole_list():
    q = Quests([step(1), step(2, era="600 A.D.", place="Cathedral", summary="Frog joins")])
    muse = JsonMuse([{"step": 2, "sure": "medium"}])
    g = run(place_in_story(muse, q, Progress(), None, "1000 A.D.", "", "Frog just joined"))
    assert g.step == 2 and "Frog joins" in muse.seen[0][0]["content"][-1]["text"]


def test_next_goal_without_a_quest_list(monkeypatch, tmp_path):
    agent, _ = agent_with(monkeypatch, tmp_path, None)
    assert run(agent._act("next_goal", {})).startswith("no quest list")


def test_go_to_place_picks_with_muse_not_shared_words(monkeypatch, tmp_path):
    write_quests(tmp_path, [step(1, place="the Millennial Fair", atlas_place="Leene Square")])
    spot = atlas_mod.Spot("1000 A.D.", 300, 300, 0.9, 0.3)
    agent, muse = agent_with(monkeypatch, tmp_path, spot, [
        {"place": "", "why": "Truce is a whole town"},
        {"map": "1000 A.D.", "place": "Leene Square", "why": "the fair"}])
    walked = []

    async def fake_walk(atlas, m, name, *a, **kw):
        walked.append(name)
        return f"arrived: went into {name} (3 s)"

    monkeypatch.setattr(atlas_mod, "walk_to", fake_walk)
    out = run(agent._act("go_to_place", {"place": "Truce"}))
    assert "no single place" in out and "Truce is a whole town" in out and not walked
    prompt = muse.seen[0][0]["content"]
    assert "Leene Square (the Millennial Fair)" in prompt  # the walkthrough's name, a hint
    assert "near Market" in prompt  # where the player is, for "the inn"
    out = run(agent._act("go_to_place", {"place": "the Millennial Fair"}))
    assert out.startswith("arrived") and walked == ["Leene Square"]
    assert Progress.load(tmp_path / "games", "Test Game").visited == ["Leene Square"]


def test_tool_list_has_the_new_tools():
    from playertwo.agent import TOOLS

    names = [t["function"]["name"] for t in TOOLS]
    assert {"where_am_i", "next_goal", "go_to_place"} <= set(names)
    json.dumps(TOOLS)  # still plain JSON
