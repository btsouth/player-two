"""The agent finding people by the player's own words inside a surveyed place: the model's
rewording doesn't win over them, two alike are asked about, "talk to X" goes with no model
turn first, and "who's here?" is a local command."""

import asyncio
import json
from types import SimpleNamespace

from test_rooms import Sim, SimPad, two_floors

from playertwo.agent import Agent
from playertwo.rooms import Person
from playertwo.screen import Frame


class Muse:
    """A model that finishes at once, noting what it was told."""

    def __init__(self):
        self.seen = []
        self.s = SimpleNamespace(image_max_side=640)

    async def tools(self, messages, tools, effort=None, max_tokens=3000):
        self.seen.append(messages)
        return SimpleNamespace(content="", tool_calls=[SimpleNamespace(
            id="c0", function=SimpleNamespace(name="finish", arguments=json.dumps(
                {"text": "Done."})),
            model_dump=lambda: {"id": "c0", "type": "function", "function": {
                "name": "finish", "arguments": json.dumps({"text": "Done."})}})])

    async def json(self, *a, **k):
        raise AssertionError("no model pick needed")


def agent_in(tmp_path, monkeypatch, ask=None, muse=None):
    from playertwo import rooms as rooms_mod

    async def no_model(*a, **k):
        raise AssertionError("the model's pick wasn't needed")

    monkeypatch.setattr(rooms_mod, "pick_exit", no_model)
    rooms, img = two_floors(tmp_path)
    sim = Sim(img, rooms.rooms[2].walk, 280.0, 173.0)
    ends = rooms.stand_by(rooms.rooms[2], (152.0, 77.9), True)
    sim.answers = lambda x, y, f: ends.get((int(y) // 16, int(x) // 16)) == f
    said = []

    async def grab():
        return Frame(await sim.shoot(), 0.0)

    async def speak(text):
        said.append(text)

    agent = Agent(muse, SimPad(sim), grab, speak, max_settle_s=0.2, ask=ask)
    agent._rooms = rooms
    agent.notes.folder = tmp_path / "notes"
    return agent, sim, said, rooms


def test_the_players_words_win_over_the_models(tmp_path, monkeypatch):
    agent, sim, said, _rooms = agent_in(tmp_path, monkeypatch)
    agent._request = "talk to the innkeeper"
    # The model reworded it by the look of someone else.
    out = asyncio.run(agent._act("talk_to", {"target": "the small animal by the bed"}))
    assert out.startswith("arrived") and "the innkeeper" in out, out
    assert said == ["Going to the innkeeper."]
    assert sim.pressed and sim.pressed[-1][0] == "a"
    # Not even when the model names someone else exactly (live: "the bearded old man at
    # the middle-right stall" for the old man at the lower stalls), till they've been
    # gone to.
    assert agent._theirs_done
    agent._theirs_done = False
    key, _ = asyncio.run(agent._resolve(*_inside(agent), "the cat", True, "go_to_exit"))
    assert key == "the innkeeper (in here)"
    # A walk the model makes for its own reasons isn't turned into the talk.
    agent._request = "talk to the innkeeper"
    out = asyncio.run(agent._resolve(*_inside(agent), "the cat", False, "walk_to"))
    assert out[0] == "the cat (in here)"


def _inside(agent):
    rooms = agent._rooms
    here = rooms.locate(asyncio.run(agent.grab()).image)
    choices, targets, _ = rooms.options(here)
    return rooms, here, choices, targets


def test_two_alike_are_asked_about(tmp_path, monkeypatch):
    asked = []

    async def ask(question):
        asked.append(question)
        return "the one by the window"

    agent, _sim, _said, rooms = agent_in(tmp_path, monkeypatch, ask=ask)
    room = rooms.rooms[2]
    room.people += [Person("the old man by the window", "person", (72.0, 60.0)),
                    Person("the old man by the stairs", "person", (280.0, 60.0))]
    agent._request = "talk to the old man"
    key, why = asyncio.run(agent._resolve(*_inside(agent), "the old man", True, "go_to_exit"))
    assert asked == ["Do you mean the old man by the window, or the old man by the stairs?"] \
        or asked == ["Do you mean the old man by the stairs, or the old man by the window?"]
    assert key == "the old man by the window (in here)" and not why
    # Asked once a request: the model's next call with other words goes there too.
    key, _ = asyncio.run(agent._resolve(*_inside(agent), "him", True, "go_to_exit"))
    assert key == "the old man by the window (in here)" and len(asked) == 1


def test_the_players_someone_doesnt_take_over_later_walks(tmp_path, monkeypatch):
    asked = []

    async def ask(question):
        asked.append(question)
        return "the left one"

    agent, _sim, _said, rooms = agent_in(tmp_path, monkeypatch, ask=ask)
    agent._request = "talk to the innkeeper, then go downstairs"
    agent._settled_pick = agent._theirs = None
    agent._theirs_done = False
    inside = _inside(agent)
    # The model's call for a way out is that way out, not the innkeeper again.
    assert asyncio.run(agent._resolve(*inside, "downstairs", None, "go_to_exit"))[0] == \
        "the inn, downstairs"
    assert asyncio.run(agent._resolve(*inside, "the stairs down", None, "go_to_exit"))[0] \
        == "the innkeeper (in here)"
    agent._theirs_done = True  # gone to them: the model's words from now on
    assert asyncio.run(agent._resolve(*inside, "the cat", None, "go_to_exit"))[0] == \
        "the cat (in here)"
    # A pick the player made counts only between the same ones.
    rooms.rooms[2].people += [Person("the old man by the window", "person", (72.0, 60.0)),
                              Person("the old man by the stairs", "person", (280.0, 60.0))]
    agent._request = "talk to the old man"
    agent._theirs_done = False
    inside = _inside(agent)
    key, _ = asyncio.run(agent._resolve(*inside, "the old man", True, "go_to_exit"))
    assert key == "the old man by the window (in here)" and len(asked) == 1
    agent._theirs_done = True
    assert asyncio.run(agent._resolve(*inside, "the cat", None, "go_to_exit"))[0] == \
        "the cat (in here)"


def test_more_after_the_someone_goes_on_to_the_model(tmp_path, monkeypatch):
    muse = Muse()
    agent, _sim, said, _rooms = agent_in(tmp_path, monkeypatch, muse=muse)
    run = asyncio.run(agent.run("take me to the innkeeper and ask about a room"))
    assert run.steps[0].tool == "go_to_exit" and run.steps[0].args == {"to": "the innkeeper"}
    assert "We're there." not in said and muse.seen  # the rest is the model's
    told = muse.seen[0][1]["content"][1]["text"]
    assert "Now the rest of what they asked" in told


def test_no_one_to_ask_says_so(tmp_path, monkeypatch):
    agent, _sim, _said, rooms = agent_in(tmp_path, monkeypatch)
    rooms.rooms[2].people += [Person("the old man by the window", "person", (72.0, 60.0)),
                              Person("the old man by the stairs", "person", (280.0, 60.0))]
    agent._request = "talk to the old man"
    key, why = asyncio.run(agent._resolve(*_inside(agent), "the old man", True, "go_to_exit"))
    assert key is None and why.startswith("not sure who: ask the player")


def test_talk_to_goes_with_no_model_turn_first(tmp_path, monkeypatch):
    muse = Muse()
    agent, sim, said, _rooms = agent_in(tmp_path, monkeypatch, muse=muse)
    run = asyncio.run(agent.run("talk to the innkeeper"))
    assert run.steps[0].tool == "go_to_exit"
    assert run.steps[0].args == {"to": "the innkeeper", "talk": True}
    assert run.steps[0].result.startswith("arrived"), run.steps[0].result
    assert said[0] == "Going to the innkeeper."
    assert sim.pressed and sim.pressed[-1][0] == "a"
    told = muse.seen[0][1]["content"][1]["text"]
    assert "was done already" in told and 'The player says: "talk to the innkeeper"' in told


def test_take_me_to_someone_here_needs_no_model(tmp_path, monkeypatch):
    muse = Muse()
    agent, sim, said, _rooms = agent_in(tmp_path, monkeypatch, muse=muse)
    run = asyncio.run(agent.run("take me to the innkeeper"))
    assert run.steps[0].tool == "go_to_exit" and run.steps[0].args == {"to": "the innkeeper"}
    assert said == ["Going to the innkeeper.", "We're there."]
    assert not muse.seen and not sim.pressed  # there, not talking


def test_talk_to_someone_unknown_leaves_it_to_the_model(tmp_path, monkeypatch):
    muse = Muse()
    agent, sim, _said, _rooms = agent_in(tmp_path, monkeypatch, muse=muse)
    run = asyncio.run(agent.run("talk to the purple monster"))
    assert not run.steps or run.steps[0].tool != "go_to_exit"
    assert not sim.pressed


def test_whos_here_is_answered_from_the_survey(tmp_path, monkeypatch):
    agent, _sim, _said, _rooms = agent_in(tmp_path, monkeypatch)
    text = asyncio.run(agent.around_here())
    assert text.startswith("In the inn, upstairs. Nearest: the cat, ")
    assert "the innkeeper, " in text and "Ways out: the inn, downstairs, " in text
    assert asyncio.run(agent.around_here(more=True)) == "That's everyone here."
    async def atlas():
        return object()

    async def off_the_map(_atlas, _image):
        return None

    agent._the_atlas, agent._spot = atlas, off_the_map
    where = asyncio.run(agent._where_am_i())
    assert where.startswith("In the inn, upstairs.") and "nearest first" in where


def test_whos_here_words():
    from playertwo.app import local_action

    for said in ("who's here", "Who is here?", "who's around", "what's around me",
                 "who can I talk to", "is anyone here", "who's nearby", "who's around here"):
        assert local_action(said) == "around", said
    for said in ("who else", "who else is here", "anyone else"):
        assert local_action(said) == "around_more", said
    assert local_action("where am i") == "describe"
    assert local_action("look around") == "describe"
    assert local_action("who's the boss here") is None


def test_no_answer_to_which_one_ends_the_request(tmp_path, monkeypatch):
    """Live 3: "talk to the old man by the stalls", asked which one and not answered: the
    model went on for 166 s, said "Got it" and asked the same three more times."""
    asked = []

    async def ask(question):
        asked.append(question)
        return ""

    muse = Muse()
    agent, sim, said, rooms = agent_in(tmp_path, monkeypatch, ask=ask, muse=muse)
    rooms.rooms[2].people += [Person("the old man by the window", "person", (72.0, 60.0)),
                              Person("the old man by the stairs", "person", (280.0, 60.0))]
    run = asyncio.run(agent.run("talk to the old man"))
    assert len(asked) == 1 and not muse.seen  # no model turn after it
    assert said[-1] == "I'll wait till you tell me which one."
    assert run.outcome == "I'll wait till you tell me which one."
    assert not sim.pressed


class AskingMuse(Muse):
    """A model that asks the player something, then (if it gets another turn) walks."""

    def __init__(self, calls):
        super().__init__()
        self.calls = list(calls)

    async def tools(self, messages, tools, effort=None, max_tokens=3000):
        self.seen.append(messages)
        name, args = self.calls.pop(0)
        dump = {"id": "c0", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}
        return SimpleNamespace(content="", tool_calls=[SimpleNamespace(
            id="c0", function=SimpleNamespace(name=name, arguments=json.dumps(args)),
            model_dump=lambda: dump)])


def test_no_answer_to_the_models_question_ends_the_request(tmp_path, monkeypatch):
    async def ask(question):
        return ""

    muse = AskingMuse([("ask_player", {"question": "Take a look, or no thanks?"}),
                       ("hold", {"direction": "up", "ms": 800})])
    agent, sim, said, _rooms = agent_in(tmp_path, monkeypatch, ask=ask, muse=muse)
    run = asyncio.run(agent.run("what should I do here?"))
    assert len(muse.seen) == 1  # no second turn
    assert said[-1] == "Okay, I'll wait till you decide."
    assert run.outcome == "Okay, I'll wait till you decide."


def test_take_me_to_the_save_point_says_so_with_no_model_turn(tmp_path, monkeypatch):
    """Live 3: after a 14 s walk onto the save point the model said "You're already
    standing on the save point"."""
    muse = Muse()
    agent, sim, said, rooms = agent_in(tmp_path, monkeypatch, muse=muse)
    rooms.rooms[2].people.append(Person("the save point", "save point", (232.0, 125.0)))
    run = asyncio.run(agent.run("take me to the save point"))
    assert run.steps[0].result.startswith("arrived: standing on the save point"), run.steps
    assert not muse.seen and not sim.pressed
    assert said[-1] == "We're on the save point. To save, open the menu; Save is lit there."
