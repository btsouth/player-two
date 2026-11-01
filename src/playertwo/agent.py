"""Voice commands: look at the screen, press buttons, look again, until it's done."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import openai
from PIL import Image

from . import ctsteam
from .ack import FIGHT, RUN
from .areamap import AreaMap
from .battle import Executor, Intent, Stop, parse_intent
from .controls import BUTTONS, DIRECTIONS, Controls
from .ctsteam import State
from .explore import ANGLES, Trail, find_me, keep_walking, words
from .muse import Muse, image_part
from .notes import GameNotes
from .prompts import AGENT, BATTLE_CT, BATTLE_DECIDE, CONFIRM, GAME_CONTEXT, LANGUAGE, WHO_GLOVE
from .screen import Frame, thumb
from .walk import Walker, small

# Never the guide button: it opens the Steam overlay or the system menu, not the game.
# Seconds of silence in a run before the model's own line about its plan is spoken.
PROGRESS_GAP_S = 8.0
# Walking around by trial and error in one request stops after this long: wandering on
# camera looks lost, and an honest "I don't know the way here yet" is better.
WANDER_LIMIT_S = 40.0
AGENT_BUTTONS = [b for b in BUTTONS if b != "guide"]

TOOLS = [
    {"type": "function", "function": {
        "name": "press",
        "description": "Press controller buttons in order. d-pad is up/down/left/right.",
        "parameters": {"type": "object", "properties": {
            "buttons": {"type": "array", "items": {"type": "string", "enum": AGENT_BUTTONS},
                        "description": "Buttons to press one after another."},
            "gap_ms": {"type": "integer", "description": "Pause between presses, default 150."},
            "say": {"type": "string", "description": "Said out loud before pressing: the "
                    "dialogue line on screen (who, then the line word for word) when a "
                    "moves it on, so it's one step."},
            "battle": {"type": "boolean", "description": "true for a command in a fight's "
                       "menus, cursor moves and confirms together (a then a to attack the "
                       "highlighted enemy): the fight goes on while you look, so a command is "
                       "one press. Not for dialogue or other menus."},
            "skipping": {"type": "boolean", "description": "true only when the player asked "
                         "to skip or fast-forward text or a cutscene: then a can be pressed "
                         "several times. Otherwise any a after another press waits for you to "
                         "look first."},
        }, "required": ["buttons"]}}},
    {"type": "function", "function": {
        "name": "hold",
        "description": "Hold a button or d-pad direction for a while (2D games walk with "
                       "the d-pad), or push a stick in a direction (3D games walk with the "
                       "left stick); or several buttons down at once with `together` (lb and "
                       "rb held together runs from a fight in Chrono Trigger).",
        "parameters": {"type": "object", "properties": {
            "input": {"type": "string", "enum": ["left_stick", "right_stick", *AGENT_BUTTONS]},
            "together": {"type": "array", "items": {"type": "string", "enum": AGENT_BUTTONS},
                         "description": "Buttons held down at the same time (then input is "
                         "ignored). Holding one and then the other isn't the same."},
            "direction": {"type": "string", "enum": list(DIRECTIONS)},
            "ms": {"type": "integer", "description": "How long, in milliseconds (max 5000; "
                   "8000 for buttons held together)."},
        }, "required": ["ms"]}}},
    {"type": "function", "function": {
        "name": "walk_to",
        "description": "Walk the player's character to something you can see on screen: "
                       "a door, stairs, a person, a chest, an exit. For games seen from "
                       "above (2D, top-down or angled); it steers by itself, goes around "
                       "what's in the way, and stops next to it, facing it. To talk to "
                       "it, open or search it, use talk_to instead. Not for 3D games with the camera behind "
                       "the character.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "What to walk to, described so "
                       "it can be found on this screen: \"the woman in the green dress\", "
                       "\"the stairs at the top left\"."},
            "me": {"type": "string", "description": "Which character on screen is the "
                   "player's, so it isn't mixed up with others: \"the boy with spiky red "
                   "hair\"."},
        }, "required": ["target"]}}},
    {"type": "function", "function": {
        "name": "talk_to",
        "description": "Walk to a person (or a sign, a chest, a door) you can see on "
                       "screen and press a when there, in games seen from above: walk_to "
                       "and the press in one. Look at the screen after to read what they "
                       "say.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "Who or what, described so it "
                       "can be found on this screen."},
            "me": {"type": "string", "description": "Which character is the player's, "
                   "if it's unclear."},
        }, "required": ["target"]}}},
    {"type": "function", "function": {
        "name": "keep_walking",
        "description": "Walk on in one direction toward somewhere not on screen yet, "
                       "in games seen from above. It watches as it goes and stops at a "
                       "wall, at a new area (a door, an exit) or when something comes up, "
                       "then tells you how far it went. Better than holding the d-pad "
                       "blind.",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": list(ANGLES)},
            "seconds": {"type": "number", "description": "At most how long, 1 to 8; "
                        "default 3."},
        }, "required": ["direction"]}}},
    {"type": "function", "function": {
        "name": "go_to_place",
        "description": "On a world map, go to a named place with the game's atlas (its "
                       "maps, if the player has them): it knows exactly where you are and "
                       "where the place is, walks the way there around water and forest, "
                       "and goes in. Use it first for any 'take me to X' on a world map; "
                       "with no atlas or no such place it says so and lists the places.",
        "parameters": {"type": "object", "properties": {
            "place": {"type": "string", "description": "The place's exact name when "
                      "you have it (where_am_i and next_goal give them), else as the "
                      "player said it."},
        }, "required": ["place"]}}},
    {"type": "function", "function": {
        "name": "go_to_era",
        "description": "Chrono Trigger, once the party has the Epoch (the winged time "
                       "machine): go to another era by its time gauge: it walks to the "
                       "parked Epoch, boards, sets the gauge, and lands near where it "
                       "arrives. For 'take me to 600 AD', 'go to the future', 'the End of "
                       "Time'. A place in another era or on another continent: go_to_place "
                       "flies there by itself.",
        "parameters": {"type": "object", "properties": {
            "era": {"type": "string", "description": "The era as the player said it "
                    "('600 AD', 'the middle ages', 'prehistoric')."},
        }, "required": ["era"]}}},
    {"type": "function", "function": {
        "name": "go_to_exit",
        "description": "Inside a place (a house, a shop, a town, a castle, a forest) the "
                       "game's room survey knows: go through the way out to where the player "
                       "wants, even off screen and through other rooms on the way: "
                       "'outside', 'upstairs', 'the north door', 'Leene Square', 'the way to "
                       "the castle'; to a thing ('the counter', 'the bell', 'the save "
                       "point'); or up to someone to talk to them ('the innkeeper', 'the "
                       "shopkeeper', 'Mom'), in this place or the next. It knows exactly "
                       "where you are, every way out and who stands where. Use it first for "
                       "any 'take me outside', 'go upstairs', 'talk to the innkeeper', "
                       "'take me to the shop counter' indoors; in a place it doesn't know it "
                       "says so quickly.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "description": "Where to or who, as the player said "
                   "it ('outside', 'downstairs', 'the fair', 'the innkeeper'), or an exact "
                   "name from where_am_i."},
            "talk": {"type": "boolean", "description": "true to talk to them (or use, "
                     "open, buy from it) once there: it presses a facing them."},
        }, "required": ["to"]}}},
    {"type": "function", "function": {
        "name": "where_am_i",
        "description": "Where the player is on a world map, from the game's atlas: the "
                       "era or map, the nearest place and which way the others are, and "
                       "the names go_to_place knows. Inside a surveyed place: which place and "
                       "its ways out, by direction and steps. Elsewhere it says so; answer "
                       "from the screen then.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "next_goal",
        "description": "What to do next in the story and where, from this game's quest "
                       "list (made from the walkthrough): the step the player is on, the "
                       "way there and the one after. If it isn't sure where they are, it "
                       "says what to ask them.",
        "parameters": {"type": "object", "properties": {
            "step": {"type": "integer", "description": "The step number the player "
                     "just confirmed they're on."},
            "player_says": {"type": "string", "description": "What the player said "
                            "about where they are in the story, if they did."},
        }}}},
    {"type": "function", "function": {
        "name": "guide",
        "description": "Look something up in this game's walkthrough, if the player has "
                       "one: where a place is and how to get there, what to do next, how a "
                       "puzzle works. Search with names of places, people and things "
                       "('Guardia Forest castle', 'Zenan Bridge'). It follows the story in "
                       "order, so check which part fits where you are.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "travel",
        "description": "Cross a big map (a world map, a large field) toward a place that "
                       "isn't on screen yet, in games seen from above: it heads that way, "
                       "slides around coasts, trees and walls in the way, and looks every "
                       "few seconds for the place. It stops when the place comes into "
                       "sight (then walk_to it), at a new area, or when that way is shut. "
                       "One call goes far; better than many keep_walking.",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": list(ANGLES)},
            "target": {"type": "string", "description": "What the place looks like on "
                       "the map, to know it when it comes into sight ('a grey stone "
                       "castle with towers')."},
            "seconds": {"type": "number", "description": "At most how long, 10 to 90; "
                        "default 45."},
        }, "required": ["direction", "target"]}}},
    {"type": "function", "function": {
        "name": "exits",
        "description": "List the ways out of the place on screen (doors, stairs, paths off "
                       "the edge, gaps in fences), nearest first, how far each is from your "
                       "character, and which one you probably came in by (2D games). Use it "
                       "to look for something that isn't on screen.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "area_map",
        "description": "See a map of this area so far, stitched from everything on "
                       "screen while walking (2D games): exits, gaps, paths and people "
                       "that have scrolled off. You get it with the next screen.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "walk_to_map",
        "description": "Walk to a spot on the area map you were shown last, even if it's "
                       "off screen now: x and y on that map image, 0 to 1000 each way "
                       "(0, 0 the top left). Stops next to it, at a wall or at a new area.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "label": {"type": "string", "description": "What's there, in a few words."},
        }, "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "wait",
        "description": "Wait without pressing anything: loading screens, fades, movies, "
                       "animations. You get a new screenshot after.",
        "parameters": {"type": "object", "properties": {
            "seconds": {"type": "number", "description": "How long, 0.5 to 8."},
        }, "required": ["seconds"]}}},
    {"type": "function", "function": {
        "name": "say",
        "description": "Say something short to the player out loud.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "ask_player",
        "description": "Ask the player a short question out loud and wait for their spoken "
                       "answer. Use it to confirm a plan or clear up what they meant.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}},
                       "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "remember",
        "description": "Keep a short fact about this game for later sessions, once you've "
                       "seen it hold: which button opens the menu or skips text, how "
                       "battles work, where a place or person is and how to get there, "
                       "what the player's character looks like. Not things that change "
                       "(health, what's on screen now).",
        "parameters": {"type": "object", "properties": {
            "note": {"type": "string"},
            "instead_of": {"type": "string", "description": "An earlier note this one "
                           "corrects, as it was written, so the wrong one goes."},
        }, "required": ["note"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "The request is done (or can't be done). Tell the player in one "
                       "short sentence.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}},
]


# Chrono Trigger on Steam: battles are played by the executor (battle.py), one command a
# call, and field menus by menu.py; in a fight they're the only way to act.
CT_TOOLS = [
    {"type": "function", "function": {
        "name": "battle",
        "description": "Chrono Trigger battle: one command for one character, done for you "
                       "(cursor moves, checks, confirm), watched until it lands. Says what "
                       "changed, or why it stopped.",
        "parameters": {"type": "object", "properties": {
            "actor": {"type": "string", "description": "Who acts: Crono, Marle, Frog... "
                      "Empty for whoever's command is up."},
            "command": {"type": "string", "enum": ["attack", "tech", "item"]},
            "name": {"type": "string", "description": "The tech or item, as the game "
                     "spells it (Cure, Cyclone, Mid-Potion, Athenian Water)."},
            "target": {"type": "string", "description": "An ally's name; left, right, top "
                       "or bottom for an enemy by where it stands; an enemy's name; or "
                       "empty for the game's own pick."},
        }, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "fight",
        "description": "Chrono Trigger battle: win plays the whole fight until it's won "
                       "(attacking, healing the low, reviving the knocked out); run tries "
                       "to run away.",
        "parameters": {"type": "object", "properties": {
            "mode": {"type": "string", "enum": ["win", "run"]},
        }, "required": ["mode"]}}},
    {"type": "function", "function": {
        "name": "field_menu",
        "description": "Chrono Trigger, outside battle: use a tech or an item on someone "
                       "through the menu ('Marle uses Cure on Crono', 'a Potion on Frog'), "
                       "or change a battle setting (Battle Mode: Wait is best; Battle Speed: "
                       "1 fastest to 8 slowest, 8 gives the player the most time). It opens "
                       "the menu, does it, checks it and closes the menu: call it straight away, "
                       "without pressing anything first.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["tech", "item", "setting"]},
            "user": {"type": "string", "description": "For a tech: who casts it."},
            "name": {"type": "string", "description": "The tech, the item or the setting "
                     "(as the game spells it)."},
            "target": {"type": "string", "description": "Who it's used on; for a setting, "
                       "the value wanted (Wait, Active, slowest, fastest)."},
        }, "required": ["action", "name"]}}},
]
FIELD_MENU = True  # the field_menu tool is offered (menu.py)
BATTLE_TOOLS = [*CT_TOOLS[:2], *(t for t in TOOLS if t["function"]["name"] in (
    "say", "ask_player", "remember", "wait", "finish"))]
FINISH_ONLY = [t for t in TOOLS if t["function"]["name"] == "finish"]
# Out of steps but not done: the player can say to keep going (that adds steps).
FINISH_OR_ASK = [t for t in TOOLS if t["function"]["name"] in ("finish", "ask_player")]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS + CT_TOOLS}
MOVES = {"up", "down", "left", "right", "lb", "rb"}
# Something broke while walking (not the model's arguments): the model hears it failed.
WALK_ERRORS = (ValueError, TypeError, IndexError, OSError, RuntimeError, openai.APIError)
WALK_FAILED = ("blocked", "timeout", "lost", "not_found", "walking failed")
# A text box or a cutscene in the way: not the way being cut off, nor a walk that got there.
WALK_PAUSED = ("didnt_move", "interrupted")


def screen_change(a: Frame, b: Frame, cell: float = 0.06) -> float:
    """Fraction of the screen that differs between two frames (small thumbnails)."""
    return float((abs(thumb(a.image) - thumb(b.image)) > cell).mean())


def skip_grid(f: Frame) -> np.ndarray:
    """A grey grid fine enough to tell one line of text from the next."""
    img = f.image.convert("L").resize((192, 108), Image.Resampling.BILINEAR)
    return np.asarray(img, np.float32) / 255


def skip_on(text: tuple | None, before: np.ndarray, after: np.ndarray) -> tuple[tuple | None, str]:
    """Whether a skip through text should go on after a press (grids from skip_grid).

    Lines of text change the same box on screen each time. When a press changes
    somewhere outside it (a menu, a Yes/No, a battle's commands) or the whole screen,
    the text is over and another a would pick something, so the skip stops there.
    Returns the box the text has used so far (top, bottom, left, right) and why to stop
    ("" to go on).
    """
    moved = abs(after - before) > 0.06
    if moved.mean() > 0.35:
        return text, "the whole screen changed"
    ys, xs = np.nonzero(moved)
    if not len(ys):
        return text, ""
    if text is not None:
        y0, y1, x0, x1 = text
        pad = 4
        inside = np.zeros_like(moved)
        inside[max(0, y0 - pad):y1 + pad + 1, max(0, x0 - pad):x1 + pad + 1] = True
        if (moved & ~inside).mean() > 0.004:
            return text, "something new showed up outside the text"
        ys, xs = np.r_[ys, y0, y1], np.r_[xs, x0, x1]
    return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())), ""


def way(angle: float | None) -> str:
    """The d-pad direction nearest a stick angle (0 up, 90 right), or ""."""
    if angle is None:
        return ""
    return ("up", "right", "down", "left")[round(angle / 90) % 4]


def split_before_confirm(buttons: list[str]) -> tuple[list[str], str]:
    """Cut a press list just before an `a` that follows any other press.

    Pressing and confirming blind is where wrong picks come from (a menu opened with
    start, a Yes/No that defaults to Yes, a second `a` that skips a line of dialogue the
    player never heard), so the agent looks at the screen before every confirm. An `a`
    first, or on its own, is fine.
    """
    for i, b in enumerate(buttons):
        if b == "a" and i > 0:
            return buttons[:i], "a"
    return buttons, ""


SEE_EPOCH = """This is Chrono Trigger's world map. Is the Epoch on it: a silver time \
machine with wings, parked on the ground (not the party walking)? JSON: {"epoch": [x, y]} \
with its middle in thousandths of the picture's width and height, or {"epoch": null}."""


# A plain request to go somewhere: on a world map the atlas knows, it's walked at once.
TRIP = re.compile(r"(?:(?:please |can you |could you )?(?:take|bring|get) (?:me|us) |"
                  r"(?:let'?s |please )?(?:go|head|walk|run) )(?:back )?(?:over )?"
                  r"(?:in)?to (?P<place>.{3,60})")


@dataclass
class Step:
    tool: str
    args: dict
    t: float
    result: str = ""


@dataclass
class Run:
    request: str
    steps: list[Step] = field(default_factory=list)
    outcome: str = ""
    seconds: float = 0.0
    # Per turn: seconds in the model, acting (presses, waits), and getting the next screen.
    turns: list[dict] = field(default_factory=list)


# Screenshots older than this many turns are dropped from what the model sees again: they
# make every later turn slower and cost more, and the latest screen is what matters.
KEEP_IMAGES = 2


def trim_images(msgs: list[dict], keep: int = KEEP_IMAGES) -> list[dict]:
    """A copy of the conversation with all but the newest `keep` screenshots replaced."""
    seen = 0
    out: list[dict] = []
    for m in reversed(msgs):
        content = m.get("content")
        if isinstance(content, list) and any(c.get("type") == "image_url" for c in content):
            seen += 1
            if seen > keep:
                m = {**m, "content": [c if c.get("type") != "image_url"
                                      else {"type": "text", "text": "(an earlier screen)"}
                                      for c in content]}
        out.append(m)
    return out[::-1]



def nearest_of_kind(m, name: str, said: str, at: tuple[float, float]) -> str:
    """Of places on a map that are the same kind ("Truce Mayor's Manor", "Porre Mayor's
    Manor"), the nearest, unless the player named which: "the mayor's house" means the
    one in this town."""
    import math

    def tokens(n: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", n.lower().replace("'s", ""))

    def kind(n: str) -> set[str]:
        w = tokens(n)
        return set(w[1:]) if len(w) > 1 else set(w)

    same = [n for n in m.places if n != name and kind(n) == kind(name) and kind(n)]
    if not same:
        return name
    # The same tokens on both sides: "mayor's" in the request isn't a town's name.
    asked = set(tokens(said))
    named = [n for n in [name, *same] if (set(tokens(n)) - kind(n)) & asked]
    if named:
        return name if name in named else named[0]  # they said which town
    return min([name, *same], key=lambda n: math.dist(m.places[n], at))


def way_round(atlas, m, rooms, name: str, here: tuple[float, float]
              ) -> tuple[str, tuple[float, float], str] | None:
    """When open ground doesn't reach `name` from here (the castle's side of Guardia
    Forest, after a trip to the castle), a place to go through: one with a way out that
    comes out on this side (its door here) and another that comes out where `name` can be
    walked to. (that place, its door here, the way out to take), or None."""
    import math

    goal, door = m.places[name], m.entrance(name)
    if rooms is None or getattr(m, "world", None) is None or \
            atlas.route(m, here, goal, door) is not None:
        return None
    best = None
    for room in rooms.rooms.values():
        outs = {}
        for e in room.exits:
            if e.to == m.world and any(e.at):
                outs.setdefault(tuple(e.at), e)
        if len(outs) < 2:
            continue
        ins = [a for a in outs if atlas.route(m, here, a) is not None]
        ends = [b for b in outs if atlas.route(m, b, goal, door) is not None]
        for a in ins:
            for b in ends:
                if math.dist(a, b) < 16:
                    continue  # the same side
                d = math.dist(here, a) + math.dist(b, goal)
                if best is None or d < best[0]:
                    best = (d, rooms.name(room.loc), a, outs[b].label)
    return best[1:] if best else None


def others_of_kind(m, name: str, said: str, at: tuple[float, float]) -> list[str]:
    """The other places of the same kind as `name`, nearest first, for when the way to it
    turns out shut; none when the player said which town ("the inn in Porre")."""
    import math

    def tokens(n: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", n.lower().replace("'s", "")))

    def kind(n: str) -> set[str]:
        w = re.findall(r"[a-z0-9]+", n.lower().replace("'s", ""))
        return set(w[1:]) if len(w) > 1 else set(w)

    same = [n for n in m.places if n != name and kind(n) == kind(name) and kind(n)]
    if any((tokens(n) - kind(n)) & tokens(said) for n in [name, *same]):
        return []
    return sorted(same, key=lambda n: math.dist(m.places[n], at))


class Agent:
    def __init__(self, muse: Muse, controls: Controls, grab: Callable[[], Awaitable[Frame]],
                 speak: Callable[[str], Awaitable[None]], game: str = "",
                 max_turns: int = 32, max_settle_s: float = 1.5,
                 ask: Callable[[str], Awaitable[str]] | None = None,
                 confirm: str = "important", language: str = "English",
                 on_action: Callable[[str, dict], None] | None = None,
                 look_before_confirm: bool = True, notes: GameNotes | None = None):
        self.muse = muse
        self.controls = controls
        self.grab = grab
        self.speak = speak
        self.game = game
        self.max_turns = max_turns
        self.max_settle_s = max_settle_s
        self.ask = ask
        self.on_action = on_action
        self.look_before_confirm = look_before_confirm
        self.confirm = confirm
        self.language = language
        self.history: list[str] = []
        self.said: list[str] = []
        self.cancelled = False
        self._walker: Walker | None = None
        self.trail = Trail()  # a rough map of the area from where walks went
        self.area = AreaMap()  # the area as seen while walking, stitched together
        self._map_image = None  # the map as last shown to the model
        self._map_shown = -1    # which map that was (_map_gen when shown)
        self._map_gen = 0       # counts new maps: one per area
        self._map_fed = 0.0     # when a frame last went on it
        self._show_map = False
        self.notes = notes or GameNotes()
        # Cleared while the player holds the talk key: no presses or speech until it's
        # known whether they said stop, asked something else, or just wanted a repeat.
        self.go = asyncio.Event()
        self.go.set()
        # Chrono Trigger on Steam: the battle executor, and the fight on screen (the
        # reader's state) while there is one.
        self._executor: Executor | None = None
        self._battle: State | None = None
        self._battle_seen = 0.0

    def hold(self) -> None:
        self.go.clear()

    def resume(self) -> None:
        self.go.set()

    def cancel(self) -> None:
        self.cancelled = True
        self.go.set()

    async def _speak(self, text: str) -> None:
        await self.go.wait()
        if not self.cancelled and text.strip():
            self.said.append(text)
            self._spoke_at = time.monotonic()
            await self.speak(text)

    def _wandering(self, name: str, args: dict) -> float | None:
        """Seconds spent so far this request walking by trial and error (walks without a
        map, held directions), or None when this tool isn't that kind of walking."""
        free = name in ("walk_to", "keep_walking", "travel", "walk_to_map", "talk_to") or (
            name == "hold" and (args.get("direction") or set(args.get("together") or [])
                                & {"up", "down", "left", "right"}
                                or str(args.get("input") or "") in ("up", "down", "left",
                                                                     "right"))) or (
            name == "press" and args.get("buttons") and all(
                b in ("up", "down", "left", "right") for b in args.get("buttons") or []))
        return getattr(self, "_wander_s", 0.0) if free else None

    def _progress_line(self, content: str, calls, t0: float) -> None:
        """Speak the model's own line about the screen or its plan, written with its tool
        calls, when the player has heard nothing for a while: a long run isn't a silent
        one. It was written from the screen before these calls run, so it can't claim
        their effect."""
        now = time.monotonic()
        if any(c.function.name in ("say", "finish", "ask_player") for c in calls):
            return
        quiet_since = max(t0, getattr(self, "_spoke_at", 0.0))
        line = " ".join(content.split())
        if not line or now - quiet_since < PROGRESS_GAP_S:
            return
        first = re.split(r"(?<=[.!?])\s+", line)
        line = " ".join(first[:2]) if len(first[0]) < 60 else first[0]
        if len(line) > 160 or line.endswith(":") or re.match(
                r"(?i)(i('ll| will) (look|check)|let me|looking|now i|ok(ay)?[,.]? )", line):
            return
        if self.said and SequenceMatcher(None, self.said[-1].lower(), line.lower()).ratio() > 0.6:
            return
        self._spoke_at = now
        asyncio.get_running_loop().create_task(self._speak(line))

    def _already_said(self, text: str) -> bool:
        """Whether the last say already said it (a finish that restates it). Only a say
        since the last action counts: "Saved the game" after "I'll save the game" and a
        press is news. Never when the numbers or a "not" differ either: "I couldn't open
        the map" after "I'll open the map" is news."""
        if len(self.said) <= self._said_before_act:
            return False
        said = self.said[-1]

        def words(t: str) -> set[str]:
            return set(re.findall(r"[a-z0-9']+", t.lower().replace("\u2019", "'")))

        def negations(w: set[str]) -> set[str]:
            return {x for x in w if x in ("not", "no", "never", "nothing", "none", "cannot")
                    or x.endswith("n't")}

        now, before = words(text), words(said)
        if not now or re.findall(r"\d+", said) != re.findall(r"\d+", text):
            return False
        if negations(now) != negations(before):
            return False
        return now <= before or len(now & before) / len(now | before) >= 0.5

    def system(self) -> str:
        parts = [AGENT, CONFIRM.get(self.confirm, CONFIRM["important"])]
        parts.append(LANGUAGE.format(language=self.language))
        if self.game:
            parts.append(GAME_CONTEXT.format(game=self.game))
            learned = self.notes.prompt(self.game)
            if learned:
                parts.append(learned)
        return "\n".join(parts)

    async def _noise(self) -> float:
        """How much the screen changes by itself (fog, idle animation) between two looks."""
        a = await self.grab()
        await asyncio.sleep(0.05)
        return screen_change(a, await self.grab())

    async def _settled(self, noise: float | None) -> Frame:
        """The screen once it has stopped changing after an action, or after max_settle_s.

        Most menus redraw within a frame or two, so this is usually much quicker than a
        fixed pause; slow transitions get the time they need.
        """
        await asyncio.sleep(0.08)
        still = max(0.004, (noise or 0.0) * 1.3 + 0.002)
        seen = [(time.monotonic(), await self.grab())]
        end = time.monotonic() + self.max_settle_s
        calm = 0
        while time.monotonic() < end:
            await asyncio.sleep(0.05)
            now, cur = time.monotonic(), await self.grab()
            # Against a frame a quarter second back, not the last one: text typing out
            # a few letters at a time looks still from one frame to the next.
            old = [f for t, f in seen if now - t >= 0.25]
            seen.append((now, cur))
            if not old:
                continue
            calm = calm + 1 if screen_change(old[-1], cur) <= still else 0
            if calm >= 2:
                return cur
        return seen[-1][1]

    async def _walk_said(self, walker: Walker) -> None:
        """A line the walk started saying ("Something's in the way") ends before the next
        step, so it isn't taken for something said after the walk; stopped, it's cut."""
        talk = getattr(walker, "_talk", None)
        if talk is None or talk.done():
            return
        if self.cancelled:
            talk.cancel()
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(talk), 8)

    async def _press_a_and_see(self) -> float:
        """Press a; how much of the screen changed in the next second (a text box takes
        a moment to open)."""
        before = await self.grab()
        await asyncio.to_thread(self.controls.press, "a")
        changed = 0.0
        for _ in range(4):
            await asyncio.sleep(0.25)
            changed = max(changed, screen_change(before, await self.grab()))
            if changed > 0.04:
                break
        return changed

    async def _act(self, name: str, a: dict) -> str:
        if name in ("press", "hold") and self._battle is not None:
            return ("not pressed: in this Chrono Trigger fight, act with battle (one "
                    "command) or fight (win or run); they do the presses")
        if name == "press":
            asked = a.get("buttons") or []
            if isinstance(asked, str):
                asked = [asked]  # one button given bare, not a list of letters
            asked = [str(b) for b in asked]
            if sum(b in ANGLES for b in asked) >= 2 and \
                    time.monotonic() > getattr(self, "_map_free_until", 0.0):
                on_map = await self._on_surveyed_map()  # walking by d-pad presses
                if on_map:
                    return on_map
            buttons = [b for b in asked if b in AGENT_BUTTONS][:12]
            ignored = [b for b in asked if b not in AGENT_BUTTONS]
            if not buttons:
                return (f"nothing pressed: {', '.join(ignored) or 'no buttons'} isn't a "
                        f"button; use {', '.join(AGENT_BUTTONS)}")
            if str(a.get("say") or "").strip():
                await self._speak(str(a["say"]))
            held_back = ""
            skipping = a.get("skipping") is True and set(buttons) <= {"a", "b", "start"}
            battle = a.get("battle") is True and "start" not in buttons
            if self.look_before_confirm and not skipping and not battle:
                buttons, held_back = split_before_confirm(buttons)
            gap = max(60, min(int(float(a.get("gap_ms") or 150)), 1000))
            if self.on_action:
                self.on_action(name, {**a, "buttons": buttons})
            # A skip watches each press land, and stops when the text is over.
            watch = skipping and self.look_before_confirm and len(buttons) > 1
            if watch:
                noise = await self._noise()
                before, box = skip_grid(await self.grab()), None
            # One press at a time, so "stop" takes effect between presses.
            pressed, stopped = [], ""
            for i, b in enumerate(buttons):
                await self.go.wait()
                if self.cancelled:
                    return f"stopped by the player after {' '.join(pressed) or 'no presses'}"
                await asyncio.to_thread(self.controls.press, b)
                pressed.append(b)
                if i == len(buttons) - 1:
                    break
                if not watch:
                    await asyncio.sleep(gap / 1000)
                    continue
                after = skip_grid(await self._settled(noise))
                box, stopped = skip_on(box, before, after)
                if stopped:
                    break
                before = after
            done = f"pressed {' '.join(pressed)}"
            if stopped:
                done += (f" and stopped skipping there, {len(pressed)} of {len(buttons)}: "
                         f"{stopped}. Look before pressing anything else")
            if ignored:
                done += f" (ignored {', '.join(ignored)}: not buttons)"
            if battle and pressed and pressed[-1] == "a" and not stopped:
                # An attack, a tech or an item takes a second or two to play out: looking
                # at once shows a menu again and the same command goes twice.
                await asyncio.sleep(1.8)
                done += (". A battle command plays out over a second or two: look for what "
                         "it did (health or MP numbers changed, an enemy hit, a message) "
                         "before doing anything again. A command list showing again is "
                         "usually the next turn, not a sign it failed; a single action the "
                         "player asked for is done once confirmed, so finish")
            if buttons == ["a"] and not battle and not str(a.get("say") or "").strip():
                # The player can't read along: a text box skipped is story lost.
                done += (". If a new line of dialogue shows now, say it word for word "
                         "(who, then the line) before pressing again.")
            if held_back:
                done += (". Stopped before confirming: look at what is highlighted now. If it "
                         "is what the player wants, press a on its own; if not, fix it first.")
            return done
        if self.on_action and name != "talk_to":  # its walk_to says it
            self.on_action(name, a)
        if name == "wait":
            secs = max(0.5, min(float(a.get("seconds") or 2), 8.0))
            end = time.monotonic() + secs
            while time.monotonic() < end and not self.cancelled:
                await asyncio.sleep(0.05)  # "stop" ends the wait at once
            return f"waited {secs:g} s"
        if name == "hold":
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            ms = max(50, min(int(float(a.get("ms") or 500)), 5000))
            together = [b for b in (a.get("together") or []) if b in AGENT_BUTTONS]
            if len(together) >= 2:
                ms = max(50, min(int(float(a.get("ms") or 500)), 8000))
                if "a" in together and self.look_before_confirm:
                    return "not held: a held with others confirms what the player can't read"
                await asyncio.to_thread(self.controls.chord, together, ms)
                return f"held {' and '.join(together)} together for {ms} ms"
            inp = a.get("input") or (together[0] if together else "left_stick")
            if ms >= 400 and (inp in ANGLES or inp.endswith("_stick")) and \
                    time.monotonic() > getattr(self, "_map_free_until", 0.0):
                on_map = await self._on_surveyed_map()
                if on_map:
                    return on_map
            if not (inp in ("left_stick", "right_stick") or inp in AGENT_BUTTONS):
                return f"can't hold {inp!r}"
            if inp.endswith("_stick"):
                d = a.get("direction", "up")
                await asyncio.to_thread(self.controls.stick, inp.split("_")[0], d, ms)
                return f"held {inp} {d} for {ms} ms"
            if inp in ANGLES and ms >= 400 and self._walker is not None \
                    and self._walker.memory is not None and await self._still_me(self._walker):
                # Walking by d-pad where it has walked and the character is in view:
                # watch it, so the model learns how far it went or what stopped it.
                return await self._act("keep_walking", {"direction": inp,
                                                        "seconds": ms / 1000})
            if inp in ANGLES:
                # Held as a steer, so "stop" lets go at once instead of after the hold.
                t_hold = time.monotonic()
                end = t_hold + ms / 1000
                try:
                    self.controls.steer(ANGLES[inp])
                    while time.monotonic() < end and not self.cancelled and self.go.is_set():
                        await asyncio.sleep(0.02)
                finally:
                    self.controls.steer(None)
                held = round((time.monotonic() - t_hold) * 1000)
                if held < ms - 60:
                    why = "the player stopped it" if self.cancelled else "the player spoke"
                    return f"held {inp} for only {held} ms of {ms}: {why}"
                return f"held {inp} for {ms} ms"
            if inp == "a" and ms > 400 and self.look_before_confirm:
                return ("not held: holding a confirms and skips text the player can't read; "
                        "press a once and look")
            await asyncio.to_thread(self.controls.press, inp, ms)
            return f"held {inp} for {ms} ms"
        if name in ("battle", "fight"):
            return await self._battle_tool(name, a)
        if name == "field_menu":
            if not FIELD_MENU:
                return "no field_menu yet: use press"
            return await self._field_menu(a)
        if name not in TOOL_NAMES:
            return f"no tool called {name}; use one of {', '.join(sorted(TOOL_NAMES))}"
        if name in ("walk_to", "talk_to", "travel", "walk_to_map", "keep_walking") and \
                time.monotonic() > getattr(self, "_map_free_until", 0.0):
            on_map = await self._on_surveyed_map()
            if on_map:
                return on_map
        if name in ("walk_to", "talk_to") and str(a.get("target") or "").strip():
            # Inside a surveyed place, someone or something it knows is walked to by the
            # place's map (straight there, off screen too), not by sight.
            known = await self._room_target(str(a["target"]), name == "talk_to")
            if known:
                return known
        if name == "talk_to":
            walked = await self._act("walk_to", a)
            if not walked.startswith("arrived"):
                return walked
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            changed = await self._press_a_and_see()
            facing = getattr(self, "_facing", None)
            if changed <= 0.04 and facing is not None and not self.cancelled:
                # Small things on the ground answer only from right beside them: one
                # step closer the way it was facing, and try again.
                self.controls.steer(facing)
                await asyncio.sleep(0.15)
                self.controls.steer(None)
                await asyncio.sleep(0.2)
                changed = await self._press_a_and_see()
                if changed > 0.04:
                    walked += "; stepped one closer"
            if changed <= 0.04:
                return (f"{walked}; pressed a, but nothing came up on screen: they may "
                        "have nothing to say, or it isn't facing them. Look before saying "
                        "anything happened.")
            return (f"{walked}; pressed a, and something came up. If they speak, say the "
                    "line word for word (who, then the line) before pressing again.")
        if name == "walk_to":
            target = str(a.get("target") or "").strip()
            if not target:
                return "say what to walk to"
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            me = str(a.get("me") or "").strip()
            walker = self.walker()
            try:
                await self._find_me(walker)
                walked = await walker.walk(target, me)
                await self._walk_said(walker)
            except WALK_ERRORS as exc:
                # Not the model's arguments: something broke while walking.
                self.controls.steer(None)
                return f"walking failed ({exc}); walk with the d-pad instead"
            steps = walker.travel / walker.size if walker.size else None
            self.trail.add(f"walk_to {target}", walked.outcome, steps)
            return walked.text()
        if name == "exits":
            from .locate import exits_text, find_exits
            from .walk import WIDTH

            img = await self.walker_image()
            found, me = await find_exits(self.muse, img)
            th = img.height * WIDTH / img.width
            step = (16 / WIDTH, 16 / th)  # about a character, when none is tracked
            walker = self._walker
            if walker is not None and walker.memory is not None \
                    and await self._still_me(walker):
                _look, at, (bw, _bh) = walker.memory
                me = (float(at[0]) / WIDTH, float(at[1]) / th)
                step = (bw / WIDTH, bw / th)
            return exits_text(found, me, step, getattr(self, "_came_in", ""))
        if name == "area_map":
            await self.walker_image()  # the screen now: maybe somewhere new since
            if self.area.frames < 2:
                return "no map yet: nothing has been walked in this area"
            self._map_image = self.area.image(marks=None)
            self._map_shown = self._map_gen
            self._show_map = True
            return "the map comes with the next screen"
        if name == "walk_to_map":
            if self._map_image is None:
                return "look at area_map first"
            await self.walker_image()
            if self._map_shown != self._map_gen:
                return "this is a different area from the map you saw: look at area_map again"
            try:
                x = max(0.0, min(float(a.get("x")), 1000.0))
                y = max(0.0, min(float(a.get("y")), 1000.0))
            except (TypeError, ValueError):
                return "x and y must be numbers from 0 to 1000"
            label = str(a.get("label") or "that spot on the map")
            spot = self.area.from_image((x / 1000 * self._map_image.width,
                                         y / 1000 * self._map_image.height))
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            walker = self.walker()
            gen = self._map_gen

            def where():
                # Gone once the map starts over (a door, the view lost): its points with it.
                return tuple(self.area.to_screen(spot)) if self._map_gen == gen else None

            try:
                await self._find_me(walker)
                walked = await walker.walk_toward(where, label)
                await self._walk_said(walker)
            except WALK_ERRORS as exc:
                self.controls.steer(None)
                return f"walking failed ({exc})"
            steps = walker.travel / walker.size if walker.size else None
            self.trail.add(f"walk_to_map {label}", walked.outcome, steps)
            return walked.text()
        if name == "go_to_place":
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            try:
                went = await self._go_to_place(str(a.get("place") or ""))
            except WALK_ERRORS as exc:
                self.controls.steer(None)
                return f"walking failed ({exc})"
            if went.startswith(("no way", "blocked", "no atlas", "no place", "not on")):
                # It can't help here: the other ways of walking are open for a while.
                self._map_free_until = time.monotonic() + 180
            return went
        if name == "go_to_era":
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            try:
                return await self._go_to_era(str(a.get("era") or ""))
            except WALK_ERRORS as exc:
                self.controls.steer(None)
                return f"flying failed ({exc})"
        if name == "go_to_exit":
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            try:
                return await self._go_to_exit(str(a.get("to") or ""),
                                              talk=a.get("talk") if isinstance(
                                                  a.get("talk"), bool) else None)
            except WALK_ERRORS as exc:
                self.controls.steer(None)
                return f"walking failed ({exc})"
        if name == "guide":
            from .guide import look_up

            return look_up(self.game or "", str(a.get("query") or ""))
        if name == "where_am_i":
            return await self._where_am_i()
        if name == "next_goal":
            try:
                step = int(a["step"]) if a.get("step") not in (None, "") else 0
            except (TypeError, ValueError):
                step = 0
            return await self._next_goal(step, str(a.get("player_says") or ""))
        if name == "travel":
            direction = str(a.get("direction") or "")
            if direction not in ANGLES:
                return f"direction must be one of {', '.join(ANGLES)}"
            secs = max(10.0, min(float(a.get("seconds") or 45), 90.0))
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            try:
                return await self._travel(direction, str(a.get("target") or ""), secs)
            except WALK_ERRORS as exc:
                self.controls.steer(None)
                return f"walking failed ({exc}); hold the d-pad instead"
        if name == "keep_walking":
            direction = str(a.get("direction") or "")
            if direction not in ANGLES:
                return f"direction must be one of {', '.join(ANGLES)}"
            secs = max(0.4, min(float(a.get("seconds") or 3), 8.0))
            await self.go.wait()
            if self.cancelled:
                return "stopped by the player"
            walker = self.walker()
            try:
                await self._find_me(walker)
                went = await keep_walking(self.walker_image, self.controls.steer, direction,
                                          secs, self.walker_keep_going, walker.memory,
                                          walker.log)
            except WALK_ERRORS as exc:
                self.controls.steer(None)
                return f"walking failed ({exc}); hold the d-pad instead"
            if went.at is None:
                walker.memory = None
            elif walker.memory is not None:
                walker.memory = (walker.memory[0], went.at, walker.memory[2])
                self.area.character_at(went.at)
            if went.steps.any() or went.outcome not in ("didnt_move", "unsure"):
                self.trail.add(f"keep_walking {direction}", went.outcome, went.steps)
            return went.text(direction.replace("_", " and "))
        return "ok"

    # -- Chrono Trigger (Steam): battles and field menus by the local executor ------------

    def _ct_steam(self, frame: Frame | None) -> bool:
        return frame is not None and "chrono trigger" in (self.game or "").lower() \
            and ctsteam.fits(frame.image)

    async def _battle_on(self, frame: Frame | None) -> State | None:
        """The fight on screen, read from pixels, or None (another game, no fight)."""
        if not self._ct_steam(frame):
            return None
        st = await asyncio.to_thread(ctsteam.read, frame.image)
        if not (st.in_battle or st.paused):
            return None
        if len(st.party) >= 2:
            self._party = [m.name for m in st.party]  # the field menu's blocks, in order
        now = time.monotonic()
        if now - self._battle_seen > 30 and self._executor is not None:
            self._executor.ally_at.clear()  # a new fight: everyone stands elsewhere
        self._battle_seen = now
        return st

    def executor(self) -> Executor:
        if self._executor is None:
            async def press(button: str, ms: int) -> None:
                await self.go.wait()
                await asyncio.to_thread(self.controls.press, button, ms)

            async def chord(buttons: list[str], ms: int) -> None:
                await self.go.wait()
                await asyncio.to_thread(self.controls.chord, buttons, ms)

            async def grab():
                return (await self.grab()).image

            def log(text: str) -> None:
                if self.on_action:
                    self.on_action("battle_note", {"text": text})

            self._executor = Executor(press, grab, identify=self._who_glove,
                                      cancelled=lambda: self.cancelled, chord=chord, log=log)
        return self._executor

    async def _who_glove(self, image, glove, names: list[str]) -> str | None:
        """Ask the model which ally the target glove is on (only when the executor can't
        tell from where each stood on their turn)."""
        try:
            img = image if isinstance(image, Image.Image) else Image.fromarray(
                np.ascontiguousarray(np.asarray(image)[..., ::-1]))
            d = await asyncio.wait_for(self.muse.json([{"role": "user", "content": [
                image_part(img, self.muse.s.image_max_side),
                {"type": "text", "text": WHO_GLOVE.format(names=", ".join(names))}]}],
                effort="low", max_tokens=1500), 20)
        except (openai.APIError, TimeoutError, ValueError, TypeError) as exc:
            if self.on_action:
                self.on_action("battle_note", {"text": f"who_glove failed: {exc}"})
            return None
        return str(d.get("name") or "") or None

    async def _decide(self, st: State, why: str) -> Intent | None:
        """One "win this fight" turn on a surprise: the model picks, from the state in
        words (no picture: quick), while the game is paused."""
        techs = ", ".join(ctsteam.TECHS.get(st.actor, ())) or "unknown"
        prompt = BATTLE_DECIDE.format(request=getattr(self, "_request", "win this fight"),
                                      state=ctsteam.describe(st), why=why or "your call",
                                      actor=st.actor, techs=techs)
        try:
            d = await asyncio.wait_for(self.muse.json(
                [{"role": "user", "content": prompt}], effort="low", max_tokens=1500), 20)
        except (openai.APIError, TimeoutError, ValueError, TypeError) as exc:
            if self.on_action:
                self.on_action("battle_note", {"text": f"decide failed: {exc}"})
            return None
        intent = parse_intent(d, st.actor)
        if intent is not None and intent.command != "attack":
            line = f"{st.actor}: {intent.name or intent.command}" + (
                f" on {intent.target}" if intent.target else "") + "."
            asyncio.get_running_loop().create_task(self._speak(line))
        return intent

    async def _hold_fight(self, st: State) -> State:
        """Pause the fight while the model thinks (Start), if it isn't already."""
        if st.paused:
            return st
        if self.on_action:
            self.on_action("battle_pause", {})  # the game window gets focus first
        try:
            return await self.executor().pause()
        except Stop:
            return st

    def _battle_note(self, lead: str = "") -> str:
        st = self._battle
        if st is None:
            return ""
        return f"{lead}{BATTLE_CT}\nThe fight now: {ctsteam.describe(st)}\n\n"

    def _tools(self, frame: Frame | None) -> list[dict]:
        if self._battle is not None:
            return BATTLE_TOOLS
        if self._ct_steam(frame) and FIELD_MENU:
            return [*TOOLS[:-1], CT_TOOLS[2], TOOLS[-1]]  # field_menu, finish still last
        return TOOLS

    async def _battle_tool(self, name: str, a: dict) -> str:
        await self.go.wait()
        if self.cancelled:
            return "stopped by the player"
        frame = await self.grab()
        if not self._ct_steam(frame):
            return "no battle tool here: it's for Chrono Trigger on Steam; use press"
        ex = self.executor()
        if name == "fight":
            mode = str(a.get("mode") or "win").lower()
            if mode.startswith(("run", "flee", "escape")):
                out = await ex.run_away()
            else:
                def now_and_then(line: str) -> None:
                    # A long fight isn't a silent one: a turn's line after a quiet spell.
                    if time.monotonic() - getattr(self, "_spoke_at", 0.0) >= PROGRESS_GAP_S:
                        self._spoke_at = time.monotonic()
                        asyncio.get_running_loop().create_task(self._speak(line))

                out = await ex.win(self._decide, say=now_and_then)
        else:
            intent = Intent(str(a.get("actor") or ""), str(a.get("command") or "attack"),
                            str(a.get("name") or ""), str(a.get("target") or ""))
            out = await ex.act(intent)
        st = out.state
        now = ctsteam.describe(st) if st is not None else "the screen can't be read"
        pressed = f" (pressed {' '.join(out.presses)})" if out.presses else ""
        held = (" The fight is paused until the player's next request." if st is not None
                and st.paused else "")
        if out.ok:
            return f"done: {out.text}. Now: {now}{held}{pressed}"
        return (f"stopped, nothing more pressed: {out.text}. Now: {now}{held} Fix it once "
                f"if the fix is plain, or tell the player.{pressed}")

    async def _field_menu(self, a: dict) -> str:
        await self.go.wait()
        if self.cancelled:
            return "stopped by the player"
        frame = await self.grab()
        if not self._ct_steam(frame):
            return "no field_menu here: it's for Chrono Trigger on Steam; use press"
        if self._battle is not None:
            return "in a fight: use battle"
        from .menu import FieldMenu

        async def press(button: str, ms: int) -> None:
            await self.go.wait()
            await asyncio.to_thread(self.controls.press, button, ms)

        async def grab():
            return (await self.grab()).image

        fm = FieldMenu(press, grab, cancelled=lambda: self.cancelled,
                       party=getattr(self, "_party", None))
        action = str(a.get("action") or "")
        if action == "setting":
            out = await fm.setting(str(a.get("name") or ""), str(a.get("target") or ""))
        else:
            out = await fm.use(action, str(a.get("name") or ""), str(a.get("user") or ""),
                               str(a.get("target") or ""))
        return out

    def _progress(self):
        from .quests import Progress

        return Progress.load(self.notes.folder, self.game or "")

    async def _the_atlas(self):
        """The game's atlas, loaded once (its map pictures take a moment), or None."""
        from .atlas import Atlas

        game = self.game or ""
        if getattr(self, "_atlas_game", None) != game:
            self._atlas = await asyncio.to_thread(Atlas.for_game, game)
            self._atlas_game = game
        return self._atlas

    async def _on_surveyed_map(self) -> str:
        """When the screen is a world map the atlas has walked (surveyed), the way to
        go anywhere on it is go_to_place: walking by sight there goes astray."""
        atlas = await self._the_atlas()
        if atlas is None:
            return ""
        spot = await self._spot(atlas, (await self.grab()).image)
        if spot is None or spot.margin < 0.1:
            return ""
        m = atlas.map(spot.map)
        if not m.surveyed:
            return ""
        return (f"Not done: this is the {m.name} world map, where walking by sight goes "
                f"astray. Use go_to_place with the place's name ({', '.join(m.places)}); "
                "it walks there and goes in.")

    async def _spot(self, atlas, image):
        """Where the screen is on the atlas's maps, or None (off the world map). A match
        no better than the next best spot is a flat patch of map matching a scene that
        isn't the map (a cutscene of sky and hills): not on it."""
        if atlas is None:
            return None
        # Around the last spot first (a hundredth of a second, not a second or more).
        spot = await asyncio.to_thread(atlas.locate, image, getattr(self, "_last_spot", None))
        spot = spot if spot is not None and spot.margin >= 0.02 else None
        self._last_spot = spot
        if spot is not None:  # the era: of two places alike in two eras, this one's
            self._last_world = getattr(atlas.map(spot.map), "world", None)
            self._world_xy = (spot.x, spot.y)  # (and the door: alike in two towns)
        return spot

    async def _where_am_i(self) -> str:
        """Where the player is on a world map, said from the places around them."""
        from . import atlas as atlas_mod
        from .quests import where_text

        atlas = await self._the_atlas()
        if atlas is None:
            return "no atlas for this game: answer from the screen"
        image = (await self.grab()).image
        spot = await self._spot(atlas, image)
        progress = self._progress()
        if spot is None:
            inside = await self._inside(image)
            if inside is not None:
                from .rooms import describe

                return describe(*inside) + "; go_to_exit goes through any of them."
            last = f" Last on the world map near {progress.visited[-1]}." \
                if progress.visited else ""
            return ("not on a world map (inside a place, a menu, a battle or a fade): "
                    "answer from the screen: the place's name if it shows, else what kind "
                    f"of place it is.{last}")
        m = atlas.map(spot.map)
        text, at = where_text(m, spot.x, spot.y, getattr(atlas_mod, "CELL", 8),
                              getattr(atlas_mod, "RISE", 0))
        if at:
            progress.went(at)
            progress.save(self.notes.folder, self.game or "")
        return f"{text} Places on this map for go_to_place: {', '.join(m.places)}."

    async def _next_goal(self, step: int = 0, player_says: str = "") -> str:
        """What to do next, from the quest list and where the player seems to be in it."""
        from .quests import Guess, Quests, place_in_story, tell

        quests = Quests.for_game(self.game or "")
        if quests is None or not quests.steps:
            return ("no quest list for this game (scripts/build_quests.py makes one from "
                    "the walkthrough): use the guide")
        progress = self._progress()
        if quests.step(step) is not None:
            progress.step, progress.confirmed = step, True
            progress.save(self.notes.folder, self.game or "")
            return tell(quests, Guess(step, "high", "the player confirmed it", ask=False))
        image = (await self.grab()).image
        atlas = await self._the_atlas()
        spot = await self._spot(atlas, image)
        era = near = ""
        if spot is not None:
            m = atlas.map(spot.map)
            era = m.name
            if m.places:
                near = m.nearest(spot.x, spot.y)[0]
        try:
            guess = await place_in_story(self.muse, quests, progress, image, era, near,
                                         player_says, self.game or "")
        except (ValueError, OSError, openai.APIError) as exc:
            return f"couldn't work out where you are in the story ({exc}): use the guide"
        if not guess.ask:
            progress.step = guess.step
            progress.save(self.notes.folder, self.game or "")
        return guess.text

    async def _go_to_place(self, place: str, limit: float = 90.0) -> str:
        """Walk to a named place on a world map by the game's atlas (see atlas.walk_to).
        Which place the player means is Muse's pick from the map's names and the other
        names the walkthrough uses for them, not shared words ("Truce" isn't Truce Inn)."""
        from .atlas import RISE, walk_to
        from .epoch import epoch_ok, era_named, era_of, era_only, left_flying, needs_epoch
        from .quests import Quests, atlas_aliases, pick_place

        atlas = await self._the_atlas()
        if atlas is None:
            return "no atlas for this game: use travel and the guide instead"
        spot = await self._spot(atlas, (await self.grab()).image)
        if spot is None:  # a blink or a cloud: one more look before going without it
            await asyncio.sleep(0.4)
            spot = await self._spot(atlas, (await self.grab()).image)
        quests = Quests.for_game(self.game or "")
        found = None
        if spot is None:
            # Inside a surveyed place: out by its way nearest the place first.
            left = await self._leave_for(atlas, place, quests)
            if left is not None:
                text, found = left
                if not text.startswith("arrived"):
                    return text
                await asyncio.sleep(1.0)  # the fade out onto the map
                spot = await self._spot(atlas, (await self.grab()).image)
        maps = [atlas.map(spot.map)] if spot is not None else atlas.maps
        era = era_named(place) if epoch_ok(atlas) else None
        if era is not None and era_only(place):
            return await self._go_to_era(place)
        if era is not None:  # "Truce Inn in 600 AD": that era's maps
            maps = sorted((m for m in atlas.maps if era_of(m.name) == era),
                          key=lambda m: spot is None or m.name != spot.map) or maps
        near = maps[0].nearest(spot.x, spot.y)[0] if spot is not None and maps[0].places \
            else ""
        if found is None or (spot is not None and found[0].name != spot.map):
            found = await pick_place(self.muse, self.game or "", place, maps,
                                     atlas_aliases(self.game or ""), near,
                                     quests.aliases() if quests else None)
        if isinstance(found, str) and epoch_ok(atlas):
            if era is not None:  # no place in that era fits: go to it, then ask
                went = await self._go_to_era(place)
                names = "; ".join(f"{m.name}: {', '.join(m.places)}" for m in maps if m.places)
                return (f"no single place in {era} for {place!r} ({found}); "
                        f"{went.removeprefix('arrived: ')}. Places there: {names}. Call "
                        "go_to_place with the exact name, or ask the player which")
            if len(maps) < len(atlas.maps):
                # Not on this map: another era's, by the Epoch (this one first).
                maps = maps + [m for m in atlas.maps if m not in maps]
                found = await pick_place(self.muse, self.game or "", place, maps,
                                         atlas_aliases(self.game or ""), near,
                                         quests.aliases() if quests else None)
        if isinstance(found, str):
            names = "; ".join(f"{m.name}: {', '.join(m.places)}" for m in maps if m.places)
            return (f"no single place on the map for {place!r} ({found}). Places: {names}. "
                    "Ask the player which, or call go_to_place with an exact name")
        m, name = found
        door_at = None
        if spot is not None and m.name == spot.map:
            name = nearest_of_kind(m, name, place, (spot.x, spot.y))
            via = await self._through(m, name, (spot.x, spot.y))
            if via is not None:  # Guardia Castle: in by Guardia Forest and out its far end
                return await self._go_through(atlas, m, name, spot, *via)
            rnd = await asyncio.to_thread(way_round, atlas, m, await self._rooms_data(), name,
                                          (spot.x, spot.y - RISE))
            if rnd is not None and rnd[0] == name:
                door_at = rnd[1]  # in by its other door, on this side
            elif rnd is not None:  # and back from the castle's side the same way
                return await self._go_through(atlas, m, name, spot, rnd[0], rnd[2], rnd[1])
        walker = self.walker()

        async def shoot():
            return (await self.grab()).image

        async def keep_going():
            return await self.walker_keep_going()

        # Said at once, while it starts walking: the player can't see it set off (unless
        # it was said on the way out of a place).
        told = asyncio.create_task(self._speak(
            "" if getattr(self, "_going_to", None) == name else f"Going to {name}."))
        self._going_to = None
        if spot is not None and (m.name != spot.map or
                                 left_flying(self.notes.folder, self.game or "")):
            result = await self._pilot(atlas).travel(m, name)  # another era, or in the air
        else:
            result = await walk_to(atlas, m, name, shoot, self.controls.steer,
                                   self.controls.press, keep_going, walker.log, limit,
                                   spot, at=door_at)
            if result.startswith(("blocked", "no way")) and spot is not None:
                # "The market" whose way is shut at this point in the story: the next
                # nearest market, before flying anywhere.
                said = getattr(self, "_request", "") or place  # the player's, not the model's
                for other in others_of_kind(m, name, said, (spot.x, spot.y))[:2]:
                    if not m.opens(other) or atlas.route(
                            m, (spot.x, spot.y - RISE), m.places[other],
                            m.entrance(other)) is None:
                        continue  # shut, or no way there either (another continent)
                    await told
                    told = asyncio.create_task(self._speak(
                        f"Can't get to {name} from here. Going to {other} instead."))
                    again = await walk_to(atlas, m, other, shoot, self.controls.steer,
                                          self.controls.press, keep_going, walker.log,
                                          limit)
                    if again.startswith("arrived"):
                        name, result = other, again
                        break
            if needs_epoch(result) and epoch_ok(atlas):
                # Another continent: fly the Epoch, when the party has it.
                flew = await self._pilot(atlas).travel(m, name)
                result = flew if not flew.startswith("no Epoch") else \
                    f"{result}. Flying there: {flew}"
        await told
        self._last_spot = None  # moved on
        if result.startswith("arrived"):
            progress = self._progress()
            progress.went(name)
            progress.save(self.notes.folder, self.game or "")
            await self._speak(f"We're at {name}.")
            result += (f'. The player has heard "Going to {name}." and "We\'re at {name}." '
                       "Add one short line only if something on screen needs them (someone "
                       "talking, a choice); otherwise finish.")
        return result

    def _pilot(self, atlas):
        """The Epoch's pilot (epoch.Pilot), with this agent's eyes and hands."""
        from .epoch import Pilot

        async def shoot():
            return (await self.grab()).image

        async def keep_going():
            return await self.walker_keep_going()

        return Pilot(atlas, self.game or "", shoot, self.controls.steer, self.controls.press,
                     keep_going, self.walker().log, self.notes.folder, find=self._see_epoch)

    async def _see_epoch(self, image):
        """Where the parked Epoch is on a picture (fractions of it), by Muse: for art
        the atlas has no sprite for (the Steam port)."""
        try:
            data = await self.muse.json([{"role": "user", "content": [
                image_part(image, 1280), {"type": "text", "text": SEE_EPOCH}]}],
                effort="low", max_tokens=600)
            x, y = (float(v) for v in data.get("epoch") or ())
        except (ValueError, TypeError, OSError, openai.APIError):
            return None
        return (x / 1000, y / 1000) if 0 <= x <= 1000 and 0 <= y <= 1000 else None

    async def _go_to_era(self, said: str) -> str:
        """Go to another era by the Epoch's time gauge (see epoch.Pilot.go_to_era)."""
        from .epoch import GAUGE, epoch_ok, era_named

        era = era_named(said)
        if era is None:
            return f"no era called {said!r}: the time gauge has {', '.join(GAUGE)}"
        atlas = await self._the_atlas()
        if atlas is None or not epoch_ok(atlas):
            return "this game's atlas has no Epoch (time machine) to fly"
        told = asyncio.create_task(self._speak(f"Off to {era.rstrip('.')}."))
        result = await self._pilot(atlas).go_to_era(era)
        await told
        return result

    async def _rooms_data(self):
        """The room survey (rooms.Rooms) for this game, loaded once, or None."""
        if getattr(self, "_rooms", None) is None:
            from .rooms import Rooms

            self._rooms = await asyncio.to_thread(Rooms.for_game, self.game or "")
        return self._rooms

    async def _through(self, m, name: str, at: tuple[float, float] | None = None
                       ) -> tuple[str, str] | None:
        """A place whose icon doesn't open, reached through another place on the map:
        (that place, its way out by this one), from the room survey."""
        import math

        if m.opens(name) or getattr(m, "world", None) is None:
            return None
        if getattr(self, "_rooms", None) is None:
            from .rooms import Rooms

            self._rooms = await asyncio.to_thread(Rooms.for_game, self.game or "")
        if self._rooms is None:
            return None
        goal = m.places[name]
        best = None
        for room in self._rooms.rooms.values():
            via = next((p for p in m.places if p != name and (cells := m.entrance(p))
                        and any(m.doors.get(c) == room.loc for c in cells)), None)
            if via is None:
                continue
            for e in room.exits:
                if e.to == m.world and any(e.at) and math.dist(e.at, goal) < 80:
                    d = math.dist(e.at, goal)
                    if best is None or d < best[0]:
                        best = (d, via, e.label)
        if best is None or (at is not None and math.dist(at, goal) < best[0] + 40):
            return None  # already on its side (come out of the forest's far end)
        return best[1], best[2]

    async def _go_through(self, atlas, m, name: str, spot, via: str, way_out: str,
                          door_at: tuple[float, float] | None = None) -> str:
        """Into `via` on the map (at `door_at`, when it's another door of it than its
        named one), out of it by `way_out`, and on to `name`."""
        from .atlas import walk_to

        walker = self.walker()

        async def shoot():
            return (await self.grab()).image

        async def keep_going():
            return await self.walker_keep_going()

        told = asyncio.create_task(self._speak(f"Going to {name}, through {via}."))
        try:
            first = await walk_to(atlas, m, via, shoot, self.controls.steer,
                                  self.controls.press, keep_going, walker.log, 90.0, spot,
                                  at=door_at)
            if not first.startswith("arrived"):
                return f"on the way to {name}, which is reached through {via}: {first}"
            await asyncio.sleep(1.0)  # the fade into it
            out = await self._go_to_exit(way_out, 180.0)  # a fight or two on the way
            if not out.startswith("arrived"):
                return (f"in {via} on the way to {name} (out by {way_out!r}, then a short "
                        f"walk): {out}")
            await asyncio.sleep(1.0)
            self._last_spot = None
            last = await walk_to(atlas, m, name, shoot, self.controls.steer,
                                 self.controls.press, keep_going, walker.log, 60.0)
        finally:
            await told
        if last.startswith("arrived"):
            await self._speak(f"We're at {name}.")
            last += (f'. The player has heard "Going to {name}, through {via}." and "We\'re '
                     f'at {name}." Add one short line only if something on screen needs '
                     "them; otherwise finish.")
        return f"through {via}, then {last}"

    async def _room_target(self, target: str, talk: bool) -> str:
        """walk_to or talk_to inside a surveyed place: to someone or something in it (or
        the next place) the survey knows, by the place's map. "" when it isn't inside a
        surveyed place or nothing known fits (then by sight)."""
        from .rooms import pick_exit, walk_out

        inside = await self._inside((await self.grab()).image)
        if inside is None:
            return ""
        rooms, here = inside
        choices, targets, hints = rooms.options(here)
        if not targets:
            return ""
        label, _why = await pick_exit(self.muse, self.game or "", target,
                                      rooms.name(here.loc),
                                      {k: choices[k] for k in targets},
                                      {k: hints.get(k, "") for k in targets})
        if label is None:
            return ""
        t = targets[label]
        walker = self.walker()

        async def shoot():
            return (await self.grab()).image

        told = asyncio.create_task(self._speak(f"Going to {t.name}."))
        try:
            out = await walk_out(rooms, t.loc, shoot, self.controls.steer,
                                 self.walker_keep_going, walker.log, 60.0, here, "", t,
                                 self.controls.press, talk)
        finally:
            await told
        if out.startswith("arrived"):
            walker.memory = None
        return f"{out} (by the place's map: {label})"

    async def _leave_for(self, atlas, place: str, quests) -> tuple[str, tuple] | None:
        """Inside a surveyed place, for a place on the world map outside it: walk out by
        the way out nearest that place. (what happened, (map, name)), or None when not
        inside a surveyed place or the place isn't on a map it leads out to."""
        import math

        from .quests import atlas_aliases, pick_place
        from .rooms import describe, walk_out

        inside = await self._inside((await self.grab()).image)
        if inside is None:
            return None
        rooms, here = inside
        outs: dict[str, tuple[int, tuple[float, float]]] = {}  # label: (world, where)
        seen, todo = {here.loc}, [here.loc]
        while todo:
            u = todo.pop()
            for e in rooms.rooms[u].exits if u in rooms.rooms else []:
                if e.to in rooms.worlds:
                    if any(e.at):
                        outs.setdefault(e.label, (e.to, e.at))
                elif e.to not in seen:
                    seen.add(e.to)
                    todo.append(e.to)
        maps = [m for m in atlas.maps if getattr(m, "world", None) in {w for w, _ in
                                                                        outs.values()}]
        if not maps:
            return None
        found = await pick_place(self.muse, self.game or "", place, maps,
                                 atlas_aliases(self.game or ""), "",
                                 quests.aliases() if quests else None)
        if isinstance(found, str):
            return None
        m, name = found
        if any(m.doors.get(c) in seen for c in m.entrance(name) or ()):
            said = (f"already in {name}: {describe(rooms, here)}. go_to_exit goes to a "
                    "way out, a thing or someone in here")
            return said, found
        goal = m.places[name]
        label = min((k for k, (w, _) in outs.items() if w == m.world),
                    key=lambda k: math.dist(outs[k][1], goal))
        walker = self.walker()

        async def shoot():
            return (await self.grab()).image

        await self._speak(f"Going to {name}, out of here first.")
        self._going_to = name
        out = await walk_out(rooms, m.world, shoot, self.controls.steer,
                             self.walker_keep_going, walker.log, 90.0, here, label)
        if not out.startswith("arrived"):
            self._going_to = None
            return (f"on the way out of {rooms.name(here.loc)} to {name}: {out}", found)
        walker.memory = None
        return out, found

    async def _inside(self, image):
        """(rooms, where) when the screen is inside a place the room survey knows."""
        from .rooms import Rooms

        if getattr(self, "_rooms", None) is None:
            self._rooms = await asyncio.to_thread(Rooms.for_game, self.game or "")
        if self._rooms is None:
            return None
        self._rooms.world_hint = getattr(self, "_last_world", None)
        self._rooms.world_spot = getattr(self, "_world_xy", None)
        here = await asyncio.to_thread(self._rooms.locate, image)
        return (self._rooms, here) if here is not None else None

    async def _go_to_exit(self, to: str, limit: float = 90.0, talk: bool | None = None
                          ) -> str:
        """Walk out of the place on screen to where the player means, or to a thing or
        someone in it or the next place, by the surveyed rooms (see rooms.walk_out): Muse
        picks from those reachable from here."""
        from .rooms import TALK, Target, pick_exit, walk_out

        inside = await self._inside((await self.grab()).image)
        if inside is None:
            return ("this place isn't in the room survey (or this isn't inside a place), so "
                    "its layout isn't known: if what they want is on screen, walk_to or "
                    "talk_to it; otherwise tell the player plainly that you don't know the "
                    "way around this place yet and what you can see. Don't walk about "
                    "trying ways.")
        rooms, here = inside
        choices, targets, hints = rooms.options(here)
        if not choices:
            return f"in {rooms.name(here.loc)}, but no ways out are surveyed: use exits"
        label, why = await pick_exit(self.muse, self.game or "", to, rooms.name(here.loc),
                                     choices, hints)
        if label is None:
            return (f"no single place from here fits {to!r} ({why}). From here: "
                    f"{'; '.join(choices)}. A place outside on the world map (another "
                    "building, a town): go_to_place, which walks out of here first. Else "
                    "ask the player which, or call go_to_exit with an exact name")
        target: Target | None = targets.get(label)
        if talk is None:
            talk = target is not None and (why.startswith("talk:") or bool(TALK.search(to)))
        walker = self.walker()

        async def shoot():
            return (await self.grab()).image

        name = target.name if target is not None else label
        told = asyncio.create_task(self._speak(
            f"Going to {name}." if target is not None or not label.startswith("outside")
            else "Going outside."))
        try:
            result = await walk_out(rooms, choices[label][0], shoot, self.controls.steer,
                                    self.walker_keep_going, walker.log, limit, here,
                                    "" if target is not None else label, target,
                                    self.controls.press, bool(talk))
        finally:
            await told
        if result.startswith("arrived"):
            walker.memory = None  # somewhere else now
            if target is not None and target.person and any(
                    p.moves and p.name == target.name
                    for p in rooms.rooms[target.loc].people):
                result += " (they walk about: if they're not there, look for them)"
        return result

    async def _travel(self, direction: str, target: str, secs: float) -> str:
        """Head `direction` across a scrolling map until `target` is in sight. The view
        follows the character there, so how far the scenery scrolls says how far it went
        (no need to pick the character out), and a way that's shut is slid along."""
        walker = self.walker()
        walker.memory = None  # the character is where the view keeps it: the middle
        t0 = time.monotonic()
        angle = ANGLES[direction]
        unit = np.array([np.sin(np.radians(angle)), -np.cos(np.radians(angle))])
        # The two ways off the line, to slide along whatever blocks it.
        sides = [d for d, v in ANGLES.items() if abs(((v - angle + 180) % 360) - 180) == 90]
        total, stalls, looks = np.zeros(2), 0, 0

        def far() -> str:
            return f"went {words(total)} in all ({time.monotonic() - t0:.0f} s)"

        while time.monotonic() - t0 < secs:
            if self.cancelled:
                return "stopped by the player"
            went = await keep_walking(self.walker_image, self.controls.steer, direction,
                                      3.0, self.walker_keep_going, None, walker.log)
            total += went.steps
            if went.outcome in ("new_area", "interrupted", "stopped"):
                return went.text(direction.replace("_", " and ")) + f"; {far()}"
            if float(went.steps @ unit) < 1:
                stalls += 1
                if stalls > 2 * len(sides) + 1:
                    return (f"blocked: the way {direction.replace('_', ' ')} is shut here "
                            f"(slid both ways); {far()}. Try another direction or a path")
                side = sides[(stalls - 1) % len(sides)]
                slid = await keep_walking(self.walker_image, self.controls.steer, side,
                                          1.5 + 0.5 * stalls, self.walker_keep_going, None,
                                          walker.log)
                total += slid.steps
                if slid.outcome in ("new_area", "interrupted", "stopped"):
                    return slid.text(side.replace("_", " and ")) + f"; {far()}"
                continue
            stalls = 0
            if target:
                looks += 1
                seen = await self._sighted(target)
                if seen:
                    return (f"in_sight: {target} is on screen now, {seen}; walk_to it. "
                            f"{far()}")
        return f"walked: no {target or 'sign of it'} in sight yet; {far()}"

    async def _sighted(self, target: str) -> str:
        """Where `target` is on the screen now, in a few words, or "" if it isn't there:
        one quick look, strict about near misses."""
        from .muse import image_part

        img = await self.walker_image()
        try:
            data = await self.muse.json([{"role": "user", "content": [
                image_part(img, 1024),
                {"type": "text", "text": (
                    f'Is this on the screen: "{target}"? Only if it clearly matches that '
                    "description; something merely like it (a town for a castle, a "
                    "house for a shop) doesn't count. Answer JSON: {\"there\": true or "
                    "false, \"where\": \"where on screen, relative to the middle, a "
                    "few words\"}")}]}], effort="minimal", max_tokens=600)
        except (ValueError, OSError):
            return ""
        return str(data.get("where") or "in view") if data.get("there") is True else ""

    async def _still_me(self, walker: Walker) -> bool:
        """Whether the character last walked is still in view near where it was (it may
        have been moved by hand, or gone through a door); updates where it is."""
        rgb = small(await self.walker_image())[0]
        if walker.memory is None:  # a new place, seen just now
            return False
        look, at, (bw, bh) = walker.memory
        x, y, score = look.find(rgb, (at[0], at[1]), 3 * max(bw, bh))
        walker.memory = (look, np.array([x, y]), (bw, bh)) if score > 0.6 else None
        return walker.memory is not None

    async def _find_me(self, walker: Walker) -> None:
        """New to this area: which character is the player's, by moving it a little. The
        model often can't tell (two people with red hair, a mother at the bedside)."""
        if walker.memory is not None:
            await self._still_me(walker)
        if walker.memory is None:
            walker.memory = await find_me(self.walker_image, self.controls.steer,
                                          self.walker_keep_going, walker.log)
        if walker.memory is not None:
            self.area.character_at(walker.memory[1])

    async def walker_image(self):
        img = (await self.grab()).image
        # Every frame a walk looks at goes on the map; it starts over in a new place.
        rgb, g = small(img)
        gap = time.monotonic() - self._map_fed > 0.3
        self._map_fed = time.monotonic()
        try:
            if gap:
                # The view may have moved meanwhile (buttons pressed, a door): find it.
                fresh = await asyncio.to_thread(self.area.resume, rgb, g)
            else:
                # Off the event loop, so listening and speech keep up while walking.
                fresh = await asyncio.to_thread(self.area.track, rgb, g)
        except (ValueError, IndexError) as exc:
            self.area.reset()  # never let the map stop a walk
            fresh = True
            if self._walker is not None:
                self._walker.log(f"area map reset ({exc})")
        if fresh:
            self._map_gen += 1  # a new place: points on the old map mean nothing
            # Somewhere else now: where the character was and the walks so far belong
            # to the last place (a walk through a door also says how it got here).
            if self._walker is not None:
                self._walker.memory = None
            self.trail.clear("entered this area")
        now = self._walker._now if self._walker is not None else None
        if now is not None:
            self.area.character_at(now[1])
        return img

    async def walker_keep_going(self) -> bool:
        if not self.go.is_set():
            self.controls.steer(None)  # let go while the player is talking
            await self.go.wait()
        return not self.cancelled

    def walker(self) -> Walker:
        """One walker for the whole session, so what it learns about the camera stays."""
        if self._walker is None:
            def note(text: str) -> None:
                if self.on_action:
                    self.on_action("walk_note", {"text": text})

            async def exact(img, phrase):
                from .locate import segment

                return [m.box for m in await segment(self.muse, img, phrase) if m.box]

            def steer(angle):
                if angle is not None:
                    self._facing = angle  # the way it last walked, to step closer by
                self.controls.steer(angle)

            self._walker = Walker(self.muse, self.walker_image, steer,
                                  self.walker_keep_going, note, say=self._speak, exact=exact)
        return self._walker

    def _check_walk(self, name: str, result: str) -> str:
        """Two walks in a row that didn't get there: say plainly that way is cut off, so
        the model stops rewording the same target."""
        if name not in ("walk_to", "talk_to", "walk_to_map"):
            return result
        if result.startswith(WALK_PAUSED):
            return result
        if not result.startswith(WALK_FAILED):
            self._failed_walks = 0
            return result
        self._failed_walks += 1
        if self._failed_walks >= 2:
            result += (". That's two walks in a row that didn't get there: that way is cut "
                       "off from here. Don't aim at it again, however you name it; look at "
                       "area_map, or keep_walking a direction you haven't tried, or tell the "
                       "player.")
        return result

    async def run(self, request: str, frame: Frame | None = None) -> Run:
        t0 = time.monotonic()
        self._failed_walks = 0
        run = Run(request)
        self._request = request  # the player's own words (tools get the model's)
        self.cancelled = False
        self.said: list[str] = []
        self._said_before_act = 0  # how many lines were said before the last action
        self._wander_s = 0.0
        frame = frame or await self.grab()
        tried = ""
        trip = TRIP.fullmatch(" ".join(request.lower().split()).strip(" .!?"))
        if trip and await self._on_surveyed_map():
            # "Take me to the fair" on a world map the atlas knows: go now, without a
            # model turn first (the player hears it set off in a second or two).
            result = await self._act("go_to_place", {"place": trip.group("place")})
            run.steps.append(Step("go_to_place", {"place": trip.group("place")},
                                  time.monotonic() - t0, result))
            if result.startswith("arrived") or self.cancelled:
                run.outcome = self.said[-1] if self.said else result
                run.seconds = time.monotonic() - t0
                self.history.append(f"{request} -> {run.outcome}")
                return run
            frame = await self.grab()
            tried = f"go_to_place({trip.group('place')!r}) was tried already: {result}\n\n"
        # Chrono Trigger on Steam in a fight: pause it while the model thinks.
        self._battle = await self._battle_on(frame)
        if self._battle is not None:
            self._battle = await self._hold_fight(self._battle)
            said = " ".join(request.lower().split())
            mode = "win" if FIGHT.search(said) else "run" if RUN.search(said) else ""
            if mode:
                # "Win this fight", "run": no model turn needed to start it.
                result = await self._act("fight", {"mode": mode})
                run.steps.append(Step("fight", {"mode": mode}, time.monotonic() - t0, result))
                if self.cancelled:
                    run.outcome = "cancelled"
                    run.seconds = time.monotonic() - t0
                    self.history.append(f"{request} -> {run.outcome}")
                    return run
                frame = await self.grab()
                # Won or got away: it's over, even while the spoils still look like a fight
                # (live, twice in six: the model called fight again and said there was none).
                self._battle = None if result.startswith("done:") else \
                    await self._battle_on(frame)
                tried += (f"fight({mode}) was done already: {result}\nTell the player how "
                          "it went in a sentence, from that result, and finish.\n\n")
        recent = "\n".join(f"- {h}" for h in self.history[-4:])
        trail = self.trail.notes()
        msgs: list[dict] = [
            {"role": "system", "content": self.system()},
            {"role": "user", "content": [
                image_part(frame.image, self.muse.s.image_max_side),
                {"type": "text", "text": (f"Earlier requests:\n{recent}\n\n" if recent else "")
                 + (f"{trail}\n\n" if trail else "") + tried + self._battle_note()
                 + f'The player says: "{request}"'},
            ]},
        ]
        # The player's answer to a question sets a new course: it gets turns of its own.
        budget, _turn, out_of_turns = self.max_turns, -1, True
        while _turn + 1 < budget:
            _turn += 1
            if self.cancelled:
                run.outcome = "cancelled"
                out_of_turns = False
                break
            t_model = time.monotonic()
            last_turn = _turn == budget - 1
            msg = await self.muse.tools(trim_images(msgs), self._tools(frame) if not last_turn
                                        else FINISH_OR_ASK if self.ask else FINISH_ONLY)
            turn = {"model_s": round(time.monotonic() - t_model, 2)}
            run.turns.append(turn)
            await self.go.wait()
            if self.cancelled:
                run.outcome = "cancelled"
                out_of_turns = False
                break
            calls = msg.tool_calls or []
            msgs.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [c.model_dump() for c in calls] or None})
            if not calls:
                if msg.content:
                    await self._speak(msg.content)
                    run.outcome = msg.content
                    out_of_turns = False
                    break
                # Nothing at all (its thinking ran out of room): ask again rather than
                # end in silence.
                msgs.append({"role": "user", "content": "You replied with nothing. Call a "
                             "tool; if you're done, finish with what to tell the player."})
                continue
            self._progress_line(msg.content or "", calls, t0)
            acted = False
            finished = None
            noise = None
            if any(c.function.name in ("press", "hold", "walk_to", "talk_to", "walk_to_map",
                                       "keep_walking", "travel")
                   for c in calls):
                noise = await self._noise()
            t_act = time.monotonic()
            for c in calls:
                if self.cancelled:
                    break
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                step = Step(c.function.name, args, time.monotonic() - t0)
                run.steps.append(step)
                if finished is not None:
                    result = "ignored: it came after finish"
                elif last_turn and c.function.name not in ("finish", "ask_player"):
                    result = ("not done: you're out of steps; ask_player whether to keep "
                              "going, or finish" if self.ask else
                              "not done: only finish is left now")
                elif c.function.name == "finish" and acted:
                    # It hasn't seen what its presses did: no telling the player yet.
                    result = "not finished: look at the screen after those actions first"
                elif c.function.name == "finish" and not str(args.get("text") or "").strip() \
                        and len(self.said) <= self._said_before_act and not last_turn:
                    result = "not finished: give the text to tell the player what happened"
                elif c.function.name == "say" and acted:
                    # Nor say what happened before seeing it.
                    result = "not said: look at the screen after those actions first"
                elif c.function.name == "say":
                    text = str(args.get("text") or "")
                    if self._already_said(text):
                        result = "you just said that; don't say it again"
                    else:
                        await self._speak(text)
                        result = "said"
                elif c.function.name == "ask_player":
                    question = str(args.get("question") or "").strip()
                    if not question:
                        # An empty question would play silence and wait for nothing.
                        result = ("ask_player needs the question itself, in the "
                                  "question field. Ask again with it, or carry on.")
                    elif self.ask is None:
                        await self._speak(question)
                        result = "The player can't answer right now. Don't do it; finish."
                    else:
                        answer = await self.ask(question)
                        result = f'The player said: "{answer}"' if answer else (
                            "No answer. Don't do it; finish and say you'll wait.")
                        if answer:
                            budget = max(budget, _turn + 1 + self.max_turns // 2)
                elif c.function.name == "remember":
                    result = self.notes.add(self.game, str(args.get("note") or ""),
                                            str(args.get("instead_of") or ""))
                elif c.function.name == "finish":
                    finished = (args.get("text") or "").strip()
                    result = "finished"
                elif (self._wandering(c.function.name, args) or 0.0) >= WANDER_LIMIT_S:
                    result = (f"not done: you've walked around for {WANDER_LIMIT_S:.0f} s "
                              "by trial and error without getting there. Stop now: finish, "
                              "and tell the player plainly that you don't know the way "
                              "through this place yet and where they are. No more tries.")
                else:
                    wander_t0 = time.monotonic()
                    try:
                        result = await self._act(c.function.name, args)
                    except (TypeError, ValueError, KeyError) as exc:
                        result = f"bad arguments ({exc}); fix them and try again"
                    if self._wandering(c.function.name, args) is not None:
                        self._wander_s += time.monotonic() - wander_t0
                    acted = True
                    self._said_before_act = len(self.said)
                    result = self._check_walk(c.function.name, result)
                    if (c.function.name in ("go_to_place", "go_to_exit", "go_to_era")
                            and result.startswith("arrived")) or result.startswith("new_area"):
                        # A new place: the wandering budget is for finding the way in one
                        # place (live, 2026-11-01: 40 s spent looking round the fair left
                        # none for walking up to the counter in Truce Market).
                        self._wander_s = 0.0
                    if result.startswith("new_area"):
                        # Which way it was going when it got here: the way back is behind.
                        self._came_in = (str(args.get("direction") or "")
                                         if c.function.name in ("keep_walking", "travel")
                                         else way(getattr(self, "_facing", None)))
                step.result = result
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": result})
            if self.cancelled:
                run.outcome = "cancelled"
                out_of_turns = False
                break
            if finished is not None:
                if not finished and last_turn and len(self.said) <= self._said_before_act:
                    finished = ("I stopped there after a lot of steps. Say describe to hear "
                                "where things stand.")
                if finished and not self._already_said(finished):
                    await self._speak(finished)
                # Kept for the next request's history even when say already said it.
                run.outcome = finished or (self.said[-1] if self.said else "done")
                out_of_turns = False
                break
            if last_turn and self.ask is not None and not self.cancelled and not any(
                    c.function.name == "ask_player" for c in calls):
                # Out of steps and not done: the player says whether it's worth more.
                answer = await self.ask("This is taking a lot of steps and I'm not there "
                                        "yet. Should I keep going?")
                if answer and not re.match(r"\s*(no|nope|stop|don't|do not|cancel)\b",
                                           answer.lower()):
                    budget = max(budget, _turn + 1 + self.max_turns // 2)
                    msgs.append({"role": "user", "content": (
                        "You were out of steps, so the player was asked whether to keep "
                        f'going. They said: "{answer}". Carry on.')})
            turn["act_s"] = round(time.monotonic() - t_act, 2)
            t_look = time.monotonic()
            # After a say too: the model may be waiting for text still typing out.
            if self._battle is not None:
                # A fight never sits still; the executor has already waited for what
                # it needed, and left it paused.
                frame = await self.grab()
            else:
                frame = await self._settled(noise)
            fought = self._battle is not None
            self._battle = await self._battle_on(frame)
            if self._battle is not None and not fought:
                self._battle = await self._hold_fight(self._battle)  # one just started
            turn["look_s"] = round(time.monotonic() - t_look, 2)
            if os.environ.get("PLAYER_TWO_FRAMES"):  # what it saw, turn by turn (debugging)
                with contextlib.suppress(OSError, AttributeError):
                    d = Path(os.environ["PLAYER_TWO_FRAMES"]).expanduser()
                    d.mkdir(parents=True, exist_ok=True)
                    frame.image.convert("RGB").save(
                        d / f"{time.strftime('%H%M%S')}-{_turn:02d}.jpg", quality=80)
            last = _turn == budget - 2
            shown = []
            if self._show_map and self._map_image is not None:
                self._show_map = False
                shown = [image_part(self._map_image, 1024), {"type": "text", "text": (
                    "The map of this area so far: dark grey is not seen yet, the yellow box "
                    "is the screen now, the cyan line is where you walked and its dot is "
                    "you. People shown may have moved since. For walk_to_map, x and y are "
                    "on this map, 0 to 1000 each way.")}]
            msgs.append({"role": "user", "content": [
                *shown,
                image_part(frame.image, self.muse.s.image_max_side),
                {"type": "text", "text": "The screen now." + self._battle_note(" ") + (
                    " You're out of steps. Done: finish. Not done: "
                    + ("ask_player, saying in a sentence how far you got and asking whether "
                       "to keep going (a yes gives you more steps); if it can't be done, "
                       if self.ask else "")
                    + "finish, telling the player plainly how far you got, where things "
                    "stand on screen, and what they could ask next." if last else "")},
            ]})
        if out_of_turns:
            await self._speak("I couldn't finish that one. Want me to try another way?")
            run.outcome = "gave up"
        run.seconds = time.monotonic() - t0
        self.history.append(f"{request} -> {run.outcome}")
        return run
