"""Voice commands: look at the screen, press buttons, look again, until it's done."""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import math
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

from . import ctsteam, ontopic
from .ack import FIGHT, RUN
from .areamap import AreaMap
from .battle import (
    Executor,
    Foes,
    Intent,
    Stop,
    act_line,
    foe_name,
    heal_line,
    heal_plan,
    heals,
    parse_intent,
    party_brief,
    plural,
    revives,
    victory,
)
from .controls import BUTTONS, DIRECTIONS, Controls
from .ctsteam import State
from .explore import ANGLES, Trail, find_me, keep_walking, words
from .muse import Muse, image_part
from .notes import GameNotes
from .prompts import (
    AGENT,
    BATTLE_CT,
    BATTLE_DECIDE,
    CONFIRM,
    FOES,
    FOES_NAMED,
    GAME_CONTEXT,
    LANGUAGE,
    WHO_GLOVE,
)
from .screen import Frame, thumb
from .walk import Walker, small

# Never the guide button: it opens the Steam overlay or the system menu, not the game.
# Seconds of silence in a run before the model's own line about its plan is spoken.
PROGRESS_GAP_S = 8.0
# "Win this fight": an attack's line ("Crono attacks.") after this long with nothing said
# (live 3: lines 8 s apart at most left silences of 4 to 14 s in a fight).
FIGHT_LINE_GAP_S = 4.0
# Tools that press nothing: after them the screen is looked at once, not waited on.
STILL_TOOLS = {"where_am_i", "next_goal", "guide", "exits", "area_map", "remember"}
# Walking around by trial and error in one request stops after this long: wandering on
# camera looks lost, and an honest "I don't know the way here yet" is better.
FOES_SIDE = 1536  # the enemy count's picture, longest side (see count_foes)
# Words that add nothing to a closing line ("We're inside the Truce Inn." after "We're at
# Truce Inn."), as ontopic.content stems them.
FILLER = {ontopic.stem(w) for w in (
    "inside", "outside", "here", "now", "arrived", "made", "there", "finally", "right",
    "safely", "safe", "back", "all", "set", "done", "ready", "got", "okay", "standing")}
# A line saying what's about to happen, not what has.
INTENT = re.compile(r"(?i)\s*(going|heading|walking|taking|trying|let me|i'll|i will|"
                    r"i'm going|okay|ok|on it|got it, i'll)\b")
# Said when the model's lines were off the point twice in one request (ontopic.py).
LOST_THREAD = "Sorry, I lost my train of thought there. Could you ask me that again?"
READ_REQUEST = re.compile(r"(?i)\bread\b|\bwhat (?:does|do|did) (?:it|that|this|they) say\b"
                          r"|\bwhat(?:'s| is) (?:it|that|this) say\b")
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
    {"type": "function", "function": {
        "name": "shop",
        "description": "Chrono Trigger, outside battle: buy from a seller, or rest at an "
                       "inn, done for you. buy walks to the seller (or uses the shop or "
                       "their words already on screen), talks, opens Buy, picks the item, "
                       "sets the count, buys, checks the gold and backs out. rest talks to "
                       "the innkeeper, pays, sleeps and checks the party's HP and MP. Call "
                       "it straight away, without walking or pressing first.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["buy", "rest"]},
            "item": {"type": "string", "description": "For buy: the item as the game "
                     "spells it (Potion, Ether, Athenian Water)."},
            "count": {"type": "integer", "description": "For buy: how many (default 1)."},
            "seller": {"type": "string", "description": "Who sells it or runs the inn, as "
                       "the place's people are named (the item seller at the purple tent, "
                       "the shopkeeper, the innkeeper). Empty when the shop is already up."},
        }, "required": ["action"]}}},
]
FIELD_MENU = True  # the field_menu tool is offered (menu.py)
BATTLE_TOOLS = [*CT_TOOLS[:2], *(t for t in TOOLS if t["function"]["name"] in (
    "say", "ask_player", "remember", "wait", "finish"))]
FINISH_ONLY = [t for t in TOOLS if t["function"]["name"] == "finish"]
TOOL_TEXT = json.dumps([*TOOLS, *CT_TOOLS])  # the tools as the model reads them
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
# "Heal Frog", "Marle, heal everyone": in a fight, chosen and done without a model turn.
HEAL = re.compile(r"(?:(?P<actor>crono|marle|lucca|robo|frog|ayla|magus),? )?"
                  r"(?:please |can you |could you )?(?:heal|cure) (?P<who>.{2,40})")
# Commands in one request ("heal Frog and attack"): a fight gets that many battle calls.
COMMANDS = re.compile(r"\b(attack|hit|heal|cure|use|cast|revive|raise)\b")
OUT = re.compile(r"(?:(?:please |can you |could you )?(?:take|bring|get) (?:me|us) |"
                 r"(?:let'?s |please )?(?:go|head|walk) )(?:back )?(?:out|outside)"
                 r"(?: of here| again)?")
TRIP = re.compile(r"(?:(?:please |can you |could you )?(?:take|bring|get) (?:me|us) |"
                  r"(?:let'?s |please )?(?:go|head|walk|run) )(?:back )?(?:over )?"
                  r"(?:in)?to (?P<place>.{3,60})")
# "Talk to the old man": inside a surveyed place, straight to them (people.resolve).
TALK_TO = re.compile(r"(?:(?:please |can you |could you |let'?s |i want to |i'd like to )?"
                     r"(?:go )?(?:talk|speak|chat) (?:to|with) )(?P<who>.{2,60})")


def plain_place(name: str) -> str:
    """A place's name the way it's said ("the west house in Truce"): rooms.plain_place."""
    from .rooms import plain_place as plain

    return plain(name)


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


TOOL_MARKUP = re.compile(r'<(?:\w+:)?invoke name="(?:\w+\.)?(\w+)">(.*?)</(?:\w+:)?invoke>', re.S)
TEXT_MARKUP = re.compile(r'<(?:\w+:)?parameter name="text">(.*?)</(?:\w+:)?parameter>', re.S)


def plain_reply(content: str) -> str:
    """What to say from a reply in words, not a tool call. The model sometimes labels it
    ("Finish: ...", "Finished with the message: ...") or writes the tool's arguments out
    ('{"text": "..."}'), and neither is for saying aloud."""
    if TOOL_MARKUP.search(content):
        # The tool call itself, written out as markup. Its words, if it was for saying;
        # otherwise nothing, and the model is asked again.
        for name, body in TOOL_MARKUP.findall(content):
            words = TEXT_MARKUP.search(body)
            if name in ("finish", "say") and words:
                return html.unescape(words.group(1)).strip()
        return ""
    text = re.sub(r"^\s*(finish(ed)?( with( the)? (message|text))?|say)\s*:\s*", "",
                  content, flags=re.IGNORECASE)
    if text.lstrip().startswith("{"):
        with contextlib.suppress(ValueError):
            args = json.loads(text)
            if isinstance(args, dict) and isinstance(args.get("text"), str):
                return args["text"]
    return text


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
        self.foes: Foes | None = None  # the enemies as last counted, this fight
        self._foes_said = 99  # the fewest said to be left, this fight: a drop is news once
        self._counting: asyncio.Task | None = None  # a count of them in flight
        # What the player heard lately (narration, the screen reader, answers): the screen
        # in words, for telling the model's lines about it from unrelated ones (ontopic).
        self.context_lines: Callable[[], list[str]] | None = None

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
        grounds = getattr(self, "_grounds", None)
        if grounds is not None and grounds.off_topic(line):
            return  # only a progress line: not worth a second look
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

    def _adds_nothing(self, text: str) -> bool:
        """Whether a closing line only says again what this request has said already
        (live 3: "We're inside the Truce Inn." after the walk's "We're at Truce Inn.";
        "It's set to Active." after "It's Active, closing the menu."): every word of it
        that matters was in those lines (numbers too), and it says no "not"."""
        if not self.said:
            return False
        words = [w for w in ontopic.content(text) if w not in FILLER]
        if not words or re.search(r"\b(not|no|never|can't|couldn't|won't|isn't|didn't)\b|"
                                  r"n't\b", text.lower().replace("\u2019", "'")):
            return False
        # Only what was said as done or seen counts: "Going to Leene Square." doesn't say
        # we're there.
        facts = [line for line in self.said if not INTENT.match(line)]
        before = {w for line in facts for w in ontopic.content(line)}
        return bool(before) and all(w in before for w in words)

    async def _on_topic(self, text: str) -> bool:
        """Whether a line the model wrote may be said: most are about what they were given
        (see ontopic.py); one that isn't gets the model's quick second look, with the
        player's request beside it, and isn't said on a no (live 3: a sentence about
        football fouls, spoken as the answer to "what's the battle mode set to?")."""
        if ontopic.about_itself(text):
            self._held = getattr(self, "_held", 0) + 1
            if self.on_action:
                self.on_action("off_topic", {"text": text, "about_itself": True, "said": False})
            return False
        grounds = getattr(self, "_grounds", None)
        if grounds is None or not grounds.off_topic(text):
            return True
        # Asked to read: the page's words are its own, known to nothing the check can see,
        # and in-game text can sound like anything (a quack doctor's advert: live, held twice,
        # so "Read it." got an apology).
        if READ_REQUEST.search(getattr(self, "_request", "")):
            return True
        verdict = await ontopic.second_look(self.muse, self.game,
                                            getattr(self, "_request", ""), text)
        share, loose = grounds.grounding(text)
        ok = verdict if verdict is not None else share >= ontopic.FALLBACK_SHARE
        if self.on_action:
            self.on_action("off_topic", {"text": text, "share": round(share, 2),
                                         "loose": loose[:8], "second_look": verdict,
                                         "said": ok})
        if not ok:
            self._held += 1
        return ok

    def _not_on_topic(self) -> str:
        """The tool result (or note) when a line wasn't said for being off the point."""
        return (f'not said: that isn\'t about what the player asked ("{self._request}") or '
                "this game. Answer that: call a tool, or finish with what to tell them. If "
                "they asked to read or repeat something they've already heard, say it again "
                "in full.")

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
            if str(a.get("say") or "").strip() and await self._on_topic(str(a["say"])):
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
        if name == "shop":
            return await self._shop(a)
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
                out = await self._go_to_exit(str(a.get("to") or ""),
                                             talk=a.get("talk") if isinstance(
                                                 a.get("talk"), bool) else None)
            except WALK_ERRORS as exc:
                self.controls.steer(None)
                return f"walking failed ({exc})"
            if out.startswith("arrived") and "pressed a" in out:
                line = await self._settled_line()
                if line:
                    self._heard(getattr(self, "_talked_to", None), line)
                    out += (f'. Their line, typed out in full: "{line}" (say it once, as is; '
                            "don't say part of it)")
            return out
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
        self.note_battle(st, new=time.monotonic() - self._battle_seen > 30)
        return st

    def note_battle(self, st: State | None = None, new: bool = False) -> None:
        """A fight is on screen (the app's reader sees it too, between requests); `new`:
        it just started, so where everyone stood and what the lists lacked are forgotten.
        (A fight with a quiet spell of more than 30 s between requests is still the same
        fight: the reader keeps this fresh.) A turn the reader saw shows where that
        fighter stands, for the target glove later (no model call to tell who it's on)."""
        if new:
            self._foes_said = 99
            self.foes = None
            if self._counting is not None:
                self._counting.cancel()
            if self._executor is not None:
                self._executor.new_fight()
        self._battle_seen = time.monotonic()
        if st is not None and st.kind == "command" and st.actor and st.bracket \
                and not st.paused:
            self.executor().ally_at[st.actor] = st.bracket
        if st is not None and (name := foe_name(st)):
            self.executor().foe_names[name] = time.monotonic()  # for the count's words

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
        # Said once it's put in (Executor.win's chose, as act_line words it: "Marle uses
        # Cure on Frog."): a pick that isn't in the list isn't said (live 3: "Marle: Arise
        # on Crono." and no Arise to cast).
        return parse_intent(d, st.actor)

    async def count_foes(self, image, party: list[str] = ()) -> Foes | None:
        """The enemies standing, counted by a look (one model call, about 2 s; 25/27
        right on the battle bench, misses off by one): the screen shows no enemy list or
        HP. Listing them by where they stand counts better than a number. Sent at 1536
        px: at 1024 a Gargoyle standing on Crono at a fight's start was missed 3 times in
        4 (live 3: "Two golden winged imps" with three up; 16/16 at 1536, and the bench
        23/27 at 1024 the same hour). None when the look fails."""
        try:
            img = image if isinstance(image, Image.Image) else Image.fromarray(
                np.ascontiguousarray(np.asarray(image)[..., ::-1]))
        except (ValueError, TypeError, IndexError):
            return None  # not a picture
        w, h = img.size
        field = img.crop((0, 0, w, round(h * 0.80)))  # above the windows
        names = sorted(self._executor.foe_names, key=self._executor.foe_names.get,
                       reverse=True)[:3] if self._executor is not None else []
        prompt = FOES.format(party=", ".join(party or getattr(self, "_party", None) or ["Crono"]),
                             named=FOES_NAMED.format(names=", ".join(names)) if names else "")
        try:
            d = await asyncio.wait_for(self.muse.json([{"role": "user", "content": [
                image_part(field, FOES_SIDE), {"type": "text", "text": prompt}]}],
                effort="minimal", max_tokens=1500), 12)
            where = d.get("enemies")
            count = len(where) if isinstance(where, list) else int(d.get("count"))
        except (openai.APIError, TimeoutError, ValueError, TypeError, AttributeError) as exc:
            if self.on_action:
                self.on_action("battle_note", {"text": f"count_foes failed: {exc}"})
            return None
        if not 0 <= count <= 8:
            return None
        look = " ".join(str(d.get("look") or "enemies").split()[:5]).strip(" .") or "enemies"
        if names and any(n.lower() in look.lower() for n in names):
            look = plural(next(n for n in names if n.lower() in look.lower()))
        return Foes(count, look, time.monotonic())

    async def what_we_fight(self, image, st: State, known: Foes | None = None,
                            now: Callable[[], Awaitable[State | None]] | None = None
                            ) -> str | None:
        """"What are we fighting?" in a fight: the enemies, counted by a look, and the
        party's HP in a few words ("Three Gargoyles. Frog's at 1 HP, Crono and Marle are
        full."). None when the look fails (the agent answers then). known: a count that
        still holds (the fight's opening one), said with no look. now: the fight as it is
        when the count is back, for the HP: the count takes about 2 s, and a fight that
        isn't paused goes on (live 3: "Everyone's at full health." said after an enemy
        combo had dropped Marle to 1 HP, read from the frame the request came with). With
        a known count nothing is waited on, so the request's own reading is fresh."""
        foes = known or await self.count_foes(image, [m.name for m in st.party])
        if foes is None:
            return None
        self.foes = foes
        brief = party_brief(st)
        if known is not None and brief == "Everyone's at full health.":
            # In a fight's first seconds an enemy combo can land just after this is said
            # (live, Nov 1: Frog down to 1 HP 2.7 s later, 3 runs of 3). Bad news is said anyway.
            brief = ""
        if now is not None and known is None:
            # Never the old reading: an attack drawn over the numbers says nothing of HP.
            fresh = None
            with contextlib.suppress(Exception):
                fresh = await now()
            brief = party_brief(fresh) if fresh is not None else ""
        return f"{foes.line()} {brief}".strip()

    def recount(self, image, party: list[str],
                say: Callable[[str], Awaitable[None]] | None = None) -> None:
        """Count the enemies again, in the background (one count at a time): fewer than
        before, when a second look agrees, is said ("One Gargoyle down, two left."). A
        look is right about 9 times in 10 (battle bench), so a drop is checked before it's
        said; more than before is taken quietly (one was hidden, or more came)."""
        if self._counting is not None and not self._counting.done():
            return
        self._counting = asyncio.get_running_loop().create_task(
            self._recount(image, party, say or self._speak))

    async def _recount(self, image, party: list[str],
                       say: Callable[[str], Awaitable[None]]) -> None:
        try:
            await self._recount_once(image, party, say)
        except Exception as exc:  # noqa: BLE001  a count is news, never a failure
            if self.on_action:
                self.on_action("battle_note", {"text": f"recount failed: {exc}"})

    async def _recount_once(self, image, party: list[str],
                            say: Callable[[str], Awaitable[None]]) -> None:
        before = self.foes
        now = await self.count_foes(image, party)
        if now is None or self.foes is not before:
            return
        if before is None or now.count >= before.count:
            if before is None or now.count > before.count:
                self.foes = now
            return
        if now.count == 0:
            return  # the fight is over: the game's own spoils say so
        frame = await self.grab()
        again = await self.count_foes(frame.image, party)
        if again is None or again.count != now.count or self.foes is not before:
            return
        if self._ct_steam(frame):
            # The counts take seconds: the fight can be won by now (live, Nov 2: "Won the
            # fight", then "One golden winged imp down, one left.").
            st = await asyncio.to_thread(ctsteam.read, frame.image)
            if not st.in_battle or victory(st):
                return
        self.foes = Foes(now.count, before.look, now.at)
        line = self.foes.since(before)
        # A hidden enemy can come back into the count and drop out again (live, Nov 1:
        # "One golden winged imp down, two left." twice, 28 s apart).
        if line and now.count < self._foes_said:
            self._foes_said = now.count
            await say(line)

    def _put_in(self, intent: Intent) -> None:
        """A heal or revive, said as it's put in ("Marle uses Cure on Frog."), over its
        animation: it can wait behind what's playing for 9 s or more (live 3, 8 to 11.6 s
        of silence)."""
        self._spoke_at = time.monotonic()
        asyncio.get_running_loop().create_task(self._speak(act_line(intent)))

    async def _heal_now(self, actor: str, who: str, run: Run, t0: float) -> str | None:
        """"Heal X" in a fight, chosen here (whoever is up with a healing tech, else an
        item) and done once: what it did, said, or None to leave it to the model."""
        st = self._battle
        ex = self.executor()
        held = ex.held if st is not None and st.paused and ex.held is not None else st
        plan = heal_plan(held, who, actor, ex.lacks) if held is not None else None
        if plan is None:
            return None
        if plan.say:
            await self._speak(plan.say)
            return plan.say
        told: list[Intent] = []

        def confirmed(done: Intent) -> None:
            # Then heal_line says only what it did.
            told.append(done)
            self._put_in(done)

        for intent in plan.intents[:4]:
            await self.go.wait()
            if self.cancelled:
                return "cancelled"
            before = ex.held or held
            ex.on_confirm = confirmed
            try:
                out = await ex.act(intent)
            finally:
                ex.on_confirm = None
            args = {"actor": intent.actor, "command": intent.command, "name": intent.name,
                    "target": intent.target}
            now = ctsteam.describe(out.state) if out.state is not None else "unreadable"
            run.steps.append(Step("battle", args, time.monotonic() - t0,
                                  f"{'done' if out.ok else 'stopped'}: {out.text}. Now: {now}"))
            if out.ok:
                line = heal_line(intent, out, before, ex.after, told=bool(told))
                await self._speak(line)
                return line
            if self.cancelled or "player" in out.stopped:
                return "cancelled"
            if out.stopped.startswith(("not in a battle", "the fight is won")) \
                    or "full HP" in out.stopped:
                return None
        return None

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
            # field_menu and shop; finish still last
            return [*TOOLS[:-1], CT_TOOLS[2], CT_TOOLS[3], TOOLS[-1]]
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
                    if time.monotonic() - getattr(self, "_spoke_at", 0.0) >= FIGHT_LINE_GAP_S:
                        self._spoke_at = time.monotonic()
                        asyncio.get_running_loop().create_task(self._speak(line))

                def turn(st: State) -> None:
                    # A new turn, after the last one's hits: how many are left (said if
                    # fewer, in the background).
                    if ex.frame is not None:
                        self.recount(ex.frame, [m.name for m in st.party])

                decide = self._decide if getattr(getattr(ex, "t", None), "model_decides",
                                                 False) else None
                out = await ex.win(decide, say=now_and_then, turn=turn, chose=self._put_in)
        else:
            intent = Intent(str(a.get("actor") or ""), str(a.get("command") or "attack"),
                            str(a.get("name") or ""), str(a.get("target") or ""))
            ex.after_frame = None
            healing = heals(intent) or revives(intent)
            # Watched till it lands (up to 14 s), so said as it's put in here too.
            ex.on_confirm = self._put_in if healing else None
            try:
                out = await ex.act(intent)
            finally:
                ex.on_confirm = None
            if out.ok and ex.after_frame is not None and not healing:
                self.recount(ex.after_frame, [m.name for m in out.state.party]
                             if out.state is not None else [])
        self._fought = out
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

    async def _settled_line(self, most: float = 3.5) -> str:
        """The text box's line once it has typed out (Steam CT: two looks agree and it
        ends like a sentence), so it's said whole, once (live: "Mother: Oh, I almost
        forgot" was said, then the whole line)."""
        from . import ctscreen

        if not self._ct_steam(await self.grab()):
            return ""
        end, last = time.monotonic() + most, ""
        while time.monotonic() < end:
            frame = await self.grab()
            if not ctscreen.sharp(frame.image):
                return ""
            lk = await asyncio.to_thread(ctscreen.look, frame.image)
            text = lk.parts.get("text") or lk.parts.get("question") or ""
            if lk.kind not in ("text", "choice") or not text:
                return ""
            if text == last and (ctscreen.finished(text) or lk.kind == "choice"):
                if lk.unsure:
                    return ""  # misread somewhere: the model reads the screen
                return text + (f" ({lk.parts['focus']})" if lk.kind == "choice"
                               and lk.parts.get("focus") else "")
            last = text
            await asyncio.sleep(0.35)
        return ""

    async def _shop(self, a: dict) -> str:
        """Buy at a shop or rest at an inn (shop.py), walking to the seller first when
        neither their words nor the shop are on screen."""
        await self.go.wait()
        if self.cancelled:
            return "stopped by the player"
        frame = await self.grab()
        if not self._ct_steam(frame):
            return "no shop tool here: it's for Chrono Trigger on Steam; use press"
        if self._battle is not None:
            return "in a fight: use battle"
        from . import ctscreen
        from .menu import FieldMenu
        from .shop import Shop

        async def press(button: str, ms: int) -> None:
            await self.go.wait()
            await asyncio.to_thread(self.controls.press, button, ms)

        async def grab():
            return (await self.grab()).image

        action = str(a.get("action") or "buy").lower()
        rest = action.startswith(("rest", "inn", "sleep", "stay"))
        seller = str(a.get("seller") or "").strip() or ("the innkeeper" if rest else "")
        sh = Shop(press, grab, cancelled=lambda: self.cancelled)
        v = await sh.look()
        lk = await asyncio.to_thread(ctscreen.look, sh.frame)
        walked = ""
        fm = FieldMenu(press, grab, cancelled=lambda: self.cancelled,
                       party=getattr(self, "_party", None))
        gold0 = vit0 = None
        if rest and v.funds is None and lk.kind not in ("text", "choice"):
            with contextlib.suppress(Exception):  # to say what the night did
                vit0, gold0 = await fm.vitals()
        if v.funds is None and lk.kind not in ("text", "choice"):
            if not seller:
                return ("not done: say who sells it (seller), as this place's people are "
                        "named; where_am_i lists them")
            walked = await self._act("go_to_exit", {"to": seller, "talk": True})
            # Not known in this place at all (not a surveyed place, or no one here fits):
            # another place. Anything else (a box in the way, a walk that failed) is said
            # as it was, not as "no innkeeper here".
            nobody = walked.startswith(("this place isn't in the room survey",
                                        "no single place from here fits"))
            if not walked.startswith("arrived"):
                if rest and nobody:
                    return ("not done: no innkeeper here. Go to an inn with go_to_place "
                            "(Truce Inn in 1000 A.D.), then call shop with action rest "
                            "again: it talks, pays, sleeps and checks HP and MP. Don't talk "
                            "to the innkeeper yourself.")
                if nobody:
                    return (f"not done: couldn't get to {seller} from here ({walked[:160]})."
                            " Go into the seller's place with go_to_place first, then call "
                            "shop again; don't talk to the seller yourself.")
                return (f"not done: the walk to {seller} didn't get there: {walked[:240]}. "
                        "Nothing was paid or bought.")
            if "pressed a to talk" not in walked:
                # There, but no line came up: the shop wouldn't open (it waited ~13 s on
                # camera for one before).
                return (f"not done: got to {seller}, but talking to them brought nothing "
                        f"up: {walked[:240]}. Nothing was paid or bought; look at the "
                        "screen before trying again.")
            walked = f"Walked to {seller} and talked. "
        if rest:
            from .shop import Inn, vitals_line

            out = await Inn(sh, fm).rest(gold0, vit0)
            if vit0:
                out += f" Before: {vitals_line(fm.party, vit0)[0]}."
            return walked + out
        item = str(a.get("item") or "").strip()
        if not item:
            return "not done: say which item to buy"
        try:
            count = int(a.get("count") or 1)
        except (TypeError, ValueError):
            count = 1
        hooks = getattr(self, "_shop_hooks", None) or (None, None)
        out = await sh.buy(item, count, priced=hooks[0], bought=hooks[1])
        said = " ".join(f'"{x}"' for x in sh.said)
        return walked + (f"The seller said: {said}. " if said else "") + out

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
                from .people import around

                text, n = around(*inside, 0, 6)
                self._around = (inside[1].loc, min(n, 6))
                return (f"{text} (Say it in a sentence or two, nearest first, with which way "
                        "each is. go_to_exit goes to any of them or through a way out.)")
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

    async def world_where(self, image) -> str:
        """Where a world-map screen is, in a few words ("World map, 1000 A.D., at Leene
        Square"), when the atlas places it surely; '' otherwise."""
        from . import atlas as atlas_mod
        from .quests import where_short

        atlas = await self._the_atlas()
        spot = await self._spot(atlas, image) if atlas is not None else None
        if spot is None or spot.margin < 0.1:
            return ""
        return where_short(atlas.map(spot.map), spot.x, spot.y, getattr(atlas_mod, "CELL", 8),
                           getattr(atlas_mod, "RISE", 0))

    async def _where_line(self) -> str:
        """"Where am I?" as said to the player, from the atlas on a world map or the room
        survey inside a place ('' when neither knows: the model answers from the screen)."""
        from . import atlas as atlas_mod
        from .people import PEOPLE_KINDS, THING_KINDS, directions
        from .quests import atlas_aliases, where_said

        frame = await self.grab()
        if not self._ct_steam(frame):
            return ""
        atlas = await self._the_atlas()
        spot = await self._spot(atlas, frame.image) if atlas is not None else None
        if spot is not None and spot.margin >= 0.1:
            m = atlas.map(spot.map)
            line, at = where_said(m, spot.x, spot.y, getattr(atlas_mod, "CELL", 8),
                                  getattr(atlas_mod, "RISE", 0), atlas_aliases(self.game or ""))
            if at:
                progress = self._progress()
                progress.went(at)
                progress.save(self.notes.folder, self.game or "")
            return line
        inside = await self._inside(frame.image, sure=True)
        if inside is None:
            return ""
        rooms, here = inside
        name = rooms.name(here.loc)
        if name.startswith("place 0x"):
            return ""
        out = f"We're at {name}."
        near = sorted((p for p in rooms.rooms[here.loc].people
                       if p.kind in PEOPLE_KINDS + THING_KINDS),
                      key=lambda p: math.dist(p.at, (here.x, here.y)))
        if near and math.dist(near[0].at, (here.x, here.y)) <= 6 * 16:
            p = near[0]
            way = directions(p.at[0] - here.x, p.at[1] - here.y)
            out += f" {p.name[:1].upper()}{p.name[1:]} is {way}."
        return out

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
            "" if getattr(self, "_going_to", None) == name else
            f"Going to {plain_place(name)}."))
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
            said = plain_place(name)
            await self._speak(f"We're at {said}.")
            result += (f'. The player has heard "Going to {said}." and "We\'re at {said}." '
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
        """The room survey (rooms.Rooms) for this game, loaded once, or None. People's
        lines heard in earlier sessions (heard.json) go with them, for "the man who
        talked about the war" and the name their lines are shown under ("Melchior")."""
        if getattr(self, "_rooms", None) is None:
            from .rooms import Rooms

            self._rooms = await asyncio.to_thread(Rooms.for_game, self.game or "")
            if self._rooms is not None:
                for key, line in self._heard_lines().items():
                    loc, _, name = key.partition("|")
                    room = self._rooms.rooms.get(int(loc)) if loc.isdigit() else None
                    for p in room.people if room is not None else ():
                        if p.name == name and not p.said:
                            p.said = line
        return self._rooms

    def _heard_path(self):
        from .notes import _slug

        return self.notes.folder / f"{_slug(self.game or '')}.heard.json"

    def _heard_lines(self) -> dict[str, str]:
        try:
            data = json.loads(self._heard_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _heard(self, target, line: str) -> None:
        """Keep the first line someone said when talked to (by the room survey's name for
        them), for knowing them by it later."""
        if target is None or not line or not getattr(target, "person", False):
            return
        room = self._rooms.rooms.get(target.loc) if getattr(self, "_rooms", None) else None
        if room is None or any(p.moves for p in room.people if p.name == target.name):
            return  # someone who walks about: it may have been someone else who answered
        lines = self._heard_lines()
        key = f"{target.loc}|{target.name}"
        if key in lines:
            return
        lines[key] = line[:300]
        with contextlib.suppress(OSError):
            self._heard_path().parent.mkdir(parents=True, exist_ok=True)
            self._heard_path().write_text(json.dumps(lines, indent=1), encoding="utf-8")
        for p in room.people:
            if p.name == target.name and not p.said:
                p.said = line

    async def _resolve(self, rooms, here, choices: dict, targets: dict, to: str,
                       talk: bool | None, tool: str) -> tuple[str | None, str]:
        """Who or what of the choices the player means: (its name, "") or (None, what to
        tell the model). The player's own words come first when the call is about what
        they asked for (a talk for "talk to the old man", a walk for "take me to the
        fountain"): the model's rewording ("the bearded man at the middle stall") took
        the wrong one live. Two or three that fit alike: the player is asked which,
        straight away, in a few words."""
        from . import people
        from .rooms import pick_words

        request = getattr(self, "_request", "") or ""
        cands = people.candidates(rooms, here, choices, targets)
        me = (here.x, here.y)
        marks = people.landmarks(rooms, here)
        mid = people.middle(rooms, here)
        hit = pick_words(to, {k: k for k in choices})
        mine = await asyncio.to_thread(people.resolve, to, cands, me, marks, people.MARGIN,
                                       mid)
        # The model's call is for a way out or another place ("upstairs" after "talk to
        # the guard, then go upstairs"): that, not the player's someone.
        elsewhere = (hit is not None and hit not in targets) or (
            mine.sure and mine.key not in targets)
        theirs = "" if getattr(self, "_theirs_done", False) or elsewhere else \
            people.wanted(request)
        talking = bool(people.TALK_WORDS.search(request))
        pick = None
        if theirs and not ((talking and talk is False) or (talking and tool == "walk_to")):
            pick = await asyncio.to_thread(people.resolve, theirs, cands, me, marks,
                                           people.MARGIN, mid)
        # The player's words when they settle it (and once they've been gone to, the
        # model's); else an exact name from the model (it may have asked them itself);
        # else the question, or the model's words.
        if pick is not None and pick.sure:
            self._theirs = pick.key
            return pick.key, ""
        if hit is not None:
            return hit, ""
        if pick is None or not pick.options:
            pick = mine
        if pick.sure:
            return pick.key, ""
        if pick.options:
            settled = getattr(self, "_settled_pick", None)
            if settled in {o.key for o in pick.options}:
                return settled, ""  # asked already, this request, between these
            if self.ask is None:
                return None, (f"not sure who: ask the player \"{pick.ask}\" and call "
                              "go_to_exit again with the name they pick")
            got = (await self.ask(pick.ask)).split(" (heard hands-free")[0]
            if not got:
                # Ends the request: no more model turns guessing (live 3: 166 s of
                # other tries, each asking the same again, and "Got it" with no answer).
                self._unanswered = "I'll wait till you tell me which one."
                return None, (f"asked the player \"{pick.ask}\" and got no answer: "
                              "finish and say you'll wait for their pick")
            key = people.answer(got, pick.options, me)
            if key is None:
                return None, (f"asked the player \"{pick.ask}\"; they said \"{got}\", "
                              "which didn't settle it: ask them once more, or call "
                              "go_to_exit with the exact name")
            self._settled_pick = key
            if theirs:
                self._theirs = key
            return key, ""
        return None, pick.why

    async def _known_here(self, who: str, frame: Frame, sure: bool = False) -> bool:
        """Whether, inside a surveyed place, `who` is someone or something the survey
        knows here or next door (one of them, or two or three alike to ask between; sure:
        just one)."""
        from . import people

        inside = await self._inside(frame.image, sure=True)
        if inside is None:
            return False
        rooms, here = inside
        choices, targets, _hints = rooms.options(here)
        targets = {k: t for k, t in targets.items() if k in choices}
        if not targets:
            return False
        cands = people.candidates(rooms, here, {k: choices[k] for k in targets}, targets)
        pick = await asyncio.to_thread(people.resolve, who, cands, (here.x, here.y),
                                       people.landmarks(rooms, here), people.MARGIN,
                                       people.middle(rooms, here))
        return pick.sure or (bool(pick.options) and not sure)

    async def around_here(self, more: bool = False) -> str:
        """\"Who's here?\" inside a surveyed place, from the survey and where the player
        stands, with no model call: "" when the screen isn't a place the survey knows.
        more: the next few further off (\"who else?\")."""
        from .people import around

        inside = await self._inside((await self.grab()).image, sure=True)
        if inside is None:
            return ""
        rooms, here = inside
        last = getattr(self, "_around", None)
        start = last[1] if more and last is not None and last[0] == here.loc else 0
        most = 4 if start == 0 else 6
        text, n = around(rooms, here, start, most)
        self._around = (here.loc, min(n, start + most))
        return text

    async def _through(self, m, name: str, at: tuple[float, float] | None = None
                       ) -> tuple[str, str] | None:
        """A place whose icon doesn't open, reached through another place on the map:
        (that place, its way out by this one), from the room survey."""
        import math

        if m.opens(name) or getattr(m, "world", None) is None:
            return None
        if await self._rooms_data() is None:
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

        told = asyncio.create_task(self._speak(f"Going to {plain_place(name)}, through {via}."))
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
            said = plain_place(name)
            await self._speak(f"We're at {said}.")
            last += (f'. The player has heard "Going to {said}, through {via}." and "We\'re '
                     f'at {said}." Add one short line only if something on screen needs '
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
        label, why = await self._resolve(rooms, here, {k: choices[k] for k in targets},
                                         targets, target, talk,
                                         "talk_to" if talk else "walk_to")
        if label is None and why.startswith(("not sure who", "asked the player")):
            return why
        if label is None:
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
                                 self.controls.press, talk, dash=self._dash())
        finally:
            await told
        if out.startswith("arrived"):
            walker.memory = None
            if label == getattr(self, "_theirs", None):
                self._theirs_done = True
            if talk and "pressed a" in out:
                line = await self._settled_line()
                if line:
                    self._heard(t, line)
                    out += (f'. Their line, typed out in full: "{line}" (say it once, as '
                            "is; don't say part of it)")
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

        await self._speak(f"Going to {plain_place(name)}, out of here first.")
        self._going_to = name
        out = await walk_out(rooms, m.world, shoot, self.controls.steer,
                             self.walker_keep_going, walker.log, 90.0, here, label,
                             dash=self._dash())
        if not out.startswith("arrived"):
            self._going_to = None
            return (f"on the way out of {rooms.name(here.loc)} to {name}: {out}", found)
        walker.memory = None
        return out, found

    def _dash(self):
        """The run button for walks inside places (b held on straight runs), when
        PLAYER_TWO_DASH=1: off until checked live on Steam (does b run there, and does
        the walker still line up at that speed)."""
        if os.environ.get("PLAYER_TWO_DASH", "") not in ("1", "on", "yes", "true"):
            return None
        return getattr(self.controls, "dash", None)

    async def _inside(self, image, sure: bool = False):
        """(rooms, where) when the screen is inside a place the room survey knows. sure:
        only a strong match with the character found (a weak one named "Residence" on
        the world map, live): for answers and trips started without the model."""
        if await self._rooms_data() is None:
            return None
        self._rooms.world_hint = getattr(self, "_last_world", None)
        self._rooms.world_spot = getattr(self, "_world_xy", None)
        here = await asyncio.to_thread(self._rooms.locate, image)
        if here is not None and sure and (not getattr(here, "sure", True)
                                          or getattr(here, "score", 1.0) < 0.75):
            return None
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
        label, why = await self._resolve(rooms, here, choices, targets, to, talk,
                                         "go_to_exit")
        if label is None and why.startswith(("not sure who", "asked the player")):
            return why
        if label is None:
            label, why = await pick_exit(self.muse, self.game or "", to,
                                         rooms.name(here.loc), choices, hints)
        if label is None:
            return (f"no single place from here fits {to!r} ({why}). From here: "
                    f"{'; '.join(choices)}. A place outside on the world map (another "
                    "building, a town): go_to_place, which walks out of here first. Else "
                    "ask the player which, or call go_to_exit with an exact name")
        target: Target | None = targets.get(label)
        if talk is None:
            mine = label == getattr(self, "_theirs", None)  # the player's someone
            talk = target is not None and (why.startswith("talk:") or bool(TALK.search(to))
                                           or (mine and bool(TALK.search(
                                               getattr(self, "_request", "") or ""))))
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
                                    self.controls.press, bool(talk), dash=self._dash())
        finally:
            await told
        self._talked_to = target if talk else None
        if result.startswith("arrived"):
            walker.memory = None  # somewhere else now
            if label == getattr(self, "_theirs", None):
                self._theirs_done = True
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
        self._settled_pick = None  # who the player picked when asked, this request
        self._theirs = None  # who the player's own words meant, this request
        self._theirs_done = False  # gone to them: later calls are the model's own
        self._unanswered = ""  # what to say when a question went unanswered: the end
        self._grounds, self._held = None, 0  # (set once the model is asked; see _on_topic)
        self.cancelled = False
        self.said: list[str] = []
        self._said_before_act = 0  # how many lines were said before the last action
        self._wander_s = 0.0
        frame = frame or await self.grab()
        tried = ""
        from . import mhstories

        mh = mhstories.asks(request) if mhstories.is_game(self.game or "") else ""
        if mh:
            # Monster Hunter Stories 3: Muse reads and picks, a short routine presses.
            fight = getattr(self, "_mh", None) or mhstories.Fight(self)
            self._mh = fight
            line = await fight.run(request, mh)
            if line or self.said:
                if line and not self.cancelled:
                    await self._speak(line)
                run.steps.append(Step("mh_fight", {"mode": mh}, time.monotonic() - t0,
                                      line or " ".join(self.said)))
                run.outcome = line or self.said[-1]
                run.seconds = time.monotonic() - t0
                self.history.append(f"{request} -> {run.outcome}")
                return run
        from .fast import try_fast, where_request

        if where_request(request):
            # "Where am I?" from the atlas or the room survey, said the same way every
            # time and with no model turn (live 3: the model reworded it each time, once
            # contradicting itself, in 4.6 to 5.6 s).
            line = await self._where_line()
            if line and not self.cancelled:
                await self._speak(line)
                run.steps.append(Step("where_am_i", {}, time.monotonic() - t0, line))
                run.outcome = line
                run.seconds = time.monotonic() - t0
                self.history.append(f"{request} -> {run.outcome}")
                return run
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
        out = OUT.fullmatch(" ".join(request.lower().split()).strip(" .!?"))
        if out and not trip and await self._inside(frame.image) is not None:
            # "Take me back outside" inside a surveyed place: out now, no model turn (live,
            # once the model went out and then back into the fair).
            result = await self._act("go_to_exit", {"to": "outside"})
            run.steps.append(Step("go_to_exit", {"to": "outside"}, time.monotonic() - t0,
                                  result))
            if result.startswith("arrived") or self.cancelled:
                if result.startswith("arrived"):
                    # Where outside, when the atlas knows (live 3: "We're outside." then
                    # the narrator's "World map.").
                    where = ""
                    with contextlib.suppress(Exception):
                        # After the fade (live, Nov 1: read during it, "the east Truce
                        # Residence" for Leene Square, 2 runs of 6).
                        await asyncio.sleep(0.8)
                        where = await self.world_where((await self.grab()).image)
                    await self._speak(f"We're outside: {where[:1].lower()}{where[1:]}."
                                      if where else "We're outside.")
                run.outcome = self.said[-1] if self.said else result
                run.seconds = time.monotonic() - t0
                self.history.append(f"{request} -> {run.outcome}")
                return run
            frame = await self.grab()
            tried += f"go_to_exit(outside) was tried already: {result}\n\n"
        from .people import wanted

        talk_to = TALK_TO.fullmatch(" ".join(request.lower().split()).strip(" .!?"))
        whole = talk_to.group("who") if talk_to and not trip and not out else \
            trip.group("place") if trip and not tried and not out else ""
        who = wanted(request) if whole else ""
        # Only that ("take me to Melchior"), or more after it ("...and buy a sword").
        only = bool(who) and who.lower().strip(" .!?,") == whole.lower().strip(" .!?,")
        if who and await self._known_here(who, frame, sure=not talk_to):
            # "Talk to Melchior" ("take me to the fountain") inside a surveyed place, who
            # the survey knows: there now, by the player's own words, no model turn first
            # (live, the model reworded them and went to someone else, or looked about).
            a = {"to": who, "talk": True} if talk_to else {"to": who}
            result = await self._act("go_to_exit", a)
            run.steps.append(Step("go_to_exit", a, time.monotonic() - t0, result))
            there = result.startswith("arrived")
            # "Take me to the fountain", there and nothing more to do: done, no model turn
            # (a save point's result says how to save; a talk's line is the model's).
            done = there and not talk_to and only and "save point" not in result \
                and "pressed a" not in result
            if done and not self.cancelled:
                await self._speak("We're there.")
            if there and only and not talk_to and not self.cancelled \
                    and result.startswith("arrived: standing on the save point"):
                # On it, and how to save, with no model turn (live 3: "You're already
                # standing on the save point" after a 14 s walk).
                await self._speak("We're on the save point. To save, open the menu; "
                                  "Save is lit there.")
                done = True
            if self._unanswered and not self.cancelled:
                await self._speak(self._unanswered)  # asked which one, no answer
                done = True
            if self.cancelled or done:
                run.outcome = "cancelled" if self.cancelled else \
                    (self.said[-1] if self.said else result)
                run.seconds = time.monotonic() - t0
                self.history.append(f"{request} -> {run.outcome}")
                return run
            line = re.search(r'Their line, typed out in full: "(.+)" \(say it once', result)
            if not (there and talk_to and not self.cancelled):
                line = None
            if line:
                # Their line, said as soon as it has typed out, not after a model turn
                # (live 3: Melchior's line 3 s after it was read). The model goes on from
                # there (a question in it is the player's), while it's said.
                from .fast import talk_line

                await self._speak(talk_line(who, line.group(1)))
                result = result.replace("(say it once, as is; don't say part of it)",
                                        "(said to the player already, word for word: don't "
                                        "say it again)")
            frame = await self.grab()
            tried += f"go_to_exit({a}) was done already: {result}\n"
            if there and not only:
                tried += "Now the rest of what they asked; don't walk to them again.\n"
            elif there and "pressed a" in result and line:
                tried += ("Their line is said; now only what's needed (a question in it is "
                          "the player's to answer); don't walk again.\n")
            elif there and "pressed a" in result:
                tried += ("Say their line as it says, then only what's needed (a question in "
                          "it is the player's to answer); don't walk again.\n")
            tried += "\n"
        # Plain requests whose words settle them ("buy three potions", "Crono, use
        # Cyclone"): done and said with no model turn (fast.py).
        fast_tried = ""
        if not tried:
            done, fast_tried = await try_fast(self, request, frame, run, t0)
            if done:
                run.outcome = run.outcome or (self.said[-1] if self.said else "done")
                run.seconds = time.monotonic() - t0
                self.history.append(f"{request} -> {run.outcome}")
                return run
            if fast_tried:
                frame = await self.grab()
                tried += fast_tried
        # Chrono Trigger on Steam in a fight: pause it while the model thinks.
        self._battle = await self._battle_on(frame)
        if self._battle is not None:
            self._battle = await self._hold_fight(self._battle)
            said = " ".join(request.lower().split())
            mode = "win" if FIGHT.search(said) else "run" if RUN.search(said) else ""
            if fast_tried:
                mode = ""  # tried already, without the model: its result is in tried
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
            heal = HEAL.fullmatch(said.strip(" .!?"))
            if heal and not mode and not fast_tried:
                line = await self._heal_now(heal.group("actor") or "", heal.group("who"),
                                            run, t0)
                if line is not None:
                    run.outcome = line
                    run.seconds = time.monotonic() - t0
                    self.history.append(f"{request} -> {run.outcome}")
                    return run
                frame = await self.grab()
                self._battle = await self._battle_on(frame)
                tried += "".join(f"battle({s.args}) was tried already: {s.result}\n"
                                 for s in run.steps) + ("\n" if run.steps else "")
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
        # What the model is given this request, to tell a line about it from one that
        # isn't (ontopic.py); tool results are added as they come.
        heard: list[str] = []
        if self.context_lines is not None:
            with contextlib.suppress(Exception):
                heard = list(self.context_lines())
        self._grounds = ontopic.Grounds(msgs[0]["content"], self.game, request, recent, trail,
                                        tried, self._battle_note(), TOOL_TEXT, BATTLE_CT,
                                        *heard)
        self._held = 0  # lines not said for being off the point, this request
        # The player's answer to a question sets a new course: it gets turns of its own.
        budget, _turn, out_of_turns = self.max_turns, -1, True
        # A fight: as many battle commands as the player named, then only finish (live:
        # "heal Frog" became three heals in five runs of nine, up to 42 s). A request that
        # names none ("take me to Guardia Castle", fights on the way) has no such limit.
        asked = len(COMMANDS.findall(" ".join(request.lower().split())))
        commanded = sum(1 for st in run.steps if st.tool == "battle"
                        and st.result.startswith("done:"))
        while _turn + 1 < budget:
            _turn += 1
            if self.cancelled:
                run.outcome = "cancelled"
                out_of_turns = False
                break
            t_model = time.monotonic()
            last_turn = _turn == budget - 1
            done_fighting = bool(asked) and self._battle is not None and commanded >= asked
            msg = await self.muse.tools(trim_images(msgs), self._tools(frame)
                                        if not last_turn and not done_fighting
                                        else FINISH_ONLY if done_fighting or not self.ask
                                        else FINISH_OR_ASK)
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
                # A reply in words, not a tool call; sometimes labelled ("Finish: ...").
                text = plain_reply(msg.content or "")
                if text.strip() and not await self._on_topic(text):
                    if self._held >= 2:
                        await self._speak(LOST_THREAD)
                        run.outcome = LOST_THREAD
                        out_of_turns = False
                        break
                    msgs.append({"role": "user", "content": self._not_on_topic()})
                    continue
                if text.strip():
                    if not self._adds_nothing(text):
                        await self._speak(text)
                    run.outcome = text
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
                elif c.function.name == "battle" and asked and commanded >= asked:
                    result = ("not done: the command the player asked for is done already "
                              "(one battle command each); finish with what it did")
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
                    elif not await self._on_topic(text):
                        result = self._not_on_topic()
                        if self._held >= 2:
                            finished = LOST_THREAD
                    else:
                        await self._speak(text)
                        result = "said"
                elif c.function.name == "ask_player":
                    question = str(args.get("question") or "").strip()
                    if not question:
                        # An empty question would play silence and wait for nothing.
                        result = ("ask_player needs the question itself, in the "
                                  "question field. Ask again with it, or carry on.")
                    elif not await self._on_topic(question):
                        result = self._not_on_topic().replace("not said", "not asked")
                        if self._held >= 2:
                            finished = LOST_THREAD
                    elif self.ask is None:
                        await self._speak(question)
                        result = "The player can't answer right now. Don't do it; finish."
                    else:
                        answer = await self.ask(question)
                        result = f'The player said: "{answer}"' if answer else (
                            "No answer. Don't do it; finish and say you'll wait.")
                        if answer:
                            budget = max(budget, _turn + 1 + self.max_turns // 2)
                        else:
                            self._unanswered = "Okay, I'll wait till you decide."
                elif c.function.name == "remember":
                    result = self.notes.add(self.game, str(args.get("note") or ""),
                                            str(args.get("instead_of") or ""))
                elif c.function.name == "finish":
                    text = str(args.get("text") or "").strip()
                    if text and not await self._on_topic(text) and self._held < 2:
                        result = self._not_on_topic().replace("not said", "not finished")
                    else:
                        # Twice off the point: said plainly, and done.
                        finished = LOST_THREAD if self._held >= 2 else text
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
                    if c.function.name == "battle" and result.startswith("done:"):
                        commanded += 1
                        if asked and commanded >= asked:
                            result += (" That was the command asked for: finish now, saying "
                                       "what it did (no other command).")
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
                self._grounds.add(result, json.dumps(args))
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": result})
                if self._unanswered and finished is None:
                    finished = self._unanswered  # asked and no answer: done here
            if self.cancelled:
                run.outcome = "cancelled"
                out_of_turns = False
                break
            if finished is not None:
                if not finished and last_turn and len(self.said) <= self._said_before_act:
                    finished = ("I stopped there after a lot of steps. Say describe to hear "
                                "where things stand.")
                if finished and not self._already_said(finished) \
                        and not self._adds_nothing(finished):
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
            if noise is None and calls and all(c.function.name in STILL_TOOLS
                                               for c in calls):
                # Nothing was pressed (where_am_i, guide): the screen is as it was, and
                # on an animated scene waiting for it to settle costs the full 1.5 s.
                frame = await self.grab()
            elif self._battle is not None:
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
            self._grounds.add(self._battle_note())
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
