"""The story as a list of steps, and where the player is on it, for "what do I do next".

A walkthrough is long prose in story order; asking a model to answer "what's next" from
search hits gives boss tables and the wrong chapter. So it is turned into a quest list
once, offline: one Muse call per chapter pulls out the ordered steps (the era or world,
the place to go, what to do there, the way there as the guide words it, a short
spoiler-light summary, key items and who joins or leaves). Then the places are linked to
the atlas's world-map names (see atlas.py), so "go there" can walk. scripts/build_quests.py
makes it; it is kept next to the guide as <guide>.quests.json.

Where the player is on the list is a judgement from cheap evidence: the era map they're
on (atlas), the last places they went, the step they last confirmed, and one look at the
screen. A guess not anchored by what the player confirmed is said as a question.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import guide
from .guide import chapters, guide_path, prose

# --- the quest list ---------------------------------------------------------------------


@dataclass
class QuestStep:
    n: int  # place in the whole game, from 1
    chapter: str  # "3. The Queen is Gone"
    era: str  # the time period, world or map it happens in ("600 A.D."), or ""
    place: str  # where to go, as the guide names it
    objective: str  # what to do there
    how: str = ""  # the way there, as the guide words it
    summary: str = ""  # one short spoiler-light sentence
    items: list[str] = field(default_factory=list)  # key items got or needed
    party: str = ""  # who joins or leaves
    optional: bool = False
    atlas_place: str = ""  # the world-map place to walk to for it (atlas name), or ""

    def line(self) -> str:
        """One line for a list the model chooses from."""
        bits = [f"{self.n}.", f"[{self.chapter}]", self.era or "?", "|", self.place, "-",
                self.summary or self.objective]
        if self.party:
            bits.append(f"(party: {self.party})")
        if self.optional:
            bits.append("(optional)")
        return " ".join(bits)

    def told(self) -> str:
        """The step, said for the agent: what, where, how to get there."""
        out = f"{self.objective} (at {self.place}"
        out += f", {self.era})" if self.era else ")"
        if self.how:
            out += f". Getting there: {self.how}"
        if self.atlas_place:
            out += f". On the world map: {self.atlas_place!r} (go_to_place can walk there)"
        if self.items:
            out += f". Key items: {', '.join(self.items)}"
        if self.party:
            out += f". Party: {self.party}"
        return out


def quests_path(game: str, folder: Path | None = None) -> Path | None:
    g = guide_path(game, folder or guide.GUIDES)
    return g.with_suffix(".quests.json") if g else None


@dataclass
class Quests:
    steps: list[QuestStep]
    game: str = ""

    @classmethod
    def load(cls, path: Path) -> Quests:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = set(QuestStep.__dataclass_fields__)
        steps = [QuestStep(**{k: v for k, v in s.items() if k in known})
                 for s in data.get("steps", [])]
        return cls(steps, data.get("game", ""))

    @classmethod
    def for_game(cls, game: str, folder: Path | None = None) -> Quests | None:
        p = quests_path(game, folder)
        if p is None or not p.exists():
            return None
        try:
            return cls.load(p)
        except (OSError, ValueError, TypeError):
            return None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"game": self.game, "made": time.strftime("%Y-%m-%d"),
                                    "steps": [asdict(s) for s in self.steps]},
                                   indent=1, ensure_ascii=False), encoding="utf-8")

    def step(self, n: int) -> QuestStep | None:
        return self.steps[n - 1] if 1 <= n <= len(self.steps) else None

    def aliases(self) -> dict[str, set[str]]:
        """Each linked atlas place and the other names the guide uses for it."""
        out: dict[str, set[str]] = {}
        for s in self.steps:
            if s.atlas_place and s.place.lower() != s.atlas_place.lower():
                out.setdefault(s.atlas_place, set()).add(s.place)
        return out


def era_key(era: str) -> str:
    """Era names compared loosely: "1000 A.D.", "1000 AD" and "AD 1000" are one era, and
    "65 million B.C." is "65,000,000 B.C."."""
    e = era.lower().replace("million", "000000").replace(",", "")
    e = re.sub(r"(\d)\s+(0{6})", r"\1\2", e)
    num = re.findall(r"\d+", e)
    side = "bc" if re.search(r"b\.?\s*c", e) else "ad" if re.search(r"a\.?\s*d", e) else ""
    if num:
        return num[0] + side
    return re.sub(r"[^a-z0-9]+", "", e.replace("the ", ""))


def same_era(a: str, b: str) -> bool:
    return bool(a and b) and era_key(a) == era_key(b)


# --- building it (offline, scripts/build_quests.py) ------------------------------------

EXTRACT = """\
You turn one chapter of a walkthrough for the video game {game} into the ordered list of \
steps a player follows, for a helper that tells a blind player what to do next and walks \
them there. Answer JSON: {{"steps": [{{"era": "...", "place": "...", "objective": "...", \
"how_to_get_there": "...", "summary": "...", "items": ["..."], "party": "...", \
"optional": false}}]}}

- One step a place the player goes to, in the order the walkthrough does them. Usually 3 \
to 12 steps a chapter. Required story steps; side things only when the walkthrough \
suggests doing them, with "optional": true. Never battle tactics, enemy lists or stats.
- era: the time period, world or map the step happens in{eras}. From the chapter, or \
from what you know of the game for sure (where the story starts, where a place is); "" \
when neither makes it clear.
- place: where to go, named the way the walkthrough names it ("Truce Inn", "Guardia \
Forest", "the Telepod at Leene Square"). The place itself, not the region. A building, \
room or spot always with the town, castle or dungeon it's in ("the inn in Choras", "the \
throne room in Guardia Castle"): many towns have an inn.
- objective: what to do there, in one short sentence (talk to someone, get an item, beat \
a boss, open a door).
- how_to_get_there: the directions the walkthrough gives, close to its words ("north of \
Truce, through Guardia Forest"), making sense on their own (not "on this side"). "" if \
it gives none. Never invent directions.
- summary: one short sentence a player could hear without spoilers: no twists, deaths, \
secret identities or reveals; plain about the goal.
- items: key items got or needed here (quest items, not shop or chest loot). [] if none.
- party: who joins or leaves here ("Marle joins"), or "".
"""

LINK = """\
These are places from a walkthrough of {game}, in the {era} era or map. For each, say \
which place on the world-map list the player walks to on the world map to get there: \
the place itself, or the town, building or area it is in or reached through ("Truce \
Market" -> "Market", "the Telepod" -> "Leene Square" when the Telepod is at the fair). \
Go by what the place is and where it is in the game, not by the way there (a place \
reached "through the forest" isn't the forest). Use exact names from the list. "" when \
none is (inside a dungeon reached another way, in another town than the one on the list, \
or the list doesn't have it). Don't match on a shared word: "Truce" alone is a town, not \
"Truce Inn". Each place comes with its chapter, what the player does there and the step \
before, to tell apart two places with one name.

World-map places: {names}

Walkthrough places:
{places}

Answer JSON: {{"links": {{"<step number>": "<world-map name or empty>"}}}}"""


async def extract_chapter(muse, game: str, title: str, text: str, eras: list[str],
                          effort: str = "low") -> list[dict]:
    """One chapter's steps, as Muse reads them (dicts, fields as in EXTRACT)."""
    era_hint = (f"; write it the way these are written when it's one of them: "
                f"{', '.join(eras)}; others in the same style" if eras else "")
    data = await muse.json([
        {"role": "system", "content": EXTRACT.format(game=game, eras=era_hint)},
        {"role": "user", "content": f"Chapter: {title}\n\n{prose(text)}"},
    ], effort=effort, max_tokens=8000)
    steps = data.get("steps") if isinstance(data, dict) else None
    return [s for s in steps or [] if isinstance(s, dict) and str(s.get("place", "")).strip()]


def to_step(n: int, chapter: str, d: dict) -> QuestStep:
    def text(k: str) -> str:
        return " ".join(str(d.get(k) or "").split())

    items = d.get("items") or []
    if isinstance(items, str):
        items = [items]
    return QuestStep(n=n, chapter=chapter, era=text("era"), place=text("place"),
                     objective=text("objective"), how=text("how_to_get_there"),
                     summary=text("summary"),
                     items=[" ".join(str(i).split()) for i in items if str(i).strip()],
                     party=text("party"), optional=d.get("optional") is True)


def fill_eras(steps: list[QuestStep]) -> None:
    """A step whose era the chapter didn't make clear is in the era of the one before."""
    last = ""
    for s in steps:
        if s.era:
            last = s.era
        else:
            s.era = last


async def link_places(muse, quests: Quests, maps: dict[str, list[str]], game: str,
                      effort: str = "low", log=print) -> int:
    """Set each step's atlas_place: one Muse call per era map that has places, each step
    with its chapter, objective and the step before it, since the same name can be two
    buildings (a mayor's manor in two towns). maps: era map name -> its place names.
    Returns how many steps are linked."""
    for s in quests.steps:
        s.atlas_place = ""
    # Maps of one era (an era split into several maps) are linked together.
    groups: dict[str, tuple[list[str], list[str]]] = {}
    for era, names in maps.items():
        if names:
            g = groups.setdefault(era_key(era), ([], []))
            g[0].append(era)
            g[1].extend(n for n in names if n not in g[1])
    for eras, names in groups.values():
        mine = [s for s in quests.steps if same_era(s.era, eras[0])]
        if not mine:
            continue
        exact = {n.lower(): n for n in names}
        lines = []
        for s in mine:
            before = quests.step(s.n - 1)
            ctx = [s.chapter, f"to: {s.objective}"]
            if s.how:
                ctx.append(f"way there: {s.how}")
            if before is not None:
                ctx.append(f"step before: {before.place}")
            lines.append(f"{s.n}. {s.place} ({'; '.join(ctx)})")
        data = await muse.json([{"role": "user", "content": LINK.format(
            game=game, era=" / ".join(eras), names="; ".join(names),
            places="\n".join(lines))}], effort=effort, max_tokens=20000)
        links = data.get("links") if isinstance(data, dict) else None
        links = links if isinstance(links, dict) else {}
        if not links:
            log(f"{' / '.join(eras)}: no links came back; only exact names are linked")
        for s in mine:
            if str(s.n) in links:
                s.atlas_place = exact.get(str(links[str(s.n)] or "").strip().lower(), "")
            else:  # no answer for it: only a name that is exactly the place's
                s.atlas_place = exact.get(s.place.lower(), "")
    return sum(1 for s in quests.steps if s.atlas_place)


async def build(muse, game: str, text: str, eras: list[str], only: set[int] | None = None,
                concurrency: int = 5, effort: str = "low", log=print) -> Quests:
    """The quest list from a chaptered walkthrough: chapters read in parallel, steps kept
    in story order. A chapter that fails is tried once more, then left out (and said)."""
    parts = [(i, t, b) for i, (t, b) in enumerate(chapters(text), 1)
             if t and (only is None or i in only)]
    gate = asyncio.Semaphore(concurrency)

    async def one(i: int, title: str, body: str) -> list[dict]:
        async with gate:
            for attempt in (1, 2):
                t0 = time.monotonic()
                try:
                    got = await extract_chapter(muse, game, title, body, eras, effort)
                except Exception as exc:  # noqa: BLE001  # a model or network failure
                    log(f"{title}: failed ({type(exc).__name__}: {exc}), try {attempt}")
                    continue
                log(f"{title}: {len(got)} steps ({time.monotonic() - t0:.0f} s)")
                if got:
                    return got
            log(f"{title}: LEFT OUT, no steps after two tries")
            return []

    found = await asyncio.gather(*(one(i, t, b) for i, t, b in parts))
    steps: list[QuestStep] = []
    for (_, title, _), got in zip(parts, found, strict=True):
        for d in got:
            steps.append(to_step(len(steps) + 1, title, d))
    fill_eras(steps)
    return Quests(steps, game)


# --- where the player is in the story --------------------------------------------------


@dataclass
class Progress:
    """Kept per game between sessions, next to the agent's notes."""
    step: int = 0  # the step the player is on, as last settled
    confirmed: bool = False  # the player said so (not only a guess)
    visited: list[str] = field(default_factory=list)  # recent places, newest last
    t: float = 0.0

    @staticmethod
    def path(folder: Path, game: str) -> Path:
        return folder / f"{re.sub(r'[^a-z0-9]+', '-', game.lower()).strip('-') or 'game'}" \
            ".progress.json"

    @classmethod
    def load(cls, folder: Path, game: str) -> Progress:
        try:
            d = json.loads(cls.path(folder, game).read_text(encoding="utf-8"))
            return cls(int(d.get("step", 0)), bool(d.get("confirmed")),
                       [str(v) for v in d.get("visited", [])][-8:], float(d.get("t", 0)))
        except (OSError, ValueError, TypeError, AttributeError):
            return cls()

    def save(self, folder: Path, game: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.t = time.time()
        self.path(folder, game).write_text(json.dumps(asdict(self)), encoding="utf-8")

    def went(self, place: str) -> None:
        if place and (not self.visited or self.visited[-1] != place):
            self.visited = [*self.visited, place][-8:]


@dataclass
class Guess:
    step: int
    sure: str  # "high", "medium" or "low", as the model put it
    why: str = ""
    other: int = 0  # the next likeliest step, or 0
    ask: bool = True  # say it as a question
    text: str = ""  # what the agent is told


PLACE_IN_STORY = """\
A player of {game} is somewhere in the story. Below are the walkthrough's steps (number, \
[chapter], era | place - summary). Which step are they on now: the first one they \
haven't finished? Evidence:
{evidence}
Use the screen too: an era or place name shown, who is in the party, what's going on. \
The step they confirmed before is a floor: they may have moved on from it, rarely back. \
Be honest: "high" only when the evidence clearly points to one step; the same era and \
place recurring in several chapters is not enough.

Steps:
{steps}

Answer JSON: {{"step": <number>, "sure": "high" | "medium" | "low", "other": <the next \
likeliest number or 0>, "why": "<a few words: what on screen or in the evidence>"}}"""


def candidates(quests: Quests, progress: Progress, era: str, player_says: str = ""
               ) -> list[QuestStep]:
    """The steps worth choosing among: all when the player said where they are (their
    word beats the map they're on), else near the last settled step, else those in the
    era the player is in (when known), else all."""
    steps = quests.steps
    if player_says.strip():
        return steps
    if progress.step:
        lo = max(0, progress.step - (1 if progress.confirmed else 4))
        return steps[lo:progress.step + 12]
    if era:
        near = {i for i, s in enumerate(steps) if same_era(s.era, era)}
        near |= {i + d for i in near for d in (-1, 1) if 0 <= i + d < len(steps)}
        if near:
            return [steps[i] for i in sorted(near)]
    return steps


def judge(step: int, sure: str, progress: Progress, era: str, quests: Quests) -> bool:
    """Whether a guess can be said as fact: anchored by a step the player confirmed, a
    little after it, not in another era than the map they're on, and the model sure."""
    s = quests.step(step)
    if s is None or sure == "low":
        return False
    if era and s.era and not same_era(era, s.era):
        return False
    if not progress.confirmed:
        return False
    return progress.step <= step <= progress.step + 3 and (sure == "high"
                                                           or step == progress.step)


async def place_in_story(muse, quests: Quests, progress: Progress, image=None, era: str = "",
                         near: str = "", player_says: str = "", game: str = "") -> Guess:
    """Which step the player is on, and whether that's sure enough to say as fact."""
    from .muse import image_part

    pool = candidates(quests, progress, era, player_says)
    ev = []
    if era:
        ev.append(f"- On the world map of era {era}" + (f", near {near}" if near else ""))
    if progress.visited:
        ev.append(f"- Places visited lately, oldest first: {', '.join(progress.visited)}")
    if progress.step:
        s = quests.step(progress.step)
        how = "confirmed by the player" if progress.confirmed else "an earlier guess"
        ev.append(f"- Last step settled: {progress.step} ({how}): {s.line() if s else ''}")
    if player_says:
        ev.append(f'- The player says: "{player_says}"')
    if not ev:
        ev.append("- Nothing but the screen.")
    content: list[dict] = []
    if image is not None:
        content.append(image_part(image, 1024))
    content.append({"type": "text", "text": PLACE_IN_STORY.format(
        game=game or quests.game or "the game", evidence="\n".join(ev),
        steps="\n".join(s.line() for s in pool))})
    data = await muse.json([{"role": "user", "content": content}], effort="low",
                           max_tokens=3000)
    valid = {s.n for s in pool}
    try:
        n = int(data.get("step") or 0)
    except (TypeError, ValueError):
        n = 0
    if n not in valid:
        n = progress.step if progress.step in valid else pool[0].n
    try:
        other = int(data.get("other") or 0)
    except (TypeError, ValueError):
        other = 0
    sure = str(data.get("sure") or "low").lower()
    if sure not in ("high", "medium", "low"):
        sure = "low"
    g = Guess(n, sure, " ".join(str(data.get("why") or "").split())[:200],
              other if other in valid and other != n else 0)
    g.ask = not judge(n, sure, progress, era, quests)
    g.text = tell(quests, g)
    return g


def tell(quests: Quests, g: Guess) -> str:
    """The tool result for next_goal."""
    now = quests.step(g.step)
    if now is None:
        return "the quest list is empty"
    after = next((s for s in quests.steps[g.step:] if not s.optional), None)
    total = len(quests.steps)
    where = f"step {now.n} of {total}, {now.chapter}"
    if g.ask:
        other = quests.step(g.other) if g.other else None
        alt = f" Next likeliest: step {other.n}: {other.summary or other.objective}." \
            if other else ""
        return (f"not sure where the player is in the story (best guess {where}: "
                f"{now.summary or now.objective}; {g.sure} confidence, {g.why}).{alt} "
                f'Ask with ask_player: "I think you\'re at {(now.summary or now.objective).rstrip(".")}'
                "; is that right?\" in your own short words. If yes, call next_goal with "
                f"step {now.n}; if they say something else, call next_goal with "
                "player_says set to their words.")
    out = f"Now ({where}): {now.told()}."
    if after is not None:
        out += f" After that (step {after.n}): {after.told()}."
    out += (f" ({g.why}.) Tell the player the next thing in a sentence or two, no "
            "spoilers past it; offer to take them there.")
    return out


# --- where the player is on a world map, and which place they mean ----------------------


def where_text(m, x: float, y: float, step: float = 8.0, rise: float = 0.0) -> tuple[str, str]:
    """Where a spot on a world map is, said from the places around it: "On the 1000 A.D.
    map, just south of Leene Square (3 steps); Truce Inn is southwest (9 steps)". m is an
    atlas WorldMap, (x, y) the middle of the view, rise how far above it the character
    stands, step map pixels to a step. Also returns the nearest place's name when the
    character is at it (within 3 steps), for the places visited."""
    from .atlas import compass

    px, py = x, y - rise
    if not m.places:
        return f"On the {m.name} map; no places are named on it yet.", ""
    near = sorted(m.places.items(), key=lambda kv: (kv[1][0] - px) ** 2 + (kv[1][1] - py) ** 2)
    name, (nx, ny) = near[0]
    steps = ((nx - px) ** 2 + (ny - py) ** 2) ** 0.5 / step
    if steps < 1:
        out = f"On the {m.name} map, at {name} (on its spot: press a to go in)"
    else:
        side = compass(px - nx, py - ny)  # the player's side of the place
        out = (f"On the {m.name} map, {'just ' if steps <= 4 else ''}{side} of {name} "
               f"({steps:.0f} step{'s' if round(steps) != 1 else ''})")
    others = []
    for n, (ox, oy) in near[1:3]:
        d = ((ox - px) ** 2 + (oy - py) ** 2) ** 0.5 / step
        if d <= 40:
            others.append(f"{n} is {compass(ox - px, oy - py)} ({d:.0f} steps)")
    if others:
        out += "; " + ", ".join(others)
    return out + ".", name if steps <= 3 else ""


def atlas_aliases(game: str) -> dict[str, set[str]]:
    """Other names for atlas places: an "aliases" table in atlas.json ({"Leene Square":
    ["the fair"]}, per map or at the top), if it has one."""
    from .atlas import ATLASES, slug

    out: dict[str, set[str]] = {}
    if not ATLASES.is_dir():
        return out
    name = slug(game)
    for d in sorted(p for p in ATLASES.iterdir() if (p / "atlas.json").exists()):
        if d.name == name or all(w in name.split("-") for w in d.name.split("-")):
            try:
                spec = json.loads((d / "atlas.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return out
            tables = [spec.get("aliases") or {}] + [m.get("aliases") or {}
                                                    for m in spec.get("maps", [])]
            for t in tables:
                for k, v in t.items() if isinstance(t, dict) else ():
                    out.setdefault(k, set()).update([v] if isinstance(v, str) else v)
            return out
    return out


PICK_PLACE = """\
A player of {game} on a world map asked to go to: "{said}".{near} The places Player \
Two can walk them to, by map (other names the walkthrough uses in brackets):
{places}
Which one do they mean? Only one that clearly is it, by its name or another name for it, \
or by what you know of the game for sure (the fair is at Leene Square). "The inn" or "the \
shop" with several on the list: the one in the town they're at or nearest. A word in \
common isn't enough: a whole town isn't one building in it. If it's a town or area with \
several places on the list, or none fits, give "" and say why in a few words. A place \
marked [doesn't open now] only when they name it: "the hut" is one that opens.
Answer JSON: {{"map": "<map name>", "place": "<exact name from the list, or empty>", \
"why": "<a few words>"}}"""


async def pick_place(muse, game: str, said: str, maps: list, aliases: dict[str, set[str]],
                     near: str = "", hints: dict[str, set[str]] | None = None
                     ) -> tuple[object, str] | str:
    """The place on these world maps (atlas WorldMaps, the one the player is on first)
    the player means: (map, exact name), or why none fits. aliases: other names that
    surely mean a place (the atlas's own); hints: names the walkthrough used for it,
    shown to the model only (a step "to Truce" that went to the market doesn't make
    Truce the market). near: the place the player is nearest, for "the inn". An exact
    name needs no model."""
    want = " ".join(said.lower().split()).strip(" .!?")
    for m in maps:
        for n in m.places:
            if n.lower() == want or want in {a.lower() for a in aliases.get(n, ())}:
                return m, n
    names = [n.lower() for m in maps for n in m.places]

    def town(h: str) -> bool:  # starts several names: a town, not one place in it
        return sum(n.startswith(h.lower() + " ") for n in names) >= 2

    known = {k: set(v) for k, v in aliases.items()}
    for k, v in (hints or {}).items():
        known.setdefault(k, set()).update(h for h in v if not town(h))
    def shut(m, n: str) -> bool:
        return hasattr(m, "opens") and not m.opens(n)

    lines = []
    for m in maps:
        if m.places:
            lines.append(f"{m.name}: " + "; ".join(
                n + (f" ({', '.join(sorted(known[n]))})" if known.get(n) else "")
                + (" [doesn't open now]" if shut(m, n) else "")
                for n in m.places))
    if not lines:
        return "no places are named on these maps yet"
    try:
        data = await muse.json([{"role": "user", "content": PICK_PLACE.format(
            game=game or "the game", said=said, places="\n".join(lines),
            near=f" They are on the {maps[0].name} map near {near}." if near else "")}],
            effort="minimal", max_tokens=800)
    except (ValueError, OSError) as exc:
        return f"couldn't choose a place ({exc})"
    place = str(data.get("place") or "").strip().lower()
    order = sorted(maps, key=lambda m: m.name != str(data.get("map") or ""))
    for m in order:
        for n in m.places:
            if n.lower() == place:
                return m, n
    return str(data.get("why") or "no place on the map fits that")
