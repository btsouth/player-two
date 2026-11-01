"""Who or what the player means inside a surveyed place, and who's around them, from the
room survey and where the player stands, with no model call.

The survey names everyone from their look ("Melchior the sword seller in the middle"),
but players say "the guy selling swords", "Melchior", "the girl", "that soldier", "the
old man by the stalls". A model choosing from the list of names took "the old man at the
lower stalls" for Melchior and couldn't find "Melchior" at all (live, 8 of 11). Here the
player's words are matched against all that's known of each one:

- the survey's name, and other names they go by (`aka` in the survey: "Melchior",
  "swordsmith", "old man");
- what they said when talked to (`said`; the speaker's name a line is shown under);
- what they are, by groups of words: "guy", "fellow" and "man" are a man; "selling
  swords", "the weapon shop" and "swordsmith" a weapon seller; "soldier" and "knight" a
  guard. A word that contradicts someone ("girl" for a man, "soldier" for a seller)
  counts against them;
- where they stand: "on the left", "lower", "in the middle" among those that fit, "by
  the bell" near something the survey placed, "that one" nearest the player.

The best fit wins when it's clearly ahead; when two or three fit about as well, a short
question names them ("Do you mean Melchior the sword seller, or the old man at the lower
stalls?") instead of a guess. Nothing that fits the main word ("the purple monster" in a
square with no monster): no pick, and the caller says so or asks a model.

around() answers "who's here?" the same way: the nearest people first, how many steps
and which way on the d-pad, then things and the ways out.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

TILE = 16

# Groups of words that mean the same kind of someone (or something), with how much a
# match counts: what they do (3), what they are (2). The player's words and the survey's
# (names, aka) both go through these.
GROUPS: dict[str, tuple[float, str]] = {
    "guard": (3, ("guard guards soldier soldiers knight knights sentry sentries guardsman "
                 "guardsmen trooper watchman")),
    "seller": (3, ("seller sellers selling sell sells shopkeeper shopkeepers shop shops "
                  "merchant merchants vendor vendors trader traders clerk salesman "
                  "storekeeper shopkeep")),
    "innkeeper": (3, "innkeeper inn lodging"),
    "music": (3, "musician musicians music band singer singing song songs"),
    "weapons": (3, "sword swords blade blades weapon weapons swordsmith blacksmith smith"),
    "armor": (3, "armor armour armors armours helmet helmets shield shields"),
    "items": (3, ("item items potion potions panacea panaceas tonic tonics ether ethers "
                 "medicine supplies")),
    "games": (3, "game games race races racing bet betting gamble prize prizes"),
    "mother": (3, "mom mother mum mama ma mommy"),
    "father": (3, "dad father papa pa daddy"),
    "mayor": (3, "mayor"),
    "save": (3, "save sparkle"),
    "chest": (3, "chest chests treasure"),
    "man": (2, ("man men guy guys fellow fella gentleman gent dude sir mister he him "
               "husband boy boys lad grandpa grandfather dad father")),
    "woman": (2, ("woman women lady ladies girl girls gal madam she her wife grandma "
                 "grandmother granny mom mother mum")),
    "old": (2, "old elderly elder aged grandpa grandma grandfather grandmother granny"),
    "young": (2, ("kid kids child children boy boys girl girls little young youngster lad "
                 "lass")),
    "cat": (2, "cat cats kitty kitten"),
    "dog": (2, "dog dogs puppy"),
    "robot": (2, "robot robots"),
    "monster": (2, "monster monsters creature creatures imp imps"),
}
WORD_GROUPS: dict[str, set[str]] = {}
for _g, (_w, _words) in GROUPS.items():
    for _word in _words.split():
        WORD_GROUPS.setdefault(_word, set()).add(_g)
LIVING = {"man", "woman", "old", "young"}
# What only people are.
PEOPLE_ONLY = LIVING | {"guard", "seller", "innkeeper", "music", "mother", "father", "mayor"}
# Things known of someone, or not: a word for a man is no miss for someone not known to be
# a man or a woman ("the old musician").
KNOWN = ({"man", "woman"}, {"old", "young"})
# Groups that can't both be true of one: a word for one counts against the other.
CLASH = {frozenset(p) for p in (
    ("man", "woman"), ("old", "young"), ("guard", "seller"), ("guard", "innkeeper"),
    ("weapons", "armor"), ("weapons", "items"), ("armor", "items"), ("mother", "man"),
    ("father", "woman"), ("innkeeper", "seller"))}
CLASH |= {frozenset((a, b)) for a in ("cat", "dog", "robot", "monster", "save", "chest")
          for b in (*LIVING, "guard", "seller", "innkeeper", "music", "mother", "father")}
CLASH |= {frozenset((a, b)) for a in ("cat", "dog", "robot", "monster", "save", "chest")
          for b in ("cat", "dog", "robot", "monster", "save", "chest") if a != b}

# Where something is: the way it lies, among those that fit (dx, dy), or the middle.
POSITIONS = {"left": (-1, 0), "west": (-1, 0), "western": (-1, 0), "leftmost": (-1, 0),
             "right": (1, 0), "east": (1, 0), "eastern": (1, 0), "rightmost": (1, 0),
             "top": (0, -1), "upper": (0, -1), "north": (0, -1), "northern": (0, -1),
             "up": (0, -1), "above": (0, -1), "back": (0, -1),
             "bottom": (0, 1), "lower": (0, 1), "south": (0, 1), "southern": (0, 1),
             "down": (0, 1), "below": (0, 1), "front": (0, 1),
             "middle": (0, 0), "center": (0, 0), "centre": (0, 0), "central": (0, 0),
             "northwest": (-1, -1), "northeast": (1, -1), "southwest": (-1, 1),
             "southeast": (1, 1)}
# "By the bell": what follows these is something to be near.
NEAR = {"by", "near", "beside", "next", "at", "behind", "around", "outside", "inside",
        "in", "on", "under", "with", "from", "close", "past", "beyond", "across"}
# "That soldier", "the one nearest": the nearest of those that fit.
NEAREST = {"that", "this", "nearest", "closest", "nearby", "near", "here", "there"}
STOP = {"the", "a", "an", "that", "this", "those", "these", "one", "ones", "there", "here",
        "who", "whose", "which", "what", "is", "are", "was", "were", "be", "it", "its",
        "standing", "stands", "stand", "sitting", "sits", "sit", "over", "please", "some",
        "someone", "somebody", "person", "people", "of", "and", "to", "me", "us", "my",
        "our", "you", "your", "i", "we", "him", "her", "them", "they", "again", "now",
        "just", "front", "side", "thing", "something", "place", "spot",
        "talking", "talked", "talks", "about", "said", "says", "say", "for", "too",
        "nearest", "closest", "nearby", "next", "close", "way", "can", "could", "would",
        "will", "lets", "let", "want", "need", "go", "get", "walk", "take", "bring",
        "head", "find", "talk", "speak", "chat", "ask", "visit", "meet", "mean", "meant",
        "oh", "um", "uh", "ok", "okay", "yeah", "yes", "no", "not", "very", "really",
        "with", "where", "whos", "wheres", "has", "have", "had", "do", "does", "did"}
POLITE = (r"^\s*(?:(?:please|hey|ok|okay|now|and|then|so|can you|could you|would you|"
          r"will you|i want to|i'd like to|i would like to|i wanna|i need to|let's|lets|"
          r"let me|go and|player two)[,\s]+)*")
VERB = (r"(?:(?:go|walk|run|head|take me|take us|bring me|bring us|get me|lead me|guide me|"
        r"move)\s+(?:over\s+|back\s+|up\s+|down\s+)?(?:to|toward|towards|over to|up to|by)\s+"
        r"|(?:go\s+)?(?:talk|speak|chat)\s+(?:to|with)\s+|(?:ask|find|visit|meet|see)\s+"
        r"|where(?:'s| is| are)\s+|(?:go\s+)?(?:say hi|say hello)\s+to\s+)")
# The words before the someone: "please take me to", "talk to", "where's".
LEAD = re.compile(POLITE + VERB + "?", re.IGNORECASE)
ASKS_FOR = re.compile(POLITE + VERB, re.IGNORECASE)
TALK_WORDS = re.compile(r"\b(talk|speak|chat|ask|say hi|say hello)\b", re.IGNORECASE)
MARGIN = 1.5  # how far ahead the best fit must be to go without asking
FAR = 2.5  # in tiles: "by" something means about this near


def stem(w: str) -> str:
    """A word's plain form: no 's, no plural s ("stalls" is "stall", "Crono's" "crono")."""
    w = w.lower().strip("'")
    w = w.removesuffix("'s")
    if len(w) > 5 and w.endswith("ed") and w[-3] not in "e":
        return w[:-2]  # "bearded" is "beard", "talked" "talk"
    if len(w) > 4 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def groups_of(words) -> set[str]:
    out: set[str] = set()
    for w in words:
        out |= WORD_GROUPS.get(w, set()) | WORD_GROUPS.get(stem(w), set())
    return out


@dataclass
class Who:
    """Someone or something the player could mean: a person, an animal, a thing or a
    spot the survey placed, or a way out (`place`)."""
    key: str  # its name among the choices ("the innkeeper (in here)")
    name: str  # its own name ("the innkeeper")
    loc: int
    at: tuple[float, float] | None
    kind: str = "person"  # person, animal, thing, save point, chest, spot, place
    here: bool = True  # in the player's place
    aka: tuple[str, ...] = ()
    said: str = ""
    moves: bool = False
    where: str = ""  # the place it's in
    groups: set[str] = field(default_factory=set)
    words: set[str] = field(default_factory=set)
    proper: set[str] = field(default_factory=set)  # names: Melchior, Crono, Bekkler
    heard: set[str] = field(default_factory=set)  # content words of what they said
    place: set[str] = field(default_factory=set)  # words of the place it's in

    def __post_init__(self):
        text = " ".join((self.name, *self.aka))
        toks = tokens(text)
        self.words = {stem(t) for t in toks}
        self.groups = groups_of(toks)
        if self.kind in ("spot", "place", "thing", "chest", "save point"):
            # "The shop counter" is where the seller is, not a seller.
            self.groups -= PEOPLE_ONLY
        if self.kind in ("animal",) and not self.groups & {"cat", "dog", "monster"}:
            self.groups.add("monster")
        if self.kind == "save point":
            self.groups.add("save")
        if self.kind == "chest":
            self.groups.add("chest")
        self.proper = {stem(w) for w in re.findall(r"\b[A-Z][a-z']+", text)
                       if w.lower() not in ("the", "a", "an")}
        spoken = re.sub(r"^[A-Z][\w' ]{0,20}:\s*", "", self.said)  # "Melchior: ..."
        if spoken != self.said:
            # The name the line is shown under: a name (Melchior), or what they are
            # ("Old man", "Soldier").
            who = tokens(self.said.split(":")[0])
            if groups_of(who):
                self.groups |= groups_of(who)
                self.words |= {stem(t) for t in who}
            else:
                self.proper |= {stem(t) for t in who}
        self.heard = {stem(t) for t in tokens(spoken) if t not in STOP and len(t) > 2}
        self.place = {stem(t) for t in tokens(self.where) if t not in STOP}

    @property
    def said_as(self) -> str:
        """What to call it out loud: its name, and where, when it's in the next place."""
        return self.name if self.here or not self.where else f"{self.name}, in {self.where}"


@dataclass
class Pick:
    key: str | None = None  # the one meant
    ask: str = ""  # a short question, when two or three fit about as well
    options: list[Who] = field(default_factory=list)
    why: str = ""

    @property
    def sure(self) -> bool:
        return self.key is not None


@dataclass
class Asked:
    """The player's words, taken apart."""
    who: list[str]  # what they are or are called ("old", "man")
    what: list[str]  # what they sell, do, wear or said ("swords", "lunch", "cap")
    near: list[str]  # what they're by ("bell")
    ways: list[str]  # left, lower, middle
    head: str | None  # the main word: the last of `who` ("man")
    nearest: bool  # "that soldier", "the nearest guard"
    talk: bool  # "talk to", "ask"


def wanted(said: str) -> str:
    """The someone or something in a request ("talk to the old man" -> "the old man"),
    or "" when it isn't that kind of request."""
    s = " ".join(said.split()).strip(" .!?,")
    m = ASKS_FOR.match(s)
    if not m:
        return ""
    rest = s[m.end():].strip()
    # "...and buy a potion", "...then come back": only the someone.
    rest = re.split(r"[,;]?\s*\b(?:and|then|so|to)\s+(?:then\s+)?(?:buy|sell|rest|ask|talk|"
                    r"see|come|go|get|check|say|find|save|open|use|pick|give|show|take|"
                    r"tell|sleep|stay|heal|look|read|back)\b", rest, maxsplit=1)[0]
    return rest.strip(" .!?,")


# After these, the words say what someone sells, does, has on or said.
WHAT = {"selling", "sells", "sell", "who", "that", "which", "with", "wearing", "holding",
        "carrying", "in"}


def parse(said: str) -> Asked:
    s = said.lower().replace("\u2019", "'")
    talk = bool(TALK_WORDS.search(s))
    m = LEAD.match(s)
    if m:
        s = s[m.end():]
    toks = tokens(s)
    nearest = any(t in NEAREST for t in toks[:2]) or any(
        t in ("nearest", "closest", "nearby") for t in toks)
    who: list[str] = []
    what: list[str] = []
    near: list[str] = []
    ways: list[str] = []
    head = None
    mode = "who"  # who they are; "what": what they sell, do, wear; "near": what they're by
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        after = toks[i + 2] if i + 2 < len(toks) else ""
        if t in ("up", "down") and nxt in ("to", "the", "by", "at"):
            continue  # "up the stairs", "down by the fountain"
        if t in POSITIONS:
            if t in ("front", "back") and nxt == "of":
                mode = "near"  # "in front of the tent": by the tent
                continue
            ways.append(t)
            continue
        if t in ("next", "close") and nxt in ("to", "by"):
            mode = "near"
            continue
        if t in NEAR or t in WHAT:
            if (nxt in ("the", "your", "my") and after in POSITIONS) or nxt in POSITIONS:
                continue  # "on the left": a way, not something to be by
            if t in WHAT and (who or head) and not (t == "in" and nxt in ("the", "a") and
                                                    after in ("middle", "center")):
                if t in ("selling", "sells", "sell"):
                    who.append("seller")
                # "the girl in green", "the guy selling swords", "the man who talked
                # about the war": what's theirs
                mode = "what" if t != "in" or mode == "who" else mode
                continue
            if t in NEAR:
                mode = "near"
                continue
        if t.endswith("ing") and len(t) > 5 and (who or head) and t not in STOP:
            mode = "what"  # "the guy eating lunch", "the man walking about"
            what.append(t)
            continue
        if t in STOP or t.isdigit():
            continue
        if mode == "near":
            near.append(t)
        elif mode == "what":
            what.append(t)
        else:
            who.append(t)
            head = t  # the main word: the last before "by", "selling", "who"
    return Asked(who, what, near, ways, head, nearest, talk)


def _word(w: str, c: Who, propers: set[str], specific: bool = False) -> float:
    """How much one of the player's words says it's c: + for a fit, - for a miss or a
    contradiction. specific: a word for what they sell, do or said ("lunch", "the war"),
    which tells one from another more than "guy" does."""
    s = stem(w)
    if s in c.proper:
        return 1.5 if specific else 5.0  # ("Melchior in Leene Square": where he is)
    if s in c.place:
        return 1.0  # "the shopkeeper up the north stairs": where they are
    if s in propers and not specific:
        return -2.0  # someone else's name
    gs = groups_of([w])
    if gs:
        weight = max(GROUPS[g][0] for g in gs)
        hit = gs & c.groups
        clash = any(frozenset((g, h)) in CLASH for g in gs for h in c.groups
                    if h not in gs)
        out = weight * len(hit) / len(gs) + (0.5 if hit and s in c.words else 0.0)
        if clash:
            out -= 3.0
        elif not hit and not any(gs & pair and not c.groups & pair for pair in KNOWN):
            out -= 0.25 * weight  # (who isn't known to be a man or a woman: no miss)
        return out
    if s in c.words:
        return 3.0 if specific else 1.5
    if s in c.heard:
        return 3.0 if specific else 1.0
    return -0.5


def _lies(way: str, c: Who, rivals: list[Who], mid: float | None) -> float:
    """How well c lies `way` (left, lower, middle) among those that fit, in its place:
    2 for the one furthest that way (and any within a step of it), 0.5 for the next, 0
    for the rest. The middle is the place's own (mid, across)."""
    near = [r for r in rivals if r.here and r.at is not None]
    if c.at is None or not c.here or len(near) < 2 or c not in rivals:
        return 0.0
    dx, dy = POSITIONS[way]

    def how(r: Who) -> float:
        if (dx, dy) == (0, 0):
            mx = mid if mid is not None else sorted(o.at[0] for o in near)[len(near) // 2]
            return -abs(r.at[0] - mx)
        return r.at[0] * dx + r.at[1] * dy

    vals = sorted({how(r) for r in near}, reverse=True)
    v = how(c)
    if vals[0] - v <= TILE:
        return 2.0
    second = next((x for x in vals if vals[0] - x > TILE), None)
    return 0.5 if second is not None and second - v <= TILE else 0.0


def _by(word: str, c: Who, marks: list[tuple[set[str], tuple[float, float], str]]) -> float:
    """0 to 2: how near c stands to something called `word` in its place."""
    if c.at is None or not c.here:
        return 0.0
    s = stem(word)
    d = min((math.dist(c.at, at) for words, at, name in marks
             if s in words and name != c.name), default=None)
    if d is None:
        return 0.0
    return 2.0 if d <= FAR * TILE else 1.0 if d <= 1.6 * FAR * TILE else 0.0


def score(asked: Asked, c: Who, marks, me: tuple[float, float] | None,
          propers: set[str], rivals: list[Who], mid: float | None = None) -> float:
    s = 0.0
    for w in asked.who:
        s += _word(w, c, propers)
    for w in asked.what:
        s += _word(w, c, propers, specific=True)
    for w in asked.near:
        lit = 1.5 if stem(w) in c.words else 0.0
        lit = max(lit, 1.0 if stem(w) in c.heard or stem(w) in c.place else 0.0)
        got = max(lit, _by(w, c, marks))
        s += got if got > 0 else -0.25
    for w in asked.ways:
        lit = 1.5 if w in c.words or any(
            POSITIONS.get(x) == POSITIONS[w] for x in c.words) else 0.0
        lit = max(lit, 1.5 if w in c.place and not c.here else 0.0)
        s += max(lit, _lies(w, c, rivals, mid))
    if asked.nearest and me is not None and c.here and c.at is not None and c in rivals:
        # "That soldier": the nearest of those that fit (and any about as near).
        near = [math.dist(r.at, me) for r in rivals if r.here and r.at is not None]
        if near and math.dist(c.at, me) - min(near) <= TILE:
            s += 2.0
    if c.here:
        s += 0.5
    if asked.talk and c.kind in ("spot", "place"):
        s -= 1.0
    if c.kind in ("spot", "place", "thing") and groups_of(asked.who) & (
            LIVING | {"guard", "seller", "innkeeper", "music", "mother", "father"}):
        s -= 1.0
    return s


def resolve(said: str, cands: list[Who], me: tuple[float, float] | None = None,
            marks: list[tuple[set[str], tuple[float, float], str]] | None = None,
            margin: float = MARGIN, mid: float | None = None) -> Pick:
    """Who or what of `cands` the player means by `said`: the one, or a short question
    naming the two or three that fit about as well, or none. me: where the player
    stands; marks: what's in the place to be "by" (landmarks()); mid: the place's
    middle, across (middle())."""
    if not cands:
        return Pick(why="nothing here to choose from")
    text = said.strip(" .!?,").lower()
    plain = re.sub(r"^(the|a|an)\s+", "", wanted(text) or text)

    def bare(n: str) -> str:
        return re.sub(r"^(the|a|an)\s+", "", n.lower())

    for c in cands:  # a name word for word
        if plain in {bare(n) for n in (c.name, c.key, *c.aka)} and (
                c.here or sum(plain == bare(o.name) for o in cands) <= 1):
            return Pick(c.key, why="named")
    asked = parse(said)
    if not (asked.who or asked.what or asked.near or asked.ways):
        return Pick(why="no one named")
    marks = marks or []
    propers = set().union(*(c.proper for c in cands))
    # Those that fit what they are (for left, lower, nearest among them).
    first = {c.key: sum(_word(w, c, propers) for w in asked.who) +
             sum(_word(w, c, propers, True) for w in asked.what) for c in cands}
    best = max(first.values())
    rivals = [c for c in cands if first[c.key] >= best - 2.0 and (
        first[c.key] > 0 or not (asked.who or asked.what))]
    scored = sorted(((score(asked, c, marks, me, propers, rivals, mid), c) for c in cands),
                    key=lambda p: (-p[0], math.dist(p[1].at, me) if me and p[1].at and
                                   p[1].here else 1e6))
    top, c = scored[0]
    if top < 1.0:
        return Pick(why=f"nothing here fits {said!r}")
    # The main word has to fit ("the purple monster" isn't the man at the purple tent);
    # a word nothing here knows ("the shop owner") leaves it to the word before.
    head = asked.head
    known = set().union(*(o.words | o.heard | o.proper | o.place for o in cands))
    if head is not None and not groups_of([head]) and stem(head) not in known:
        heads = [w for w in asked.who if groups_of([w]) or stem(w) in known]
        head = heads[-1] if heads else head
    if head is not None and _word(head, c, propers) < 0:
        return Pick(why=f"nothing here is {head!r}")
    close = [o for s, o in scored if top - s < margin]
    if c.here:
        # The best fit is in here: those next door only if named for it.
        close = [o for o in close if o.here]
    if len({o.name for o in close}) == 1:
        return Pick(close[0].key, why="fits best")  # the same name twice: the nearest
    options = close[:3]
    return Pick(None, question(options), options, why="two or more fit")


def question(options: list[Who]) -> str:
    names = [o.said_as for o in options]
    if len(names) == 2:
        return f"Do you mean {names[0]}, or {names[1]}?"
    return f"Do you mean {', '.join(names[:-1])}, or {names[-1]}?"


ORDINALS = {"first": 0, "1st": 0, "former": 0, "second": 1, "2nd": 1, "latter": -1,
            "third": 2, "3rd": 2, "last": -1}


def answer(said: str, options: list[Who], me: tuple[float, float] | None = None) -> str | None:
    """The option the player's answer to the question picks, or None."""
    toks = tokens(said)
    for t in toks:
        if t in ORDINALS and len(toks) <= 4:
            i = ORDINALS[t]
            return options[i].key if -len(options) <= i < len(options) else None
    got = resolve(said, options, me, margin=1.0)
    return got.key


def middle(rooms, here) -> float | None:
    """The middle of the player's place, across: of the ground that can be walked."""
    room = rooms.rooms.get(here.loc)
    if room is None or not room.walk:
        return None
    xs = sorted(c * TILE + TILE / 2 for _r, c in room.walk)
    return (xs[0] + xs[-1]) / 2


def directions(dx: float, dy: float) -> str:
    """How far and which way on the d-pad: "4 steps up and to your left"."""
    d = math.hypot(dx, dy)
    steps = max(1, round(d / TILE))
    if d < 1.5 * TILE:
        return "right next to you"
    if abs(dx) < 0.4 * abs(dy):
        way = "up" if dy < 0 else "down"
    elif abs(dy) < 0.4 * abs(dx):
        way = "to your left" if dx < 0 else "to your right"
    else:
        way = f"{'up' if dy < 0 else 'down'} and to your {'left' if dx < 0 else 'right'}"
    return f"{steps} step{'s' if steps != 1 else ''} {way}"


PEOPLE_KINDS = ("person", "animal")
THING_KINDS = ("thing", "save point", "chest")


def around(rooms, here, start: int = 0, most: int = 4) -> tuple[str, int]:
    """"Who's here?": the nearest people in the player's place first (from `start`,
    `most` of them), how far and which way each is on the d-pad, then the things and the
    ways out. (the answer, how many people there are)."""
    from .rooms import centre

    room = rooms.rooms[here.loc]
    me = (here.x, here.y)

    def far(at) -> float:
        return math.dist(at, me)

    people = sorted((p for p in room.people if p.kind in PEOPLE_KINDS), key=lambda p: far(p.at))
    n = len(people)
    parts = []
    if start == 0:
        parts.append(f"In {rooms.name(here.loc)}.")
    chunk = people[start:start + most]
    if chunk:
        lines = [f"{p.name}, {directions(p.at[0] - here.x, p.at[1] - here.y)}"
                 f"{', walks about' if p.moves else ''}" for p in chunk]
        lead = "Nearest: " if start == 0 else "Also here: "
        parts.append(lead + "; ".join(lines) + ".")
    elif start:
        parts.append("That's everyone here." if n else "No one's here.")
    else:
        parts.append("No one to talk to here.")
    left = n - start - len(chunk)
    if left > 0:
        parts.append(f"{left} more further off: say who else.")
    if start == 0:
        things = [(p.name, p.at) for p in room.people if p.kind in THING_KINDS]
        names = {t[0] for t in things}
        things += [(k, v) for k, v in room.spots.items() if k not in names]
        things.sort(key=lambda t: far(t[1]))
        if things:
            parts.append("Things: " + "; ".join(
                f"{k}, {directions(v[0] - here.x, v[1] - here.y)}" for k, v in things[:2])
                + ".")
        outs: dict[str, tuple[float, float]] = {}
        # One way out onto the world map: "the way out" ("outside, by Truce Residence"
        # names the map's nearest place, not this one).
        worlds = {e.label for e in room.exits if e.to in rooms.worlds}
        for e in room.exits:
            lab = e.label or rooms.name(e.to)
            if e.to in rooms.worlds and len(worlds) == 1:
                lab = "the way out"
            c = centre(e.frm)
            if lab not in outs or far(c) < far(outs[lab]):
                outs[lab] = c
        if outs:
            parts.append("Ways out: " + "; ".join(
                f"{k}, {directions(v[0] - here.x, v[1] - here.y)}"
                for k, v in sorted(outs.items(), key=lambda kv: far(kv[1]))[:4]) + ".")
    return " ".join(parts), n


def candidates(rooms, here, choices: dict, targets: dict) -> list[Who]:
    """Everyone and everything among the choices a walk can go to (rooms.options), as
    Whos: people and things with all the survey knows of them, ways out by their name."""
    from .rooms import centre

    out: list[Who] = []
    room = rooms.rooms.get(here.loc)
    for key in choices:
        t = targets.get(key)
        if t is not None:
            other = rooms.rooms.get(t.loc)
            p = next((p for p in (other.people if other else []) if p.name == t.name
                      and tuple(p.at) == tuple(t.at)), None)
            out.append(Who(key, t.name, t.loc, t.at,
                           p.kind if p is not None else "spot", t.loc == here.loc,
                           tuple(getattr(p, "aka", ()) or ()) if p else (),
                           getattr(p, "said", "") if p else "", bool(p and p.moves),
                           rooms.name(t.loc)))
        else:
            at = None
            if room is not None:
                ways = [e for e in room.exits if e.label == key]
                if ways:
                    at = min((centre(e.frm) for e in ways),
                             key=lambda c: math.dist(c, (here.x, here.y)))
            out.append(Who(key, key, choices[key][0], at, "place", True))
    return out


def landmarks(rooms, here) -> list[tuple[set[str], tuple[float, float], str]]:
    """What can be stood "by" in the player's place: things and spots, and people (as
    "by the old man"), with their words."""
    room = rooms.rooms.get(here.loc)
    if room is None:
        return []
    out = [({stem(t) for t in tokens(" ".join((p.name, *getattr(p, "aka", ()))))}, p.at,
            p.name) for p in room.people]
    out += [({stem(t) for t in tokens(k)}, v, k) for k, v in room.spots.items()]
    return out
