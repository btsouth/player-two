"""Whether a line the model wrote is about the player's game and request, before it's said.

Once in about 60 requests (live 3), the model answered "what's the battle mode set to?" with a
sentence about football fouls, and it was spoken as is. A line the model writes is said only
when enough of its words are grounded: found in what the model was given this request (the
player's words, the recent requests, the tool results, the game's pack and notes, the
instructions) or in the few words a game helper always uses. A line that isn't is checked once
more, by a quick question to the model with the request beside it (agent.py), and never said
when that says no. Measured offline on every agent line in the live logs: see
scripts/ontopic_bench.py.
"""

from __future__ import annotations

import re

# Words that say nothing of what a line is about.
STOP = set("""
a an the and or but if then than so as of to in on at by for with from into onto off up down
out over under about after before again once all any both each few more most other some such
only own same too very just now still already yet also even ever never not no nor here there
where when why how what which who whom whose this that these those it its it's itself i i'm
i'll i've i'd me my mine myself you you're you'll you've your yours we we're we'll we've our
ours us they they're they'll their them he he's him his she she's her hers is are was were
be been being am have has had having do does did doing done can can't cannot could couldn't
will won't would wouldn't shall should shouldn't may might must let let's get gets got getting
go goes going went gone come comes came make makes made take takes took see sees saw seen
look looks looked looking say says said tell told want wants need needs like know knows
think one two three four five six seven eight nine ten first last next back right left
okay ok yes sure well oh hey there's that's what's here's isn't aren't wasn't weren't don't
doesn't didn't haven't hasn't hadn't
""".split())

# What any line from a game helper may talk about, in any game, without the screen or
# the request saying it first.
GAME = set("""
game play player playing level hp mp health heal heals healed healing hurt damage hit hits
fight fights fighting fought battle battles enemy enemies foe foes monster monsters attack
attacked attacks win won lose lost run ran away escape flee party character characters turn
turns ready queued menu menus screen box text item items gold money buy bought sell sold shop
shopping seller store potion potions save saved saving load map world place places room door
doors way path walk walked walking stand standing stood talk talked talking said asked answer
north south east west up down inside outside near nearby beside behind ahead front far close
stairs exit entrance town city village castle forest cave house inn square fair full low knocked
experience exp points tech techs skill skills magic spell setting settings option options mode
mode wait active button buttons press pressed controller pad cursor selected select pick picked
choose chose choice question yes no done finished stop stopped try tried trying again sorry
couldn't can't unable time moment second seconds minute
""".split())

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def stem(w: str) -> str:
    """A rough stem: "potions" and "potion", "healed" and "heal", "walking" and "walk"."""
    w = w.replace("’", "'")
    w = re.sub(r"'(s|re|ll|ve|d|m|t)$", "", w)
    if w.isdigit() or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and not w.endswith("ss"):
            return w[: -len(suffix)]
    return w


def content(text: str) -> list[str]:
    """The words of a line that say what it's about (stemmed)."""
    out = []
    for w in WORD.findall(text.lower().replace("’", "'")):
        if w in STOP or (len(w) < 3 and not w.isdigit()):
            continue
        out.append(stem(w))
    return out


class Grounds:
    """What the model was given this request, as stems: the player's words, recent
    requests, tool results, what the screen says, the instructions and the game pack,
    with the words a game helper always uses."""

    def __init__(self, *texts: str):
        self.vocab: set[str] = {stem(w) for w in GAME}
        self.add(*texts)

    def add(self, *texts: str) -> None:
        for t in texts:
            if t:
                self.vocab.update(stem(w) for w in WORD.findall(t.lower().replace("\u2019",
                                                                                  "'")))

    def grounding(self, line: str) -> tuple[float, list[str]]:
        """The share of a line's words found in what it was given, and the ones that
        weren't."""
        words = content(line)
        if not words:
            return 1.0, []
        loose = [w for w in words if w not in self.vocab]
        return 1.0 - len(loose) / len(words), loose

    def off_topic(self, line: str) -> bool:
        """Whether a line looks unrelated to the game and the request: a longer one with
        few of its words grounded, a short one with none."""
        words = content(line)
        share, _loose = self.grounding(line)
        if len(words) < MIN_WORDS:
            return len(words) >= 2 and share == 0.0
        return share < MIN_SHARE


# A line with at least this many words, fewer than this share of them grounded, gets the
# model's second look before it's said (scripts/ontopic_bench.py measures it).
MIN_WORDS = 4
MIN_SHARE = 0.6
# The model's look failed (no answer in time): said only when at least this share is
# grounded (the football line: 0.2).
FALLBACK_SHARE = 0.4


async def second_look(muse, game: str, request: str, line: str, timeout: float = 4.0
                      ) -> bool | None:
    """The model's yes or no on whether a line held back is a reply to the request (about
    a second, text only); None when it didn't answer in time."""
    import asyncio

    prompt = RELEVANT.format(game=game or "a video game", request=request.replace('"', "'"),
                             line=line.replace('"', "'"))
    try:
        d = await asyncio.wait_for(muse.json([{"role": "user", "content": prompt}],
                                             effort="minimal", max_tokens=400), timeout)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(d, dict) or not isinstance(d.get("game"), bool):
        return None
    return d["game"]


RELEVANT = """\
A voice helper plays {game} with a player who can't see the screen. The player said: \
"{request}". The helper is about to say: "{line}"
Is that line about the game or what the helper is doing in it (any place, person, item, \
fight, menu, action or state, even if it's unexpected or wrong)? Or is it about something \
with no connection to the game at all (sports rules, cooking, the news, a text from \
elsewhere)? Answer with JSON: {{"game": true}} if it's about the game, {{"game": false}} if \
it's unrelated."""
