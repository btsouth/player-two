"""An instant spoken acknowledgement for a request, with no model call.

The model's first words come seconds after a request, and an action request could be silent
for a minute. This says straight away that the request was heard, in the player's own words
(so a misheard one is caught at once). It states intent only, never an effect.
"""
from __future__ import annotations

import re

LOOK = "Let me look."
FIGHT_LINE = "Okay, I'll take this fight."
RUN_LINE = "Okay, trying to run."

QUESTION = re.compile(r"^(what|where|who|which|how|why|is|are|am|do (you|we|i|they)|does|did|can you see|"
                      r"read|tell me|describe)\b|\?$")
FIGHT = re.compile(r"\b(win|finish|handle|take) (this|the) (fight|battle)\b|fight for me|"
                   r"keep (attacking|fighting)")
RUN = re.compile(r"^(run|flee|escape|get away)( away| from (this|the) (fight|battle))?\b")
# Trips say "Going to X." within seconds by themselves; an echo before it is just noise.
TRIP = re.compile(r"^(take (me|us)|go|let'?s go|head|bring (me|us)|fly|travel)\b")
VERB = re.compile(r"^(heal|use|cast|attack|hit|open|cross|talk to|save|equip|buy|sell|"
                  r"pick up|select|climb|enter|leave|revive|cure)\b(.*)")
POLITE = re.compile(r"^(please |can you |could you |would you |can we |i want to |"
                    r"i need to |let's |lets )+")
SWAP = ((r"\bme\b", "you"), (r"\bmy\b", "your"), (r"\bus\b", "you"), (r"\bour\b", "your"))


def ack(request: str) -> str | None:
    """The line to say now, or None to stay quiet (the request answers itself quickly)."""
    text = " ".join(request.split()).strip(" .!")
    text = re.sub(r"(?i)^(hey )?player two,?\s*", "", text)
    low = text.lower()
    if not low:
        return None
    if QUESTION.search(low):
        return LOOK
    if FIGHT.search(low):
        return FIGHT_LINE
    if RUN.search(low):
        return RUN_LINE
    body = POLITE.sub("", low)
    if TRIP.match(body):
        return None
    who = re.match(r"^([A-Za-z]+), (.+)$", text)  # "Crono, use Cyclone"
    if who:
        text = who.group(2)
    words = POLITE.sub("", text.lower())
    m = VERB.match(words)
    if not m or len(words) > 70:
        return "On it."
    rest = text[len(text) - len(m.group(2)):].strip() if m.group(2) else ""
    for a, b in SWAP:
        rest = re.sub(a, b, rest, flags=re.IGNORECASE)
    action = f"{m.group(1)} {rest}".strip()
    if who:
        return f"Okay, {who.group(1).capitalize()} will {action}."
    return f"Okay, I'll {action}."
