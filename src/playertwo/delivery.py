"""Spoken presentation only: keep facts, numbers and quoted game words intact."""
import re

LINES = {
    "Frog's at 1 HP, Crono and Marle are full.":
        "Frog's at 1 HP; Crono and Marle are at full health.",
    "That's the fight. We won.": "We've won this fight!",
    "We made it to Guardia Castle, but the guard stopped you, and Marle said to check out the fair instead, so we're back on the world map.":
        "We reached Guardia Castle, but the guard stopped you. Marle wants to visit the fair instead, so we're back on the world map.",
    "Hit the head with the harp, the legs with the hammer, and the tail with the sword.":
        "Use the harp on its head, the hammer on its legs and the sword on its tail.",
    "Your Journal is open on Brief, for Can a Swindler Change His Spots.":
        "Your journal's open to the Brief for Can a Swindler Change His Spots.",
    "Soldier, Stop right there!": 'The soldier says, "Stop right there!"',
    "Marle, Where are you going? C'mon, let's check out the fair!":
        'Marle says, "Where are you going? C\'mon, let\'s check out the fair!"',
}


def companion_line(text: str) -> str:
    text = LINES.get(text, text)
    text = re.sub(r"^(\w+) uses ([^.]+)\.$", r"\1's using \2.", text)
    text = re.sub(r"^(\w+)'s level increased!$", r"\1 went up a level!", text)
    text = re.sub(r"^Won the fight: (.+)\.$", r"We won! You earned \1.", text)
    text = re.sub(r"^One (.+) down, (one|two|[0-9]+) left\.$",
                  r"One \1's down; \2 left.", text)
    return text
