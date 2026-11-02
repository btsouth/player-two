"""Spoken presentation only: keep facts, numbers and quoted game words intact."""
import re


def companion_line(text: str) -> str:
    if text in {"Let me get a look at it.", "I'll just go for damage.",
                "Going for damage again.", "More damage.", "On it.",
                "No target line this turn, so I'll just go for damage."}:
        return ""
    if text == "That's the fight. We won.":
        return "We've won this fight!"
    # Older dialogue producers put a comma after these speakers. Attribute the
    # quotation explicitly without changing a word inside it.
    m = re.fullmatch(r"(Soldier|Marle), (.+)", text)
    if m:
        who = "The soldier" if m[1] == "Soldier" else m[1]
        return f'{who} says, "{m[2]}"'
    text = re.sub(r"^(\w+)'s at (\d+) HP, (.+) are full\.$",
                  r"\1's at \2 HP; \3 are at full health.", text)
    text = re.sub(r"^We made it to ([^,]+), but (.+?), and (.+?), so (.+)\.$",
                  r"We reached \1, but \2. \3, so \4.", text)
    text = re.sub(r"^Hit the (.+) with the (.+), the (.+) with the (.+), and the (.+) with the (.+)\.$",
                  r"Use the \2 on its \1, the \4 on its \3 and the \6 on its \5.", text)
    text = re.sub(r"^Your Journal is open on (.+), for (.+)\.$",
                  r"Your journal's open to the \1 for \2.", text)
    text = re.sub(r"^(\w+) uses ([^.]+)\.$", r"\1's using \2.", text)
    text = re.sub(r"^(\w+)'s level increased!$", r"\1 went up a level!", text)
    text = re.sub(r"^Won the fight: (.+)\.$", r"We won! You earned \1.", text)
    text = re.sub(r"^One (.+) down, (one|two|[0-9]+) left\.$",
                  r"One \1's down; \2 left.", text)
    return text
