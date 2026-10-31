"""A walkthrough to look things up in: where a place is and how to get there, what to do
next. Blind players already play with text guides; this reads one so they don't have to.

Guides are plain text files the player puts in ~/.config/player-two/guides/, named after
the game ("chrono-trigger.txt"), with blank lines between paragraphs; a line starting
with a number and a dot ("3. The Queen is Gone") starts a chapter. A search is plain
word matching, rarer words counting more, so it needs no index or model.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG_DIR

GUIDES = CONFIG_DIR / "guides"
STOP = set("""a an and are as at be but by can do does for from get go going how i if in into is
it its me my of on or so than that the then there this to up was what when where
which who why will with you your""".split())  # noqa: SIM905
CHAPTER = re.compile(r"^\d+\.\s+\S")


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 1]


def slug(name: str) -> str:
    return "-".join(words(re.sub(r"\(.*?\)", "", name)))


@dataclass
class Passage:
    chapter: str
    text: str


class Guide:
    def __init__(self, text: str):
        self.passages: list[Passage] = []
        chapter = ""
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if not para:
                continue
            if CHAPTER.match(para) and len(para) < 80:
                chapter = para
                continue
            self.passages.append(Passage(chapter, para))
        n = len(self.passages) or 1
        df: dict[str, int] = {}
        self._words = [words(p.text) for p in self.passages]
        for ws in self._words:
            for w in set(ws):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}

    @classmethod
    def for_game(cls, game: str, folder: Path = GUIDES) -> Guide | None:
        """The guide whose file name is this game's name, or is all in it."""
        if not folder.is_dir():
            return None
        name = slug(game)
        for f in sorted(folder.glob("*.txt")):
            if f.stem == name or (f.stem and all(w in name.split("-")
                                                 for w in f.stem.split("-"))):
                return cls(f.read_text(encoding="utf-8", errors="replace"))
        return None

    def search(self, query: str, limit: int = 5, max_chars: int = 2400) -> list[Passage]:
        q = words(query)
        if not q:
            return []
        pairs = list(itertools.pairwise(q))  # place names are often two words
        scored = []
        for i, ws in enumerate(self._words):
            if not ws:
                continue
            have = set(ws)
            s = sum(self.idf.get(w, 0) for w in set(q) if w in have)
            text = " ".join(ws)
            s += sum(2 * (self.idf.get(a, 0) + self.idf.get(b, 0)) / 2
                     for a, b in pairs if f"{a} {b}" in text)
            if s:
                scored.append((s / (1 + len(ws) / 400), i))
        out, used = [], 0
        for _, i in sorted(scored, reverse=True)[:limit]:
            p = self.passages[i]
            text = clip(p.text, q, max_chars // limit)
            if used + len(text) > max_chars and out:
                break
            out.append(Passage(p.chapter, text))
            used += len(text)
        return out


def clip(text: str, q: list[str], n: int) -> str:
    """Up to about n characters of text, around where most of the query's words are."""
    if len(text) <= n:
        return text
    lower = text.lower()
    hits = [m.start() for w in q for m in re.finditer(re.escape(w), lower)]
    mid = sorted(hits)[len(hits) // 2] if hits else 0
    start = max(0, min(mid - n // 2, len(text) - n))
    return ("..." if start else "") + text[start:start + n].strip() + (
        "..." if start + n < len(text) else "")


def look_up(game: str, query: str) -> str:
    guide = Guide.for_game(game)
    if guide is None:
        return (f"no guide for {game}: the player can put a text walkthrough in "
                f"{GUIDES} named {slug(game)}.txt")
    found = guide.search(query)
    if not found:
        return "nothing in the guide about that; try other words (place or person names)"
    return "From the guide (it follows the story in order; you may be at another point):\n" \
        + "\n\n".join(f"[{p.chapter or 'start'}] {p.text}" for p in found)
