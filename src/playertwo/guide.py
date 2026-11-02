"""A walkthrough to look things up in: where a place is and how to get there, what to do
next. Blind players already play with text guides; this reads one so they don't have to.

Guides are plain text files the player puts in ~/.config/player-two/guides/, named after
the game ("chrono-trigger.txt"), with blank lines between paragraphs; a line starting
with a number and a dot ("3. The Queen is Gone") starts a chapter. A search is plain
word matching, rarer words counting more, so it needs no index or model. Boss stat
tables, move names and enemy lists are left out: they match names but say nothing about
where to go. A short heading with a place in brackets ("Guardia Castle (1000 A.D.)")
labels the passages under it.
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
SECTION = re.compile(r"^[A-Z][^.!?:]{2,50}\(\d[^)]*[A-Za-z][^)]*\)$")  # "Lab 16 (2300 A.D.)"
STATS = re.compile(r"\bLevel HP\b|Tech Points|Treasures:|^Combo: |\bDef\. ")


def noise(para: str) -> bool:
    """Not walkthrough prose: a stat table, an enemy or treasure list, a boss or move
    name, a "Chapters" link."""
    if STATS.search(para):
        return True
    digits = sum(c.isdigit() for c in para)
    if digits > 12 and digits > len(para) * 0.12:
        return True
    return len(para) < 60 and not re.search(r"[.!?]$", para)


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 1]


def slug(name: str) -> str:
    return "-".join(words(re.sub(r"\(.*?\)", "", name)))


def guide_path(game: str, folder: Path = GUIDES) -> Path | None:
    """The guide whose file name is this game's name, or is all in it."""
    if not folder.is_dir():
        return None
    name = slug(game)
    for f in sorted(folder.glob("*.txt")):
        if f.stem == name or (f.stem and all(w in name.split("-") for w in f.stem.split("-"))):
            return f
    return None


def chapters(text: str) -> list[tuple[str, str]]:
    """The guide cut into its numbered chapters: (title, text). Chapter numbers go up one
    at a time, so a numbered list inside a chapter ("1. Rename characters at...") isn't
    taken for one."""
    out: list[tuple[str, list[str]]] = []
    want = 1
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\.\s+(\S.*)$", line)
        if m and int(m.group(1)) == want and len(line.strip()) < 70:
            out.append((line.strip(), []))
            want += 1
        elif out:
            out[-1][1].append(line)
        elif line.strip():
            out.append(("", [line]))
    return [(title, "\n".join(body).strip()) for title, body in out]


def prose(text: str) -> str:
    """A chapter's text without stat tables and lists (see noise), paragraphs kept."""
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paras if p and (SECTION.match(p) or not noise(p)))


@dataclass
class Passage:
    chapter: str
    text: str


class Guide:
    def __init__(self, text: str):
        self.passages: list[Passage] = []
        chapter = section = ""
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if not para:
                continue
            if CHAPTER.match(para) and len(para) < 80:
                chapter, section = para, ""
                continue
            if SECTION.match(para):
                section = para
                continue
            if noise(para):
                continue
            self.passages.append(Passage(f"{chapter} / {section}" if section else chapter,
                                         para))
        n = len(self.passages) or 1
        df: dict[str, int] = {}
        self._words = [words(p.text) for p in self.passages]
        for ws in self._words:
            for w in set(ws):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}

    @classmethod
    def for_game(cls, game: str, folder: Path = GUIDES) -> Guide | None:
        f = guide_path(game, folder)
        return cls(f.read_text(encoding="utf-8", errors="replace")) if f else None

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
