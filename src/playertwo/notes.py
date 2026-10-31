"""What the agent learned about each game, kept between sessions: which button opens the
menu, how battles go, where things are, what the player's character looks like. A
sighted friend remembers these from last time; without them every session starts over.

One plain text file a game, a line a note, in ~/.config/player-two/games/, so the player
(or a friend) can read and fix them.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import CONFIG_DIR

MAX_NOTES = 40


def _slug(game: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", game.lower()).strip("-") or "game"


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


class GameNotes:
    def __init__(self, folder: Path | None = None):
        self.folder = folder or CONFIG_DIR / "games"

    def path(self, game: str) -> Path:
        return self.folder / f"{_slug(game)}.txt"

    def read(self, game: str) -> list[str]:
        if not game:
            return []
        try:
            text = self.path(game).read_text(encoding="utf-8")
        except OSError:
            return []
        return [line.strip()[2:].strip() if line.strip().startswith("- ") else line.strip()
                for line in text.splitlines() if line.strip()]

    def add(self, game: str, note: str, instead_of: str = "") -> str:
        """Keep `note` for `game`. `instead_of` names an older note it corrects, which
        goes; the same note again isn't kept twice. Notes that only sound alike ("the
        inn is north of the square", "the shop is north of the square") both stay."""
        note = " ".join(note.split())[:300]
        if not game or not note:
            return "nothing to remember"
        notes = self.read(game)

        def like(a: str, b: str) -> float:
            wa, wb = _words(a), _words(b)
            return len(wa & wb) / max(len(wa | wb), 1)

        kept = [n for n in notes if like(n, note) < 0.9]
        if instead_of.strip():
            old = max(kept, key=lambda n: like(n, instead_of), default=None)
            if old is not None and like(old, instead_of) >= 0.5:
                kept.remove(old)
        replaced = len(kept) < len(notes)
        kept = [*kept, note][-MAX_NOTES:]
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path(game).write_text("".join(f"- {n}\n" for n in kept), encoding="utf-8")
        return "updated that note" if replaced else "remembered"

    def pack(self, game: str) -> str:
        """How the game works from its game pack (<game>.pack.txt beside the notes: its
        controls, battle menus), written once for everyone who plays it."""
        if not game:
            return ""
        # One pack for every version: "Chrono Trigger (SNES)" uses chrono-trigger's.
        for name in (_slug(game), _slug(re.sub(r"\(.*?\)", "", game))):
            try:
                return self.folder.joinpath(f"{name}.pack.txt").read_text(
                    encoding="utf-8").strip()
            except OSError:
                continue
        return ""

    def prompt(self, game: str) -> str:
        notes = self.read(game)
        pack = self.pack(game)
        parts = []
        if pack:
            parts.append("How this game works, from its game pack (trust the screen over "
                         "it if they disagree):\n" + pack)
        if notes:
            parts.append("What you learned about this game in earlier sessions (trust the "
                         "screen over these if they disagree, and correct a wrong one with "
                         "remember):\n" + "\n".join(f"- {n}" for n in notes))
        return "\n\n".join(parts)
