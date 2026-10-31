"""Chrono Trigger (SNES) battle memory, as offsets into the 128 KB of work RAM ($7E0000),
found by comparing memory in fights (scripts/ct_battle.py). `read(off, n) -> bytes` is
any reader, such as the emulator's READ_CORE_RAM.

- 0x117: 1 while a fight is on.
- 0x5E30 + 0x80 * slot: one block per fighter. Slots 0-2 are the party in party order,
  3-10 the enemies. +0 HP, +2 max HP, +4 MP, +6 max MP (16 bits each). An enemy slot
  not in this fight holds a stand-in with the bytes EMPTY at +0x34.
- 0x2980: the party's character numbers (0 Crono, 1 Marle, 2 Lucca, 3 Robo, 4 Frog,
  5 Ayla, 6 Magus); 0x80 and up is no one.
- 0x2600 + 0x50 * character: the character outside battle, +3 HP, +5 max HP, +7 MP,
  +9 max MP.
- 0x2400 the inventory's item numbers, 0x2500 how many of each.
"""

from __future__ import annotations

import time
from collections.abc import Callable

Read = Callable[[int, int], bytes]

BATTLE = 0x117
LOCATION = 0x100
STATS, SLOT = 0x5E30, 0x80
PARTY = 0x2980
CHARS, CHAR = 0x2600, 0x50
ITEMS, COUNTS = 0x2400, 0x2500
EMPTY = bytes.fromhex("0a010a640132017f")
NAMES = ["Crono", "Marle", "Lucca", "Robo", "Frog", "Ayla", "Magus"]


def w(b: bytes, o: int) -> int:
    return b[o] | b[o + 1] << 8


def in_battle(read: Read) -> bool:
    return read(BATTLE, 1)[0] == 1


def snapshot(read: Read) -> dict:
    """The fight (or the party outside one) as plain numbers."""
    fight = in_battle(read)
    ids = [i for i in read(PARTY, 3) if i < len(NAMES)]
    stats = read(STATS, SLOT * 11)
    chars = read(CHARS, CHAR * len(NAMES))
    party = []
    for n, i in enumerate(ids):
        if fight:
            o = n * SLOT
            hp, mhp, mp, mmp = (w(stats, o + k) for k in (0, 2, 4, 6))
        else:
            o = i * CHAR
            hp, mhp, mp, mmp = (w(chars, o + k) for k in (3, 5, 7, 9))
        party.append({"name": NAMES[i], "hp": hp, "max_hp": mhp, "mp": mp, "max_mp": mmp})
    enemies = []
    for slot in range(3, 11):
        o = slot * SLOT
        if stats[o + 0x34:o + 0x3C] != EMPTY:
            enemies.append({"slot": slot, "hp": w(stats, o), "max_hp": w(stats, o + 2)})
    counts = read(COUNTS, 0x100)
    return {"t": round(time.time(), 2), "battle": fight,
            "location": w(read(LOCATION, 2), 0) & 0x1FF, "party": party,
            "enemies": enemies, "items": sum(counts)}


def battle(read: Read) -> dict | None:
    return snapshot(read) if in_battle(read) else None
