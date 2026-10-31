"""The battle judge (scripts/battle_judge.py) on made-up memory readings, and the
Chrono Trigger memory reader (scripts/ct_ram.py) on a made-up memory."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import battle_judge as J  # noqa: E402
import ct_ram  # noqa: E402


def reading(battle, party, enemies, items=10):
    return {"battle": battle, "items": items,
            "party": [{"name": n, "hp": hp, "max_hp": 100, "mp": mp, "max_mp": 20}
                      for n, hp, mp in party],
            "enemies": [{"slot": s, "hp": hp, "max_hp": 30} for s, hp in enemies]}


def test_won_and_alive():
    s = [reading(True, [("Crono", 100, 8)], [(3, 30), (4, 30)]),
         reading(True, [("Crono", 90, 8)], [(3, 0), (4, 30)]),
         reading(True, [("Crono", 90, 8)], [(3, 0), (4, 0)]),
         reading(False, [("Crono", 90, 8)], [(3, 0), (4, 0)], items=11)]
    ok, why = J.judge(s, ["won", "alive", "over", "hit:3", "all_hit"])
    assert ok, why
    assert not J.judge(s, ["fled"])[0]
    assert J.judge(s, ["acts<=1"])[0]  # two readings in a row: one action
    apart = s[:2] + s[1:2] + s[2:]  # a quiet reading between the two hits
    assert not J.judge(apart, ["acts<=1"])[0]
    assert J.judge(apart, ["acts<=2"])[0]
    assert not J.judge(s, ["item"])[0]  # an item dropped is no item used


def test_fled_and_game_over():
    fled = [reading(True, [("Crono", 100, 8)], [(3, 30)]),
            reading(False, [("Crono", 100, 8)], [(3, 30)])]
    assert J.judge(fled, ["fled", "alive", "acts<=0"])[0]
    lost = [reading(True, [("Crono", 10, 8)], [(3, 30)]),
            reading(True, [("Crono", 0, 8)], [(3, 30)]),
            reading(False, [("Crono", 0, 8)], [(3, 30)])]
    ok, why = J.judge(lost, ["fled"])
    assert not ok and "lost" in why
    assert not J.judge(lost, ["alive"])[0]


def test_heal_tech_item():
    s = [reading(True, [("Crono", 100, 8), ("Marle", 40, 20)], [(7, 30)]),
         reading(True, [("Crono", 100, 8), ("Marle", 40, 18)], [(7, 30)]),
         reading(True, [("Crono", 100, 8), ("Marle", 90, 18)], [(7, 30)], items=9)]
    assert J.judge(s, ["healed:Marle", "mp:Marle", "item", "in_battle"])[0]
    assert not J.judge(s, ["healed:Crono"])[0]
    assert not J.judge(s, ["mp:Crono"])[0]
    with pytest.raises(ValueError):
        J.judge(s, ["nonsense"])


def test_snapshot_reads_the_blocks():
    mem = bytearray(0x20000)
    mem[ct_ram.BATTLE] = 1
    mem[ct_ram.LOCATION:ct_ram.LOCATION + 2] = (0x77).to_bytes(2, "little")
    mem[ct_ram.PARTY:ct_ram.PARTY + 3] = bytes([0, 1, 0x80])
    for slot in range(11):
        o = ct_ram.STATS + slot * ct_ram.SLOT
        mem[o + 0x34:o + 0x3C] = ct_ram.EMPTY
    for slot, hp in ((0, 232), (1, 45), (7, 30)):
        o = ct_ram.STATS + slot * ct_ram.SLOT
        mem[o:o + 4] = hp.to_bytes(2, "little") * 2
        if slot == 7:
            mem[o + 0x34:o + 0x3C] = bytes(8)
    mem[ct_ram.COUNTS:ct_ram.COUNTS + 2] = bytes([3, 4])
    s = ct_ram.snapshot(lambda off, n: bytes(mem[off:off + n]))
    assert s["battle"] and s["location"] == 0x77 and s["items"] == 7
    assert [(p["name"], p["hp"]) for p in s["party"]] == [("Crono", 232), ("Marle", 45)]
    assert s["enemies"] == [{"slot": 7, "hp": 30, "max_hp": 30}]
