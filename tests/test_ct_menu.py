"""Chrono Trigger (Steam) field menus: the executor against a fake menu, and its reading
helpers on live frames (skipped where they aren't on disk)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

from playertwo import ctsteam
from playertwo.glyphread import Run
from playertwo.menu import BATTLE, MAIN, SETTINGS, FieldMenu

PARTY = ["Crono", "Frog", "Marle"]
ITEMS = ["Potion", "Mid-Potion", "Hi-Potion", "Mid-Ether", "Panacea", "Athenian Water",
         "Shelter", "Lapis", "Hi-Ether", "Ether"]
GREEN = {"Potion", "Mid-Potion", "Hi-Potion", "Lapis", "Hi-Ether", "Ether"}


def run(text, x, y, colour="white"):
    return Run(text, x, x + 20 * len(text), y, colour, 0.0)


class FakeMenu:
    def __init__(self):
        self.screen = "field"
        self.i = 0
        self.hp = {"Crono": 175, "Frog": 494, "Marle": 121}
        self.log: list[str] = []
        self.speed = 5
        self.swaps = 0

    async def press(self, b, ms):
        self.log.append(b)
        s = self.screen
        if s == "field":
            if b == "y":
                self.screen, self.i = "main", 0
            return
        if b == "b":
            back = {"main": "field", "inv_top": "main", "items": "inv_top",
                    "pick": "items", "set_left": "main", "set_right": "set_left",
                    "users": "main", "techs": "users"}
            self.screen, self.i = back.get(s, "field"), 0
            return
        n = {"main": len(MAIN), "items": len(ITEMS), "pick": 3, "set_left": len(SETTINGS),
             "set_right": len(BATTLE), "users": 3, "techs": 7}.get(s, 1)
        if b in ("up", "down"):
            step = 2 if s == "items" else 1
            self.i = max(0, min(n - 1, self.i + (step if b == "down" else -step)))
        elif b in ("left", "right") and s == "items":
            self.i = self.i + 1 if b == "right" and self.i % 2 == 0 else (
                self.i - 1 if b == "left" and self.i % 2 else self.i)
        elif b in ("left", "right") and s == "set_right" and BATTLE[self.i] == "Battle Speed":
            self.speed = max(1, min(8, self.speed + (1 if b == "right" else -1)))
        elif b == "a":
            if s == "main":
                self.screen = {"Inventory": "inv_top", "Settings": "set_left",
                               "Techs": "users"}.get(MAIN[self.i], s)
                self.i = 0
            elif s == "inv_top":
                self.screen = "items"
            elif s == "items":
                if ITEMS[self.i] in GREEN:
                    self.item, self.screen, self.back_i = ITEMS[self.i], "pick", self.i
                    self.i = 0
                else:
                    self.swaps += 1
            elif s == "pick":
                who = PARTY[self.i]
                self.hp[who] += 50  # the picker stays up with the new HP
            elif s == "set_left" and SETTINGS[self.i] == "Battle":
                self.screen, self.i = "set_right", 0

    async def grab(self):
        return None

    def read(self, _frame):
        s, i = self.screen, self.i
        lines, tips = [], []
        if s in ("main", "inv_top", "set_left", "users"):
            for k, w in enumerate(MAIN):
                lines.append([run(w, 339, 100 + 80 * k)])
            for k, n in enumerate(PARTY):
                lines.append([run(n, 1070, 132 + 400 * k)])
        if s == "main":
            tips = [(296, 100 + 80 * i)]
        elif s == "inv_top":
            lines.append([run("Use / Switch", 330, 60), run("Sort list", 1500, 60)])
            tips = [(296, 60)]
        elif s in ("items", "pick"):
            for k, w in enumerate(ITEMS):
                lines.append([run(w, 330 + 1170 * (k % 2), 340 + 64 * (k // 2),
                                  "green" if w in GREEN else "white")])
            j = self.back_i if s == "pick" else i
            tips = [(296 + 1170 * (j % 2), 340 + 64 * (j // 2))]
            if self.swaps:
                tips.append((900, 900))
            if s == "pick":
                for k, n in enumerate(PARTY):
                    y = 144 + 380 * k
                    lines += [[run(n, 1615, y)],
                              [run(f"HP : {self.hp[n]}/ 999", 1614, y + 61)]]
                tips = [(1384, 224 + 380 * i)]
        elif s == "set_left":
            lines = [[run(w, 339, 70 + 80 * k)] for k, w in enumerate(SETTINGS)]
            tips = [(296, 70 + 80 * i)]
        elif s == "set_right":
            lines = [[run(w, 339, 70 + 80 * k)] for k, w in enumerate(SETTINGS)]
            for k, w in enumerate(BATTLE):
                y = 70 + 246 * k
                lines.append([run(w, 1533, y)])
                val = str(self.speed) + " (NORMAL)" if w == "Battle Speed" else "WAIT"
                lines.append([run(val, 1550, y + 109)])
            tips = [(1164, 70 + 246 * i)]
        return [SimpleNamespace(tip=t) for t in tips], lines


def menu(game):
    return FieldMenu(game.press, game.grab, read=game.read, gap_s=0, wait_s=0.2,
                     party=PARTY)


async def test_an_item_on_someone_and_the_menu_closed():
    game = FakeMenu()
    out = await menu(game).use("item", "Hi-Ether", "", "Frog")
    assert out.startswith("used Hi-Ether on Frog: Frog HP 494/999 to 544/999"), out
    assert game.screen == "field" and game.swaps == 0
    assert game.log[:3] == ["y", "down", "a"]


async def test_an_item_that_cant_be_used_here_never_starts_a_swap():
    game = FakeMenu()
    out = await menu(game).use("item", "Shelter", "", "Frog")
    assert out.startswith("stopped: Shelter can't be used here"), out
    assert game.swaps == 0 and game.screen == "field"


async def test_battle_speed_to_the_slowest_reads_back():
    game = FakeMenu()
    out = await menu(game).setting("Battle Speed", "slowest")
    assert out == "Battle Speed is now 8", out
    assert game.speed == 8 and game.screen == "field"


LIVE = Path(os.environ.get("PLAYER_TWO_CT_FRAMES",
                           "~/.cache/player-two/live/steam-2026-11-01")).expanduser()


@pytest.mark.skipif(not (LIVE / "frames").is_dir(), reason="no live Chrono Trigger frames")
@pytest.mark.parametrize(("frame", "who", "tech"), [
    ("140416-04", "Frog", "Slurp"), ("170901-08", "Marle", "Aura"),
    ("140419-05", "Frog", "Heal"), ("170905-09", "Marle", "Cure")])
def test_the_highlighted_tech_is_told_from_its_neighbours(frame, who, tech):
    hands, lines = ctsteam.menu_read(cv2.imread(str(LIVE / "frames" / f"{frame}.jpg")))
    tip = max(h.tip for h in hands)
    assert FieldMenu.in_order(lines, tip, ctsteam.TECHS[who]) == tech


@pytest.mark.skipif(not (LIVE / "frames").is_dir(), reason="no live Chrono Trigger frames")
@pytest.mark.parametrize(("frame", "want"), [
    ("170836-00", "Equipment"), ("135431-01", "Techs"), ("135434-02", "Inventory")])
def test_the_main_menu_entry(frame, want):
    hands, lines = ctsteam.menu_read(cv2.imread(str(LIVE / "frames" / f"{frame}.jpg")))
    assert FieldMenu.in_order(lines, min(h.tip for h in hands), MAIN) == want


@pytest.mark.skipif(not (LIVE / "frames").is_dir(), reason="no live Chrono Trigger frames")
def test_a_highlighted_block_is_the_one_name_missing():
    fm = FieldMenu(None, None, party=PARTY)
    hands, lines = ctsteam.menu_read(cv2.imread(str(LIVE / "frames" / "170850-05.jpg")))
    assert fm.who(lines, min(h.tip for h in hands)) == "Frog"
    hands, lines = ctsteam.menu_read(cv2.imread(str(LIVE / "frames" / "170907-10.jpg")))
    tip = max(h.tip for h in hands)
    assert (fm.who(lines, tip), FieldMenu.hp_near(lines, tip)) == ("Crono", "175/811")
