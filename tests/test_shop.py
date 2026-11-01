"""The Chrono Trigger (Steam) shop and inn executor: the shop screen read from live frames,
and a whole purchase against a small model of the shop."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from PIL import Image

from playertwo import ctsteam, shop
from playertwo.glyphread import Run
from playertwo.shop import Inn, Shop

FRAMES = Path(os.environ.get("PLAYER_TWO_CT_FRAMES",
                             "~/.cache/player-two/live/steam-2026-11-01")).expanduser() / "frames"


def read(name: str) -> shop.View:
    hands, lines = ctsteam.menu_read(Image.open(FRAMES / f"{name}.png"))
    return shop.parse([h.tip for h in hands], lines)


@pytest.mark.skipif(not (FRAMES / "shop-melchior-1.png").exists(), reason="no shop frames")
def test_the_shop_screen_read_from_live_frames():
    v = read("shop-melchior-1")
    assert (v.kind, v.on, v.funds) == ("main", "Buy", 496)
    assert [(g.price) for g in v.goods] == [350, 4000]
    v = read("shop-melchior-2-buy-list")
    assert (v.kind, v.on, v.funds, v.stock) == ("list", "Bronze Blade", 496, 0)
    v = read("shop-melchior-3-item-picked")
    assert (v.kind, v.count, v.total) == ("count", 1, 350)


def test_numbers_as_the_font_reads_them():
    assert shop.number("4.000") == 4000 and shop.number("4,000") == 4000
    assert shop.number("496 G") == 496 and shop.number("35O") == 350
    assert shop.item_name("k3 Bronze Blade") == "Bronze Blade"


def test_the_good_meant_is_the_same_name_first():
    goods = [shop.Good("Mid-Potion", 100, 88), shop.Good("Potion", 10, 166)]
    assert shop.find(goods, "potions").name == "Potion"
    assert shop.find(goods, "Mid Potion").name == "Mid-Potion"
    assert shop.find(goods, "Elixir") is None


class FakeShop:
    """The fair's item seller: a line, then the shop screen (Buy/Sell/Equip, the goods,
    the count), then a goodbye line. Lines and hands as the menu reader gives them."""

    GOODS: ClassVar = [("Potion", 10), ("Panacea", 10), ("Athenian Water", 200), ("Shelter", 150)]

    def __init__(self, funds=400, confirm=False):
        self.mode, self.funds, self.row, self.count = "talk", funds, 0, 1
        self.main, self.stock, self.confirm, self.asked = 0, {}, confirm, False
        self.log: list[str] = []

    async def press(self, b: str, ms: int) -> None:
        self.log.append(b)
        m = self.mode
        if m == "talk" and b == "a":
            self.mode = "main"
        elif m == "main":
            if b in ("up", "down"):
                self.main = (self.main + (1 if b == "down" else -1)) % 3
            elif b == "a" and self.main == 0:
                self.mode = "list"
            elif b == "b":
                self.mode = "bye"
        elif m == "list":
            if b in ("up", "down"):
                self.row = max(0, min(len(self.GOODS) - 1, self.row + (1 if b == "down" else -1)))
            elif b == "a":
                self.mode, self.count = "count", 1
            elif b == "b":
                self.mode = "main"
        elif m == "count":
            price = self.GOODS[self.row][1]
            if b == "right" and (self.count + 1) * price <= self.funds:
                self.count += 1
            elif b == "left" and self.count > 1:
                self.count -= 1
            elif b == "a" and self.confirm and not self.asked:
                self.asked = True
            elif b == "a":
                self.funds -= price * self.count
                name = self.GOODS[self.row][0]
                self.stock[name] = self.stock.get(name, 0) + self.count
                self.mode, self.asked = "list", False
            elif b == "b":
                self.mode = "list"
        elif m == "bye" and b == "a":
            self.mode = "field"

    async def grab(self):
        return self.mode

    def read(self, _frame):
        def run(text, x, y):
            return Run(text, x, x + 20 * len(text), y, "white", 5.0)

        if self.mode not in ("main", "list", "count"):
            return [], []
        lines = [[run("Funds:", 227, 486), run(f"{self.funds} G", 627, 486)]]
        name = self.GOODS[self.row][0]
        lines.append([run("Stock:", 254, 612), run(str(self.stock.get(name, 0)), 742, 612)])
        hand = (328, shop.MAIN_Y[self.main])
        if self.mode in ("main", "list"):
            for k, (n, p) in enumerate(self.GOODS):
                lines.append([run(f"k3 {n}", 939, 88 + 78 * k), run(str(p), 1743, 88 + 78 * k)])
            if self.mode == "list":
                hand = (912, 100 + 78 * self.row)
        else:
            lines.append([run(f"k {name}", 916, 243), run(str(self.count), 1653, 243)])
            lines.append([run(f"{self.count * self.GOODS[self.row][1]} G", 1779, 378)])
            hand = (912, 252)
        return [SimpleNamespace(tip=hand)], lines

    def dialogue(self, _frame):
        if self.mode == "talk":
            return SimpleNamespace(kind="text", parts={"text": "Welcome! Have a look."})
        if self.mode == "bye":
            return SimpleNamespace(kind="text", parts={"text": "Come again!"})
        return SimpleNamespace(kind="none", parts={})


def make(game: FakeShop) -> Shop:
    return Shop(game.press, game.grab, read=game.read, dialogue=game.dialogue, gap_s=0.0,
                wait_s=0.2)


def test_buy_three_from_the_sellers_line_and_back_out():
    game = FakeShop(funds=400)
    out = asyncio.run(make(game).buy("potions", 3))
    assert game.funds == 370 and game.stock == {"Potion": 3}
    assert "Bought 3 Potion for 30 G (funds 400 G to 370 G)" in out
    assert game.mode == "bye" and "Come again!" in out


def test_an_item_further_down_and_a_confirm_step():
    game = FakeShop(funds=400, confirm=True)
    out = asyncio.run(make(game).buy("Shelter", 2))
    assert game.stock == {"Shelter": 2} and game.funds == 100, out


def test_only_what_the_funds_allow_and_it_says_so():
    game = FakeShop(funds=450)
    out = asyncio.run(make(game).buy("Athenian Water", 3))
    assert game.stock == {"Athenian Water": 2} and "only 2 of the 3" in out


def test_something_not_sold_here_stops_backed_out():
    game = FakeShop()
    out = asyncio.run(make(game).buy("Elixir", 1))
    assert out.startswith("stopped: no Elixir here") and "Potion (10 G)" in out
    assert game.mode == "bye" and game.funds == 400


class FakeInn:
    def __init__(self):
        self.mode, self.log = "ask", []

    async def press(self, b, ms):
        self.log.append(b)
        if self.mode == "ask" and b == "a":
            self.mode, self.dark = "night", 3

    async def grab(self):
        if self.mode == "night":
            self.dark -= 1
            if self.dark < 0:
                self.mode = "morning"
            return Image.new("L", (64, 36), 0)
        return Image.new("L", (64, 36), 120)

    def dialogue(self, _frame):
        if self.mode == "ask":
            return SimpleNamespace(kind="choice", parts={
                "question": "Lodging is 10 G. Would you like to stay for the night?",
                "focus": "Yes selected"})
        return SimpleNamespace(kind="none", parts={})


def test_rest_says_yes_waits_the_night_and_reads_the_party(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", lambda s: _now())  # the night passes at once
    inn = FakeInn()
    sh = Shop(inn.press, inn.grab, read=lambda f: ([], []), dialogue=inn.dialogue, gap_s=0.0)

    async def vitals():
        return [(70, 70, 8, 8), (65, 65, 12, 12)], 390

    menu = SimpleNamespace(vitals=vitals, party=["Crono", "Marle"])
    out = asyncio.run(Inn(sh, menu).rest(400))
    assert inn.log == ["a"]
    assert out.startswith("Stayed the night (10 G: 400 to 390 G)") and "everyone full" in out
    assert "Crono 70/70 HP, 8/8 MP" in out


async def _now():
    return None


def test_the_inns_question_with_a_hand_that_doesnt_read_is_answered_yes(monkeypatch):
    """Live: "Lodging is 10" was read while typing, then the Yes/No read as a menu with
    "Tut-" under the hand; the inn waited 45 s for a question it had already seen."""
    monkeypatch.setattr(asyncio, "sleep", lambda s: _now())
    inn = FakeInn()
    looks = iter([SimpleNamespace(kind="text", parts={"text": "Lodging is 10"})])

    def dialogue(frame):
        if inn.mode == "ask":
            return next(looks, SimpleNamespace(kind="menu", parts={"focus": "Tut-"}))
        return SimpleNamespace(kind="none", parts={})

    def read(frame):
        return ([SimpleNamespace(tip=(972, 956))], []) if inn.mode == "ask" else ([], [])

    pressed = []

    async def press(b, ms):
        pressed.append(b)
        if b == "a" and len(pressed) >= 2:
            inn.mode, inn.dark = "night", 2

    sh = Shop(press, inn.grab, read=read, dialogue=dialogue, gap_s=0.0)

    async def vitals():
        return [(70, 70, 8, 8)], 390

    out = asyncio.run(Inn(sh, SimpleNamespace(vitals=vitals, party=["Crono"])).rest(400))
    assert pressed == ["a", "a"] and out.startswith("Stayed the night"), out


def test_the_menu_after_the_night_gets_a_few_tries(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", lambda s: _now())
    inn = FakeInn()
    sh = Shop(inn.press, inn.grab, read=lambda f: ([], []), dialogue=inn.dialogue, gap_s=0.0)
    tries = []

    async def vitals():
        tries.append(1)
        if len(tries) < 3:
            raise RuntimeError("the menu didn't open (y)")
        return [(70, 70, 8, 8)], 390

    out = asyncio.run(Inn(sh, SimpleNamespace(vitals=vitals, party=["Crono"])).rest(400))
    assert len(tries) == 3 and "everyone full" in out
