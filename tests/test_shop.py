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
    assert shop.number("1u") == 10  # the fair's Potion, live
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
    assert game.mode == "field" and "Come again!" in out  # the goodbye read, pressed on


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
    assert game.mode == "field" and game.funds == 400



def test_only_the_good_by_its_name_never_one_like_it():
    """Live goods at the fair: "Hi-Potion" or "Mid-Potion" isn't "Potion" (a loose match
    bought Potion for both), and "Potion" isn't Mid-Potion where that's sold instead."""
    fair = [shop.Good(n, p, 88 + 78 * k) for k, (n, p) in enumerate(FakeShop.GOODS)]
    assert shop.find(fair, "Hi-Potion") is None and shop.find(fair, "Mid-Potion") is None
    assert shop.find(fair, "Hi-Potions") is None
    assert shop.find(fair, "potions").name == "Potion"
    assert shop.find(fair, "athenian water").name == "Athenian Water"
    assert shop.find([shop.Good("Mid-Potion", 100, 88), shop.Good("Ether", 800, 166)],
                     "Potion") is None
    # A name the font misread is still the good meant (Melchior's list, live).
    melchior = [shop.Good("Bronze Blade", 350, 88), shop.Good("Silver Sv)ord", 4000, 168)]
    assert shop.find(melchior, "Silver Sword").price == 4000
    assert shop.find(melchior, "Iron Blade") is None
    assert shop.same_good("Brohze Blade", "Bronze Blade")
    assert not shop.same_good("Potion", "Hi-Potion") and not shop.same_good("Ether", "Mid-Ether")
    # The count view: another good stops the last a; a name that reads as nothing doesn't.
    potion = fair[0]
    assert shop.other_good("Panacea", potion, fair) == "Panacea"
    assert shop.other_good("Hi-Potion", potion, fair) == "Hi-Potion"
    assert shop.other_good("Pctlcn", potion, fair) is None
    assert shop.other_good("Potion", potion, fair) is None


def test_hi_potion_isnt_bought_where_only_potion_is_sold():
    game = FakeShop(funds=400)
    out = asyncio.run(make(game).buy("Hi-Potion", 2))
    assert out.startswith("stopped: no Hi-Potion here") and "Potion (10 G)" in out, out
    assert game.funds == 400 and game.stock == {} and game.mode == "field"


@pytest.mark.skipif(not (FRAMES / "shop-melchior-3-item-picked.png").exists(),
                    reason="no shop frames")
def test_the_count_view_names_the_good_picked_from_the_list():
    """The count view's name is read again (and misread: "Brohze Blade"); it's still the
    good the hand was put on, so the last a goes ahead."""
    picked = read("shop-melchior-2-buy-list").on
    v = read("shop-melchior-3-item-picked")
    goods = read("shop-melchior-2-buy-list").goods
    good = next(g for g in goods if g.name == picked)
    assert v.kind == "count" and shop.other_good(v.on, good, goods) is None
    silver = next(g for g in goods if g is not good)
    assert shop.other_good(v.on, silver, goods) == "Bronze Blade"


class FakeShopPlus(FakeShop):
    """The fair seller with what went wrong: the hand left in the Sell list, the choice's
    hand on "No, thanks", a count that doesn't read or reads one too many, a count view
    showing another good."""

    SELL: ClassVar = [("Potion", 5), ("Tonic", 5)]

    def __init__(self, funds=400, mode="talk", choice=False, count_shown=lambda c: c,
                 count_row=0, paid=None):
        super().__init__(funds)
        self.paid = paid  # what the last a really costs each (another good than shown)
        self.mode, self.choice, self.on_no = mode, choice, choice
        self.count_shown, self.count_row, self.sold = count_shown, count_row, 0
        if mode.startswith("sell"):
            self.main = 1

    async def press(self, b, ms):
        m = self.mode
        if m == "talk" and self.choice and b in ("up", "down"):
            self.log.append(b)
            self.on_no = not self.on_no
            return
        if m == "talk" and self.choice and b == "a" and self.on_no:
            self.log.append(b)
            self.mode = "bye"  # declined: the seller says goodbye, no shop
            return
        if m == "main" and b == "a" and self.main == 1:
            self.log.append(b)
            self.mode, self.row = "sell", 0
            return
        if m == "sell":
            self.log.append(b)
            if b in ("up", "down"):
                self.row = max(0, min(len(self.SELL) - 1, self.row + (1 if b == "down" else -1)))
            elif b == "a":
                self.mode, self.count = "sellcount", 1
            elif b == "b":
                self.mode = "main"
            return
        if m == "count" and b == "a" and self.paid is not None:
            self.log.append(b)
            self.funds -= self.paid * self.count
            self.mode = "list"
            return
        if m == "sellcount":
            self.log.append(b)
            if b == "right":
                self.count += 1
            elif b == "a":
                self.funds += self.SELL[self.row][1] * self.count
                self.sold += self.count
                self.mode = "sell"
            elif b == "b":
                self.mode = "sell"
            return
        await super().press(b, ms)

    def read(self, frame):
        def run(text, x, y):
            return Run(text, x, x + 20 * len(text), y, "white", 5.0)

        if self.mode in ("sell", "sellcount"):
            lines = [[run("Funds:", 227, 486), run(f"{self.funds} G", 627, 486)]]
            if self.mode == "sell":
                for k, (n, p) in enumerate(self.SELL):
                    lines.append([run(f"k3 {n}", 939, 88 + 78 * k),
                                  run(str(p), 1743, 88 + 78 * k)])
                return [SimpleNamespace(tip=(912, 100 + 78 * self.row))], lines
            name = self.SELL[self.row][0]
            lines.append([run(f"k {name}", 916, 243), run(str(self.count), 1653, 243)])
            lines.append([run(f"{self.count * self.SELL[self.row][1]} G", 1779, 378)])
            return [SimpleNamespace(tip=(912, 252))], lines
        hands, lines = super().read(frame)
        if self.mode == "count":
            name = self.GOODS[(self.row + self.count_row) % len(self.GOODS)][0]
            shown = self.count_shown(self.count)
            lines[-2] = [Run(f"k {name}", 916, 916 + 20 * len(name), 243, "white", 5.0)] + (
                [Run(str(shown), 1653, 1673, 243, "white", 5.0)] if shown is not None else [])
        return hands, lines

    def dialogue(self, frame):
        if self.mode == "talk" and self.choice:
            return SimpleNamespace(kind="choice", parts={
                "question": "Welcome! Have a look?",
                "focus": "No, thanks selected, or Take a look" if self.on_no
                else "Take a look selected, or No, thanks"})
        return super().dialogue(frame)


def test_the_hand_left_in_the_sell_list_buys_from_buy_not_sells():
    """An open list may be the Sell list: an a there sold, and was reported as "Bought 3
    Potion for -15 G"."""
    game = FakeShopPlus(funds=400, mode="sell")
    out = asyncio.run(make(game).buy("Potion", 3))
    assert game.sold == 0 and game.funds == 370 and game.stock == {"Potion": 3}, out
    assert "Bought 3 Potion for 30 G" in out


def test_a_sale_is_never_reported_as_a_purchase():
    """Whatever got it there, gold that went up is a sale, said as one."""
    game = FakeShopPlus(funds=400, mode="sell")
    sh = make(game)

    async def straight_in(v):
        return v  # the old way: any open list is taken for the Buy list

    sh.to_list = straight_in
    out = asyncio.run(sh.buy("Potion", 3))
    assert out.startswith("stopped: the funds went up from 400 G to 415 G"), out


def test_the_sellers_choice_is_answered_with_take_a_look_not_no_thanks():
    """"No, thanks selected, or Take a look": the hand is on No, thanks, though "Take a
    look" is in the line."""
    game = FakeShopPlus(funds=400, choice=True)
    out = asyncio.run(make(game).buy("Potion", 1))
    assert game.stock == {"Potion": 1} and game.funds == 390, out
    assert game.log[:2] == ["up", "a"]


@pytest.mark.skipif(not (FRAMES / "box-melchior-options-no-question.png").exists(),
                    reason="no seller choice frame")
def test_the_live_sellers_choice_reads_as_the_shop_expects():
    from playertwo import ctscreen

    lk = ctscreen.look(Image.open(FRAMES / "box-melchior-options-no-question.png"))
    assert lk.kind == "choice"
    assert lk.parts["focus"].split(" selected")[0] in shop.OPEN


def test_a_count_that_doesnt_read_buys_nothing():
    game = FakeShopPlus(funds=400, count_shown=lambda c: "?")  # read, not a number
    out = asyncio.run(make(game).buy("Potion", 3))
    assert out.startswith("stopped: the count didn't read"), out
    assert game.funds == 400 and game.stock == {} and game.mode == "field"


def test_a_count_read_one_too_many_is_said_as_the_gold_says():
    """It read 3 where the game had 2: the report goes by the gold spent."""
    game = FakeShopPlus(funds=400, count_shown=lambda c: c + 1)
    out = asyncio.run(make(game).buy("Potion", 3))
    assert game.stock == {"Potion": 2}
    assert "Bought 2 Potion for 20 G" in out and "the gold says 2" in out, out


def test_gold_that_doesnt_fit_the_good_isnt_turned_into_a_count():
    """2 bought at 200 G each where Potion is 10 G: not "Bought 40 Potion"."""
    game = FakeShopPlus(funds=1000, paid=200)
    out = asyncio.run(make(game).buy("Potion", 2))
    assert "Bought 2 Potion for 400 G" in out and "doesn't fit" in out, out


def test_the_count_for_another_good_isnt_bought():
    game = FakeShopPlus(funds=400, count_row=1)
    out = asyncio.run(make(game).buy("Potion", 1))
    assert out.startswith("stopped: the count shows Panacea, not Potion"), out
    assert game.funds == 400 and game.stock == {}

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


def _fake_clock(monkeypatch):
    """Time that passes only in the shop's own sleeps: a 45 s wait runs at once."""
    clock = [1000.0]

    async def sleep(s):
        clock[0] += s

    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(shop, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    return clock


def test_no_night_and_no_gold_spent_isnt_a_night_at_the_inn(monkeypatch):
    """Not enough gold: Yes, then the innkeeper says so and no night comes. Before, it
    waited 45 s and said "Stayed the night"."""
    clock = _fake_clock(monkeypatch)
    t0 = clock[0]
    inn = FakeInn()

    async def press(b, ms):
        inn.log.append(b)
        if inn.mode == "ask" and b == "a":
            inn.mode = "broke"
        elif inn.mode == "broke" and b == "a":
            inn.mode = "room"

    def dialogue(frame):
        if inn.mode == "broke":
            return SimpleNamespace(kind="text", parts={"text": "You don't have enough money!"})
        return inn.dialogue(frame)

    sh = Shop(press, inn.grab, read=lambda f: ([], []), dialogue=dialogue, gap_s=0.0)

    async def vitals():
        return [(20, 70, 8, 8)], 5

    out = asyncio.run(Inn(sh, SimpleNamespace(vitals=vitals, party=["Crono"])).rest(5))
    assert out.startswith("stopped: didn't stay: no night came after Yes and the gold is "
                          "still 5 G"), out
    assert "You don't have enough money!" in out
    assert clock[0] - t0 < 20


def test_a_night_the_looks_missed_counts_when_the_gold_went_down(monkeypatch):
    _fake_clock(monkeypatch)
    inn = FakeInn()

    async def press(b, ms):
        inn.log.append(b)
        if inn.mode == "ask" and b == "a":
            inn.mode = "room"  # the fade came and went between looks

    sh = Shop(press, inn.grab, read=lambda f: ([], []), dialogue=inn.dialogue, gap_s=0.0)

    async def vitals():
        return [(70, 70, 8, 8)], 390

    out = asyncio.run(Inn(sh, SimpleNamespace(vitals=vitals, party=["Crono"])).rest(400))
    assert out.startswith("Stayed the night (10 G: 400 to 390 G)"), out

    async def no_gold():
        return [(70, 70, 8, 8)], None

    inn.mode = "ask"
    out = asyncio.run(Inn(sh, SimpleNamespace(vitals=no_gold, party=["Crono"])).rest(400))
    assert out.startswith("stopped: no night came after Yes and the gold didn't read"), out


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


@pytest.mark.skipif(not (FRAMES / "menu-main-1000ad-crono-marle.png").exists(),
                    reason="no menu frame")
def test_the_menu_page_hp_and_mp_with_misread_labels():
    """"MF l 70/ 70" is Crono's HP, "MP : 8/" "8" his MP; the gold under the time."""
    from playertwo.menu import FieldMenu

    img = Image.open(FRAMES / "menu-main-1000ad-crono-marle.png")

    async def grab():
        return img

    async def press(b, ms):
        pass

    async def nothing():
        return None

    fm = FieldMenu(press, grab, wait_s=0.1)
    fm.open = fm.close = nothing
    assert asyncio.run(fm.vitals()) == ([(70, 70, 8, 8), (65, 65, 12, 12)], 366)


@pytest.mark.skipif(not (FRAMES / "shop-fair-item-seller-list.png").exists(),
                    reason="no fair shop frames")
def test_the_fair_item_sellers_goods_and_the_reader_on_them():
    from playertwo import ctscreen

    v = read("shop-fair-item-seller-list")
    assert [(g.name, g.price) for g in v.goods] == [
        ("Potion", 10), ("Panacea", 10), ("Athenian Water", 200), ("Shelter", 150)]
    assert (v.kind, v.on, v.funds, v.stock) == ("list", "Potion", 276, 18)
    lk = ctscreen.look(Image.open(FRAMES / "shop-fair-item-seller-list.png"))
    assert lk.parts == {"focus": "Potion, 10 G", "desc": "Restores 50 HP."}
    lk = ctscreen.look(Image.open(FRAMES / "shop-fair-item-seller-main.png"))
    assert lk.parts == {"focus": "Buy", "desc": "Funds 276 G"}


def _shop_agent(monkeypatch, walked: str):
    """The agent's shop tool with the walk to the seller answering `walked`; what the
    shop and inn were asked to do is noted."""
    from playertwo import ctscreen
    from playertwo.agent import Agent
    from playertwo.menu import FieldMenu
    from playertwo.screen import Frame

    async def grab():
        return Frame(Image.new("RGB", (64, 36)), 0.0)

    async def speak(text):
        pass

    agent = Agent(SimpleNamespace(), SimpleNamespace(press=lambda b, ms: None), grab, speak,
                  game="Chrono Trigger")
    agent._ct_steam = lambda frame: True
    done = []

    async def act(name, a):
        done.append((name, a))
        return walked

    async def look(self):
        self.frame = Image.new("RGB", (64, 36))
        return shop.View()

    async def buy(self, item, count=1):
        done.append(("buy", item))
        return "Bought"

    async def rest(self, gold0=None, vit0=None):
        done.append(("rest",))
        return "Stayed the night"

    async def vitals(self):
        return [(70, 70, 8, 8)], 400

    agent._act = act
    monkeypatch.setattr(shop.Shop, "look", look)
    monkeypatch.setattr(shop.Shop, "buy", buy)
    monkeypatch.setattr(shop.Inn, "rest", rest)
    monkeypatch.setattr(FieldMenu, "vitals", vitals)
    monkeypatch.setattr(ctscreen, "look", lambda image, party=(): ctscreen.Look())
    return agent, done


def test_the_shop_tool_says_what_the_walk_to_the_seller_did(monkeypatch):
    """Live review: any failed walk was "no innkeeper here, go to Truce Inn", even inside
    the inn; any arrival was "talked", and the purchase then waited ~13 s for a shop."""
    nothing = ("arrived: at the innkeeper (12 s), facing them, but pressing a brought "
               "nothing up: look")
    agent, done = _shop_agent(monkeypatch, nothing)
    out = asyncio.run(agent._shop({"action": "rest"}))
    assert out.startswith("not done: got to the innkeeper, but talking to them brought "
                          "nothing up") and "brought nothing up: look" in out, out
    assert ("rest",) not in done
    agent, done = _shop_agent(monkeypatch, nothing.replace("innkeeper", "item seller"))
    out = asyncio.run(agent._shop({"action": "buy", "item": "Potion",
                                   "seller": "the item seller"}))
    assert out.startswith("not done: got to the item seller") and ("buy", "Potion") not in done

    boxed = ("interrupted: a text box (or a menu) is up: read it to the player, press a to "
             "go on, then go_to_exit again")
    agent, done = _shop_agent(monkeypatch, boxed)
    out = asyncio.run(agent._shop({"action": "rest"}))
    assert "no innkeeper here" not in out and "interrupted: a text box" in out, out

    elsewhere = "no single place from here fits 'the innkeeper' (no innkeeper listed nearby)"
    agent, done = _shop_agent(monkeypatch, elsewhere)
    out = asyncio.run(agent._shop({"action": "rest"}))
    assert out.startswith("not done: no innkeeper here. Go to an inn"), out

    talked = ("arrived: at the innkeeper, facing them, and pressed a to talk (5 s): read "
              "their line to the player word for word")
    agent, done = _shop_agent(monkeypatch, talked)
    out = asyncio.run(agent._shop({"action": "rest"}))
    assert out.startswith("Walked to the innkeeper and talked. Stayed the night"), out


def test_the_price_and_the_purchase_are_told_before_the_goodbye():
    """So they're said as the gold shows them, not after the seller's goodbye (live 3:
    the goodbye typed out for 2.5 s before anything was said)."""
    game = FakeShop()
    told = []

    async def priced(n, name, total):
        told.append(("price", n, name, total, game.mode))

    async def bought(n, name, spent, funds):
        told.append(("bought", n, name, spent, funds, game.mode))

    out = asyncio.run(make(game).buy("Potion", 3, priced=priced, bought=bought))
    assert out.startswith("Bought 3 Potion for 30 G (funds 400 G to 370 G)"), out
    assert told == [("price", 3, "Potion", 30, "count"),
                    ("bought", 3, "Potion", 30, 370, "list")]  # still in the shop
