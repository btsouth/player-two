"""Requests done without a model turn (fast.py): the words, the lines said, and the
agent's side of a purchase."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from PIL import Image

from playertwo import fast
from playertwo.battle import Intent, Outcome
from playertwo.ctsteam import Member, State


def test_buy_words_say_how_many_of_what():
    assert fast.buy_request("buy three potions") == (3, "Potion")
    assert fast.buy_request("Buy three potions.") == (3, "Potion")
    assert fast.buy_request("Player Two, buy three more potions please") == (3, "Potion")
    assert fast.buy_request("buy 3 potions") == (3, "Potion")
    assert fast.buy_request("buy a hi-potion") == (1, "Hi-Potion")
    assert fast.buy_request("could you get two ethers") == (2, "Ether")
    # How many, or what, not said: the model asks.
    assert fast.buy_request("buy some potions") is None
    assert fast.buy_request("buy potions") is None
    assert fast.buy_request("buy three swords") is None
    assert fast.buy_request("buy three potions and an ether") is None


def test_rest_words():
    assert fast.rest_request("rest at the inn") and fast.rest_request("stay the night")
    assert fast.rest_request("let's rest at the inn please")
    assert not fast.rest_request("rest") and not fast.rest_request("take me to the inn")


def test_battle_words_settle_one_command_or_none():
    br = fast.battle_request
    assert br("Crono, use Cyclone.") == Intent("Crono", "tech", "Cyclone", "")
    assert br("use cyclone") == Intent("Crono", "tech", "Cyclone", "")  # only Crono has it
    assert br("Crono, Cyclone") == Intent("Crono", "tech", "Cyclone", "")
    assert br("Attack the one on the left.") == Intent("", "attack", "", "left")
    assert br("attack the left one") == Intent("", "attack", "", "left")
    assert br("Frog, attack the one on the right") == Intent("Frog", "attack", "", "right")
    assert br("attack") == Intent("", "attack", "", "")
    assert br("Marle, use Ice on the one on the right") == Intent("Marle", "tech", "Ice", "right")
    assert br("Marle, use cure on Frog") == Intent("Marle", "tech", "Cure", "Frog")
    assert br("Marle, cure on me") == Intent("Marle", "tech", "Cure", "Marle")
    assert br("use a potion on Frog") == Intent("", "item", "Potion", "Frog")
    assert br("Frog, heal") == Intent("Frog", "tech", "Heal", "")
    # The model's: two commands, an enemy by name, the middle, a tech two people have.
    for words in ("Crono use cyclone and attack", "attack the gargoyle",
                  "attack the middle one", "use Lightning II", "attack whichever is on the left",
                  "what are we fighting", "win this fight"):
        assert br(words) is None, words


def test_where_words():
    assert fast.where_request("where am I?") and fast.where_request("Where are we")
    # Both ways "where am I" came in before (speed's and quality's words): one path now.
    for words in ("so, where am I?", "Player Two, where are we now?", "where am I exactly",
                  "tell me where we are please", "OK where are we right now"):
        assert fast.where_request(words), words
    assert not fast.where_request("where's the inn") and not fast.where_request(
        "where am i on the map")


def test_the_one_item_seller_in_a_place():
    people = [SimpleNamespace(name="Melchior the sword seller in the middle", kind="person"),
              SimpleNamespace(name="the armor seller at the blue tent on the left",
                              kind="person"),
              SimpleNamespace(name="the item seller at the purple tent", kind="person"),
              SimpleNamespace(name="the fountain", kind="thing")]
    assert fast.seller_for(people) == "the item seller at the purple tent"
    assert fast.seller_for(people, rest=True) == ""
    two = [*people, SimpleNamespace(name="the item shop clerk", kind="person")]
    assert fast.seller_for(two) == ""  # two fit: the model picks
    inn = [SimpleNamespace(name="the innkeeper behind the counter", kind="person")]
    assert fast.seller_for(inn, rest=True) == "the innkeeper behind the counter"


def test_purchase_lines():
    assert fast.price_line(3, "Potion", 30) == "Three Potions, 30 gold."
    assert fast.bought_line(3, "Potion", 30, 370, 3) == (
        "Got three Potions for 30 gold. You have 370 left.")
    assert fast.bought_line(3, "Potion", 30, 370, 3, priced=True) == (
        "Bought. You have 370 gold left.")
    assert fast.bought_line(2, "Potion", 20, 0, 3, priced=True) == (
        "Got two Potions for 20 gold. You have 0 left. That's all the gold allowed, not "
        "three.")
    assert fast.not_sold_line(
        "stopped: no Hi-Potion here; this shop sells: Potion (10 G), Panacea (10 G), "
        "Shelter (150 G). Out of the shop.", "Hi-Potion") == (
        "This shop doesn't sell Hi-Potions. It has Potion for 10, Panacea for 10 and "
        "Shelter for 150 gold.")


def _st(*hps, kind="command"):
    names = ("Crono", "Frog", "Marle")
    tops = (780, 494, 430)
    return State(kind=kind, party=[Member(n, h, t, 60, 64, "white")
                                   for n, h, t in zip(names, hps, tops, strict=False)])


def test_a_command_said_from_what_the_screen_showed():
    before = _st(780, 494, 430)
    after = _st(780, 1, 1)
    out = Outcome(True, "Crono: tech Cyclone", after)
    assert fast.battle_line(Intent("Crono", "tech", "Cyclone"), out, before, after, True) == (
        "Crono used Cyclone. Frog and Marle dropped to 1 HP.")
    # Hits before the request aren't news; the danger still is (low_line's words).
    out = Outcome(True, "Crono: attack on left", after)
    assert fast.battle_line(Intent("Crono", "attack", "", "left"), out, after, after, True) == (
        "Crono attacked the one on the left. Frog and Marle are at 1 HP.")
    assert fast.battle_line(Intent("Crono", "tech", "Cyclone"), out, after, after, False) == (
        "Crono picked Cyclone; the fight is paused, so it goes off when we go on. Frog and "
        "Marle are at 1 HP.")
    out = Outcome(True, "Crono: attack", _st(0, 494, 430))
    assert fast.battle_line(Intent("", "attack"), out, before, None, True) == (
        "Crono attacked. Crono is down.")
    # An item in act_line's words; a tech at an enemy says which side.
    out = Outcome(True, "Crono: item Hi-Potion on Marle", _st(780, 494, 430))
    assert fast.battle_line(Intent("Crono", "item", "Hi-Potion", "Marle"), out,
                            _st(780, 494, 100), None, True) == (
        "Crono gave Marle a Hi-Potion. Marle is up to 430.")
    out = Outcome(True, "Marle: tech Ice on right", before)
    assert fast.battle_line(Intent("Marle", "tech", "Ice", "right"), out, before, before,
                            True) == "Marle used Ice on the one on the right."


def test_a_command_line_takes_hp_from_the_newest_full_reading():
    """The HP said is the newest reading with everyone's HP read: the screen once the
    command played out, else the paused one (an attack drawn over the box hides
    numbers; live 3 said "full health" over numbers it couldn't see)."""
    before = _st(780, 494, 430)
    hidden = _st(780, 494, 430)
    hidden.party[1].hp = None
    paused = _st(780, 1, 430)
    out = Outcome(True, "Crono: attack", paused)
    assert fast.battle_line(Intent("Crono", "attack"), out, before, hidden, True) == (
        "Crono attacked. Frog dropped to 1 HP.")


def test_stops_said_plainly_or_left_to_the_model():
    cyc = Intent("Crono", "tech", "Cyclone")
    stop = lambda why: Outcome(False, why, None, stopped=why)
    assert fast.stopped_line(cyc, stop("Crono is knocked out")) == (
        "Crono's knocked out, so they can't use Cyclone.")
    assert fast.stopped_line(cyc, stop("Crono isn't ready for a command yet (ready: Marle)")
                             ) == "Crono isn't ready yet. Ask again in a moment."
    assert fast.stopped_line(cyc, stop("the target glove went away")) == ""
    aura = Intent("Marle", "tech", "Aura", "Crono")
    assert fast.stopped_line(aura, stop("Crono is at full HP (780/780): nothing to heal, so "
                                        "nothing was pressed")) == (
        "Crono is at full health already, 780 of 780.")


def test_a_won_fight_in_a_line():
    """As the fight's lines say it (live 3 read out "Won after 6 commands."): won, the
    spoils that matter, and who's still low, not everyone's HP."""
    st = _st(529, 494, 430, kind="none")
    assert fast.won_line("done: won the fight. Earned 648 EXP. Earned 18 TP. (9 commands). "
                         "Now: ...", st) == "Won the fight: 648 experience and 18 tech points."
    assert fast.won_line("done: won the fight. Earned 10 ExP. Crono's level increased! "
                         "Frog's at 1 HP. (2 commands)", _st(780, 1, 430)) == (
        "Won the fight: 10 experience. Crono went up a level. Frog is at 1 HP.")
    assert fast.won_line("done: won the fight. (2 commands)", _st(780, 494, 430)) == (
        "Won the fight.")
    assert fast.won_line("stopped: everyone is knocked out", st) == ""


def test_where_on_a_world_map_is_quests_where_said():
    """One "where am I" line (fast.py's own was dropped for quests.where_said, which also
    knows doors and rooms): the scene fast.py's test had, in its words."""
    from playertwo.quests import where_said

    m = SimpleNamespace(name="1000 A.D.", places={
        "Leene Square": (424, 214), "Truce Residence (east)": (424, 256),
        "Crono's House": (446, 270), "Guardia Castle": (100, 100)})
    line, at = where_said(m, 424, 233, 8, 0)
    assert line == ("On the world map, 1000 A.D., just south of Leene Square. The east Truce "
                    "Residence is south and Crono's House is southeast.")
    assert at == "Leene Square"
    assert not hasattr(fast, "where_line")


def test_a_purchase_is_said_as_the_gold_shows_it(monkeypatch):
    """The price from the count screen, then the purchase as soon as the gold shows it,
    with no model turn (live 3: the model's two turns and a settle look were 4 to 5 s of
    each "buy three potions")."""
    from playertwo.agent import Agent, Run
    from playertwo.screen import Frame

    said = []

    async def grab():
        return Frame(Image.new("RGB", (64, 36)), 0.0)

    async def speak(text):
        said.append(text)

    agent = Agent(SimpleNamespace(), SimpleNamespace(press=lambda b, ms: None), grab, speak,
                  game="Chrono Trigger")
    agent._ct_steam = lambda frame: True
    seller = SimpleNamespace(name="the item seller at the purple tent", kind="person")
    rooms = SimpleNamespace(rooms={5: SimpleNamespace(people=[seller])})

    async def inside(image, sure=False):
        return rooms, SimpleNamespace(loc=5)

    calls = []

    async def act(name, a):
        calls.append((name, a))
        price, bought = agent._shop_hooks
        await price(3, "Potion", 30)
        await bought(3, "Potion", 30, 370)
        return ("Walked to the item seller at the purple tent and talked. Bought 3 Potion "
                "for 30 G (funds 400 G to 370 G). Out of the shop.")

    agent._inside = inside
    agent._act = act
    run = Run("buy three potions")
    done, tried = asyncio.run(fast._shop(agent, (3, "Potion"), run, 0.0))
    assert done and not tried
    assert calls == [("shop", {"action": "buy", "seller": "the item seller at the purple tent",
                               "item": "Potion", "count": 3})]
    assert said == ["Three Potions, 30 gold.", "Bought. You have 370 gold left."]
    assert run.outcome == said[-1]

    # The seller's goodbye left up over the place (Shop.leave tells it by its window):
    # not done here, or no one would close it. The model gets it, and what was said.
    async def act_box_left(name, a):
        price, bought = agent._shop_hooks
        await price(3, "Potion", 30)
        await bought(3, "Potion", 30, 370)
        return ("Bought 3 Potion for 30 G (funds 400 G to 370 G). Out of the shop; the "
                "seller said: \"Thanks!\". A text box is still up over the place: read it "
                "and press a to close it.")

    agent._act = act_box_left
    said.clear()
    run = Run("buy three potions")
    done, tried = asyncio.run(fast._shop(agent, (3, "Potion"), run, 0.0))
    assert not done and "text box is still up" in tried and "the player heard" in tried
    assert said == ["Three Potions, 30 gold.", "Bought. You have 370 gold left."]


def test_someones_line_said_with_the_name_the_player_used():
    assert fast.talk_line("Melchior", "At your service!") == "Melchior: At your service!"
    assert fast.talk_line("the guy selling swords", "At your service!") == "At your service!"


def test_a_night_at_the_inn_in_a_line():
    assert fast.rested_line(
        "Walked to the innkeeper and talked. Stayed the night (10 G: 400 to 390 G). Now: "
        "Crono 70/70 HP, 8/8 MP; Marle 65/65 HP, 12/12 MP (everyone full).") == (
        "Rested for 10 gold, 390 left. Everyone's at full health.")
    assert fast.rested_line("Stayed the night. Now: HP not read.") == ""
