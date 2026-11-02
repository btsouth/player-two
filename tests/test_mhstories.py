"""The Monster Hunter Stories 3 routine: the choices, the words, and the presses against a fake
game that behaves as the real one was seen to (a Skills cursor that stays where it was, a greyed
row that ignores A). Pixel reads are checked on saved frames by scripts/mh_bench.py."""

import asyncio
import types

import numpy as np
import pytest

from playertwo import mhstories as m


def part(name, works, broken=False, gauge=0.5):
    return m.Part(name, [works], broken, gauge, "", None)


def gear():
    return [m.Weapon("Cluster Hammer", "blunt", "none", [
                m.Skill("Sweet Spot", 20, 50, 60, "all", False),
                m.Skill("Spinning Meteor", 50, 120, 30, "one", True),
                m.Skill("Sleep Chaser", 20, 50, 10, "one", False)]),
            m.Weapon("Frozen Speartuna", "slash", "ice", [
                m.Skill("Charge", 0, 0, 0, "none", False, False),
                m.Skill("Rage Slash", 50, 130, 20, "one", True)]),
            m.Weapon("Seahorse Harp", "pierce", "water", [])]


def view(**over):
    base = {"parts": [part("Tail", "slash", gauge=0.8), part("Legs", "blunt", gauge=0.4),
                      part("Head", "pierce", broken=True)],
            "weapons": gear(), "in_hand": 1, "menu_type": "Speed"}
    return m.View(**{**base, **over})


def test_asks_reads_the_requests_for_the_clip():
    assert m.asks("Take this fight for me") == "fight"
    assert m.asks("Where's it weak?") == "weak"
    assert m.asks("What are we fighting?") == "what"
    assert m.asks("How's my health?") == "health"
    assert m.asks("Heal me") == "heal"
    assert m.asks("What's the capital of France?") == ""


def test_the_part_nearest_to_breaking_and_the_weapon_that_fits_it():
    plan = m.choose(view())
    assert plan.part.name == "Legs"
    assert gear()[plan.weapon].damage == "blunt"


def test_a_part_being_worked_on_stays_until_it_breaks():
    assert m.choose(view(target="Tail")).part.name == "Tail"
    parts = [part("Tail", "slash", broken=True), part("Legs", "blunt", gauge=0.4)]
    assert m.choose(view(parts=parts, target="Tail")).part.name == "Legs"


def test_a_part_whose_weapon_it_resists_comes_last():
    parts = [part("Head", "pierce", gauge=0.2), part("Legs", "blunt", gauge=0.9)]
    assert m.choose(view(parts=parts, resists=["water"])).part.name == "Legs"
    head_only = [part("Head", "pierce"), part("Legs", "blunt", broken=True)]
    assert m.choose(view(parts=head_only, resists=["water"], in_hand=0)).part.name == "Legs"
    assert m.choose(view(parts=parts)).part.name == "Head"


def test_the_type_that_beats_the_one_it_has_been_using():
    for seen, want in (("Power", "Speed"), ("Speed", "Technical"), ("Technical", "Power")):
        plan = m.choose(view(arc="yellow", toward="Zerk", monster_type=seen))
        assert plan.type == want and plan.why_type == "seen"


def test_what_it_usually_uses_is_said_as_usual_and_gives_way_to_what_is_seen():
    v = view(arc="yellow", toward="Zerk", usual_type="Power")
    plan = m.choose(v)
    assert plan.type == "Speed" and plan.why_type == "usual"
    assert "usually" in m.Lines().reason(v, plan)
    assert m.choose(view(arc="yellow", usual_type="Power", monster_type="Technical")).type == "Power"


def test_an_allys_pick_is_used_only_as_a_hint_and_said_as_one():
    v = view(arc="blue", toward="Ogden", hint_type="Power", hint_from="Ogden")
    plan = m.choose(v)
    assert plan.type == "Power" and plan.why_type == "ally"
    line = m.Lines().reason(v, plan)
    assert "Ogden" in line and "Technical" not in line


def test_an_untyped_attack_changes_no_type_and_claims_none():
    v = view(arc="red", toward="Canyne", monster_type="Power", menu_type="Technical")
    plan = m.choose(v)
    assert plan.type == "Technical" and plan.why_type == "none"
    assert "counter" not in m.Lines().reason(v, plan)
    assert "counter" not in m.Lines().reason(view(arc="red", toward="Ogden"), m.Plan("attack"))


def test_no_arc_claims_nothing_about_who_gets_hit():
    v = view(arc="none")
    line = m.Lines().reason(v, m.choose(v))
    assert "party" not in line and "everybody" not in line


def test_a_double_attack_when_canyne_is_on_the_winning_type():
    plan = m.choose(view(arc="yellow", toward="Zerk", monster_type="Power", canyne_type="Speed",
                         gauge=90))
    assert plan.double and plan.do == "attack" and plan.skill is None


def test_the_skill_that_breaks_parts_when_the_gauge_pays_for_it():
    plan = m.choose(view(gauge=60))
    assert plan.do == "skill" and plan.skill.name == "Spinning Meteor"
    assert m.choose(view(gauge=15)).do == "attack"  # nothing it can pay for


def test_healing_comes_first_and_goes_to_who_is_worst_off():
    plan = m.choose(view(hp={"Zerk": 0.2, "Canyne": 0.3, "Ogden": 0.9, "Chirpy": 1.0}))
    assert (plan.do, plan.on, plan.item) == ("item", "Zerk", "Potion")
    assert m.choose(view(hp={"Zerk": 0.65, "Canyne": 0.8})).do == "attack"


def test_no_line_is_said_twice():
    lines, said = m.Lines(), []
    v = view(arc="yellow", toward="Zerk", monster_type="Power")
    for _ in range(8):
        p = m.choose(v)
        said += [lines.reason(v, p), lines.choice(v, p, False)]
    said = [x for x in said if x]
    assert len(said) == len(set(said))


def test_lines_carry_no_triangle_strings():
    lines = m.Lines()
    v = view(arc="blue", toward="Ogden", monster_type="Technical")
    p = m.choose(v)
    assert "Power Rapid" not in lines.choice(v, p, False)
    assert not any(t in lines.choice(v, p, True) for t in m.TYPES)


def test_where_weak_takes_muses_words_only_when_they_fit_the_wheel():
    f = m.Fight(types.SimpleNamespace(on_action=None))
    f.parts = [part("Tail", "slash"), part("Legs", "blunt"), part("Head", "pierce", broken=True)]
    f.weak_line = "Hit the tail with the sword and the legs with the hammer."
    assert f.where_weak() == f.weak_line
    f.weak_line = "Go for the tail."
    f.weapons = gear()
    assert f.where_weak() == "Sword on the tail, hammer on the legs."
    f.weak_line = "Hit the tail with the hammer and the legs with the sword."
    assert f.where_weak() == "Sword on the tail, hammer on the legs."
    f.weak_line = "Hit the tail with the sword and the legs with the hammer."
    f.muse_agrees = False  # its list of diamonds was off, but the sentence is right
    assert f.where_weak() == f.weak_line


# A fake game: the routine's pixel reads look at its state instead of a picture.

def title(name):
    """A part's name as a wheel title's letters: a different mask for each."""
    i = ["Tail", "Head", "Legs"].index(name)
    return np.array([[j % (i + 2) == 0 for j in range(40)] for _ in range(8)])


class Game:
    def __init__(self):
        self.screen = "menu"
        self.type = "Speed"
        self.tab = 0
        self.cursor = {0: 1, 1: 0, 2: 0}  # the hammer's list stays on Sweet Spot (take 2)
        self.greyed = {(0, 2)}  # Spinning Meteor
        self.parts = [("Tail", "slash"), ("Head", "pierce"), ("Legs", "blunt")]
        self.wheel = 0
        self.item_row = 1
        self.ring = 0
        self.confirmed = []
        self.presses = []

    def row_list(self):
        return {0: 3, 1: 2, 2: 0}[self.tab]

    def press(self, b, hold_ms=100):
        self.presses.append(b)
        s = self.screen
        if s == "menu":
            self.screen = {"a": "wheel", "x": "skills", "y": "items", "rt": "status"}.get(b, s)
        elif s == "wheel":
            if b == "b":
                self.screen = "menu"
            elif b == "rb":
                self.wheel = (self.wheel + 1) % len(self.parts)
            elif b == "a":
                row = self.cursor[self.tab] if getattr(self, "armed", False) else 0
                self.confirmed.append((self.tab, row, self.parts[self.wheel][0], self.type))
                self.armed = False
                self.screen = "playing"
        elif s == "skills":
            if b == "b":
                self.screen = "menu"
                self.cursor[self.tab] = self.cursor[self.tab]
            elif b == "rb":
                self.tab = (self.tab + 1) % 3
            elif b == "lb":
                self.tab = (self.tab - 1) % 3
            elif b == "down":
                self.cursor[self.tab] = min(self.row_list(), self.cursor[self.tab] + 1)
            elif b == "up":
                self.cursor[self.tab] = max(0, self.cursor[self.tab] - 1)
            elif b == "a" and (self.tab, self.cursor[self.tab]) not in self.greyed:
                self.armed = True
                self.screen = "wheel"
        elif s == "items":
            if b == "b":
                self.screen = "menu"
            elif b == "down":
                self.item_row += 1
            elif b == "up":
                self.item_row -= 1
            elif b == "a":
                self.screen = "ring"
        elif s == "ring":
            if b == "a":
                self.confirmed.append(("item", self.item_row, m.ALLIES[self.ring]))
                self.screen = "playing"
            elif b == "b":
                self.screen = "items"
        elif s == "status" and b in ("rt", "b"):
            self.screen = "menu"

    def stick(self, which, way, ms, strength=1.0):
        if self.screen == "ring":
            self.ring = (self.ring + 1) % 4
        elif way in ("up", "down"):
            self.press(way)
        else:
            step = 1 if way == "right" else -1
            self.type = m.TYPES[(m.TYPES.index(self.type) + step) % 3]

    def snap(self):
        return types.SimpleNamespace(
            screen=self.screen, type=self.type, tab=self.tab, row=self.cursor[self.tab],
            part=self.parts[self.wheel], item=self.item_row, ring=self.ring, greyed=set(self.greyed))


@pytest.fixture
def game(monkeypatch):
    g = Game()
    patch = {
        "menu_up": lambda i: i.screen == "menu",
        "menu_type": lambda i: i.type,
        "skills_tab": lambda i: i.tab if i.screen in ("skills", "items") else None,
        "skill_row": lambda i: i.row,
        "skill_greyed": lambda i, row: (i.tab, row) in i.greyed,
        "items_open": lambda i: i.screen == "items",
        "lit_row": lambda i, rows: i.item,
        "wheel_open": lambda i: i.screen == "wheel",
        "wheel_has_parts": lambda i: i.screen == "wheel",
        "wheel_title": lambda i: title(i.part[0]),
        "ring_is_small": lambda i: False,
        "wheel_works": lambda i: [i.part[1]],
        "part_broken": lambda i: False,
        "wheel_gauge": lambda i: 0.5,
        "ring_who": lambda i: m.ALLIES[i.ring] if i.screen == "ring" else "",
        "status_open": lambda i: i.screen == "status",
        "arc_seen": lambda i: ("yellow", "Zerk"),
        "ribbon": lambda i, who: "",
        "bar_share": lambda i, who, y=684: 0.9,
        "monster_health": lambda i: 0.9,
        "monster_ribbon": lambda i: "",
        "clash_type": lambda i: "",
        "toast_up": lambda i: False,
    }
    for name, fn in patch.items():
        monkeypatch.setattr(m, name, fn)
    return g


def agent_for(g, said):
    go = asyncio.Event()
    go.set()

    async def grab():
        return types.SimpleNamespace(image=g.snap())

    async def speak(text):
        said.append(text)

    return types.SimpleNamespace(
        go=go, cancelled=False, controls=g, grab=grab, _speak=speak, on_action=None, muse=None)


def fight_for(g, said):
    f = m.Fight(agent_for(g, said))
    f.weapons = gear()
    f.parts = [m.Part(n, [w], False, 0.5, "", title(n))
               for n, w in g.parts]
    f.status = {"name": "Tetranadon"}
    f.studied = True
    return f


def run(coro):
    return asyncio.run(coro)


def test_a_skill_cursor_left_on_another_row_doesnt_land_on_a_greyed_skill(game):
    """Take 2: the list opened on Sweet Spot, one down landed on greyed Spinning Meteor, and A
    did nothing for 30 s. Here the cursor is read."""
    said = []
    f = fight_for(game, said)
    plan = m.Plan("skill", type="Speed", weapon=0, part=f.parts[0],
                  skill=f.weapons[0].skills[2])  # Sleep Chaser, row 3

    async def go():
        game.press("a")
        state = await f.select(plan, "Speed")
        assert state == "ready"
        await f.confirm()

    run(go())
    assert game.confirmed == [(0, 3, "Tail", "Speed")]
    assert game.screen == "playing"


def test_a_greyed_skill_falls_back_to_the_plain_attack_and_is_not_named(game):
    said = []
    f = fight_for(game, said)
    plan = m.Plan("skill", type="Speed", weapon=0, part=f.parts[2],
                  skill=f.weapons[0].skills[1])  # Spinning Meteor, greyed

    async def go():
        game.press("a")
        state = await f.select(plan, "Speed")
        assert state == "ready"
        await f.confirm()
        return m.Lines().choice(view(), plan, False)

    line = run(go())
    assert plan.skill is None and plan.do == "attack"
    assert "Spinning" not in line
    assert game.confirmed == [(0, 0, "Legs", "Speed")]


def test_a_weapon_and_type_change_end_on_that_weapon_type_and_part(game):
    said = []
    f = fight_for(game, said)
    plan = m.Plan("attack", type="Technical", weapon=1, part=f.parts[0])

    async def go():
        game.press("a")
        assert await f.select(plan, "Speed") == "ready"
        await f.confirm()

    run(go())
    assert game.confirmed == [(1, 0, "Tail", "Technical")]
    assert f.weapon == 1


def test_a_whole_turn_says_why_then_what_in_order(game):
    said = []
    f = fight_for(game, said)
    f.seen_type = "Power"

    async def go():
        plan = await f.take_turn(game.snap(), "take this fight")
        await f.said()
        return plan

    plan = run(go())
    assert plan.type == "Speed"
    assert game.screen == "playing"
    assert "so speed" in said[0] and "guessing power" in said[0]
    assert len(said) == 2


def test_a_potion_goes_on_who_it_was_said_for(game, monkeypatch):
    said = []
    f = fight_for(game, said)
    monkeypatch.setattr(m, "bar_share", lambda i, who, y=684: 0.2 if who == "Zerk" else 0.9)

    async def go():
        await f.take_turn(game.snap(), "heal me")
        await f.said()

    run(go())
    assert game.confirmed == [("item", 1, "Zerk")]
    assert said[0].startswith("You're down to about a quarter")
    assert "Potion on you." in said


def test_heal_on_request_uses_a_potion_even_when_no_one_is_low(game):
    said = []
    f = fight_for(game, said)

    async def go():
        await f.take_turn(game.snap(), "use a potion", heal=True)
        await f.said()

    run(go())
    assert game.confirmed == [("item", 1, "Zerk")]


def test_a_part_the_player_named_is_the_one_that_gets_hit(game):
    said = []
    f = fight_for(game, said)

    async def go():
        await f.take_turn(game.snap(), "attack the tail")
        await f.said()

    run(go())
    assert game.confirmed[0][2] == "Tail"


def test_a_menu_left_open_during_the_turn_is_backed_out_of_fast(game, monkeypatch):
    """No more 30 seconds of a party waiting on a menu."""
    said = []
    f = fight_for(game, said)
    game.screen = "skills"
    real = game.press

    async def go():
        t = asyncio.get_running_loop().time()
        img = await f.play_out()
        return asyncio.get_running_loop().time() - t, img

    took, img = run(go())
    assert img is not None and took < 3.0
    assert "b" in game.presses
    del real


def test_a_title_a_few_pixels_off_is_still_that_part():
    tail, head = title("Tail"), title("Head")
    assert m._title_gap(np.roll(tail, 4, axis=1), tail) < m._title_gap(np.roll(tail, 4, axis=1), head)


def test_jargon_and_repeats_stay_out_of_the_lines():
    lines = m.Lines()
    assert lines.monster_low(0.3) == "It's getting weak."
    assert lines.monster_low(0.25) == ""
    assert lines.monster_low(0.5) == ""
    assert lines.monster_low(0.1) == "It's almost done."
    plan = m.Plan("attack", type="Speed")
    assert "line" not in lines.reason(m.View([], [], 0, "Speed", arc="none"), plan)
    assert lines.reason(m.View([], [], 0, "Speed", arc="red", toward="Canyne"), plan) == ""


def test_a_part_muse_never_named_is_not_said():
    part = m.Part("part 2", ["blunt"], False, 0.5, "", title("Legs"))
    plan = m.Plan("attack", type="Speed", part=part)
    line = m.Lines().choice(m.View([part], [], 0, "Speed"), plan, False)
    assert "part" not in line


def test_muse_words_are_used_only_when_each_weapon_fits_its_part():
    def part(n, w):
        return m.Part(n, [w], False, 0.5, "", title(n))

    parts = [part("Head", "pierce"), part("Legs", "blunt"), part("Tail", "slash")]
    weapons = [m.Weapon("Cluster Hammer", "blunt"), m.Weapon("Frozen Speartuna", "slash"),
               m.Weapon("Seahorse Harp", "pierce")]
    good = "hit the head with the harp, the legs with the hammer, and the tail with the sword."
    assert m.muse_line_fits(good, parts, weapons)
    assert m.muse_line_fits("use the seahorse harp on its head, cluster hammer on its legs and "
                            "ice spear on its tail", parts, weapons)
    assert not m.muse_line_fits("hit the head with the hammer, the legs with the harp, and the "
                                "tail with the sword.", parts, weapons)
    assert not m.muse_line_fits("hit the legs with the hammer.", parts, weapons)


def test_a_line_is_not_said_twice_however_long_the_fight():
    lines = m.Lines()
    assert lines.pick("Sleep Chaser.") == "Sleep Chaser."
    assert lines.pick("Sleep Chaser.") == ""


def test_health_names_who_is_a_bit_hurt():
    said = m.Lines().health({"Zerk": 0.98, "Canyne": 0.98, "Ogden": 0.7, "Chirpy": 0.97})
    assert "Ogden" in said and "full health" in said


def test_one_early_top_up_a_fight_when_nothing_is_urgent():
    hurt = {"Zerk": 0.98, "Canyne": 0.98, "Ogden": 0.72, "Chirpy": 0.97}
    assert m.choose(view(hp=hurt)).do != "item"
    v = view(hp=hurt)
    v.top_up = True
    plan = m.choose(v)
    assert (plan.do, plan.on) == ("item", "Ogden")
    v = view(hp={"Zerk": 0.98, "Canyne": 0.98, "Ogden": 0.0, "Chirpy": 0.97})  # riding
    v.top_up = True
    assert m.choose(v).do != "item"


def test_a_change_of_state_changes_what_it_usually_hits_with():
    f = m.Fight(types.SimpleNamespace(on_action=None))
    f.status = {"name": "Somnacanth", "state": "Normal"}
    assert f.usual() == "Technical"
    f.status["state"] = "Angry"
    assert f.usual() == "Speed"


def test_four_fifths_of_a_bar_is_not_two_thirds():
    assert m.amount(0.8) == "about three quarters"
    assert m.amount(0.66) == "about two thirds"


def test_a_broken_part_stays_broken():
    part = m.Part("Stomach", ["blunt"], True, 0.0, "", title("Legs"))
    tail = m.Part("Tail", ["slash"], False, 0.5, "", title("Tail"))
    v = view()
    v.parts, v.target = [part, tail], "Stomach"
    assert m.pick_part(v) is tail
