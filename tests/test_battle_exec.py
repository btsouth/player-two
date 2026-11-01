"""The Chrono Trigger (Steam) battle executor against a fake game: presses go to a small
model of the battle menus, and each look gets the State the reader would give."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

from playertwo import battle
from playertwo.battle import Executor, Intent, Tuning, merge_rows
from playertwo.ctsteam import Member, State

FAST = Tuning(after_tap_s=0.0, poll_s=0.0, pause_wait_s=0.2, ready_wait_s=0.5,
              actor_wait_s=0.3, target_wait_s=0.2, playout_min_s=0.0, playout_quiet_s=0.0,
              playout_max_s=0.2, spoils_wait_s=0.05, win_max_s=5.0, run_chunk_ms=10,
              heal_playout_s=1.0, gone_s=0.3)
ITEMS = ["Potion", "Mid-Potion", "Hi-Potion", "Mid-Ether", "Panacea", "Athenian Water",
         "Lapis", "Hi-Ether", "Ether", "Elixir", "Barrier Sphere", "Shield Sphere"]
HEALS = {"Potion": 50, "Mid-Potion": 200, "Hi-Potion": 500, "Cure": 300, "Aura": 100}


@dataclass
class Fighter:
    name: str
    hp: int
    hp_max: int
    mp: int = 60
    at: tuple[int, int] = (0, 0)  # sprite, where the bracket and the ally glove go
    techs: list[str] = field(default_factory=list)


class FakeCT:
    """Wait mode, cursor memory, two-column lists with three rows in view, Start pause."""

    def __init__(self, party, enemies, ready=(0,), items=ITEMS):
        self.party = party
        self.enemies = enemies  # [(x, y, name, hp)]
        self.ready = list(ready)
        self.hand_col = self.ready[0] if self.ready else None
        self.cmd = 0
        self.mode = "command" if self.ready else "none"
        self.paused = False
        self.items = list(items)
        self.list: list[str] = []
        self.memory: dict[str, int] = {}
        self.idx = 0
        self.top_row = 0
        self.targets: list[tuple[int, int, str]] = []
        self.glove = 0
        self.top = ""
        self.log: list[str] = []
        self.grey: set[str] = set()
        self.drop_when_paused = True
        self.back_after = 0  # looks until the next actor is ready again
        self.looks = 0
        self.taps_since_look = 0
        self.bursts: list[int] = []  # taps between looks
        self.drop_starts = 0
        self.ko_at_target: int | None = None  # this fighter goes down as the glove opens

    # The controller.
    async def press(self, button: str, ms: int) -> None:
        self.log.append(button)
        self.taps_since_look += 1
        if button == "start" and self.drop_starts:
            self.drop_starts -= 1
            return
        if button == "start":
            self.paused = not self.paused
            return
        if self.paused and self.drop_when_paused:
            return
        getattr(self, f"_{self.mode}")(button)

    async def chord(self, buttons, ms):
        self.log.append("+".join(buttons))
        self.mode, self.ready, self.party_gone = "none", [], True

    def _command(self, b):
        if b in ("up", "down"):
            self.cmd = max(0, min(2, self.cmd + (1 if b == "down" else -1)))
        elif b in ("left", "right"):
            cols = sorted(self.ready)
            i = cols.index(self.hand_col) + (1 if b == "right" else -1)
            if 0 <= i < len(cols):
                self.hand_col = cols[i]
        elif b == "a":
            if self.cmd == 0:
                self._aim_enemies()
            else:
                actor = self.party[self.hand_col]
                key = "item" if self.cmd == 2 else actor.name
                self.list = self.items if self.cmd == 2 else actor.techs
                self.idx = self.memory.get(key, 0)
                self.top_row = max(0, self.idx // 2 - 2)
                self.mode = "list"

    def _aim_enemies(self):
        self.targets = sorted((x, y, n) for x, y, n, hp in self.enemies if hp > 0)
        self.glove = len(self.targets) // 2
        self.mode = "target"

    def _list(self, b):
        n, i = len(self.list), self.idx
        if b == "down":
            if i + 2 < n:
                i += 2
            elif i % 2 == 0 and i + 1 < n:
                i += 1
        elif b == "up" and i >= 2:
            i -= 2
        elif b == "right" and i % 2 == 0 and i + 1 < n:
            i += 1
        elif b == "left" and i % 2 == 1:
            i -= 1
        elif b == "b":
            self.mode = "command"
            if self.hand_col not in self.ready:  # they went down: the next one's window
                self.hand_col = self.ready[0] if self.ready else None
                self.mode, self.cmd = ("command", 0) if self.ready else ("none", 0)
            return
        elif b == "a":
            name = self.list[i]
            if name in self.grey:
                return
            self.memory["item" if self.cmd == 2 else self.party[self.hand_col].name] = i
            self.picked = name
            if name == "Athenian Water" and not any(m.hp == 0 for m in self.party):
                return  # no one to use it on: nothing happens
            heal = name in HEALS or name == "Athenian Water"
            if heal:
                self.targets = [(m.at[0] - 8, m.at[1] + 60, m.name) for m in self.party]
                down = [k for k, m in enumerate(self.party) if m.hp == 0]
                self.glove = down[0] if name == "Athenian Water" else self.hand_col
                self.mode = "target"
                if self.ko_at_target is not None:
                    self.party[self.ko_at_target].hp = 0
                    self.ready.remove(self.ko_at_target)
                    self.ko_at_target = None
            else:
                self._aim_enemies()
            return
        self.idx = i
        r = i // 2
        if r > self.top_row + 2:
            self.top_row = r - 2
        self.top_row = min(self.top_row, r)

    def _target(self, b):
        if b in ("left", "right", "up", "down"):
            self.glove = (self.glove + (1 if b in ("right", "down") else -1)) % len(self.targets)
        elif b == "b":
            self.mode = "list" if self.cmd else "command"
        elif b == "a":
            x, y, name = self.targets[self.glove]
            actor = self.hand_col
            if self.cmd == 0 or getattr(self, "picked", "") not in HEALS | {"Athenian Water": 1}:
                for k, e in enumerate(self.enemies):
                    if e[2] == name and (e[0], e[1]) == (x, y):
                        self.enemies[k] = (e[0], e[1], e[2], e[3] - 1)
                self.top = "" if any(e[3] > 0 for e in self.enemies) else "Earned 10 ExP."
            else:
                m = next(m for m in self.party if m.name == name)
                m.hp = 100 if self.picked == "Athenian Water" else min(
                    m.hp_max, m.hp + HEALS[self.picked])
            self.ready.remove(actor)
            if self.top.startswith("Earned"):
                self.ready = []  # won: the menus go
            self.mode = "command" if self.ready else "none"
            self.hand_col = self.ready[0] if self.ready else None
            self.cmd = 0
            self.back_after = 2
            self.came_back = actor

    def _none(self, b):
        if b == "a" and self.top:
            self.top = "Crono's level increased!" if "ExP" in self.top else ""
            if not self.top:
                self.over = True

    # The screen.
    async def grab(self):
        self.looks += 1
        if self.taps_since_look:
            self.bursts.append(self.taps_since_look)
        self.taps_since_look = 0
        if self.mode == "none" and not self.paused and self.back_after and \
                not self.top:
            self.back_after -= 1
            if not self.back_after:
                self.ready.append(self.came_back)
                self.hand_col, self.mode, self.cmd = self.came_back, "command", 0
        return self.looks

    def read(self, _frame, full=True) -> State:
        st = State(paused=self.paused, top=self.top if full else "")
        if getattr(self, "over", False) or getattr(self, "party_gone", False):
            return State()
        party = [Member(m.name, m.hp, m.hp_max, m.mp, 99, "white") for m in self.party]
        if self.mode in ("command", "none", "target"):
            st.party = party
        if self.mode == "command":
            st.kind = "command"
            st.columns = sorted(self.ready)
            st.actor_col = self.hand_col
            st.command = ("Attack", "Tech", "Item")[self.cmd]
            st.actor = self.party[self.hand_col].name
            party[self.hand_col].colour = "yellow"
            if self.party[self.hand_col].name not in getattr(self, "no_bracket", ()):
                st.bracket = self.party[self.hand_col].at
            st.hand = (300, 1200)
        elif self.mode == "list" and not self.list:
            st.kind, st.hand = "list", (300, 1232)  # an empty tech list: the MP column only
        elif self.mode == "list":
            st.kind = "list"
            rows = [self.list[k:k + 2] for k in range(0, len(self.list), 2)]
            st.entries = rows[self.top_row:self.top_row + 3]
            st.usable_rows = [[e not in self.grey for e in r] for r in st.entries]
            st.entry = self.list[self.idx]
            st.entry_rc = (self.idx // 2 - self.top_row, self.idx % 2)
            st.usable = st.entry not in self.grey
            st.hand = (300, 1300)
        elif self.mode == "target":
            st.kind = "target"
            x, y, name = self.targets[self.glove]
            st.gloves = [(x, y)]
            if self.cmd == 0:
                st.top = name
        if self.paused:
            st.party, st.kind = [], "none"  # the dim pause reads badly
        return st


def party():
    return [Fighter("Crono", 780, 780, 64, (1884, 960), ["Cyclone", "Wind Slash", "Raise"]),
            Fighter("Frog", 494, 494, 65, (1100, 1200), ["Slurp", "Heal"]),
            Fighter("Marle", 430, 430, 66, (1336, 848), ["Aura", "Allure", "Ice", "Cure"])]


def enemies():
    return [(1000, 500, "Gargoyle", 3), (1400, 450, "Gargoyle", 3), (1800, 520, "Imp", 3)]


def executor(game: FakeCT) -> Executor:
    return Executor(game.press, game.grab, read=game.read, tuning=FAST, chord=game.chord)


async def test_attack_with_whoever_is_up_is_two_presses():
    game = FakeCT(party(), enemies())
    out = await executor(game).act(Intent("", "attack"))
    assert out.ok, out.text
    assert game.log[:2] == ["a", "a"]
    assert game.paused  # left paused for the player's next request
    assert sum(e[3] for e in game.enemies) == 8


async def test_an_attack_with_nothing_seen_says_it_went_in():
    game = FakeCT(party(), enemies())
    out = await executor(game).act(Intent("", "attack"))
    assert "attack went in" in out.text and "No party HP" not in out.text


async def test_paused_first_then_the_named_actor_in_another_column():
    game = FakeCT(party(), enemies(), ready=(0, 2))
    game.paused = True
    out = await executor(game).act(Intent("Marle", "attack", target="left"))
    assert out.ok, out.text
    assert game.log[:3] == ["start", "right", "a"]
    assert game.enemies[0][3] == 2  # the one on the left
    assert out.text.startswith("Marle")


async def test_presses_while_paused_would_be_lost_so_it_unpauses_first():
    game = FakeCT(party(), enemies())
    game.paused = True
    out = await executor(game).act(Intent("Crono", "attack"))
    assert out.ok and game.log[0] == "start"


async def test_revive_navigates_the_two_column_list_from_where_it_was_left():
    p = party()
    p[0].hp = 0  # Crono knocked out; Marle acts
    game = FakeCT(p, enemies(), ready=(2,))
    game.memory["item"] = ITEMS.index("Panacea")
    out = await executor(game).act(Intent("Marle", "item", "Athenian Water", "Crono"))
    assert out.ok, out.text
    assert p[0].hp == 100
    assert "Crono HP 0 to 100 (revived)" in out.text
    # down twice to Item, a, right to Athenian Water, a, a on the knocked-out one
    assert game.log[:6] == ["down", "down", "a", "right", "a", "a"]


async def test_a_revive_starts_from_a_used_up_items_empty_slot():
    # Live, Giga Gaia 2026-11-01: an item ran out, the hand was left on its empty slot
    # and every revive stopped at "can't see the hand in the list" until the party fell.
    p = party()
    p[0].hp = 0
    items = [*ITEMS[:6], "", *ITEMS[6:]]  # an empty slot left of Lapis
    game = FakeCT(p, enemies(), ready=(2,), items=items)
    game.memory["item"] = 6
    out = await executor(game).act(Intent("Marle", "item", "Athenian Water", "Crono"))
    assert out.ok, out.text
    assert p[0].hp == 100
    assert game.log[:7] == ["down", "down", "a", "up", "right", "a", "a"]


async def test_the_one_choosing_knocked_out_at_the_glove_hands_it_to_the_next():
    # Live, Woe 2026-11-01: Frog went down with his Athenian Water's glove up; it steered
    # the glove round for a minute before the model asked Marle instead.
    p = party()
    p[0].hp = 0
    game = FakeCT(p, enemies(), ready=(1, 2))
    game.ko_at_target = 1
    out = await executor(game).act(Intent("", "item", "Athenian Water", "Crono"))
    assert out.ok, out.text
    assert out.text.startswith("Marle: item Athenian Water on Crono")
    assert p[0].hp == 100 and p[1].hp == 0


async def test_a_tech_when_none_are_learned_yet_stops_and_says_so():
    # Live, Guardia Forest 2026-11-01: Marle at level 1 has no techs; the empty list read
    # as "not a battle" and the agent tried for minutes.
    p = party()
    p[2].techs = []
    p[0].hp = 300
    game = FakeCT(p, enemies(), ready=(2,))
    out = await executor(game).act(Intent("Marle", "tech", "Aura", "Crono"))
    assert not out.ok
    assert "list is empty" in out.text and "Aura" in out.text
    assert game.mode == "command"  # backed out to the commands


async def test_after_a_pause_start_and_the_command_go_in_one_burst():
    game = FakeCT(party(), enemies(), ready=(0, 2))
    game.party[0].hp = 300
    ex = executor(game)
    await ex.pause()  # the request came in: paused while the model thinks
    game.log.clear()
    game.bursts.clear()
    out = await ex.act(Intent("Crono", "item", "Mid-Potion", "Crono"))
    assert out.ok, out.text
    assert game.log[:4] == ["start", "down", "down", "a"]
    assert game.bursts[0] == 4  # no look until the Item list is open


async def test_a_dropped_start_drops_the_burst_and_it_recovers():
    game = FakeCT(party(), enemies(), ready=(0,))
    ex = executor(game)
    await ex.pause()
    game.drop_starts = 1
    out = await ex.act(Intent("Crono", "attack", target="left"))
    assert out.ok, out.text
    assert game.enemies[0][3] == 2


async def test_an_entry_out_of_view_is_found_by_scrolling():
    game = FakeCT(party(), enemies(), ready=(2,))
    game.memory["item"] = 0
    out = await executor(game).act(Intent("Marle", "item", "Shield Sphere"))
    assert out.ok, out.text
    assert game.memory["item"] == ITEMS.index("Shield Sphere")


async def test_an_item_that_isnt_there_stops_backed_out_and_paused():
    game = FakeCT(party(), enemies(), ready=(2,))
    ex = executor(game)
    out = await ex.act(Intent("Marle", "item", "Megalixir"))
    assert not out.ok
    assert "isn't in the item list" in out.text and "Shield Sphere" in out.text
    assert game.mode == "command" and game.paused
    # Next time it knows the list: no scan needed to say what's there.
    assert ("Barrier Sphere", "Shield Sphere") in ex.lists["item"]


async def test_a_greyed_out_tech_stops():
    game = FakeCT(party(), enemies(), ready=(0,))
    game.grey.add("Raise")
    out = await executor(game).act(Intent("Crono", "tech", "Raise", "Frog"))
    assert not out.ok and "greyed out" in out.text


async def test_a_revive_with_no_one_down_says_the_pick_did_nothing():
    game = FakeCT(party(), enemies(), ready=(2,))
    out = await executor(game).act(Intent("Marle", "item", "Athenian Water", "Crono"))
    assert not out.ok
    assert "did nothing" in out.text
    assert game.mode == "command"


async def test_heal_an_ally_moves_the_glove_by_where_they_stood_on_their_turn():
    p = party()
    p[1].hp = 100
    game = FakeCT(p, enemies(), ready=(0, 1, 2))
    ex = executor(game)
    # Everyone's turn comes up once first, so the bracket shows where each one stands.
    for name in ("Crono", "Frog"):
        assert (await ex.act(Intent(name, "attack"), watch=False, pause_after=False)).ok
    out = await ex.act(Intent("Marle", "tech", "Cure", "Frog"))
    assert out.ok, out.text
    assert p[1].hp == 400
    assert "Frog HP 100 to 400" in out.text


async def test_an_actor_knocked_out_or_not_ready_stops():
    p = party()
    p[1].hp = 0
    game = FakeCT(p, enemies(), ready=(0,))
    out = await executor(game).act(Intent("Frog", "attack"))
    assert not out.ok and "knocked out" in out.text
    out = await executor(game).act(Intent("Marle", "attack"))
    assert not out.ok and "isn't ready" in out.text


async def test_win_attacks_until_the_spoils_and_reads_them():
    game = FakeCT(party(), [(1000, 500, "Imp", 2)], ready=(0, 1, 2))
    out = await executor(game).win()
    assert out.ok, out.text
    assert "Earned 10 ExP." in out.text and "Crono's level increased!" in out.text


async def test_win_heals_on_a_surprise_with_the_decision():
    p = party()
    p[1].hp = 40
    game = FakeCT(p, [(1000, 500, "Imp", 2)], ready=(2,))
    asked = []

    async def decide(st, why):
        asked.append(why)
        return Intent(st.actor, "tech", "Cure", "Frog")

    async def identify(_frame, glove, names):  # the model, asked only when needed
        return next(n for x, y, n in game.targets if (x, y) == glove)

    ex = executor(game)
    ex.identify = identify
    out = await ex.win(decide)
    assert out.ok, out.text
    assert asked == ["Frog low on HP"]
    assert p[1].hp == 340


async def test_win_falls_back_to_items_when_the_decision_fails():
    p = party()
    p[0].hp = 0
    game = FakeCT(p, [(1000, 500, "Imp", 1)], ready=(1,))

    async def decide(st, why):
        raise TimeoutError

    out = await executor(game).win(decide)
    assert out.ok, out.text
    assert p[0].hp == 100  # Frog has no revive tech: Athenian Water


async def test_run_away_holds_l_and_r():
    game = FakeCT(party(), enemies())
    out = await executor(game).run_away()
    assert out.ok and "lb+rb" in game.log


async def test_stop_from_the_player_pauses():
    game = FakeCT(party(), enemies())
    out = await Executor(game.press, game.grab, read=game.read, tuning=FAST,
                         cancelled=lambda: True).act(Intent("", "attack"))
    assert not out.ok and "player" in out.text
    assert game.log == ["start"]


def test_list_rows_merge_in_order():
    order = merge_rows([], [["Potion", "Mid-Potion"], ["Hi-Potion", "Mid-Ether"],
                            ["Panacea", "Athenian Water"]])
    order = merge_rows(order, [["Hi-Potion", "Mid-Ether"], ["Panacea", "Athenian Water"],
                               ["Lapis", "Hi-Ether"]])
    assert [r[0] for r in order] == ["Potion", "Hi-Potion", "Panacea", "Lapis"]
    order = merge_rows(order[2:], [["Potion", "Mid-Potion"], ["Hi-Potion", "Mid-Ether"],
                                   ["Panacea", "Athenian Water"]])
    assert [r[0] for r in order] == ["Potion", "Hi-Potion", "Panacea", "Lapis"]


@pytest.mark.parametrize(("hp", "want"), [
    ((780, 494, 430), "attack"),
    ((0, 494, 430), "revive"),
    ((780, 100, 430), "heal"),
])
def test_the_standing_plan_and_its_fallbacks(hp, want):
    st = State(kind="command", actor="Marle", party=[
        Member(n, h, m, 60, 66, "white") for n, h, m in
        zip(("Crono", "Frog", "Marle"), hp, (780, 494, 430), strict=True)])
    if want == "attack":
        assert battle.plan(st) == Intent("Marle", "attack")
        return
    assert battle.plan(st) is None
    first = battle.fallbacks(st)[0]
    if want == "revive":
        assert (first.command, first.name, first.target) == ("item", "Athenian Water", "Crono")
    else:
        assert (first.command, first.name, first.target) == ("tech", "Cure", "Frog")


async def test_the_agent_pauses_a_ct_fight_and_offers_only_the_battle_tools(monkeypatch):
    import json
    from types import SimpleNamespace

    from PIL import Image

    from playertwo import agent as agent_mod
    from playertwo.battle import Outcome
    from playertwo.controls import DryRun
    from playertwo.screen import Frame

    fight = State(kind="command", actor="Crono", columns=[0], actor_col=0, command="Attack",
                  party=[Member("Crono", 780, 780, 64, 64, "yellow")])
    monkeypatch.setattr(agent_mod.ctsteam, "read", lambda img, full=True: fight)

    class Muse:
        s = SimpleNamespace(image_max_side=640)

        def __init__(self, replies):
            self.replies, self.tools_seen, self.texts = replies, [], []

        async def tools(self, messages, tools, effort=None, max_tokens=0):
            self.tools_seen.append({t["function"]["name"] for t in tools})
            self.texts.append(json.dumps(messages[-1]["content"], default=str))
            name, args = self.replies.pop(0)
            call = SimpleNamespace(id="c", function=SimpleNamespace(
                name=name, arguments=json.dumps(args)),
                model_dump=lambda: {"id": "c", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args)}})
            return SimpleNamespace(content="", tool_calls=[call])

    class Ex:
        held = None

        def __init__(self):
            self.acts, self.paused, self.ally_at = [], 0, {}

        def new_fight(self):
            self.ally_at.clear()

        async def pause(self, st=None):
            self.paused += 1
            return fight

        async def act(self, intent):
            self.acts.append(intent)
            return Outcome(True, "Crono: attack. Seen: the bar said: 120", fight)

    async def grab():
        return Frame(Image.new("RGB", (1280, 720)), 0)

    async def speak(_text):
        pass

    muse = Muse([("press", {"buttons": ["a"]}),
                 ("battle", {"actor": "Crono", "command": "attack", "target": "left"}),
                 ("finish", {"text": "Crono hit it."})])
    ag = agent_mod.Agent(muse, DryRun(), grab, speak, game="Chrono Trigger")
    ag._executor = ex = Ex()
    run = await ag.run("Crono, attack the one on the left")
    assert ex.paused >= 1  # paused before the model thought
    assert "battle" in muse.tools_seen[0] and "press" not in muse.tools_seen[0]
    assert "Crono HP 780/780" in muse.texts[0]
    assert run.steps[0].result.startswith("not pressed")
    assert ex.acts == [Intent("Crono", "attack", "", "left")]
    assert run.steps[1].result.startswith("done: Crono: attack")


async def test_after_a_won_fight_the_model_only_reports_it(monkeypatch):
    # Live, Woe 2026-11-01: "win this fight" won, the spoils screen still read as a fight,
    # and the model called fight again (2 of 6), then said there was no fight.
    import json
    from types import SimpleNamespace

    from PIL import Image

    from playertwo import agent as agent_mod
    from playertwo.battle import Outcome
    from playertwo.controls import DryRun
    from playertwo.screen import Frame

    spoils = State(kind="none", party=[Member("Crono", 653, 780, 54, 64, "white")],
                   top="Earned 648 EXP.")
    monkeypatch.setattr(agent_mod.ctsteam, "read", lambda img, full=True: spoils)

    class Muse:
        s = SimpleNamespace(image_max_side=640)

        def __init__(self):
            self.tools_seen, self.texts = [], []

        async def tools(self, messages, tools, effort=None, max_tokens=0):
            self.tools_seen.append({t["function"]["name"] for t in tools})
            self.texts.append(json.dumps(messages[-1]["content"], default=str))
            call = SimpleNamespace(id="c", function=SimpleNamespace(
                name="finish", arguments=json.dumps({"text": "We won."})),
                model_dump=lambda: {"id": "c", "type": "function", "function": {
                    "name": "finish", "arguments": json.dumps({"text": "We won."})}})
            return SimpleNamespace(content="", tool_calls=[call])

    class Ex:
        ally_at: ClassVar[dict] = {}
        held = None

        def new_fight(self):
            pass

        async def pause(self, st=None):
            return spoils

        async def win(self, decide, say=None, turn=None):
            return Outcome(True, "won after 9 commands. Earned 648 EXP.", spoils)

    async def grab():
        return Frame(Image.new("RGB", (1280, 720)), 0)

    async def speak(_text):
        pass

    muse = Muse()
    ag = agent_mod.Agent(muse, DryRun(), grab, speak, game="Chrono Trigger")
    ag._executor = Ex()
    run = await ag.run("win this fight for me")
    assert run.steps[0].tool == "fight" and run.steps[0].result.startswith("done: won")
    assert "fight" not in muse.tools_seen[0] and "battle" not in muse.tools_seen[0]
    assert "fight(win) was done already" in muse.texts[0]


def test_specks_on_the_bar_are_not_a_message():
    from playertwo.battle import message

    assert not message("-") and not message("- .") and not message("Q") and not message("")
    assert message("Heal") and message("Combo Attack: Reduces HP to 1")


async def test_a_tech_whose_name_never_shows_went_off_when_its_mp_is_spent():
    """Live 2026-11-01: "Crono, use Cyclone" twice reported "hadn't gone off yet" while
    Crono's MP had gone 64 to 62 and the imps were gone: the name never showed on the bar."""
    game = FakeCT(party(), enemies(), ready=(0,))
    target = game._target

    def spend(b):
        if b == "a" and game.cmd == 1:
            game.party[0].mp -= 2
        target(b)

    game._target = spend
    out = await executor(game).act(Intent("Crono", "tech", "Cyclone"))
    assert out.ok, out.text
    assert "hadn't gone off" not in out.text and "Crono MP 64 to 62" in out.text


async def test_a_heal_is_watched_from_its_mp_until_the_hp_lands():
    """Live 2026-11-01: Frog's Heal spent its MP, the watch stopped before the HP moved
    ("no party HP changed"), and the model healed twice more for one "heal Marle"."""
    game = FakeCT(party(), enemies(), ready=(1,))
    game.party[2].hp = 1
    target, read = game._target, game.read
    spent_at = []

    def spend(b):
        if b == "a" and game.cmd == 1:
            game.party[1].mp -= 2
            spent_at.append(time.monotonic())
            game.mode, game.ready = "none", []  # nothing lands at the confirm: see later
            return
        target(b)

    def later(frame, full=True):
        if spent_at and time.monotonic() - spent_at[0] > 0.5 and game.party[2].hp == 1:
            game.party[2].hp = 200
        return read(frame, full)

    game._target, game.read = spend, later
    out = await executor(game).act(Intent("Frog", "tech", "Heal"))
    assert "Marle HP 1 to 200" in out.text, out.text


async def test_left_past_the_last_enemy_onto_an_ally_comes_back():
    """Live: "left" from the leftmost Gargoyle put the glove on Frog over the windows,
    where it didn't read; the attack stopped with the glove still up, and the next two
    commands found no command window."""
    game = FakeCT(party(), enemies())
    target = game._target
    read = game.read

    def onto_ally(b):
        if b == "left" and game.glove == 0 and not getattr(game, "on_ally", False):
            game.on_ally = True  # off the enemies, onto Frog over the windows
            return
        if b == "right" and getattr(game, "on_ally", False):
            game.on_ally = False
            return
        target(b)

    def reads(frame, full=True):
        if getattr(game, "on_ally", False):
            return State(kind="command", party=[Member(m.name, m.hp, m.hp_max, m.mp, 99,
                                                       "white") for m in game.party])
        return read(frame, full)

    game._target, game.read = onto_ally, reads
    out = await executor(game).act(Intent("", "attack", target="left"))
    assert out.ok, out.text
    assert game.enemies[0][3] == 2  # the leftmost one was hit


async def test_a_glove_left_on_an_ally_over_the_windows_is_backed_out_first():
    game = FakeCT(party(), enemies())
    game.on_ally, game.mode = True, "target"  # left so by the last request
    target, read = game._target, game.read

    def keys(b):
        if game.on_ally:
            if b == "b":
                game.on_ally = False
                game.mode = "command"
            return
        target(b)

    def reads(frame, full=True):
        if game.on_ally:
            return State(kind="command", columns=[0, 1, 2], party=[
                Member(m.name, m.hp, m.hp_max, m.mp, 99, "white") for m in game.party])
        return read(frame, full)

    game._target, game.read = keys, reads
    out = await executor(game).act(Intent("", "attack"))
    assert out.ok, out.text
    assert game.log[0] == "b"


async def test_a_heal_glove_opens_on_its_user_so_no_model_is_asked():
    """Live 2026-11-01: Frog stands behind the windows and was never bracketed, so a glove
    on him (or on the one healing, before their bracket was seen) asked the model who it
    was: 3 to 10 s a look, 13 s of a 25 s "heal Crono"."""
    p = party()
    p[1].hp = 100
    game = FakeCT(p, enemies(), ready=(0, 2))
    game.no_bracket = {"Frog", "Marle"}

    async def identify(_frame, glove, names):
        raise AssertionError("asked the model")

    ex = executor(game)
    ex.identify = identify
    assert (await ex.act(Intent("Crono", "attack"), watch=False, pause_after=False)).ok
    out = await ex.act(Intent("Marle", "tech", "Cure", "Frog"))
    assert out.ok, out.text
    assert p[1].hp == 400 and "Frog HP 100 to 400" in out.text
    assert {"Marle", "Frog"} <= set(ex.ally_at)  # learnt from the glove for next time


async def test_a_heal_on_someone_at_full_health_presses_nothing():
    """Live: "heal Marle" with Marle at 430/430 had Frog use Slurp on her."""
    game = FakeCT(party(), enemies(), ready=(1,))
    ex = executor(game)
    await ex.pause()  # the request came in: paused while the model thinks
    game.log.clear()
    out = await ex.act(Intent("", "tech", "Slurp", "Marle"))
    assert not out.ok and "Marle is at full HP (430/430)" in out.text
    assert game.log == [] and game.paused
    out = await ex.act(Intent("Frog", "tech", "Heal"))
    assert not out.ok and "everyone standing is at full HP" in out.text


async def test_win_doesnt_end_when_the_party_box_drops_out_a_moment():
    """Live, Guardia Forest: leaves behind the windows hid the party box for 1.5 s and
    "the fight ended after 1 command", three times on one trip."""
    game = FakeCT(party(), [(1000, 500, "Imp", 2)], ready=(0,))
    read = game.read
    blank = {"n": 0}

    def flaky(frame, full=True):
        st = read(frame, full)
        if st.kind == "none" and st.party and blank["n"] < 3:
            blank["n"] += 1
            return State(top="r. utvA-,t r")  # nothing read: no party box
        return read(frame, full)

    game.read = flaky
    ex = Executor(game.press, game.grab, read=game.read,
                  tuning=Tuning(**{**FAST.__dict__, "gone_s": 0.5, "poll_s": 0.2}))
    out = await ex.win()
    assert out.ok and "Earned 10 ExP." in out.text, out.text


async def test_win_with_nothing_to_heal_with_attacks_without_asking_each_turn():
    """A level 1 party: no techs, no items. Low on HP, it attacks; it looked once."""
    p = [Fighter("Crono", 20, 70, 8, (1884, 960), []),
         Fighter("Marle", 65, 65, 12, (1336, 848), [])]
    game = FakeCT(p, [(1000, 500, "Hetake", 6)], ready=(0, 1), items=["Shelter"])
    asked = []

    async def decide(st, why):
        asked.append(why)
        return Intent(st.actor, "item", "Potion", "Crono")

    out = await executor(game).win(decide)
    assert out.ok and "won after 6 commands" in out.text, out.text
    # Once each (Crono's items, Marle's techs): then it knows there's nothing to use.
    assert len(asked) <= 2


@pytest.mark.parametrize(("said", "hp", "ready", "want"), [
    ("Frog", (780, 100, 430), (2,), ("Marle", "tech", "Cure", "Frog")),
    ("Frog", (780, 100, 430), (1,), ("Frog", "tech", "Slurp", "Frog")),
    ("crono", (300, 494, 430), (0,), ("Crono", "item", "Mid-Potion", "Crono")),
    ("everyone", (300, 100, 430), (1, 2), ("Frog", "tech", "Heal", "")),
    ("whoever is hurt", (780, 494, 200), (0, 2), ("Marle", "tech", "Cure", "Marle")),
    ("Marle", (780, 494, 430), (1,), "Marle is at full health already, 430 of 430."),
    ("everyone up", (780, 494, 430), (1,), "Everyone's at full health already."),
    ("the castle", (780, 494, 430), (1,), None),
])
def test_heal_x_is_chosen_from_the_screen(said, hp, ready, want):
    names = ("Crono", "Frog", "Marle")
    st = State(kind="command", columns=list(ready), actor_col=ready[0],
               actor=names[ready[0]], slots=list(names), party=[
                   Member(n, h, m, 60, 66, "white") for n, h, m in
                   zip(names, hp, (780, 494, 430), strict=True)])
    plan = battle.heal_plan(st, said)
    if want is None:
        assert plan is None
    elif isinstance(want, str):
        assert plan.say == want and not plan.intents
    else:
        first = plan.intents[0]
        assert (first.actor, first.command, first.name, first.target) == want


def test_a_heal_is_said_from_the_hp_it_gave():
    before = State(party=[Member("Frog", 1, 494, 60, 65, "white")])
    after = State(party=[Member("Frog", 126, 494, 58, 65, "white")])
    out = battle.Outcome(True, "Frog: tech Slurp on Frog. Seen: Frog HP 1 to 126", after)
    assert battle.heal_line(Intent("Frog", "tech", "Slurp", "Frog"), out, before) == \
        "Frog's Slurp healed Frog: 126 of 494."
    out = battle.Outcome(True, "Frog: tech Slurp on Frog.", before)
    assert battle.heal_line(Intent("Frog", "tech", "Slurp", "Frog"), out, before) == \
        "Frog's Slurp is on its way to Frog."


def _muse(replies):
    import json
    from types import SimpleNamespace

    class Muse:
        s = SimpleNamespace(image_max_side=640)

        def __init__(self):
            self.replies, self.tools_seen = list(replies), []

        async def tools(self, messages, tools, effort=None, max_tokens=0):
            self.tools_seen.append({t["function"]["name"] for t in tools})
            name, args = self.replies.pop(0)
            call = SimpleNamespace(id="c", function=SimpleNamespace(
                name=name, arguments=json.dumps(args)),
                model_dump=lambda: {"id": "c", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args)}})
            return SimpleNamespace(content="", tool_calls=[call])

    return Muse()


def _agent(monkeypatch, game, muse):
    from PIL import Image

    from playertwo import agent as agent_mod
    from playertwo.controls import DryRun
    from playertwo.screen import Frame

    monkeypatch.setattr(agent_mod.ctsteam, "read", lambda img, full=True: game.read(img, full))
    said = []

    async def grab():
        return Frame(Image.new("RGB", (1280, 720)), 0)

    async def speak(text):
        said.append(text)

    ag = agent_mod.Agent(muse, DryRun(), grab, speak, game="Chrono Trigger")
    ag._executor = executor(game)
    return ag, said


async def test_heal_x_in_a_fight_is_one_command_and_no_model_turn(monkeypatch):
    """Live: "heal X" passed 7/9 in 13 to 42 s; five runs healed two or three times."""
    p = party()
    p[1].hp = 100
    game = FakeCT(p, enemies(), ready=(0,))
    muse = _muse([])
    ag, said = _agent(monkeypatch, game, muse)
    ag.note_battle(new=True)
    ag.note_battle(game.read(None))  # the reader saw Crono's turn before the request
    game.ready, game.hand_col = [2], 2
    run = await ag.run("heal Frog")
    assert muse.tools_seen == []  # no model call at all
    assert p[1].hp == 400
    assert said == ["Marle's Cure healed Frog: 400 of 494."] and run.outcome == said[0]
    assert [s.tool for s in run.steps] == ["battle"]
    run = await ag.run("heal Marle")
    assert said[-1] == "Marle is at full health already, 430 of 430." and len(run.steps) == 0


async def test_the_model_gets_one_battle_command_for_one_asked(monkeypatch):
    game = FakeCT(party(), enemies(), ready=(0, 2))
    muse = _muse([("battle", {"command": "attack", "target": "left"}),
                  ("battle", {"actor": "Marle", "command": "attack"}),
                  ("finish", {"text": "Crono hit the one on the left."})])
    ag, said = _agent(monkeypatch, game, muse)
    run = await ag.run("attack the one on the left")
    assert "battle" in muse.tools_seen[0]
    assert muse.tools_seen[1] == {"finish"}
    assert run.steps[1].result.startswith("not done: the command the player asked for")
    assert sum(e[3] for e in game.enemies) == 8  # one hit, not two
    assert said[-1] == "Crono hit the one on the left."


async def test_a_request_naming_no_command_isnt_held_to_one(monkeypatch):
    """A trip's fights ("take me to Guardia Castle") or "deal with them": no limit."""
    game = FakeCT(party(), enemies(), ready=(0, 2))
    muse = _muse([("battle", {"command": "attack", "target": "left"}),
                  ("battle", {"actor": "Marle", "command": "attack"}),
                  ("finish", {"text": "Two hits."})])
    ag, _said = _agent(monkeypatch, game, muse)
    run = await ag.run("deal with these monsters")
    assert "battle" in muse.tools_seen[1]
    assert run.steps[1].result.startswith("done:")


def _counting_muse(counts):
    """A model that answers each enemy count in turn (None: a failed look)."""
    from types import SimpleNamespace

    class Muse:
        s = SimpleNamespace(image_max_side=640)

        def __init__(self):
            self.counts, self.prompts = list(counts), []

        async def json(self, messages, effort=None, max_tokens=0):
            self.prompts.append(messages[0]["content"][-1]["text"])
            n = self.counts.pop(0)
            if n is None:
                return {}
            named = "Gargoyle" in self.prompts[-1]  # told the name the screen showed
            return {"enemies": ["somewhere"] * n,
                    "look": "Gargoyles" if named else "golden winged imps"}

    return Muse()


PIC = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (64, 36))


async def _drain(ag):
    while ag._counting is not None and not ag._counting.done():
        await ag._counting


async def test_enemies_counted_fewer_twice_are_said_down_once(monkeypatch):
    from playertwo.battle import Foes

    game = FakeCT(party(), enemies(), ready=(0,))
    muse = _counting_muse([2, 2])
    ag, said = _agent(monkeypatch, game, muse)
    ag.foes = Foes(3, "Gargoyles")
    ag.recount(PIC, ["Crono", "Frog", "Marle"])
    ag.recount(PIC, ["Crono", "Frog", "Marle"])  # one count at a time: this one is dropped
    await _drain(ag)
    assert said == ["One Gargoyle down, two left."]
    assert ag.foes.count == 2 and ag.foes.look == "Gargoyles"


async def test_a_drop_a_second_look_doesnt_see_isnt_said(monkeypatch):
    from playertwo.battle import Foes

    game = FakeCT(party(), enemies(), ready=(0,))
    ag, said = _agent(monkeypatch, game, _counting_muse([2, 3, 4, None]))
    ag.foes = Foes(3, "Gargoyles")
    ag.recount(PIC, [])
    await _drain(ag)
    assert said == [] and ag.foes.count == 3  # a miscount (bench: 9 in 10 right)
    ag.recount(PIC, [])
    await _drain(ag)
    assert said == [] and ag.foes.count == 4  # more: taken quietly
    ag.recount(PIC, [])
    await _drain(ag)
    assert said == [] and ag.foes.count == 4  # a failed look changes nothing


async def test_what_are_we_fighting_counts_them_and_gives_the_partys_hp(monkeypatch):
    from playertwo.battle import foe_name

    p = party()
    p[1].hp = 1
    game = FakeCT(p, enemies(), ready=(0,))
    muse = _counting_muse([3])
    ag, _said = _agent(monkeypatch, game, muse)
    st = game.read(None)
    st.kind, st.top, st.gloves = "target", "Gargoyle", [(1200, 500)]
    assert foe_name(st) == "Gargoyle"
    ag.note_battle(st)  # the reader saw the glove name one
    line = await ag.what_we_fight(PIC, game.read(None))
    assert line == "Three Gargoyles. Crono's full, Frog's at 1 of 494, Marle's full."
    assert "Gargoyle" in muse.prompts[0]  # the name the screen showed, for the look's words
    st.top = "Frog"
    assert foe_name(st) == ""  # the glove on an ally


def test_turn_lines_name_who_acted_and_who_is_in_danger_only():
    from playertwo.battle import Intent, party_brief, progress

    p = party()
    st = FakeCT(p, enemies()).read(None)
    assert progress(st, Intent("Crono", "attack")) == "Crono attacks."
    p[1].hp, p[2].hp = 1, 0
    st = FakeCT(p, enemies()).read(None)
    assert progress(st, Intent("Crono", "attack")) == "Crono attacks. Frog 1 HP, Marle is down."
    assert party_brief(st) == "Crono's full, Frog's at 1 of 494, Marle's down."
