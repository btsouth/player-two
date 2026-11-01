"""The Chrono Trigger (Steam) battle executor against a fake game: presses go to a small
model of the battle menus, and each look gets the State the reader would give."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from playertwo import battle
from playertwo.battle import Executor, Intent, Tuning, merge_rows
from playertwo.ctsteam import Member, State

FAST = Tuning(after_tap_s=0.0, poll_s=0.0, pause_wait_s=0.2, ready_wait_s=0.5,
              actor_wait_s=0.3, target_wait_s=0.2, playout_min_s=0.0, playout_quiet_s=0.0,
              playout_max_s=0.2, spoils_wait_s=0.05, win_max_s=5.0, run_chunk_ms=10)
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
            st.bracket = self.party[self.hand_col].at
            st.hand = (300, 1200)
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


async def test_after_a_pause_start_and_the_command_go_in_one_burst():
    game = FakeCT(party(), enemies(), ready=(0, 2))
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
        def __init__(self):
            self.acts, self.paused, self.ally_at = [], 0, {}

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
