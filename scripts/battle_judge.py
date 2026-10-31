"""Judge a battle task by the game's memory, for scripts/emu_tasks.py: a thread reads the
fight every half second while Player Two plays it, and the task's `ram_check` names what
must have happened. All must hold:

    "over"            the fight ended (it was on at the start)
    "won"             it ended with every enemy at 0 HP
    "fled"            it ended with an enemy still standing
    "alive"           everyone in the party has HP at the end (and at the fight's end)
    "no_ko"           no one in the party was at 0 HP at any time
    "hit" "hit:7"     an enemy (or the one in that slot) lost HP
    "all_hit"         every enemy lost HP
    "healed:Marle"    her HP went up at some point
    "mp" "mp:Lucca"   someone's (her) MP went down: a tech
    "item"            an item was used: the inventory has fewer, or someone was healed
                      without MP spent (the fight's inventory is written back after it)
    "acts<=2"         at most that many party actions: runs of readings in a row where an
                      enemy lost HP, someone was healed, MP was spent or an item used
    "in_battle"       the fight is still on at the end

Game-specific memory comes from the module the task file names in [game.ram] watch
(scripts/ct_ram.py for Chrono Trigger).
"""

from __future__ import annotations

import importlib
import json
import threading
import time


class Watch:
    def __init__(self, box, module: str = "ct_ram", every: float = 0.5):
        self.box, self.every = box, every
        self.mod = importlib.import_module(module)
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def read(self, off: int, n: int) -> bytes:
        for i in range(6):
            r = self.box.cmd(f"READ_CORE_RAM {off:x} {n}", wait=0.5).split()
            if len(r) == n + 2 and r[2] != "-1":
                return bytes(int(x, 16) for x in r[2:])
            time.sleep(0.1 * (i + 1))
        raise OSError(f"memory read failed at {off:x}")

    def sample(self) -> dict | None:
        try:
            return self.mod.snapshot(self.read)
        except (OSError, ValueError, IndexError):
            return None

    def start(self) -> None:
        s = self.sample()
        if s:
            self.samples.append(s)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            s = self.sample()
            if s:
                self.samples.append(s)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        s = self.sample()
        if s:
            self.samples.append(s)

    def save(self, path) -> None:
        with open(path, "w") as f:
            f.writelines(json.dumps(s) + "\n" for s in self.samples)

    def judge(self, checks: list[str]) -> dict:
        ok, why = judge(self.samples, checks)
        return {"done": ok, "why": why, "ram": summary(self.samples)}


def summary(samples: list[dict]) -> dict:
    if not samples:
        return {}
    a, b = samples[0], samples[-1]

    def short(s):
        return {"battle": s["battle"],
                "party": {p["name"]: f"{p['hp']}/{p['max_hp']} {p['mp']}mp" for p in s["party"]},
                "enemies": {e["slot"]: f"{e['hp']}/{e['max_hp']}" for e in s["enemies"]},
                "items": s["items"]}
    return {"start": short(a), "end": short(b), "samples": len(samples)}


def _events(samples: list[dict]) -> dict:
    """What changed between readings: hits per slot, heals and MP spent per name, items."""
    ev = {"hit": {}, "healed": {}, "mp": {}, "item": 0, "acts": 0}
    before = False  # the reading before showed a change too: the same action going on
    for a, b in zip(samples, samples[1:]):
        if not (a["battle"] or b["battle"]):
            before = False
            continue
        acted, healed, spent = False, False, False
        ea = {e["slot"]: e["hp"] for e in a["enemies"]}
        for e in b["enemies"]:
            if e["slot"] in ea and e["hp"] < ea[e["slot"]]:
                ev["hit"][e["slot"]] = ev["hit"].get(e["slot"], 0) + 1
                acted = True
        pa = {p["name"]: p for p in a["party"]}
        for p in b["party"]:
            q = pa.get(p["name"])
            if q is None or not b["battle"] or not a["battle"]:
                continue
            if p["hp"] > q["hp"]:
                ev["healed"][p["name"]] = ev["healed"].get(p["name"], 0) + 1
                acted = healed = True
            if p["mp"] < q["mp"]:
                ev["mp"][p["name"]] = ev["mp"].get(p["name"], 0) + 1
                acted = spent = True
        # The inventory in memory is only written back after the fight: a heal that cost
        # no MP was an item.
        if b["items"] < a["items"] or (healed and not spent):
            ev["item"] += 1
            acted = True
        ev["acts"] += acted and not before
        before = acted
    return ev


def judge(samples: list[dict], checks: list[str]) -> tuple[bool, str]:
    if len(samples) < 2:
        return False, "no memory readings"
    first, last = samples[0], samples[-1]
    ev = _events(samples)
    start_enemies = {e["slot"] for e in first["enemies"]}
    end_hp = {e["slot"]: e["hp"] for e in last["enemies"]}
    # A party all at 0 HP is a game over, not a fight that ended.
    wiped = any(s["party"] and all(p["hp"] == 0 for p in s["party"]) for s in samples)
    fought = [s for s in samples if s["battle"]]
    over = first["battle"] and not last["battle"] and not wiped
    won = over and all(end_hp.get(s, 0) == 0 for s in start_enemies)
    failed = []
    for c in checks:
        name, _, arg = c.partition(":")
        if name.startswith("acts<="):
            good = ev["acts"] <= int(name[6:])
        elif name == "over":
            good = over
        elif name == "won":
            good = won
        elif name == "fled":
            good = over and not won
        elif name == "alive":
            good = not wiped and all(p["hp"] > 0 for s in (fought[-1:] + [last])
                                     for p in s["party"]) and bool(last["party"])
        elif name == "no_ko":
            good = all(p["hp"] > 0 for s in samples for p in s["party"])
        elif name == "hit":
            good = int(arg) in ev["hit"] if arg else bool(ev["hit"])
        elif name == "all_hit":
            good = start_enemies <= set(ev["hit"])
        elif name == "healed":
            good = arg in ev["healed"] if arg else bool(ev["healed"])
        elif name == "mp":
            good = arg in ev["mp"] if arg else bool(ev["mp"])
        elif name == "item":
            good = ev["item"] > 0
        elif name == "in_battle":
            good = last["battle"]
        else:
            raise ValueError(f"unknown ram_check {c!r}")
        if not good:
            failed.append(c)
    said = (f"fight {'lost' if wiped else 'over' if over else 'on'}{' (won)' if won else ''}; hits "
            f"{ev['hit'] or 'none'}, heals {ev['healed'] or 'none'}, mp "
            f"{ev['mp'] or 'none'}, items used {ev['item']}, actions {ev['acts']}")
    return not failed, (f"failed {failed}: " if failed else "") + said
