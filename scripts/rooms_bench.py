"""Trips out of places by the surveyed rooms, no model, judged by the game's location
number: each trip loads a state in a hidden box, runs scripts/rooms_walk.py there, then
reads where the game is.

    python scripts/rooms_bench.py scripts/tasks/ct_rooms.toml [--boxes 11,12,13]
        [--repeat 3] [--only ID ...] [--out FILE]

Trips are the task file's [[task]]s with a `to` (the location number to walk to, hex):
the same tasks scripts/emu_tasks.py gives the full agent by their `request`. `exit` names
the way out to take (rooms.py's label for it), and `near` = [x, y] says where on a world
map the trip must come out (within 32 pixels), for places with more than one way out.
`target` names someone or something in the place `to` (as the survey named them): the
trip must end standing by them (on a tile rooms.Rooms.stand_by gives, within 6 pixels of
its middle across the way it faces), and with `talk = true` pressing a there must open a
text box (the game's memory: [game.ram] dialog, when it's known).

PLAYER_TWO_ROOMS: a folder of rooms to try instead of the game's own (copied into each box).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emu_tasks import KEYMAP, ROOT, Box
from survey import Game

PY = str(ROOT / ".venv" / "bin" / "python")


def rooms_here():
    from playertwo.atlas import ATLASES
    from playertwo.rooms import Rooms

    own = os.environ.get("PLAYER_TWO_ROOMS")
    return Rooms(Path(own) if own else ATLASES / "chrono-trigger" / "rooms")


def by_target(box: Box, task: dict, loc: int) -> tuple[bool, str]:
    """Whether the player stands by the task's target, facing the right way to reach it
    (and talking to it when asked): the game's memory, against the survey."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from survey_rooms import Field

    ram = {k: int(v, 16) if isinstance(v, str) else v for k, v in box.game["ram"].items()}
    g = Field(box, ram)
    rooms = rooms_here()
    room = rooms.rooms.get(loc)
    if room is None:
        return False, "place not surveyed"
    hit = [p for p in room.people if p.name == task["target"]]
    at = hit[0].at if hit else room.spots.get(task["target"])
    if at is None:
        return False, f"no {task['target']!r}"
    person = bool(hit) and hit[0].kind != "save point"
    x, y = g.pos()
    ends = rooms.stand_by(room, at, person)
    from playertwo.rooms import centre

    ok = False
    for t, (_fx, fy) in ends.items():
        cx, cy = centre(t)
        off = abs(cx - x) if fy else abs(cy - y)
        if (int(y) // 16, int(x) // 16) == t and (off <= 6 or not person):
            ok = True
    why = f"at {x:.0f},{y:.0f}, by it" if ok else f"at {x:.0f},{y:.0f}, not by it"
    if task.get("talk"):  # bytes that may say a text box is up (to find which does)
        why += " " + " ".join(f"{a:x}={g.read(a, 1)[0]}" for a in (0x11F, 0x127, 0x134, 0x1EE,
                                                                  0x6E2))
    if ok and task.get("talk") and "dialog" in ram:
        talking = g.read(ram["dialog"], 1)[0] != ram.get("dialog_none", 0)
        ok = ok and talking
        why += ", talking" if talking else ", no text box"
    return ok, why


def trip(box: Box, task: dict, states: Path, game: str) -> dict:
    t0 = time.monotonic()
    box.load(states / f"{task['state']}.state")
    out = box.ob("run", "--pass", "PLAYER_TWO_KEYS", "--pass", "PLAYER_TWO_KEYMAP", "--",
                 PY, str(ROOT / "scripts" / "rooms_walk.py"), game, task["to"],
                 "--limit", str(task.get("walk_s", 90)),
                 *(["--exit", task["exit"]] if task.get("exit") else []),
                 *(["--target", task["target"]] if task.get("target") else []),
                 *(["--talk"] if task.get("talk") else []),
                 timeout=float(task.get("walk_s", 90)) + 90)
    time.sleep(1.5)  # a fade into the last place
    got = box.location()
    want = task["location"] if isinstance(task["location"], list) else [task["location"]]
    want = [int(w, 16) if isinstance(w, str) else w for w in want]
    at = None
    if task.get("near") and got in want:
        ram = {k: int(v, 16) if isinstance(v, str) else v for k, v in box.game["ram"].items()}
        at = Game(box, ram).pos()
        if math.dist(at, task["near"]) > 32:
            got = None
    if task.get("target") and got in want:
        ok, at = by_target(box, task, got)
        if not ok:
            got = None
    result = next((ln.split("RESULT", 1)[1].strip() for ln in out.splitlines()
                   if "RESULT" in ln), out.strip().splitlines()[-1] if out.strip() else "")
    return {"id": task["id"], "done": got in want, "location": None if got is None else hex(got),
            "seconds": round(time.monotonic() - t0, 1), "result": result[:200], "at": at,
            "log": out[-3000:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--boxes", default="11,12,13")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    spec = tomllib.loads(args.tasks.read_text())
    game = spec["game"]
    states = Path(game["states"]).expanduser()
    tasks = [t for t in spec.get("task", []) if t.get("to")
             and (not args.only or t["id"] in args.only)]
    out = args.out or Path.home() / ".cache/player-two/tasks/runs" / \
        f"rooms-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["PLAYER_TWO_KEYS"] = "hyprland"
    os.environ.setdefault("PLAYER_TWO_KEYMAP", KEYMAP)
    work: queue.Queue = queue.Queue()
    for n in range(args.repeat):
        for t in tasks:
            work.put((t, n + 1))
    lock = threading.Lock()
    results: list[dict] = []

    def worker(n: int) -> None:
        box = Box(n, game)
        box.up()  # after the survey: the box gets the rooms as they are now
        if os.environ.get("PLAYER_TWO_ROOMS"):  # rooms being tried: in place of the game's
            import shutil

            dest = box.home / ".config/player-two/atlas/chrono-trigger/rooms"
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(os.environ["PLAYER_TWO_ROOMS"], dest)
        try:
            while True:
                try:
                    t, k = work.get_nowait()
                except queue.Empty:
                    return
                try:
                    r = trip(box, t, states, game["name"])
                except (RuntimeError, OSError) as exc:
                    r = {"id": t["id"], "done": False, "error": str(exc)}
                r["run"] = k
                with lock:
                    results.append(r)
                    with out.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    print(f"{'PASS' if r['done'] else 'FAIL'} {t['id']:18} #{k} "
                          f"{r.get('seconds', '?'):>6} s  {r.get('location')}  "
                          f"{r.get('result', r.get('error', ''))[:110]}", flush=True)
        finally:
            box.down()

    threads = [threading.Thread(target=worker, args=(int(b),))
               for b in args.boxes.split(",")]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    by: dict[str, list[bool]] = {}
    for r in results:
        by.setdefault(r["id"], []).append(bool(r["done"]))
    print()
    for tid, oks in by.items():
        print(f"{tid:20} {sum(oks)}/{len(oks)}")
    print(f"\n{sum(1 for r in results if r['done'])} of {len(results)} done; {out}")


if __name__ == "__main__":
    main()
