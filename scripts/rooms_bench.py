"""Trips out of places by the surveyed rooms, no model, judged by the game's location
number: each trip loads a state in a hidden box, runs scripts/rooms_walk.py there, then
reads where the game is.

    python scripts/rooms_bench.py scripts/tasks/ct_rooms.toml [--boxes 11,12,13]
        [--repeat 3] [--only ID ...] [--out FILE]

Trips are the task file's [[task]]s with a `to` (the location number to walk to, hex):
the same tasks scripts/emu_tasks.py gives the full agent by their `request`. `exit` names
the way out to take (rooms.py's label for it), and `near` = [x, y] says where on a world
map the trip must come out (within 32 pixels), for places with more than one way out.
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


def trip(box: Box, task: dict, states: Path, game: str) -> dict:
    t0 = time.monotonic()
    box.load(states / f"{task['state']}.state")
    out = box.ob("run", "--pass", "PLAYER_TWO_KEYS", "--pass", "PLAYER_TWO_KEYMAP", "--",
                 PY, str(ROOT / "scripts" / "rooms_walk.py"), game, task["to"],
                 "--limit", str(task.get("walk_s", 90)),
                 *(["--exit", task["exit"]] if task.get("exit") else []),
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
