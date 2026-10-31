"""Tasks in an emulated game, several at once, each in its own hidden omabox desktop: load
a save state, give Player Two one request (or a few), and judge the screen it ends on.

    python scripts/emu_tasks.py scripts/tasks/ct_snes.toml [--boxes 4] [--repeat 3]
        [--only ID ...] [--out FILE]

Nothing touches the real desktop. Each box runs RetroArch with the game, its own network
command port (for loading states and reading memory) and its own Player Two, which sends
keys through the box's compositor (RetroArch reads key codes, not key names). Results go
to --out as JSON lines with each run's events and last screen beside it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contextlib

from PIL import Image
from play_tasks import JUDGE, judge

from playertwo.config import Settings, api_key
from playertwo.muse import Muse

ROOT = Path(__file__).resolve().parent.parent
P2 = str(ROOT / ".venv" / "bin" / "player-two")
EMU = Path.home() / ".cache/player-two/emu"
BASE_PORT = 55410
# The game's keys in base.cfg: a Enter, b Escape, x BackSpace, y Tab, l q, r e, start F5,
# select F6, the arrows. Player Two's own map has start and back on keys taken there.
KEYMAP = "start=F5,back=F6"


def sh(*cmd: str, timeout: float = 60, check: bool = False) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)
    return out.stdout + out.stderr


class Box:
    """One hidden desktop with the game and a Player Two."""

    def __init__(self, n: int, game: dict):
        self.name, self.port, self.game = f"emu-{n}", BASE_PORT + n, game
        self.rom = Path(game["rom"]).expanduser()
        self.home = Path()

    def ob(self, *args: str, timeout: float = 60) -> str:
        return sh("omabox", "-b", self.name, *args, timeout=timeout)

    def cmd(self, c: str, wait: float = 2.0) -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(wait)
        try:
            s.sendto(c.encode(), ("127.0.0.1", self.port))
            return s.recv(65536).decode().strip()
        except OSError:
            return ""
        finally:
            s.close()

    def up(self) -> None:
        sh("omabox", "down", "-b", self.name, timeout=60)
        (EMU / f"port-{self.port}.cfg").write_text(f'network_cmd_port = "{self.port}"\n')
        out = sh("omabox", "up", self.name, "--ro-bind", str(self.rom.parent),
                 "--ro-bind", str(EMU), "--ro-bind", str(ROOT), timeout=180)
        if f"box '{self.name}' up" not in out:
            raise RuntimeError(f"{self.name} didn't come up: {out[-300:]}")
        self.home = Path(sh("omabox", "path", "--home", "-b", self.name).strip().splitlines()[-1])
        # Its first start in a new HOME sometimes crashes while it writes its config:
        # the next one doesn't.
        for _ in range(3):
            self.ob("run", "-d", "--wait", "--", "retroarch", "-L", self.game["core"],
                    f"--appendconfig={EMU / 'base.cfg'}|{EMU / f'port-{self.port}.cfg'}",
                    str(self.rom), timeout=60)
            if wait_for(lambda: "PLAYING" in self.cmd("GET_STATUS"), 20):
                return
        raise RuntimeError(f"{self.name}: RetroArch isn't answering on {self.port}")

    def load(self, state: Path) -> None:
        d = self.home / ".config/retroarch/states" / self.game["core_name"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(state, d / f"{self.rom.stem}.state")
        self.cmd("LOAD_STATE", wait=0.5)
        time.sleep(2)

    @property
    def log(self) -> Path:
        return self.home / "p2.jsonl"

    def p2(self, *args: str, timeout: float = 30) -> str:
        return self.ob("run", "--", P2, "ctl", *args, timeout=timeout)

    def start_p2(self) -> None:
        self.ob("run", "--", "sh", "-c", "pkill -f '[b]in/player-two run'; true")
        self.log.unlink(missing_ok=True)
        cmd = (f"exec {shlex.quote(P2)} run --controls keyboard --window RetroArch "
               f"--companion off --game {shlex.quote(self.game['name'])} --log $HOME/p2.jsonl")
        self.ob("run", "-d", "--pass", "MODEL_API_KEY", "--pass", "PLAYER_TWO_KEYMAP",
                "--pass", "PLAYER_TWO_KEYS", "--", "sh", "-c", cmd)
        if not wait_for(lambda: any(e["kind"] == "ready" for e in self.events()), 60):
            raise RuntimeError(f"{self.name}: Player Two didn't start")

    def events(self) -> list[dict]:
        try:
            lines = self.log.read_text(errors="replace").splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            with contextlib.suppress(json.JSONDecodeError):
                out.append(json.loads(line))
        return out

    def shot(self, dest: Path) -> Path | None:
        out = self.ob("shot").strip().splitlines()  # RetroArch fills the screen
        if out and Path(out[-1]).exists():
            shutil.copy2(out[-1], dest)
            return dest
        return None

    def down(self) -> None:
        sh("omabox", "down", "-b", self.name, timeout=60)


def wait_for(pred, timeout: float, every: float = 1.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(every)
    return pred()


def do(box: Box, request: str, limit: float, answers: list[str]) -> dict | None:
    """One request: answer its questions in order ("yes" once out), up to the limit."""
    seen = len(box.events())
    box.p2("do", request)
    answered = 0
    end = time.monotonic() + limit
    while time.monotonic() < end:
        evs = box.events()[seen:]
        done = next((e for e in evs if e["kind"] == "agent"), None)
        if done:
            return done
        asks = [e for e in evs if e["kind"] == "ask"]
        if len(asks) > answered:
            answered += 1
            box.p2("do", answers.pop(0) if answers else "yes")
        time.sleep(1)
    box.p2("stop")
    wait_for(lambda: any(e["kind"] == "agent" for e in box.events()[seen:]), 30)
    return None


def run_task(box: Box, task: dict, tid: str, states: Path, shots: Path,
             runner: asyncio.Runner, muse: Muse) -> dict:
    r: dict = {"id": task["id"], "run": tid, "box": box.name,
               "category": task.get("category", ""), "request": task["request"]}
    try:
        box.load(states / f"{task['state']}.state")
        box.start_p2()
        t = time.monotonic()
        requests = task["request"] if isinstance(task["request"], list) else [task["request"]]
        answers = list(task.get("answers", []))
        run = None
        for req in requests:
            run = do(box, req, float(task.get("limit_s", 300)), answers)
        r["seconds"] = round(time.monotonic() - t, 1)
        r.update(outcome=(run or {}).get("outcome", "timed out"),
                 steps=len((run or {}).get("steps", [])))
        shot = box.shot(shots / f"{tid}.end.png")
        shutil.copy2(box.log, shots / f"{tid}.events.jsonl")
        if task.get("expect"):  # a question with a known answer: in its words
            said = r["outcome"].lower()
            missing = [w for w in task["expect"] if w.lower() not in said]
            r.update(done=not missing, why=f"missing {missing}" if missing else "right answer")
        elif shot:
            v = runner.run(judge(muse, Image.open(shot), JUDGE.format(
                request=" Then: ".join(requests), check=task["check"], said=r["outcome"])))
            r.update(done=bool(v.get("done")), why=v.get("why", v), seen=v.get("seen", ""))
        else:
            r.update(done=False, error="no screenshot")
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        r.update(done=False, error=str(exc))
    finally:
        box.p2("quit", timeout=10)
    return r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tasks", type=Path)
    p.add_argument("--boxes", type=int, default=4)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    spec = tomllib.loads(args.tasks.read_text())
    game = spec["game"]
    states = Path(game["states"]).expanduser()
    tasks = [t for t in spec["task"] if not args.only or t["id"] in args.only]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or Path.home() / ".cache/player-two/tasks/runs" / f"emu-{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    shots = out.with_suffix("")
    shots.mkdir(exist_ok=True)
    os.environ["MODEL_API_KEY"] = api_key()  # passed into each box, never printed
    os.environ.setdefault("PLAYER_TWO_KEYMAP", KEYMAP)
    os.environ["PLAYER_TWO_KEYS"] = "hyprland"

    work: queue.Queue = queue.Queue()
    for n in range(args.repeat):
        for task in tasks:
            work.put((task, f"{task['id']}#{n + 1}"))
    lock, results = threading.Lock(), []

    def worker(n: int) -> None:
        box = Box(n, game)
        runner = asyncio.Runner()  # one loop per worker: the judge's client lives on it
        muse = Muse(Settings.load())
        try:
            box.up()
        except (RuntimeError, subprocess.SubprocessError) as exc:
            print(f"{box.name}: {exc}", flush=True)
            return
        try:
            while True:
                try:
                    task, tid = work.get_nowait()
                except queue.Empty:
                    return
                r = run_task(box, task, tid, states, shots, runner, muse)
                with lock:
                    results.append(r)
                    with out.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    print(f"{'PASS' if r.get('done') else 'FAIL'} {tid:24} "
                          f"{r.get('seconds', '?'):>6} s  {r.get('why', r.get('error', ''))}",
                          flush=True)
        finally:
            box.down()
            runner.close()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(args.boxes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    by: dict[str, list[bool]] = {}
    for r in results:
        by.setdefault(r["id"], []).append(bool(r.get("done")))
    print()
    for tid, oks in by.items():
        print(f"{tid:20} {sum(oks)}/{len(oks)}")
    passed = sum(1 for r in results if r.get("done"))
    print(f"\n{passed} of {len(results)} done; results in {out}")


if __name__ == "__main__":
    main()
