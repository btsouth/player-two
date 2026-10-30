"""Real tasks in a real game, scored: start from a known save, give Player Two one
request, and judge the screen it ends on.

    python scripts/play_tasks.py scripts/tasks/chrono_trigger.toml --saves DIR \
        [--only ID ...] [--repeat N] [--host "omabox host --"]

It launches the game, starts Player Two with a virtual controller, and drives both, so
the desktop must be free. Each task names a save set, a folder in --saves holding a copy
of the game's save files; the player's own saves are backed up first and put back at
the end, with the game closed. Results go to --out as JSON lines (default
~/.cache/player-two/tasks/runs/), and a summary is printed.

The judge is Muse Spark looking at the last screen against the task's check, plus what
Player Two said it did. Read the failures' screenshots before trusting a score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

from PIL import Image

from playertwo.config import Settings
from playertwo.muse import Muse, image_part

ROOT = Path(__file__).resolve().parent.parent
P2 = str(ROOT / ".venv" / "bin" / "player-two")

JUDGE = """You are checking whether an AI helper finished a task in a video game for a \
blind player. The task was: "{request}".
It counts as done if: {check}
The helper's last words to the player: "{said}"
Look at this last screenshot. Did it get done? Be strict: close isn't done. Answer JSON: \
{{"done": true or false, "why": "one short sentence"}}"""

READY = """Does this screenshot show: {ready}? Answer JSON: {{"yes": true or false, \
"why": "a few words"}}"""


def sh(host: list[str], *cmd: str, check: bool = False, timeout: float = 30) -> str:
    out = subprocess.run([*host, *cmd], capture_output=True, text=True, timeout=timeout,
                         check=check)
    return out.stdout


def ctl(*args: str, timeout: float = 60) -> str:
    out = subprocess.run([P2, "ctl", *args], check=False, capture_output=True, text=True,
                         timeout=timeout)
    return out.stdout.strip()


def events(log: Path, since: float) -> list[dict]:
    try:
        lines = log.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("t", 0) >= since:
            out.append(e)
    return out


def wait_for(pred, timeout: float, every: float = 1.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(every)
    return False


def save_dir(spec: str, app_id: str) -> Path:
    """The game's save folder. "{prefix}" is its Proton prefix, in whichever Steam library
    holds it, so task files don't name anyone's own disks."""
    from playertwo.games import steam_libraries

    if "{prefix}" in spec:
        for lib in steam_libraries():
            prefix = lib / "compatdata" / app_id / "pfx"
            if prefix.is_dir():
                return Path(spec.replace("{prefix}", str(prefix)))
        raise SystemExit(f"no Proton prefix for app {app_id} in any Steam library")
    return Path(spec).expanduser()


class Game:
    def __init__(self, cfg: dict, host: list[str]):
        self.name = cfg["name"]
        self.app_id = str(cfg["app_id"])
        self.saves = save_dir(cfg["save_dir"], self.app_id)
        self.window = cfg.get("window") or f"steam_app_{self.app_id}"
        self.process = cfg.get("process", "")
        self.settle = float(cfg.get("start_delay_s", 3))  # after the window opens
        self.host = host

    def running(self) -> bool:
        ps = subprocess.run(["ps", "-eo", "cmd"], check=False, capture_output=True,
                            text=True).stdout
        return any((self.process and self.process in line) or f"AppId={self.app_id}" in line
                   for line in ps.splitlines() if "play_tasks" not in line)

    def window_open(self) -> bool:
        try:
            clients = json.loads(sh(self.host, "hyprctl", "-j", "clients"))
        except (json.JSONDecodeError, subprocess.SubprocessError):
            return False
        return any(c.get("class") == self.window for c in clients)

    def start(self) -> None:
        sh(self.host, "sh", "-c",
           f"steam steam://rungameid/{self.app_id} >/dev/null 2>&1 &")
        if not wait_for(self.window_open, 120, 2):
            raise RuntimeError(f"{self.name} didn't open a window")
        time.sleep(self.settle)

    def stop(self) -> None:
        sh(self.host, "hyprctl", "dispatch",
           f'hl.dsp.window.close({{ window = "class:{self.window}" }})')
        if not wait_for(lambda: not self.running(), 150, 1):  # Proton can take a minute
            raise RuntimeError(f"{self.name} is still running")

    def put_saves(self, folder: Path | None) -> None:
        if self.running():
            raise RuntimeError("the game is running: saves not changed")
        if folder is None:
            return
        for f in folder.iterdir():
            shutil.copy2(f, self.saves / f.name)


class PlayerTwo:
    def __init__(self, host: list[str], game: str, log: Path):
        self.host, self.game, self.log = host, game, log

    def start(self) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        cmd = (f"setsid -f {shlex.quote(P2)} run --controls gamepad --companion off "
               f"--game {shlex.quote(self.game)} --log {shlex.quote(str(self.log))} "
               f">{shlex.quote(str(self.log.with_suffix('.out')))} 2>&1")
        sh(self.host, "sh", "-c", cmd)
        t = time.time() - 5
        if not wait_for(lambda: any(e["kind"] == "ready" for e in events(self.log, t)), 60):
            raise RuntimeError("Player Two didn't start")

    def stop(self) -> None:
        ctl("quit", timeout=10)
        wait_for(lambda: not subprocess.run(["pgrep", "-f", "[b]in/player-two run"],
                                            check=False, capture_output=True).stdout,
                 20, 0.5)

    def do(self, request: str, limit: float, answers: list[str] | None = None) -> dict | None:
        """Ask for something and wait for the agent to finish. Its questions get the next
        of `answers` ("yes" once they run out), as a player would answer them."""
        t = time.time()
        ctl("do", request)
        done = None
        answered = 0
        answers = list(answers or [])

        def finished() -> bool:
            nonlocal done, answered
            evs = events(self.log, t)
            runs = [e for e in evs if e["kind"] == "agent"]
            done = runs[-1] if runs else None
            asks = [e for e in evs if e["kind"] == "ask"]
            if done is None and len(asks) > answered:
                answered += 1
                ctl("do", answers.pop(0) if answers else "yes")
            return done is not None

        if not wait_for(finished, limit, 2):
            ctl("stop")
            wait_for(finished, 30, 1)
        return done


async def judge(muse: Muse, img: Image.Image, prompt: str) -> dict:
    msgs = [{"role": "user", "content": [image_part(img, 1280),
                                         {"type": "text", "text": prompt}]}]
    try:
        return await muse.json(msgs, effort="low", max_tokens=2000)
    except ValueError as exc:
        return {"error": str(exc)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tasks", type=Path)
    p.add_argument("--saves", type=Path, required=True,
                   help="a folder of save sets, one subfolder each")
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--host", default="", help='prefix for desktop commands, e.g. '
                   '"omabox host --"')
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    spec = tomllib.loads(args.tasks.read_text())
    host = shlex.split(args.host)
    game = Game(spec["game"], host)
    tasks = [t for t in spec["task"] if not args.only or t["id"] in args.only]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or Path.home() / ".cache/player-two/tasks/runs" / f"{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    shots = out.with_suffix("")
    shots.mkdir(exist_ok=True)
    muse = Muse(Settings.load())

    if game.running():
        sys.exit(f"{game.name} is running: close it first (its saves get backed up)")
    backup = Path(tempfile.mkdtemp(prefix="p2-saves-"))
    shutil.copytree(game.saves, backup, dirs_exist_ok=True)
    print(f"saves backed up to {backup}")
    results = []
    try:
        for n in range(args.repeat):
            for task in tasks:
                tid = f"{task['id']}#{n + 1}"
                print(f"--- {tid}: {task['request']}")
                r = run_task(task, tid, game, muse, shots, args.saves)
                results.append(r)
                with out.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"    {'PASS' if r.get('done') else 'FAIL'} "
                      f"{r.get('seconds', '?')} s: {r.get('why', r.get('error', ''))}")
    finally:
        if game.running():
            game.stop()
        for f in backup.iterdir():
            shutil.copy2(f, game.saves / f.name)
            (game.saves / f.name).touch()  # so the cloud keeps these, not the tests'
        print(f"saves restored from {backup}")
    passed = sum(1 for r in results if r.get("done"))
    print(f"\n{passed} of {len(results)} done; results in {out}")


def run_task(task: dict, tid: str, game: Game, muse: Muse, shots: Path,
             saves: Path) -> dict:
    r: dict = {"id": task["id"], "run": tid, "request": task["request"]}
    p2 = PlayerTwo(game.host, game.name, shots / f"{tid}.events.jsonl")
    try:
        save = task.get("save")
        game.put_saves(saves / save if save else None)
        game.start()
        p2.start()
        if task.get("setup"):
            ctl("press", task["setup"], timeout=120)
        shot = shots / f"{tid}.start.png"
        ctl("shot", str(shot))
        if task.get("ready"):
            ok = asyncio.run(judge(muse, Image.open(shot),
                                   READY.format(ready=task["ready"])))
            if not ok.get("yes"):
                r.update(done=False, error=f"setup didn't reach the start: {ok}")
                return r
        t = time.monotonic()
        # A task is one request, or a few in a row (each with its own time limit).
        requests = task["request"] if isinstance(task["request"], list) else [task["request"]]
        run = None
        for request in requests:
            run = p2.do(request, float(task.get("limit_s", 300)), task.get("answers"))
        r["seconds"] = round(time.monotonic() - t, 1)
        shot = shots / f"{tid}.end.png"
        ctl("shot", str(shot))
        said = (run or {}).get("outcome", "")
        r.update(outcome=said, steps=len((run or {}).get("steps", [])),
                 status=ctl("status"))
        verdict = asyncio.run(judge(muse, Image.open(shot), JUDGE.format(
            request=" Then: ".join(requests), check=task["check"], said=said)))
        r.update(done=bool(verdict.get("done")), why=verdict.get("why", verdict))
    except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
        r.update(done=False, error=str(exc))
    finally:
        p2.stop()
        if game.running():
            game.stop()
    return r


if __name__ == "__main__":
    main()
