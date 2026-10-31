"""Chrono Trigger (SNES) save states for every era, made from battery saves (.srm).

A save slot keeps the party, the place (a location or world map ID) and the position.
This loads a slot's .srm into a hidden omabox desktop's RetroArch, continues it from the
title screen, sets Battle Mode to Wait and saves a RetroArch state with a screenshot.

    python scripts/ct_states.py up -n 20                # box emu-20 with the game
    python scripts/ct_states.py slots SRM ...           # what each slot holds
    python scripts/ct_states.py heal SRM OUT            # full HP/MP in every slot
    python scripts/ct_states.py boot -n 20 SRM          # start the game with that .srm
    python scripts/ct_states.py keys -n 20 down a ...   # press game buttons
    python scripts/ct_states.py shot -n 20 OUT.png      # screenshot (+ locate on a map)
    python scripts/ct_states.py save -n 20 NAME [...]   # save a state + .png, index it
    python scripts/ct_states.py make -n 20 RECIPE ...   # a recipe below, end to end
    python scripts/ct_states.py where -n 20             # location ID now ($7E0100)
    python scripts/ct_states.py finish -n 20 NAME --at 0x1F5 --meta JSON
                                                        # Wait, save, index (by hand)
    python scripts/ct_states.py verify -n 20 [NAME ...] # reload each, check, reshoot
    python scripts/ct_states.py down -n 20

States, their .png and index.json go to ~/.cache/player-two/emu/states/ct-snes. In the
field X opens the menu; Config is its 4th icon, Battle Mode its 2nd line. On the world
map the party enters a place on its door tile with A; in the Epoch Y opens the time gauge.

SRAM layout (datacrystal.tcrf.net/wiki/Chrono_Trigger): three 0xA00-byte slots at 0,
0xA00 and 0x1400; per slot 0x200 characters (0x50 each: +3 HP, +5 max HP, +7 MP, +9 max
MP, +0x12 level), 0x580 party, 0x5F3 location (0x1F0-0x1F7 are world maps), 0x5F5 x, y.
0x1FE0 is the slot saved last, where the title screen's cursor starts.
The checksum at 0x1FF0 + 2*slot is a 16-bit add-with-carry of the slot's words, last
word first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emu_tasks import EMU, ROOT, Box, wait_for

from playertwo.atlas import RISE

STATES = EMU / "states/ct-snes"
INDEX = STATES / "index.json"
QUIET = EMU / "ct-states.cfg"  # no "Saved state" notices on the screenshots
SLOT = 0xA00
NAMES = ["Crono", "Marle", "Lucca", "Robo", "Frog", "Ayla", "Magus"]
WORLDS = {0x1F0: "1000 AD", 0x1F1: "600 AD", 0x1F2: "2300 AD", 0x1F3: "65,000,000 BC",
          0x1F4: "12,000 BC", 0x1F5: "12,000 BC (Zeal)", 0x1F6: "12,000 BC (after the fall)",
          0x1F7: "1999 AD"}
# The game's buttons as base.cfg binds them, as Hyprland key names.
BUTTONS = {"a": "Return", "b": "Escape", "x": "BackSpace", "y": "Tab", "l": "q", "r": "e",
           "start": "F5", "select": "F6", "up": "Up", "down": "Down", "left": "Left",
           "right": "Right"}


def game() -> dict:
    return tomllib.loads((ROOT / "scripts/tasks/ct_snes.toml").read_text())["game"]


# --- battery saves -------------------------------------------------------------------

def checksum(slot: bytes) -> int:
    a = c = 0  # from the last word back to the first: the carry order matters
    for i in range(len(slot) - 2, -1, -2):
        a += int.from_bytes(slot[i:i + 2], "little") + c
        c, a = a >> 16, a & 0xFFFF
    return a


def slot_info(srm: bytes, s: int) -> dict:
    sl = srm[s * SLOT:(s + 1) * SLOT]
    w = lambda o: int.from_bytes(sl[o:o + 2], "little")
    chars = {}
    for i, name in enumerate(NAMES):
        o = 0x200 + i * 0x50
        chars[name] = {"level": sl[o + 0x12], "hp": [w(o + 3), w(o + 5)],
                       "mp": [w(o + 7), w(o + 9)]}
    party = [NAMES[x] for x in sl[0x580:0x583] if x < len(NAMES)]
    loc = w(0x5F3)
    return {"slot": s + 1, "location": loc, "world": WORLDS.get(loc),
            "x": sl[0x5F5], "y": sl[0x5F6], "party": party,
            "levels": {n: chars[n]["level"] for n in party},
            "hp": {n: chars[n]["hp"] for n in party},
            "checksum_ok": checksum(sl) == int.from_bytes(srm[0x1FF0 + 2 * s:0x1FF2 + 2 * s],
                                                          "little")}


def heal(srm: bytes) -> bytes:
    """Every character at full HP and MP, in every used slot, checksums redone."""
    b = bytearray(srm)
    for s in range(3):
        o = s * SLOT
        if not any(b[o:o + SLOT]):
            continue
        for i in range(len(NAMES)):
            c = o + 0x200 + i * 0x50
            b[c + 3:c + 5] = b[c + 5:c + 7]
            b[c + 7:c + 9] = b[c + 9:c + 11]
        b[0x1FF0 + 2 * s:0x1FF2 + 2 * s] = checksum(bytes(b[o:o + SLOT])).to_bytes(2, "little")
    return bytes(b)


# --- the box -------------------------------------------------------------------------

class CT(Box):
    def srm_path(self) -> Path:
        return self.home / ".config/retroarch/saves" / self.game["core_name"] / \
            f"{self.rom.stem}.srm"

    def state_path(self) -> Path:
        return self.home / ".config/retroarch/states" / self.game["core_name"] / \
            f"{self.rom.stem}.state"

    def attach(self) -> None:
        self.home = Path(self.ob("path", "--home").strip().splitlines()[-1])

    def boot(self, srm: Path | bytes) -> None:
        """Restart RetroArch with this battery save, at the title screen."""
        if not self.home.name:
            self.attach()
        if "PLAYING" in self.cmd("GET_STATUS"):
            self.cmd("QUIT", wait=0.5)
            self.ob("run", "--", "sh", "-c", "pkill -x retroarch; true")
            wait_for(lambda: not self.cmd("GET_STATUS", wait=0.5), 15)
        QUIET.write_text('menu_enable_widgets = "false"\nvideo_font_enable = "false"\n')
        p = self.srm_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(srm if isinstance(srm, bytes) else Path(srm).read_bytes())
        for _ in range(3):
            self.ob("run", "-d", "--wait", "--", "retroarch", "-L", self.game["core"],
                    f"--appendconfig={EMU / 'base.cfg'}|{EMU / f'port-{self.port}.cfg'}"
                    f"|{QUIET}",
                    str(self.rom), timeout=60)
            if wait_for(lambda: "PLAYING" in self.cmd("GET_STATUS"), 20):
                return
        raise RuntimeError(f"{self.name}: RetroArch isn't answering")

    def key(self, button: str, hold: float = 0.08, gap: float = 0.25) -> None:
        k = BUTTONS.get(button, button)
        for st in ("down", "up"):
            self.ob("hyprctl", "dispatch",
                    f'hl.dsp.send_key_state({{ mods = "", key = "{k}", state = "{st}" }})')
            if st == "down":
                time.sleep(hold)
        time.sleep(gap)

    def keys(self, seq: list[str]) -> None:
        """Buttons in order; "w1.5" waits 1.5 s, "right*3" presses three times,
        "up:2" holds up for 2 s."""
        for item in seq:
            if item.startswith("w") and item[1:].replace(".", "").isdigit():
                time.sleep(float(item[1:]))
                continue
            n = 1
            if "*" in item:
                item, cnt = item.split("*")
                n = int(cnt)
            hold = 0.08
            if ":" in item:
                item, h = item.split(":")
                hold = float(h)
            for _ in range(n):
                self.key(item, hold)

    def ram(self, off: int, n: int) -> bytes:
        out = self.cmd(f"READ_CORE_RAM {off:x} {n}")
        parts = out.split()
        if len(parts) < 3 or parts[2] == "-1":
            return b""
        return bytes(int(x, 16) for x in parts[2:])

    def where(self) -> int | None:
        """The location ID ($7E0100) the game is in now."""
        b = self.ram(0x100, 2)
        return int.from_bytes(b, "little") if len(b) == 2 else None

    def save(self, dest: Path) -> Path:
        self.state_path().unlink(missing_ok=True)
        self.cmd("SAVE_STATE", wait=0.5)
        wait_for(lambda: self.state_path().exists() and self.state_path().stat().st_size, 10)
        time.sleep(1)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.state_path(), dest)
        return dest


_ATLAS = None


def atlas():
    """The game's atlas with every map rip in its folder, not only those atlas.json
    lists (the other eras' rips aren't labelled yet)."""
    global _ATLAS
    if _ATLAS is None:
        import cv2

        from playertwo.atlas import ATLASES, Atlas, WorldMap
        _ATLAS = Atlas.for_game("Chrono Trigger")
        folder = ATLASES / "chrono-trigger"
        have = {m["image"] for m in json.loads((folder / "atlas.json").read_text())["maps"]}
        for png in sorted(folder.glob("w*.png")):
            if png.name not in have:
                img = cv2.imread(str(png), cv2.IMREAD_COLOR)
                if img is not None:
                    _ATLAS.maps.append(WorldMap(png.stem, img, {}))
    return _ATLAS


def locate(png: Path) -> dict | None:
    """Atlas.locate's spot, or (when it finds none) the best match anyway, marked."""
    import cv2
    from PIL import Image

    a = atlas()
    s = a.locate(Image.open(png))
    if s is not None:
        return {"map": s.map, "x": round(s.x), "y": round(s.y),
                "score": round(s.score, 3), "margin": round(s.margin, 3)}
    v = a.view(Image.open(png))
    nw, nh = a.native
    t = v[16:nh - 24, 16:nw - 16]
    best = None
    for m in a.maps:
        _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(m.image, t, cv2.TM_CCOEFF_NORMED))
        if best is None or score > best["score"]:
            best = {"map": m.name, "x": round(x - 16 + nw / 2), "y": round(y - 16 + nh / 2),
                    "score": round(float(score), 3)}
    return {**best, "below_threshold": True} if best else None


def index_add(entry: dict) -> None:
    import fcntl
    with open(INDEX.with_suffix(".lock"), "w") as lock:  # two boxes may finish at once
        fcntl.flock(lock, fcntl.LOCK_EX)
        idx = json.loads(INDEX.read_text()) if INDEX.exists() else {"states": []}
        idx["states"] = [e for e in idx["states"] if e["file"] != entry["file"]] + [entry]
        idx["states"].sort(key=lambda e: e["file"])
        INDEX.write_text(json.dumps(idx, indent=2) + "\n")


# --- recipes -------------------------------------------------------------------------
# Continue a slot from the title screen (its list: New Game, then slots 1 to 3), check
# the place it loaded by $7E0100, set Battle Mode to Wait and save. `then` is more
# buttons (see CT.keys) to play on from the slot, `at` the location ID to end on.

SAVES = Path.home() / ".cache/player-two/research/saves/fantasyanime/srm"
SRC = "fantasyanime.com/squaresoft/ctsaves-srm.htm"
RECIPES: dict[str, dict] = {
    "world-600": {"srm": "ctsave04-masamune", "slot": 1, "at": 0x1F1,
                      "near": "Zenan Bridge, north end (Guardia side is south)"},
    "world-2300": {"srm": "ctsave03-future_more", "slot": 2, "at": 0x1F2,
                       "near": "outside the Factory (Robo just joined)"},
    "world-2300-bangor": {"srm": "ctsave02-future", "slot": 2, "at": 0x1F2,
                              "near": "outside Bangor Dome, just arrived in 2300 AD"},
    "world-65m": {"srm": "ctsave05-prehistoric", "slot": 1, "at": 0x1F3,
                      "near": "outside the Forest Maze"},
    "world-12000": {"srm": "ctsave08-zeal", "slot": 2, "at": 0x1F4,
                        "near": "by the Skyway up to Zeal, just arrived in 12000 BC"},
    "world-1000-epoch": {"srm": "ctsave14-side_quests3", "slot": 1, "at": 0x1F0, "epoch": True,
                             "near": "outside Lucca's house (Sun Stone side quest done)"},
    "world-600-epoch": {"srm": "ctsave14-side_quests3", "slot": 2, "at": 0x1F1, "epoch": True,
                            "near": "outside Hero's Grave (Frog's side quest done)"},
    "world-2300-epoch": {"srm": "ctsave14-side_quests3", "slot": 3, "at": 0x1F2, "epoch": True,
                             "near": "outside Geno Dome (Robo's side quest done)"},
    "room-end-of-time": {"srm": "ctsave03-future_more", "slot": 3, "at": 0x1D0,
                             "era": "End of Time", "near": "End of Time, by the save point"},
    "room-arris-dome": {"srm": "ctsave02-future", "slot": 3, "at": 0x0D6, "era": "2300 AD",
                            "near": "Arris Dome, food storage down the ladder"},
    "room-magus-castle": {"srm": "ctsave06-magus_castle", "slot": 3, "at": 0x1D7,
                              "era": "600 AD", "near": "inside Magus's Castle, halfway"},
    "room-earthbound": {"srm": "ctsave11-crono_gone", "slot": 2, "at": 0x1A9,
                            "era": "12,000 BC", "near": "Earthbound Village (after Zeal fell)"},
    "room-ocean-palace": {"srm": "ctsave10-ocean_palace", "slot": 1, "at": 0x194,
                              "era": "12,000 BC", "near": "Ocean Palace entrance"},
}
WAIT_REF = SAVES.parent.parent / "config-wait-full.png"  # its label reads Wait
LABEL = (240 + 300, 860 + 40, 240 + 900, 860 + 130)  # "Battle Mode  Wait" on the screen


def label_diff(png: Path) -> float:
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(png).convert("L").crop(LABEL), dtype=float)
    b = np.asarray(Image.open(WAIT_REF).convert("L").crop(LABEL), dtype=float)
    return float(np.abs(a - b).mean())


def set_wait(box: CT, tmp: Path) -> bool:
    """Menu (X), Config (4th icon), Battle Mode (2nd line): Wait if it isn't, out again."""
    box.keys(["x", "w1.2", "right*3", "a", "w1.2", "down", "w0.6"])
    ok = False
    for _ in range(2):
        box.shot(tmp)
        d = label_diff(tmp)
        print(f"  battle mode label diff {d:.1f}", flush=True)
        if d < 6:
            ok = True
            break
        box.keys(["a", "w0.6"])
    box.keys(["b", "w0.8", "b", "w1.5"])
    return ok


def make(box: CT, name: str, r: dict) -> dict:
    srm = (SAVES / f"{r['srm']}.srm").read_bytes()
    info = slot_info(srm, r["slot"] - 1)
    b = bytearray(heal(srm))
    b[0x1FE0] = r["slot"] - 1  # the last slot saved: the title's cursor starts on it
    box.boot(bytes(b))
    box.keys(["w7", "a", "w5", "a", "w5"])
    loc = box.where()
    print(f"{name}: loaded at {loc and hex(loc)}", flush=True)
    if r.get("then"):
        box.keys(r["then"])
        loc = box.where()
    if loc != r["at"]:
        raise RuntimeError(f"{name}: at {loc and hex(loc)}, wanted {r['at']:#x}")
    return finish(box, name, r["at"], {
        "era": r.get("era") or WORLDS.get(r["at"]), "near": r["near"],
        "party": {n: info["levels"][n] for n in info["party"]},
        "party_hp": "full (healed in the .srm)",
        "epoch": r.get("epoch", "no" if r["at"] in WORLDS else None),
        "source": f"{SRC} {r['srm']}.zip slot {r['slot']}" + (
            f", then {' '.join(r['then'])}" if r.get("then") else ""),
        "made_with": "scripts/ct_states.py make " + name})


def finish(box: CT, name: str, at: int, meta: dict) -> dict:
    """Where the game is now: check the place, Battle Mode to Wait, save and index."""
    if box.where() != at:
        raise RuntimeError(f"{name}: at {box.where()}, wanted {at:#x}")
    tmp = STATES.parent.parent / f"tmp-{box.name}.png"
    wait = set_wait(box, tmp)
    tmp.unlink(missing_ok=True)
    if box.where() != at:
        raise RuntimeError(f"{name}: left {at:#x} while setting Wait")
    dest = box.save(STATES / f"{name}.state")
    png = STATES / f"{name}.png"
    box.shot(png)
    entry = {"file": dest.name, "png": png.name, "era": meta.pop("era", WORLDS.get(at)),
             "kind": "world map" if at in WORLDS else "interior", "location_id": hex(at),
             **meta, "battle_mode_wait": wait, "locate": locate(png)}
    index_add(entry)
    return entry


def verify(box: CT, names: list[str]) -> None:
    """Load each indexed state, check where it is, and retake its screenshot."""
    idx = json.loads(INDEX.read_text())
    for e in idx["states"]:
        name = e["file"].removesuffix(".state")
        if names and name not in names:
            continue
        box.load(STATES / e["file"])
        time.sleep(2)
        loc = box.where()
        if e.get("location_id") is None and loc is not None:  # added by hand
            e["location_id"] = hex(loc)
        png = STATES / e["png"]
        if png.exists() and "ct_states" not in e.get("made_with", ""):
            png = STATES.parent.parent / f"tmp-{box.name}.png"  # not ours: leave it be
        box.shot(png)
        e["locate"] = locate(png)
        e.pop("nearest_place", None)
        if e["locate"] and not e["locate"].get("below_threshold"):
            m = atlas().map(e["locate"]["map"])
            if m.places:  # the player stands a little above the view's middle
                place, dist, way = m.nearest(e["locate"]["x"], e["locate"]["y"] - RISE)
                e["nearest_place"] = {"name": place, "map_px": round(dist), "way": way}
        e["verified"] = {"loads": loc is not None and hex(loc) == e["location_id"],
                         "location_id": loc and hex(loc), "at": time.strftime("%F %T")}
        print(name, e["verified"], e["locate"], flush=True)
        index_add(e)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slots")
    s.add_argument("srm", nargs="+", type=Path)
    s = sub.add_parser("heal")
    s.add_argument("srm", type=Path)
    s.add_argument("out", type=Path)
    for name in ("up", "boot", "keys", "shot", "save", "where", "down", "make", "verify",
                 "finish"):
        s = sub.add_parser(name)
        s.add_argument("-n", type=int, default=20)
        if name == "boot":
            s.add_argument("srm", type=Path)
            s.add_argument("--heal", action="store_true")
        if name == "keys":
            s.add_argument("seq", nargs="+")
        if name == "finish":
            s.add_argument("name")
            s.add_argument("--at", required=True, type=lambda v: int(v, 0))
            s.add_argument("--meta", default="{}", help="JSON for its index entry")
        if name == "verify":
            s.add_argument("names", nargs="*")
        if name == "make":
            s.add_argument("recipes", nargs="+", choices=sorted(RECIPES))
        if name == "shot":
            s.add_argument("out", type=Path)
        if name == "save":
            s.add_argument("name")
            s.add_argument("--meta", default="{}", help="JSON for its index entry")
    a = p.parse_args()

    if a.cmd == "slots":
        for f in a.srm:
            b = f.read_bytes()
            for i in range(3):
                print(f.name, json.dumps(slot_info(b, i)))
        return
    if a.cmd == "heal":
        a.out.write_bytes(heal(a.srm.read_bytes()))
        return

    box = CT(a.n, game())
    if a.cmd == "up":
        box.up()
        print(box.home)
        return
    if a.cmd == "down":
        box.down()
        return
    box.attach()
    if a.cmd == "boot":
        b = a.srm.read_bytes()
        box.boot(heal(b) if a.heal else b)
    elif a.cmd == "make":
        failed = []
        for name in a.recipes:
            try:
                print(json.dumps(make(box, name, RECIPES[name]), indent=1), flush=True)
            except RuntimeError as exc:
                print(f"FAILED {exc}", flush=True)
                failed.append(name)
        if failed:
            sys.exit(f"failed: {failed}")
    elif a.cmd == "finish":
        print(json.dumps(finish(box, a.name, a.at, json.loads(a.meta)), indent=1))
    elif a.cmd == "verify":
        verify(box, a.names)
    elif a.cmd == "keys":
        box.keys(a.seq)
    elif a.cmd == "where":
        loc = box.where()
        print(hex(loc) if loc is not None else None, WORLDS.get(loc or -1, ""))
    elif a.cmd == "shot":
        box.shot(a.out)
        print(a.out, locate(a.out))
    elif a.cmd == "save":
        dest = box.save(STATES / f"{a.name}.state")
        png = STATES / f"{a.name}.png"
        box.shot(png)
        loc = box.where()
        entry = {"file": dest.name, "png": png.name,
                 "location_id": hex(loc) if loc is not None else None,
                 "locate": locate(png), **json.loads(a.meta)}
        index_add(entry)
        print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
