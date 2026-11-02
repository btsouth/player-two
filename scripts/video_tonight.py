#!/usr/bin/env python3
"""One workstation command: transfer only mapped take files, render on devbox, copy proxy back."""

import argparse
import json
import shlex
import subprocess
import tempfile
import time
from pathlib import Path


def run(*args, **kw):
    return subprocess.run(args, check=True, **kw)


def main():
    started = time.monotonic()
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("folder", type=Path)
    a.add_argument("map", type=Path)
    a.add_argument("out", type=Path)
    args = a.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new output name; current cuts are never overwritten")
    board = Path.home() / ".cache/player-two/lead/BOARD.md"
    if board.exists():
        print(board.read_text()[-8000:], flush=True)
    mapping = json.loads(args.map.read_text())
    names = set(mapping.get("voiceovers", {}).values())
    for e in mapping.get("takes", {}).values():
        v = Path(e["video"])
        names.update(
            [
                str(v),
                e["start"],
                e["events"],
                str(v.with_suffix(".game.wav")),
                str(v.with_suffix(".voice.wav")),
            ]
        )
    for name in names:
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or not (args.folder / p).is_file():
            raise ValueError(f"Input must be an existing relative take file: {name}")
    remote = subprocess.check_output(
        ["ssh", "devbox-bts", "mktemp -d /workspace/tmp/p2-video/tonight-XXXXXX"], text=True
    ).strip()
    if not remote.startswith("/workspace/tmp/p2-video/tonight-"):
        raise ValueError("Unexpected remote staging path")
    root = "/workspace/tmp/p2-video"
    repo = Path(__file__).resolve().parents[1]
    mapping["template"] = mapping.get("devbox_template", root + "/d4-takes")
    with tempfile.TemporaryDirectory(prefix="p2-video-map-") as td:
        t = Path(td)
        (t / "map.json").write_text(json.dumps(mapping))
        (t / "files").write_text("\n".join(sorted(names)) + "\n")
        run("ssh", "devbox-bts", f"mkdir -p {shlex.quote(remote)}/incoming")
        run(
            "rsync",
            "-aL",
            "--files-from=" + str(t / "files"),
            str(args.folder.resolve()) + "/",
            f"devbox-bts:{remote}/incoming/",
        )
        run("rsync", "-a", str(t / "map.json"), f"devbox-bts:{remote}/map.json")
        run(
            "rsync",
            "-a",
            str(repo / "scripts/make_video.py"),
            str(repo / "scripts/video_dropin.py"),
            str(repo / "scripts/demo_captions.py"),
            f"devbox-bts:{root}/scripts/",
        )
        command = [
            "/workspace/repos/player-two/.venv/bin/python",
            root + "/scripts/video_dropin.py",
            remote + "/incoming",
            remote + "/map.json",
            remote + "/stage",
            "--out",
            remote + "/proxy.mp4",
        ]
        run("ssh", "devbox-bts", shlex.join(command))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        run("rsync", "-a", f"devbox-bts:{remote}/proxy.mp4", str(args.out))
        run(
            "rsync",
            "-a",
            f"devbox-bts:{remote}/proxy.sheet.png",
            str(args.out.with_suffix(".sheet.png")),
        )
        run(
            "rsync",
            "-a",
            f"devbox-bts:{remote}/pace-report.md",
            str(args.out.with_suffix(".pace-report.md")),
        )
    print(
        "Proxy:",
        args.out,
        "Remote evidence:",
        remote,
        "Elapsed seconds:",
        round(time.monotonic() - started, 2),
    )


if __name__ == "__main__":
    main()
