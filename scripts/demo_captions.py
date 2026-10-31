#!/usr/bin/env python3
"""Captions for a demo recording, from Player Two's event log.

The player's words ("heard") and everything Player Two says ("say") become subtitles timed
to the video, so the recording reads with the sound off too.

    player-two run --log demo.events.jsonl      # while recording with scripts/demo_record.sh
    python scripts/demo_captions.py demo.events.jsonl --video demo.mp4 --burn demo-captioned.mp4

The video's start time comes from <video>.start (written by demo_record.sh) or --start.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

CHARS_PER_S = 14.0  # Player Two's voice at speech rate 1.1, roughly


def _stamp(s: float) -> str:
    s = max(0.0, s)
    h, rem = divmod(int(s * 1000), 3_600_000)
    m, rem = divmod(rem, 60_000)
    sec, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _chunks(text: str, limit: int = 90) -> list[str]:
    """Sentences, with long ones split at a comma near the middle."""
    out: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        while len(sent) > limit:
            cut = sent.rfind(", ", 0, limit)
            cut = cut + 1 if cut > limit // 3 else sent.rfind(" ", 0, limit)
            if cut <= 0:
                break
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            out.append(sent)
    return out


def captions(events: list[dict], start: float, narrator: bool = True) -> list[tuple[float, float, str]]:
    """(from, to, text) in seconds from the video's start."""
    out: list[tuple[float, float, str]] = []
    statuses = [(e["t"], e.get("state")) for e in events if e.get("kind") == "status"]
    for e in events:
        kind, text = e.get("kind"), str(e.get("text") or "").strip()
        if not text or kind not in ("heard", "say"):
            continue
        if kind == "say" and not narrator and e.get("source") == "narrator":
            continue
        t0 = e["t"]
        guess = max(2.0, len(text) / CHARS_PER_S + 0.6)
        if kind == "say":
            # Speech ends when the status leaves "speaking" after this line started.
            ends = [t for t, s in statuses if t > t0 + 0.3 and s != "speaking"]
            t1 = min(ends[0] + 0.4, t0 + guess + 4) if ends else t0 + guess
            t1 = max(t1, t0 + 1.5)
            label = f"Player Two: {text}"
        else:
            t1 = t0 + max(2.5, guess)
            label = f"You: {text}"
        # One sentence at a time, each shown for its share of the spoken time.
        parts = _chunks(text)
        total = sum(len(x) for x in parts) or 1
        a = t0
        for i, part in enumerate(parts):
            b = a + (t1 - t0) * len(part) / total
            who = label.split(":", 1)[0]
            out.append((a - start, b - start, f"{who}: {part}" if i == 0 else part))
            a = b
    out = [c for c in out if c[1] > 0]
    out.sort()
    # Never let one caption run over the next one.
    for i in range(len(out) - 1):
        a, b, txt = out[i]
        if b > out[i + 1][0]:
            out[i] = (a, max(a + 0.8, out[i + 1][0] - 0.05), txt)
    return out


def srt(caps: list[tuple[float, float, str]]) -> str:
    return "".join(f"{i}\n{_stamp(a)} --> {_stamp(b)}\n{txt}\n\n"
                   for i, (a, b, txt) in enumerate(caps, 1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("events", help="the events.jsonl from player-two run --log")
    p.add_argument("--video", help="the recording; its .start file gives the start time")
    p.add_argument("--start", type=float, help="the video's first frame, Unix seconds")
    p.add_argument("--offset", type=float, default=0.0, help="shift captions by seconds")
    p.add_argument("--no-narrator", action="store_true", help="leave out automatic narration")
    p.add_argument("--srt", help="where to write the subtitles (default: beside the video)")
    p.add_argument("--burn", help="also write a copy of the video with the captions burned in")
    a = p.parse_args()

    start = a.start
    if start is None and a.video:
        f = Path(a.video).with_suffix(Path(a.video).suffix + ".start")
        if f.exists():
            start = float(f.read_text().split()[0])
    if start is None:
        sys.exit("give --start, or --video with a .start file beside it")
    events = [json.loads(line) for line in Path(a.events).read_text().splitlines() if line.strip()]
    caps = captions(events, start - a.offset, narrator=not a.no_narrator)
    out = Path(a.srt or (Path(a.video).with_suffix(".srt") if a.video else "captions.srt"))
    out.write_text(srt(caps), encoding="utf-8")
    print(f"{len(caps)} captions -> {out}")
    if a.burn:
        if not a.video:
            sys.exit("--burn needs --video")
        style = ("FontName=DejaVu Sans,Fontsize=13,PrimaryColour=&H00FFFFFF,"
                 "OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=0,"
                 "BackColour=&H99000000,MarginV=22,Alignment=2")
        path = str(out.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.video, "-vf",
                        f"subtitles='{path}':force_style='{style}'", "-c:a", "copy", a.burn],
                       check=True)
        print(f"burned -> {a.burn}")


if __name__ == "__main__":
    main()
