#!/usr/bin/env python3
"""Internal synthetic rehearsal. Never produces submission-eligible footage."""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import struct
import subprocess
import time
from pathlib import Path

import make_video as m
import numpy as np
import soundfile as sf

TAG = "TEMP-NOT-FOR-SUBMISSION"


def read(path):
    return np.frombuffer(
        subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-ar",
                "24000",
                "-ac",
                "2",
                "-f",
                "f32le",
                "-",
            ]
        ),
        np.float32,
    ).reshape(-1, 2)


def say(k, text, path, speed=1):
    stamp = path.with_suffix(".tts.json")
    key = json.dumps([text, "am_michael", speed])
    if not path.exists() or not stamp.exists() or stamp.read_text() != key:
        audio, rate = k.create(text, voice="am_michael", speed=speed, lang="en-us")
        sf.write(path, audio, rate)
        stamp.write_text(key)
    marker = b"PLAYER_TWO_SYNTHETIC_STAND_IN"
    if marker not in path.read_bytes()[-512:]:
        chunk = b"P2ST" + struct.pack("<I", len(marker)) + marker + b"\0" * (len(marker) % 2)
        with path.open("r+b") as f:
            if f.read(4) != b"RIFF":
                raise ValueError("Stand-in provenance needs a RIFF WAV")
            f.seek(0, 2)
            f.write(chunk)
            length = f.tell()
            f.seek(4)
            f.write(struct.pack("<I", length - 8))
    x, rate = sf.read(path)
    return len(x) / rate


async def prepare(args):
    from video_mix_check import configure_key

    configure_key()
    from playertwo.app import game_keywords
    from playertwo.listen import transcribe_file
    from playertwo.voice import load_kokoro

    if TAG not in str(args.folder):
        raise ValueError("Rehearsal must live in a TEMP-NOT-FOR-SUBMISSION folder")
    args.folder.mkdir(parents=True, exist_ok=True)
    (args.folder / "STAND-IN.json").write_text('{"synthetic":true,"voice":"Kokoro am_michael"}\n')
    k, _ = load_kokoro()
    cfg = json.loads((args.template / "pacing.json").read_text())
    manifest = json.loads((args.template / "takes.json").read_text())
    base = []
    new = {}
    durations = []
    cache_path = args.folder / f"{TAG}-asr-cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    # Independent copies: never write a synthetic voice or event beside real footage.
    for p0 in cfg["shots"]:
        p = m._piece_from(p0)
        p.stand_in = True
        p.proof = ""
        p.overlays = [o for o in p.overlays if not o.where.startswith("trace")]
        if p.vo_prompt:
            key = {"identity": "intro", "c-how": "how", "limit": "limit"}[p.id]
            normal = args.folder / f"{TAG}-vo-{key}.wav"
            slow = args.folder / f"{TAG}-vo-{key}-slow.wav"
            n = say(k, p.vo_prompt, normal)
            s = say(k, p.vo_prompt, slow, 0.85)
            durations.append(
                {
                    "slot": key,
                    "allowed": p.out_s,
                    "normal": n,
                    "slower": s,
                    "normal_fits": n <= p.out_s,
                    "slower_fits": s <= p.out_s,
                }
            )
            if n + 0.7 > p.out_s:
                extra = n + 0.7 - p.out_s
                if p.kind == "card":
                    p.secs += extra
                else:
                    p.b += extra
                for o in p.overlays:
                    o.f1 += extra
            p.vo = str(normal.resolve())
        if p.kind == "clip":
            t = m.load_take(args.template, p.take, p.take == "take-B")
            heard = [
                (x, e)
                for x, e in t.heard()
                if p.a <= x < p.b and not p.freeze and p.audio_mode != "game"
            ]
            if heard:
                requests = []
                for i, (x, e) in enumerate(heard):
                    file = args.folder / f"{TAG}-{p.id}-request-{i}.wav"
                    n = say(k, e["text"], file)
                    keywords=["Player Two",*game_keywords(t.game_name())]
                    key = hashlib.sha256(file.read_bytes()+json.dumps(keywords).encode()).hexdigest()
                    if key not in cache:
                        before = time.monotonic()
                        transcript = await transcribe_file(str(file),keywords=keywords)
                        cache[key] = (transcript, time.monotonic() - before)
                        cache_path.write_text(json.dumps(cache, indent=2) + "\n")
                    transcript, elapsed = cache[key]
                    requests.append((x, e, file, n, transcript, elapsed))
                lead = max(
                    0, max(n + 0.3 - (x - p.a) for x, e, file, n, transcript, elapsed in requests)
                )
                name = TAG + "-" + p.id
                video = args.folder / (name + ".mp4")
                length = p.b - p.a + lead
                source_key = json.dumps(
                    [str(t.video.resolve()), t.video.stat().st_mtime_ns, p.a, p.b, lead]
                )
                source_stamp = video.with_suffix(".source.json")
                if (
                    not video.exists()
                    or not source_stamp.exists()
                    or source_stamp.read_text() != source_key
                ):
                    m._ff(
                        "-ss",
                        str(p.a),
                        "-t",
                        str(p.b - p.a),
                        "-i",
                        str(t.video),
                        "-an",
                        "-vf",
                        f"scale=1920:1080,fps=30,tpad=start_mode=clone:start_duration={lead}",
                        "-t",
                        str(length),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-crf",
                        "24",
                        str(video),
                    )
                    source_stamp.write_text(source_key)
                count = round(length * 24000)
                voice = np.zeros((count, 2), np.float32)
                game = voice.copy()
                for target, suffix in [(voice, ".voice.wav"), (game, ".game.wav")]:
                    src = read(t.video.with_suffix(suffix))
                    a = int(p.a * 24000)
                    clip = src[a : a + int((p.b - p.a) * 24000)]
                    dest = round(lead * 24000)
                    n = min(len(clip), count - dest)
                    target[dest : dest + n] += clip[:n]
                events = []
                epoch = 1000.0
                for e in t.events:
                    if e.get("kind") in ("heard", "status"):
                        continue
                    if e.get("kind") == "spoken" and not (
                        p.a - 0.025 <= e["began"] - t.start and e["ended"] - t.start <= p.b + 0.025
                    ):
                        continue
                    event = dict(e)
                    for key in ("t", "began", "ended"):
                        if isinstance(event.get(key), (int, float)):
                            event[key] = epoch + event[key] - t.start - p.a + lead
                    event["synthetic"] = True
                    events.append(event)
                for x, _e, file, n, text, elapsed in requests:
                    at = lead + x - p.a - n - 0.2
                    previous=[ev['ended']-epoch for ev in events if ev.get('kind')=='spoken' and ev['ended']-epoch<=lead+x-p.a]
                    if previous:at=max(at,max(previous)+.7)
                    if at+n>lead+x-p.a:raise ValueError('Synthetic request cannot fit without overlapping a reply; cut a whole beat or expand the source slot')
                    audio = read(file)
                    dest = max(0, round(at * 24000))
                    nframes = min(len(audio), count - dest)
                    voice[dest : dest + nframes] += audio[:nframes]
                    events.extend(
                        [
                            {
                                "kind": "status",
                                "state": "listening",
                                "t": epoch + at,
                                "synthetic": True,
                            },
                            {
                                "kind": "heard",
                                "text": text,
                                "raw_transcript": text,
                                "t": epoch + lead + x - p.a,
                                "typed": False,
                                "hands_free": True,
                                "transcribe_s": elapsed,
                                "synthetic": True,
                                "origin": "Kokoro rehearsal, file transcription; not microphone capture",
                            },
                        ]
                    )
                sf.write(video.with_suffix(".voice.wav"), voice, 24000, subtype="PCM_24")
                sf.write(video.with_suffix(".game.wav"), game, 24000, subtype="PCM_24")
                stamp = args.folder / (name + ".mp4.start")
                stamp.write_text(str(epoch) + "\n")
                log = args.folder / (name + ".events.jsonl")
                log.write_text("".join(json.dumps(e) + "\n" for e in events))
                new[name] = {
                    "video": video.name,
                    "start": stamp.name,
                    "events": log.name,
                    "edit": {"paced": True},
                }
                old_a = p.a
                p.take = name
                p.a = 0
                p.b = length
                p.black_for += lead if p.black_for else 0
                for o in p.overlays:
                    o.f0 = o.f0 + lead if o.f0 else 0
                    o.f1 += lead
                if p.result_at is not None:
                    p.result_at = p.result_at - old_a + lead
                p.voice_boosts = [
                    [a - old_a + lead, b - old_a + lead, g] for a, b, g in p.voice_boosts
                ]
                p.game_ducks = [[a - old_a + lead, b - old_a + lead, g] for a, b, g in p.game_ducks]
                p.box_until = p.box_until - old_a + lead if p.box_until else 0
            else:
                # Real source is referenced read-only, but the whole rehearsal stays synthetic.
                new[p.take] = {
                    **manifest.get(p.take, {}),
                    "video": str(t.video.resolve()),
                    "start": str(
                        (
                            args.template
                            / manifest.get(p.take, {}).get("start", p.take + ".mp4.start")
                        ).resolve()
                    ),
                    "events": str(
                        (
                            args.template
                            / manifest.get(p.take, {}).get("events", p.take + ".events.jsonl")
                        ).resolve()
                    ),
                }
        p.overlays.append(
            m.Overlay(TAG + " / SYNTHETIC REQUESTS AND FOUNDER VO", 24, "draft", 0, p.out_s)
        )
        base.append(p)
    new["_stand_in"] = True
    (args.folder / "takes.json").write_text(json.dumps(new, indent=2) + "\n")
    takes = {
        name: m.load_take(args.folder, name, name == "take-B")
        for name, entry in new.items()
        if isinstance(entry, dict) and "video" in entry
    }
    for t in takes.values():
        t.edit["paced"] = True
    # Evidence and boxes come from the newly staged logs; no hardcoded transcription badge.
    m.evidence_overlays(base, takes)
    for p in base:
        p.overlays = [
            o
            for o in p.overlays
            if o.text != "You:\nRequest begins"
            and not (p.id in ("ct-castle", "ct-forest-walk") and o.where.startswith("trace"))
        ]
        for o in p.overlays:
            if o.where.startswith("trace"):
                o.text = "\n".join(o.text.splitlines()[:2])
    m.sam_boxes(base, takes)
    (args.folder / "pacing.json").write_text(
        json.dumps(
            {
                "shots": [dataclasses.asdict(p) for p in base],
                "exceptions": cfg.get("exceptions", {}),
                "_stand_in": True,
            },
            indent=2,
        )
        + "\n"
    )
    (args.folder / f"{TAG}-cut.json").write_text(
        json.dumps([dataclasses.asdict(p) for p in base], indent=2) + "\n"
    )
    (args.folder / f"{TAG}-voice-durations.json").write_text(json.dumps(durations, indent=2) + "\n")
    report, blocked = m.pacing_report(base, takes, cfg.get("exceptions", {}))
    (args.folder / f"{TAG}-pace-report.md").write_text(
        report
        + "\nInternal rehearsal is intentionally not submission eligible. Inserted speech/holds expose the real space needed. Production pacing gate remains unchanged.\n"
    )
    print("VO durations", durations, flush=True)
    print("Rehearsal duration", sum(p.out_s for p in base), "pacing issues", blocked, flush=True)
    return base, takes


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("template", type=Path)
    a.add_argument("folder", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--prepare-only", action="store_true")
    args = a.parse_args()
    if TAG not in str(args.out):
        raise ValueError("Rehearsal output requires explicit TEMP-NOT-FOR-SUBMISSION name")
    parts, takes = asyncio.run(prepare(args))
    if not args.prepare_only:
        m.PROXY = True
        m.FPS = 30
        m.build(parts, takes, args.out, "libx264", 2)
        m.contact_sheet(args.out, args.out.with_suffix(".sheet.png"), 2)
