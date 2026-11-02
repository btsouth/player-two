#!/usr/bin/env python3
"""Stage a reviewed map of tonight's takes, then run the normal guarded proxy path."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def stage(source: Path, mapping: dict, output: Path):
    if output.exists():
        raise ValueError("Use a new staging folder; never overwrite the current cut")
    template = Path(mapping["template"]).resolve()
    manifest = json.loads((template / "takes.json").read_text())
    pacing = json.loads((template / "pacing.json").read_text())
    for key in ('guide_mode','opening_quote'):
        if key in mapping:pacing[key]=mapping[key]
    if pacing.get('guide_mode','walk') not in ('walk','fallback'):
        raise ValueError('guide_mode must be walk or fallback')
    synthetic = (source / "STAND-IN.json").exists() or bool(mapping.get("stand_in"))
    if synthetic and "TEMP-NOT-FOR-SUBMISSION" not in str(output):
        raise ValueError("Stand-in staging needs TEMP-NOT-FOR-SUBMISSION in its path")
    output.mkdir(parents=True)
    for f in template.iterdir():
        if f.name in ("takes.json", "pacing.json", "STAND-IN.json"):
            continue
        if f.is_file():
            (output / f.name).symlink_to(f.resolve())
    for name, entry in mapping.get("takes", {}).items():
        if name=='guide' and pacing.get('guide_mode')=='fallback':continue
        v = (source / entry["video"]).resolve()
        stamp = (source / entry["start"]).resolve()
        events = (source / entry["events"]).resolve()
        required = [v, stamp, events, v.with_suffix(".game.wav"), v.with_suffix(".voice.wav")]
        for f in required:
            if not f.is_file():
                raise ValueError(f"Missing take companion: {f}")
        alias = "TEMP-NOT-FOR-SUBMISSION-" + name if synthetic else "incoming-" + name
        for f, dest in [
            (v, alias + ".mp4"),
            (stamp, alias + ".mp4.start"),
            (events, alias + ".events.jsonl"),
            (required[3], alias + ".game.wav"),
            (required[4], alias + ".voice.wav"),
        ]:
            (output / dest).symlink_to(f)
        old = manifest.get(name, {})
        manifest[name] = {
            **old,
            "video": alias + ".mp4",
            "start": alias + ".mp4.start",
            "events": alias + ".events.jsonl",
            "edit": {**old.get("edit", {}), **entry.get("edit", {})},
        }
        if not synthetic:manifest[name]['edit'].pop('guide_fixture',None)
        if name in mapping.get("unchanged_windows", []):
            original = template / old.get("video", name + ".mp4")
            if (
                not os.path.samefile(v, original)
                and hashlib.sha256(v.read_bytes()).digest()
                != hashlib.sha256(original.read_bytes()).digest()
            ):
                raise ValueError(f"{name}: unchanged_windows requires byte-identical footage")
        else:
            if "recording_delay" not in entry.get("edit", {}):
                raise ValueError("A changed take needs its measured recording_delay")
            delay=entry['edit']['recording_delay']
            if not isinstance(delay,(int,float)) or not math.isfinite(delay):
                raise ValueError('Recorder delay must be measured numeric seconds')
            if name == "rdr-menu" and entry.get("edit", {}).get("backend") != "keyboard":
                raise ValueError("New Red Dead menu take needs explicit reviewed keyboard backend")
            if name=='game2' and pacing.get('require_battle_speed') and 'battle_speed' not in entry.get('edit',{}):
                raise ValueError('New Monster Hunter take needs its reviewed battle_speed')
            missing = [
                p["id"]
                for p in pacing["shots"]
                if p.get("take") == name
                and not {"a", "b", "overlays"} <= mapping.get("shots", {}).get(p["id"], {}).keys()
            ]
            if missing:
                raise ValueError(
                    f"New {name} needs reviewed a/b/overlays for every shot: {missing}"
                )
    known = {p["id"] for p in pacing["shots"]}
    if set(mapping.get("shots", {})) - known:
        raise ValueError("Unknown shot ID in map")
    for p in pacing["shots"]:
        if p.get('guide') and pacing.get('guide_mode')=='fallback':continue
        patch = mapping.get("shots", {}).get(p["id"], {})
        if patch and any(not isinstance(patch.get(k), (int,float)) or not math.isfinite(patch[k]) for k in ('a','b') if k in patch):
            raise ValueError(f"{p['id']}: enter measured numeric a/b before importing")
        if p.get("take") in mapping.get("takes", {}) and p.get("take") not in mapping.get(
            "unchanged_windows", []
        ):
            # Source clocks never carry over silently to a different recording.
            for key, value in [
                ("game_ducks", []),
                ("voice_boosts", []),
                ("result_at", None),
                ("picture_until", None),
            ]:
                p[key] = patch.get(key, value)
            if p.get("black_for") and "black_for" not in patch:
                raise ValueError("New opening needs its frame-checked reveal point")
        p.update(patch)
        if p.get('take') in mapping.get('takes',{}) and not synthetic:
            p['stand_in']=False
            if p.get('guide') and not mapping['takes'][p['take']].get('edit',{}).get('eyes_covered'):
                raise ValueError('Real guide needs reviewed eyes_covered evidence')
    for key, name in mapping.get("voiceovers", {}).items():
        if key==('limit' if pacing.get('guide_mode')=='fallback' else 'limit-fallback'):continue
        f = (source / name).resolve()
        if not f.is_file():
            raise ValueError(f"Missing founder audio: {f}")
        target=output / f"vo-{key}{f.suffix}"
        if target.is_symlink():target.unlink()
        target.symlink_to(f)
        for p in pacing['shots']:
            if {'intro':'identity','how':'c-how','limit':'limit','limit-fallback':'limit'}.get(key)==p['id'] and not synthetic:
                p['stand_in']=False
    if synthetic:
        manifest["_stand_in"] = True
        (output / "STAND-IN.json").write_text(
            json.dumps({"synthetic": True, "source": str(source)}) + "\n"
        )
    (output / "takes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "pacing.json").write_text(json.dumps(pacing, indent=2) + "\n")
    (output / "drop-in-map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    return output


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("source", type=Path)
    a.add_argument("map", type=Path)
    a.add_argument("stage", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--full", action="store_true")
    a.add_argument("--prepare-only", action="store_true")
    args = a.parse_args()
    stage(args.source, json.loads(args.map.read_text()), args.stage)
    if not args.prepare_only and (args.stage / "STAND-IN.json").exists():
        if args.full:
            raise ValueError("Rehearsal is proxy-only and never submission eligible")
        import dataclasses

        import make_video as m

        config = json.loads((args.stage / "pacing.json").read_text())
        parts = [m._piece_from(p) for p in config["shots"]]
        for p in parts:
            p.stand_in = True
            if p.vo:
                p.vo = str(Path(p.vo).resolve())
        manifest = json.loads((args.stage / "takes.json").read_text())
        takes = {
            n: m.load_take(args.stage, n, n == "take-B")
            for n, e in manifest.items()
            if isinstance(e, dict) and "video" in e
        }
        for t in takes.values():
            t.edit["paced"] = True
        report, blocked = m.pacing_report(parts, takes, config.get("exceptions", {}))
        (args.stage / "TEMP-NOT-FOR-SUBMISSION-pace-report.md").write_text(
            report
            + "\nInternal overlength rehearsal only. Production and final gates are unchanged.\n"
        )
        if any(x.startswith(("partial:", "speech-speed:")) for x in blocked):
            raise ValueError("Even a rehearsal must keep selected source speech whole at1x")
        m.PROXY = True
        m.FPS = 30
        m.build(parts, takes, args.out, "libx264", 2)
        m.contact_sheet(args.out, args.out.with_suffix(".sheet.png"), 2)
        args.out.with_name(args.out.name + ".cut.json").write_text(
            json.dumps([dataclasses.asdict(p) for p in parts], indent=2) + "\n"
        )
    elif not args.prepare_only:
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("make_video.py")),
            str(args.stage),
            "--out",
            str(args.out),
            "--sheet",
            "--jobs",
            "2",
        ]
        if not args.full:
            cmd.append("--proxy")
        subprocess.run(cmd, check=True)
