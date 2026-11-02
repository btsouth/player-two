#!/usr/bin/env python3
"""Machine intelligibility audit of an existing, immutable rendered video."""

import argparse
import asyncio
import difflib
import hashlib
import json
import re
import subprocess
from pathlib import Path

import make_video as m
import numpy as np


def words(s):
    s = s.lower().replace("h.p.", "hp")
    return re.findall(r"[a-z0-9]+", s)


def coverage(expected, actual):
    a, b = words(expected), words(actual)
    return sum(x.size for x in difflib.SequenceMatcher(None, a, b).get_matching_blocks()) / max(
        1, len(a)
    )


def pcm(path):
    return np.frombuffer(
        subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-",
            ]
        ),
        np.float32,
    )


def db(x):
    return float(20 * np.log10(max(1e-12, float(np.sqrt(np.mean(x * x)))))) if len(x) else -240.0


def wav(path, audio):
    import soundfile as sf

    sf.write(path, audio, 24000, subtype="PCM_16")


def configure_key():
    import os

    key = Path.home() / ".config/meta-hackathon/meta-api-key"
    if key.exists():
        os.environ.setdefault("META_API_KEY", key.read_text().strip())


async def audit(args):
    configure_key()
    from playertwo.listen import transcribe_file

    args.out.mkdir(parents=True, exist_ok=True)
    parts = [m._piece_from(x) for x in json.loads(args.cut.read_text())]
    mix = pcm(args.video)
    rows = []
    at = 0
    # Whole-film -14 LUFS then 2kHz low-pass, never normalize individual phrases.
    lp = args.out / "laptop-2khz.wav"
    m._ff(
        "-i",
        str(args.video),
        "-vn",
        "-af",
        "loudnorm=I=-14:TP=-1:LRA=9,lowpass=f=2000",
        "-ar",
        "24000",
        "-ac",
        "1",
        str(lp),
    )
    laptop = pcm(lp)
    for p in parts:
        if p.take and p.audio_mode != "game" and not p.freeze:
            t = m.load_take(args.takes, p.take, False)
            stem = pcm(t.video.with_suffix(".voice.wav"))
            game = pcm(t.video.with_suffix(".game.wav"))
            duck = args.out / f"{p.id}-duck.wav"
            m._ff(
                "-ss",
                str(p.a),
                "-t",
                str(p.b - p.a),
                "-i",
                str(t.video.with_suffix(".game.wav")),
                "-ss",
                str(p.a),
                "-t",
                str(p.b - p.a),
                "-i",
                str(t.video.with_suffix(".voice.wav")),
                "-filter_complex",
                "[0:a]apad=pad_dur=2"
                + m.game_duck_filter(p)
                + "[g];[1:a]apad=pad_dur=2[v];[g][v]sidechaincompress=threshold=0.025:ratio=8:attack=15:release=220[d]",
                "-map",
                "[d]",
                "-t",
                str(p.b - p.a),
                "-ar",
                "24000",
                "-ac",
                "1",
                str(duck),
            )
            bed = pcm(duck)
            for e in t.events:
                if e.get("kind") != "spoken":
                    continue
                a, b = e["began"] - t.start, e["ended"] - t.start
                if not p.a - 0.025 <= a < p.b:
                    continue
                lo, hi = max(0, a - 0.12), min(t.duration, b + 0.12)
                n = len(rows)
                variants = {}
                for name, audio, x, y in [
                    ("mix", mix, at + lo - p.a, at + hi - p.a),
                    ("voice", stem, lo, hi),
                    ("laptop", laptop, at + lo - p.a, at + hi - p.a),
                ]:
                    path = args.out / f"{n:02}-{name}.wav"
                    wav(path, audio[max(0, int(x * 24000)) : int(y * 24000)])
                    variants[name] = str(path)
                v = stem[int(a * 24000) : int(b * 24000)]
                g = game[int(a * 24000) : int(b * 24000)]
                d = bed[max(0, int((a - p.a) * 24000)) : int((b - p.a) * 24000)]
                rows.append(
                    {
                        "shot": p.id,
                        "at": at + a - p.a,
                        "expected": e["text"],
                        "voice_db": db(v),
                        "game_db": db(g),
                        "raw_margin": db(v) - db(g),
                        "duck_margin": db(v) - db(d),
                        "variants": variants,
                    }
                )
        at += p.out_s
    cache_path = args.out / "asr-cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    sem = asyncio.Semaphore(3)

    async def one(row, name, path):
        key = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        async with sem:
            if key not in cache:
                try:
                    cache[key] = {"text": await transcribe_file(path)}
                except Exception as e:  # noqa: BLE001 API/session failures are recorded per slice
                    cache[key] = {"error": str(e)[:200]}
                cache_path.write_text(json.dumps(cache, indent=2) + "\n")
            row[name] = cache[key]

    await asyncio.gather(*(one(row, n, p) for row in rows for n, p in row["variants"].items()))
    for row in rows:
        for name in ("mix", "voice", "laptop"):
            row[name]["coverage"] = coverage(row["expected"], row[name].get("text", ""))
        row["mask_candidate"] = row["voice"]["coverage"] >= 0.95 and row["mix"]["coverage"] < 0.95
        row["laptop_candidate"] = (
            row["voice"]["coverage"] >= 0.95 and row["laptop"]["coverage"] < 0.95
        )
    (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    report = [
        "# Draft 3j machine mix check",
        "",
        "Real Muse Voice Transcribe, one API session per slice/variant. **Machine check, not human listening.** No keywords supplied. Original 3j is unchanged. The laptop copy is whole-film normalized to -14 LUFS then low-passed at2kHz; this is a crude bandwidth check, not a speaker simulation. Levels are RMS dBFS from synchronized stems; duck margin uses the renderer’s sidechain settings. Coverage is ordered word matching, not a listening score. Numbers/homophones and native adjacent chunks need manual interpretation.",
        "",
        "| Film s / shot | Said | Mix transcript | Voice-only transcript | 2kHz transcript | Voice / game dBFS | Raw / duck margin dB | Flag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cell = lambda x: str(x).replace("|", "/").replace("\n", " ")
        flag = (
            "MASK CANDIDATE"
            if r["mask_candidate"]
            else "2kHz candidate"
            if r["laptop_candidate"]
            else "ASR/source mismatch"
            if r["voice"]["coverage"] < 0.95
            else "pass"
        )
        report.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    [
                        f"{r['at']:.2f} {r['shot']}",
                        r["expected"],
                        r["mix"].get("text", r["mix"].get("error")),
                        r["voice"].get("text", r["voice"].get("error")),
                        r["laptop"].get("text", r["laptop"].get("error")),
                        f"{r['voice_db']:.1f} / {r['game_db']:.1f}",
                        f"{r['raw_margin']:.1f} / {r['duck_margin']:.1f}",
                        flag,
                    ],
                )
            )
            + " |"
        )
    report += [
        "",
        f"{len(rows)} speech slices, {sum(r['mask_candidate'] for r in rows)} mix candidates, {sum(r['laptop_candidate'] for r in rows)} low-pass candidates. Raw API results/cached slices are alongside this report. A candidate is not proof of masking: inspect exact words and residual alignment first.",
    ]
    args.report.write_text("\n".join(report) + "\n")
    print(report[-1])
    errors = sum("error" in r[n] for r in rows for n in ("mix", "voice", "laptop"))
    if errors:
        raise RuntimeError(f"Incomplete Muse audit: {errors} API errors; see table/cache")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("video", type=Path)
    a.add_argument("cut", type=Path)
    a.add_argument("takes", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--report", type=Path, required=True)
    asyncio.run(audit(a.parse_args()))
