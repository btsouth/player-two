#!/usr/bin/env python3
"""How well Player Two hears a game's own voices: the speech detector and SpeechTrack on
recorded game audio, against when the lines are really spoken.

    python scripts/game_speech_bench.py CLIP.wav ... [--grid] [--show N]

Each clip is 24 kHz mono 16-bit WAV. Beside it:
- CLIP.txt: XCOM 2's movie subtitle timings (a first line, then "start_ms end_ms key"), or
  CLIP.spans.json ([[start_s, end_s], ...]): when someone speaks. A clip with neither is
  taken to have no speech at all (music, effects).
- CLIP.exclude.json ([[start_s, end_s], ...]): stretches not to score, such as Player
  Two's own voice in a desktop recording (make it with --events).

Reported, over all clips:
- lines heard: subtitle lines with any detected speech inside them;
- start: when speech was first detected after a line's subtitle came up (median, 90th);
- false speech: seconds a minute detected outside every line (1 s margin), and how many
  times a minute speech "started" there: each would hold narration back or cut it off.

The detector's scores are cached beside each clip (CLIP.p.npy), so --grid (a sweep of
SpeechTrack's thresholds) is fast. Game audio stays out of the repo.

To score a desktop recording (game plus Player Two's voice), make its exclude file from
Player Two's event log first:
    python scripts/game_speech_bench.py --events events.jsonl --start EPOCH --exclude-for CLIP.wav
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from playertwo.gameaudio import FRAME_BYTES, FRAME_S, SILENCE_RMS, SpeechTrack


def read_wav(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        if w.getframerate() != 24000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 24 kHz mono 16-bit")
        return w.readframes(w.getnframes())


def scores(path: Path) -> np.ndarray:
    """The detector's speech probability for each 80 ms frame (cached)."""
    cache = path.with_suffix(".p.npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache)
    from playertwo.listen import SpeechDetector, ensure_vad

    det = SpeechDetector(ensure_vad())
    data = read_wav(path)
    out = []
    for i in range(0, len(data) - FRAME_BYTES + 1, FRAME_BYTES):
        frame = data[i:i + FRAME_BYTES]
        x = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        if float(np.sqrt(np.mean(x * x))) < SILENCE_RMS:
            det.reset()
            out.append(0.0)
        else:
            out.append(det(frame))
    p = np.array(out, dtype=np.float32)
    np.save(cache, p)
    return p


def spans_for(path: Path) -> list[tuple[float, float]] | None:
    js = path.with_suffix(".spans.json")
    if js.exists():
        return [tuple(x) for x in json.loads(js.read_text())]
    txt = path.with_suffix(".txt")
    if txt.exists():
        out = []
        for line in txt.read_text(errors="replace").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
                out.append((int(parts[0]) / 1000, int(parts[1]) / 1000))
        return out
    return None


def excluded(path: Path) -> list[tuple[float, float]]:
    js = path.with_suffix(".exclude.json")
    return [tuple(x) for x in json.loads(js.read_text())] if js.exists() else []


def run_track(p: np.ndarray, **kw) -> list[tuple[float, float]]:
    track = SpeechTrack(keep_s=1e9, **kw)
    for i, v in enumerate(p):
        track.push((i + 1) * FRAME_S, float(v))
    end = len(p) * FRAME_S
    return [(s, e if e is not None else end) for s, e in track.segments]


def inside(t: float, spans, margin: float = 0.0) -> bool:
    return any(a - margin <= t <= b + margin for a, b in spans)


def score(clips, **kw) -> dict:
    lines = heard = 0
    starts: list[float] = []
    false_s = false_n = 0.0
    minutes = 0.0
    neg_minutes = 0.0
    for _path, p, spans, skip in clips:
        segs = run_track(p, **kw)
        dur = len(p) * FRAME_S
        truth = spans or []
        # Time scored as "no speech should be here": outside lines (1 s margin), not skipped.
        grid = np.arange(0, dur, FRAME_S)
        free = [t for t in grid if not inside(t, truth, 1.0) and not inside(t, skip)]
        minutes += len(free) * FRAME_S / 60
        if spans is None:
            neg_minutes += len(free) * FRAME_S / 60
        for t in free:
            if any(s <= t <= e for s, e in segs):
                false_s += FRAME_S
        for s, _e in segs:
            if not inside(s, truth, 1.0) and not inside(s, skip):
                false_n += 1
        for a, b in truth:
            if inside(a, skip):
                continue
            lines += 1
            hits = [s for s, e in segs if s <= b and e >= a]
            if hits:
                heard += 1
                after = [max(0.0, s - a) for s, e in segs if s <= b and e >= a]
                starts.append(min(after))
    return {
        "lines": lines, "heard": heard,
        "start_med": statistics.median(starts) if starts else None,
        "start_p90": sorted(starts)[int(0.9 * (len(starts) - 1))] if starts else None,
        "false_s_per_min": false_s / minutes if minutes else 0.0,
        "false_starts_per_min": false_n / minutes if minutes else 0.0,
        "free_min": minutes, "negative_min": neg_minutes,
    }


def policy(clips, gated: bool = True, line_s: float = 2.0, every_s: float = 3.0,
           most_s: float = 4.0) -> dict:
    """Narration played over each clip as the session plays it: a line of `line_s` seconds
    wants to start every `every_s` seconds (one at a time). Gated, it starts only when the
    game isn't speaking, stops the moment speech starts and begins again after it, and is
    dropped after `most_s` seconds without a gap. Ungated, it just plays. How much of what
    was played overlapped the game's real speech, and how long lines waited."""
    played = over = 0.0
    lines = dropped = 0
    waits: list[float] = []
    per_line: list[float] = []  # talk-over in each of the game's lines
    for _path, p, spans, _skip in clips:
        if not spans:
            continue
        talked = [0.0] * len(spans)
        track = SpeechTrack(keep_s=1e9)
        state, pos, since, due = "idle", 0.0, 0.0, 1.0
        for i, v in enumerate(p):
            t = (i + 1) * FRAME_S
            was = track.speaking()
            track.push(t, float(v))
            talking = track.speaking()
            if state == "idle" and t >= due:
                state, since, pos = "waiting", t, 0.0
                lines += 1
            if state == "waiting":
                if not gated or not talking:
                    state = "playing"
                    waits.append(t - since)
                elif t - since > most_s:
                    state, dropped, due = "idle", dropped + 1, t + every_s
            elif state == "playing":
                if gated and talking and not was:
                    state, held_at, pos = "held", t, 0.0  # the sentence starts over after
                else:
                    pos += FRAME_S
                    played += FRAME_S
                    for k, (a_, b_) in enumerate(spans):
                        if a_ <= t <= b_:
                            over += FRAME_S
                            talked[k] += FRAME_S
                            break
                    if pos >= line_s:
                        state, due = "idle", t + every_s - line_s
            elif state == "held":
                if not talking:
                    state = "playing"
                elif t - held_at > most_s:
                    state, dropped, due = "idle", dropped + 1, t + every_s
        per_line += talked
    hit = [x for x in per_line if x > 0]
    return {"game_lines": len(per_line), "talked_over": len(hit),
            "over_0.5s": sum(1 for x in per_line if x > 0.5),
            "over_med_s": round(statistics.median(hit), 2) if hit else 0.0,
            "lines": lines, "dropped": dropped, "played_s": round(played, 1),
            "overlap_s": round(over, 1), "overlap_pct": round(100 * over / played, 1)
            if played else 0.0, "wait_med": round(statistics.median(waits), 2) if waits
            else None}


def exclude_from_events(events: Path, start: float, out: Path, margin: float = 0.6) -> None:
    """Player Two's own speaking stretches (status "speaking" until the next status), in
    seconds from the recording's start, widened by `margin`."""
    evs = []
    for line in events.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "status":
            evs.append((e["t"] - start, e.get("state")))
    spans = []
    for (t, state), nxt in zip(evs, [*evs[1:], (evs[-1][0] + 30, "end")], strict=True):
        if state == "speaking":
            spans.append([round(t - margin, 2), round(nxt[0] + margin, 2)])
    out.with_suffix(".exclude.json").write_text(json.dumps(spans))
    print(f"{len(spans)} stretches of Player Two's voice -> {out.with_suffix('.exclude.json')}")


LINES = [
    "Commander, the squad is in position. We're ready when you are.",
    "Stay sharp. ADVENT patrols sweep this sector every few minutes.",
    "I never thought I'd see this place again.",
    "Get down! They're coming over the wall!",
    "You know what they say about the Midnight Channel, right?",
    "We leave at dawn. Get some rest while you can.",
    "That's the last of them. Good work, everyone.",
    "Hold on, something's not right here.",
    "If we don't stop them now, there won't be anything left to save.",
    "Welcome back. We've got a lot to catch up on.",
    "Over here! I think I found something.",
    "Listen to me. You have to trust me on this one.",
]
VOICES = ["am_michael", "af_bella", "bm_george", "af_sarah", "am_fenrir", "bf_emma",
          "am_onyx", "af_nicole"]


def _radio(x: np.ndarray) -> np.ndarray:
    """A comms-radio voice: 300 to 3400 Hz and a little clipping."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / 24000)
    spec[(f < 300) | (f > 3400)] = 0
    y = np.fft.irfft(spec, len(x))
    return np.tanh(3 * y / (np.abs(y).max() + 1e-9)) * np.abs(y).max()


def make_mixes(backgrounds: list[Path], out: Path, n: int, seed: int = 7) -> None:
    """Game audio without speech, with spoken lines laid over it at known times and at
    three levels against the background (+6, 0 and -6 dB), half of them as radio."""
    from playertwo.voice import load_kokoro

    rng = np.random.default_rng(seed)
    kokoro, _ = load_kokoro()
    out.mkdir(parents=True, exist_ok=True)
    beds = []
    for b in backgrounds:
        x = np.frombuffer(read_wav(b), dtype="<i2").astype(np.float32) / 32768.0
        if spans_for(b) is not None:
            continue  # it has speech in it
        skip = excluded(b)
        if skip:  # only stretches of at least 20 s with none of Player Two's voice in them
            ok = np.ones(len(x), bool)
            for a_, b_ in skip:
                ok[max(0, int(a_ * 24000)):max(0, int(b_ * 24000))] = False
            edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
            for s, e in zip(edges[::2], edges[1::2], strict=False):
                if e - s >= 20 * 24000:
                    beds.append((b.stem, x[s:e]))
        elif len(x) >= 20 * 24000:
            beds.append((b.stem, x))
    speech = {}
    for i in range(n):
        name, bed = beds[rng.integers(len(beds))]
        length = min(len(bed), 45 * 24000)
        start = rng.integers(0, len(bed) - length + 1)
        mix = bed[start:start + length].copy()
        bed_rms = float(np.sqrt(np.mean(mix ** 2))) or 1e-3
        level = (6, 0, -6)[i % 3]
        radio = i % 2 == 1
        spans, t = [], 1.0 + rng.uniform(0, 2)
        while True:
            line = LINES[rng.integers(len(LINES))]
            voice = VOICES[rng.integers(len(VOICES))]
            key = (line, voice)
            if key not in speech:
                y, _sr = kokoro.create(line, voice=voice, speed=1.0, lang="en-us")
                y = np.asarray(y, np.float32)
                loud = np.flatnonzero(np.abs(y) > 0.02 * np.abs(y).max())
                speech[key] = y[loud[0]:loud[-1] + 1]
            y = speech[key]
            if radio:
                y = _radio(y)
            if t + len(y) / 24000 > length / 24000 - 1.0:
                break
            gain = bed_rms * 10 ** (level / 20) / (float(np.sqrt(np.mean(y ** 2))) or 1e-3)
            s = int(t * 24000)
            mix[s:s + len(y)] += gain * y
            spans.append([round(t, 3), round(t + len(y) / 24000, 3)])
            t += len(y) / 24000 + rng.uniform(1.5, 5.0)
        mix = np.clip(mix, -1, 1)
        path = out / f"mix{i:02d}_{name}_{level:+d}dB{'_radio' if radio else ''}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes((mix * 32767).astype("<i2").tobytes())
        path.with_suffix(".spans.json").write_text(json.dumps(spans))
    print(f"{n} mixes in {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clips", nargs="*", type=Path)
    ap.add_argument("--mix", type=Path, help="make N labelled mixes from the clips (no speech in"
                    " them) into this folder")
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--grid", action="store_true", help="sweep SpeechTrack's thresholds")
    ap.add_argument("--show", type=int, default=0, help="list N false starts and missed lines")
    ap.add_argument("--events", type=Path)
    ap.add_argument("--start", type=float)
    ap.add_argument("--exclude-for", type=Path)
    a = ap.parse_args()
    if a.events:
        exclude_from_events(a.events, a.start, a.exclude_for)
        return
    if a.mix:
        make_mixes(a.clips, a.mix, a.n)
        return
    clips = [(c, scores(c), spans_for(c), excluded(c)) for c in a.clips]
    pos = sum(1 for c in clips if c[2] is not None)
    print(f"{len(clips)} clips ({pos} with lines), "
          f"{sum(len(c[1]) for c in clips) * FRAME_S / 60:.1f} min")
    default = score(clips)
    if any(c[2] for c in clips):
        print("narration ungated", json.dumps(policy(clips, gated=False)))
        print("narration gated  ", json.dumps(policy(clips)))
    print("default", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in default.items()}))
    if a.show:
        for path, p, spans, skip in clips:
            segs = run_track(p)
            fs = [round(s, 1) for s, _e in segs
                  if not inside(s, spans or [], 1.0) and not inside(s, skip)]
            miss = [(a_, b_) for a_, b_ in (spans or [])
                    if not any(s <= b_ and e >= a_ for s, e in segs) and not inside(a_, skip)]
            if fs or miss:
                print(f"{path.name}: false starts at {fs[:a.show]} missed {miss[:a.show]}")
    if a.grid:
        rows = []
        for on, off, k, hang in itertools.product((0.5, 0.6, 0.7, 0.8), (0.3, 0.35, 0.5),
                                                  (1, 2, 3, 4), (0.3, 0.5, 0.8)):
            if off > on:
                continue
            r = score(clips, on=on, off=off, start_frames=k, hang_s=hang)
            rows.append((on, off, k, hang, r))
        rows.sort(key=lambda x: (x[4]["false_starts_per_min"], -x[4]["heard"]))
        print("on off k hang | heard/lines start_med p90 | false s/min starts/min")
        for on, off, k, hang, r in rows:
            print(f"{on} {off} {k} {hang} | {r['heard']}/{r['lines']} "
                  f"{r['start_med'] or 0:.2f} {r['start_p90'] or 0:.2f} | "
                  f"{r['false_s_per_min']:.2f} {r['false_starts_per_min']:.2f}")


if __name__ == "__main__":
    main()
