"""demo_check: the bar each run sheet line is held to, on small logs and the recorded ones."""
from __future__ import annotations

import argparse
import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "demo_check.py"
spec = importlib.util.spec_from_file_location("demo_check", SCRIPT)
dc = importlib.util.module_from_spec(spec)
sys.modules["demo_check"] = dc
spec.loader.exec_module(dc)

LIVE = Path.home() / ".cache/player-two/live"
LIVE3 = LIVE / "steam-2026-11-01c/events.jsonl"
LIVE2 = LIVE / "steam-2026-11-01b/events.jsonl"
line = {ln.id: ln for ln in dc.LINES}
T = 1000.0


def run(lid: str, parts: list[tuple[float, float, str, str]], extra=(),
        agent: float | None = None, budgets: str = "predicted", frames=None):
    """parts: (began, ended, source, text), seconds after the request."""
    ev = [{"t": T, "kind": "heard", "text": line[lid].request}]
    for a, b, src, text in parts:
        ev.append({"t": T + a - 0.01, "kind": "say", "text": text, "source": src})
        ev.append({"t": T + b, "kind": "spoken", "text": text, "began": T + a, "ended": T + b})
    ev += [{"t": T + t, "kind": k} for t, k in extra]
    if agent is not None:
        ev.append({"t": T + agent, "kind": "agent", "request": "x", "outcome": "x", "steps": []})
    ev.sort(key=lambda e: e["t"])
    return dc.judge(line[lid], ev[0], ev, T + 200, "t", budgets, frames)


FAIR = [(0.5, 1.5, "agent", "Going to Leene Square."), (3.0, 4.0, "agent", "We're at Leene Square.")]


def test_a_good_line_passes():
    r = run("A3", FAIR, agent=3.1)
    assert r.ok, r.fails
    assert r.first == pytest.approx(0.5)
    assert r.total == pytest.approx(3.1)
    assert r.silence == pytest.approx(1.5)
    assert [s.text for s in r.said] == [p[3] for p in FAIR]


def test_first_word_late_fails():
    r = run("A3", [(2.4, 3.2, "agent", "Going to Leene Square.")], agent=3.3)
    assert any("first word at 2.40" in f for f in r.fails)


def test_silence_over_three_seconds_fails():
    r = run("A3", [(0.5, 0.8, "agent", "Going to Leene Square."),
                   (4.2, 4.4, "agent", "We're at Leene Square.")], agent=4.3)
    assert any("3.4 s of silence" in f for f in r.fails)


def test_animation_silence_is_listed_up_to_its_cap():
    said = [(0.1, 1.5, "p2", "Okay, Crono will use Cyclone."),
            (5.5, 6.5, "agent", "Cyclone went off.")]
    r = run("B2", said, agent=5.6)
    assert r.ok, r.fails
    assert "Cyclone's animation" in r.allowed[0]
    r = run("B2", [said[0], (7.0, 8.0, "agent", "Cyclone went off.")], agent=7.1)
    assert any("over the 4.5 s allowed" in f for f in r.fails)


def test_a_walk_silence_needs_the_walker_in_the_log():
    said = [(0.1, 1.0, "p2", "Going outside."), (6.0, 7.0, "agent", "We're outside.")]
    assert any("silence" in f for f in run("A5", said, agent=6.1).fails)
    r = run("A5", said, extra=[(3.0, "walk_note")], agent=6.1)
    assert r.ok, r.fails
    assert "the walk out" in r.allowed[0]


def test_budget_comes_from_the_chosen_set():
    said = [(0.1, 1.0, "p2", "Okay."), (1.2, 2.0, "agent", "Going to the seller."),
            (2.2, 4.0, "reader", "Potion."), (4.2, 7.0, "reader", "Potion, 10."),
            (7.2, 10.0, "reader", "Panacea, 10."), (10.2, 13.0, "reader", "Shelter, 150."),
            (13.2, 15.0, "reader", "Water."), (15.2, 17.0, "reader", "Three."),
            (17.2, 18.0, "agent", "Got three potions.")]
    assert any("budget 16.0" in f for f in run("A4", said, agent=17.3).fails)
    assert run("A4", said, agent=17.3, budgets="sheet").ok


@pytest.mark.parametrize("text", [
    "No tool calls needed for this turn.",
    "Sorry, I lost my train of thought there. Could you ask me that again?",
    "I can't reach the model right now. Check the internet connection. I'll keep trying.",
    "Finish: You're fighting two small winged gold imp-like monsters.",
])
def test_internal_phrases_fail(text):
    r = run("A3", [*FAIR, (4.2, 6.0, "agent", text)], agent=3.1)
    assert any("internal phrase" in f for f in r.fails), r.fails


def test_off_topic_line_fails():
    text = ("If a tackle involves contact that endangers the safety of an opponent, it must be "
            "sanctioned as serious foul play.")
    r = run("A3", [*FAIR, (4.2, 6.0, "agent", text)], agent=3.1)
    assert any("off topic" in f for f in r.fails), r.fails


def test_said_twice_fails_but_a_fight_may_come_back_to_a_line():
    r = run("A3", [*FAIR, (4.2, 5.2, "agent", "We're at Leene Square.")], agent=3.1)
    assert any("said twice" in f for f in r.fails)
    said = [(0.1, 1.0, "p2", "Okay, I'll take this fight."), (1.1, 2.0, "agent", "Frog: Heal."),
            (6.5, 7.0, "agent", "Frog: Heal."), (7.2, 8.0, "agent", "Won that fight for you.")]
    assert not [f for f in run("B4", said, agent=7.3).fails if "twice" in f]
    said[2] = (2.5, 3.0, "agent", "Frog: Heal.")
    assert [f for f in run("B4", said, agent=7.3).fails if "twice" in f]


def test_the_lines_purpose_is_required_and_a_refusal_fails():
    r = run("A3", [(0.5, 1.0, "agent", "Okay."), (1.1, 1.5, "agent", "Done.")], agent=1.2)
    assert any("what this line is for" in f for f in r.fails)
    r = run("B2", [(0.1, 2.0, "agent", "Crono's knocked out, so he can't use Cyclone.")], agent=0.2)
    assert any("retakes" in f for f in r.fails)


@pytest.mark.parametrize("text,want", [
    ("Three golden winged imps. Everyone's at full health.", [("all_full", "", 0)]),
    ("Won that fight, and everyone's health is full.", [("all_full", "", 0)]),
    ("Crono's at 1 of 780, Frog's full, Marle's full.",
     [("hp", "Crono", 1), ("full", "Frog", 0), ("full", "Marle", 0)]),
    ("Crono used Cyclone. Frog and Marle are still at 1 HP.",
     [("hp", "Frog", 1), ("hp", "Marle", 1)]),
    ("Marle took 429, down to 1.", [("hp", "Marle", 1)]),
    ("Frog's Slurp healed Frog: 101 of 494.", [("hp", "Frog", 101)]),
    ("Marle uses Cure on Frog. Frog 1 HP, Marle 1 HP.", [("hp", "Frog", 1), ("hp", "Marle", 1)]),
    ("Cyclone went off, but Marle dropped to 1 HP", [("hp", "Marle", 1)]),
    ("Got you three potions for 30 gold. You have 370 left.",
     [("gold_left", "", 370), ("gold_spent", "", 30)]),
    ("Three Potions, 30 gold.", [("gold_spent", "", 30)]),
    ("Bought. You have 370 gold left.", [("gold_left", "", 370)]),
    ("Going to Leene Square.", []),
])
def test_claims(text, want):
    got = sorted((c.kind, c.name, c.value) for c in dc.claims(text))
    assert got == sorted(want)


class Frames:
    def __init__(self, seen):
        self.seen = seen

    def at(self, t):
        return t


def fake_reading(monkeypatch, fn):
    monkeypatch.setattr(dc, "reading", fn)


def test_a_figure_must_match_the_frame_when_its_words_are_said(monkeypatch):
    # True when the line begins; healed by the time its second sentence is said.
    def seen(t):
        return {"hp": {"Frog": (1 if t < T + 2 else 131, 494), "Marle": (1, 430)}}

    fake_reading(monkeypatch, seen)
    s = dc.Said(T, "agent", "Marle uses Cure on Frog. Frog 1 HP, Marle 1 HP.", T, T + 6)
    fails, _ = dc.check_figures(s, [s], Frames(None))
    assert any("said Frog 1" in f and "131/494" in f for f in fails), fails
    assert not any("said Marle" in f for f in fails)


def test_full_health_must_hold_for_as_long_as_its_said(monkeypatch):
    fake_reading(monkeypatch, lambda t: {"hp": {"Marle": (430 if t < T + 2 else 1, 430)}})
    s = dc.Said(T, "screen", "Three golden winged imps. Everyone's at full health.", T, T + 4)
    fails, _ = dc.check_figures(s, [s], Frames(None))
    assert any("Marle 1/430" in f for f in fails), fails


def test_the_reader_not_able_to_tell_is_a_note_not_a_fail(monkeypatch):
    fake_reading(monkeypatch, lambda t: {})
    s = dc.Said(T, "agent", "Got three potions for 30 gold. You have 370 left.", T, T + 3)
    fails, notes = dc.check_figures(s, [s], Frames(None))
    assert not fails
    assert any("can't tell" in n for n in notes)


def test_gold_against_the_shop_screen(monkeypatch):
    fake_reading(monkeypatch, lambda t: {"funds": 370 if t > T - 3 else 400})
    good = dc.Said(T, "agent", "Got three potions for 30 gold. You have 370 left.", T, T + 3)
    assert dc.check_figures(good, [good], Frames(None))[0] == []
    bad = dc.Said(T, "agent", "Got three potions for 40 gold. You have 360 left.", T, T + 3)
    fails = dc.check_figures(bad, [bad], Frames(None))[0]
    assert any("360 gold left" in f for f in fails) and any("40 gold spent" in f for f in fails)


def test_old_logs_are_timed_from_the_voices_status():
    ev = [{"t": T, "kind": "heard", "text": "Take me to the fair."},
          {"t": T + 0.1, "kind": "status", "state": "acting"},
          {"t": T + 0.5, "kind": "say", "text": "Going to Leene Square.", "source": "agent"},
          {"t": T + 0.6, "kind": "status", "state": "speaking"},
          {"t": T + 2.0, "kind": "status", "state": "acting"},
          {"t": T + 3.0, "kind": "say", "text": "We're at Leene Square.", "source": "agent"},
          {"t": T + 3.1, "kind": "agent", "request": "x", "outcome": "x", "steps": []},
          {"t": T + 3.2, "kind": "status", "state": "speaking"},
          {"t": T + 4.5, "kind": "status", "state": "watching"}]
    r = dc.judge(line["A3"], ev[0], ev, T + 100, "t", "sheet")
    assert r.ok, r.fails
    assert r.first == pytest.approx(0.6)


def test_report_keeps_the_latest_of_each_take(tmp_path):
    a = run("A3", FAIR, agent=3.1)
    b = run("B1", [(0.1, 2.0, "screen", "Three golden winged imps.")], agent=0.2)
    store = tmp_path / "check-report"
    dc.latest_takes(store, "A", [a])
    both = dc.latest_takes(store, "B", [b])
    assert [r.line.id for r in both] == ["A3", "B1"]
    assert both[1].ok
    report = tmp_path / "check-report.md"
    dc.write_report(both, report, "x")
    assert "| A3 Take me to the fair. | 1/1 |" in report.read_text()
    rates = tmp_path / "rates.txt"
    dc.write_rates(both, rates, "Measured:")
    assert rates.read_text().splitlines() == ["Measured:", "“Take me to the fair” 1/1",
                                              "“What are we fighting?” 1/1"]


def test_a_live_run_sends_every_line_in_order_and_judges_them(tmp_path):
    log = tmp_path / "events.jsonl"
    log.touch()
    sent = tmp_path / "sent.txt"
    fake = tmp_path / "player-two"
    fake.write_text(f"""#!{sys.executable}
import json, sys, time
log, sent = {str(log)!r}, {str(sent)!r}
args = sys.argv[2:]
def emit(**e):
    e.setdefault("t", time.time())
    open(log, "a").write(json.dumps(e) + "\\n")
if args[0] == "status":
    print("{{}}")
elif args[0] == "do":
    text = args[1]
    open(sent, "a").write(text + "\\n")
    now = time.time()
    emit(kind="heard", text=text, typed=True, t=now)
    answer = "Going to Leene Square." if "fair" in text else "You're at Leene Square."
    emit(kind="say", text=answer, source="agent", t=now + 0.1)
    emit(kind="spoken", text=answer, began=now + 0.1, ended=now + 0.6, t=now + 0.6)
    emit(kind="agent", request=text, outcome=answer, steps=[], turns=[], t=now + 0.2)
    emit(kind="status", state="watching", t=now + 0.7)
""")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    if sys.platform == "win32":
        # Windows does not execute a shebang script as a native program. Keep a real
        # subprocess integration test, using a batch launcher for the Python fixture.
        launcher = tmp_path / "player-two.cmd"
        launcher.write_text(f'@"{sys.executable}" "{fake}" %*\n')
        fake = launcher
    a = argparse.Namespace(p2=str(fake), log=str(log), out=str(tmp_path / "out"), frames="none",
                           monitor="DP-2", launcher="", budgets="predicted", repeats=2, take="A",
                           skip="A2,A4,A5", add="", reload="never",
                           report=str(tmp_path / "check-report.md"))
    assert dc.cmd_run(a) == 0
    assert sent.read_text().splitlines() == ["Where am I?", "Take me to the fair."] * 2
    report = (tmp_path / "check-report.md").read_text()
    assert "| A1 Where am I? | 2/2 |" in report
    assert "| A3 Take me to the fair. | 2/2 |" in report
    runs = sorted((tmp_path / "out").glob("take-A-*/report.txt"))
    assert len(runs) == 2
    assert "A3 PASS" in runs[0].read_text()
    assert (tmp_path / "check-rates.txt").read_text().startswith("Measured on live Steam")


def _found(path: Path, want: str) -> list:
    events = dc.read_log(path)
    return [r for r in dc.replay(events, budgets="sheet") if r.line.id == want]


@pytest.mark.skipif(not LIVE3.exists(), reason="the live 3 log isn't here")
def test_recorded_live3():
    events = dc.read_log(LIVE3)
    results = dc.replay(events, budgets="sheet")
    by = {}
    for r in results:
        by.setdefault(r.line.id, []).append(r)
    # The early cut's stale line.
    stale = [r for r in by["B1"] if any("Everyone's at full" in s.text for s in r.said)]
    assert len(stale) == 1
    assert any(f.startswith("stale") for f in stale[0].fails)
    # The good ones pass: the fair every time, at 3 to 4 s.
    assert all(r.ok for r in by["A3"]) and len(by["A3"]) == 11
    # "I can't reach the model" on "what does it look like?"
    bad = [r for r in by["A3b"] if any("reach the model" in s.text for s in r.said)]
    assert len(bad) == 2 and all(any("internal" in f for f in r.fails) for r in bad)
    # The right refusal isn't a take.
    refused = [r for r in by["B2"] if any("can't use" in s.text for s in r.said)]
    assert refused and all(any("retakes" in f for f in r.fails) for r in refused)
    # Before the speed work: the long silences SPEED profiled are all caught.
    assert not any(r.ok for r in by["A4"])
    assert not any(r.ok for r in by["B4"])


@pytest.mark.skipif(not LIVE2.exists(), reason="the live 2 log isn't here")
def test_recorded_live2():
    results = _found(LIVE2, "B1")
    assert any(any("internal" in f for f in r.fails) for r in results
               if any(s.text.startswith("Finish:") for s in r.said))
    assert any(any("reptite" in f for f in r.fails) for r in results
               if any("reptites" in s.text for s in r.said))


def test_log_reader_skips_a_line_being_written(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps({"t": 1, "kind": "say", "text": "x"}) + '\n{"t": 2, "ki')
    assert len(dc.read_log(p)) == 1
