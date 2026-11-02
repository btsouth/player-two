"""Demo captions start and end with the speech, and the voice says when that was."""

import importlib.util
import itertools
from pathlib import Path

import numpy as np

from playertwo.audio import RATE, Mixer

_spec = importlib.util.spec_from_file_location(
    "demo_captions", Path(__file__).parents[1] / "scripts" / "demo_captions.py")
demo_captions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo_captions)


def test_the_mixer_says_when_each_line_was_heard_and_where_one_was_cut():
    m = Mixer("test")  # no sound card: the callback is called by hand
    heard = []
    m.on_played = lambda text, began, ended, cut: heard.append((text, ended >= began, cut))
    m.push_speech(np.zeros(RATE // 2, np.float32), "Three Gargoyles.")
    m.push_speech(np.zeros(RATE, np.float32), "Marle took 429.")
    out = np.zeros((480, 2), np.float32)
    for _ in range(RATE // 2 // 480 + 1):  # a little past the first line
        m._callback(out, 480, None, None)
    assert heard == [("Three Gargoyles.", True, False)]
    m.clear_speech()  # cut off part way
    assert [h[0] for h in heard] == ["Three Gargoyles.", "Marle took 429."]
    assert heard[1][2] is True
    m.clear_speech()  # nothing playing: nothing to report
    assert len(heard) == 2


def _take_b():
    """Take B (live 3): the answer to "what are we fighting?" and the reader's hit line
    queued behind it; the voice was speaking from 21.92 to 27.27."""
    return [
        {"t": 19.91, "kind": "heard", "text": "What are we fighting?"},
        {"t": 21.78, "kind": "say", "text": "Three golden winged imps. Everyone's at full "
                                           "health.", "source": "screen"},
        {"t": 21.92, "kind": "status", "state": "speaking"},
        {"t": 21.94, "kind": "say", "text": "Marle took 429, down to 1.", "source": "reader"},
        {"t": 27.27, "kind": "status", "state": "watching"},
    ]


def test_old_logs_caption_lines_in_the_order_they_were_said():
    """Before: "Everyone's at full health." was put after "Marle took 429" and stayed up
    to 27.7 s, while Marle was at 1 HP on screen."""
    caps = demo_captions.captions(_take_b(), 0.0)
    p2 = [(round(a, 2), round(b, 2), txt) for a, b, who, txt in caps if who == "Player Two"]
    assert [t for _a, _b, t in p2] == ["Player Two: Three golden winged imps.",
                                       "Everyone's at full health.",
                                       "Player Two: Marle took 429, down to 1."]
    assert p2[0][0] >= 21.78 and p2[-1][1] <= 27.3
    assert all(b <= nxt[0] for (_a, b, _t), nxt in itertools.pairwise(p2))


def test_captions_follow_what_the_voice_logged_as_heard():
    events = [*_take_b(),
        {"t": 23.0, "kind": "spoken", "text": "Three golden winged imps.", "began": 21.95,
         "ended": 23.4},
        {"t": 24.6, "kind": "spoken", "text": "Everyone's at full health.", "began": 23.45,
         "ended": 24.6},
        {"t": 26.0, "kind": "spoken", "text": "Marle took 429,", "began": 24.65,
         "ended": 25.5},
        {"t": 26.5, "kind": "spoken", "text": "down to 1.", "began": 25.55, "ended": 26.4},
        {"t": 30.0, "kind": "spoken", "text": "Frog and Marle are at 1 H.P.", "began": 30.0,
         "ended": 32.0}]
    caps = [(round(a, 2), round(b, 2), txt) for a, b, who, txt in
            demo_captions.captions(events, 0.0) if who == "Player Two"]
    assert caps == [(21.95, 23.4, "Player Two: Three golden winged imps."),
                    (23.45, 24.6, "Player Two: Everyone's at full health."),
                    (24.65, 26.4, "Player Two: Marle took 429, down to 1."),
                    (30.0, 32.0, "Player Two: Frog and Marle are at 1 HP.")]


def test_narration_left_out_by_its_source():
    events = [{"t": 5.0, "kind": "say", "text": "A cliff ledge with chains.",
               "source": "narrator"},
              {"t": 5.0, "kind": "spoken", "text": "A cliff ledge with chains.", "began": 3.5,
               "ended": 5.0},
              {"t": 9.0, "kind": "say", "text": "Fight.", "source": "reader"},
              {"t": 9.6, "kind": "spoken", "text": "Fight.", "began": 9.1, "ended": 9.6}]
    caps = demo_captions.captions(events, 0.0, narrator=False)
    assert [c[3] for c in caps] == ["Player Two: Fight."]
    assert caps[0][1] - caps[0][0] >= 0.8  # a short line stays up long enough to read


def test_a_line_held_for_the_games_voice_is_captioned_as_it_was_heard():
    """What was heard before the pause, then the whole line once it started over: never
    a caption up through the game's own voice."""
    line = "A soldier walks in."
    events = [{"t": 11.0, "kind": "spoken", "text": line, "began": 10.0, "ended": 11.0,
               "cut": True},
              {"t": 15.0, "kind": "spoken", "text": line, "began": 13.0, "ended": 15.0}]
    caps = [(a, b, txt) for a, b, who, txt in demo_captions.captions(events, 0.0)
            if who == "Player Two"]
    assert caps == [(10.0, 11.0, f"Player Two: {line}"), (13.0, 15.0, f"Player Two: {line}")]


def test_hp_abbreviation_does_not_break_a_continuous_combo_warning():
    events=[{"t":2,"kind":"spoken","text":"Combo Attack: Reduces H.P.","began":0,"ended":2},
            {"t":2.8,"kind":"spoken","text":"to 1.","began":2,"ended":2.8}]
    caps=demo_captions.captions(events,0)
    assert caps == [(0,2.8,"Player Two","Player Two: Combo Attack: Reduces HP to 1.")]
