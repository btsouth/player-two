import asyncio

import pytest

from playertwo import earguide as g
from playertwo.locate import Match


def m(cx, area=0.01, cy=0.5):
    side = area ** 0.5 / 2
    return Match(cx, cy, area, (cx - side, cy - side, cx + side, cy + side))


@pytest.mark.parametrize("text,what", [
    ("walk me to my horse", "horse"),
    ("Walk me to my horse.", "horse"),
    ("guide me to the door", "door"),
    ("Lead me to a red barrel?", "red barrel"),
    ("lead me towards that tree,", "tree"),
    ("please guide me to the old mill, please", "old mill"),
    ("Okay, walk me over to the wagon", "wagon"),
])
def test_guide_requests(text, what):
    assert g.guide_request(text) == what


@pytest.mark.parametrize("text", [
    "where's my horse", "take me to my horse", "walk to the door", "walk me", "guide me",
    "open the journal",
])
def test_not_guide_requests(text):
    assert g.guide_request(text) is None


def test_the_first_look_says_the_side_and_a_steady_side_isnt_repeated():
    guide = g.Guide(0.0)
    step = guide.look([m(0.2)], 1.0)
    assert step.say == "Left." and step.pan == -0.6 and step.beeps == 1
    guide.said(step.say, 1.0)
    assert guide.look([m(0.22)], 2.0).say == ""


def test_a_new_side_is_said_after_two_looks_and_only_once():
    guide = g.Guide(0.0)
    guide.said(guide.look([m(0.2)], 1.0).say, 1.0)
    assert guide.look([m(0.5)], 2.0).say == ""
    step = guide.look([m(0.5)], 3.0)
    assert step.say == "Straight ahead."
    guide.said(step.say, 3.0)
    assert guide.look([m(0.5)], 4.0).say == ""


def test_straight_ahead_doesnt_flip_at_the_edge():
    guide = g.Guide(0.0)
    guide.said(guide.look([m(0.5)], 1.0).say, 1.0)
    for i, cx in enumerate((0.6, 0.62, 0.6, 0.62)):  # pan 0.2 to 0.24: still ahead once ahead
        assert guide.look([m(cx)], 2.0 + i).say == ""


def test_a_line_not_really_said_is_offered_again():
    guide = g.Guide(0.0)
    assert guide.look([m(0.2)], 1.0).say == "Left."
    assert guide.look([m(0.2)], 2.0).say == "Left."  # the game was talking: not said


def test_pitch_and_beeps_rise_with_the_box():
    far = g.Guide(0.0).look([m(0.5, 0.002)], 1.0)
    mid = g.Guide(0.0).look([m(0.5, 0.02)], 1.0)
    near = g.Guide(0.0).look([m(0.5, 0.07)], 1.0)
    assert far.hz == pytest.approx(660.0) and far.hz < mid.hz < near.hz
    assert (far.beeps, mid.beeps, near.beeps) == (1, 2, 3)
    assert far.wait_s > mid.wait_s > near.wait_s


def test_getting_close_once():
    guide = g.Guide(0.0)
    guide.said(guide.look([m(0.5, 0.02)], 1.0).say, 1.0)
    step = guide.look([m(0.5, 0.04)], 2.0)
    assert step.say == "Getting close."
    guide.said(step.say, 2.0)
    assert guide.look([m(0.5, 0.05)], 3.0).say == ""


def test_arrival_needs_two_looks_in_a_row():
    guide = g.Guide(0.0)
    assert guide.look([m(0.5, 0.11)], 1.0).end == ""
    assert guide.look([m(0.5, 0.06)], 2.0).end == ""  # one big box was a fluke
    assert guide.look([m(0.5, 0.11)], 3.0).end == ""
    step = guide.look([m(0.5, 0.11)], 4.0)
    assert step.end == "there" and step.say == "You're there."


def test_a_thing_end_on_arrives_by_height():
    guide = g.Guide(0.0)
    narrow = Match(0.5, 0.5, 0.04, (0.46, 0.3, 0.55, 0.75))
    assert guide.look([narrow], 1.0).end == ""
    assert guide.look([narrow], 2.0).end == "there"


def test_out_of_frame_says_turn_slowly_once_in_a_while_and_no_ping():
    guide = g.Guide(0.0)
    guide.said(guide.look([m(0.5)], 1.0).say, 1.0)
    assert guide.look([], 3.0).say == ""
    step = guide.look([], 5.5)
    assert step.say == "I don't see it. Turn slowly." and step.pan is None and step.beeps == 0
    guide.said(step.say, 5.5)
    assert guide.look([], 10.0).say == ""
    assert guide.look([], 21.0).say == "I don't see it. Turn slowly."


def test_the_side_is_said_again_when_it_comes_back():
    guide = g.Guide(0.0)
    guide.said(guide.look([m(0.2)], 1.0).say, 1.0)
    guide.said(guide.look([], 6.0).say, 6.0)
    step = guide.look([m(0.2)], 7.0)
    assert step.say == "Left." and step.pan == -0.6


def test_follows_the_same_one_among_several():
    guide = g.Guide(0.0)
    near, far = m(0.3, 0.03), m(0.8, 0.005)
    assert guide.look([far, near], 1.0).target is near
    # The other one's box is bigger now, but it's over there; ours moved a little.
    assert guide.look([m(0.8, 0.05), m(0.35, 0.03)], 2.0).target.cx == 0.35


def test_timeout_says_what_is_true():
    guide = g.Guide(0.0)
    assert guide.look([], 121.0).say == (
        "That's two minutes, and I never found it, so I've stopped.")
    guide = g.Guide(0.0)
    guide.look([m(0.5)], 1.0)
    step = guide.look([m(0.5)], 121.0)
    assert step.end == "timeout" and step.say == (
        "That's two minutes, and we didn't get there, so I've stopped.")


def session(monkeypatch, looks, speaking=False):
    """A session running one guide over scripted looks."""
    from types import SimpleNamespace

    from PIL import Image

    from playertwo import locate
    from playertwo.screen import Frame

    from test_app import bare_session

    s = bare_session()
    said, cues, emitted = [], [], []
    frame = Frame(Image.new("RGB", (64, 36)), 0)

    async def grab_now(show=False):
        return frame

    async def say(text, interrupt=True):
        said.append(text)

    async def seen(muse, img, phrase):
        await asyncio.sleep(0)
        return (looks.pop(0) if len(looks) > 1 else looks[0]), 0.5

    monkeypatch.setattr(locate, "locate", seen)
    monkeypatch.setattr(g, "LOST_S", 0.0)
    s.narrating = None
    s.pending = "x"
    s.grab_now = grab_now
    s.muse = None
    s.voice = SimpleNamespace(say=say, stop=lambda: None)
    s.mixer = SimpleNamespace(cue=lambda mono, pan=0.0: cues.append(pan))
    s.announce = lambda text, **k: _done(said.append(text))
    s.listening = lambda: False
    s._ears_on = lambda: speaking
    s.ears = SimpleNamespace(speaking=lambda: speaking)
    s.GAP_MAX_S = 0.0
    s.events.emit = lambda kind, **data: emitted.append((kind, data))
    return s, said, cues, emitted


def _done(_value):
    f = asyncio.get_event_loop().create_future()
    f.set_result(None)
    return f


@pytest.mark.asyncio
async def test_a_walk_to_arrival_logs_every_look(monkeypatch):
    s, said, cues, emitted = session(monkeypatch, [
        [m(0.2, 0.05)], [m(0.25, 0.11)], [m(0.25, 0.11)]])
    await s.guide("horse")
    await asyncio.sleep(0)
    assert said == ["Okay. Listen for the ping.", "Left.", "Getting close.", "You're there."]
    looks = [d for k, d in emitted if k == "guide"]
    assert [d["cycle"] for d in looks] == [1, 2, 3]
    assert looks[0]["pan"] == -0.6 and looks[0]["box"] == pytest.approx(
        list(m(0.2, 0.05).box)) and looks[0]["seconds"] == 0.5 and looks[0]["said"] == "Left."
    assert looks[0]["boxes"] == [pytest.approx(list(m(0.2, 0.05).box))]
    assert cues[0] == -0.6
    assert emitted[-1][0] == "guide_end" and emitted[-1][1]["reason"] == "there"


@pytest.mark.asyncio
async def test_game_speech_holds_the_words_not_the_pings(monkeypatch):
    s, said, cues, emitted = session(monkeypatch, [[m(0.2, 0.01)]], speaking=True)
    task = asyncio.create_task(s.guide("horse"))
    s.guide_task = task
    await asyncio.sleep(0.05)
    assert said == ["Okay. Listen for the ping."] and cues == [-0.6]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert emitted[-1] == ("guide_end", {"phrase": "horse", "reason": "stopped"})


@pytest.mark.asyncio
async def test_stop_ends_it_with_a_true_line(monkeypatch):
    s, said, cues, emitted = session(monkeypatch, [[m(0.2, 0.01)]])
    from playertwo.app import Session

    s.stop_side = lambda: Session.stop_side(s)
    s.spawn = asyncio.ensure_future
    s.agent_task = None
    s.answer = None
    s.side_task = s.guide_task = asyncio.create_task(s.guide("horse"))
    await asyncio.sleep(0.05)
    assert s.guiding()
    await s._local("stop")
    await asyncio.sleep(0.01)
    assert said[-1] == "Okay, stopped." and not s.guiding()
    assert ("guide_end", {"phrase": "horse", "reason": "stopped"}) in emitted


def test_a_box_bigger_in_one_look_waits_for_the_next_to_agree():
    guide = g.Guide(0.0)
    guide.look([m(0.5, 0.008)], 1.0)
    # Live: a box on the player's own character, for one look, over a far horse.
    step = guide.look([m(0.47, 0.03, cy=0.7)], 2.0)
    assert step.pan is None and step.say == "" and step.beeps == 0
    step = guide.look([m(0.5, 0.01)], 3.0)
    assert step.target.area == 0.01
    guide.look([m(0.5, 0.04)], 4.0)
    assert guide.look([m(0.5, 0.04)], 5.0).target.area == 0.04  # it was real: now believed


def test_a_box_the_camera_carries_is_dropped():
    guide = g.Guide(0.0)
    john = m(0.5, 0.05, cy=0.7)
    horse = m(0.2, 0.01, cy=0.4)
    assert guide.look([john, horse], 1.0, 0.1).target is john  # it can't tell yet
    guide.look([john, horse], 2.0, 0.1)
    step = guide.look([john, horse], 3.0, 0.1)  # 3 looks in one spot as the view moved
    assert step.target is not john
    assert guide.look([john], 4.0, 0.1).target is None


def test_a_still_view_keeps_a_still_box():
    guide = g.Guide(0.0)
    still = m(0.5, 0.01)
    for i in range(6):
        assert guide.look([still], float(i + 1), 0.0).target is still


def test_it_says_when_there_are_several_and_when_it_may_have_switched():
    guide = g.Guide(0.0)
    step = guide.look([m(0.2, 0.01), m(0.8, 0.002)], 1.0)
    assert step.say == "I see more than one. Going to the biggest." and step.target.cx == 0.2
    guide.said(step.say, 1.0)
    # Locked: a bigger one elsewhere doesn't take over, the one nearest the last does.
    assert guide.look([m(0.8, 0.02), m(0.22, 0.01)], 2.0).target.cx == 0.22
    # The one followed is gone and only the far one is left: one look, then honest.
    assert guide.look([m(0.95, 0.002)], 3.0).pan is None
    step = guide.look([m(0.95, 0.002)], 4.0)
    assert step.say == "Not sure that's the same one." and step.target.cx == 0.95


def test_picture_change():
    import numpy as np

    a = np.zeros((4, 4), dtype=np.uint8)
    assert g.picture_change(None, a) == 0.0 and g.picture_change(a, a) == 0.0
    assert g.picture_change(a, a + 255) == 1.0
