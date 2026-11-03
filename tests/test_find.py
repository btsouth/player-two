"""find: SAM 3.1's masks kept, measured, and put in front of Muse Spark."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from playertwo.agent import STILL_TOOLS, TOOLS, Agent, merge_finds
from playertwo.locate import (
    Found,
    Match,
    Shape,
    find_all,
    find_phrases,
    found_record,
    found_text,
    measure,
    parse_boxes,
    ping_order,
    where_request,
)
from playertwo.screen import Frame

RAW = (Path(__file__).parent / "data" / "sam_raw_one_mask.txt").read_text()


def test_a_real_sam_answer_keeps_its_mask():
    [m] = parse_boxes(RAW)
    assert m.box == (741 / 1280, 327 / 720, 906 / 1280, 480 / 720)
    assert (m.mask.height, m.mask.width, m.mask.left, m.mask.top) == (154, 166, 741, 327)
    assert (m.mask.frame_w, m.mask.frame_h) == (1280, 720)
    grid = m.mask.raster()
    assert grid.shape == (154, 166)
    assert set(np.unique(grid)) == {0, 1}
    # The box-based numbers are as they always were.
    assert round(m.cx, 4) == round((741 + 906) / 2 / 1280, 4)
    assert m.area == (906 - 741) * (480 - 327) / (1280 * 720)


def test_measure_goes_by_the_pixels_not_the_box():
    [m] = parse_boxes(RAW)
    full = m.mask.full_frame()
    assert full.shape == (720, 1280)
    assert full.sum() == m.mask.raster().sum()
    s = measure(m)
    assert s.area == full.sum() / (1280 * 720)
    assert s.area < m.area * 0.6  # a horse doesn't fill its box
    assert 741 / 1280 <= s.x1 < s.cx < s.x2 <= 907 / 1280
    assert 327 / 720 <= s.y1 < s.cy < s.y2 <= 481 / 720


def test_a_mask_reaching_past_the_frame_is_clipped():
    from playertwo.locate import Mask

    class Over(Mask):
        def raster(self):
            return np.ones((4, 4), dtype=np.uint8)

    assert Over(4, 4, "~", 8, 8, 10, 10).full_frame().sum() == 4


def test_a_box_without_a_usable_mask_is_still_a_match():
    boxes = parse_boxes("<0f>0<|box;x1=0;y1=0;x2=100;y2=100;w=1000;h=1000|><|mask;x=0|>,"
                        "1<|box;x1=800;y1=400;x2=900;y2=600;w=1000;h=1000|>")
    assert [b.mask for b in boxes] == [None, None]
    s = measure(boxes[1])
    assert (s.cx, s.cy, s.area) == (boxes[1].cx, boxes[1].cy, boxes[1].area)


def test_two_records_each_keep_their_own_mask():
    token = RAW.split("<|mask;", 1)[1]
    text = ("<0f>0<|box;x1=741;y1=327;x2=906;y2=480;w=1280;h=720|><|mask;" + token.strip()
            + ",7<|box;x1=1;y1=2;x2=166;y2=155;w=1280;h=720|><|mask;" + token.strip())
    a, b = parse_boxes(text)
    assert (a.mask.left, b.mask.left, b.mask.top) == (741, 1, 2)
    assert a.mask.payload == b.mask.payload


def test_find_phrases_are_kept_as_written():
    assert find_phrases(["Big Rock", " tree ", "big rock", ""]) == ["Big Rock", "tree"]
    assert find_phrases("a horse") == ["a horse"]
    assert find_phrases(None) == []


def call(name, args, id="1"):
    return SimpleNamespace(id=id, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(args)), model_dump=lambda: {
            "id": id, "type": "function", "function": {"name": name,
                                                       "arguments": json.dumps(args)}})


def test_split_finds_become_one():
    calls = [call("find", {"phrases": ["rock"]}, "1"), call("say", {"text": "hi"}, "2"),
             call("find", {"phrases": ["tree", "rock"]}, "3"),
             call("find", {"phrases": "horse"}, "4")]
    merged = merge_finds(calls)
    assert [c.id for c in merged] == ["1", "2"]
    assert json.loads(merged[0].function.arguments) == {"phrases": ["rock", "tree", "horse"]}


def test_one_find_or_none_is_left_alone():
    calls = [call("find", {"phrases": ["rock"]})]
    assert merge_finds(calls) == calls
    assert merge_finds([]) == []


def test_find_is_offered_and_presses_nothing():
    assert "find" in {t["function"]["name"] for t in TOOLS}
    assert "find" in STILL_TOOLS
    assert TOOLS[-1]["function"]["name"] == "finish"  # _tools puts things before it


class Stream:
    def __init__(self, text):
        self.text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def events():
            yield SimpleNamespace(type="response.output_text.delta", delta=self.text)
        return events()


class FakeMuse:
    """Spark splits its phrases across two finds, then answers; SAM answers from RAW."""

    def __init__(self, bad=(), answer="A horse right of the middle of the screen."):
        self.asked, self.images, self.efforts, self.seen, self.offered = [], [], [], [], []
        self.bad = bad
        self.answer = answer
        self.s = SimpleNamespace(image_max_side=1280, effort="minimal", agent_effort="low")
        self.client = SimpleNamespace(responses=SimpleNamespace(stream=self._sam))

    def _sam(self, model, input):
        assert model == "sam-3.1"
        text, image = input[0]["content"]
        self.asked.append(text["text"])
        self.images.append(image["image_url"])
        if text["text"] in self.bad:
            raise RuntimeError("SAM is down")
        return Stream(RAW if text["text"] == "horse" else "")

    async def tools(self, messages, tools, effort=None, max_tokens=3000):
        self.efforts.append(effort)
        self.seen.append(messages)
        self.offered.append([t["function"]["name"] for t in tools])
        if len(self.efforts) == 1:
            return SimpleNamespace(content="", tool_calls=[
                call("find", {"phrases": ["horse"]}, "a"),
                call("find", {"phrases": ["boulder"]}, "b")])
        return SimpleNamespace(content="", tool_calls=[
            call("finish", {"text": "A horse right of the middle of the screen."}, "c")])

    async def stream(self, messages, effort=None, max_tokens=1500):
        self.efforts.append(effort)
        self.seen.append(messages)
        for word in self.answer.split(" "):
            yield word + " "


async def run_agent(muse, request="What's around me, and what could I hide behind?"):
    img = Image.new("RGB", (1280, 720), (90, 120, 60))
    said, acts = [], []

    async def grab():
        return Frame(img, 1.0)

    async def speak(text):
        said.append(text)

    agent = Agent(muse, SimpleNamespace(), grab, speak, game="Some Game",
                  on_action=lambda n, a: acts.append((n, a)))
    return await agent.run(request), said, acts


async def test_an_open_question_gathers_split_finds_on_one_frame_and_answers():
    muse = FakeMuse()
    run, said, acts = await run_agent(muse)
    assert sorted(muse.asked) == ["boulder", "horse"]  # one phrase a SAM call, as written
    assert len(set(muse.images)) == 1  # one frame
    assert muse.offered == [["find"]]
    assert [s.tool for s in run.steps] == ["find"]
    # What SAM found at once, then what Spark makes of it.
    assert said == ["I can see a horse.", "A horse right of the middle of the screen."]
    assert run.outcome == " ".join(said)
    # Short prompts both ways, at the quickest effort: the player's waiting.
    assert muse.efforts == ["minimal", "minimal"]
    assert all(len(m[0]["content"]) < 1500 for m in muse.seen)
    # The answer is written from the result, which says what it can't.
    result = run.steps[0].result
    assert "horse: 1 found" in result and "boulder: none found" in result
    assert "not how far away" in result
    assert "mask" not in result.lower()
    assert result in muse.seen[1][1]["content"][1]["text"]
    assert "never give a distance" in muse.seen[1][0]["content"]
    # The log gets the whole of it through on_action.
    [(name, data)] = [a for a in acts if a[0] == "found"]
    assert name == "found" and data["phrases"] == ["horse", "boulder"]
    assert data["frame_t"] > 0
    [horse, boulder] = data["found"]
    assert horse.matches[0].mask is not None and not boulder.matches
    assert not any(a[0] == "find" for a in acts)  # no focus change, nothing pressed


async def test_a_find_in_any_other_request_is_an_ordinary_tool_of_the_full_agent():
    muse = FakeMuse()
    run, said, _ = await run_agent(muse, "Get me to shelter")
    assert sorted(muse.asked) == ["boulder", "horse"]
    assert "find" in muse.offered[0] and "press" in muse.offered[0]
    assert [s.tool for s in run.steps] == ["find", "finish"]
    assert said == ["A horse right of the middle of the screen."]


async def test_an_open_question_without_a_find_goes_to_the_full_agent():
    class NoFind(FakeMuse):
        async def tools(self, messages, tools, effort=None, max_tokens=3000):
            self.offered.append([t["function"]["name"] for t in tools])
            if len(self.offered) == 1:
                return SimpleNamespace(content="Lots of things.", tool_calls=[])
            return SimpleNamespace(content="", tool_calls=[
                call("finish", {"text": "A horse, on the right of the screen."}, "c")])

    muse = NoFind()
    run, said, _ = await run_agent(muse, "What's around me?")
    assert muse.offered[0] == ["find"] and "press" in muse.offered[1]
    assert said == ["A horse, on the right of the screen."]
    assert [s.tool for s in run.steps] == ["finish"]


async def test_one_phrase_failing_leaves_the_others():
    img = Image.new("RGB", (64, 36))
    found = await find_all(FakeMuse(bad=("boulder",)), img, ["horse", "boulder"])
    assert [f.phrase for f in found] == ["horse", "boulder"]
    assert len(found[0].shapes) == 1
    assert found[1].error == "SAM is down"
    assert "boulder: couldn't look (SAM is down)" in found_text(found)


def test_a_thing_two_phrases_outline_counts_once():
    from playertwo.locate import Mask, drop_repeats

    class Grid(Mask):
        def __new__(cls, grid):
            self = object.__new__(cls)
            object.__setattr__(self, "grid", grid)
            return self

        def __init__(self, grid):
            pass

        def full_frame(self):
            return self.grid

    def at(x1, x2):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[2:8, x1:x2] = 1
        return Match(0.5, 0.5, 0.1, mask=Grid(grid))

    rock, small_inside, tree = at(0, 8), at(2, 4), at(8, 10)
    same_rock = at(0, 7)
    found = [Found("rock", [rock, tree], [shape(0.4), shape(0.9)]),
             Found("boulder", [same_rock, small_inside], [shape(0.35), shape(0.3)])]
    drop_repeats(found)
    assert [len(f.matches) for f in found] == [2, 1]
    assert found[1].matches == [small_inside] and len(found[1].shapes) == 1


def shape(cx, area=0.01):
    return Shape(cx, 0.5, area, cx - 0.01, 0.4, cx + 0.01, 0.6)


def test_found_text_gives_position_and_area_and_never_distance():
    text = found_text([Found("door", [Match(0.1, 0.5, 0.3)], [shape(0.1, 0.25)]),
                       Found("key")])
    assert "door: 1 found" in text and "far left of the screen" in text and "area 25.0%" in text
    assert "key: none found" in text
    body = text.split("\n", 1)[1]
    for word in ("close", "far off", "nearby", "nearest", "distance", "steps"):
        assert word not in body


def test_found_text_lists_the_biggest_and_counts_the_rest():
    many = [shape(i / 10, area=(i + 1) / 100) for i in range(9)]
    line = found_text([Found("tree", [Match(0, 0, 0)] * 9, many)]).split("\n")[1]
    assert line.startswith("tree: 9 found") and line.endswith("and 3 smaller")
    assert line.index("area 9.0%") < line.index("area 8.0%")


async def test_pings_come_after_the_found_line_and_before_the_answer():
    muse = FakeMuse()
    order = []
    img = Image.new("RGB", (1280, 720))

    async def grab():
        return Frame(img, 1.0)

    async def speak(text):
        order.append(text)

    async def ping():
        order.append("ping")

    agent = Agent(muse, SimpleNamespace(), grab, speak, game="Some Game")
    agent.ping_found = ping
    await agent.run("What's around me?")
    assert order == ["I can see a horse.", "ping", "A horse right of the middle of the screen."]


async def test_an_answer_naming_nothing_found_still_gets_the_second_look():
    muse = FakeMuse(answer="Football fouls are called by the referee on the pitch every game.")
    verdicts = []

    async def no(*a):
        verdicts.append(a)
        return False

    from playertwo import ontopic

    keep, ontopic.second_look = ontopic.second_look, no
    try:
        run, said, _ = await run_agent(muse, "What's around me?")
    finally:
        ontopic.second_look = keep
    assert verdicts and said == ["I can see a horse."]
    assert run.outcome == "I can see a horse."


def test_found_line_says_what_was_found_in_order():
    from playertwo.locate import found_line

    m = Match(0.5, 0.5, 0.1)
    assert found_line([Found("horse", [m]), Found("large rock", [m] * 3), Found("cactus"),
                       Found("leafy tree", [m] * 8), Found("oak", [m])]) == (
        "I can see a horse, three large rocks, several leafy trees and an oak.")
    assert found_line([Found("bush", [m, m])]) == "I can see two bushes."
    assert found_line([Found("cactus", [m, m])]) == "I can see two cactuses."
    assert found_line([Found("the distant mountains", [m, m])]) == (
        "I can see two distant mountains.")
    assert found_line([Found("person", [m] * 5), Found("dry grass", [m])]) == (
        "I can see several people and some dry grass.")
    assert found_line([Found("door")]) == "I can't pick out any of those on screen."


def test_ping_order_is_per_object_a_phrase_at_a_time_left_to_right():
    found = [Found("rock", shapes=[shape(0.9), shape(0.2)]), Found("door"),
             Found("horse", shapes=[shape(0.75)])]
    assert ping_order(found) == [("rock", -0.6), ("rock", 0.8), ("horse", 0.5)]


def test_ping_order_caps_with_every_phrase_heard():
    trees = [shape(i / 20 + 0.05, area=(i + 1) / 100) for i in range(12)]
    order = ping_order([Found("tree", shapes=trees), Found("horse", shapes=[shape(0.5)])], 8)
    assert len(order) == 8
    assert order.count(("horse", 0.0)) == 1
    pans = [p for ph, p in order if ph == "tree"]
    assert pans == sorted(pans) and pans[-1] == round((11 / 20 + 0.05) * 2 - 1, 2)  # the biggest


def test_found_record_keeps_a_log_line_small_and_masks_beside_it(tmp_path):
    [m] = parse_boxes(RAW)
    found = [Found("a big horse!", [m], [measure(m)]), Found("door", error="no answer in time")]
    record = found_record(found, tmp_path, "12.5-")
    item = record[0]["items"][0]
    assert item["mask"] == "12.5-a-big-horse-1.png"
    assert item["mask_at"] == [741, 327] and item["mask_size"] == [154, 166]
    with Image.open(tmp_path / item["mask"]) as png:
        assert png.size == (166, 154)
        assert np.count_nonzero(np.asarray(png)) == m.mask.raster().sum()
    assert record[1] == {"phrase": "door", "count": 0, "error": "no answer in time", "items": []}
    assert len(json.dumps(record)) < 1000
    assert "mask" not in found_record(found)[0]["items"][0]


def test_session_logs_masks_beside_the_log_and_pings_each_object(tmp_path):
    import asyncio

    from playertwo.app import Events, Session

    [m] = parse_boxes(RAW)
    s = object.__new__(Session)
    s.events = Events()
    s.events.log_path = tmp_path / "take.events.jsonl"
    cues, focused = [], []
    s.focus_game = lambda: focused.append(1)

    async def wait_idle(timeout):
        cues.append("idle")

    s.voice = SimpleNamespace(wait_idle=wait_idle,
                              cue=lambda name, pan=0.0: cues.append((name, pan)))
    found = [Found("horse", [m], [measure(m)])]
    s.on_action("found", {"frame_t": 12.5, "phrases": ["horse"], "found": found, "seconds": 2.1})
    assert not focused  # nothing was pressed: the game's focus is left alone
    event = json.loads(s.events.log_path.read_text().splitlines()[-1])
    assert event["kind"] == "find" and event["frame_t"] == 12.5
    assert event["found"][0]["items"][0]["box"][0] == round(741 / 1280, 4)
    assert (tmp_path / "take.events.masks" / "12.500-horse-1.png").exists()

    asyncio.run(s._ping_found())
    # Said first, then one thing gets three pings, wide for the ear.
    pan = cues[1][1]
    assert cues == ["idle", ("ping", pan), ("ping", pan), ("ping", pan)]
    assert pan == 0.54  # 0.29 on screen, spread
    assert s.found is None


def test_open_request_spots_open_questions_only():
    from playertwo.locate import open_request

    for text in ("What could I use for cover, and is my horse near any of it?",
                 "What\u2019s around me?", "Is there anywhere to hide?", "What's out there ahead?",
                 "ok what can I see from here", "Is anyone nearby?"):
        assert open_request(text), text
    for text in ("What's the battle mode set to?", "Where's my horse?", "What does it say?",
                 "What's my health?", "Attack the one in front"):
        assert not open_request(text), text


def test_where_questions_stay_on_the_fast_path_and_open_ones_go_to_the_agent():
    from playertwo.earguide import guide_request
    from playertwo.fast import where_request as where_am_i

    assert where_request("Where's my horse?") == "horse"
    assert where_request("where's the door") == "door"
    for open_request in ("What could I use for cover, and is my horse near any of it?",
                         "What's around me?", "Is there anywhere to hide?",
                         "What can I see from here?"):
        assert where_request(open_request) is None
        assert guide_request(open_request) is None
        assert not where_am_i(open_request)
