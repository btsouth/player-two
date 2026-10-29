"""Walking on into what isn't on screen yet, and the rough map of an area."""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from playertwo.explore import Trail, keep_walking, words


def texture(w, h, seed=3):
    rng = np.random.default_rng(seed)
    noise = (rng.random((h // 8, w // 8)) * 255).astype(np.uint8)
    img = Image.fromarray(noise).resize((w, h), Image.BICUBIC)
    return img.filter(ImageFilter.GaussianBlur(2)).convert("RGB")


class World:
    """A top-down world bigger than the screen, the camera on the character, a wall
    and a doorway that fades to black."""

    VIEW = (960, 540)

    def __init__(self, start=(1200, 700), wall=None, door=None, room=False, speed=7.0,
                 far=False):
        self.world = texture(2400, 1400)
        # A far-off layer over half the screen, scrolling at half speed (clouds, a sky).
        self.far = texture(2400, 1400, seed=9) if far else None
        self.pos = np.array(start, float)
        self.wall = wall  # (x0, y0, x1, y1): can't be walked through
        self.door = door  # (x0, y0, x1, y1): walking in fades to a new place
        self.room = room  # a fixed camera: the character walks across a still screen
        self.speed = speed
        self.stick = None
        self.dark = False
        self.text_box = False

    def steer(self, angle):
        self.stick = angle

    def _in(self, r, p):
        return r is not None and r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]

    def camera(self):
        w, h = self.VIEW
        if self.room:
            return np.array([700.0, 400.0])
        return np.array([min(max(self.pos[0] - w / 2, 0), self.world.width - w),
                         min(max(self.pos[1] - h / 2, 0), self.world.height - h)])

    async def grab(self):
        w, h = self.VIEW
        if self.stick is not None and not self.text_box:
            a = math.radians(self.stick)
            nxt = self.pos + np.array([math.sin(a), -math.cos(a)]) * self.speed
            if not self._in(self.wall, nxt):
                self.pos = nxt
            if self._in(self.door, self.pos):
                self.dark = True
        if self.dark:
            return Image.new("RGB", (w, h))
        cam = self.camera()
        view = self.world.crop((int(cam[0]), int(cam[1]), int(cam[0]) + w, int(cam[1]) + h))
        if self.far is not None:
            fx, fy = int(cam[0] / 2), int(cam[1] / 2)
            view.paste(self.far.crop((fx, fy, fx + w // 2, fy + h)), (0, 0))
        d = ImageDraw.Draw(view)
        if self.wall:
            x0, y0, x1, y1 = self.wall
            d.rectangle((x0 - cam[0], y0 - cam[1], x1 - cam[0], y1 - cam[1]), fill=(40, 40, 40))
        px, py = self.pos - cam
        d.rectangle((px - 12, py - 18, px + 12, py + 18), fill=(200, 30, 30))
        d.rectangle((px - 6, py - 14, px + 6, py - 4), fill=(250, 250, 250))
        if self.text_box:
            d.rectangle((40, 360, 920, 520), fill=(20, 20, 120))
        return view


async def test_keeps_walking_onto_new_ground_and_says_how_far():
    world = World()
    went = await keep_walking(world.grab, world.steer, "left", seconds=2.0)
    assert went.outcome == "walked", went
    assert world.stick is None  # let go
    assert went.steps[0] < -5 and abs(went.steps[1]) < 2
    assert "going left, still moving" in went.text("left")


async def test_a_far_off_layer_scrolling_at_its_own_speed_is_not_a_popup():
    world = World(far=True, speed=14.0)
    went = await keep_walking(world.grab, world.steer, "down", seconds=1.5)
    assert went.outcome == "walked", went


async def test_stops_at_a_wall_and_says_so():
    world = World(wall=(1050, 0, 1080, 1400))
    went = await keep_walking(world.grab, world.steer, "left", seconds=6.0)
    assert went.outcome == "blocked", went
    assert went.seconds < 4
    assert world.stick is None


async def test_stops_when_it_walks_into_a_new_place():
    world = World(door=(1040, 600, 1100, 800))
    went = await keep_walking(world.grab, world.steer, "left", seconds=6.0)
    assert went.outcome == "new_area", went
    assert "new place" in went.text("left")


async def test_follows_the_character_across_a_still_screen():
    world = World(room=True, wall=(1460, 0, 1500, 1400))
    went = await keep_walking(world.grab, world.steer, "right", seconds=6.0)
    assert went.outcome == "blocked", went
    assert went.steps[0] > 5


async def test_a_text_box_stops_it():
    world = World()
    world.text_box = True
    went = await keep_walking(world.grab, world.steer, "up", seconds=3.0)
    assert went.outcome in ("didnt_move", "interrupted"), went


async def test_the_trail_maps_an_area_and_starts_over_in_a_new_one():
    trail = Trail()
    trail.add("walk_to the stairs", "arrived", np.array([3.0, -2.0]))
    trail.add("keep_walking up", "blocked", np.array([0.0, -4.0]))
    assert "6 steps up and 3 right" in trail.notes()
    trail.add("walk_to the door", "new_area", np.array([1.0, 0.0]))
    assert trail.lines == ["entered this area by walk_to the door"]
    assert words(np.array([0.2, 0.1])) == "hardly anywhere"
    assert words(np.array([-1.0, 0.0])) == "1 step left"


async def test_the_agent_explores_and_remembers_where_it_went():
    import json
    from types import SimpleNamespace

    from playertwo.agent import Agent
    from playertwo.controls import DryRun
    from playertwo.screen import Frame

    world = World(wall=(1050, 0, 1080, 1400))

    class Pad(DryRun):
        def steer(self, angle, strength=1.0):
            super().steer(angle, strength)
            world.steer(angle)

    class Muse:
        s = SimpleNamespace(image_max_side=640)

        def __init__(self):
            self.replies = [("keep_walking", {"direction": "left", "seconds": 6}),
                            ("keep_walking", {"direction": "up", "seconds": 1.5}),
                            ("finish", {"text": "A wall to the west."})]
            self.results: list[str] = []
            self.asked: list[str] = []

        async def tools(self, messages, tools, effort=None, max_tokens=0):
            self.asked.append(messages[1]["content"][-1]["text"])
            self.results = [m["content"] for m in messages if m["role"] == "tool"]
            name, args = self.replies.pop(0)
            call = SimpleNamespace(id="c", function=SimpleNamespace(
                name=name, arguments=json.dumps(args)),
                model_dump=lambda: {"id": "c", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args)}})
            return SimpleNamespace(content="", tool_calls=[call])

    async def grab():
        return Frame(await world.grab(), 0)

    async def speak(_text):
        pass

    pad, muse = Pad(), Muse()
    agent = Agent(muse, pad, grab, speak, max_settle_s=0.1)
    result = await agent.run("head west")
    walks = muse.results
    assert result.outcome == "A wall to the west."
    assert walks[0].startswith("blocked: something stops you going left"), walks
    assert walks[1].startswith("walked: went") and "going up" in walks[1], walks
    assert world.stick is None and pad.log[-1] == "steer:off"
    # Once it has walked here, holding the d-pad is watched too.
    muse.replies = [("hold", {"input": "up", "ms": 1200}), ("finish", {"text": "ok"})]
    await agent.run("go up a bit")
    assert "steps up going up" in muse.results[-1], muse.results
    # The next request knows where this one went.
    muse.replies = [("finish", {"text": "ok"})]
    await agent.run("where have we been?")
    notes = muse.asked[-1]
    assert "keep_walking left: blocked; now" in notes and "steps up and" in notes
    assert "keep_walking up: walked" in notes
