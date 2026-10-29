"""Walking to things on screen, against a small simulated top-down game."""

import asyncio
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from playertwo.walk import Look, Walker, new_place, phase_shift


def texture(w, h, seed=1):
    rng = np.random.default_rng(seed)
    noise = (rng.random((h // 8, w // 8)) * 255).astype(np.uint8)
    img = Image.fromarray(noise).resize((w, h), Image.BICUBIC)
    return img.filter(ImageFilter.GaussianBlur(2)).convert("RGB")


class Game:
    """A top-down world bigger than the screen; the camera follows the character."""

    VIEW = (960, 540)

    def __init__(self, start=(150, 700), goal=(880, 560), wall=None, speed=7.0,
                 camera_turn=0.0):
        self.world = texture(2400, 1400)
        self.pos = np.array(start, float)
        self.goal = np.array(goal, float)
        self.wall = wall  # (x0, y0, x1, y1) in world pixels, can't be walked through
        self.speed = speed
        self.turn = camera_turn  # stick angle minus the direction it walks on screen
        self.stick = None
        self.frames = 0
        self.asked = []
        self.held_when_asked = []

    def steer(self, angle):
        self.stick = angle

    def _blocked(self, p):
        if self.wall is None:
            return False
        x0, y0, x1, y1 = self.wall
        return x0 - 12 <= p[0] <= x1 + 12 and y0 - 18 <= p[1] <= y1 + 18

    def camera(self):
        w, h = self.VIEW
        cx = min(max(self.pos[0] - w / 2, 0), self.world.width - w)
        cy = min(max(self.pos[1] - h / 2, 0), self.world.height - h)
        return np.array([cx, cy])

    async def grab(self):
        self.frames += 1
        if self.stick is not None:
            a = math.radians(self.stick - self.turn)
            step = np.array([math.sin(a), -math.cos(a)]) * self.speed
            nxt = self.pos + step
            if not self._blocked(nxt):
                self.pos = nxt
        cam = self.camera()
        w, h = self.VIEW
        view = self.world.crop((int(cam[0]), int(cam[1]), int(cam[0]) + w, int(cam[1]) + h))
        d = ImageDraw.Draw(view)
        if self.wall:
            x0, y0, x1, y1 = self.wall
            d.rectangle((x0 - cam[0], y0 - cam[1], x1 - cam[0], y1 - cam[1]), fill=(40, 40, 40))
        gx, gy = self.goal - cam
        d.rectangle((gx - 14, gy - 20, gx + 14, gy + 20), fill=(30, 160, 30))
        px, py = self.pos - cam
        # The character: a red body with a white face, so it's easy to follow.
        d.rectangle((px - 12, py - 18, px + 12, py + 18), fill=(200, 30, 30))
        d.rectangle((px - 6, py - 14, px + 6, py - 4), fill=(250, 250, 250))
        return view

    def box(self, center, hw, hh):
        cam = self.camera()
        x, y = center - cam
        w, h = self.VIEW
        return [(x - hw) / w * 1000, (y - hh) / h * 1000, (x + hw) / w * 1000,
                (y + hh) / h * 1000]


class FakeMuse:
    """Points at the character and the target the way Muse Spark would."""

    def __init__(self, game, via=None):
        self.game = game
        self.via = via or []

    async def json(self, messages, effort=None, max_tokens=0):
        g = self.game
        g.asked.append(messages[0]["content"][-1]["text"])
        g.held_when_asked.append(g.stick)
        cam, (w, h) = g.camera(), g.VIEW
        via = [[(x - cam[0]) / w * 1000, (y - cam[1]) / h * 1000] for x, y in self.via]
        self.via = []
        return {"player": g.box(g.pos, 12, 18), "target": g.box(g.goal, 14, 20),
                "via": via, "note": ""}


async def walk(game, muse=None, timeout=15.0):
    w = Walker(muse or FakeMuse(game), game.grab, game.steer)
    return await w.walk("the green chest", timeout=timeout), w


def test_template_and_scroll_measurement():
    img = np.asarray(texture(480, 270).convert("L"), dtype=np.float32)
    moved = np.roll(img, (5, -7), axis=(0, 1))
    x, y, _ = phase_shift(img, moved)
    assert (round(x), round(y)) == (-7, 5)
    # A character found by its colours, facing either way.
    scene = np.repeat(img[..., None], 3, axis=2).astype(np.uint8)
    hero = scene.copy()
    hero[100:130, 200:220] = (200, 30, 30)
    hero[104:110, 204:216] = (250, 250, 250)
    look = Look(hero, (200, 100, 220, 130))
    turned = scene.copy()
    turned[110:140, 240:260] = (200, 30, 30)
    turned[114:120, 248:256] = (240, 240, 245)
    x, y, score = look.find(turned, (215, 115), radius=40)
    assert abs(x - 250) <= 1 and abs(y - 125) <= 1 and score > 0.8
    assert not new_place(img, moved)
    assert new_place(img, img * 0.2)


async def test_walks_across_a_scrolling_world_to_the_target():
    game = Game()
    result, _ = await walk(game)
    assert result.outcome == "arrived", result.text()
    assert np.hypot(*(game.pos - game.goal)) < 80
    assert min(abs(game.pos - game.goal)) < 14  # lined up with it, to talk or open
    assert game.stick is None  # let go of the stick
    assert len(game.asked) == 1  # one look by the model; the rest was local


async def test_goes_around_a_wall_by_the_waypoint():
    game = Game(start=(150, 600), goal=(800, 600), wall=(450, 450, 510, 760))
    result, _ = await walk(game, FakeMuse(game, via=[(480, 380)]))
    assert result.outcome == "arrived", result.text()


async def test_asks_again_when_blocked_without_a_way_around():
    game = Game(start=(150, 600), goal=(800, 600), wall=(450, 0, 510, 1400))
    muse = FakeMuse(game)
    result, _ = await walk(game, muse, timeout=12)
    assert result.outcome in ("blocked", "timeout")
    assert any("blocked" in a for a in game.asked[1:])  # it asked for a way around


async def test_not_found_and_stopped():
    game = Game()

    class NoTarget(FakeMuse):
        async def json(self, *a, **k):
            r = await super().json(*a, **k)
            return {**r, "target": None, "note": "probably off screen to the right"}

    result, _ = await walk(game, NoTarget(game))
    assert result.outcome == "not_found" and "right" in result.detail

    async def no():
        return False

    w = Walker(FakeMuse(game), game.grab, game.steer, keep_going=no)
    result = await w.walk("the green chest")
    assert result.outcome == "stopped" and game.stick is None


async def test_a_door_ends_the_walk():
    game = Game()
    real = game.grab

    async def fade():
        img = await real()
        return img if game.frames < 8 else Image.new("RGB", img.size, (0, 0, 0))

    w = Walker(FakeMuse(game), fade, game.steer)
    result = await w.walk("the green chest")
    assert result.outcome == "new_area"


async def test_the_agent_walks_there_and_lets_go_while_the_player_talks():
    import json
    from types import SimpleNamespace

    from playertwo.agent import Agent
    from playertwo.controls import DryRun
    from playertwo.screen import Frame

    game = Game()

    class Pad(DryRun):
        def steer(self, angle, strength=1.0):
            super().steer(angle, strength)
            game.steer(angle)

    class Muse(FakeMuse):
        def __init__(self, game):
            super().__init__(game)
            self.s = SimpleNamespace(image_max_side=640)
            self.replies = [("walk_to", {"target": "the green chest"}),
                            ("finish", {"text": "There."})]
            self.results: list[str] = []

        async def tools(self, messages, tools, effort=None, max_tokens=0):
            self.results += [m["content"] for m in messages if m["role"] == "tool"]
            name, args = self.replies.pop(0)
            call = SimpleNamespace(id="c", function=SimpleNamespace(
                name=name, arguments=json.dumps(args)),
                model_dump=lambda: {"id": "c", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args)}})
            return SimpleNamespace(content="", tool_calls=[call])

    async def grab():
        return Frame(await game.grab(), 0)

    async def speak(_text):
        pass

    pad, muse = Pad(), Muse(game)
    agent = Agent(muse, pad, grab, speak, max_settle_s=0.1)
    run = asyncio.create_task(agent.run("go to the chest"))
    while game.stick is None:
        await asyncio.sleep(0.01)
    agent.hold()  # the player starts talking mid-walk
    await asyncio.sleep(0.3)
    assert game.stick is None
    agent.resume()
    result = await run
    assert result.outcome == "There."
    assert muse.results[0].startswith("arrived"), muse.results
    assert "steer:off" in pad.log


async def test_a_character_that_wont_move_ends_the_walk_quickly():
    game = Game(speed=0.0)  # a cutscene: the stick does nothing
    result, _ = await walk(game)
    assert result.outcome == "didnt_move" and result.seconds < 3


async def test_remembers_the_character_when_the_model_points_at_someone_else():
    game = Game(goal=(500, 640))
    muse = FakeMuse(game)
    w = Walker(muse, game.grab, game.steer)
    assert (await w.walk("the green chest")).outcome == "arrived"
    game.goal = np.array([900.0, 760.0])

    class Mixup(FakeMuse):
        async def json(self, *a, **k):
            r = await super().json(*a, **k)
            x1, y1, x2, y2 = r["player"]
            return {**r, "player": [x1 + 150, y1 - 120, x2 + 150, y2 - 120]}

    w.muse = Mixup(game)
    result = await w.walk("the green chest")
    assert result.outcome == "arrived", result.text()
    assert np.hypot(*(game.pos - game.goal)) < 80


async def test_starting_up_against_a_wall_is_not_a_cutscene():
    # Pressed against the near side of a wall, straight at the target behind it.
    game = Game(start=(436, 600), goal=(800, 600), wall=(450, 520, 510, 680))
    result, _ = await walk(game, FakeMuse(game, via=[(480, 480)]))
    assert result.outcome == "arrived", result.text()


async def test_lets_go_while_asking_again_and_a_waypoint_isnt_being_stuck():
    game = Game(start=(150, 600), goal=(800, 600), wall=(450, 450, 510, 760), speed=3.0)
    result, _ = await walk(game, FakeMuse(game, via=[(480, 380)]), timeout=40)
    assert result.outcome == "arrived", result.text()
    assert len(game.asked) == 1, game.asked  # reaching the waypoint isn't "no progress"
    game2 = Game(start=(150, 600), goal=(800, 600), wall=(450, 0, 510, 1400))
    await walk(game2, timeout=12)
    assert len(game2.asked) > 1 and all(s is None for s in game2.held_when_asked)


def test_a_box_at_the_screen_edge_is_followed_sensibly():
    rng = np.random.default_rng(3)
    scene = (rng.random((270, 480, 3)) * 80).astype(np.uint8)
    scene[100:130, 0:12] = (200, 30, 30)
    look = Look(scene, (-5, 100, 15, 130))
    assert look.usable and look.base > 0.05
    x, _, score = look.find(scene, (5, 115), radius=20)
    assert x < 20 and 0.8 < score < 1.5
    blank = Look(scene, (300, 200, 320, 230))  # just background: nothing to follow
    assert not blank.usable or blank.find(scene, (310, 215), 10)[2] < 2


def test_keyboards_walk_with_the_arrow_keys(monkeypatch):
    import subprocess

    from playertwo.controls import WaylandKeys

    started, ran = [], []

    class Proc:
        def __init__(self, args):
            started.append(args)

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", Proc)
    keys = WaylandKeys()
    monkeypatch.setattr(keys, "_run", lambda *a: ran.append(a))
    keys.steer(135)
    keys.steer(135)  # the same way: nothing new
    keys.steer(None)
    assert len(started) == 1 and "Down" in started[0] and "Right" in started[0]
    assert ran == [("-p", "Down", "-p", "Right")]


async def test_a_text_box_or_battle_popping_up_stops_the_walk():
    game = Game()
    real = game.grab

    async def popup():
        img = await real()
        if game.frames >= 10:
            ImageDraw.Draw(img).rectangle((40, 380, 920, 520), fill=(30, 30, 120))
        return img

    w = Walker(FakeMuse(game), popup, game.steer)
    result = await w.walk("the green chest")
    assert result.outcome == "interrupted" and game.stick is None
