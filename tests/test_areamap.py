"""The area map: frames stitched where the camera was, people walking about left out."""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from playertwo.areamap import UNSEEN, AreaMap
from playertwo.walk import phase_shift, small


def texture(w, h, seed=3):
    rng = np.random.default_rng(seed)
    noise = (rng.random((h // 8, w // 8)) * 255).astype(np.uint8)
    img = Image.fromarray(noise).resize((w, h), Image.BICUBIC)
    return img.filter(ImageFilter.GaussianBlur(2)).convert("RGB")


class World:
    """A top-down world bigger than the screen, the camera on the character (as in
    test_explore), with a wall, someone walking about and a fixed-camera room."""

    VIEW = (960, 540)

    def __init__(self, start=(1200, 700), wall=None, room=False, speed=7.0, walker=None):
        self.world = texture(2400, 1400)
        if wall:
            ImageDraw.Draw(self.world).rectangle(wall, fill=(40, 40, 40))
        self.pos = np.array(start, float)
        self.wall = wall
        self.room = room
        self.speed = speed
        self.walker = walker  # (x, y) of someone pacing left and right, in world pixels
        self.t = 0

    def camera(self):
        w, h = self.VIEW
        if self.room:
            return np.array([700.0, 400.0])
        return np.array([min(max(self.pos[0] - w / 2, 0), self.world.width - w),
                         min(max(self.pos[1] - h / 2, 0), self.world.height - h)])

    def walker_at(self):
        x, y = self.walker
        return np.array([x + 120 * math.sin(self.t / 6), y])

    def step(self, angle):
        """Walk one frame toward `angle` (None stands still) and grab the screen."""
        self.t += 1
        if angle is not None:
            a = math.radians(angle)
            nxt = self.pos + np.array([math.sin(a), -math.cos(a)]) * self.speed
            x0, y0, x1, y1 = self.wall or (0, 0, -1, -1)
            if not (x0 - 12 <= nxt[0] <= x1 + 12 and y0 - 18 <= nxt[1] <= y1 + 18):
                self.pos = nxt
        w, h = self.VIEW
        cam = self.camera()
        view = self.world.crop((int(cam[0]), int(cam[1]), int(cam[0]) + w, int(cam[1]) + h))
        d = ImageDraw.Draw(view)
        if self.walker:
            bx, by = self.walker_at() - cam
            d.rectangle((bx - 12, by - 18, bx + 12, by + 18), fill=(30, 30, 220))
        px, py = self.pos - cam
        d.rectangle((px - 12, py - 18, px + 12, py + 18), fill=(200, 30, 30))
        d.rectangle((px - 6, py - 14, px + 6, py - 4), fill=(250, 250, 250))
        return view

    def truth(self):
        """The world without anyone in it, at tracking size."""
        return np.asarray(self.world.resize((1200, 700), Image.BILINEAR), dtype=np.float32)


def walk(world, area, moves, bias=(0.0, 0.0)):
    """Walk the moves ((angle, frames), ...), feeding each frame to the map with the
    scroll phase_shift measures (plus `bias`, a made-up error each frame). Returns the
    first camera position, at tracking size."""
    cam0 = world.camera() / 2
    rgb, prev = small(world.step(None))
    area.add(rgb, (0.0, 0.0), prev)
    for angle, n in moves:
        for _ in range(n):
            rgb, g = small(world.step(angle))
            at = area.to_screen(area.to_map((240, 135)))  # the camera keeps it in the middle
            mask = (int(at[0] - 16), int(at[1] - 20), int(at[0] + 16), int(at[1] + 20))
            sx, sy, _ = phase_shift(prev, g, None if world.room else mask)
            area.add(rgb, (sx + bias[0], sy + bias[1]), g)
            prev = g
    return cam0


def error(area, world, cam0, away=()):
    """Mean gray difference between the map and the empty world under it, leaving out
    the map's rim (half-pixel steps) and boxes `away` (x0, y0, x1, y1 in map pixels)."""
    pic = area.picture().astype(np.float32).mean(axis=2)
    x0, y0, _, _ = area.bounds()
    wx, wy = round(cam0[0] + x0), round(cam0[1] + y0)
    truth = world.truth().mean(axis=2)[wy:wy + pic.shape[0], wx:wx + pic.shape[1]]
    keep = np.zeros(pic.shape, bool)
    keep[3:-3, 3:-3] = True
    keep &= (area.picture() != UNSEEN).any(axis=2)
    for bx0, by0, bx1, by1 in away:
        keep[max(0, by0 - y0):max(0, by1 - y0), max(0, bx0 - x0):max(0, bx1 - x0)] = False
    return float(np.abs(pic - truth)[keep].mean())


def red(area):
    pic = area.picture().astype(int)
    return (pic[..., 0] > 150) & (pic[..., 1] < 90) & (pic[..., 2] < 90)


def test_walking_stitches_the_world_without_the_character_smeared():
    world = World()
    area = AreaMap()
    cam0 = walk(world, area, [(90, 40), (180, 30), (270, 20)])
    x0, y0, x1, y1 = area.bounds()
    # It walked 280 px right, 210 down and 140 back left in the world: half that here.
    assert abs((x1 - x0) - (480 + 140)) <= 2 and abs((y1 - y0) - (270 + 105)) <= 2
    truth_cam = world.camera() / 2 - cam0
    assert np.hypot(*(area.camera - truth_cam)) < 1.0
    here = area.to_map((240, 135))
    near = (int(here[0]) - 20, int(here[1]) - 20, int(here[0]) + 20, int(here[1]) + 20)
    assert error(area, world, cam0, [near]) < 4
    # The character shows where it is now, and nowhere it walked through.
    r = red(area)
    ys, xs = np.nonzero(r)
    far = np.hypot(xs + x0 - here[0], ys + y0 - here[1]) > 20
    assert far.sum() < 20


def test_a_wall_lands_where_it_is():
    world = World(wall=(1500, 500, 1540, 900))
    area = AreaMap()
    cam0 = walk(world, area, [(90, 60)])  # into the wall, which stops it
    assert world.pos[0] < 1500 - 11
    # The wall, at tracking size, in map coordinates.
    wx0, wy0 = 1500 / 2 - cam0[0], 500 / 2 - cam0[1]
    wall = area.picture()[int(wy0) + 20 - area.bounds()[1]:int(wy0) + 120 - area.bounds()[1],
                          int(wx0) + 3 - area.bounds()[0]:int(wx0) + 17 - area.bounds()[0]]
    assert np.abs(wall.astype(int) - 40).max() <= 6
    assert error(area, world, cam0, [(*(area.to_map((220, 115)).astype(int)),
                                      *(area.to_map((260, 155)).astype(int)))]) < 4


def test_someone_walking_about_does_not_smear():
    world = World(walker=(1250, 600))
    area = AreaMap()
    cam0 = walk(world, area, [(90, 40), (None, 20), (270, 40)])
    pic = area.picture().astype(int)
    blue = (pic[..., 2] > 180) & (pic[..., 0] < 80) & (pic[..., 1] < 80)
    # At most about one of them (24 x 36 in the world, 12 x 18 here) where it is now.
    assert blue.sum() < 12 * 18 * 1.5
    w = world.walker_at() / 2 - cam0
    here = area.to_map((240, 135))
    away = [(int(w[0]) - 10, int(w[1]) - 13, int(w[0]) + 10, int(w[1]) + 13),
            (int(here[0]) - 20, int(here[1]) - 20, int(here[0]) + 20, int(here[1]) + 20)]
    assert error(area, world, cam0, away) < 4


def test_a_fixed_camera_gives_the_screen():
    world = World(room=True, start=(800, 700))
    area = AreaMap()
    cam0 = walk(world, area, [(90, 40), (0, 20), (270, 30)])
    assert area.bounds() == (0, 0, 480, 270)
    assert np.abs(area.camera).max() < 0.5
    # The room as it is, the character where it is now and nowhere else.
    here = world.pos / 2 - cam0
    near = (int(here[0]) - 12, int(here[1]) - 15, int(here[0]) + 12, int(here[1]) + 15)
    assert error(area, world, cam0, [near]) < 3
    r = red(area)
    ys, xs = np.nonzero(r)
    assert (np.hypot(xs - here[0], ys - here[1]) > 15).sum() < 10


def test_checking_against_the_map_stops_drift():
    world = World()
    area = AreaMap()
    # Each scroll a third of a pixel off: 30 px off after 90 frames if trusted.
    cam0 = walk(world, area, [(90, 30), (180, 30), (270, 30)], bias=(0.33, -0.33))
    truth_cam = world.camera() / 2 - cam0
    assert np.hypot(*(area.camera - truth_cam)) < 2.0
    here = area.to_map((240, 135))
    near = (int(here[0]) - 20, int(here[1]) - 20, int(here[0]) + 20, int(here[1]) + 20)
    assert error(area, world, cam0, [near]) < 6


def test_positions_path_and_picture():
    world = World()
    area = AreaMap()
    walk(world, area, [(90, 10)])
    p = area.to_map((100, 50))
    assert np.allclose(area.to_screen(p), (100, 50))
    assert area.on_screen(p) and not area.on_screen(np.add(p, (480, 0)))
    assert area.seen(p) and not area.seen(np.add(p, (2000, 0)))
    area.character_at((240, 135))
    area.character_at((240, 135))
    walk_more = [(180, 10)]
    for angle, n in walk_more:
        for _ in range(n):
            rgb, g = small(world.step(angle))
            area.add(rgb, (0.0, -3.5), g)
    area.character_at((240, 135))
    assert len(area.path) == 2 and area.path[1][1] - area.path[0][1] > 30
    img = area.image(marks={"gate": (300, 200)}, size=1024)
    assert max(img.size) == 1024
    back = area.from_image(((300 - area.bounds()[0]) * area._fit[2],
                            (200 - area.bounds()[1]) * area._fit[2]))
    assert np.allclose(back, (300, 200))
    a = np.asarray(img)
    # Unseen corners are flat grey (it walked right, then down).
    assert (a == UNSEEN).all(axis=2).sum() > 1000


def fade(a, b, steps):
    """Frames fading from a to black and up to b, over `steps` frames each way, as
    SNES games do through a door."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    out = [a * (1 - k / steps) for k in range(1, steps + 1)]
    out += [np.zeros_like(a)] * 4
    out += [b * (k / steps) for k in range(1, steps + 1)]
    return [f.astype(np.uint8) for f in out]


def test_track_measures_scroll_and_starts_over_in_a_new_place():
    world = World()
    area = AreaMap()
    t = 0.0
    for _ in range(10):
        t += 0.04
        assert not area.track(*small(world.step(90)), now=t)
    assert area.camera[0] > 25
    # Through a door that fades out and in, each frame changing too little to tell.
    last = small(world.step(None))[0]
    world.world = texture(2400, 1400, seed=11)
    there = small(world.step(None))[0]
    fresh = []
    for f in [*fade(last, there, 8), *[there] * 6]:
        t += 0.04
        fresh.append(area.track(f, now=t))
    assert fresh.count(True) == 1 and fresh[-1] is False
    # Nothing of the fade or the old place is on the new map.
    assert area.bounds() == (0, 0, 480, 270)
    assert np.abs(area.picture().astype(int) - there.astype(int)).mean() < 3


def test_a_flash_is_not_a_new_place():
    world = World()
    area = AreaMap()
    t = 0.0
    for _ in range(10):
        t += 0.04
        area.track(*small(world.step(90)), now=t)
    here = small(world.step(None))[0]
    seen = area.bounds()
    for f in [*fade(here, here, 3), *[here] * 6]:
        t += 0.04
        assert not area.track(f, now=t)
    assert area.bounds() == seen


def test_a_text_box_after_a_gap_keeps_the_map():
    world = World()
    area = AreaMap()
    walk(world, area, [(90, 40)])
    seen = area.bounds()
    rgb = small(world.step(None))[0].copy()
    rgb[180:260, 20:460] = (20, 20, 120)  # a text box over the bottom third
    assert not area.resume(rgb)
    assert area.bounds() == seen
    # and the box isn't drawn into the map
    pic = area.picture()
    assert not ((pic[..., 2] == 120) & (pic[..., 0] == 20)).any()


def test_after_a_gap_it_finds_the_screen_on_the_map_or_starts_over():
    world = World()
    area = AreaMap()
    cam0 = walk(world, area, [(90, 60), (180, 30), (270, 60)])
    # Moved by hand while nobody watched: back where it came in, further than any
    # frame-to-frame scroll.
    world.pos = np.array([1200.0, 700.0])
    assert np.hypot(*(world.camera() / 2 - cam0 - area.camera)) > 60
    assert not area.resume(*small(world.step(None)))
    assert np.hypot(*(area.camera - (world.camera() / 2 - cam0))) < 2
    # Through a door: somewhere else.
    world.world = texture(2400, 1400, seed=11)
    assert area.resume(*small(world.step(None)))
    assert area.frames == 1


def test_a_frame_of_another_size_starts_a_new_map_instead_of_failing():
    world = World()
    area = AreaMap()
    for _ in range(5):
        area.track(*small(world.step(90)))
    taller = np.zeros((360, 480, 3), np.uint8)
    assert area.track(taller)
    assert area.resume(np.zeros((270, 480, 3), np.uint8))
