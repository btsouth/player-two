"""Screen capture that doesn't need a desktop: fake mss, fake time."""

import sys
import types

import pytest
from PIL import Image

from playertwo import screen
from playertwo.screen import BLANK, WAYLAND, CaptureError, MssCapture


class Shot:
    def __init__(self, color):
        img = Image.new("RGB", (64, 36), color)
        self.size = img.size
        self.rgb = img.tobytes()


def fake_mss(monkeypatch, color=(0, 0, 0), fail=False):
    class Sct:
        monitors = ({}, {"left": 0, "top": 0, "width": 64, "height": 36})

        def grab(self, mon):
            if fail:
                raise RuntimeError("XGetImage() failed")
            return Shot(color)

    monkeypatch.setitem(sys.modules, "mss", types.SimpleNamespace(mss=Sct))


@pytest.mark.parametrize(("wayland", "says"), [(True, WAYLAND), (False, BLANK)])
def test_black_whole_screen_says_how_to_fix_it(monkeypatch, wayland, says):
    fake_mss(monkeypatch)
    monkeypatch.setattr(screen.sys, "platform", "linux")
    monkeypatch.setattr(screen, "wayland", lambda: wayland)
    now = [100.0]
    monkeypatch.setattr(screen.time, "monotonic", lambda: now[0])
    cap = MssCapture()
    cap.grab()
    assert cap.notice == ""
    now[0] += cap.BLANK_S + 1
    cap.grab()
    assert cap.notice == says
    cap.notice = ""
    now[0] += 60
    cap.grab()
    assert cap.notice == ""  # once


def test_a_picture_is_never_blank(monkeypatch):
    fake_mss(monkeypatch, color=(90, 120, 60))
    now = [0.0]
    monkeypatch.setattr(screen.time, "monotonic", lambda: now[0])
    cap = MssCapture()
    for _ in range(3):
        now[0] += 30
        cap.grab()
    assert cap.notice == ""


def test_wayland_capture_failure_says_how_to_fix_it(monkeypatch):
    fake_mss(monkeypatch, fail=True)
    monkeypatch.setattr(screen.sys, "platform", "linux")
    monkeypatch.setattr(screen, "wayland", lambda: True)
    cap = MssCapture()  # opening it can't fail start-up
    with pytest.raises(CaptureError, match="X11"):
        cap.grab()


def test_mss_is_a_normal_dependency():
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    assert any(d.startswith("mss") for d in project["project"]["dependencies"])
