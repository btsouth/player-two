"""The game window's capture: a window that fails once isn't captured again straight away."""
import io
import subprocess

from PIL import Image

from playertwo import screen


def _ppm() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (40, 80, 120)).save(buf, "PPM")
    return buf.getvalue()


def test_failed_window_capture_backs_off(monkeypatch):
    calls: list[list[str]] = []

    def run(cmd, **kw):
        calls.append(cmd)
        if "-T" in cmd:
            raise subprocess.CalledProcessError(-11, cmd)  # grim segfaulting
        return _ppm()

    clock = [100.0]
    monkeypatch.setattr(screen.subprocess, "check_output", run)
    monkeypatch.setattr(screen.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(screen, "monitor_name", lambda i: "DP-2")
    cap = screen.HyprlandCapture()
    win = {"stableId": "abc", "monitor": 1, "title": "CHRONO TRIGGER"}
    monkeypatch.setattr(cap, "pick", lambda: win)

    cap.grab()  # window capture fails, the monitor is grabbed instead
    assert sum("-T" in c for c in calls) == 1
    for _ in range(5):
        clock[0] += 0.5
        cap.grab()
    assert sum("-T" in c for c in calls) == 1  # not again within 5 s
    clock[0] += 5.0
    cap.grab()
    assert sum("-T" in c for c in calls) == 2


def test_gone_window_is_noticed_on_next_grab(monkeypatch):
    def run(cmd, **kw):
        if "-T" in cmd:
            raise subprocess.CalledProcessError(-11, cmd)
        return _ppm()

    monkeypatch.setattr(screen.subprocess, "check_output", run)
    monkeypatch.setattr(screen, "monitor_name", lambda i: "DP-2")
    cap = screen.HyprlandCapture()
    picks = iter([{"stableId": "abc", "monitor": 1}, None])
    monkeypatch.setattr(cap, "pick", lambda: next(picks))
    cap.grab()
    try:
        cap.grab()
    except screen.NoGame:
        return
    raise AssertionError("expected NoGame once the window is gone")
