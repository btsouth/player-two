"""The game window's capture: a window that crashed grim isn't captured again straight away
(one crash per close); a slow capture of a window still open is just tried again."""
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
    monkeypatch.setattr(screen, "hypr_clients", lambda: [win])  # still listed: closing

    cap.grab()  # window capture fails, the monitor is grabbed instead
    assert sum("-T" in c for c in calls) == 1
    for _ in range(5):
        clock[0] += 0.5
        cap.grab()
    assert sum("-T" in c for c in calls) == 1  # not again within 5 s
    clock[0] += 5.0
    cap.grab()
    assert sum("-T" in c for c in calls) == 2


def test_gone_window_is_noticed_at_once_and_never_captured_again(monkeypatch):
    calls: list[list[str]] = []

    def run(cmd, **kw):
        calls.append(cmd)
        if "-T" in cmd:
            raise subprocess.CalledProcessError(-11, cmd)
        return _ppm()

    monkeypatch.setattr(screen.subprocess, "check_output", run)
    monkeypatch.setattr(screen, "monitor_name", lambda i: "DP-2")
    monkeypatch.setattr(screen, "hypr_clients", list)  # closed: an empty client list
    cap = screen.HyprlandCapture()
    picks = iter([{"stableId": "abc", "monitor": 1}, None, None])
    monkeypatch.setattr(cap, "pick", lambda: next(picks))
    for _ in range(2):
        try:
            cap.grab()
        except screen.NoGame:
            continue
        raise AssertionError("expected NoGame once the window is gone")
    assert sum("-T" in c for c in calls) == 1  # one crash for the close
    assert not any("-T" not in c for c in calls)  # and no monitor picture


def _window_then(monkeypatch, failures):
    """A window that captured fine, then fails as `failures` says (an exception or None
    for a picture), with a clock to move."""
    calls: list[list[str]] = []
    left = list(failures)

    def run(cmd, **kw):
        calls.append(cmd)
        if "-T" in cmd and left:
            exc = left.pop(0)
            if exc is not None:
                raise exc
        return _ppm()

    clock = [100.0]
    monkeypatch.setattr(screen.subprocess, "check_output", run)
    monkeypatch.setattr(screen.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(screen, "monitor_name", lambda i: "DP-2")
    win = {"stableId": "abc", "monitor": 1, "title": "CHRONO TRIGGER"}
    monkeypatch.setattr(screen, "hypr_clients", lambda: [win])
    cap = screen.HyprlandCapture()
    monkeypatch.setattr(cap, "pick", lambda: win)
    return cap, calls, clock


def test_one_slow_capture_is_tried_again_not_five_seconds_of_the_monitor(monkeypatch):
    """Live review: one grim timeout (a busy machine, recording) gave 5 s of whole-monitor
    frames with no label, which the reader misread and walks got lost on."""
    slow = subprocess.TimeoutExpired(["grim"], 3)
    cap, calls, clock = _window_then(monkeypatch, [None, slow])
    assert cap.grab().label == "CHRONO TRIGGER"
    clock[0] += 0.5
    assert cap.grab().label == "CHRONO TRIGGER"  # tried again at once: the window
    for _ in range(4):
        clock[0] += 0.5
        assert cap.grab().label == "CHRONO TRIGGER"
    assert all("-T" in c for c in calls)


def test_two_failures_in_a_row_give_one_look_at_the_monitor_then_the_window(monkeypatch):
    err = subprocess.CalledProcessError(1, ["grim"])
    cap, _calls, clock = _window_then(monkeypatch, [None, err, err])
    assert cap.grab().label == "CHRONO TRIGGER"
    clock[0] += 0.5
    assert cap.grab().label == ""  # this one look: the monitor
    clock[0] += 0.5
    assert cap.grab().label == "CHRONO TRIGGER"  # and the window straight after


def test_a_hyprland_that_cant_capture_a_window_shows_the_monitor_without_retrying(
        monkeypatch):
    """Older Hyprland: a window capture never works; the monitor it's on is shown, and the
    window is only tried again after a while, not twice for every picture."""
    err = subprocess.CalledProcessError(1, ["grim"])
    cap, calls, clock = _window_then(monkeypatch, [err] * 10)
    for _ in range(6):
        assert cap.grab().label == ""
        clock[0] += 0.5
    assert sum("-T" in c for c in calls) == 1



def test_a_frame_is_decoded_before_two_threads_read_it(monkeypatch):
    """Live 3: "what does it look like?" hands the new frame to the model and to the room
    match at once; a lazily opened picture read by two threads at the same time came out
    "image file is truncated" for both, every time."""
    import threading

    big = io.BytesIO()
    Image.new("RGB", (1280, 720), (40, 80, 120)).save(big, "PPM")
    monkeypatch.setattr(screen.subprocess, "check_output", lambda cmd, **kw: big.getvalue())
    monkeypatch.setattr(screen, "monitor_name", lambda i: "DP-2")
    win = {"stableId": "abc", "monitor": 1, "title": "CHRONO TRIGGER"}
    monkeypatch.setattr(screen, "hypr_clients", lambda: [win])
    cap = screen.HyprlandCapture()
    monkeypatch.setattr(cap, "pick", lambda: win)
    for _ in range(5):
        frame = cap.grab()
        errors: list[str] = []

        def read(img=frame.image):
            try:
                img.convert("L").resize((64, 36))
            except OSError as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=read) for _ in range(2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errors


def test_window_capture_runs_without_core_dumps(monkeypatch):
    """A failed window capture segfaults grim 1.5; it mustn't put a crash notice on the
    desktop. The monitor capture has no such bug and runs as it is."""
    monkeypatch.setattr(screen.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert screen._no_core_dump(["grim", "-T", "x", "-"])[:2] == ["prlimit", "--core=1:"]
    monkeypatch.setattr(screen.shutil, "which", lambda name: None)
    assert screen._no_core_dump(["grim", "-T", "x", "-"]) == ["grim", "-T", "x", "-"]


def test_no_core_dump_really_sets_the_limit_and_keeps_the_signal():
    import shutil

    if not shutil.which("prlimit"):
        return
    out = subprocess.check_output(screen._no_core_dump(["grep", "core", "/proc/self/limits"]))
    assert out.split()[4] == b"1"  # the soft limit
    done = subprocess.run(screen._no_core_dump(["sh", "-c", "kill -SEGV $$"]), check=False)
    assert done.returncode == -11  # seen as a crash, the same as without the wrapper
