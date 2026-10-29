from playertwo.controls import WindowsKeys


def test_windows_keys_steer_releases_on_close():
    # Exercise held keys without needing SendInput on the test machine.
    keys = WindowsKeys.__new__(WindowsKeys)
    keys.map = {"up": "Up", "right": "Right", "down": "Down", "left": "Left"}
    keys.focus = None
    events = []
    keys._key = lambda name, up: events.append((name, up))

    keys.steer(45)
    assert events == [("Up", False), ("Right", False)]
    events.clear()
    keys.steer(90)
    assert events == [("Up", True)]
    events.clear()
    keys.close()
    assert events == [("Right", True)]
    assert keys._held == []
    events.clear()
    keys.close()
    assert events == []


def test_held_wayland_keys_are_let_go_on_close():
    import subprocess

    from playertwo.controls import WaylandKeys

    keys = WaylandKeys()
    ran = []
    keys._run = lambda *args: ran.append(args)
    holder = subprocess.Popen(["sleep", "30"])  # stands in for "wtype -P Up -s 60000"
    keys._held = (["Up"], holder)
    keys.close()
    assert holder.poll() is not None and keys._held is None
    assert ran == [("-p", "Up")]
    keys.close()  # nothing held: nothing to do
    assert ran == [("-p", "Up")]
