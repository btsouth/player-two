"""Windows windows: listing them, giving one focus, and copying one's pixels.

The list is shaped like Hyprland's clients, so games.pick_game picks the game the same
way on both: `class` is the program's name ("HollowKnight", or `steam_app_<id>` for a
game in a Steam library), `focusHistoryID` the window's place in the stacking order,
`fullscreen` 2 when it covers its monitor, and `game_dir` when the program lives in a
games folder (Epic, GOG, Xbox, C:\\Games).
"""

from __future__ import annotations

import contextlib
import ctypes
import os
from ctypes import wintypes
from pathlib import Path

from PIL import Image

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi")

HWND, HANDLE, HDC, DWORD, BOOL = wintypes.HWND, wintypes.HANDLE, wintypes.HDC, wintypes.DWORD, wintypes.BOOL
WNDENUMPROC = ctypes.WINFUNCTYPE(BOOL, HWND, wintypes.LPARAM)


def _sig(fn, restype, *argtypes):
    fn.restype, fn.argtypes = restype, list(argtypes)


_sig(user32.EnumWindows, BOOL, WNDENUMPROC, wintypes.LPARAM)
_sig(user32.IsWindowVisible, BOOL, HWND)
_sig(user32.IsIconic, BOOL, HWND)
_sig(user32.IsWindow, BOOL, HWND)
_sig(user32.GetWindow, HWND, HWND, wintypes.UINT)
_sig(user32.GetWindowLongW, wintypes.LONG, HWND, ctypes.c_int)
_sig(user32.GetWindowTextLengthW, ctypes.c_int, HWND)
_sig(user32.GetWindowTextW, ctypes.c_int, HWND, wintypes.LPWSTR, ctypes.c_int)
_sig(user32.GetClassNameW, ctypes.c_int, HWND, wintypes.LPWSTR, ctypes.c_int)
_sig(user32.GetWindowThreadProcessId, DWORD, HWND, ctypes.POINTER(DWORD))
_sig(user32.GetForegroundWindow, HWND)
_sig(user32.SetForegroundWindow, BOOL, HWND)
_sig(user32.BringWindowToTop, BOOL, HWND)
_sig(user32.ShowWindow, BOOL, HWND, ctypes.c_int)
_sig(user32.AttachThreadInput, BOOL, DWORD, DWORD, BOOL)
_sig(user32.GetWindowRect, BOOL, HWND, ctypes.POINTER(wintypes.RECT))
_sig(user32.MonitorFromWindow, HANDLE, HWND, DWORD)
_sig(user32.GetWindowDC, HDC, HWND)
_sig(user32.ReleaseDC, ctypes.c_int, HWND, HDC)
_sig(user32.PrintWindow, BOOL, HWND, HDC, wintypes.UINT)
_sig(kernel32.OpenProcess, HANDLE, DWORD, BOOL, DWORD)
_sig(kernel32.CloseHandle, BOOL, HANDLE)
_sig(kernel32.QueryFullProcessImageNameW, BOOL, HANDLE, DWORD, wintypes.LPWSTR,
     ctypes.POINTER(DWORD))
_sig(kernel32.GetCurrentThreadId, DWORD)
_sig(kernel32.GetConsoleWindow, HWND)
_sig(gdi32.CreateCompatibleDC, HDC, HDC)
_sig(gdi32.CreateCompatibleBitmap, wintypes.HBITMAP, HDC, ctypes.c_int, ctypes.c_int)
_sig(gdi32.SelectObject, wintypes.HGDIOBJ, HDC, wintypes.HGDIOBJ)
_sig(gdi32.DeleteObject, BOOL, wintypes.HGDIOBJ)
_sig(gdi32.DeleteDC, BOOL, HDC)
_sig(gdi32.GetDIBits, ctypes.c_int, HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
     ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT)
_sig(dwmapi.DwmGetWindowAttribute, ctypes.c_long, HWND, DWORD, ctypes.c_void_p, DWORD)


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                ("dwFlags", DWORD)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
                ("biCompression", DWORD), ("biSizeImage", DWORD),
                ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", DWORD), ("biClrImportant", DWORD)]


_sig(user32.GetMonitorInfoW, BOOL, HANDLE, ctypes.POINTER(MONITORINFO))

GW_OWNER = 4
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x80
DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
MONITOR_DEFAULTTONEAREST = 2
SW_RESTORE = 9
PW_RENDERFULLCONTENT = 2
GAME_FOLDERS = {"steamapps", "epic games", "gog games", "gog galaxy", "xboxgames",
                "games", "itch", "ea games", "ubisoft game launcher", "riot games"}


def dpi_aware() -> None:
    """Window sizes and screen pixels in the same (physical) units, on any scaling."""
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per monitor v2
    except (AttributeError, OSError):
        with contextlib.suppress(AttributeError, OSError):
            ctypes.windll.shcore.SetProcessDpiAwareness(2)


def _text(fn, hwnd, size: int = 512) -> str:
    buf = ctypes.create_unicode_buffer(size)
    fn(hwnd, buf, size)
    return buf.value


def window_pid(hwnd) -> int:
    pid = DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def exe_path(pid: int) -> str:
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ""  # some anti-cheat protected games refuse even this
    try:
        buf = ctypes.create_unicode_buffer(1024)
        n = DWORD(len(buf))
        return buf.value if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) else ""
    finally:
        kernel32.CloseHandle(h)


def frame_rect(hwnd) -> tuple[int, int, int, int] | None:
    """The window's visible rectangle, without the invisible resize borders."""
    r = wintypes.RECT()
    dwm = dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(r),
                                       ctypes.sizeof(r))
    if dwm != 0 and not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        return None
    return r.left, r.top, r.right, r.bottom


def monitor_rect(hwnd) -> tuple[int, int, int, int] | None:
    info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
    mon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    if not mon or not user32.GetMonitorInfoW(mon, ctypes.byref(info)):
        return None
    m = info.rcMonitor
    return m.left, m.top, m.right, m.bottom


def _cloaked(hwnd) -> bool:
    """Hidden by Windows itself: suspended Store apps, other virtual desktops."""
    v = DWORD()
    ok = dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(v), ctypes.sizeof(v))
    return ok == 0 and v.value != 0


def in_game_folder(path: str) -> bool:
    parts = {p.lower() for p in Path(path).parts[:-1]}
    return bool(parts & GAME_FOLDERS)


def clients() -> list[dict]:
    """Top-level app windows, front to back, shaped like Hyprland's clients."""
    from .games import steam_app_for_path

    own = {kernel32.GetConsoleWindow() or 0}
    fg = user32.GetForegroundWindow() or 0
    found: list[int] = []

    @WNDENUMPROC
    def each(hwnd, _):
        found.append(hwnd or 0)
        return True

    user32.EnumWindows(each, 0)
    out = []
    for hwnd in found:
        if hwnd in own or not user32.IsWindowVisible(hwnd) or user32.GetWindow(hwnd, GW_OWNER):
            continue
        if user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW or _cloaked(hwnd):
            continue
        rect = frame_rect(hwnd)
        minimized = bool(user32.IsIconic(hwnd))
        if rect is None or (not minimized and (rect[2] - rect[0] < 64 or rect[3] - rect[1] < 64)):
            continue
        pid = window_pid(hwnd)
        if pid == os.getpid():
            continue
        path = exe_path(pid)
        app_id = steam_app_for_path(path) if path else ""
        cls = f"steam_app_{app_id}" if app_id else (Path(path).stem if path
                                                   else _text(user32.GetClassNameW, hwnd))
        mon = monitor_rect(hwnd)
        out.append({
            "class": cls, "title": _text(user32.GetWindowTextW, hwnd),
            "focusHistoryID": -1 if hwnd == fg else len(out), "mapped": True,
            "fullscreen": 2 if mon and rect == mon else 0,
            "game_dir": bool(path) and in_game_folder(path),
            "address": str(hwnd), "hwnd": hwnd, "path": path, "minimized": minimized,
        })
    return out


def alive(hwnd: int) -> bool:
    return bool(user32.IsWindow(hwnd))


def minimized(hwnd: int) -> bool:
    return bool(user32.IsIconic(hwnd))


def foreground() -> int:
    return user32.GetForegroundWindow() or 0


def focus(hwnd: int) -> None:
    """Bring a window to the front with keyboard focus. Windows only lets the program
    that got the last input do this, so borrow the front window's input for a moment."""
    if minimized(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    fg = foreground()
    if fg == hwnd:
        return
    me = kernel32.GetCurrentThreadId()
    them = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    attached = bool(them) and them != me and user32.AttachThreadInput(me, them, True)
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(me, them, False)


def print_window(hwnd: int) -> Image.Image | None:
    """The window's own pixels, even under other windows (most DirectX games included).
    None when Windows can't give them."""
    outer = wintypes.RECT()
    inner = frame_rect(hwnd)
    if not user32.GetWindowRect(hwnd, ctypes.byref(outer)) or inner is None:
        return None
    w, h = outer.right - outer.left, outer.bottom - outer.top
    if w <= 0 or h <= 0:
        return None
    wdc = user32.GetWindowDC(hwnd)
    if not wdc:
        return None
    mem = gdi32.CreateCompatibleDC(wdc)
    bmp = gdi32.CreateCompatibleBitmap(wdc, w, h)
    old = gdi32.SelectObject(mem, bmp)
    try:
        if not user32.PrintWindow(hwnd, mem, PW_RENDERFULLCONTENT):
            return None
        gdi32.SelectObject(mem, old)  # GetDIBits needs the bitmap deselected
        old = None
        bmi = BITMAPINFOHEADER(biSize=ctypes.sizeof(BITMAPINFOHEADER), biWidth=w,
                               biHeight=-h, biPlanes=1, biBitCount=32)
        buf = ctypes.create_string_buffer(w * h * 4)
        if gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bmi), 0) != h:
            return None
    finally:
        if old is not None:
            gdi32.SelectObject(mem, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(hwnd, wdc)
    img = Image.frombuffer("RGB", (w, h), buf, "raw", "BGRX", 0, 1)
    box = (inner[0] - outer.left, inner[1] - outer.top,
           inner[2] - outer.left, inner[3] - outer.top)
    return img.crop(box) if box != (0, 0, w, h) else img.copy()
