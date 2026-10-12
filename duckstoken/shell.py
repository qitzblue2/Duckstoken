"""Native desktop shell: the app window, the system-tray icon, start-with-Windows and the duck icon.

pywebview (the window), pystray (the tray) and Pillow (the icon bitmap) are all optional. Each one is imported only when
it is needed, and anything that goes wrong with it makes the app fall back instead of stopping it: without pywebview the
UI opens in the default browser, without pystray or Pillow there is simply no tray icon.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import threading
import webbrowser
from pathlib import Path

from . import bundle, paths

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
_WINDOW_SIZE = (1200, 800)
_MIN_WINDOW_SIZE = (900, 600)
_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_RUN_VALUE = "Duckstoken"  # the same name the installer's "start with Windows" task uses

# ------------------------------------------------------------------------------------------------------------- icon
_SUPERSAMPLE = 4  # the duck is drawn this many times larger and then scaled down, which anti-aliases the edges
_MARGIN = 0.06  # empty border around the duck, so the outline is never clipped
_BODY, _WING, _BEAK = (255, 208, 36, 255), (244, 170, 0, 255), (255, 122, 0, 255)
_OUTLINE, _EYE, _GLINT = (122, 70, 0, 255), (30, 24, 16, 255), (255, 255, 255, 255)
# Back-to-front shapes of a duck facing right, in unit coordinates: (kind, geometry, fill colour).
_DUCK = (
    ("polygon", ((0.07, 0.66), (0.02, 0.43), (0.31, 0.58)), _BODY),  # tail
    ("ellipse", (0.09, 0.46, 0.83, 0.93), _BODY),  # body
    ("ellipse", (0.40, 0.27, 0.82, 0.72), _BODY),  # neck
    ("ellipse", (0.78, 0.30, 0.99, 0.46), _BEAK),
    ("ellipse", (0.42, 0.07, 0.84, 0.49), _BODY),  # head
)
_WING_BOX = (0.24, 0.56, 0.62, 0.82)
_EYE_CENTRE, _EYE_RADIUS = (0.66, 0.25), 0.045


def icon_image(size: int = 64):
    """The Duckstoken duck as an RGBA Pillow image, drawn from shapes so no font, emoji or image file is needed."""
    from PIL import Image, ImageDraw  # Pillow is optional: callers that can live without an icon catch ImportError

    if not isinstance(size, int) or not 8 <= size <= 1024:
        raise ValueError("The icon size must be a whole number between 8 and 1024 pixels")
    n = size * _SUPERSAMPLE

    def px(*unit: float) -> tuple[float, ...]:
        return tuple((_MARGIN + v * (1 - 2 * _MARGIN)) * n for v in unit)

    pen = max(0.035 * n, 1.2 * _SUPERSAMPLE)  # outline thickness: 3.5% of the icon, never thinner than ~1 final pixel
    canvas = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for kind, geometry, _ in _DUCK:  # outlines first, so the shapes overlap without an outline line between them
        if kind == "ellipse":
            x0, y0, x1, y1 = px(*geometry)
            draw.ellipse((x0 - pen, y0 - pen, x1 + pen, y1 + pen), fill=_OUTLINE)
        else:
            points = [px(*p) for p in geometry]
            draw.polygon(points, fill=_OUTLINE)
            draw.line(points + points[:1], fill=_OUTLINE, width=round(2 * pen), joint="curve")
    for kind, geometry, fill in _DUCK:
        if kind == "ellipse":
            draw.ellipse(px(*geometry), fill=fill)
        else:
            draw.polygon([px(*p) for p in geometry], fill=fill)
    draw.ellipse(px(*_WING_BOX), fill=_WING)
    cx, cy = px(*_EYE_CENTRE)
    r = _EYE_RADIUS * (1 - 2 * _MARGIN) * n
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=_EYE)
    gx, gy, g = cx - 0.1 * r, cy - 0.45 * r, 0.4 * r  # the glint that makes the eye look alive
    draw.ellipse((gx - g, gy - g, gx + g, gy + g), fill=_GLINT)
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def write_icons(out_dir) -> list[Path]:
    """Writes duckstoken.ico (each size in ICON_SIZES drawn natively, not scaled) and a 256 px icon.png."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = [icon_image(s) for s in ICON_SIZES]
    ico, png = out / "duckstoken.ico", out / "icon.png"
    frames[-1].save(ico, format="ICO", sizes=[f.size for f in frames], append_images=frames[:-1])
    frames[-1].save(png, format="PNG", optimize=True)
    return [ico, png]


# -------------------------------------------------------------------------------------------------------- autostart
def _registry():
    """The winreg module on Windows, None elsewhere."""
    if os.name != "nt":
        return None
    import winreg

    return winreg


def _quote(path: str) -> str:
    if not path or any(c in path for c in '"\r\n\0'):
        raise ValueError(f"Cannot register {path!r} to start with Windows")
    return f'"{path}"'


def _autostart_command(exe_path: str | None) -> str:
    if exe_path:
        return f"{_quote(exe_path)} --minimized"
    if bundle.frozen():
        return f"{_quote(sys.executable)} --minimized"
    python = Path(sys.executable)  # running from source: start the same interpreter, without a console window
    windowless = python.with_name("pythonw.exe")
    interpreter = _quote(str(windowless if windowless.is_file() else python))
    launcher = bundle.app_dir() / "duckstoken_launcher.py"
    target = _quote(str(launcher)) if launcher.is_file() else "-m duckstoken"
    return f"{interpreter} {target} --minimized"


def set_autostart(enabled: bool, exe_path: str | None = None) -> bool:
    """Adds or removes Duckstoken's per-user "start when I sign in" entry; it always starts hidden (--minimized).

    exe_path defaults to the running program. Returns True when the registry now matches `enabled` (removing an entry
    that does not exist counts), False when that is impossible: not Windows, or the registry refused. Raises ValueError
    for an exe_path that cannot be quoted safely.
    """
    reg = _registry()
    if reg is None:
        return False
    command = _autostart_command(exe_path) if enabled else ""
    try:
        with reg.CreateKeyEx(reg.HKEY_CURRENT_USER, _RUN_KEY, 0, reg.KEY_SET_VALUE) as key:
            if enabled:
                reg.SetValueEx(key, _RUN_VALUE, 0, reg.REG_SZ, command)
            else:
                with contextlib.suppress(FileNotFoundError):  # already off
                    reg.DeleteValue(key, _RUN_VALUE)
    except OSError:
        return False
    return True


def autostart_enabled() -> bool:
    reg = _registry()
    if reg is None:
        return False
    try:
        with reg.OpenKey(reg.HKEY_CURRENT_USER, _RUN_KEY, 0, reg.KEY_READ) as key:
            reg.QueryValueEx(key, _RUN_VALUE)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------------------------------------- window
def choose_mode(settings: dict, have_webview: bool, headless: bool) -> str:
    """"headless" (serve only), "native" (own window) or "browser" (default web browser)."""
    if headless:
        return "headless"
    if settings.get("window_mode", "native") == "native" and have_webview:
        return "native"
    return "browser"


class Shell:
    """The window and tray around the UI served at `url`. Settings are read through `get_settings` whenever they matter.

    Besides the app settings it understands the one-off key "headless" (serve only: no window, tray or browser). `run`
    must be called on the main thread (pywebview requires it); the tray runs on its own thread, so `show`, `quit` and
    the tray menu are safe to use from any thread. `on_quit` is called exactly once, by `run`, after the window and
    tray are gone, whichever way the app ended (tray Quit, window closed for good, `quit()`, Ctrl+C).
    """

    def __init__(self, url: str, get_settings, on_quit, title: str = "Duckstoken", icon_path: str | None = None,
                 webview_module=None, tray_module=None):
        if not url.startswith(("http://", "https://")):
            raise ValueError("The UI address must be an http(s) URL")
        self._url, self._title, self._icon_path = url, title, icon_path
        self._get_settings, self._on_quit = get_settings, on_quit
        self._webview_module, self._tray_module = webview_module, tray_module
        self._window = None
        self._tray = None
        self._quitting = threading.Event()

    # ---------------------------------------------------------------------------------------------- public API
    def run(self) -> None:
        """Blocks until the app is quit: in its own window, else the default browser (also when the window fails)."""
        settings = self._get_settings()
        headless = bool(settings.get("headless"))
        webview = None if headless else self._load("webview", self._webview_module)
        mode = choose_mode(settings, webview is not None, headless)
        try:
            if mode != "headless":
                self._start_tray()
            hidden = bool(settings.get("start_minimized")) and self._tray is not None  # hidden needs a way back
            if mode == "native":
                if self._run_native(webview, hidden):
                    return
                open_ui = not hidden  # the window failed, so the browser takes over
            else:
                open_ui = mode == "browser" and not hidden and bool(settings.get("open_browser", True))
            if open_ui and not self._quitting.is_set():
                self._open_in_browser()
            while not self._quitting.wait(0.5):  # short waits keep Ctrl+C working on Windows
                pass
        finally:
            self.quit()
            self._on_quit()

    def show(self) -> None:
        """Brings the window back from the tray; without a window, opens the UI in the browser instead."""
        window = self._window
        if window is not None:
            try:
                window.show()
                return
            except Exception as exc:  # noqa: BLE001 - a broken window must not leave "Open" doing nothing
                self._note(f"Could not show the window ({exc}); opening the browser")
        self._open_in_browser()

    def quit(self) -> None:
        """Removes the tray icon and closes the window, which makes `run` return. Idempotent, any thread."""
        if self._quitting.is_set():
            return
        self._quitting.set()
        tray, self._tray = self._tray, None
        if tray is not None:
            try:
                tray.stop()
            except Exception as exc:  # noqa: BLE001
                self._note(f"Could not remove the tray icon ({exc})")
        window = self._window
        if window is not None:
            try:
                window.destroy()
            except Exception as exc:  # noqa: BLE001
                self._note(f"Could not close the window ({exc})")

    # ------------------------------------------------------------------------------------------------- internals
    def _load(self, name: str, injected):
        if injected is not None:
            return injected
        try:
            return importlib.import_module(name)
        except Exception:  # noqa: BLE001 - not installed, or its GUI toolkit is unusable here: that feature is just off
            return None

    def _run_native(self, webview, hidden: bool) -> bool:
        """Shows the window and blocks until it is gone. False if it could not be created or run."""
        try:
            self._window = webview.create_window(self._title, self._url, width=_WINDOW_SIZE[0], height=_WINDOW_SIZE[1],
                                                 min_size=_MIN_WINDOW_SIZE, hidden=hidden)
            self._window.events.closing += self._on_closing
            webview.start(**self._start_options())
        except Exception as exc:  # noqa: BLE001 - e.g. no WebView2 runtime: use the browser instead
            self._note(f"Native window unavailable ({exc}); using the browser")
            return False
        finally:
            self._window = None
        return True

    def _start_options(self) -> dict:
        """Keep the UI's local storage (chat history) between runs: pywebview forgets everything by default."""
        profile = paths.data_dir() / "webview"
        try:
            profile.mkdir(exist_ok=True)
        except OSError:
            return {}
        return {"private_mode": False, "storage_path": str(profile)}

    def _on_closing(self, *_) -> bool:
        """pywebview's closing handler (False cancels the close): hide instead when the tray can bring it back.

        pywebview keeps the GUI thread waiting for this verdict, so the window is hidden from another thread afterwards.
        """
        window = self._window
        if (window is None or self._quitting.is_set() or self._tray is None
                or not self._get_settings().get("minimize_to_tray", True)):
            return True
        threading.Thread(target=self._hide, args=(window,), daemon=True, name="hide-window").start()
        return False

    def _hide(self, window) -> None:
        try:
            window.hide()
        except Exception as exc:  # noqa: BLE001 - cannot hide: quit rather than leave a window that ignores its X
            self._note(f"Could not hide the window ({exc}); quitting")
            self.quit()

    def _start_tray(self) -> None:
        pystray = self._load("pystray", self._tray_module)
        image = self._tray_image() if pystray is not None else None
        if image is None:
            return
        try:
            menu = pystray.Menu(
                pystray.MenuItem("Open", lambda icon, item: self.show(), default=True),
                pystray.MenuItem("Open in browser", lambda icon, item: self._open_in_browser()),
                pystray.MenuItem("Quit", lambda icon, item: self.quit()))
            tray = pystray.Icon("duckstoken", image, self._title, menu)
            tray.run_detached()
        except Exception as exc:  # noqa: BLE001 - no tray is fine
            self._note(f"System tray unavailable ({exc})")
            return
        self._tray = tray

    def _tray_image(self):
        """The icon file if readable, else the drawn duck; None when Pillow is missing (then there is no tray)."""
        try:
            from PIL import Image
        except ImportError:
            return None
        if self._icon_path:
            try:
                with Image.open(self._icon_path) as picture:
                    return picture.convert("RGBA")
            except (OSError, ValueError):
                pass
        return icon_image(64)

    def _open_in_browser(self) -> None:
        try:
            webbrowser.open(self._url)
        except webbrowser.Error as exc:
            self._note(f"Could not open the browser ({exc})")

    @staticmethod
    def _note(message: str) -> None:
        print(message, file=sys.stderr, flush=True)  # lands in duckstoken.log for windowed builds
