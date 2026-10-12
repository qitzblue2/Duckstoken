"""Native shell: duck icon, autostart registry entry, mode choice and the window/tray behaviour.

pywebview, pystray and winreg are replaced by small fakes (nothing real is imported, opened or written), so these tests
describe the behaviour the shell expects from those libraries rather than exercising them.
"""
import importlib.util
import io
import os
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DUCKSTOKEN_HOME", tempfile.mkdtemp(prefix="duck-shell-"))

from duckstoken import shell  # noqa: E402

try:
    from PIL import Image, ImageChops, ImageStat
except ImportError:  # Pillow is optional: icon and tray tests are skipped without it
    Image = None
needs_pillow = unittest.skipUnless(Image, "Pillow is not installed")

URL = "http://127.0.0.1:8765/"


def wait_for(condition, timeout=5.0):
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            return False
        time.sleep(0.01)
    return True


# ------------------------------------------------------------------------------------------------------------ fakes
class FakeEvent:
    """pywebview's event: handlers are attached with +=, and a handler returning False vetoes the action."""

    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class FakeWindow:
    def __init__(self, hidden=False, hide_error=None, show_error=None, hide_gate=None):
        self.events = types.SimpleNamespace(closing=FakeEvent())
        self.hidden, self.hide_error, self.show_error, self.hide_gate = hidden, hide_error, show_error, hide_gate
        self.closed = threading.Event()
        self.destroy_calls = 0

    def hide(self):
        if self.hide_gate:
            self.hide_gate.wait(5)  # a hide() that needs the GUI thread, which is busy answering the closing event
        if self.hide_error:
            raise self.hide_error
        self.hidden = True

    def show(self):
        if self.show_error:
            raise self.show_error
        self.hidden = False

    def click_close(self) -> bool:
        """The user presses the window's X. True when the window really closed, False when a handler vetoed it."""
        if False in [handler(self) for handler in self.events.closing.handlers]:
            return False
        self.closed.set()
        return True

    def destroy(self):
        self.destroy_calls += 1
        self.click_close()  # like pywebview, destroying a window fires its closing event first


class FakeWebview:
    """Stands in for the `webview` module; `script(window)` plays the user's actions once start() is running."""

    def __init__(self, script=None, create_error=None, start_error=None, **window_options):
        self.script, self.create_error, self.start_error = script, create_error, start_error
        self.window_options = window_options
        self.windows, self.create_calls, self.start_kwargs = [], [], None
        self.start_timed_out = False

    def create_window(self, title, url, **kwargs):
        self.create_calls.append((title, url, kwargs))
        if self.create_error:
            raise self.create_error
        window = FakeWindow(hidden=kwargs.get("hidden", False), **self.window_options)
        self.windows.append(window)
        return window

    def start(self, **kwargs):
        """Blocks like pywebview does, until the window is gone."""
        self.start_kwargs = kwargs
        if self.start_error:
            raise self.start_error
        window = self.windows[0]
        if self.script:
            self.script(window)
        self.start_timed_out = not window.closed.wait(5)


class FakePystray:
    """Stands in for the `pystray` module."""

    def __init__(self, run_error=None):
        self.icons, self.run_error = [], run_error
        outer = self

        class MenuItem:
            def __init__(self, text, action, default=False):
                self.text, self.action, self.default = text, action, default

        class Menu:
            def __init__(self, *items):
                self.items = items

        class Icon:
            def __init__(self, name, image, title, menu):
                self.name, self.image, self.title, self.menu = name, image, title, menu
                self.running = self.stopped = False
                outer.icons.append(self)

            def run_detached(self):
                if outer.run_error:
                    raise outer.run_error
                self.running = True

            def stop(self):
                self.running, self.stopped = False, True

            def click(self, label):
                item = next(i for i in self.menu.items if i.text == label)
                item.action(self, item)

        self.Menu, self.MenuItem, self.Icon = Menu, MenuItem, Icon


class FakeRegistry:
    """A tiny winreg: one dict of values per key path."""

    HKEY_CURRENT_USER, KEY_SET_VALUE, KEY_READ, REG_SZ = "HKCU", 2, 3, 1

    class _Key:
        def __init__(self, values):
            self.values = values

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def __init__(self, error=None):
        self.keys, self.error = {}, error

    def CreateKeyEx(self, root, path, reserved, access):  # noqa: N802 - mirrors winreg
        if self.error:
            raise self.error
        return self._Key(self.keys.setdefault((root, path), {}))

    def OpenKey(self, root, path, reserved, access):  # noqa: N802
        if self.error:
            raise self.error
        if (root, path) not in self.keys:
            raise FileNotFoundError(path)
        return self._Key(self.keys[(root, path)])

    def SetValueEx(self, key, name, reserved, kind, data):  # noqa: N802
        key.values[name] = (data, kind)

    def DeleteValue(self, key, name):  # noqa: N802
        if name not in key.values:
            raise FileNotFoundError(name)
        del key.values[name]

    def QueryValueEx(self, key, name):  # noqa: N802
        if name not in key.values:
            raise FileNotFoundError(name)
        return key.values[name]


# ------------------------------------------------------------------------------------------------------------ icon
@needs_pillow
class IconTests(unittest.TestCase):
    def test_every_size_is_a_transparent_square_duck(self):
        for size in shell.ICON_SIZES:
            with self.subTest(size=size):
                img = shell.icon_image(size)
                self.assertEqual((img.size, img.mode), ((size, size), "RGBA"))
                alpha = img.getchannel("A")
                self.assertEqual(alpha.getpixel((0, 0)), 0)  # transparent corners
                self.assertEqual(alpha.getpixel((size - 1, size - 1)), 0)
                coverage = sum(1 for a in alpha.getdata() if a > 127) / (size * size)
                self.assertTrue(0.3 < coverage < 0.8, coverage)  # a body, not an empty or a filled square

    def test_duck_is_yellow_with_an_orange_beak_and_a_dark_eye(self):
        pixels = [p for p in shell.icon_image(128).getdata() if p[3] == 255]

        def near(colour, tolerance=20):
            return sum(1 for p in pixels if max(abs(a - b) for a, b in zip(p, colour, strict=True)) <= tolerance)

        self.assertGreater(near(shell._BODY), len(pixels) * 0.4)
        self.assertGreater(near(shell._BEAK), 50)
        self.assertGreater(near(shell._EYE), 20)

    def test_default_size_and_determinism(self):
        self.assertEqual(shell.icon_image().size, (64, 64))
        self.assertEqual(shell.icon_image(48).tobytes(), shell.icon_image(48).tobytes())

    def test_rejects_unreasonable_sizes(self):
        for bad in (0, 7, 1025, -64, 64.0, "64", None):
            with self.subTest(size=bad), self.assertRaises(ValueError):
                shell.icon_image(bad)

    def test_write_icons_produces_a_multi_size_ico_and_a_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "icons"
            ico, png = shell.write_icons(out)
            self.assertEqual((ico.name, png.name), ("duckstoken.ico", "icon.png"))
            self.assertEqual(sorted(entry[0] for entry in read_ico(ico)), sorted(shell.ICON_SIZES))
            with Image.open(png) as img:
                self.assertEqual((img.size, img.mode, img.format), ((256, 256), "RGBA", "PNG"))

    def test_make_icons_cli_writes_into_the_requested_folder_and_into_a_checkout(self):
        spec = importlib.util.spec_from_file_location("make_icons", ROOT / "assets" / "make_icons.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as tmp, mock.patch("sys.stdout", new_callable=io.StringIO):
            with mock.patch.object(sys, "argv", ["make_icons.py", "--out", str(Path(tmp) / "x")]):
                cli.main()
            self.assertTrue((Path(tmp) / "x" / "duckstoken.ico").is_file())
            self.assertTrue((Path(tmp) / "x" / "icon.png").is_file())
            checkout = Path(tmp) / "checkout"
            (checkout / "assets").mkdir(parents=True)
            (checkout / "duckstoken" / "web").mkdir(parents=True)
            with mock.patch.object(cli, "ROOT", checkout), mock.patch.object(sys, "argv", ["make_icons.py"]):
                cli.main()
            self.assertTrue((checkout / "assets" / "duckstoken.ico").is_file())
            self.assertTrue((checkout / "duckstoken" / "web" / "icon.png").is_file())

    def test_committed_icons_match_the_drawing(self):
        ico, png = ROOT / "assets" / "duckstoken.ico", ROOT / "duckstoken" / "web" / "icon.png"
        for size, frame in read_ico(ico):
            self.assertLess(mean_difference(frame, shell.icon_image(size)), 2.0, f"ico {size}px is stale")
        with Image.open(png) as img:
            self.assertEqual(img.size, (256, 256))
            self.assertLess(mean_difference(img.convert("RGBA"), shell.icon_image(256)), 2.0, "icon.png is stale")


def read_ico(path):
    """[(size, decoded image)] for every entry of an .ico, parsed by hand so the file layout itself is checked."""
    data = Path(path).read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data)
    assert (reserved, kind) == (0, 1), "not an .ico file"
    entries = []
    for i in range(count):
        width, height, _, _, _, bits, length, offset = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
        width, height = width or 256, height or 256
        assert bits == 32
        with Image.open(io.BytesIO(data[offset:offset + length])) as img:
            frame = img.convert("RGBA")
        assert frame.size == (width, height)
        entries.append((width, frame))
    return entries


def mean_difference(a, b) -> float:
    return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 4


# ------------------------------------------------------------------------------------------------------- autostart
class AutostartTests(unittest.TestCase):
    RUN_KEY = ("HKCU", r"Software\Microsoft\Windows\CurrentVersion\Run")

    def setUp(self):
        self.reg = FakeRegistry()
        patcher = mock.patch.object(shell, "_registry", return_value=self.reg)
        patcher.start()
        self.addCleanup(patcher.stop)

    def stored(self):
        return self.reg.keys.get(self.RUN_KEY, {}).get("Duckstoken")

    def test_enable_writes_a_quoted_minimized_command(self):
        exe = r"C:\Program Files\Duck stoken\Duckstoken.exe"
        self.assertTrue(shell.set_autostart(True, exe))
        self.assertEqual(self.stored(), (f'"{exe}" --minimized', FakeRegistry.REG_SZ))
        self.assertTrue(shell.autostart_enabled())

    def test_disable_removes_the_entry_and_is_idempotent(self):
        shell.set_autostart(True, r"C:\Duck\Duckstoken.exe")
        self.assertTrue(shell.set_autostart(False))
        self.assertIsNone(self.stored())
        self.assertFalse(shell.autostart_enabled())
        self.assertTrue(shell.set_autostart(False))  # nothing left to remove is still success

    def test_enabling_again_replaces_a_stale_path(self):
        shell.set_autostart(True, r"C:\Old\Duckstoken.exe")
        shell.set_autostart(True, r"C:\New\Duckstoken.exe")
        self.assertEqual(self.stored()[0], r'"C:\New\Duckstoken.exe" --minimized')

    def test_frozen_build_registers_itself(self):
        with mock.patch.object(shell.bundle, "frozen", return_value=True), \
                mock.patch.object(sys, "executable", r"C:\Apps\Duckstoken.exe"):
            self.assertTrue(shell.set_autostart(True))
        self.assertEqual(self.stored()[0], r'"C:\Apps\Duckstoken.exe" --minimized')

    def test_source_checkout_registers_the_interpreter_and_launcher(self):
        with mock.patch.object(shell.bundle, "frozen", return_value=False):
            self.assertTrue(shell.set_autostart(True))
        command = self.stored()[0]
        self.assertTrue(command.startswith('"') and command.endswith('duckstoken_launcher.py" --minimized'), command)
        self.assertIn(str(ROOT), command)

    def test_installed_package_registers_the_module(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(shell.bundle, "frozen", return_value=False), \
                mock.patch.object(shell.bundle, "app_dir", return_value=Path(tmp)):
            self.assertTrue(shell.set_autostart(True))
        self.assertTrue(self.stored()[0].endswith('" -m duckstoken --minimized'), self.stored())

    def test_unquotable_paths_are_refused_before_touching_the_registry(self):
        for bad in ('C:\\a"b.exe', "C:\\a\nb.exe", "C:\\a\0b.exe"):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                shell.set_autostart(True, bad)
        self.assertEqual(self.reg.keys, {})

    def test_registry_errors_report_failure(self):
        self.reg.error = PermissionError("denied")
        self.assertFalse(shell.set_autostart(True, r"C:\Duck\Duckstoken.exe"))
        self.assertFalse(shell.set_autostart(False))
        self.assertFalse(shell.autostart_enabled())


@unittest.skipIf(os.name == "nt", "checks the behaviour on systems without a Windows registry")
class AutostartOffWindowsTests(unittest.TestCase):
    def test_nothing_happens_and_nothing_is_reported_enabled(self):
        self.assertFalse(shell.set_autostart(True, "/opt/duck/duckstoken"))
        self.assertFalse(shell.set_autostart(False))
        self.assertFalse(shell.autostart_enabled())


# ------------------------------------------------------------------------------------------------------ choose_mode
class ChooseModeTests(unittest.TestCase):
    def test_headless_wins_over_everything(self):
        self.assertEqual(shell.choose_mode({"window_mode": "native"}, True, True), "headless")
        self.assertEqual(shell.choose_mode({"window_mode": "browser"}, False, True), "headless")

    def test_native_needs_both_the_setting_and_pywebview(self):
        self.assertEqual(shell.choose_mode({"window_mode": "native"}, True, False), "native")
        self.assertEqual(shell.choose_mode({}, True, False), "native")  # native is the default
        self.assertEqual(shell.choose_mode({"window_mode": "native"}, False, False), "browser")

    def test_browser_setting_ignores_pywebview(self):
        self.assertEqual(shell.choose_mode({"window_mode": "browser"}, True, False), "browser")
        self.assertEqual(shell.choose_mode({"window_mode": "browser"}, False, False), "browser")

    def test_unknown_values_use_the_browser(self):
        self.assertEqual(shell.choose_mode({"window_mode": "hologram"}, True, False), "browser")


# ------------------------------------------------------------------------------------------------------------ Shell
class ShellTestCase(unittest.TestCase):
    """Common set-up: private data dir, no real browser, no real pywebview/pystray, captured stderr."""

    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.home = Path(home.name)
        self.patch(mock.patch.dict(os.environ, {"DUCKSTOKEN_HOME": home.name}))
        self.patch(mock.patch.dict(sys.modules, {"webview": None, "pystray": None}))  # "not installed"
        self.browser = self.patch(mock.patch("webbrowser.open", return_value=True))
        self.stderr = self.patch(mock.patch("sys.stderr", new_callable=io.StringIO))
        self.quits = []
        self.settings = {"window_mode": "native", "minimize_to_tray": True, "start_minimized": False,
                         "open_browser": True}

    def patch(self, patcher):
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    def make(self, webview=None, tray=None, icon_path=None, url=URL):
        return shell.Shell(url, lambda: self.settings, lambda: self.quits.append(1), icon_path=icon_path,
                           webview_module=webview, tray_module=tray)

    def run_in_thread(self, sh):
        """Runs sh.run() in the background; the thread is stopped at the end of the test whatever happens."""
        errors = []

        def target():
            try:
                sh.run()
            except BaseException as exc:  # noqa: BLE001 - reported through the test
                errors.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self.addCleanup(lambda: (sh.quit(), thread.join(5)))
        return thread, errors

    def assert_finished(self, thread, errors):
        thread.join(5)
        self.assertFalse(thread.is_alive(), "run() did not return")
        self.assertEqual(errors, [])


@needs_pillow
class NativeWindowTests(ShellTestCase):
    def test_window_and_tray_are_set_up(self):
        self.settings["minimize_to_tray"] = False
        tray = FakePystray()
        seen = {}

        def script(window):
            icon = tray.icons[0]
            seen.update(running=icon.running, title=icon.title, items=[(i.text, i.default) for i in icon.menu.items],
                        image=icon.image)
            window.click_close()

        webview = FakeWebview(script)
        self.make(webview, tray).run()
        (title, url, kwargs), = webview.create_calls
        self.assertEqual((title, url), ("Duckstoken", URL))
        self.assertGreaterEqual(kwargs["min_size"], (900, 600))
        self.assertTrue(kwargs["width"] >= 900 and kwargs["height"] >= 600)
        self.assertFalse(kwargs["hidden"])
        self.assertEqual(webview.start_kwargs["private_mode"], False)  # the UI keeps its chat history
        profile = Path(webview.start_kwargs["storage_path"])
        self.assertEqual(profile, self.home / "webview")
        self.assertTrue(profile.is_dir())
        self.assertTrue(seen["running"])
        self.assertEqual(seen["title"], "Duckstoken")
        self.assertEqual(seen["items"], [("Open", True), ("Open in browser", False), ("Quit", False)])
        self.assertEqual((seen["image"].mode, seen["image"].size), ("RGBA", (64, 64)))
        self.assertTrue(tray.icons[0].stopped)
        self.assertEqual(self.quits, [1])
        self.browser.assert_not_called()

    def test_closing_hides_to_the_tray_and_the_tray_brings_it_back_and_quits(self):
        tray, steps = FakePystray(), {}

        def script(window):
            icon = tray.icons[0]
            steps["closed_for_good"] = window.click_close()
            steps["hidden"] = wait_for(lambda: window.hidden)
            steps["tray_alive"] = icon.running
            steps["quits_while_hidden"] = list(self.quits)
            icon.click("Open")
            steps["shown"] = not window.hidden
            icon.click("Quit")

        webview = FakeWebview(script)
        self.make(webview, tray).run()
        self.assertEqual(steps, {"closed_for_good": False, "hidden": True, "tray_alive": True,
                                 "quits_while_hidden": [], "shown": True})
        self.assertFalse(webview.start_timed_out)
        self.assertEqual(webview.windows[0].destroy_calls, 1)
        self.assertTrue(webview.windows[0].closed.is_set())
        self.assertTrue(tray.icons[0].stopped)
        self.assertEqual(self.quits, [1])

    def test_closing_quits_when_minimize_to_tray_is_off(self):
        tray = FakePystray()

        def script(window):
            self.assertFalse(window.click_close())  # on at first: hidden
            self.assertTrue(wait_for(lambda: window.hidden))
            self.settings["minimize_to_tray"] = False  # the setting is read live, not cached
            self.assertTrue(window.click_close())

        webview = FakeWebview(script)
        self.make(webview, tray).run()
        self.assertFalse(webview.start_timed_out)
        self.assertTrue(tray.icons[0].stopped)
        self.assertEqual(self.quits, [1])

    def test_closing_quits_when_there_is_no_tray(self):
        webview = FakeWebview(lambda window: self.assertTrue(window.click_close()))  # pystray is "not installed"
        self.make(webview).run()
        self.assertFalse(webview.start_timed_out)
        self.assertEqual(self.quits, [1])

    def test_closing_handler_answers_before_the_window_is_hidden(self):
        tray, gate = FakePystray(), threading.Event()

        def script(window):
            self.assertFalse(window.click_close())  # returned although hide() is still waiting for the gate
            self.assertFalse(window.hidden)
            gate.set()
            self.assertTrue(wait_for(lambda: window.hidden))
            tray.icons[0].click("Quit")

        webview = FakeWebview(script, hide_gate=gate)
        self.make(webview, tray).run()
        self.assertFalse(webview.start_timed_out)
        self.assertEqual(self.quits, [1])

    def test_a_window_that_cannot_be_hidden_quits_the_app(self):
        tray = FakePystray()

        def script(window):
            self.assertFalse(window.click_close())  # the veto is given before hiding is attempted

        webview = FakeWebview(script, hide_error=RuntimeError("no"))
        self.make(webview, tray).run()
        self.assertFalse(webview.start_timed_out)
        self.assertIn("Could not hide the window (no); quitting", self.stderr.getvalue())
        self.assertEqual(self.quits, [1])
        self.assertTrue(tray.icons[0].stopped)

    def test_start_minimized_creates_the_window_hidden_when_a_tray_exists(self):
        self.settings["start_minimized"] = True
        tray = FakePystray()

        def script(window):
            self.assertTrue(window.hidden)
            tray.icons[0].click("Quit")

        webview = FakeWebview(script)
        self.make(webview, tray).run()
        self.assertTrue(webview.create_calls[0][2]["hidden"])
        self.assertEqual(self.quits, [1])

    def test_start_minimized_is_ignored_without_a_tray(self):
        self.settings["start_minimized"] = True
        webview = FakeWebview(lambda window: window.click_close())
        self.make(webview).run()
        self.assertFalse(webview.create_calls[0][2]["hidden"])

    def test_tray_open_in_browser_uses_the_default_browser(self):
        tray = FakePystray()

        def script(window):
            tray.icons[0].click("Open in browser")
            tray.icons[0].click("Quit")

        self.make(FakeWebview(script), tray).run()
        self.browser.assert_called_once_with(URL)

    def test_show_falls_back_to_the_browser_if_the_window_cannot_be_shown(self):
        tray = FakePystray()

        def script(window):
            window.click_close()  # hidden in the tray
            self.assertTrue(wait_for(lambda: window.hidden))
            tray.icons[0].click("Open")
            tray.icons[0].click("Quit")

        webview = FakeWebview(script, show_error=RuntimeError("gone"))
        self.make(webview, tray).run()
        self.browser.assert_called_once_with(URL)
        self.assertIn("Could not show the window", self.stderr.getvalue())

    def test_quit_from_another_thread_closes_everything(self):
        tray = FakePystray()
        webview = FakeWebview()  # the window just stays open
        sh = self.make(webview, tray)
        thread, errors = self.run_in_thread(sh)
        self.assertTrue(wait_for(lambda: webview.start_kwargs is not None))
        self.assertTrue(thread.is_alive())
        sh.quit()
        sh.quit()  # idempotent
        self.assert_finished(thread, errors)
        self.assertEqual(webview.windows[0].destroy_calls, 1)
        self.assertTrue(tray.icons[0].stopped)
        self.assertEqual(self.quits, [1])

    def test_on_quit_still_runs_when_the_app_is_interrupted(self):
        tray = FakePystray()

        def script(window):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.make(FakeWebview(script), tray).run()
        self.assertTrue(tray.icons[0].stopped)
        self.assertEqual(self.quits, [1])

    def test_tray_that_cannot_start_means_no_tray(self):
        tray = FakePystray(run_error=RuntimeError("no notification area"))
        webview = FakeWebview(lambda window: self.assertTrue(window.click_close()))
        self.make(webview, tray).run()
        self.assertIn("System tray unavailable", self.stderr.getvalue())
        self.assertEqual(self.quits, [1])

    def test_icon_file_is_used_for_the_tray_and_a_broken_one_falls_back_to_the_drawn_duck(self):
        good, bad = self.home / "icon.png", self.home / "broken.png"
        Image.new("RGB", (32, 32), "red").save(good)
        bad.write_text("not an image")
        sizes = {}
        for name, path in (("good", good), ("bad", bad), ("missing", self.home / "nope.png")):
            tray = FakePystray()
            webview = FakeWebview(lambda window: window.click_close())
            self.settings["minimize_to_tray"] = False
            self.make(webview, tray, icon_path=str(path)).run()
            sizes[name] = (tray.icons[0].image.size, tray.icons[0].image.mode)
        self.assertEqual(sizes, {"good": ((32, 32), "RGBA"), "bad": ((64, 64), "RGBA"), "missing": ((64, 64), "RGBA")})
        good.unlink()  # nothing may keep the file open (Windows)

    def test_no_tray_without_pillow(self):
        tray = FakePystray()
        webview = FakeWebview(lambda window: window.click_close())
        with mock.patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            self.make(webview, tray).run()
        self.assertEqual(tray.icons, [])
        self.assertEqual(self.quits, [1])


@needs_pillow
class BrowserFallbackTests(ShellTestCase):
    def test_window_creation_failure_opens_the_browser_and_lives_on_in_the_tray(self):
        tray = FakePystray()
        webview = FakeWebview(create_error=RuntimeError("WebView2 is missing"))
        sh = self.make(webview, tray)
        thread, errors = self.run_in_thread(sh)
        self.assertTrue(wait_for(lambda: self.browser.called))
        self.browser.assert_called_once_with(URL)
        self.assertIn("Native window unavailable (WebView2 is missing)", self.stderr.getvalue())
        self.assertTrue(tray.icons[0].running)
        self.assertTrue(thread.is_alive())
        tray.icons[0].click("Open")  # without a window "Open" means the browser
        self.assertEqual(self.browser.call_count, 2)
        tray.icons[0].click("Quit")
        self.assert_finished(thread, errors)
        self.assertEqual(self.quits, [1])

    def test_start_failure_is_handled_the_same_way(self):
        tray = FakePystray()
        sh = self.make(FakeWebview(start_error=RuntimeError("no GUI backend")), tray)
        thread, errors = self.run_in_thread(sh)
        self.assertTrue(wait_for(lambda: self.browser.called))
        tray.icons[0].click("Quit")
        self.assert_finished(thread, errors)
        self.assertEqual(self.quits, [1])

    def test_failed_window_stays_hidden_when_started_minimized_with_a_tray(self):
        self.settings["start_minimized"] = True
        tray = FakePystray()
        sh = self.make(FakeWebview(create_error=RuntimeError("x")), tray)
        thread, errors = self.run_in_thread(sh)
        self.assertTrue(wait_for(lambda: tray.icons and tray.icons[0].running))
        time.sleep(0.2)
        self.browser.assert_not_called()
        tray.icons[0].click("Quit")
        self.assert_finished(thread, errors)

    def test_missing_pywebview_uses_the_browser(self):
        tray = FakePystray()
        sh = self.make(None, tray)  # `webview` is "not installed" (see setUp)
        thread, errors = self.run_in_thread(sh)
        self.assertTrue(wait_for(lambda: self.browser.called))
        tray.icons[0].click("Quit")
        self.assert_finished(thread, errors)
        self.assertEqual(self.quits, [1])

    def test_browser_setting_never_touches_pywebview(self):
        self.settings["window_mode"] = "browser"
        tray, webview = FakePystray(), FakeWebview()
        sh = self.make(webview, tray)
        thread, errors = self.run_in_thread(sh)
        self.assertTrue(wait_for(lambda: self.browser.called))
        self.assertEqual((webview.create_calls, webview.start_kwargs), ([], None))
        self.assertTrue(tray.icons[0].running)
        tray.icons[0].click("Quit")
        self.assert_finished(thread, errors)
        self.assertTrue(tray.icons[0].stopped)
        self.assertEqual(self.quits, [1])

    def test_browser_mode_respects_open_browser_and_start_minimized(self):
        for overrides, opens in (({"open_browser": False}, False), ({"start_minimized": True}, False),
                                 ({"open_browser": True, "start_minimized": False}, True)):
            with self.subTest(overrides=overrides):
                self.browser.reset_mock()
                self.settings.update(window_mode="browser", **overrides)
                tray = FakePystray()
                sh = self.make(None, tray)
                thread, errors = self.run_in_thread(sh)
                self.assertTrue(wait_for(lambda t=tray: t.icons and t.icons[0].running))
                time.sleep(0.2)
                self.assertEqual(self.browser.called, opens)
                tray.icons[0].click("Quit")
                self.assert_finished(thread, errors)

    def test_browser_mode_without_a_tray_opens_the_browser_even_when_started_minimized(self):
        self.settings.update(window_mode="browser", start_minimized=True)
        sh = self.make(None, None)
        thread, errors = self.run_in_thread(sh)
        self.assertTrue(wait_for(lambda: self.browser.called))
        sh.quit()
        self.assert_finished(thread, errors)
        self.assertEqual(self.quits, [1])


class HeadlessTests(ShellTestCase):
    def test_headless_only_waits(self):
        self.settings["headless"] = True
        tray, webview = FakePystray(), FakeWebview()
        sh = self.make(webview, tray)
        thread, errors = self.run_in_thread(sh)
        time.sleep(0.3)
        self.assertTrue(thread.is_alive())
        self.assertEqual((webview.create_calls, tray.icons), ([], []))
        self.browser.assert_not_called()
        sh.quit()
        self.assert_finished(thread, errors)
        self.assertEqual(self.quits, [1])

    def test_browser_mode_without_any_optional_piece_waits_for_quit(self):
        self.settings["open_browser"] = False
        sh = self.make()
        thread, errors = self.run_in_thread(sh)
        time.sleep(0.3)
        self.assertTrue(thread.is_alive())
        self.browser.assert_not_called()
        sh.quit()
        self.assert_finished(thread, errors)


class ConstructorTests(unittest.TestCase):
    def test_only_http_urls_are_accepted(self):
        for bad in ("file:///etc/passwd", "javascript:alert(1)", "127.0.0.1:8765", ""):
            with self.subTest(url=bad), self.assertRaises(ValueError):
                shell.Shell(bad, dict, lambda: None)
        shell.Shell("https://127.0.0.1:1/", dict, lambda: None)


if __name__ == "__main__":
    unittest.main()
