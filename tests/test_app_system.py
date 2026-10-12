"""Update endpoints through the real HTTP handler (no network: the checker's GitHub call is replaced)."""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_app_integration as ti  # noqa: E402
from duckstoken import app as appmod, updater  # noqa: E402

AVAILABLE = {"available": True, "channel": "stable", "version": "9.9.9", "sha": "a" * 40, "notes": "new things",
             "published": "", "asset": {"name": "Duckstoken.exe", "url": "https://github.com/x/y/releases/download/v9/Duckstoken.exe",
                                        "size": 10}, "sha256": "b" * 64, "html_url": "", "reason": ""}


class UpdateEndpoints(ti.Api):
    def setUp(self):
        super().setUp()
        a = appmod.APP
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        a.updater = updater.UpdateChecker(a.settings.all, self.tmp.name, check_fn=lambda ch: dict(AVAILABLE))
        a.update_job = type(a.update_job)()

    def test_status_reports_current_build_and_channel(self):
        s = self.call("update/status")
        self.assertEqual(s["channel"], "stable")
        self.assertIn("version", s["current"])
        self.assertFalse(s["can_apply"])  # a source checkout cannot update itself
        self.assertEqual(s["download"]["state"], "idle")
        self.assertIsNone(s["result"])

    def test_check_off_channel_is_refused(self):
        appmod.APP.settings.update({"update_channel": "off"})
        r = self.call("update/check", {})
        self.assertEqual(r["_status"], 400)

    def test_check_then_download_then_apply_needs_confirmation(self):
        r = self.call("update/check", {})
        self.assertTrue(r["result"]["available"])
        self.assertEqual(r["result"]["version"], "9.9.9")
        fake = Path(self.tmp.name) / "updates"
        fake.mkdir()
        exe = fake / "Duckstoken.exe"
        exe.write_bytes(b"x" * 10)

        def fake_download(plan, dest, progress, cancel=None, **kw):
            progress(5, 10)
            progress(10, 10)
            return exe

        with mock.patch.object(updater, "download", fake_download):
            self.call("update/download", {})
            for _ in range(100):
                st = self.call("update/status")
                if st["download"]["state"] != "downloading":
                    break
                time.sleep(0.05)
        self.assertEqual(st["download"]["state"], "downloaded")
        self.assertEqual(st["downloaded"], str(exe))
        self.assertEqual(self.call("update/apply", {})["_status"], 400)  # no explicit confirm
        r = self.call("update/apply", {"confirm": True})
        self.assertEqual(r["_status"], 400)  # not the packaged Windows app
        self.assertTrue(exe.exists())  # nothing was touched

    def test_download_without_update_is_refused(self):
        r = self.call("update/download", {})
        self.assertEqual(r["_status"], 400)

    def test_check_failure_is_reported_not_raised(self):
        def boom(channel):
            raise updater.UpdateError("GitHub is down")
        a = appmod.APP
        a.updater = updater.UpdateChecker(a.settings.all, self.tmp.name, check_fn=boom)
        r = self.call("update/check", {})
        self.assertEqual(r["_status"], 502)
        self.assertIn("GitHub is down", r["error"])


class Shutdown(unittest.TestCase):
    """Review findings: Quit / 'Install & restart' left the process alive, and auto-install ran Setup.exe twice."""

    def test_shutdown_runs_once_and_ends_the_native_shell(self):
        a = appmod.App()  # App() binds itself into the api_* modules; give them the shared instance back afterwards
        self.addCleanup(a.db.close)
        self.addCleanup(lambda: [m.bind(appmod.APP) for m in (appmod.api_learning, appmod.api_system, appmod.api_vgpu)])
        calls = []

        class FakeShell:
            def quit(self):
                calls.append("quit")

        a.shell = FakeShell()
        with mock.patch.object(appmod.api_system, "apply_on_exit", side_effect=lambda app: calls.append("apply")):
            a.shutdown()
            a.shutdown()  # Shell.run's finally and run()'s finally both call it
        self.assertEqual(calls.count("quit"), 1)
        self.assertEqual(calls.count("apply"), 1)

    def test_auto_install_launches_the_installer_only_once(self):
        from duckstoken import api_system

        class FakeUpdater:
            def __init__(self):
                self.downloaded = "C:/x/Duckstoken-Setup.exe"
                self.applied = 0

            def status(self):
                return {"downloaded": self.downloaded}

            def mark_applied(self):
                self.downloaded = None

        class FakeApp:
            updater = FakeUpdater()

            class settings:
                @staticmethod
                def get(k, d=None):
                    return k == "update_auto_install"

        launched = []
        with mock.patch.object(api_system.updater, "apply", side_effect=lambda p, restart=True: launched.append(str(p))), \
                mock.patch.object(api_system.updater, "current_info", return_value={"frozen": True}):
            api_system.apply_on_exit(FakeApp)
            api_system.apply_on_exit(FakeApp)
        self.assertEqual(len(launched), 1)


if __name__ == "__main__":
    unittest.main()
