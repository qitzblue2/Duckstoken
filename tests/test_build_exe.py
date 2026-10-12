"""The packaging command and the --selftest report (review finding: pywebview's code was never bundled)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import build_exe  # noqa: E402
from duckstoken import selftest  # noqa: E402


class PackagingCommand(unittest.TestCase):
    def test_webview_code_is_collected_not_just_its_data(self):
        for mode in ("onefile", "onedir"):
            cmd = build_exe.command(mode)
            pairs = list(zip(cmd, cmd[1:]))
            self.assertIn(("--collect-all", "webview"), pairs)
            self.assertNotIn(("--collect-data", "webview"), pairs)
            self.assertIn(("--hidden-import", "pystray._win32"), pairs)
            self.assertIn(f"--{mode}", cmd)
            self.assertIn("--windowed", cmd)
            self.assertTrue(any(c.startswith("engine") for c in cmd), "the llama.cpp engine must be bundled")
            self.assertEqual(cmd[-1], "duckstoken_launcher.py")

    def test_distpath_and_bad_mode(self):
        self.assertIn("dist-installer", build_exe.command("onedir", "dist-installer"))
        with self.assertRaises(ValueError):
            build_exe.command("zip")

    def test_ci_and_bat_use_the_shared_builder(self):
        for f in (ROOT / "build_windows.bat", ROOT / ".github" / "workflows" / "release.yml"):
            text = f.read_text(encoding="utf-8")
            self.assertIn("build_exe.py", text, f.name)
            self.assertNotIn("--collect-data webview", text, f.name)


class SelfTest(unittest.TestCase):
    def test_report_lists_every_check_and_passes_from_source(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            code = selftest.run(str(out))
            report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)              # not frozen: missing optional packages are reported, not fatal
        self.assertFalse(report["frozen"])
        names = {c["check"] for c in report["checks"]}
        self.assertTrue({"web ui files", "pywebview module", "bundled llama.cpp engine", "numpy (built-in engine)"} <= names)
        ui = next(c for c in report["checks"] if c["check"] == "web ui files")
        self.assertTrue(ui["ok"])

    def test_cli_flag_writes_the_report_and_exits(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "cli.json"
            r = subprocess.run([sys.executable, "-m", "duckstoken", "--selftest", str(out), "--data-dir", d], cwd=ROOT,
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(json.loads(out.read_text(encoding="utf-8"))["checks"])

    def test_a_frozen_build_missing_a_required_part_fails(self):
        from unittest import mock
        checks = [("x", lambda: (_ for _ in ()).throw(ImportError("gone")), True), ("y", lambda: "fine", True)]
        with mock.patch.object(selftest, "CHECKS", checks), mock.patch.object(selftest.bundle, "frozen", return_value=True):
            self.assertEqual(selftest.run(None), 1)
        with mock.patch.object(selftest, "CHECKS", checks), mock.patch.object(selftest.bundle, "frozen", return_value=False):
            self.assertEqual(selftest.run(None), 0)


if __name__ == "__main__":
    unittest.main()
