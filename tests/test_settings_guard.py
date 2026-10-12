"""Review finding: importing a settings file (or loading a profile) silently granted PC access and switched off approvals."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_app_integration as ti  # noqa: E402
from duckstoken import app as appmod, settings as st  # noqa: E402

RISKY = {"tools_enabled": True, "tool_run_command": True, "confirm_commands": False, "full_pc_access": True,
         "llama_server_path": "/bin/sh", "extra_args": "-c id"}


class RiskyChanges(unittest.TestCase):
    def test_only_changes_away_from_the_safe_default_are_risky(self):
        cur = st.DEFAULTS
        keys = {c["key"] for c in st.risky_changes(cur, RISKY)}
        self.assertTrue({"tool_run_command", "confirm_commands", "full_pc_access", "llama_server_path", "extra_args"} <= keys)
        self.assertEqual(st.risky_changes(cur, {"temperature": 0.2, "ctx_size": 4096, "theme": "light"}), [])
        # turning an approval back ON is the safe direction
        self.assertEqual(st.risky_changes({**cur, "confirm_commands": False}, {"confirm_commands": True}), [])
        # a value that does not change is never risky
        self.assertEqual(st.risky_changes(cur, {"confirm_commands": True}), [])

    def test_secrets_are_not_echoed_back(self):
        out = st.risky_changes(st.DEFAULTS, {"llama_server_path": "x", "api_key": "k"})
        self.assertEqual([c["key"] for c in out], ["llama_server_path"])


class ImportEndpoint(ti.Api):
    def test_risky_import_needs_confirmation_and_changes_nothing_until_confirmed(self):
        before = appmod.APP.settings.all()
        r = self.call("settings/import", {"values": RISKY})
        self.assertEqual(r["_status"], 409)
        self.assertTrue(r["needs_confirmation"])
        self.assertEqual(appmod.APP.settings.all(), before)
        r = self.call("settings/import", {"values": RISKY, "confirm_risky": True})
        self.assertNotIn("_status", r)
        self.assertTrue(appmod.APP.settings.get("tools_enabled"))
        self.assertFalse(appmod.APP.settings.get("confirm_commands"))

    def test_harmless_import_needs_no_confirmation(self):
        r = self.call("settings/import", {"values": {"temperature": 0.3, "ctx_size": 4096}})
        self.assertNotIn("_status", r)
        self.assertEqual(appmod.APP.settings.get("temperature"), 0.3)

    def test_profile_load_is_guarded_the_same_way(self):
        # loading a profile re-applies vgpu_monitor, which (rightly) starts the telemetry threads: stop them again
        self.addCleanup(appmod.APP.vgpu.monitor.stop)
        self.addCleanup(appmod.APP.vgpu.sampler.stop)
        appmod.APP.settings.update(RISKY)
        self.call("profiles/save", {"name": "risky"})
        appmod.APP.settings.reset()
        r = self.call("profiles/load", {"name": "risky"})
        self.assertEqual(r["_status"], 409)
        self.assertFalse(appmod.APP.settings.get("tools_enabled"))
        r = self.call("profiles/load", {"name": "risky", "confirm_risky": True})
        self.assertTrue(appmod.APP.settings.get("tools_enabled"))
        self.call("profiles/delete", {"name": "risky"})


if __name__ == "__main__":
    unittest.main()
