"""JobRunner: one job at a time, progress, cancel, failure capture, result retention."""
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckstoken.vgpu import runner as rn  # noqa: E402


class TestJobRunner(unittest.TestCase):
    def setUp(self):
        self.r = rn.JobRunner()
        self.addCleanup(self._drain)

    def _drain(self):
        self.r.cancel()
        self.r.wait(5)

    def test_idle_status_shape(self):
        s = self.r.status()
        self.assertEqual(s["state"], "idle")
        self.assertIsNone(s["id"])
        self.assertEqual(s["progress"], {"step": 0, "total": 0, "message": "", "pct": None})
        self.assertEqual(s["log"], [])
        for key in ("kind", "result", "error", "started", "finished"):
            self.assertIn(key, s)
        self.assertTrue(self.r.wait(0.01))            # nothing to wait for
        self.assertFalse(self.r.cancel())             # nothing to cancel

    def test_runs_to_done_with_result_args_and_progress(self):
        def job(cancel, progress, a, b):
            progress(1, 2, "half way")
            progress(2, 2, "finished")
            return {"sum": a + b}

        jid = self.r.start("demo", job, 2, 3)
        self.assertTrue(self.r.wait(5))
        s = self.r.status()
        self.assertEqual((s["id"], s["kind"], s["state"]), (jid, "demo", "done"))
        self.assertEqual(s["result"], {"sum": 5})
        self.assertEqual(s["progress"], {"step": 2, "total": 2, "message": "finished", "pct": 100})
        self.assertIsNotNone(s["started"])
        self.assertGreaterEqual(s["finished"], s["started"])
        self.assertTrue(any("half way" in line for line in s["log"]))
        self.assertEqual(s["error"], "")

    def test_only_one_job_at_a_time(self):
        release = threading.Event()
        self.r.start("first", lambda cancel, progress: release.wait(5))
        with self.assertRaises(rn.Busy):
            self.r.start("second", lambda cancel, progress: None)
        self.assertEqual(self.r.status()["kind"], "first")
        self.assertTrue(self.r.active())
        release.set()
        self.assertTrue(self.r.wait(5))
        self.assertFalse(self.r.active())
        self.r.start("second", lambda cancel, progress: "ok")      # free again once finished
        self.assertTrue(self.r.wait(5))
        self.assertEqual(self.r.status()["result"], "ok")

    def test_cancel_sets_shared_event_and_state_becomes_cancelled(self):
        seen = {}
        started = threading.Event()

        def job(cancel, progress):
            seen["event"] = cancel
            started.set()
            while not cancel.wait(0.01):
                pass
            return "partial"

        self.r.start("long", job)
        self.assertTrue(started.wait(5))
        self.assertTrue(self.r.cancel())
        self.assertTrue(self.r.status()["cancel_requested"] or self.r.status()["state"] == "cancelled")
        self.assertTrue(self.r.wait(5))
        s = self.r.status()
        self.assertEqual(s["state"], "cancelled")
        self.assertTrue(seen["event"].is_set())
        self.assertEqual(s["result"], "partial")          # whatever the job returned is kept
        self.assertFalse(s["cancel_requested"])
        self.assertFalse(self.r.cancel())                 # already finished

    def test_job_may_raise_jobcancelled(self):
        def job(cancel, progress):
            raise rn.JobCancelled("stopped by user")

        self.r.start("x", job)
        self.r.wait(5)
        s = self.r.status()
        self.assertEqual((s["state"], s["error"]), ("cancelled", "stopped by user"))

    def test_exception_becomes_failed_with_error_and_traceback_in_log(self):
        def job(cancel, progress):
            raise ValueError("boom")

        self.r.start("x", job)
        self.r.wait(5)
        s = self.r.status()
        self.assertEqual(s["state"], "failed")
        self.assertEqual(s["error"], "ValueError: boom")
        self.assertIsNone(s["result"])
        self.assertTrue(any("ValueError" in line for line in s["log"]))
        self.r.start("again", lambda cancel, progress: 1)   # a failed job does not wedge the runner
        self.r.wait(5)
        self.assertEqual(self.r.status()["state"], "done")

    def test_result_retained_until_next_start(self):
        self.r.start("a", lambda cancel, progress: {"v": 1})
        self.r.wait(5)
        self.assertEqual(self.r.status()["result"], {"v": 1})
        self.assertEqual(self.r.status()["result"], {"v": 1})     # reading does not consume it
        release = threading.Event()
        self.r.start("b", lambda cancel, progress: release.wait(5))
        self.assertIsNone(self.r.status()["result"])               # new job started -> old result gone
        release.set()
        self.r.wait(5)

    def test_progress_pct_and_unknown_total(self):
        gate, go = threading.Event(), threading.Event()

        def job(cancel, progress):
            progress(1, 4, "quarter")
            gate.set()
            go.wait(5)
            progress(3, 0, "unknown total")
            progress("garbage", None, "bad numbers are tolerated")

        self.r.start("p", job)
        self.assertTrue(gate.wait(5))
        self.assertEqual(self.r.status()["progress"]["pct"], 25)
        go.set()
        self.r.wait(5)
        p = self.r.status()["progress"]
        self.assertEqual((p["step"], p["total"], p["pct"]), (0, 0, None))

    def test_progress_log_helper_and_log_is_bounded(self):
        def job(cancel, progress):
            for i in range(500):
                progress.log(f"line {i}")

        self.r.start("log", job)
        self.r.wait(5)
        log = self.r.status()["log"]
        self.assertEqual(len(log), rn.LOG_LINES)
        self.assertIn("line 499", log[-2])
        self.assertTrue(log[-1].endswith("done"))

    def test_progress_after_finish_is_ignored(self):
        holder = {}
        self.r.start("x", lambda cancel, progress: holder.setdefault("p", progress))
        self.r.wait(5)
        holder["p"](9, 9, "late")
        self.assertNotEqual(self.r.status()["progress"]["message"], "late")

    def test_wait_timeout_returns_false_while_running(self):
        release = threading.Event()
        self.r.start("slow", lambda cancel, progress: release.wait(5))
        t0 = time.time()
        self.assertFalse(self.r.wait(0.05))
        self.assertLess(time.time() - t0, 2)
        release.set()
        self.assertTrue(self.r.wait(5))

    def test_two_runners_are_independent(self):
        other = rn.JobRunner()
        release = threading.Event()
        self.r.start("a", lambda cancel, progress: release.wait(5))
        other.start("b", lambda cancel, progress: "done")     # another instance is not blocked
        other.wait(5)
        self.assertEqual(other.status()["result"], "done")
        release.set()


if __name__ == "__main__":
    unittest.main()
