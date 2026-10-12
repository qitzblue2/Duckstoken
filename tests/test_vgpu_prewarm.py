"""Sequential page-cache prewarm: bounded by max_bytes and free RAM, cancellable, stops under pressure."""
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckstoken.vgpu import prewarm as pw  # noqa: E402

MB = 1024 * 1024
BIG_RAM = lambda: 64.0  # noqa: E731


class TestPrewarm(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def file(self, name="m.gguf", size=5 * MB):
        p = self.dir / name
        p.write_bytes(os.urandom(1024) * (size // 1024))
        return p

    def run_prewarm(self, target, max_bytes=10**12, **kw):
        kw.setdefault("avail_ram_fn", BIG_RAM)
        kw.setdefault("pressure_fn", lambda: "ok")
        kw.setdefault("chunk_bytes", 64 * 1024)
        return pw.prewarm(target, max_bytes, **kw)

    def test_reads_whole_file_and_reports_measured_speed(self):
        p = self.file(size=5 * MB)
        calls = []
        r = self.run_prewarm(p, progress=lambda *a: calls.append(a))
        self.assertTrue(r["completed"])
        self.assertIsNone(r["stopped_because"])
        self.assertEqual(r["bytes_read"], 5 * MB)
        self.assertEqual(r["requested_bytes"], 5 * MB)
        self.assertEqual(r["mb_per_s"]["status"], "measured")
        self.assertGreater(r["mb_per_s"]["value"], 0)
        self.assertIn("does not make a slow drive faster", r["note"])
        self.assertEqual(r["files"][0]["bytes_read"], 5 * MB)
        self.assertEqual(calls[-1][0], 5 * MB)                 # final progress call reports everything
        done = [c[0] for c in calls]
        self.assertEqual(done, sorted(done))

    def test_budget_cap_never_reads_more_than_max_bytes(self):
        p = self.file(size=5 * MB)
        r = self.run_prewarm(p, max_bytes=2 * MB + 100)
        self.assertEqual(r["bytes_read"], 2 * MB + 100)
        self.assertFalse(r["completed"])
        self.assertEqual(r["stopped_because"], "budget")
        self.assertEqual(r["limit_bytes"], 2 * MB + 100)

    def test_bounded_by_available_ram_minus_reserve(self):
        p = self.file(size=5 * MB)
        r = self.run_prewarm(p, avail_ram_fn=lambda: 4.0 + 3 * MB / 2**30, reserve_gb=4.0)
        self.assertEqual(r["stopped_because"], "ram_headroom")
        self.assertLessEqual(r["bytes_read"], 3 * MB + 1)
        self.assertGreater(r["bytes_read"], 0)

    def test_no_ram_headroom_reads_nothing(self):
        p = self.file()
        r = self.run_prewarm(p, avail_ram_fn=lambda: 3.0, reserve_gb=4.0)
        self.assertEqual((r["bytes_read"], r["stopped_because"], r["completed"]), (0, "no_ram_headroom", False))
        self.assertEqual(r["mb_per_s"]["status"], "unavailable")

    def test_unknown_ram_reads_nothing(self):
        for fn in (lambda: None, lambda: 0, lambda: 0.0):
            r = self.run_prewarm(self.file(), avail_ram_fn=fn)
            self.assertEqual((r["bytes_read"], r["stopped_because"]), (0, "ram_unknown"))

    def test_zero_budget(self):
        r = self.run_prewarm(self.file(), max_bytes=0)
        self.assertEqual((r["bytes_read"], r["stopped_because"]), (0, "budget"))

    def test_cancel_midway(self):
        p = self.file(size=8 * MB)
        r = self.run_prewarm(p, cancel=CancelAfter(20))          # 20 chunks of 64 KiB, then "stop"
        self.assertEqual(r["stopped_because"], "cancelled")
        self.assertFalse(r["completed"])
        self.assertEqual(r["bytes_read"], 20 * 64 * 1024)

    def test_cancelled_before_start(self):
        ev = threading.Event()
        ev.set()
        r = self.run_prewarm(self.file(), cancel=ev)
        self.assertEqual((r["bytes_read"], r["stopped_because"]), (0, "cancelled"))

    def test_stops_when_memory_pressure_turns_high(self):
        p = self.file(size=6 * MB)
        levels = iter(["ok", "ok", "high", "high", "high"])
        r = self.run_prewarm(p, pressure_fn=lambda: next(levels, "high"), check_every_bytes=1 * MB)
        self.assertEqual(r["stopped_because"], "memory_pressure")
        self.assertLess(r["bytes_read"], 6 * MB)
        self.assertGreater(r["bytes_read"], 0)

    def test_stops_when_available_ram_drops_below_reserve_while_reading(self):
        p = self.file(size=6 * MB)
        avail = iter([64.0, 64.0, 2.0, 2.0])
        r = self.run_prewarm(p, avail_ram_fn=lambda: next(avail, 2.0), reserve_gb=4.0, check_every_bytes=1 * MB)
        self.assertEqual(r["stopped_because"], "memory_pressure")
        self.assertLess(r["bytes_read"], 6 * MB)

    def test_critical_pressure_also_stops(self):
        r = self.run_prewarm(self.file(size=3 * MB), pressure_fn=lambda: "critical", check_every_bytes=1)
        self.assertEqual(r["stopped_because"], "memory_pressure")

    def test_elevated_pressure_keeps_going(self):
        r = self.run_prewarm(self.file(size=3 * MB), pressure_fn=lambda: "elevated", check_every_bytes=1 * MB)
        self.assertTrue(r["completed"])

    def test_split_model_reads_every_shard(self):
        a = self.file("big-00001-of-00003.gguf", 2 * MB)
        self.file("big-00002-of-00003.gguf", 1 * MB)
        self.file("big-00003-of-00003.gguf", 3 * MB)
        r = self.run_prewarm(a)
        self.assertTrue(r["completed"])
        self.assertEqual(r["bytes_read"], 6 * MB)
        self.assertEqual(len(r["files"]), 3)

    def test_budget_across_shards_stops_inside_second_file(self):
        a = self.file("big-00001-of-00002.gguf", 2 * MB)
        self.file("big-00002-of-00002.gguf", 2 * MB)
        r = self.run_prewarm(a, max_bytes=3 * MB)
        self.assertEqual(r["bytes_read"], 3 * MB)
        self.assertEqual([f["bytes_read"] for f in r["files"]], [2 * MB, 1 * MB])

    def test_list_of_paths_deduplicates(self):
        a, b = self.file("a.gguf", 1 * MB), self.file("b.gguf", 1 * MB)
        r = self.run_prewarm([a, b, a])
        self.assertEqual(r["bytes_read"], 2 * MB)
        self.assertEqual(len(r["files"]), 2)

    def test_missing_files(self):
        r = self.run_prewarm(self.dir / "nope.gguf")
        self.assertEqual((r["stopped_because"], r["completed"], r["bytes_read"]), ("no_files", False, 0))
        self.assertEqual(self.run_prewarm([])["stopped_because"], "no_files")
        self.assertEqual(self.run_prewarm(None)["stopped_because"], "no_files")

    def test_file_deleted_mid_read_is_reported_not_raised(self):
        p = self.file(size=4 * MB)
        real_open = open

        def flaky(path, *a, **k):
            if str(path) == str(p):
                raise PermissionError("locked")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", flaky):
            r = self.run_prewarm(p)
        self.assertEqual(r["stopped_because"], "read_error")
        self.assertIn("locked", r["error"])

    def test_file_shrinking_under_us_does_not_loop_forever(self):
        p = self.file(size=2 * MB)
        real_stat = Path.stat

        class Big:
            st_size = 4 * MB
            st_mode = real_stat(p).st_mode

        with mock.patch.object(Path, "stat", lambda self, *a, **k: Big() if self == p else real_stat(self, *a, **k)):
            r = self.run_prewarm(p)
        self.assertFalse(r["completed"])
        self.assertEqual(r["bytes_read"], 2 * MB)

    @unittest.skipUnless(hasattr(os, "posix_fadvise"), "POSIX only")
    def test_uses_posix_fadvise_sequential_and_ignores_its_failure(self):
        p = self.file(size=2 * MB)
        with mock.patch("os.posix_fadvise") as fa:
            r = self.run_prewarm(p)
        self.assertTrue(r["completed"])
        advices = {c.args[3] for c in fa.call_args_list}
        self.assertIn(os.POSIX_FADV_SEQUENTIAL, advices)
        self.assertIn(os.POSIX_FADV_WILLNEED, advices)
        with mock.patch("os.posix_fadvise", side_effect=OSError("unsupported")):
            self.assertTrue(self.run_prewarm(p)["completed"])

    def test_works_without_fadvise(self):
        p = self.file(size=2 * MB)
        with mock.patch.object(pw.os, "posix_fadvise", None, create=True):
            self.assertTrue(self.run_prewarm(p)["completed"])

    def test_default_functions_do_not_crash(self):
        gb = pw._default_avail_gb()
        self.assertTrue(gb is None or gb > 0)
        self.assertIn(pw._default_pressure(), (None, "ok", "elevated", "high", "critical"))

    def test_default_pressure_degrades_if_monitoring_is_broken(self):
        with mock.patch.dict(sys.modules, {"duckstoken.vgpu.pressure": None}):
            self.assertIsNone(pw._default_pressure())


class CancelAfter:
    """Event-like: reports cancelled once `is_set` has been asked `n` times."""

    def __init__(self, n):
        self.n, self.calls = n, 0

    def is_set(self):
        self.calls += 1
        return self.calls > self.n


if __name__ == "__main__":
    unittest.main()
