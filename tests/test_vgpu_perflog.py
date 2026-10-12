import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from duckstoken.vgpu.perflog import MAX_LINE, MAX_STR, PerfLog


class PerfLogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "sub" / "perf.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def test_log_and_tail(self):
        log = PerfLog(self.path)
        self.assertTrue(log.log("benchmark_start", model="m.gguf", tokens=64, ok=True, ratio=0.5))
        log.log("benchmark_end", tps=4.2)
        recs = log.tail(10)
        self.assertEqual([r["event"] for r in recs], ["benchmark_start", "benchmark_end"])
        self.assertEqual((recs[0]["model"], recs[0]["tokens"], recs[0]["ok"], recs[0]["ratio"]), ("m.gguf", 64, True, 0.5))
        self.assertIsInstance(recs[0]["t"], float)
        self.assertLessEqual(abs(recs[0]["t"] - time.time()), 5)
        self.assertEqual(len(log.tail(1)), 1)
        self.assertEqual(log.tail(1)[0]["event"], "benchmark_end")
        self.assertEqual(log.tail(0), [])
        self.assertEqual(log.stats()["written"], 2)

    def test_one_json_object_per_line(self):
        log = PerfLog(self.path)
        log.log("a", x=1)
        log.log("b", text="line1\nline2")
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[1])["text"], "line1\nline2")

    def test_values_are_truncated_and_cleaned(self):
        log = PerfLog(self.path)
        log.log("big", text="x" * 5000, nan=float("nan"), inf=float("inf"), items=list(range(500)), nested={"a": {"b": {"c": {"d": {"e": 1}}}}},
                data=b"bytes", obj=object(), tup=(1, 2), nonstr={1: 2})
        r = log.tail(1)[0]
        self.assertLessEqual(len(r["text"]), MAX_STR + 20)
        self.assertTrue(r["text"].endswith("[truncated]"))
        self.assertIsNone(r["nan"])
        self.assertIsNone(r["inf"])
        self.assertEqual(len(r["items"]), 51)
        self.assertIn("more", r["items"][-1])
        self.assertEqual(r["data"], "bytes")
        self.assertTrue(r["obj"].startswith("<object"))
        self.assertEqual(r["tup"], [1, 2])
        self.assertEqual(r["nonstr"], {"1": 2})
        json.loads(self.path.read_text())  # strictly valid JSON (no NaN tokens)

    def test_huge_record_collapses_to_a_marker(self):
        log = PerfLog(self.path)
        log.log("huge", **{f"k{i}": "y" * 480 for i in range(60)})
        self.assertLess(len(self.path.read_bytes()), MAX_LINE + 100)
        r = log.tail(1)[0]
        self.assertTrue(r["_truncated"])
        self.assertIn("k0", r["fields"])

    def test_reserved_field_names_do_not_clobber_the_record(self):
        log = PerfLog(self.path)
        log.log("e", t="mine")
        r = log.tail(1)[0]
        self.assertEqual(r["event"], "e")
        self.assertIsInstance(r["t"], float)
        self.assertEqual(r["field_t"], "mine")

    def test_rotation_keeps_the_size_bounded_and_the_newest_data(self):
        log = PerfLog(self.path, max_bytes=2048, backups=2)
        for i in range(300):
            log.log("tick", i=i, pad="p" * 40)
        files = sorted(p.name for p in self.path.parent.iterdir())
        self.assertEqual(files, ["perf.jsonl", "perf.jsonl.1", "perf.jsonl.2"])
        for p in self.path.parent.iterdir():
            self.assertLessEqual(p.stat().st_size, 2048)
        self.assertGreater(log.stats()["rotations"], 3)
        tail = log.tail(1000)
        self.assertEqual(tail[-1]["i"], 299)
        idx = [r["i"] for r in tail]
        self.assertEqual(idx, sorted(idx))  # chronological across files
        self.assertEqual(idx, list(range(idx[0], 300)))  # contiguous: nothing lost in the retained window
        self.assertLess(len(idx), 300)  # the oldest were dropped

    def test_no_backups(self):
        log = PerfLog(self.path, max_bytes=1024, backups=0)
        for i in range(100):
            log.log("tick", i=i, pad="p" * 40)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["perf.jsonl"])
        self.assertEqual(log.tail(1)[0]["i"], 99)

    def test_read_since(self):
        log = PerfLog(self.path)
        log.log("old")
        time.sleep(0.02)
        mark = time.time()
        time.sleep(0.02)
        log.log("new1")
        log.log("new2")
        self.assertEqual([r["event"] for r in log.read_since(mark)], ["new1", "new2"])
        self.assertEqual(len(log.read_since(0)), 3)
        self.assertEqual(log.read_since(time.time() + 100), [])
        self.assertEqual([r["event"] for r in log.read_since(0, limit=1)], ["new2"])

    def test_corrupt_lines_are_skipped(self):
        log = PerfLog(self.path)
        log.log("good1")
        with open(self.path, "ab") as f:
            f.write(b"{not json\n\n[1,2]\n\xff\xfe garbage\n")
        log.log("good2")
        self.assertEqual([r["event"] for r in log.tail(10)], ["good1", "good2"])

    def test_missing_file_reads_empty(self):
        log = PerfLog(self.path)
        self.assertEqual(log.tail(5), [])
        self.assertEqual(log.read_since(0), [])
        self.assertEqual(log.stats()["size_bytes"], 0)

    def test_never_raises_and_counts_dropped_records(self):
        blocker = self.dir / "file"
        blocker.write_text("x")
        log = PerfLog(blocker / "perf.jsonl")  # the parent "directory" is a file: every write fails
        self.assertFalse(log.log("x", a=1))
        self.assertFalse(log.log("y"))
        s = log.stats()
        self.assertEqual((s["written"], s["dropped"]), (0, 2))
        self.assertTrue(s["last_error"])
        self.assertEqual(log.tail(3), [])

    def test_unserialisable_or_weird_values_do_not_raise(self):
        log = PerfLog(self.path)

        class Bad:
            def __str__(self):
                raise RuntimeError("no str")

        self.assertFalse(log.log("bad", x=Bad()))  # dropped, not raised
        self.assertTrue(log.log("fine"))
        self.assertEqual(log.stats()["dropped"], 1)
        self.assertEqual(log.tail(5)[-1]["event"], "fine")

    def test_threads_do_not_interleave_lines(self):
        log = PerfLog(self.path, max_bytes=10_000_000)

        def work(n):
            for i in range(100):
                log.log("w", n=n, i=i, pad="z" * 100)

        ts = [threading.Thread(target=work, args=(n,)) for n in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 800)
        for line in lines:
            json.loads(line)
        self.assertEqual(log.stats()["written"], 800)

    def test_files_can_be_deleted_afterwards(self):
        log = PerfLog(self.path, max_bytes=1024, backups=1)
        for i in range(50):
            log.log("tick", pad="p" * 60)
        log.tail(5)
        for p in list(self.path.parent.iterdir()):
            p.unlink()  # would fail on Windows if a handle were left open
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_clear(self):
        log = PerfLog(self.path, max_bytes=1024, backups=2)
        for i in range(60):
            log.log("tick", pad="p" * 60)
        log.clear()
        self.assertEqual(list(self.path.parent.iterdir()), [])
        self.assertTrue(log.log("again"))
        self.assertEqual(len(log.tail(5)), 1)

    def test_tiny_max_bytes_is_clamped(self):
        log = PerfLog(self.path, max_bytes=1)
        self.assertGreaterEqual(log.max_bytes, 1024)
        self.assertTrue(log.log("ok"))

    def test_default_path_helper(self):
        import os
        from unittest import mock

        from duckstoken import paths
        with mock.patch.dict(os.environ, {"DUCKSTOKEN_HOME": str(self.dir / "home")}):
            self.assertEqual(paths.perf_log_file().name, "perf.jsonl")
            self.assertEqual(paths.perf_log_file().parent, paths.vgpu_dir())
            self.assertTrue(str(paths.vgpu_dir()).startswith(str(self.dir)))
            self.assertTrue(PerfLog(paths.perf_log_file()).log("x"))


if __name__ == "__main__":
    unittest.main()
