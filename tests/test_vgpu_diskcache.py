"""BoundedCache (LRU, exact budget, thrash detection) and StagingCache (SSD staging of a model, safe deletion)."""
import errno
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckstoken.vgpu import diskcache as dc  # noqa: E402

KB = 1024
MB = 1024 * 1024


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class TestBoundedCache(unittest.TestCase):
    def test_put_get_and_default_sizes(self):
        c = dc.BoundedCache(100)
        self.assertTrue(c.put("a", b"12345"))
        self.assertTrue(c.put("b", "héllo"))                # str is measured in UTF-8 bytes (6)
        self.assertEqual(c.stats()["bytes"], 5 + 6)
        self.assertEqual(c.get("a"), b"12345")
        self.assertIsNone(c.get("zzz"))
        self.assertEqual(c.get("zzz", "dflt"), "dflt")
        with self.assertRaises(TypeError):
            c.put("c", {"not": "sized"})
        self.assertTrue(c.put("c", {"not": "sized"}, size=7))
        self.assertEqual(c.stats()["bytes"], 18)

    def test_eviction_order_is_least_recently_used(self):
        c = dc.BoundedCache(30)
        for k in "abc":
            c.put(k, "x" * 10)
        self.assertEqual(c.keys(), ["a", "b", "c"])
        c.get("a")                                           # a becomes most recent
        self.assertTrue(c.put("d", "y" * 10))
        self.assertNotIn("b", c)                              # b was the least recently used
        self.assertEqual(c.keys(), ["c", "a", "d"])           # oldest first
        self.assertEqual(c.stats()["evictions"], 1)

    def test_budget_is_exact(self):
        c = dc.BoundedCache(10)
        self.assertTrue(c.put("a", b"x" * 10))                # exactly the budget fits
        self.assertEqual(c.stats()["bytes"], 10)
        self.assertEqual(c.stats()["evictions"], 0)
        self.assertTrue(c.put("b", b"y"))                     # one byte more evicts the old one
        self.assertEqual(c.keys(), ["b"])
        self.assertEqual(c.stats()["bytes"], 1)

    def test_item_bigger_than_budget_is_refused_and_cache_unchanged(self):
        c = dc.BoundedCache(10)
        c.put("a", b"12345")
        self.assertFalse(c.put("big", b"x" * 11))
        self.assertFalse(c.put("a", b"x" * 11))               # also for an existing key: old value kept
        self.assertEqual(c.get("a"), b"12345")
        self.assertEqual(c.stats()["bytes"], 5)
        self.assertEqual(c.stats()["evictions"], 0)

    def test_replacing_a_key_adjusts_bytes(self):
        c = dc.BoundedCache(100)
        c.put("a", b"x" * 40)
        c.put("a", b"y" * 10)
        self.assertEqual((len(c), c.stats()["bytes"]), (1, 10))

    def test_max_items(self):
        c = dc.BoundedCache(1000, max_items=2)
        for k in "abc":
            c.put(k, b"x")
        self.assertEqual(c.keys(), ["b", "c"])
        self.assertEqual(c.stats()["evictions"], 1)
        z = dc.BoundedCache(1000, max_items=0)
        self.assertFalse(z.put("a", b"x"))

    def test_zero_and_negative_budget(self):
        self.assertFalse(dc.BoundedCache(0).put("a", b"x"))
        self.assertTrue(dc.BoundedCache(0).put("a", b"", size=0))
        with self.assertRaises(ValueError):
            dc.BoundedCache(-1)

    def test_stats_hit_rate(self):
        c = dc.BoundedCache(100)
        self.assertIsNone(c.stats()["hit_rate"])              # no lookups yet: not 0%
        c.put("a", b"1")
        c.get("a")
        c.get("a")
        c.get("nope")
        s = c.stats()
        self.assertEqual((s["hits"], s["misses"]), (2, 1))
        self.assertAlmostEqual(s["hit_rate"], 2 / 3)

    def test_pop_and_clear(self):
        c = dc.BoundedCache(100)
        c.put("a", b"12")
        c.put("b", b"345")
        self.assertEqual(c.pop("a"), b"12")
        self.assertIsNone(c.pop("a"))
        self.assertEqual(c.stats()["bytes"], 3)
        c.clear()
        self.assertEqual((len(c), c.stats()["bytes"]), (0, 0))

    def test_thrash_detection(self):
        clock = FakeClock()
        c = dc.BoundedCache(20, clock=clock)
        self.assertFalse(c.thrashing())
        for i in range(10):                                   # scan bigger than the cache: pure churn
            c.put(f"k{i}", b"x" * 10)
        self.assertGreater(c.stats()["evictions"], 0)
        self.assertTrue(c.thrashing())
        for _ in range(10):                                   # now it is being useful
            c.get("k9")
        self.assertFalse(c.thrashing())
        clock.t += 120                                        # the churn is outside the window
        self.assertFalse(c.thrashing(window_s=60))
        c.put("z1", b"x" * 20)
        c.put("z2", b"x" * 20)
        self.assertTrue(c.thrashing(window_s=60))

    def test_thread_safe_accounting(self):
        c = dc.BoundedCache(1000)
        errors = []

        def work(n):
            try:
                for i in range(300):
                    c.put(f"{n}-{i % 20}", b"x" * (1 + i % 7))
                    c.get(f"{n}-{(i * 3) % 20}")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        ts = [threading.Thread(target=work, args=(n,)) for n in range(6)]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        self.assertEqual(errors, [])
        s = c.stats()
        self.assertLessEqual(s["bytes"], 1000)
        self.assertEqual(s["bytes"], sum(len(c.get(k)) for k in c.keys()))


class FakeCancel:
    """Event-like object whose `is_set` turns True after N checks."""

    def __init__(self, after: int):
        self.after, self.n = after, 0

    def is_set(self):
        self.n += 1
        return self.n > self.after


class StagingBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.src_dir = self.root / "slow"
        self.cache_dir = self.root / "fast" / "staging"
        self.src_dir.mkdir()
        self.disks = [{"drive": str(self.src_dir), "kind": "hdd"}, {"drive": str(self.root / "fast"), "kind": "nvme"}]
        self.hw = {"disks": self.disks}
        self.cache = self.make_cache(budget_gb=1.0)

    def make_cache(self, budget_gb=1.0, min_free_gb=0.0, free_fn=None):
        cache = dc.StagingCache(self.cache_dir, budget_gb, min_free_gb, free_fn=free_fn, chunk_bytes=64 * KB)
        return cache

    def model(self, name="model.gguf", size=3 * MB, fill=b"abc"):
        p = self.src_dir / name
        block = (fill * (size // len(fill) + 1))[:size]
        p.write_bytes(block)
        return p

    def tree(self):
        """Every path under the cache dir, relative."""
        if not self.cache_dir.exists():
            return []
        return sorted(str(p.relative_to(self.cache_dir)) for p in self.cache_dir.rglob("*"))


class TestPlanStage(StagingBase):
    def test_worthwhile_hdd_to_nvme_and_nothing_is_copied(self):
        p = self.model()
        plan = self.cache.plan_stage(p, self.hw)
        self.assertTrue(plan["worthwhile"], plan["reason"])
        self.assertEqual((plan["src_kind"], plan["cache_kind"]), ("hdd", "nvme"))
        self.assertEqual(plan["size_bytes"], 3 * MB)
        self.assertFalse(plan["already_staged"])
        self.assertEqual(self.tree(), [])                       # planning has no side effects

    def test_not_worthwhile_when_source_already_fast(self):
        p = self.model()
        self.disks[0]["kind"] = "nvme"
        plan = self.cache.plan_stage(p, self.hw)
        self.assertFalse(plan["worthwhile"])
        self.assertIn("already on a fast drive", plan["reason"])

    def test_not_worthwhile_when_cache_drive_is_not_known_fast(self):
        p = self.model()
        for kind in ("hdd", "unknown"):
            self.disks[1]["kind"] = kind
            plan = self.cache.plan_stage(p, self.hw)
            self.assertFalse(plan["worthwhile"], kind)
            self.assertIn("SSD", plan["reason"])

    def test_unknown_source_drive_counts_as_slow(self):
        p = self.model()
        self.disks[0]["kind"] = "unknown"
        self.assertTrue(self.cache.plan_stage(p, self.hw)["worthwhile"])

    def test_no_disk_info_at_all_is_not_worthwhile(self):
        plan = self.cache.plan_stage(self.model(), {"disks": []})
        self.assertFalse(plan["worthwhile"])

    def test_too_big_for_budget(self):
        p = self.model(size=3 * MB)
        small = dc.StagingCache(self.cache_dir, budget_gb=2 * MB / dc.GIB, min_free_gb=0)
        plan = small.plan_stage(p, self.hw)
        self.assertFalse(plan["worthwhile"])
        self.assertIn("larger than the staging budget", plan["reason"])

    def test_not_enough_free_space_mentions_numbers(self):
        p = self.model()
        cache = self.make_cache(min_free_gb=10, free_fn=lambda: 5 * dc.GIB)
        plan = cache.plan_stage(p, self.hw)
        self.assertFalse(plan["worthwhile"])
        self.assertIn("Not enough free space", plan["reason"])
        self.assertIn("5.0 GB free", plan["reason"])
        ok = self.make_cache(min_free_gb=10, free_fn=lambda: 50 * dc.GIB)
        self.assertTrue(ok.plan_stage(p, self.hw)["worthwhile"])

    def test_missing_source_and_unreadable_free_space(self):
        self.assertFalse(self.cache.plan_stage(self.src_dir / "nope.gguf", self.hw)["worthwhile"])

        def boom():
            raise OSError("no")

        cache = self.make_cache(free_fn=boom)
        plan = cache.plan_stage(self.model(), self.hw)
        self.assertFalse(plan["worthwhile"])
        self.assertIn("Cannot read the free space", plan["reason"])

    def test_already_staged_is_reported(self):
        p = self.model()
        self.assertTrue(self.cache.stage(p, confirm=True)["ok"])
        plan = self.cache.plan_stage(p, self.hw)
        self.assertFalse(plan["worthwhile"])
        self.assertTrue(plan["already_staged"])
        self.assertTrue(plan["staged_path"].endswith("model.gguf"))

    def test_plan_lists_what_would_be_evicted(self):
        a, b = self.model("a.gguf", 3 * MB), self.model("b.gguf", 3 * MB, fill=b"xyz")
        cache = dc.StagingCache(self.cache_dir, budget_gb=4 * MB / dc.GIB, min_free_gb=0, chunk_bytes=64 * KB)
        self.assertTrue(cache.stage(a, confirm=True)["ok"])
        plan = cache.plan_stage(b, self.hw)
        self.assertTrue(plan["worthwhile"])
        self.assertEqual(len(plan["would_evict"]), 1)
        self.assertTrue(plan["would_evict"][0].startswith("a-"))

    def test_posix_disk_lookup_prefers_longest_mount(self):
        if os.name == "nt":
            self.skipTest("POSIX mount-point matching")
        disks = [{"drive": "/", "kind": "ssd"}, {"drive": str(self.src_dir), "kind": "hdd"}]
        self.assertEqual(dc.StagingCache._disk_for(str(self.src_dir / "m.gguf"), {"disks": disks})["kind"], "hdd")
        self.assertEqual(dc.StagingCache._disk_for(str(self.root / "elsewhere.gguf"), {"disks": disks})["kind"], "ssd")
        self.assertIsNone(dc.StagingCache._disk_for("/x", {"disks": []}))


class TestStage(StagingBase):
    def test_refuses_without_confirm_and_creates_nothing(self):
        p = self.model()
        for confirm in (False, None, 1, "yes"):
            r = self.cache.stage(p, confirm=confirm)
            self.assertFalse(r["ok"])
            self.assertEqual(r["code"], "confirm_required")
        self.assertFalse(self.cache.stage(p)["ok"])
        self.assertFalse(self.cache_dir.exists())

    def test_copy_is_exact_verified_and_leaves_no_part_files(self):
        p = self.model(size=3 * MB + 123)
        calls = []
        r = self.cache.stage(p, confirm=True, progress=lambda *a: calls.append(a))
        self.assertTrue(r["ok"], r)
        staged = Path(r["path"])
        self.assertEqual(staged.read_bytes(), p.read_bytes())
        self.assertEqual(r["size_bytes"], 3 * MB + 123)
        self.assertTrue(r["verified"]["size"] and r["verified"]["edges_sha256"])
        self.assertEqual(r["mb_per_s"]["status"], "measured")
        self.assertFalse(any(t.endswith(".part") for t in self.tree()))
        self.assertEqual(self.cache.usage()["pending"], [])
        done = [c[0] for c in calls]
        self.assertEqual(done, sorted(done))
        self.assertEqual(done[-1], 3 * MB + 123)
        self.assertEqual(calls[0][1], 3 * MB + 123)
        self.assertEqual(self.cache.lookup(p), str(staged))
        self.assertEqual(p.read_bytes()[:3], b"abc")            # the original is untouched

    def test_second_stage_of_current_copy_is_a_no_op(self):
        p = self.model()
        self.cache.stage(p, confirm=True)
        r = self.cache.stage(p, confirm=True)
        self.assertTrue(r["ok"])
        self.assertTrue(r["already_staged"])

    def test_split_model_all_shards_staged_together(self):
        a = self.model("big-00001-of-00002.gguf", MB, fill=b"AAA")
        b = self.model("big-00002-of-00002.gguf", MB + 5, fill=b"BBB")
        r = self.cache.stage(a, confirm=True)
        self.assertTrue(r["ok"], r)
        staged = Path(r["path"])
        self.assertEqual(staged.name, "big-00001-of-00002.gguf")
        self.assertEqual((staged.parent / "big-00002-of-00002.gguf").read_bytes(), b.read_bytes())
        self.assertEqual(r["size_bytes"], 2 * MB + 5)
        self.assertEqual(self.cache.lookup(a), str(staged))

    def test_lookup_refuses_a_copy_when_the_original_changed(self):
        p = self.model()
        self.cache.stage(p, confirm=True)
        self.assertIsNotNone(self.cache.lookup(p))
        p.write_bytes(b"x" * (3 * MB))                          # same size, different mtime/content
        os.utime(p, ns=(1, 1))
        self.assertIsNone(self.cache.lookup(p))
        self.assertFalse(self.cache.list()[0]["current"])
        r = self.cache.stage(p, confirm=True)                   # restaging replaces the outdated copy
        self.assertTrue(r["ok"] and not r["already_staged"])
        self.assertEqual(Path(r["path"]).read_bytes(), p.read_bytes())
        self.assertEqual(len(self.cache.list()), 1)

    def test_lookup_misses_when_staged_file_was_deleted_or_never_staged(self):
        p = self.model()
        self.assertIsNone(self.cache.lookup(p))
        r = self.cache.stage(p, confirm=True)
        Path(r["path"]).unlink()
        self.assertIsNone(self.cache.lookup(p))

    def test_lookup_survives_a_manifest_that_cannot_be_saved(self):
        # review finding: StoreError (not OSError) escaped lookup() and blocked starting the model server
        from unittest import mock
        from duckstoken.vgpu.store import StoreError
        p = self.model()
        self.assertTrue(self.cache.stage(p, confirm=True)["ok"])
        with mock.patch.object(self.cache, "_save", side_effect=StoreError("cannot write manifest.json: [Errno 28] No space left")):
            staged = self.cache.lookup(p)
        self.assertIsNotNone(staged)
        self.assertTrue(Path(staged).is_file())

    def test_stage_reports_an_unwritable_manifest_instead_of_raising(self):
        from unittest import mock
        from duckstoken.vgpu.store import StoreError
        p = self.model()
        with mock.patch.object(self.cache, "_save", side_effect=StoreError("cannot write manifest.json")):
            r = self.cache.stage(p, confirm=True)
        self.assertFalse(r["ok"])
        self.assertEqual(r["code"], "io_error")

    def test_lookup_bumps_last_used(self):
        p = self.model()
        self.cache.stage(p, confirm=True)
        t0 = self.cache.list()[0]["last_used"]
        with mock.patch("duckstoken.vgpu.staging.time.time", return_value=t0 + 500):
            self.cache.lookup(p)
        self.assertEqual(self.cache.list()[0]["last_used"], t0 + 500)

    def test_cancel_midway_removes_partial_and_registration(self):
        p = self.model(size=3 * MB)
        r = self.cache.stage(p, confirm=True, cancel=FakeCancel(after=5))
        self.assertFalse(r["ok"])
        self.assertEqual(r["code"], "cancelled")
        self.assertTrue(r["cleaned"])
        self.assertEqual([t for t in self.tree() if t != "manifest.json"], [])
        self.assertEqual(self.cache.usage()["pending"], [])
        self.assertEqual(self.cache.list(), [])
        self.assertEqual(p.stat().st_size, 3 * MB)
        self.assertTrue(self.cache.stage(p, confirm=True)["ok"])  # and a retry works

    def test_threading_event_cancel(self):
        ev = threading.Event()
        ev.set()
        r = self.cache.stage(self.model(), confirm=True, cancel=ev)
        self.assertEqual(r["code"], "cancelled")

    def test_disk_full_during_copy_cleans_up(self):
        p = self.model()
        with mock.patch("duckstoken.vgpu.staging.os.fsync", side_effect=OSError(errno.ENOSPC, "No space left on device")):
            r = self.cache.stage(p, confirm=True)
        self.assertFalse(r["ok"])
        self.assertEqual(r["code"], "disk_full")
        self.assertTrue(r["cleaned"])
        self.assertEqual([t for t in self.tree() if t != "manifest.json"], [])
        self.assertEqual(self.cache.usage()["pending"], [])

    def test_other_io_error_is_reported_not_raised(self):
        p = self.model()
        with mock.patch("duckstoken.vgpu.staging.os.fsync", side_effect=OSError(errno.EIO, "I/O error")):
            r = self.cache.stage(p, confirm=True)
        self.assertEqual(r["code"], "io_error")
        self.assertEqual([t for t in self.tree() if t != "manifest.json"], [])

    def test_free_space_precheck_blocks_before_copying(self):
        p = self.model()
        cache = self.make_cache(min_free_gb=10, free_fn=lambda: 1 * dc.GIB)
        r = cache.stage(p, confirm=True)
        self.assertEqual(r["code"], "disk_full")
        self.assertEqual([t for t in self.tree() if t != "manifest.json"], [])

    def test_larger_than_budget(self):
        p = self.model(size=3 * MB)
        small = dc.StagingCache(self.cache_dir, budget_gb=2 * MB / dc.GIB, min_free_gb=0)
        r = small.stage(p, confirm=True)
        self.assertEqual(r["code"], "over_budget")

    def test_source_changed_during_copy_is_detected(self):
        p = self.model()
        real = self.cache._copy_file

        def sneaky(src, dst, done, total, progress, cancel):
            res = real(src, dst, done, total, progress, cancel)
            os.utime(src, ns=(5, 5))                            # somebody touched the original mid-copy
            return res

        with mock.patch.object(self.cache, "_copy_file", side_effect=sneaky):
            r = self.cache.stage(p, confirm=True)
        self.assertEqual(r["code"], "source_changed")
        self.assertEqual([t for t in self.tree() if t != "manifest.json"], [])

    def test_verification_failure_removes_copy(self):
        p = self.model()
        with mock.patch.object(dc.StagingCache, "_edges", side_effect=["aaa", "bbb"]):
            r = self.cache.stage(p, confirm=True)
        self.assertEqual(r["code"], "verify_failed")
        self.assertEqual([t for t in self.tree() if t != "manifest.json"], [])

    def test_wrong_size_copy_is_rejected(self):
        p = self.model()

        def short(src, dst, done, total, progress, cancel):
            dst.write_bytes(b"short")
            return None

        with mock.patch.object(self.cache, "_copy_file", side_effect=short):
            r = self.cache.stage(p, confirm=True)
        self.assertEqual(r["code"], "verify_failed")
        self.assertEqual([t for t in self.tree() if t != "manifest.json"], [])

    def test_unexpected_exception_still_cleans_up_and_propagates(self):
        p = self.model()

        def explode(src, dst, done, total, progress, cancel):
            dst.write_bytes(b"half")
            raise RuntimeError("bug")

        with mock.patch.object(self.cache, "_copy_file", side_effect=explode):
            with self.assertRaises(RuntimeError):
                self.cache.stage(p, confirm=True)
        self.assertEqual([t for t in self.tree() if t != "manifest.json"], [])
        self.assertEqual(self.cache.usage()["pending"], [])
        self.assertTrue(self.cache.stage(p, confirm=True)["ok"])   # the lock was released

    def test_missing_source_and_source_inside_cache(self):
        self.assertEqual(self.cache.stage(self.src_dir / "nope.gguf", confirm=True)["code"], "source_missing")
        p = self.model()
        r = self.cache.stage(p, confirm=True)
        self.assertEqual(self.cache.stage(r["path"], confirm=True)["code"], "source_in_cache")

    def test_hw_given_and_not_worthwhile_refuses(self):
        p = self.model()
        self.disks[0]["kind"] = "nvme"
        r = self.cache.stage(p, confirm=True, hw=self.hw)
        self.assertEqual(r["code"], "not_worthwhile")
        self.assertIn("already on a fast drive", r["error"])
        self.assertFalse(self.cache_dir.exists())
        self.disks[0]["kind"] = "hdd"
        self.assertTrue(self.cache.stage(p, confirm=True, hw=self.hw)["ok"])

    def test_only_one_stage_at_a_time(self):
        p, q = self.model("a.gguf"), self.model("b.gguf", fill=b"zzz")
        gate, release = threading.Event(), threading.Event()
        real = self.cache._copy_file

        def slow(*a):
            gate.set()
            release.wait(10)
            return real(*a)

        results = {}
        with mock.patch.object(self.cache, "_copy_file", side_effect=slow):
            t = threading.Thread(target=lambda: results.update(first=self.cache.stage(p, confirm=True)))
            t.start()
            self.assertTrue(gate.wait(10))
            second = self.cache.stage(q, confirm=True)
            release.set()
            t.join(20)
        self.assertEqual(second["code"], "busy")
        self.assertTrue(results["first"]["ok"])


class TestEvictionAndSafety(StagingBase):
    def small_cache(self):
        return dc.StagingCache(self.cache_dir, budget_gb=7 * MB / dc.GIB, min_free_gb=0, chunk_bytes=64 * KB)

    def test_lru_eviction_removes_only_the_least_recently_used_copy(self):
        cache = self.small_cache()
        a, b, c = self.model("a.gguf", 3 * MB), self.model("b.gguf", 3 * MB, fill=b"bbb"), self.model("c.gguf", 3 * MB, fill=b"ccc")
        ra, rb = cache.stage(a, confirm=True), cache.stage(b, confirm=True)
        self.assertTrue(ra["ok"] and rb["ok"])
        with mock.patch("duckstoken.vgpu.staging.time.time", return_value=time_future()):
            cache.lookup(a)                                     # a is now more recent than b
        rc = cache.stage(c, confirm=True)
        self.assertTrue(rc["ok"], rc)
        names = {e["name"].split("-")[0] for e in cache.list()}
        self.assertEqual(names, {"a", "c"})
        self.assertFalse(Path(rb["path"]).exists())
        self.assertFalse(Path(rb["path"]).parent.exists())      # its (now empty) directory went too
        self.assertTrue(b.exists() and a.exists() and c.exists())   # originals never touched
        self.assertLessEqual(cache.usage()["used_bytes"], cache.budget_bytes)

    def test_never_deletes_files_it_did_not_create(self):
        cache = self.small_cache()
        self.cache_dir.mkdir(parents=True)
        foreign_file = self.cache_dir / "notes.txt"
        foreign_file.write_text("mine")
        foreign_dir = self.cache_dir / "user-stuff"
        foreign_dir.mkdir()
        (foreign_dir / "keep.bin").write_bytes(b"1")
        stray_part = self.cache_dir / "random.part"
        stray_part.write_bytes(b"not ours")
        a, b, c = self.model("a.gguf", 3 * MB), self.model("b.gguf", 3 * MB, fill=b"bbb"), self.model("c.gguf", 3 * MB, fill=b"ccc")
        for m in (a, b, c):
            self.assertTrue(cache.stage(m, confirm=True)["ok"])
        for e in cache.list():
            cache.evict(e["name"])
        cache.cleanup()
        self.assertEqual(foreign_file.read_text(), "mine")
        self.assertEqual((foreign_dir / "keep.bin").read_bytes(), b"1")
        self.assertEqual(stray_part.read_bytes(), b"not ours")
        usage = cache.usage()
        self.assertIn("notes.txt", usage["unmanaged"])           # reported, not deleted
        self.assertIn("random.part", usage["unmanaged"])

    def test_evict_unknown_name_deletes_nothing(self):
        cache = self.small_cache()
        self.cache_dir.mkdir(parents=True)
        victim = self.cache_dir / "important.txt"
        victim.write_text("x")
        for bad in ("important.txt", "../fast", "", ".", "..", "a/b"):
            r = cache.evict(bad)
            self.assertFalse(r["ok"], bad)
            self.assertEqual(r["code"], "not_found")
        self.assertTrue(victim.exists())

    def test_tampered_manifest_cannot_make_it_delete_outside_files(self):
        cache = self.small_cache()
        outside = self.root / "outside.txt"
        outside.write_text("precious")
        inner = self.cache_dir / "legit"
        inner.mkdir(parents=True)
        (inner / "real.gguf").write_bytes(b"1234")
        manifest = {"version": 1, "pending": ["../../outside.txt", "legit/../../outside.txt", "/etc/passwd"],
                    "entries": {"legit": {"name": "legit", "src": "/x", "size": 4, "last_used": 1,
                                          "files": [{"name": "../../outside.txt", "size": 8},
                                                    {"name": "..", "size": 1}, {"name": "real.gguf", "size": 4}]}}}
        cache.manifest_path.write_text(json.dumps(manifest))
        cache.cleanup()
        r = cache.evict("legit")
        self.assertTrue(r["ok"])
        self.assertEqual(outside.read_text(), "precious")
        self.assertFalse((inner / "real.gguf").exists())         # only the safe, listed file was removed

    def test_symlink_pointing_outside_is_not_followed(self):
        if os.name == "nt":
            self.skipTest("symlinks need privileges on Windows")
        cache = self.small_cache()
        outside = self.root / "outside.bin"
        outside.write_bytes(b"precious")
        d = self.cache_dir / "entry"
        d.mkdir(parents=True)
        os.symlink(outside, d / "link.gguf")
        manifest = {"version": 1, "pending": [], "entries": {"entry": {"name": "entry", "src": "/x", "size": 8, "last_used": 1,
                                                                        "files": [{"name": "link.gguf", "size": 8}]}}}
        cache.manifest_path.write_text(json.dumps(manifest))
        cache.evict("entry")
        self.assertEqual(outside.read_bytes(), b"precious")

    def test_cleanup_removes_registered_partials_only(self):
        cache = self.small_cache()
        d = self.cache_dir / "crashed-12345678"
        d.mkdir(parents=True)
        part = d / "model.gguf.part"
        part.write_bytes(b"half a model")
        foreign = self.cache_dir / "other.part"
        foreign.write_bytes(b"x")
        cache.manifest_path.write_text(json.dumps(
            {"version": 1, "entries": {}, "pending": ["crashed-12345678/model.gguf.part"]}))
        self.assertEqual(cache.usage()["pending"], ["crashed-12345678/model.gguf.part"])
        self.assertEqual(cache.cleanup(), {"removed": 1})
        self.assertFalse(part.exists())
        self.assertFalse(d.exists())
        self.assertTrue(foreign.exists())
        self.assertEqual(cache.usage()["pending"], [])

    def test_next_stage_cleans_partials_left_by_a_crash(self):
        cache = self.small_cache()
        p = self.model()
        name = cache._entry_name(cache._norm(p))
        d = self.cache_dir / name
        d.mkdir(parents=True)
        (d / "model.gguf.part").write_bytes(b"stale partial from a crash")
        cache.manifest_path.write_text(json.dumps(
            {"version": 1, "entries": {}, "pending": [f"{name}/model.gguf.part"]}))
        r = cache.stage(p, confirm=True)
        self.assertTrue(r["ok"], r)
        self.assertEqual(Path(r["path"]).read_bytes(), p.read_bytes())
        self.assertFalse(any(t.endswith(".part") for t in self.tree()))

    def test_evict_reports_in_use_and_keeps_entry(self):
        cache = self.small_cache()
        r = cache.stage(self.model(), confirm=True)
        name = cache.list()[0]["name"]
        with mock.patch.object(Path, "unlink", side_effect=PermissionError("file is in use")):
            res = cache.evict(name)
        self.assertFalse(res["ok"])
        self.assertEqual(res["code"], "in_use")
        self.assertEqual(len(cache.list()), 1)
        self.assertTrue(Path(r["path"]).exists())
        self.assertTrue(cache.evict(name)["ok"])

    def test_evict_frees_bytes_and_usage_is_consistent(self):
        cache = self.small_cache()
        cache.stage(self.model(size=2 * MB), confirm=True)
        u = cache.usage()
        self.assertEqual((u["used_bytes"], u["entries"]), (2 * MB, 1))
        self.assertEqual(u["free_budget_bytes"], u["budget_bytes"] - 2 * MB)
        self.assertEqual(u["disk_free_gb"]["status"], "measured")
        res = cache.evict(cache.list()[0]["name"])
        self.assertEqual(res, {"ok": True, "freed_bytes": 2 * MB})
        self.assertEqual(cache.usage()["used_bytes"], 0)
        self.assertEqual([t for t in self.tree() if t != "manifest.json"], [])

    def test_corrupt_manifest_is_treated_as_empty_and_files_survive(self):
        cache = self.small_cache()
        r = cache.stage(self.model(), confirm=True)
        cache.manifest_path.write_text("{ broken")
        self.assertEqual(cache.list(), [])
        self.assertTrue(Path(r["path"]).exists())                # unknown files are never deleted
        self.assertTrue(cache.stage(self.model(), confirm=True)["ok"])


def time_future():
    return time.time() + 1000


if __name__ == "__main__":
    unittest.main()
