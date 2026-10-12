"""VGpuStore: bounded JSONL + profile persistence, corruption tolerance, atomic writes, profile keys."""
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckstoken.vgpu import store as st  # noqa: E402


def hw(ram=48.0, gpu="NVIDIA GeForce GTX 750", vram=2048, cpu="Intel Core i7-8700K", cores=6, logical=12):
    return {"cpu": {"name": cpu, "physical_cores": cores, "logical_cores": logical},
            "ram": {"total_gb": ram}, "gpus": [{"name": gpu, "vram_mb": vram}] if gpu else [], "disks": []}


MODEL = {"name": "gpt-oss-20b", "size_bytes": 12_100_000_000, "file_type": 38, "path": "/m/a.gguf"}


class StoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "vgpu"
        self.store = st.VGpuStore(self.dir, max_records=5, fsync=False)


class TestBenchmarks(StoreCase):
    def test_add_assigns_id_and_created_and_roundtrips(self):
        rec = self.store.add_benchmark({"label": "x", "model": {"name": "m1", "path": "/p/m1.gguf"}, "config_hash": "h1"})
        self.assertTrue(rec["id"].startswith("bench-"))
        self.assertRegex(rec["created"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual(self.store.get_benchmark(rec["id"]), rec)
        self.assertIsNone(self.store.get_benchmark("nope"))

    def test_caller_supplied_id_is_kept_and_replaces_same_id(self):
        self.store.add_benchmark({"id": "fixed", "v": 1})
        self.store.add_benchmark({"id": "fixed", "v": 2})
        self.assertEqual(self.store.get_benchmark("fixed")["v"], 2)
        self.assertEqual(len(self.store.benchmarks(10)), 1)

    def test_newest_first_and_limit(self):
        for i in range(4):
            self.store.add_benchmark({"id": f"b{i}"})
        self.assertEqual([r["id"] for r in self.store.benchmarks(limit=3)], ["b3", "b2", "b1"])
        self.assertEqual(self.store.benchmarks(limit=0), [])

    def test_filters_by_model_name_path_and_config_hash(self):
        self.store.add_benchmark({"id": "a", "model": {"name": "m1", "path": "/x/m1.gguf"}, "config_hash": "c1"})
        self.store.add_benchmark({"id": "b", "model": {"name": "m2", "path": "/x/m2.gguf"}, "config_hash": "c2"})
        self.store.add_benchmark({"id": "c", "model": {"name": "m1", "path": "/x/m1.gguf"}, "config_hash": "c2"})
        self.assertEqual([r["id"] for r in self.store.benchmarks(model="m1")], ["c", "a"])
        self.assertEqual([r["id"] for r in self.store.benchmarks(model="/x/m2.gguf")], ["b"])
        self.assertEqual([r["id"] for r in self.store.benchmarks(config_hash="c2")], ["c", "b"])
        self.assertEqual([r["id"] for r in self.store.benchmarks(model="m1", config_hash="c2")], ["c"])
        self.assertEqual(self.store.benchmarks(model="unknown"), [])

    def test_bounded_oldest_dropped(self):
        for i in range(9):
            self.store.add_benchmark({"id": f"b{i}"})
        ids = [r["id"] for r in self.store.benchmarks(100)]
        self.assertEqual(ids, ["b8", "b7", "b6", "b5", "b4"])
        self.assertEqual(len(self.store.benchmarks_path.read_text().splitlines()), 5)

    def test_delete(self):
        self.store.add_benchmark({"id": "a"})
        self.store.add_benchmark({"id": "b"})
        self.assertTrue(self.store.delete_benchmark("a"))
        self.assertFalse(self.store.delete_benchmark("a"))
        self.assertEqual([r["id"] for r in self.store.benchmarks(10)], ["b"])

    def test_corrupted_lines_are_skipped_not_fatal(self):
        self.store.add_benchmark({"id": "good1"})
        with open(self.store.benchmarks_path, "a", encoding="utf-8") as f:
            f.write("{this is not json\n")
            f.write("[1, 2, 3]\n")          # valid JSON but not a record
            f.write("\n")
            f.write('{"id": "truncated", "x": ')   # torn write at the end, no newline
        self.assertEqual([r["id"] for r in self.store.benchmarks(10)], ["good1"])
        self.assertGreaterEqual(self.store.corrupt_lines, 2)
        self.store.add_benchmark({"id": "good2"})      # the next write also heals the file
        self.assertEqual([r["id"] for r in self.store.benchmarks(10)], ["good2", "good1"])
        for line in self.store.benchmarks_path.read_text().splitlines():
            json.loads(line)

    def test_missing_dir_is_created_on_first_write_only(self):
        self.assertFalse(self.dir.exists())
        self.assertEqual(self.store.benchmarks(), [])
        self.assertFalse(self.dir.exists())
        self.store.add_benchmark({"id": "a"})
        self.assertTrue(self.store.benchmarks_path.exists())

    def test_atomic_write_failure_keeps_old_file_and_cleans_tmp(self):
        self.store.add_benchmark({"id": "a"})
        before = self.store.benchmarks_path.read_text()
        with mock.patch("duckstoken.vgpu.store.os.replace", side_effect=OSError("disk exploded")):
            with self.assertRaises(st.StoreError):
                self.store.add_benchmark({"id": "b"})
        self.assertEqual(self.store.benchmarks_path.read_text(), before)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["benchmarks.jsonl"])  # no stray .tmp

    def test_unwritable_location_raises_store_error(self):
        blocker = Path(self._tmp.name) / "afile"
        blocker.write_text("x")
        bad = st.VGpuStore(blocker / "sub")
        with self.assertRaises(st.StoreError):
            bad.add_benchmark({"id": "a"})

    def test_oversized_record_loses_bulky_fields_then_is_refused(self):
        big = {"id": "big", "label": "keep", "log_tail": "x" * (st.MAX_RECORD_BYTES + 10)}
        rec = self.store.add_benchmark(big)
        self.assertNotIn("log_tail", rec)
        self.assertTrue(rec["truncated"])
        with self.assertRaises(ValueError):
            self.store.add_benchmark({"id": "huge", "blob": "y" * (st.MAX_RECORD_BYTES + 10)})
        self.assertIsNone(self.store.get_benchmark("huge"))

    def test_thread_safety_no_lost_records_no_corruption(self):
        store = st.VGpuStore(self.dir, max_records=500, fsync=False)
        errors = []

        def work(n):
            try:
                for i in range(10):
                    store.add_benchmark({"id": f"t{n}-{i}"})
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
        [t.start() for t in threads]
        [t.join(30) for t in threads]
        self.assertEqual(errors, [])
        self.assertEqual(len(store.benchmarks(1000)), 60)
        self.assertEqual(store.corrupt_lines, 0)

    def test_non_json_values_are_stringified_not_fatal(self):
        rec = self.store.add_benchmark({"id": "p", "path": Path("/a/b"), "s": {1, 2}.__class__})
        self.assertIsInstance(rec["path"], str)


class TestHistoryAndProfiles(StoreCase):
    def test_history_newest_first_bounded(self):
        for i in range(8):
            self.store.add_history({"n": i})
        self.assertEqual([r["n"] for r in self.store.history(3)], [7, 6, 5])
        self.assertEqual(len(self.store.history(100)), 5)
        self.assertTrue(self.store.history(1)[0]["id"].startswith("opt-"))

    def test_profile_roundtrip_overwrite_delete(self):
        key = st.profile_key(hw(), MODEL, "build 1")
        self.assertIsNone(self.store.get_profile(key))
        saved = self.store.save_profile(key, {"patch": {"n_gpu_layers": 0}})
        self.assertEqual(saved["key"], key)
        self.assertIn("saved", saved)
        self.assertEqual(self.store.get_profile(key)["patch"], {"n_gpu_layers": 0})
        self.store.save_profile(key, {"patch": {"n_gpu_layers": 8}})
        self.assertEqual(self.store.get_profile(key)["patch"], {"n_gpu_layers": 8})
        self.assertEqual(list(self.store.profiles()), [key])
        self.assertTrue(self.store.delete_profile(key))
        self.assertFalse(self.store.delete_profile(key))
        self.assertEqual(self.store.profiles(), {})

    def test_corrupt_profiles_file_reads_as_empty_and_heals(self):
        self.dir.mkdir(parents=True)
        self.store.profiles_path.write_text("{ not json", encoding="utf-8")
        self.assertEqual(self.store.profiles(), {})
        self.store.save_profile("k", {"a": 1})
        self.assertEqual(self.store.get_profile("k")["a"], 1)

    def test_profiles_bounded_drops_oldest(self):
        for i in range(8):
            with mock.patch("duckstoken.vgpu.store._now_iso", return_value=f"2026-01-0{i + 1}T00:00:00Z"):
                self.store.save_profile(f"k{i}", {"i": i})
        self.assertEqual(sorted(self.store.profiles()), ["k3", "k4", "k5", "k6", "k7"])

    def test_stats(self):
        self.store.add_benchmark({"id": "a"})
        self.store.add_history({"x": 1})
        self.store.save_profile("k", {})
        s = self.store.stats()
        self.assertEqual((s["benchmarks"], s["history"], s["profiles"]), (1, 1, 1))


class TestProfileKey(unittest.TestCase):
    def test_available_as_module_function_and_as_store_method(self):
        self.assertEqual(st.VGpuStore.profile_key(hw(), MODEL, "b"), st.profile_key(hw(), MODEL, "b"))
        self.assertEqual(st.VGpuStore("/nonexistent/never/created").profile_key(hw(), MODEL, "b"), st.profile_key(hw(), MODEL, "b"))

    def test_stable_and_wellformed(self):
        a = st.profile_key(hw(), MODEL, "build 7")
        b = st.profile_key(hw(), dict(MODEL, path="/other/place.gguf"), "build 7")
        self.assertEqual(a, b)                      # the path is not part of the identity
        self.assertRegex(a, r"^[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}$")

    def test_changes_with_each_ingredient(self):
        base = st.profile_key(hw(), MODEL, "b1")
        variants = {
            "ram": st.profile_key(hw(ram=64), MODEL, "b1"),
            "gpu name": st.profile_key(hw(gpu="NVIDIA GeForce RTX 3060"), MODEL, "b1"),
            "vram": st.profile_key(hw(vram=4096), MODEL, "b1"),
            "no gpu": st.profile_key(hw(gpu=None), MODEL, "b1"),
            "cpu name": st.profile_key(hw(cpu="AMD Ryzen 5"), MODEL, "b1"),
            "cores": st.profile_key(hw(cores=4, logical=8), MODEL, "b1"),
            "model name": st.profile_key(hw(), dict(MODEL, name="other"), "b1"),
            "model size": st.profile_key(hw(), dict(MODEL, size_bytes=MODEL["size_bytes"] + 1), "b1"),
            "quant": st.profile_key(hw(), dict(MODEL, file_type=15), "b1"),
            "engine": st.profile_key(hw(), MODEL, "b2"),
            "engine unknown": st.profile_key(hw(), MODEL, None),
        }
        for what, k in variants.items():
            self.assertNotEqual(k, base, what)
        self.assertEqual(len(set(variants.values())), len(variants))

    def test_ram_rounding_absorbs_tiny_differences(self):
        self.assertEqual(st.profile_key(hw(ram=47.9), MODEL, "b"), st.profile_key(hw(ram=48.2), MODEL, "b"))

    def test_gpu_order_does_not_matter(self):
        h1, h2 = hw(), hw()
        h1["gpus"] = [{"name": "A", "vram_mb": 1}, {"name": "B", "vram_mb": 2}]
        h2["gpus"] = list(reversed(h1["gpus"]))
        self.assertEqual(st.hardware_fingerprint(h1), st.hardware_fingerprint(h2))

    def test_tolerates_empty_inputs(self):
        self.assertRegex(st.profile_key({}, {}, None), r"^[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}$")


class TestProfileStatus(StoreCase):
    def test_missing_fresh_stale(self):
        key = st.profile_key(hw(), MODEL, "b1")
        self.assertEqual(self.store.profile_status(key), "missing")
        self.store.save_profile(key, {"patch": {}})
        self.assertEqual(self.store.profile_status(key), "fresh")
        # same model, more RAM  -> tuned on different hardware
        self.assertEqual(self.store.profile_status(st.profile_key(hw(ram=64), MODEL, "b1")), "stale")
        # same model + hardware, new engine build
        self.assertEqual(self.store.profile_status(st.profile_key(hw(), MODEL, "b2")), "stale")
        # a different model was never tuned
        self.assertEqual(self.store.profile_status(st.profile_key(hw(), dict(MODEL, name="z"), "b1")), "missing")
        self.assertEqual(self.store.profile_status("not-a-key"), "missing")

    def test_stale_profiles_lists_the_old_ones(self):
        old = st.profile_key(hw(), MODEL, "b1")
        self.store.save_profile(old, {"patch": {"x": 1}})
        newer = st.profile_key(hw(ram=64), MODEL, "b1")
        self.assertEqual([p["key"] for p in self.store.stale_profiles(newer)], [old])
        self.assertEqual(self.store.stale_profiles(old), [])
        self.assertEqual(self.store.stale_profiles("garbage"), [])


if __name__ == "__main__":
    unittest.main()
