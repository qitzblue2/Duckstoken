"""run_benchmark against a fake llama-server: honesty of values, cold/warm separation, no leaked processes,
cancel/timeout handling, rejection without starting anything, reproducibility, comparison."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vgpu_kit as K  # noqa: E402

from duckstoken.vgpu import bench_client as bc  # noqa: E402
from duckstoken.vgpu import benchmark as B  # noqa: E402

SMALL = dict(prompt_tokens=64, gen_tokens=8, repeats=2)


class BenchCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.model = K.make_model(self.dir)
        self.factory = K.ServerFactory()
        p = mock.patch.object(B, "ServerManager", self.factory)
        p.start()
        self.addCleanup(p.stop)
        self.hw = K.mock_hw()
        self.probe = K.FakeProbe()
        self.patch_modules(probe=self.probe)       # the real probe spawns the executable twice; tested once below

    def settings(self, extra="--fake-load-s 0.05", **kw):
        return K.fake_settings(self.dir, extra, **kw)

    def bench(self, extra="--fake-load-s 0.05", spec=None, settings=None, **kw):
        spec = spec or B.BenchSpec(**SMALL)
        s = settings or self.settings(extra)
        kw.setdefault("hw", self.hw)
        return B.run_benchmark(s, self.model, kw.pop("label", "t"), spec=spec, **kw)

    def use_planner(self, planner):
        self.patch_modules(planner=planner)

    def patch_modules(self, **mods):
        """Replace sibling modules the benchmark looks up (a value of None means "not importable")."""
        real = B._mod
        p = mock.patch.object(B, "_mod", lambda name: mods[name] if name in mods else real(name))
        p.start()
        self.addCleanup(p.stop)

    def assertNoLeaks(self):
        self.assertEqual(self.factory.alive(), [])
        self.assertEqual(K.child_pids(), [])
        self.assertEqual(K.thread_names(), [])


class TestSuccessfulRun(BenchCase):
    def test_record_shape_and_honest_labels(self):
        r = self.bench()
        self.assertEqual(r["status"], "ok", r["error"])
        for key in ("id", "created", "label", "status", "error", "reasons", "hardware", "model", "backend", "config",
                    "config_hash", "spec", "load", "cold", "warm", "metrics", "comparison", "bottleneck", "reproducibility"):
            self.assertIn(key, r)
        self.assertTrue(r["id"].startswith("bench-"))
        self.assertEqual(r["hardware"]["fingerprint"], B.hardware_fingerprint(self.hw))
        self.assertEqual(r["model"]["name"], "tiny-dense")
        self.assertEqual(r["model"]["file_type"], 15)
        self.assertEqual(r["model"]["path"], self.model)
        self.assertEqual(r["load"]["seconds"]["status"], "measured")
        self.assertEqual(len(r["warm"]["runs"]), SMALL["repeats"])
        for run in [r["cold"], *r["warm"]["runs"]]:
            for k in ("ttft_s", "prompt_tokens", "prompt_tps", "gen_tokens", "gen_tps", "total_s"):
                self.assertIn("status", run[k], k)
            self.assertEqual(run["gen_tps"]["status"], "measured")
            self.assertEqual(run["gen_tps"]["source"], "llama-server timings")
            self.assertEqual(run["gen_tokens"]["value"], SMALL["gen_tokens"])
        # prompt-processing and generation speeds are separate numbers, never one blended figure
        self.assertNotEqual(r["cold"]["prompt_tps"]["value"], r["cold"]["gen_tps"]["value"])
        self.assertEqual(r["warm"]["gen_tps"]["n"], SMALL["repeats"])
        self.assertEqual(r["warm"]["gen_tps"]["status"], "measured")
        self.assertIn("stdev", r["warm"]["prompt_tps"])
        self.assertEqual(r["metrics"]["transfer_bandwidth_gbs"]["status"], "unavailable")
        self.assertIn("does not expose", r["metrics"]["transfer_bandwidth_gbs"]["note"])
        self.assertIn("cmdline", r["config"])
        self.assertIn("--port", r["config"]["cmdline"])
        self.assertRegex(r["reproducibility"]["prompt_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("page cache not purged", r["reproducibility"]["note"])
        json.dumps(r)                                         # the record is plain JSON (it goes to the store and the UI)
        self.assertNoLeaks()

    def test_cold_run_is_separate_and_warm_up_is_not_counted(self):
        r = self.bench("--fake-load-s 0.05 --fake-cold-factor 4")
        cold, warm = r["cold"], r["warm"]
        self.assertLess(cold["gen_tps"]["value"], warm["gen_tps"]["value"] / 3)
        self.assertLess(cold["prompt_tps"]["value"], warm["prompt_tps"]["value"] / 3)
        self.assertEqual(warm["gen_tps"]["min"], warm["gen_tps"]["max"])     # no cold-run contamination
        self.assertEqual(cold["phase"], "cold")
        self.assertTrue(all(x["phase"] == "warm" for x in warm["runs"]))
        self.assertEqual([x["index"] for x in warm["runs"]], [1, 2])

    def test_warmup_flag_controls_extra_discarded_request(self):
        counted = []
        for warmup in (True, False):
            steps = []
            self.bench(spec=B.BenchSpec(**SMALL, warmup=warmup), progress=lambda *a, s=steps: s.append(a[2]))
            counted.append(any("Warm-up" in m for m in steps))
        self.assertEqual(counted, [True, False])

    def test_prompt_is_reprocessed_every_run(self):
        # the fake behaves like llama-server: with cache_prompt true the repeated prompt would not be re-processed
        r = self.bench("--fake-load-s 0.05 --fake-cache-prompt")
        pts = [x["prompt_tokens"]["value"] for x in [r["cold"], *r["warm"]["runs"]]]
        self.assertEqual(len(set(pts)), 1)
        self.assertGreater(pts[0], 40)
        self.assertEqual({x["cached_tokens"] for x in [r["cold"], *r["warm"]["runs"]]}, {0})

    def test_prompt_cache_reuse_is_flagged_if_the_server_does_it_anyway(self):
        def fake_stream(base, ep, payload, cancel, timeout):
            n = 1 if getattr(fake_stream, "n", 0) else 0
            fake_stream.n = 1
            return {"outcome": "ok", "t_sent": 0.0, "t_first": 0.1, "t_last": 0.2, "t_end": 0.2, "chunks": 4,
                    "text": "x", "endpoint": ep, "final": {"stop_type": "limit"},
                    "timings": {"cache_n": 60 if n else 0, "prompt_n": 4 if n else 64, "prompt_per_second": 100.0,
                                "predicted_n": 4, "predicted_per_second": 20.0}}

        with mock.patch.object(bc, "stream_request", fake_stream):
            r = self.bench()
        self.assertTrue(any("cached prompt tokens" in w for w in r["warnings"]))
        self.assertEqual(r["warm"]["runs"][0]["cached_tokens"], 60)

    def test_without_server_timings_values_are_estimates_and_say_so(self):
        r = self.bench("--fake-load-s 0.05 --fake-no-timings --fake-tg-tps 400 --fake-pp-tps 4000")
        self.assertEqual(r["status"], "ok", r["error"])
        for run in [r["cold"], *r["warm"]["runs"]]:
            self.assertEqual(run["source"], "client_side")
            self.assertEqual(run["gen_tps"]["status"], "estimated")
            self.assertEqual(run["prompt_tps"]["status"], "estimated")
            self.assertIn("lower bound", run["prompt_tps"]["note"])
            self.assertEqual(run["gen_tokens"]["status"], "estimated")
            self.assertEqual(run["ttft_s"]["status"], "measured")        # a stopwatch reading is still measured
        self.assertEqual(r["warm"]["gen_tps"]["status"], "estimated")
        gen = r["warm"]["gen_tps"]["value"]
        # roughly the simulated 400 tokens/s, client side. Windows sleeps in ~15 ms steps, so the simulated 2.5 ms/token is
        # really ~15 ms there and the measured speed is far lower; the label is what this test is about.
        self.assertTrue((20 if os.name == "nt" else 150) < gen < 450, gen)

    def test_falls_back_to_chat_endpoint_on_older_servers(self):
        r = self.bench("--fake-load-s 0.05 --fake-chat-only")
        self.assertEqual(r["status"], "ok", r["error"])
        self.assertTrue(all(x["endpoint"] == "/v1/chat/completions" for x in [r["cold"], *r["warm"]["runs"]]))

    @unittest.skipIf(os.name == "nt", "tests/fake_llama_server.py is a shebang script and cannot be launched directly on Windows")
    def test_old_test_server_that_drops_the_connection_also_falls_back(self):
        old = K.ROOT / "tests" / "fake_llama_server.py"
        self.patch_modules(probe=None)
        s = self.settings(llama_server_path=str(old), extra_args="")
        r = B.run_benchmark(s, self.model, spec=B.BenchSpec(**SMALL), hw=self.hw, work_dir=self.dir)
        self.assertEqual(r["status"], "ok", r["error"])
        self.assertEqual(r["cold"]["endpoint"], "/v1/chat/completions")
        self.assertEqual(r["cold"]["gen_tps"]["source"], "llama-server timings")

    def test_backend_info_comes_from_the_engine_probe(self):
        r = self.bench()
        self.assertEqual(r["backend"]["engine"], "llama.cpp")
        self.assertEqual((r["backend"]["version"], r["backend"]["build"]), ("9999", 9999))
        self.assertEqual(r["backend"]["probe"], "ok")
        self.assertEqual(Path(r["backend"]["path"]).stem, "llama-server")   # llama-server.cmd on Windows

    def test_real_probe_on_the_fake_executable(self):
        from duckstoken.vgpu import probe as real_probe
        self.patch_modules(probe=real_probe)         # replace the canned probe with the real module
        r = self.bench()
        self.assertEqual(r["status"], "ok", r["error"])
        self.assertEqual((r["backend"]["version"], r["backend"]["build"], r["backend"]["commit"]), ("0.6.0-dev", 9999, "abc1234"))

    def test_probe_failure_does_not_stop_the_benchmark(self):
        class Boom:
            def probe_engine(self, p, **k):
                raise OSError("cannot run")

        self.patch_modules(probe=Boom())
        r = self.bench()
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["backend"]["probe"], "error")
        self.assertIsNone(r["backend"]["version"])
        self.patch_modules(probe=None)
        self.assertEqual(self.bench()["backend"]["probe"], "unavailable")

    def test_log_verbosity_added_for_buffer_lines_but_not_part_of_config_hash(self):
        a = self.bench()
        self.assertIn("-lv 4", a["config"]["cmdline"])
        self.assertNotIn("-lv", a["config"]["extra_args"])
        b = self.bench(settings=self.settings("--fake-load-s 0.05", verbose=True))
        self.assertNotIn("-lv 4", b["config"]["cmdline"])      # the user asked for verbose already
        self.assertEqual(a["config_hash"], b["config_hash"])

    def test_comparison_reads_buffer_sizes_from_the_load_log(self):
        r = self.bench()
        cmp_ = r["comparison"]
        self.assertIn("estimated", cmp_)
        self.assertEqual(cmp_["measured"]["weights_ram_gb"]["status"], "measured")
        self.assertEqual(cmp_["measured"]["weights_ram_gb"]["source"], "llama.cpp load log")
        self.assertIn("CPU_Mapped", cmp_["measured"]["raw_mib"]["model_buffers"])
        self.assertIn("kv_cache_gb", cmp_["delta_gb"])

    def test_gpu_buffers_are_reported_in_the_vram_tier(self):
        big = K.make_model(self.dir, "mid.gguf", pad=8 * 2**20)
        r = B.run_benchmark(self.settings("--fake-load-s 0.05 --fake-gpu-mib 6", n_gpu_layers=2), big,
                            spec=B.BenchSpec(**SMALL), hw=K.mock_hw(vram_mb=4096))
        self.assertEqual(r["status"], "ok", r["error"])
        self.assertGreater(r["comparison"]["measured"]["weights_vram_gb"]["value"], 0)

    def test_an_http_proxy_in_the_environment_does_not_intercept_the_local_server(self):
        with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9",
                                           "ALL_PROXY": "http://127.0.0.1:9"}):
            r = self.bench()
        self.assertEqual(r["status"], "ok", r["error"])

    def test_user_verbosity_flags_are_respected(self):
        for extra in ("--fake-load-s 0.1 -lv 5", "--fake-load-s 0.1 -v", "--fake-load-s 0.1 --verbosity 3"):
            r = self.bench(extra)
            self.assertEqual(r["config"]["cmdline"].count("-lv 4"), 0, extra)

    def test_context_is_raised_when_the_prompt_would_not_fit(self):
        r = self.bench(settings=self.settings(ctx_size=128), spec=B.BenchSpec(prompt_tokens=200, gen_tokens=8, repeats=1))
        self.assertEqual(r["status"], "ok", r["error"])
        self.assertGreaterEqual(r["config"]["ctx_size"], 200 + 8)
        self.assertTrue(any("context size set to" in w for w in r["warnings"]))

    def test_progress_steps_are_monotonic_and_reach_total(self):
        steps = []
        r = self.bench(progress=lambda *a: steps.append(a))
        self.assertEqual(r["status"], "ok")
        nums = [s[0] for s in steps]
        self.assertEqual(nums, sorted(nums))
        self.assertEqual(steps[-1][0], steps[-1][1])
        self.assertTrue(all(isinstance(s[2], str) and s[2] for s in steps))

    def test_perflog_gets_start_and_done(self):
        events = []

        class Log:
            def log(self, event, **f):
                events.append((event, f))

        self.bench(perflog=Log())
        self.assertEqual([e[0] for e in events], ["benchmark_start", "benchmark_done"])
        self.assertEqual(events[1][1]["status"], "ok")

    def test_broken_progress_and_perflog_do_not_break_the_run(self):
        class Bad:
            def log(self, *a, **k):
                raise OSError("disk full")

        def bad_progress(*a):
            raise RuntimeError("ui gone")

        self.assertEqual(self.bench(perflog=Bad(), progress=bad_progress)["status"], "ok")

    def test_temp_dir_is_removed(self):
        made = []
        real_mkdtemp = tempfile.mkdtemp

        def mk(prefix=""):
            d = real_mkdtemp(prefix=prefix, dir=self.dir)
            made.append(d)
            return d

        with mock.patch.object(B.tempfile, "mkdtemp", mk):
            self.bench()
        self.assertEqual(len(made), 1)
        self.assertFalse(Path(made[0]).exists())

    def test_given_work_dir_keeps_other_files_but_not_the_log(self):
        wd = self.dir / "wd"
        wd.mkdir()
        (wd / "keep.txt").write_text("mine")
        r = self.bench(work_dir=wd)
        self.assertEqual(sorted(p.name for p in wd.iterdir()), ["keep.txt"])
        self.assertTrue(r["log_tail"])


class TestRejectionAndFailures(BenchCase):
    def test_planner_rejection_starts_no_process(self):
        self.use_planner(K.FakePlanner(reject=lambda cfg: ["Not enough RAM: needs 80 GB, 12 GB available."]))
        r = self.bench()
        self.assertEqual(r["status"], "rejected")
        self.assertEqual(r["reasons"], ["Not enough RAM: needs 80 GB, 12 GB available."])
        self.assertIn("Not enough RAM", r["error"])
        self.assertEqual(self.factory.servers, [])            # not even a ServerManager was created
        self.assertEqual(K.child_pids(), [])
        self.assertEqual(r["warm"]["runs"], [])
        self.assertEqual(r["cold"]["gen_tps"]["status"], "unavailable")

    def test_unreadable_or_missing_model_is_rejected_without_a_process(self):
        r = B.run_benchmark(self.settings(), self.dir / "missing.gguf", spec=B.BenchSpec(**SMALL), hw=self.hw)
        self.assertEqual(r["status"], "rejected")
        self.assertIn("cannot read the model file", r["error"])
        (self.dir / "junk.gguf").write_bytes(b"not a gguf at all")
        r = B.run_benchmark(self.settings(), self.dir / "junk.gguf", spec=B.BenchSpec(**SMALL), hw=self.hw)
        self.assertEqual(r["status"], "rejected")
        self.assertEqual(self.factory.servers, [])

    def test_engine_not_installed(self):
        s = self.settings(llama_server_path="")
        with mock.patch("duckstoken.vgpu.benchmark.find_llama_server", return_value=None):
            r = B.run_benchmark(s, self.model, spec=B.BenchSpec(**SMALL), hw=self.hw)
        self.assertEqual(r["status"], "failed")
        self.assertIn("not installed", r["error"])
        self.assertEqual(self.factory.servers, [])

    def test_garbage_arguments_give_a_failed_record_not_an_exception(self):
        r = B.run_benchmark(self.settings(), self.model, hw=self.hw, spec=B.BenchSpec(prompt_tokens="lots"))
        self.assertEqual(r["status"], "failed")
        self.assertIn("invalid benchmark arguments", r["error"])
        self.assertEqual(self.factory.servers, [])

    def test_hardware_detection_failure_is_recorded(self):
        with mock.patch.object(B.hardware, "get_hardware", side_effect=OSError("wmi broke")):
            r = B.run_benchmark(self.settings(), self.model, spec=B.BenchSpec(**SMALL))
        self.assertEqual(r["status"], "failed")
        self.assertIn("wmi broke", r["error"])

    def test_planner_missing_fails_closed(self):
        self.use_planner(None)
        r = self.bench()
        self.assertEqual(r["status"], "failed")
        self.assertIn("planner is unavailable", r["error"])
        self.assertEqual(self.factory.servers, [])

    def test_planner_crash_is_recorded_not_raised(self):
        class Boom(K.FakePlanner):
            def plan(self, *a, **k):
                raise ZeroDivisionError("planner bug")

        self.use_planner(Boom())
        r = self.bench()
        self.assertEqual(r["status"], "failed")
        self.assertIn("planner bug", r["error"])
        self.assertEqual(self.factory.servers, [])

    def test_load_failure_reports_exit_code_and_log_tail(self):
        r = self.bench("--fake-load-s 0.05 --fake-exit-code 7")
        self.assertEqual(r["status"], "failed")
        self.assertIn("code 7", r["error"])
        self.assertTrue(any("failed to load model" in line for line in r["log_tail"]), r["log_tail"])
        self.assertEqual(r["cold"]["gen_tps"]["status"], "unavailable")
        self.assertNoLeaks()

    def test_load_timeout_kills_the_server(self):
        t0 = time.time()
        r = self.bench("--fake-hang-load", spec=B.BenchSpec(**SMALL, load_timeout_s=1.0))
        self.assertEqual(r["status"], "timeout")
        self.assertIn("did not finish loading", r["error"])
        self.assertLess(time.time() - t0, 10)
        self.assertEqual(len(self.factory.procs), 1)
        self.assertNoLeaks()

    def test_request_timeout(self):
        r = self.bench("--fake-load-s 0.05 --fake-tg-tps 2", spec=B.BenchSpec(prompt_tokens=64, gen_tokens=50, repeats=1,
                                                                             request_timeout_s=1.0))
        self.assertEqual(r["status"], "timeout")
        self.assertIn("longer than", r["error"])
        self.assertNoLeaks()

    def test_server_crash_mid_run_keeps_what_was_measured(self):
        r = self.bench("--fake-load-s 0.05 --fake-crash-after 1")
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["cold"]["gen_tps"]["status"], "measured")      # the cold run finished before the crash
        self.assertEqual(r["warm"]["runs"], [])
        self.assertTrue(r["error"])
        self.assertNoLeaks()

    def test_http_error_from_the_server(self):
        r = self.bench("--fake-load-s 0.05 --fake-status 500")
        self.assertEqual(r["status"], "failed")
        self.assertIn("simulated failure", r["error"])
        self.assertNoLeaks()

    def test_server_that_ignores_sigterm_is_killed(self):
        if os.name == "nt":
            self.skipTest("SIGTERM semantics differ on Windows")
        t0 = time.time()
        r = self.bench("--fake-load-s 0.05 --fake-ignore-term", stop_grace_s=0.5)
        self.assertEqual(r["status"], "ok")
        self.assertLess(time.time() - t0, 8)
        self.assertNoLeaks()


class TestCancel(BenchCase):
    def test_cancel_during_load_is_prompt(self):
        cancel = threading.Event()
        threading.Timer(0.5, cancel.set).start()
        t0 = time.time()
        r = self.bench("--fake-hang-load", spec=B.BenchSpec(**SMALL, load_timeout_s=60), cancel=cancel)
        self.assertEqual(r["status"], "cancelled")
        self.assertLess(time.time() - t0, 6)
        self.assertNoLeaks()

    def test_cancel_in_the_middle_of_generation(self):
        cancel = threading.Event()
        threading.Timer(1.0, cancel.set).start()
        t0 = time.time()
        r = self.bench("--fake-load-s 0.05 --fake-tg-tps 4", spec=B.BenchSpec(prompt_tokens=64, gen_tokens=200, repeats=3),
                       cancel=cancel)
        self.assertEqual(r["status"], "cancelled")
        self.assertLess(time.time() - t0, 8)          # nowhere near the 50 s the generation would take
        self.assertEqual(r["warm"]["runs"], [])
        self.assertNoLeaks()

    def test_cancel_between_warm_runs_keeps_partial_results(self):
        cancel = threading.Event()
        seen = []

        def progress(step, total, message):
            seen.append(message)
            if message.startswith("Warm run 2"):
                cancel.set()

        r = self.bench(spec=B.BenchSpec(prompt_tokens=64, gen_tokens=8, repeats=4), cancel=cancel, progress=progress)
        self.assertEqual(r["status"], "cancelled")
        self.assertEqual(len(r["warm"]["runs"]), 1)
        self.assertEqual(r["warm"]["gen_tps"]["n"], 1)
        self.assertNoLeaks()

    def test_already_cancelled_never_starts_a_server(self):
        cancel = threading.Event()
        cancel.set()
        r = self.bench(cancel=cancel)
        self.assertEqual(r["status"], "cancelled")
        self.assertEqual(self.factory.procs, [])

    def test_no_leaks_across_many_runs_of_every_kind(self):
        before = K.child_pids()
        for extra, spec in (("--fake-load-s 0.1", None), ("--fake-exit-code 3", None), ("--fake-status 500", None),
                            ("--fake-load-s 0.1 --fake-crash-after 0", None),
                            ("--fake-hang-load", B.BenchSpec(**SMALL, load_timeout_s=1.0))):
            self.bench(extra, spec=spec)
        self.assertEqual(self.factory.alive(), [])
        self.assertEqual(len(self.factory.procs), 5)
        self.assertEqual(K.child_pids(), before)
        self.assertEqual(K.thread_names(), [])


class TestMetricsFromSampler(BenchCase):
    def test_sampler_values_become_measured_metrics(self):
        n = {"i": 0}

        def sampler(pid):
            n["i"] += 1
            i = n["i"]
            return {"process": {"rss_gb": {"value": 1.0 + i / 10, "status": "measured"},
                                "peak_rss_gb": {"value": 1.5 + i / 10, "status": "measured"},
                                "page_faults": {"value": 1000 * i, "status": "measured"},
                                "io_read_gb": {"value": 0.5 * i, "status": "measured"},
                                "cpu_seconds": {"value": 2.0 * i, "status": "measured"}},
                    "system": {"cpu": {"util_pct": {"value": 50.0 + i, "status": "measured"}},
                               "gpus": [{"util_pct": {"value": 10.0 * i, "status": "measured"},
                                         "vram_used_mb": {"value": 500 + i, "status": "measured"}}]}}

        r = self.bench(sampler_fn=sampler, spec=B.BenchSpec(**SMALL, sample_interval_s=0.05))
        m = r["metrics"]
        self.assertEqual(m["peak_rss_gb"]["status"], "measured")
        self.assertGreaterEqual(m["peak_rss_gb"]["value"], 1.6)
        self.assertGreater(m["page_faults"]["value"], 0)
        self.assertGreater(m["io_read_gb"]["value"], 0)
        self.assertGreater(m["cpu_util_pct"]["value"], 50)
        self.assertGreater(m["gpu_util_pct"]["value"], 0)
        self.assertGreater(m["peak_vram_mb"]["value"], 500)
        self.assertIn("device-wide", m["peak_vram_mb"]["note"])
        self.assertGreaterEqual(m["samples"], 2)

    def test_failing_sampler_degrades_to_unavailable(self):
        def bad(pid):
            raise RuntimeError("telemetry broke")

        r = self.bench(sampler_fn=bad)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["metrics"]["peak_rss_gb"]["status"], "unavailable")
        self.assertEqual(r["metrics"]["peak_rss_gb"]["value"], None)

    def test_aggregate_samples_never_invents_numbers(self):
        m = B.aggregate_samples([])
        for k in ("peak_rss_gb", "page_faults", "io_read_gb", "cpu_util_pct", "gpu_util_pct", "peak_vram_mb",
                  "transfer_bandwidth_gbs", "host_bandwidth_gbs"):
            self.assertEqual(m[k]["status"], "unavailable", k)
            self.assertIsNone(m[k]["value"], k)

    def test_aggregate_reports_unavailable_leaf_reasons_and_plain_numbers(self):
        samples = [{"t": 0.0, "process": {"rss_gb": 2.0, "page_faults": {"value": None, "status": "unavailable", "note": "no /proc"}},
                    "gpus": [], "sources": {"gpu": {"note": "nvidia-smi not found"}}},
                   {"t": 1.0, "process": {"rss_gb": 3.5, "page_faults": {"value": None, "status": "unavailable", "note": "no /proc"}},
                    "gpus": []}]
        m = B.aggregate_samples(samples, host_bw={"value": 12.0, "status": "measured", "source": "x", "note": ""})
        self.assertEqual(m["peak_rss_gb"]["value"], 3.5)
        self.assertEqual(m["page_faults"]["status"], "unavailable")
        self.assertEqual(m["page_faults"]["note"], "no /proc")
        self.assertIn("nvidia-smi not found", m["gpu_util_pct"]["note"])
        self.assertEqual(m["host_bandwidth_gbs"]["value"], 12.0)

    def test_aggregate_deltas_and_peaks(self):
        v = lambda x: {"value": x, "status": "measured"}  # noqa: E731
        samples = [{"t": 0.0, "process": {"rss_gb": v(1.0), "peak_rss_gb": v(1.2), "page_faults": v(100), "major_page_faults": v(5),
                                          "io_read_gb": v(0.1), "cpu_seconds": v(1.0)}, "cpu": {"util_pct": v(40.0)},
                    "gpus": [{"util_pct": v(10.0), "vram_used_mb": v(900)}]},
                   {"t": 2.0, "process": {"rss_gb": v(2.0), "peak_rss_gb": v(2.5), "page_faults": v(1100), "major_page_faults": v(25),
                                          "io_read_gb": v(0.6), "cpu_seconds": v(5.0)}, "cpu": {"util_pct": v(80.0)},
                    "gpus": [{"util_pct": v(30.0), "vram_used_mb": v(1500)}]}]
        m = B.aggregate_samples(samples)
        self.assertEqual(m["peak_rss_gb"]["value"], 2.5)
        self.assertEqual(m["page_faults"]["value"], 1000)
        self.assertEqual(m["major_page_faults"]["value"], 20)
        self.assertAlmostEqual(m["io_read_gb"]["value"], 0.5)
        self.assertEqual(m["process_cpu_cores"]["value"], 2.0)       # 4 cpu-seconds over 2 s
        self.assertEqual(m["cpu_util_pct"]["value"], 60.0)
        self.assertEqual(m["gpu_util_pct"]["value"], 20.0)
        self.assertEqual(m["peak_vram_mb"]["value"], 1500)
        self.assertEqual({m[k]["status"] for k in ("peak_rss_gb", "page_faults", "io_read_gb", "cpu_util_pct", "peak_vram_mb")}, {"measured"})

    def test_real_telemetry_is_used_by_default_and_never_zero_filled(self):
        r = self.bench(spec=B.BenchSpec(**SMALL, sample_interval_s=0.05))
        rss = r["metrics"]["peak_rss_gb"]
        if os.path.isdir("/proc"):
            self.assertEqual(rss["status"], "measured")
            self.assertGreater(rss["value"], 0.001)       # a real python process, tens of MB
        self.assertGreaterEqual(r["metrics"]["samples"], 2)
        self.assertEqual(r["metrics"]["transfer_bandwidth_gbs"]["status"], "unavailable")
        self.assertIn(r["metrics"]["gpu_util_pct"]["status"], ("measured", "unavailable"))


class TestReproducibility(BenchCase):
    def shape(self, obj):
        if isinstance(obj, dict):
            return {k: self.shape(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.shape(obj[0])] if obj else []
        return type(obj).__name__ if not isinstance(obj, (int, float)) else "num"

    def test_same_spec_twice_gives_same_hashes_and_shape(self):
        a, b = self.bench(), self.bench()
        self.assertEqual(a["status"], "ok")
        self.assertEqual(a["reproducibility"]["prompt_sha256"], b["reproducibility"]["prompt_sha256"])
        self.assertEqual(a["reproducibility"]["spec_hash"], b["reproducibility"]["spec_hash"])
        self.assertEqual(a["config_hash"], b["config_hash"])
        self.assertEqual(a["reproducibility"]["config_hash"], a["config_hash"])
        self.assertNotEqual(a["id"], b["id"])
        self.assertEqual(self.shape(a["warm"]["gen_tps"]).keys(), self.shape(b["warm"]["gen_tps"]).keys())
        self.assertEqual(set(a), set(b))
        self.assertEqual(set(a["cold"]), set(b["cold"]))
        self.assertEqual(set(a["metrics"]), set(b["metrics"]))
        self.assertEqual(a["cold"]["prompt_tokens"]["value"], b["cold"]["prompt_tokens"]["value"])

    def test_changed_config_changes_config_hash_only(self):
        a = self.bench()
        b = self.bench(settings=self.settings("--fake-load-s 0.05", ubatch_size=128))
        self.assertNotEqual(a["config_hash"], b["config_hash"])
        self.assertEqual(a["reproducibility"]["prompt_sha256"], b["reproducibility"]["prompt_sha256"])

    def test_port_and_model_path_are_not_part_of_the_config_hash(self):
        a = self.bench(settings=self.settings("--fake-load-s 0.05", port=1111))
        b = self.bench(settings=self.settings("--fake-load-s 0.05", port=2222, api_key="secret"))
        self.assertEqual(a["config_hash"], b["config_hash"])
        self.assertNotIn("secret", str(b["config"]))

    def test_spec_hash_ignores_timeouts_but_not_the_workload(self):
        base = B.BenchSpec()
        self.assertEqual(base.spec_hash(), B.BenchSpec(load_timeout_s=5, request_timeout_s=5, sample_interval_s=3).spec_hash())
        for change in ({"prompt_tokens": 256}, {"gen_tokens": 32}, {"repeats": 5}, {"seed": 1}, {"temperature": 0.5},
                       {"warmup": False}, {"ctx": 8192}):
            self.assertNotEqual(base.spec_hash(), B.BenchSpec(**change).spec_hash(), change)

    def test_spec_normalization_clamps_nonsense(self):
        n = B.BenchSpec(prompt_tokens=-5, gen_tokens=0, repeats=0, load_timeout_s=-1, sample_interval_s=0).normalized()
        self.assertGreaterEqual(n.prompt_tokens, 8)
        self.assertGreaterEqual(n.gen_tokens, 2)
        self.assertEqual(n.repeats, 1)
        self.assertGreater(n.sample_interval_s, 0)


class TestRealPlannerIntegration(BenchCase):
    def test_runs_end_to_end_with_the_real_planner_and_records_its_estimate(self):
        r = self.bench()
        self.assertEqual(r["status"], "ok", r["error"])
        est = r["comparison"]["estimated"]
        self.assertEqual(est["verdict"], "ok")
        self.assertEqual(est["weights_gb"]["status"], "estimated")
        self.assertEqual(est["tokens_per_s"]["status"], "estimated")
        self.assertIn(r["bottleneck"]["kind"], ("memory_bandwidth", "compute", "storage_paging", "vram_capacity",
                                                "pcie_transfer", "thermal", "cpu_threads", "unknown"))

    def test_real_planner_rejects_when_the_disk_is_too_small_for_the_model(self):
        hw = K.mock_hw(disk_free_gb=0.0)
        big = K.make_model(self.dir, "big.gguf", pad=3 * 2**20)
        with mock.patch.object(B.gguf, "summarize", side_effect=lambda p, _s=B.gguf.summarize: {**_s(p), "size_bytes": 50 * 2**30, "size_gb": 50.0}):
            r = B.run_benchmark(self.settings(), big, spec=B.BenchSpec(**SMALL), hw=hw)
        self.assertEqual(r["status"], "rejected", r)
        self.assertTrue(r["reasons"])
        self.assertEqual(self.factory.servers, [])


class TestBenchClient(unittest.TestCase):
    def test_prompt_is_deterministic_and_scales(self):
        t1, h1 = bc.build_prompt(512)
        t2, h2 = bc.build_prompt(512)
        self.assertEqual((t1, h1), (t2, h2))
        self.assertNotEqual(h1, bc.build_prompt(256)[1])
        self.assertGreater(len(t1.split()), 400)
        self.assertEqual(len(h1), 64)
        self.assertTrue(bc.build_prompt(0)[0])

    def test_summarize_runs(self):
        s = bc.summarize_runs([1.0, 2.0, 3.0])
        self.assertEqual((s["mean"], s["min"], s["max"], s["n"]), (2.0, 1.0, 3.0, 3))
        self.assertAlmostEqual(s["stdev"], 1.0)
        self.assertEqual(bc.summarize_runs([5.0])["stdev"], 0.0)
        self.assertEqual(bc.summarize_runs([]), {"mean": None, "stdev": None, "min": None, "max": None, "n": 0})
        self.assertEqual(bc.summarize_runs([1.0, float("nan"), None, True, "x"])["n"], 1)

    def test_stat_val_status_rules(self):
        m = lambda v: {"value": v, "status": "measured", "source": "s", "note": ""}  # noqa: E731
        e = lambda v: {"value": v, "status": "estimated", "source": "s", "note": ""}  # noqa: E731
        u = {"value": None, "status": "unavailable", "source": "s", "note": "why not"}
        self.assertEqual(bc.stat_val([{"k": m(1.0)}, {"k": m(3.0)}], "k")["status"], "measured")
        mixed = bc.stat_val([{"k": m(1.0)}, {"k": e(3.0)}], "k")
        self.assertEqual((mixed["status"], mixed["value"]), ("estimated", 2.0))
        part = bc.stat_val([{"k": m(1.0)}, {"k": u}], "k")
        self.assertEqual((part["status"], part["n"]), ("measured", 1))
        none = bc.stat_val([{"k": u}], "k")
        self.assertEqual((none["status"], none["value"], none["note"]), ("unavailable", None, "why not"))
        self.assertEqual(bc.stat_val([], "k")["status"], "unavailable")

    def test_run_metrics_with_server_timings(self):
        raw = {"t_sent": 0.0, "t_first": 0.5, "t_last": 1.5, "t_end": 1.6, "chunks": 11, "endpoint": "/completion",
               "final": {"stop_type": "limit"},
               "timings": {"cache_n": 0, "prompt_n": 100, "prompt_per_second": 250.0, "predicted_n": 11,
                           "predicted_per_second": 12.5}}
        r = bc.run_metrics(raw, 1, "warm", 100)
        self.assertEqual((r["prompt_tps"]["value"], r["gen_tps"]["value"]), (250.0, 12.5))
        self.assertEqual({r["prompt_tps"]["status"], r["gen_tps"]["status"]}, {"measured"})
        self.assertEqual(r["ttft_s"]["value"], 0.5)
        self.assertEqual(r["total_s"]["value"], 1.6)

    def test_run_metrics_client_side_fallback_is_labelled_estimated(self):
        raw = {"t_sent": 0.0, "t_first": 0.5, "t_last": 1.5, "t_end": 1.6, "chunks": 11, "endpoint": "/completion",
               "final": {"tokens_evaluated": 100}, "timings": None}
        r = bc.run_metrics(raw, 0, "cold", 100)
        self.assertEqual(r["gen_tps"]["status"], "estimated")
        self.assertAlmostEqual(r["gen_tps"]["value"], 10.0)         # (11 - 1) tokens over 1.0 s
        self.assertAlmostEqual(r["prompt_tps"]["value"], 200.0)     # 100 tokens / 0.5 s: a lower bound
        self.assertEqual(r["prompt_tokens"]["status"], "measured")  # the server did report tokens_evaluated

    def test_run_metrics_with_nothing_streamed_is_unavailable_not_zero(self):
        raw = {"t_sent": 0.0, "t_first": None, "t_last": None, "t_end": None, "chunks": 0, "endpoint": "/completion"}
        r = bc.run_metrics(raw, 0, "cold", 100)
        for k in ("ttft_s", "total_s", "gen_tps", "prompt_tps", "gen_tokens"):
            self.assertEqual(r[k]["status"], "unavailable", k)
            self.assertIsNone(r[k]["value"], k)

    def test_single_token_has_no_generation_speed(self):
        raw = {"t_sent": 0.0, "t_first": 0.5, "t_last": 0.5, "t_end": 0.6, "chunks": 1, "endpoint": "/completion"}
        self.assertEqual(bc.run_metrics(raw, 0, "cold", 10)["gen_tps"]["status"], "unavailable")

    def test_free_port_is_usable(self):
        import socket
        p = bc.free_port()
        with socket.socket() as s:
            s.bind(("127.0.0.1", p))

    def test_payload_never_allows_prompt_cache_reuse(self):
        for ep in (bc.COMPLETION, bc.CHAT):
            pl = bc.completion_payload(ep, "hello", 16, 7, 0.0)
            self.assertIs(pl["cache_prompt"], False)
            self.assertTrue(pl["stream"] and pl["ignore_eos"])
            self.assertEqual((pl["seed"], pl["temperature"]), (7, 0.0))
        self.assertEqual(bc.completion_payload(bc.COMPLETION, "x", 5, 1, 0)["n_predict"], 5)
        self.assertEqual(bc.completion_payload(bc.CHAT, "x", 5, 1, 0)["max_tokens"], 5)

    def test_stream_request_to_a_closed_port_reports_error_not_exception(self):
        port = bc.free_port()
        r = bc.stream_request(f"http://127.0.0.1:{port}", bc.COMPLETION, {}, None, 5)
        self.assertEqual(r["outcome"], "error")
        self.assertTrue(r["error"])
        self.assertEqual(K.thread_names(), [])


class TestCompare(unittest.TestCase):
    def rec(self, gen, name="m", size=100, spec="s1", hw="h1", status="ok", st="measured", rss=2.0):
        v = lambda x, s=st: {"value": x, "status": s, "source": "", "note": ""}  # noqa: E731
        return {"status": status, "error": "", "config_hash": "c", "model": {"name": name, "size_bytes": size, "file_type": 15},
                "hardware": {"fingerprint": hw}, "reproducibility": {"spec_hash": spec},
                "warm": {"gen_tps": v(gen), "prompt_tps": v(gen * 10), "ttft_s": v(1.0)},
                "load": {"seconds": v(5.0)}, "metrics": {"peak_rss_gb": v(rss)}}

    def test_ratio_and_winner(self):
        c = B.compare_benchmarks(self.rec(10.0), self.rec(15.0))
        self.assertTrue(c["comparable"])
        self.assertEqual(c["ratios"]["gen_tps"]["ratio"], 1.5)
        self.assertEqual(c["winner"], "b")
        self.assertEqual(B.compare_benchmarks(self.rec(15.0), self.rec(10.0))["winner"], "a")
        self.assertEqual(B.compare_benchmarks(self.rec(10.0), self.rec(10.2))["winner"], "tie")

    def test_warns_when_not_like_for_like(self):
        c = B.compare_benchmarks(self.rec(10.0), self.rec(12.0, name="other", spec="s2", hw="h2"))
        self.assertFalse(c["comparable"])
        text = " ".join(c["warnings"])
        for word in ("different models", "different benchmark specs", "different hardware"):
            self.assertIn(word, text)

    def test_failed_or_missing_numbers_give_no_ratio_and_no_winner(self):
        bad = self.rec(1.0, status="failed")
        bad["warm"]["gen_tps"] = {"value": None, "status": "unavailable", "source": "", "note": ""}
        c = B.compare_benchmarks(self.rec(10.0), bad)
        self.assertIsNone(c["ratios"]["gen_tps"]["ratio"])
        self.assertIsNone(c["winner"])
        self.assertFalse(c["comparable"])
        self.assertTrue(any("did not finish" in w for w in c["warnings"]))

    def test_mixed_measured_and_estimated_is_flagged(self):
        c = B.compare_benchmarks(self.rec(10.0), self.rec(12.0, st="estimated"))
        self.assertTrue(any("measured and the other estimated" in w for w in c["warnings"]))

    def test_empty_records_do_not_crash(self):
        c = B.compare_benchmarks({}, {})
        self.assertIsNone(c["winner"])


class TestServerOwnership(BenchCase):
    def test_an_injected_server_is_used_and_stopped(self):
        srv = K.RecordingServer(log_file=self.dir / "mine.log")
        r = B.run_benchmark(self.settings(), self.model, "inj", hw=self.hw, spec=B.BenchSpec(**SMALL), server=srv)
        self.assertEqual(r["status"], "ok", r["error"])
        self.assertEqual(len(srv.procs), 1)
        self.assertIsNotNone(srv.procs[0].poll())
        self.assertEqual(srv.status()["state"], "stopped")
        self.assertEqual(self.factory.servers, [])                # the factory was not used: the given server was

    def test_a_server_that_is_already_running_is_never_stopped_by_a_failed_benchmark(self):
        chat = K.RecordingServer(log_file=self.dir / "chat.log")
        s = self.settings()
        chat_port = B.bc.free_port()
        chat.start({**s, "port": chat_port, "model_path": self.model, "host": "127.0.0.1"})
        try:
            r = B.run_benchmark(s, self.model, "clash", hw=self.hw, spec=B.BenchSpec(**SMALL), server=chat)
            self.assertEqual(r["status"], "failed")
            self.assertIn("already running", r["error"])
            self.assertIsNone(chat.procs[0].poll(), "the benchmark killed a server it did not start")
            self.assertTrue(chat.status()["running"])
        finally:
            chat.stop()
        self.assertIsNotNone(chat.procs[0].poll())


class TestJobRunnerIntegration(BenchCase):
    def test_cancel_through_the_job_runner(self):
        from duckstoken.vgpu.runner import JobRunner
        runner = JobRunner()
        self.addCleanup(lambda: (runner.cancel(), runner.wait(10)))
        s = self.settings("--fake-load-s 0.1 --fake-tg-tps 5")
        spec = B.BenchSpec(prompt_tokens=64, gen_tokens=200, repeats=3)

        def job(cancel, progress):
            return B.run_benchmark(s, self.model, "job", hw=self.hw, spec=spec, cancel=cancel, progress=progress)

        runner.start("benchmark", job)
        deadline = time.time() + 10
        while time.time() < deadline and runner.status()["progress"]["step"] < 5:
            time.sleep(0.05)                                      # wait until the cold run is under way
        self.assertTrue(runner.cancel())
        self.assertTrue(runner.wait(15))
        st = runner.status()
        self.assertEqual(st["state"], "cancelled")
        self.assertEqual(st["result"]["status"], "cancelled")
        self.assertTrue(any("Cold run" in line for line in st["log"]))
        self.assertNoLeaks()

    def test_second_benchmark_cannot_overlap(self):
        from duckstoken.vgpu.runner import Busy, JobRunner
        runner = JobRunner()
        self.addCleanup(lambda: (runner.cancel(), runner.wait(10)))
        s = self.settings("--fake-load-s 0.1 --fake-tg-tps 5")
        runner.start("benchmark", lambda cancel, progress: B.run_benchmark(
            s, self.model, hw=self.hw, spec=B.BenchSpec(prompt_tokens=64, gen_tokens=100, repeats=2), cancel=cancel, progress=progress))
        with self.assertRaises(Busy):
            runner.start("benchmark", lambda cancel, progress: None)
        runner.cancel()
        runner.wait(15)
        self.assertNoLeaks()


class TestMetricsPlumbing(BenchCase):
    def test_host_bandwidth_is_measured_once_and_cached(self):
        from duckstoken.vgpu import bench_metrics as M
        calls = []
        fake = type("T", (), {"measure_host_bandwidth": staticmethod(lambda: calls.append(1) or {"value": 7.0, "status": "measured"})})
        with mock.patch.dict(M._HOST_BW, {"t": 0.0, "value": None}), mock.patch.object(M, "_mod", lambda n: fake):
            a, b = M.host_bandwidth(), M.host_bandwidth()
        self.assertEqual((a["value"], b["value"], len(calls)), (7.0, 7.0, 1))

    def test_host_bandwidth_failure_is_unavailable_not_zero(self):
        from duckstoken.vgpu import bench_metrics as M

        def boom():
            raise OSError("no memory")

        fake = type("T", (), {"measure_host_bandwidth": staticmethod(boom)})
        with mock.patch.dict(M._HOST_BW, {"t": 0.0, "value": None}), mock.patch.object(M, "_mod", lambda n: fake):
            self.assertIsNone(M.host_bandwidth())
        r = B.aggregate_samples([], None)
        self.assertEqual(r["host_bandwidth_gbs"]["status"], "unavailable")

    def test_no_telemetry_module_means_unavailable_metrics_but_a_working_benchmark(self):
        from duckstoken.vgpu import bench_metrics as M
        with mock.patch.object(M, "_mod", lambda n: None):
            r = self.bench()
            self.assertIsNone(M.make_sampler(1, 0.1))
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["metrics"]["peak_rss_gb"]["status"], "unavailable")
        self.assertEqual(r["metrics"]["samples"], 0)

    def test_sampler_fn_that_returns_garbage_is_tolerated(self):
        r = self.bench(sampler_fn=lambda pid: None)
        self.assertEqual(r["status"], "ok")
        r = self.bench(sampler_fn=lambda pid: {"process": "not a dict", "system": 5})
        self.assertEqual(r["status"], "ok")


REAL_LLAMA, REAL_MODEL = os.environ.get("DUCK_REAL_LLAMA"), os.environ.get("DUCK_REAL_MODEL")


@unittest.skipUnless(REAL_LLAMA and REAL_MODEL and Path(REAL_LLAMA).is_file() and Path(REAL_MODEL).is_file(),
                     "opt-in: set DUCK_REAL_LLAMA=/path/to/llama-server and DUCK_REAL_MODEL=/path/to/tiny.gguf")
class TestRealEngine(unittest.TestCase):
    """Runs the real llama-server on a tiny model.  Not part of the hermetic suite (skipped unless both variables are set)."""

    def setUp(self):
        from duckstoken import hardware, settings as settings_mod
        self.hw = hardware.get_hardware()
        self.settings = dict(settings_mod.DEFAULTS)
        self.settings.update(llama_server_path=REAL_LLAMA, ctx_size=2048, n_gpu_layers=0, kv_offload=False)
        self.before = K.child_pids()

    def test_benchmark_numbers_are_real_separate_and_labelled(self):
        spec = B.BenchSpec(prompt_tokens=128, gen_tokens=16, repeats=2)
        r = B.run_benchmark(self.settings, REAL_MODEL, "real", hw=self.hw, spec=spec)
        self.assertEqual(r["status"], "ok", r["error"] or r["log_tail"][-5:])
        runs = [r["cold"], *r["warm"]["runs"]]
        for run in runs:
            self.assertEqual(run["source"], "server_timings")
            self.assertEqual(run["cached_tokens"], 0)
            self.assertEqual(run["gen_tokens"]["value"], 16)
            self.assertTrue(0.7 * 128 <= run["prompt_tokens"]["value"] <= 1.6 * 128, run["prompt_tokens"])
            self.assertGreater(run["gen_tps"]["value"], 0)
            self.assertGreater(run["prompt_tps"]["value"], run["gen_tps"]["value"])      # prompt processing is much faster per token
            self.assertEqual(run["gen_tps"]["status"], "measured")
        self.assertEqual(r["warm"]["gen_tps"]["n"], 2)
        self.assertEqual(r["load"]["seconds"]["status"], "measured")
        measured_buffers = r["comparison"]["measured"]
        self.assertGreater(measured_buffers["weights_ram_gb"]["value"], 0.05)
        self.assertGreater(measured_buffers["kv_cache_gb"]["value"], 0)
        self.assertIn("weights_ram_gb", r["comparison"]["delta_gb"])
        self.assertEqual(r["backend"]["probe"], "ok")
        self.assertTrue(r["backend"]["version"])
        if os.path.isdir("/proc"):
            self.assertEqual(r["metrics"]["peak_rss_gb"]["status"], "measured")
            self.assertGreater(r["metrics"]["peak_rss_gb"]["value"], 0.1)
        self.assertEqual(K.child_pids(), self.before)
        print("\nREAL ENGINE (this sandbox, not the owner's PC): warm generation "
              f"{r['warm']['gen_tps']['mean']} tokens/s, warm prompt {r['warm']['prompt_tps']['mean']} tokens/s, "
              f"cold generation {r['cold']['gen_tps']['value']} tokens/s, load {r['load']['seconds']['value']} s")

    def test_cancel_stops_the_real_server(self):
        cancel = threading.Event()
        threading.Timer(1.5, cancel.set).start()
        t0 = time.time()
        r = B.run_benchmark(self.settings, REAL_MODEL, "cancel", hw=self.hw, cancel=cancel,
                            spec=B.BenchSpec(prompt_tokens=512, gen_tokens=400, repeats=5))
        self.assertEqual(r["status"], "cancelled")
        self.assertLess(time.time() - t0, 30)
        self.assertEqual(K.child_pids(), self.before)

    def test_request_timeout_stops_the_real_server(self):
        r = B.run_benchmark(self.settings, REAL_MODEL, "timeout", hw=self.hw,
                            spec=B.BenchSpec(prompt_tokens=512, gen_tokens=2000, repeats=1, request_timeout_s=2.0))
        self.assertEqual(r["status"], "timeout")
        self.assertEqual(K.child_pids(), self.before)

    def test_optimizer_on_the_real_engine(self):
        from duckstoken.vgpu import optimizer as O
        from duckstoken.vgpu.store import VGpuStore
        with tempfile.TemporaryDirectory() as d:
            store = VGpuStore(d, fsync=False)
            r = O.run_optimization(self.settings, [REAL_MODEL], "max_speed", self.hw, store=store, max_candidates=3,
                                   spec=B.BenchSpec(prompt_tokens=64, gen_tokens=12, repeats=2, warmup=False))
            self.assertEqual(r["status"], "ok", r)
            rows = r["models"][0]["candidates"]
            self.assertEqual(rows[0]["id"], "cpu-baseline")
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(r["best"]["basis"], "measured")
            self.assertEqual(store.profile_status(r["best"]["profile_key"]), "fresh")
            self.assertEqual(K.child_pids(), self.before)
            print("\nREAL OPTIMIZER (this sandbox):", [(c["id"], c["status"], c["measured_gen_tps"]["value"]) for c in rows],
                  "->", r["best"]["candidate_id"])


if __name__ == "__main__":
    unittest.main()
