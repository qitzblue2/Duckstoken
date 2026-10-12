"""Optimizer: objectives, selection, refusal of rejected/failed candidates, pressure handling, persistence, apply/revert.

Most tests inject a deterministic planner and a fake `bench_fn`; the integration tests at the bottom run the real
benchmark against the fake llama-server (and one against the real planner)."""
import copy
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vgpu_kit as K  # noqa: E402

from duckstoken import settings as settings_mod  # noqa: E402
from duckstoken.vgpu import benchmark as B  # noqa: E402
from duckstoken.vgpu import optimizer as O  # noqa: E402
from duckstoken.vgpu import scheduler as S  # noqa: E402
from duckstoken.vgpu.store import VGpuStore, profile_key  # noqa: E402

PROBE_CUDA = {"devices": [{"id": "CUDA0", "backend": "CUDA", "name": "Test GPU"}], "backends_usable": ["CUDA"], "status": "ok"}
SPEC = B.BenchSpec(prompt_tokens=64, gen_tokens=8, repeats=2)


def val(v, status="measured"):
    return {"value": v, "status": status, "source": "test", "note": ""}


def stat(v, status="measured", sd=0.0):
    return {**val(v, status), "mean": v, "stdev": sd, "min": v, "max": v, "n": 2}


def make_rec(gen=10.0, rss=2.0, status="ok", error="", prompt=100.0, cold=None, vram=None, bid=None, ngl=0, sd=0.0):
    """A benchmark record with just the fields the optimizer reads."""
    return {"id": bid or f"bench-{gen}-{rss}", "status": status, "error": error, "created": "t",
            "warm": {"runs": [{}, {}], "gen_tps": stat(gen, sd=sd) if gen is not None else val(None, "unavailable"),
                     "prompt_tps": stat(prompt), "ttft_s": stat(0.1)},
            "cold": {"gen_tps": val(cold if cold is not None else gen)},
            "metrics": {"peak_rss_gb": val(rss) if rss is not None else val(None, "unavailable"),
                        "peak_vram_mb": val(vram) if vram is not None else val(None, "unavailable")},
            "backend": {"engine": "llama.cpp", "version": "0.6.0-dev", "build": 1, "commit": "abc1234"},
            "reproducibility": {"spec_hash": "s1"}, "config_hash": f"c{ngl}", "warnings": []}


class Script:
    """A fake `bench_fn`: records every call and answers from `plan(overrides, path) -> record | None`."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def __call__(self, settings, path, label="", **kw):
        ov = {k: settings.get(k) for k in ("n_gpu_layers", "n_cpu_moe", "kv_offload", "threads", "threads_batch", "cache_type_k",
                                           "ubatch_size", "extra_args")}
        self.calls.append({"settings": settings, "path": path, "label": label, "kw": kw, "ov": ov})
        return self.answer(settings, path, len(self.calls), label.split(': ')[-1])


class OptCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.planner = K.FakePlanner()
        for mod in (S, O):
            p = mock.patch.object(mod, "_mod", lambda name, _pl=self.planner: _pl if name == "planner" else None)
            p.start()
            self.addCleanup(p.stop)
        self.hw = K.mock_hw(vram_mb=4096)
        self.store = VGpuStore(self.dir / "store", fsync=False)
        self.model = K.make_model(self.dir, "m.gguf")

    def settings(self, **kw):
        s = dict(settings_mod.DEFAULTS)
        s.update(ctx_size=4096, **kw)
        return s

    def run_opt(self, script, objective="max_speed", models=None, **kw):
        kw.setdefault("spec", SPEC)
        kw.setdefault("store", self.store)
        kw.setdefault("probe", PROBE_CUDA)
        kw.setdefault("pressure_fn", lambda: "ok")
        return O.run_optimization(kw.pop("settings", self.settings()), models or [self.model], objective, self.hw,
                                  bench_fn=script, **kw)

    def cand_ids(self, result, model=0):
        return [c["id"] for c in result["models"][model]["candidates"]]


class TestSelection(unittest.TestCase):
    def entry(self, gen, res, size=100, frac=None, cid="c"):
        return {"model": {"size_bytes": size, "name": f"m{size}", "path": f"/m{size}"}, "cand": {"id": cid, "label": cid},
                "rec": None, "gen_tps": gen, "resident_gb": res, "resident": val(res), "ram_fraction": frac, "basis": "measured"}

    def test_max_speed_takes_the_fastest(self):
        best, why = O.pick_best("max_speed", [self.entry(5, 4, cid="a"), self.entry(9, 8, cid="b"), self.entry(7, 1, cid="c")])
        self.assertEqual(best["cand"]["id"], "b")
        self.assertIn("highest generation speed", why)

    def test_max_speed_tie_goes_to_lower_memory(self):
        best, why = O.pick_best("max_speed", [self.entry(10.0, 9, cid="a"), self.entry(9.8, 3, cid="b"), self.entry(8.0, 1, cid="c")])
        self.assertEqual(best["cand"]["id"], "b")                     # within 3 % of the best, uses less memory
        self.assertIn("within", why)
        best, _ = O.pick_best("max_speed", [self.entry(10.0, 9, cid="a"), self.entry(9.5, 3, cid="b")])
        self.assertEqual(best["cand"]["id"], "a")                     # 5 % slower is not a tie

    def test_balanced_formula_and_ram_limit(self):
        a, b = self.entry(10.0, 16.0, cid="a"), self.entry(7.0, 4.0, cid="b")
        self.assertAlmostEqual(O.score_of("balanced", a), 10 / 4)
        self.assertAlmostEqual(O.score_of("balanced", b), 7 / 2)
        best, why = O.pick_best("balanced", [a, b])
        self.assertEqual(best["cand"]["id"], "b")
        self.assertIn("sqrt", why)
        hungry = self.entry(50.0, 1.0, frac=0.95, cid="hungry")      # fastest and tiny, but >90 % of the RAM
        best, _ = O.pick_best("balanced", [hungry, b])
        self.assertEqual(best["cand"]["id"], "b")
        none, why = O.pick_best("balanced", [hungry])
        self.assertIsNone(none)
        self.assertIn("90", why)

    def test_balanced_tolerates_zero_memory(self):
        e = self.entry(10.0, 0.0, cid="z")
        e["resident_gb"] = 0.05
        self.assertTrue(O.score_of("balanced", e) > 0)

    def test_max_model_size_takes_the_largest_model_that_meets_min_tps(self):
        small, big = self.entry(20, 2, size=100, cid="s"), self.entry(2.5, 9, size=900, cid="b")
        best, _ = O.pick_best("max_model_size", [small, big], min_tps=1.0)
        self.assertEqual(best["cand"]["id"], "b")
        best, _ = O.pick_best("max_model_size", [small, big], min_tps=3.0)       # the big one is too slow now
        self.assertEqual(best["cand"]["id"], "s")

    def test_max_model_size_single_model_smallest_memory_that_meets_min_tps(self):
        es = [self.entry(5.0, 9.0, cid="fast-big"), self.entry(1.2, 3.0, cid="slow-small"), self.entry(0.5, 1.0, cid="too-slow")]
        best, why = O.pick_best("max_model_size", es, min_tps=1.0)
        self.assertEqual(best["cand"]["id"], "slow-small")
        self.assertIn("1 tokens/s", why)
        self.assertEqual(O.pick_best("max_model_size", es, min_tps=1.0)[0]["cand"]["id"], "slow-small")

    def test_max_model_size_near_equal_memory_prefers_speed(self):
        es = [self.entry(4.0, 3.00, cid="a"), self.entry(6.0, 3.05, cid="b"), self.entry(9.0, 5.0, cid="c")]
        self.assertEqual(O.pick_best("max_model_size", es)[0]["cand"]["id"], "b")

    def test_nothing_qualifies(self):
        best, why = O.pick_best("max_model_size", [self.entry(0.4, 1.0)], min_tps=1.0)
        self.assertIsNone(best)
        self.assertIn("minimum of 1 tokens/s", why)
        self.assertEqual(O.pick_best("max_speed", [])[0], None)

    def test_unknown_objective(self):
        with self.assertRaises(O.OptimizerError):
            O.pick_best("fastest-ever", [self.entry(1, 1)])

    def test_resident_prefers_measurement_and_adds_vram(self):
        r = O.resident_of(make_rec(rss=3.0, vram=2048), None)
        self.assertEqual((r["status"], r["value"]), ("measured", 5.0))
        self.assertIn("VRAM", r["note"])
        planner = K.FakePlanner()
        with mock.patch.object(O, "_mod", lambda name: planner):
            plan = planner.plan({}, {"name": "x"}, {"n_gpu_layers": 0})
            e = O.resident_of(make_rec(rss=None), plan)
            self.assertEqual(e["status"], "estimated")
            self.assertEqual(O.resident_of(None, None)["status"], "unavailable")

    def test_backend_id(self):
        self.assertEqual(O.backend_id({"version": "1", "build": 2, "commit": "c"}), "1|2|c")
        self.assertIsNone(O.backend_id({}))
        self.assertIsNone(O.backend_id(None))


class TestRunOptimization(OptCase):
    def fast_ok(self, settings, path, n, cid=None):
        # more GPU layers -> faster; the baseline is the slowest
        return make_rec(gen=5.0 + (settings.get("n_gpu_layers") or 0), rss=2.0, ngl=settings.get("n_gpu_layers") or 0,
                        bid=f"bench-{n}")

    def test_baseline_first_then_candidates_in_scheduler_order(self):
        script = Script(self.fast_ok)
        r = self.run_opt(script)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(script.calls[0]["ov"]["n_gpu_layers"], 0)
        self.assertIs(script.calls[0]["ov"]["kv_offload"], False)
        self.assertEqual(script.calls[0]["label"], "max_speed: cpu-baseline")
        order = [c["label"].split(": ")[1] for c in script.calls]
        self.assertEqual(order, [c["id"] for c in r["models"][0]["candidates"] if c["status"] == "ok"])
        self.assertEqual(order[0], "cpu-baseline")
        self.assertIs(r["applied"], False)
        self.assertEqual(r["models"][0]["baseline_id"], "cpu-baseline")

    def test_best_is_the_fastest_and_the_patch_is_complete(self):
        r = self.run_opt(Script(self.fast_ok))
        best = r["best"]
        self.assertEqual(best["candidate_id"], "gpu-14-kv-host")          # most layers on the GPU = fastest in this script
        self.assertEqual(best["basis"], "measured")
        self.assertEqual(best["patch"]["n_gpu_layers"], 14)
        self.assertEqual(best["patch"]["model_path"], self.model)
        self.assertEqual(best["gen_tps"]["status"], "measured")
        self.assertEqual(best["score"]["status"], "measured")
        settings_mod.validate_patch(best["patch"])                      # only real settings, valid values
        rows = {c["id"]: c for c in r["models"][0]["candidates"]}
        self.assertEqual(rows["cpu-baseline"]["speedup_vs_baseline"], 1.0)
        self.assertGreater(rows["gpu-12"]["speedup_vs_baseline"], 2.0)
        self.assertGreater(rows["gpu-14-kv-host"]["speedup_vs_baseline"], rows["gpu-12"]["speedup_vs_baseline"])

    def test_comparison_rows_show_estimate_next_to_measurement(self):
        r = self.run_opt(Script(self.fast_ok))
        row = next(c for c in r["comparison"] if c["candidate"] == "gpu-12")
        self.assertEqual(set(row) >= {"candidate", "estimated_tps", "measured_tps", "measured_gen_tps", "delta", "status"}, True)
        self.assertEqual(row["estimated_tps"]["status"], "estimated")
        self.assertEqual(row["measured_gen_tps"]["status"], "measured")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["delta"]["status"], "estimated")
        self.assertAlmostEqual(row["delta"]["value"], 17.0 - 6.0, places=1)    # measured 17 - estimate midpoint 6

    def test_rejected_candidates_never_reach_the_benchmark(self):
        self.planner.reject = lambda cfg: ["VRAM overcommit"] if (cfg.get("n_gpu_layers") or 0) >= 8 else None
        script = Script(self.fast_ok)
        r = self.run_opt(script)
        run_ngl = [c["ov"]["n_gpu_layers"] for c in script.calls]
        self.assertTrue(all(n < 8 for n in run_ngl), run_ngl)
        rows = {c["id"]: c for c in r["models"][0]["candidates"]}
        for cid in ("gpu-8", "gpu-12"):
            self.assertEqual(rows[cid]["status"], "rejected")
            self.assertIn("VRAM overcommit", rows[cid]["reason"])
            self.assertEqual(rows[cid]["measured_gen_tps"]["status"], "unavailable")
        self.assertIn({"model": "tiny-dense", "candidate": "gpu-12", "reason": "VRAM overcommit"}, r["rejected"])
        self.assertNotIn(r["best"]["candidate_id"], ("gpu-8", "gpu-12"))

    def test_failed_and_timed_out_candidates_never_win_and_are_listed(self):
        def answer(settings, path, n, cid):
            if cid == "gpu-12":
                return make_rec(gen=500.0, status="failed", error="llama-server exited while loading (code 1)", bid="b-fail")
            if cid == "gpu-8":
                return make_rec(gen=400.0, status="timeout", error="a request took longer than 600 s", bid="b-timeout")
            return self.fast_ok(settings, path, n, cid)

        r = self.run_opt(Script(answer))
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["best"]["candidate_id"], "gpu-14-kv-host")       # fastest of those that really ran
        self.assertNotIn(r["best"]["candidate_id"], ("gpu-12", "gpu-8"))
        reverted = {x["candidate"]: x for x in r["reverted"]}
        self.assertEqual(set(reverted), {"gpu-12", "gpu-8"})
        self.assertEqual(reverted["gpu-12"]["status"], "failed")
        self.assertIn("code 1", reverted["gpu-12"]["reason"])
        self.assertEqual(reverted["gpu-8"]["status"], "timeout")
        self.assertEqual(reverted["gpu-12"]["benchmark_id"], "b-fail")
        rows = {c["id"]: c for c in r["models"][0]["candidates"]}
        self.assertEqual(rows["gpu-12"]["status"], "failed")

    def test_best_is_never_a_failed_run_even_if_it_is_the_only_one_with_numbers(self):
        def answer(settings, path, n, cid):
            return make_rec(gen=50.0, status="failed", error="boom") if n > 1 else make_rec(gen=3.0)

        script = Script(answer)
        r = self.run_opt(script)
        self.assertEqual(r["best"]["candidate_id"], "cpu-baseline")
        self.assertEqual(len(r["reverted"]), len(script.calls) - 1)          # every run after the baseline failed

    def test_baseline_failure_stops_everything_else(self):
        script = Script(lambda s, p, n, c: make_rec(status="failed", error="out of memory"))
        r = self.run_opt(script)
        self.assertEqual(len(script.calls), 1)
        self.assertEqual(r["status"], "no_result")
        self.assertIsNone(r["best"])
        self.assertEqual(r["reverted"][0]["candidate"], "cpu-baseline")
        self.assertTrue(all(c["status"] in ("failed", "skipped", "rejected") for c in r["models"][0]["candidates"]))
        self.assertTrue(r["skipped"])
        self.assertTrue(all("baseline" in x["reason"] for x in r["skipped"]))
        self.assertEqual(self.store.profiles(), {})

    def test_baseline_rejected_by_the_planner_still_allows_admitted_gpu_candidates(self):
        self.planner.reject = lambda cfg: ["CPU-only needs 40 GB of RAM"] if (cfg.get("n_gpu_layers") or 0) == 0 else None
        script = Script(self.fast_ok)
        r = self.run_opt(script)
        self.assertTrue(script.calls)
        self.assertTrue(all(c["ov"]["n_gpu_layers"] > 0 for c in script.calls))
        rows = {c["id"]: c for c in r["models"][0]["candidates"]}
        self.assertEqual(rows["cpu-baseline"]["status"], "rejected")
        self.assertIsNone(rows["gpu-12"]["speedup_vs_baseline"])           # no baseline to compare with
        self.assertIsNotNone(r["best"])

    def test_nothing_admitted_runs_nothing(self):
        self.planner.reject = lambda cfg: ["does not fit"]
        script = Script(self.fast_ok)
        r = self.run_opt(script)
        self.assertEqual(script.calls, [])
        self.assertEqual(r["status"], "no_result")
        self.assertIsNone(r["best"])

    def test_max_candidates_limits_runs_and_baseline_always_runs(self):
        script = Script(self.fast_ok)
        r = self.run_opt(script, max_candidates=3)
        self.assertEqual(len(script.calls), 3)
        rows = r["models"][0]["candidates"]
        self.assertEqual(sum(1 for c in rows if c["status"] == "ok"), 3)
        limited = [c for c in rows if c["status"] == "skipped"]
        self.assertTrue(limited)
        self.assertTrue(all("limit of 3 candidates" in c["reason"] for c in limited))
        one = Script(self.fast_ok)
        self.run_opt(one, max_candidates=1)
        self.assertEqual(len(one.calls), 1)

    def test_high_pressure_skips_a_candidate_critical_aborts_the_run(self):
        levels = iter(["ok", "high", "ok", "ok", "ok", "ok", "ok", "ok", "ok"])
        script = Script(self.fast_ok)
        r = self.run_opt(script, pressure_fn=lambda: next(levels, "ok"))
        rows = r["models"][0]["candidates"]
        skipped = [c for c in rows if c["status"] == "skipped"]
        self.assertTrue(any("pressure is high" in c["reason"] for c in skipped))
        self.assertEqual(r["status"], "ok")
        self.assertEqual(len(script.calls), len([c for c in rows if c["status"] == "ok"]))

        levels = iter(["ok", "critical"])
        script = Script(self.fast_ok)
        self.store = VGpuStore(self.dir / "store2", fsync=False)
        r = self.run_opt(script, pressure_fn=lambda: next(levels, "critical"), store=self.store)
        self.assertEqual(r["status"], "aborted")
        self.assertEqual(len(script.calls), 1)                               # only the baseline ran
        self.assertIn("critically short of memory", r["error"])
        self.assertIsNone(r["best"])
        self.assertEqual(self.store.profiles(), {})
        self.assertTrue(all(c["status"] in ("ok", "skipped", "rejected") for c in r["models"][0]["candidates"]))

    def test_critical_pressure_before_the_baseline_runs_nothing(self):
        script = Script(self.fast_ok)
        r = self.run_opt(script, pressure_fn=lambda: "critical")
        self.assertEqual(script.calls, [])
        self.assertEqual(r["status"], "aborted")

    def test_cancel_stops_after_the_current_candidate(self):
        cancel = threading.Event()

        def answer(settings, path, n, cid):
            if n == 2:
                cancel.set()
            return self.fast_ok(settings, path, n, cid)

        script = Script(answer)
        r = self.run_opt(script, cancel=cancel)
        self.assertEqual(r["status"], "cancelled")
        self.assertEqual(len(script.calls), 2)
        self.assertIsNone(r["best"])
        self.assertFalse(r["applied"])
        self.assertEqual(self.store.profiles(), {})
        self.assertIn("did not finish", " ".join(r["notes"]))

    def test_a_cancelled_benchmark_ends_the_run_and_is_not_a_result(self):
        script = Script(lambda s, p, n, c: make_rec(status="cancelled", error="cancelled by the user") if n == 2 else make_rec(gen=3.0))
        r = self.run_opt(script)
        self.assertEqual(r["status"], "cancelled")
        self.assertEqual(len(script.calls), 2)
        self.assertEqual(r["reverted"][0]["status"], "cancelled")

    def test_settings_are_never_modified(self):
        s = self.settings(threads=3)
        before = copy.deepcopy(s)
        r = self.run_opt(Script(self.fast_ok), settings=s)
        self.assertEqual(s, before)
        self.assertIs(r["applied"], False)

    def test_each_run_gets_a_merged_copy_not_the_users_dict(self):
        script = Script(self.fast_ok)
        s = self.settings()
        self.run_opt(script, settings=s)
        self.assertTrue(all(c["settings"] is not s for c in script.calls))
        self.assertEqual(s["n_gpu_layers"], settings_mod.DEFAULTS["n_gpu_layers"])

    def test_benchmarks_history_and_profile_are_persisted(self):
        r = self.run_opt(Script(self.fast_ok))
        ids = [x["id"] for x in self.store.benchmarks(100)]
        self.assertEqual(sorted(ids), sorted(r["benchmarks"]))
        hist = self.store.history(5)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["best"]["candidate_id"], r["best"]["candidate_id"])
        self.assertFalse(hist[0]["estimate_only"])
        key = r["best"]["profile_key"]
        prof = self.store.get_profile(key)
        self.assertEqual(prof["patch"], r["best"]["patch"])
        self.assertEqual(prof["objective"], "max_speed")
        self.assertFalse(prof["applied"])
        self.assertIsNone(prof["previous"])
        self.assertEqual(prof["measured"]["gen_tps"]["status"], "measured")
        self.assertFalse(prof["estimated_only"])
        self.assertIn("hardware_fingerprint", prof)
        self.assertEqual(self.store.profile_status(key), "fresh")
        model = {"name": "tiny-dense", "size_bytes": Path(self.model).stat().st_size, "file_type": 15}
        self.assertEqual(key, profile_key(self.hw, model, "0.6.0-dev|1|abc1234"))
        self.assertEqual(self.store.profile_status(profile_key(K.mock_hw(ram_gb=64, vram_mb=4096), model, "0.6.0-dev|1|abc1234")), "stale")

    def test_store_failure_is_noted_not_fatal(self):
        class BadStore:
            def add_benchmark(self, rec):
                raise OSError("disk full")

            add_history = save_profile = add_benchmark

        r = self.run_opt(Script(self.fast_ok), store=BadStore())
        self.assertEqual(r["status"], "ok")
        self.assertIsNotNone(r["best"])
        self.assertFalse(r["best"]["profile_saved"])
        self.assertTrue(any("disk full" in n for n in r["notes"]))

    def test_progress_and_perflog(self):
        steps, events = [], []

        class Log:
            def log(self, event, **f):
                events.append((event, f))

        script = Script(self.fast_ok)
        r = self.run_opt(script, progress=lambda *a: steps.append(a), perflog=Log())
        self.assertEqual([s[0] for s in steps], list(range(1, len(script.calls) + 1)))
        self.assertTrue(all(s[1] >= s[0] for s in steps))
        self.assertIn("CPU only (baseline)", steps[0][2])
        self.assertEqual(events[-1][0], "optimizer_done")
        self.assertEqual(events[-1][1]["best"], r["best"]["candidate_id"])

    def test_apply_cb_is_called_only_with_a_good_result_and_its_failure_is_contained(self):
        got = []
        r = self.run_opt(Script(self.fast_ok), apply_cb=lambda patch, profile: got.append((patch, profile)) or True)
        self.assertTrue(r["applied"])
        self.assertEqual(got[0][0], r["best"]["patch"])
        self.assertEqual(got[0][1]["key"], r["best"]["profile_key"])

        def boom(patch, profile):
            raise RuntimeError("settings file locked")

        r = self.run_opt(Script(self.fast_ok), apply_cb=boom)
        self.assertFalse(r["applied"])
        self.assertTrue(any("nothing was changed" in n for n in r["notes"]))

        got.clear()
        r = self.run_opt(Script(lambda s, p, n, c: make_rec(status="failed")), apply_cb=lambda *a: got.append(a))
        self.assertEqual(got, [])
        self.assertFalse(r["applied"])

    def test_result_is_plain_json(self):
        r = self.run_opt(Script(self.fast_ok))
        json.dumps(r)                                   # the HTTP layer returns it as is
        json.dumps(O.estimate_only(self.settings(), [self.model], "balanced", self.hw, probe=PROBE_CUDA))

    def test_winner_reports_its_margin_over_the_baseline(self):
        r = self.run_opt(Script(self.fast_ok))
        vb = r["best"]["vs_baseline"]
        self.assertGreater(vb["ratio"], 2.0)
        self.assertTrue(vb["distinguishable"])
        self.assertFalse(any("measurement noise" in n for n in r["notes"]))

    def test_a_win_inside_the_noise_is_called_a_tie_not_a_win(self):
        def answer(settings, path, n, cid):
            ngl = settings.get("n_gpu_layers") or 0
            gen = 11.0 if cid == "gpu-4" else 10.0
            return make_rec(gen=gen, rss=2.0, sd=2.0, ngl=ngl)          # 20 % run-to-run spread

        r = self.run_opt(Script(answer))
        vb = r["best"]["vs_baseline"]
        self.assertIsNotNone(vb)
        self.assertFalse(vb["distinguishable"])
        self.assertIn("equivalent", vb["note"])
        self.assertTrue(any("measurement noise" in n for n in r["notes"]))

    def test_vs_baseline_edge_cases(self):
        e = {"gen_tps": 12.0, "rec": make_rec(gen=12.0)}
        base = {"gen_tps": 10.0, "rec": make_rec(gen=10.0)}
        self.assertIsNone(O.vs_baseline(e, None))
        self.assertIsNone(O.vs_baseline(e, e))
        self.assertIsNone(O.vs_baseline(e, {"gen_tps": 0, "rec": None}))
        self.assertEqual(O.vs_baseline(e, base)["ratio"], 1.2)
        self.assertTrue(O.vs_baseline(e, base)["distinguishable"])
        slower = {"gen_tps": 8.0, "rec": make_rec(gen=8.0)}
        self.assertTrue(O.vs_baseline(slower, base)["distinguishable"])          # clearly slower is also a real difference
        self.assertIsNone(O._spread({"warm": {"gen_tps": {"mean": 5.0, "stdev": 1.0, "n": 1}}}))

    def test_default_never_applies(self):
        self.assertIs(self.run_opt(Script(self.fast_ok))["applied"], False)

    def test_bad_arguments(self):
        with self.assertRaises(O.OptimizerError):
            self.run_opt(Script(self.fast_ok), objective="fastest")
        with self.assertRaises(O.OptimizerError):
            O.run_optimization(self.settings(), [], "max_speed", self.hw)
        with self.assertRaises(O.OptimizerError):
            O.estimate_only(self.settings(), [], "max_speed", self.hw)

    def test_unreadable_model_is_reported_and_the_others_still_run(self):
        junk = self.dir / "junk.gguf"
        junk.write_bytes(b"nope")
        script = Script(self.fast_ok)
        r = self.run_opt(script, models=[str(junk), self.model])
        self.assertEqual(r["status"], "ok")
        self.assertIn("cannot read the model file", r["models"][0]["error"])
        self.assertEqual(r["rejected"][0]["model"], str(junk))
        self.assertTrue(script.calls and all(c["path"] == self.model for c in script.calls))

    def test_models_are_deduplicated(self):
        script = Script(self.fast_ok)
        r = self.run_opt(script, models=[self.model, self.model])
        self.assertEqual(len(r["models"]), 1)

    def test_candidates_whose_planner_run_crashes_are_reported_per_model(self):
        class Crashy(K.FakePlanner):
            def plan(self, *a, **k):
                raise KeyError("planner bug")

        crashy = Crashy()
        with mock.patch.object(S, "_mod", lambda name: crashy if name == "planner" else None):
            script = Script(self.fast_ok)
            r = self.run_opt(script)
        self.assertEqual(script.calls, [])
        self.assertIn("cannot plan candidates", r["models"][0]["error"])
        self.assertEqual(r["status"], "no_result")


class TestObjectivesEndToEndWithScriptedNumbers(OptCase):
    def two_models(self):
        small = K.make_model(self.dir, "small.gguf", pad=100 * 1024)
        big = K.make_model(self.dir, "big.gguf", {**K.DENSE, "general.name": "big-dense"}, pad=5 * 1024 * 1024)
        return small, big

    def test_max_model_size_prefers_the_bigger_model_when_it_is_usable(self):
        small, big = self.two_models()

        def answer(settings, path, n, cid):
            return make_rec(gen=2.0 if path == big else 30.0, rss=9.0 if path == big else 1.0)

        r = self.run_opt(Script(answer), objective="max_model_size", models=[small, big], max_candidates=2)
        self.assertEqual(r["best"]["model_path"], big)
        self.assertEqual(r["best"]["patch"]["model_path"], big)

    def test_max_model_size_falls_back_when_the_big_model_is_too_slow(self):
        small, big = self.two_models()

        def answer(settings, path, n, cid):
            return make_rec(gen=0.4 if path == big else 30.0, rss=9.0 if path == big else 1.0)

        r = self.run_opt(Script(answer), objective="max_model_size", models=[small, big], max_candidates=2, min_tps=1.0)
        self.assertEqual(r["best"]["model_path"], small)
        self.assertIn("largest model that reaches 1 tokens/s", r["best"]["reason"])
        rows = {c["id"]: c for m in r["models"] for c in m["candidates"] if m["path"] == big}
        self.assertEqual(rows["cpu-baseline"]["status"], "ok")           # the big model was measured, just too slow

    def test_max_model_size_single_model_picks_the_leanest_config_that_is_fast_enough(self):
        def answer(settings, path, n, cid):
            ngl = settings.get("n_gpu_layers") or 0
            return make_rec(gen=1.5 if ngl == 0 else 8.0, rss=2.0 if ngl == 0 else 7.0, ngl=ngl)

        r = self.run_opt(Script(answer), objective="max_model_size")
        self.assertEqual(r["best"]["candidate_id"], "cpu-baseline")
        self.assertEqual(r["best"]["resident_gb"]["status"], "measured")

    def test_balanced_excludes_configs_that_use_over_ninety_percent_of_ram(self):
        hw = K.mock_hw(ram_gb=10.0, vram_mb=4096)

        def answer(settings, path, n, cid):
            ngl = settings.get("n_gpu_layers") or 0
            return make_rec(gen=40.0 if ngl else 5.0, rss=9.6 if ngl else 2.0, ngl=ngl)

        script = Script(answer)
        r = O.run_optimization(self.settings(), [self.model], "balanced", hw, spec=SPEC, store=self.store, probe=PROBE_CUDA,
                               bench_fn=script, pressure_fn=lambda: "ok")
        self.assertEqual(r["best"]["candidate_id"], "cpu-baseline")
        self.assertEqual(r["best"]["objective"], "balanced")
        self.assertEqual(r["best"]["score"]["status"], "measured")


class TestEstimateOnly(OptCase):
    def test_no_process_no_measurement_everything_labelled_estimated(self):
        script = Script(self.run_must_not_happen)
        r = O.estimate_only(self.settings(), [self.model], "max_speed", self.hw, probe=PROBE_CUDA, store=self.store)
        self.assertEqual(r["status"], "estimate_only")
        self.assertEqual(script.calls, [])
        self.assertEqual(r["best"]["basis"], "estimated")
        self.assertEqual(r["best"]["gen_tps"]["status"], "estimated")
        self.assertEqual(r["best"]["prompt_tps"]["status"], "unavailable")
        self.assertEqual(r["best"]["score"]["status"], "estimated")
        self.assertEqual(r["benchmarks"], [])
        self.assertIsNone(r["best"]["profile_key"])
        self.assertIs(r["applied"], False)
        for row in r["comparison"]:
            self.assertEqual(row["measured_gen_tps"]["status"], "unavailable")
            self.assertEqual(row["estimated_tps"]["status"], "estimated")
            self.assertIn(row["status"], ("estimated", "rejected"))
        self.assertEqual(self.store.profiles(), {})            # an estimate is not a learned profile
        self.assertTrue(self.store.history(1)[0]["estimate_only"])
        self.assertTrue(any("Estimates only" in n for n in r["notes"]))

    def run_must_not_happen(self, *a):
        raise AssertionError("a benchmark was started in estimate-only mode")

    def test_allow_run_false_routes_to_estimates(self):
        with mock.patch.object(B, "run_benchmark", side_effect=AssertionError("must not run")):
            r = O.run_optimization(self.settings(), [self.model], "max_speed", self.hw, allow_run=False, probe=PROBE_CUDA)
        self.assertEqual(r["status"], "estimate_only")

    def test_estimate_selection_uses_planner_numbers(self):
        self.planner.speed_fn = lambda cfg: (2.0, 4.0) if not cfg.get("n_gpu_layers") else (10.0, 14.0)
        r = O.estimate_only(self.settings(), [self.model], "max_speed", self.hw, probe=PROBE_CUDA)
        self.assertNotEqual(r["best"]["candidate_id"], "cpu-baseline")
        self.assertEqual(r["best"]["gen_tps"]["value"], 12.0)
        self.assertIn("model_path", r["best"]["patch"])

    def test_estimate_respects_planner_rejections(self):
        self.planner.reject = lambda cfg: ["no"] if (cfg.get("n_gpu_layers") or 0) > 0 else None
        self.planner.speed_fn = lambda cfg: (50.0, 60.0) if cfg.get("n_gpu_layers") else (2.0, 3.0)
        r = O.estimate_only(self.settings(), [self.model], "max_speed", self.hw, probe=PROBE_CUDA)
        self.assertEqual(r["best"]["patch"]["n_gpu_layers"], 0)

    def test_nothing_admitted_gives_no_best(self):
        self.planner.reject = lambda cfg: ["no"]
        r = O.estimate_only(self.settings(), [self.model], "max_model_size", self.hw)
        self.assertIsNone(r["best"])


class TestApplyRevert(OptCase):
    def setUp(self):
        super().setUp()
        self.result = self.run_opt(Script(lambda s, p, n, c: make_rec(gen=5.0 + (s.get("n_gpu_layers") or 0), ngl=s.get("n_gpu_layers") or 0)))
        self.key = self.result["best"]["profile_key"]
        self.patch = self.result["best"]["patch"]
        self.current = self.settings(n_gpu_layers=99, kv_offload=True, threads=5)
        self.applied = []

    def updater(self, patch):
        self.applied.append(dict(patch))
        self.current.update(patch)
        return patch

    def test_apply_stores_previous_values_and_calls_the_updater(self):
        res = O.apply_best(self.store, self.key, self.updater, current=dict(self.current))
        self.assertTrue(res["applied"])
        self.assertTrue(res["revertible"])
        self.assertEqual(self.applied, [self.patch])
        self.assertEqual(res["previous"]["n_gpu_layers"], 99)
        self.assertEqual(res["previous"]["kv_offload"], True)
        self.assertEqual(res["previous"]["model_path"], "")
        prof = self.store.get_profile(self.key)
        self.assertTrue(prof["applied"])
        self.assertEqual(prof["previous"], res["previous"])
        self.assertIn("applied_at", prof)

    def test_revert_restores_exactly_what_was_replaced(self):
        original = dict(self.current)
        O.apply_best(self.store, self.key, self.updater, current=dict(self.current))
        self.assertNotEqual(self.current["n_gpu_layers"], original["n_gpu_layers"])
        res = O.revert_profile(self.store, self.key, self.updater)
        self.assertTrue(res["reverted"])
        for k in self.patch:
            self.assertEqual(self.current[k], original[k], k)
        prof = self.store.get_profile(self.key)
        self.assertFalse(prof["applied"])
        self.assertIsNone(prof["previous"])
        self.assertFalse(O.revert_profile(self.store, self.key, self.updater)["reverted"])     # nothing left to revert

    def test_rerunning_the_optimizer_after_apply_keeps_undo_working(self):
        # review finding: a second run replaced the stored profile and erased the saved previous values
        original = dict(self.current)
        O.apply_best(self.store, self.key, self.updater, current=dict(self.current))
        again = self.run_opt(Script(lambda s, p, n, c: make_rec(gen=7.0 + (s.get("n_gpu_layers") or 0), ngl=s.get("n_gpu_layers") or 0)))
        self.assertEqual(again["best"]["profile_key"], self.key)
        prof = self.store.get_profile(self.key)
        self.assertTrue(prof["applied"])
        self.assertEqual(prof["previous"]["n_gpu_layers"], original["n_gpu_layers"])
        res = O.revert_profile(self.store, self.key, self.updater)
        self.assertTrue(res["reverted"], res)
        self.assertEqual(self.current["n_gpu_layers"], original["n_gpu_layers"])

    def test_applying_twice_keeps_the_original_previous_values(self):
        original = dict(self.current)
        O.apply_best(self.store, self.key, self.updater, current=dict(self.current))
        again = O.apply_best(self.store, self.key, self.updater, current=dict(self.current))       # `current` is already patched
        self.assertEqual(again["previous"]["n_gpu_layers"], original["n_gpu_layers"])
        O.revert_profile(self.store, self.key, self.updater)
        self.assertEqual(self.current["n_gpu_layers"], original["n_gpu_layers"])

    def test_without_current_it_applies_but_says_it_cannot_be_reverted(self):
        res = O.apply_best(self.store, self.key, self.updater)
        self.assertTrue(res["applied"])
        self.assertFalse(res["revertible"])
        self.assertIn("cannot be reverted", res["note"])
        self.assertFalse(O.revert_profile(self.store, self.key, self.updater)["reverted"])

    def test_failing_updater_leaves_the_profile_unapplied(self):
        def broken(patch):
            raise ValueError("invalid value")

        with self.assertRaises(ValueError):
            O.apply_best(self.store, self.key, broken, current=self.current)
        self.assertFalse(self.store.get_profile(self.key)["applied"])
        self.assertIsNone(self.store.get_profile(self.key)["previous"])

    def test_unknown_key_and_estimates_are_refused(self):
        with self.assertRaises(O.OptimizerError):
            O.apply_best(self.store, "nope", self.updater)
        with self.assertRaises(O.OptimizerError):
            O.revert_profile(self.store, "nope", self.updater)
        self.store.save_profile("est", {"patch": {"n_gpu_layers": 3}, "estimated_only": True})
        with self.assertRaises(O.OptimizerError):
            O.apply_best(self.store, "est", self.updater)
        self.assertEqual(self.applied, [])

    def test_stale_profile_is_visible_when_hardware_changes(self):
        model = {"name": "tiny-dense", "size_bytes": Path(self.model).stat().st_size, "file_type": 15}
        newer = profile_key(K.mock_hw(ram_gb=64, vram_mb=4096), model, "0.6.0-dev|1|abc1234")
        self.assertEqual(self.store.profile_status(newer), "stale")
        self.assertEqual([p["key"] for p in self.store.stale_profiles(newer)], [self.key])


class TestIntegrationWithTheFakeEngine(unittest.TestCase):
    """Real benchmark runs against the fake llama-server; processes must never leak."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.model = K.make_model(self.dir)
        self.factory = K.ServerFactory()
        for patcher in (mock.patch.object(B, "ServerManager", self.factory),
                        mock.patch.object(B, "_mod", lambda name, _r=B._mod: K.FakeProbe() if name == "probe" else _r(name))):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.store = VGpuStore(self.dir / "store", fsync=False)
        self.hw = K.mock_hw(vram_mb=4096)
        self.settings = K.fake_settings(self.dir, "--fake-load-s 0.05")

    def scripted_candidates(self, speeds):
        """Candidates that differ only in the fake engine's speed knob (and are all admitted)."""
        planner = K.FakePlanner()
        base = {"cpu-baseline": {"n_gpu_layers": 0, "extra_args": f"--fake-load-s 0.05 --fake-tg-tps {speeds[0]}"}}
        cands = [{"id": "cpu-baseline", "label": "baseline", "overrides": base["cpu-baseline"], "group": "baseline",
                  "plan": planner.plan(self.hw, {"name": "x"}, {"n_gpu_layers": 0}), "admitted": True, "reject_reason": "", "expected": "x"}]
        for i, tps in enumerate(speeds[1:], 1):
            cands.append({"id": f"fast-{i}", "label": f"candidate {i}", "group": "gpu",
                          "overrides": {"n_gpu_layers": i, "extra_args": f"--fake-load-s 0.05 --fake-tg-tps {tps}"},
                          "plan": planner.plan(self.hw, {"name": "x"}, {"n_gpu_layers": i}), "admitted": True,
                          "reject_reason": "", "expected": "x"})
        return planner, cands

    def test_measured_winner_is_chosen_and_no_process_leaks(self):
        planner, cands = self.scripted_candidates([300, 900, 600])
        before = K.child_pids()
        with mock.patch.object(O.scheduler, "candidate_configs", lambda *a, **k: cands), \
                mock.patch.object(O, "_mod", lambda name: planner if name == "planner" else None), \
                mock.patch.object(B, "_mod", lambda name, _r=B._mod: planner if name == "planner" else (K.FakeProbe() if name == "probe" else _r(name))):
            r = O.run_optimization(self.settings, [self.model], "max_speed", self.hw, spec=B.BenchSpec(prompt_tokens=32, gen_tokens=6, repeats=2),
                                   store=self.store, pressure_fn=lambda: "ok")
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(r["best"]["candidate_id"], "fast-1")
        self.assertEqual(r["best"]["gen_tps"]["status"], "measured")
        rows = {c["id"]: c for c in r["models"][0]["candidates"]}
        self.assertGreater(rows["fast-1"]["speedup_vs_baseline"], 2.0)
        self.assertEqual(len(self.store.benchmarks(10)), 3)
        self.assertEqual(self.factory.alive(), [])
        self.assertEqual(len(self.factory.procs), 3)
        self.assertEqual(K.child_pids(), before)
        self.assertEqual(K.thread_names(), [])
        # the first benchmark in the store is the baseline (run first)
        self.assertEqual(self.store.benchmarks(10)[-1]["label"], "max_speed: cpu-baseline")

    def test_a_crashing_candidate_is_reverted_and_the_next_one_still_runs(self):
        planner, cands = self.scripted_candidates([300, 900, 600])
        cands[1]["overrides"]["extra_args"] += " --fake-crash-after 0"
        with mock.patch.object(O.scheduler, "candidate_configs", lambda *a, **k: cands), \
                mock.patch.object(O, "_mod", lambda name: planner if name == "planner" else None), \
                mock.patch.object(B, "_mod", lambda name, _r=B._mod: planner if name == "planner" else (K.FakeProbe() if name == "probe" else _r(name))):
            r = O.run_optimization(self.settings, [self.model], "max_speed", self.hw, spec=B.BenchSpec(prompt_tokens=32, gen_tokens=6, repeats=1),
                                   store=self.store, pressure_fn=lambda: "ok")
        self.assertEqual([x["candidate"] for x in r["reverted"]], ["fast-1"])
        self.assertEqual(r["best"]["candidate_id"], "fast-2")
        self.assertEqual(self.factory.alive(), [])

    def test_cancel_mid_optimization_stops_the_server_and_keeps_nothing_applied(self):
        planner, cands = self.scripted_candidates([20, 20, 20])
        cancel = threading.Event()
        threading.Timer(0.8, cancel.set).start()
        with mock.patch.object(O.scheduler, "candidate_configs", lambda *a, **k: cands), \
                mock.patch.object(O, "_mod", lambda name: planner if name == "planner" else None), \
                mock.patch.object(B, "_mod", lambda name, _r=B._mod: planner if name == "planner" else (K.FakeProbe() if name == "probe" else _r(name))):
            r = O.run_optimization(self.settings, [self.model], "max_speed", self.hw, spec=B.BenchSpec(prompt_tokens=32, gen_tokens=60, repeats=2),
                                   store=self.store, cancel=cancel, pressure_fn=lambda: "ok")
        self.assertEqual(r["status"], "cancelled")
        self.assertIsNone(r["best"])
        self.assertFalse(r["applied"])
        self.assertEqual(self.factory.alive(), [])
        self.assertEqual(K.child_pids(), [])
        self.assertEqual(self.store.profiles(), {})

    def test_real_scheduler_and_real_planner_end_to_end(self):
        r = O.run_optimization(self.settings, [self.model], "balanced", self.hw, probe=PROBE_CUDA, max_candidates=3,
                               spec=B.BenchSpec(prompt_tokens=32, gen_tokens=6, repeats=1), store=self.store,
                               pressure_fn=lambda: "ok")
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(r["models"][0]["candidates"][0]["id"], "cpu-baseline")
        self.assertEqual(r["models"][0]["candidates"][0]["status"], "ok")
        self.assertEqual(len([c for c in r["models"][0]["candidates"] if c["status"] == "ok"]), 3)
        self.assertTrue(r["best"]["profile_key"])
        self.assertEqual(self.store.profile_status(r["best"]["profile_key"]), "fresh")
        self.assertEqual(self.factory.alive(), [])
        for c in r["models"][0]["candidates"]:
            self.assertIn(c["verdict"], ("ok", "tight", "paging", "reject"))
        self.assertEqual(K.child_pids(), [])


if __name__ == "__main__":
    unittest.main()
