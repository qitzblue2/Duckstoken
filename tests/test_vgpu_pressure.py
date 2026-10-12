import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from duckstoken.vgpu import hwmodel, planner, pressure, telemetry
from duckstoken.vgpu.values import measured, unavailable

GPT120 = {"name": "gpt-oss-120b", "architecture": "gpt-oss", "size_bytes": int(63.4e9), "block_count": 36, "expert_count": 128,
          "expert_used_count": 4, "is_moe": True, "embedding_length": 2880, "head_count": 64, "head_count_kv": 8, "key_length": 64,
          "value_length": 64, "expert_feed_forward_length": 2880, "feed_forward_length": 2880, "sliding_window": 128,
          "context_length": 131072, "file_type": 38, "file_type_name": "MXFP4_MOE", "vocab_size": 201088, "remote": True}
GPT20 = dict(GPT120, name="gpt-oss-20b", size_bytes=int(12.1e9), block_count=24, expert_count=32)


def sample(total=48.0, avail=30.0, swap_total=0.0, swap_pct=0.0, commit_pct=None, psi=None, commit_src="GetPerformanceInfo", t=None):
    m = lambda v, src="test": measured(v, src)
    s = {"t": t if t is not None else time.time(),
         "ram": {"total_gb": m(total), "avail_gb": m(avail), "used_pct": m(100 * (1 - avail / total)),
                 "pressure_pct": m(psi) if psi is not None else unavailable("no PSI")},
         "swap": {"total_gb": m(swap_total), "used_gb": m(swap_total * swap_pct / 100), "used_pct": m(swap_pct)},
         "commit": {"used_pct": measured(commit_pct, commit_src) if commit_pct is not None else unavailable("none")},
         "cpu": {"util_pct": m(50.0)}, "gpus": [], "sources": {}}
    return s


class ClassifyTests(unittest.TestCase):
    def level(self, **kw):
        return pressure.classify_pressure(sample(**kw))["level"]

    def test_default_percentage_thresholds(self):
        self.assertEqual(self.level(total=100, avail=60), "ok")
        self.assertEqual(self.level(total=100, avail=26), "ok")
        self.assertEqual(self.level(total=100, avail=24), "elevated")
        self.assertEqual(self.level(total=100, avail=11), "high")
        self.assertEqual(self.level(total=100, avail=5), "critical")

    def test_absolute_critical_floor(self):
        self.assertEqual(self.level(total=64, avail=1.4), "critical")  # 2.2% anyway
        self.assertEqual(self.level(total=16, avail=1.4), "critical")  # 8.75% but under 1.5 GB
        self.assertEqual(self.level(total=16, avail=1.6), "high")

    def test_boundaries_are_strict(self):
        self.assertEqual(self.level(total=100, avail=25.0), "ok")
        self.assertEqual(self.level(total=100, avail=12.0), "elevated")
        self.assertEqual(self.level(total=100, avail=6.0), "high")

    def test_result_shape(self):
        r = pressure.classify_pressure(sample(total=48, avail=6))
        self.assertEqual(set(r), {"level", "avail_gb", "avail_pct", "known", "reasons"})
        self.assertEqual((r["avail_gb"], r["avail_pct"], r["known"]), (6, 12.5, True))
        self.assertEqual(r["level"], "elevated")
        self.assertTrue(r["reasons"])

    def test_custom_thresholds(self):
        s = sample(total=100, avail=40)
        self.assertEqual(pressure.classify_pressure(s)["level"], "ok")
        self.assertEqual(pressure.classify_pressure(s, {"avail_elevated_pct": 50})["level"], "elevated")
        self.assertEqual(pressure.classify_pressure(s, {"avail_high_pct": 45, "avail_elevated_pct": 50})["level"], "high")

    def test_swap_raises_but_never_lowers(self):
        self.assertEqual(self.level(total=100, avail=60, swap_total=8, swap_pct=60), "elevated")
        self.assertEqual(self.level(total=100, avail=60, swap_total=8, swap_pct=85), "high")
        self.assertEqual(self.level(total=100, avail=5, swap_total=8, swap_pct=0), "critical")
        self.assertEqual(self.level(total=100, avail=60, swap_total=0.5, swap_pct=90), "ok")  # tiny swap is ignored

    def test_windows_commit_charge(self):
        self.assertEqual(self.level(total=100, avail=60, commit_pct=92), "elevated")
        self.assertEqual(self.level(total=100, avail=60, commit_pct=98), "high")
        self.assertEqual(self.level(total=100, avail=60, commit_pct=50), "ok")

    def test_linux_overcommit_is_not_pressure(self):
        self.assertEqual(self.level(total=100, avail=60, commit_pct=250, commit_src="/proc/meminfo"), "ok")

    def test_linux_memory_stall_time(self):
        self.assertEqual(self.level(total=100, avail=60, psi=12), "elevated")
        self.assertEqual(self.level(total=100, avail=60, psi=30), "high")
        self.assertEqual(self.level(total=100, avail=60, psi=80), "critical")
        self.assertEqual(self.level(total=100, avail=60, psi=1), "ok")

    def test_unknown_memory_is_not_called_safe(self):
        bad = {"ram": {"total_gb": unavailable("x"), "avail_gb": unavailable("x")}}
        r = pressure.classify_pressure(bad)
        self.assertFalse(r["known"])
        self.assertIsNone(r["avail_gb"])
        self.assertIn("could not be read", r["reasons"][0])
        for junk in ({}, None, {"ram": None}):
            self.assertFalse(pressure.classify_pressure(junk)["known"])

    def test_accepts_a_real_sample(self):
        r = pressure.classify_pressure(telemetry.system_sample(gpu=False))
        self.assertIn(r["level"], pressure.LEVELS)


def window(**kw):
    m = lambda v: measured(v, "sampler")
    u = unavailable("n/a")
    w = {"n": 10, "ram_avail_gb": {"min": u, "mean": u}, "swap_used_gb": {"max": u}, "cpu_util_pct": {"mean": u, "max": u},
         "gpu_util_pct": {"mean": u, "max": u}, "vram_used_mb": {"max": u}, "gpu_temp_c": {"max": u},
         "gpu_sm_clock_mhz": {"min": u, "max": u}, "rss_gb": {"max": u}, "major_fault_rate_per_s": u,
         "page_fault_rate_per_s": u, "io_read_mb_per_s": u}
    for k, v in kw.items():
        if isinstance(v, dict):
            w[k] = {kk: m(vv) for kk, vv in v.items()}
        else:
            w[k] = m(v)
    return w


def bench(gen_tps=None, **metrics):
    b = {"metrics": {k: measured(v, "bench") for k, v in metrics.items()}, "config": {"threads": 6, "n_gpu_layers": 0}, "warm": {}}
    if gen_tps is not None:
        b["warm"] = {"gen_tps": {"mean": measured(gen_tps, "bench"), "stdev": measured(0.1, "bench")}}
    return b


class DiagnoseTests(unittest.TestCase):
    def setUp(self):
        self.hw = hwmodel.preset_hardware("duck-user")
        self.plan120 = planner.plan(self.hw, GPT120)
        self.plan20 = planner.plan(self.hw, GPT20)

    def test_shape(self):
        d = pressure.diagnose_bottleneck()
        self.assertEqual(set(d), {"kind", "confidence", "explanation", "evidence", "suggestions", "basis"})
        self.assertEqual((d["kind"], d["confidence"], d["basis"]), ("unknown", "low", "none"))
        self.assertTrue(d["suggestions"])

    def test_major_faults_mean_paging(self):
        d = pressure.diagnose_bottleneck(self.plan120, window(major_fault_rate_per_s=400.0, io_read_mb_per_s=600.0), hw=self.hw)
        self.assertEqual((d["kind"], d["confidence"], d["basis"]), ("storage_paging", "high", "measured"))
        self.assertTrue(any("major page faults" in e for e in d["evidence"]))
        self.assertTrue(any("RAM" in s or "SSD" in s for s in d["suggestions"]))

    def test_rereading_the_model_is_paging_evidence(self):
        b = bench(gen_tps=0.8, io_read_gb=130.0)
        b["model"] = {"size_bytes": int(63.4e9)}
        d = pressure.diagnose_bottleneck(self.plan120, None, b, self.hw)
        self.assertEqual(d["kind"], "storage_paging")
        self.assertIn("2.2x the model size", " ".join(d["evidence"]))

    def test_bandwidth_bound_from_measured_speed(self):
        # gpt-oss-20b reads ~2.6 GB per token; 5 tokens/s = 13 GB/s of a 34 GB/s theoretical = 38%
        active = next(a["value"]["total"] for a in self.plan20["assumptions"] if a["key"] == "speed.active_gb_per_token")
        b = bench(gen_tps=0.45 * 34.1 / active, cpu_util_pct=100.0, host_bandwidth_gbs=6.0)
        d = pressure.diagnose_bottleneck(self.plan20, window(cpu_util_pct={"mean": 100.0, "max": 100.0}), b, self.hw)
        self.assertEqual(d["kind"], "memory_bandwidth")
        self.assertEqual(d["confidence"], "high")
        self.assertIn("GB/s of weights", " ".join(d["evidence"]))

    def test_cpu_saturated_far_below_the_bandwidth_ceiling_is_compute(self):
        active = next(a["value"]["total"] for a in self.plan20["assumptions"] if a["key"] == "speed.active_gb_per_token")
        b = bench(gen_tps=0.05 * 34.1 / active)
        d = pressure.diagnose_bottleneck(self.plan20, window(cpu_util_pct={"mean": 99.0, "max": 100.0}), b, self.hw)
        self.assertEqual(d["kind"], "compute")
        self.assertEqual(d["confidence"], "low")

    def test_too_few_threads(self):
        b = bench(gen_tps=1.0)
        b["config"]["threads"] = 2
        d = pressure.diagnose_bottleneck(None, window(cpu_util_pct={"mean": 30.0, "max": 40.0}), b, self.hw)
        self.assertEqual(d["kind"], "cpu_threads")
        self.assertIn("2 threads for 6 physical cores", " ".join(d["evidence"]))

    def test_full_vram(self):
        plan = planner.plan(self.hw, GPT20, {"n_gpu_layers": 24, "n_cpu_moe": 24})
        d = pressure.diagnose_bottleneck(plan, window(vram_used_mb={"max": 2040.0}), None, self.hw)
        self.assertEqual(d["kind"], "vram_capacity")
        self.assertEqual(d["confidence"], "medium")
        self.assertTrue(any("--no-op-offload" in s for s in d["suggestions"]))

    def test_vram_full_is_ignored_when_nothing_is_offloaded(self):
        plan = planner.plan(self.hw, GPT20, {"n_gpu_layers": 0})
        d = pressure.diagnose_bottleneck(plan, window(vram_used_mb={"max": 2040.0}), None, self.hw)
        self.assertNotEqual(d["kind"], "vram_capacity")

    def test_thermal(self):
        d = pressure.diagnose_bottleneck(None, window(gpu_temp_c={"max": 88.0}, gpu_sm_clock_mhz={"min": 600.0, "max": 1500.0}), None, self.hw)
        self.assertEqual((d["kind"], d["confidence"]), ("thermal", "high"))

    def test_idle_gpu_with_a_busy_cpu_points_at_transfers(self):
        plan = planner.plan(self.hw, GPT20, {"n_gpu_layers": 12})
        d = pressure.diagnose_bottleneck(plan, window(gpu_util_pct={"mean": 8.0, "max": 20.0}, cpu_util_pct={"mean": 95.0, "max": 100.0}), None, self.hw)
        self.assertEqual(d["kind"], "pcie_transfer")
        self.assertIn("cannot be measured", " ".join(d["evidence"]))

    def test_measured_signals_beat_the_estimate(self):
        # the plan says storage; the measurement says bandwidth
        active = next(a["value"]["total"] for a in self.plan120["assumptions"] if a["key"] == "speed.active_gb_per_token")
        b = bench(gen_tps=0.5 * 34.1 / active)
        d = pressure.diagnose_bottleneck(self.plan120, window(cpu_util_pct={"mean": 95.0, "max": 100.0}, major_fault_rate_per_s=0.0), b, self.hw)
        self.assertEqual(self.plan120["speed"]["bound_by"], "storage")
        self.assertEqual(d["kind"], "memory_bandwidth")
        self.assertEqual(d["basis"], "measured")

    def test_estimate_only_fallback_is_labelled_low_confidence(self):
        d = pressure.diagnose_bottleneck(self.plan120, None, None, self.hw)
        self.assertEqual((d["kind"], d["confidence"], d["basis"]), ("storage_paging", "low", "estimated"))
        self.assertIn("estimate", d["explanation"])
        self.assertIn("planner", d["explanation"])
        d = pressure.diagnose_bottleneck(self.plan20, None, None, self.hw)
        self.assertEqual((d["kind"], d["confidence"]), ("memory_bandwidth", "low"))

    def test_unavailable_measurements_are_not_evidence(self):
        d = pressure.diagnose_bottleneck(self.plan120, window(), None, self.hw)  # an all-unavailable window
        self.assertEqual(d["basis"], "estimated")
        self.assertEqual(d["confidence"], "low")

    def test_garbage_inputs_never_raise(self):
        for args in ((None, {"x": 1}, {"warm": 5, "metrics": []}, {}), ({}, {}, {}, {}), ("x", [], "y", 3)):
            try:
                d = pressure.diagnose_bottleneck(*args)
            except Exception as e:
                self.fail(f"raised {e!r} for {args!r}")
            self.assertIn(d["kind"], ("unknown", "storage_paging", "memory_bandwidth", "compute"))


class FakeSampler:
    def __init__(self):
        self.items = []

    def history(self, n=None):
        return list(self.items)

    def add(self, avail, total=100.0, t=None, **kw):
        self.items.append(sample(total=total, avail=avail, t=t if t is not None else 1000.0 + len(self.items), **kw))


class MonitorTests(unittest.TestCase):
    def make(self, settings=None, **kw):
        self.events = []
        self.sampler = FakeSampler()
        return pressure.Monitor(self.sampler, lambda: settings if settings is not None else {}, self.events.append, **kw)

    def test_debounce_up_needs_two_samples(self):
        m = self.make()
        self.sampler.add(60)
        self.sampler.add(20)
        self.assertEqual(m.poll(), [])
        self.assertEqual(m.level, "ok")
        self.sampler.add(20)
        ev = m.poll()
        self.assertEqual(len(ev), 1)
        self.assertEqual((ev[0]["level"], ev[0]["previous"], ev[0]["type"]), ("elevated", "ok", "pressure"))
        self.assertEqual(self.events, ev)
        self.assertEqual(m.level, "elevated")

    def test_a_blip_does_not_emit(self):
        m = self.make()
        for a in (60, 20, 60, 20, 60):
            self.sampler.add(a)
        m.poll()
        self.assertEqual(self.events, [])

    def test_debounce_down_needs_three_samples(self):
        m = self.make()
        for a in (20, 20, 60, 60):
            self.sampler.add(a)
        m.poll()
        self.assertEqual(m.level, "elevated")
        self.sampler.add(60)
        m.poll()
        self.assertEqual(m.level, "ok")
        self.assertEqual([e["level"] for e in self.events], ["elevated", "ok"])

    def test_pending_resets_when_the_level_flips_back(self):
        m = self.make()
        for a in (20, 60, 20, 60, 20):
            self.sampler.add(a)
        m.poll()
        self.assertEqual(self.events, [])

    def test_skipping_levels(self):
        m = self.make(confirm_up=1)
        self.sampler.add(3)
        m.poll()
        self.assertEqual((self.events[0]["level"], self.events[0]["previous"]), ("critical", "ok"))

    def test_action_follows_the_setting_only_at_critical(self):
        for setting, expected in (("warn", "warn"), ("stop_big_model", "stop_big_model"), ("stop_all", "stop_all"), ("bogus", "warn")):
            m = self.make({"vgpu_pressure_action": setting}, confirm_up=1)
            self.sampler.add(3)
            m.poll()
            self.assertEqual(self.events[-1]["action"], expected, setting)
        m = self.make({"vgpu_pressure_action": "stop_all"}, confirm_up=1)
        self.sampler.add(10)  # high
        m.poll()
        self.assertEqual((self.events[0]["level"], self.events[0]["action"]), ("high", "warn"))
        m = self.make({"vgpu_pressure_action": "stop_all"}, confirm_up=1)
        self.sampler.add(20)
        m.poll()
        self.assertEqual(self.events[0]["action"], "none")

    def test_default_action_is_warn(self):
        m = self.make({}, confirm_up=1)
        self.sampler.add(3)
        m.poll()
        self.assertEqual(self.events[0]["action"], "warn")

    def test_event_carries_the_sample_and_reasons(self):
        m = self.make(confirm_up=1)
        self.sampler.add(3)
        m.poll()
        ev = self.events[0]
        self.assertEqual(ev["sample"]["ram"]["avail_gb"]["value"], 3)
        self.assertTrue(ev["reasons"])
        self.assertIn("t", ev)

    def test_unreadable_samples_do_not_change_the_level(self):
        m = self.make(confirm_up=1)
        self.sampler.items.append({"t": 1.0, "ram": {"total_gb": unavailable("x"), "avail_gb": unavailable("x")}})
        m.poll()
        self.assertEqual(m.level, "ok")
        self.assertEqual(m.blind_samples, 1)
        self.assertEqual(self.events, [])

    def test_samples_are_processed_once(self):
        m = self.make(confirm_up=1)
        self.sampler.add(3)
        self.assertEqual(len(m.poll()), 1)
        self.assertEqual(m.poll(), [])
        self.assertEqual(len(self.events), 1)

    def test_callback_errors_are_contained(self):
        m = pressure.Monitor(FakeSampler(), dict, lambda e: 1 / 0, confirm_up=1)
        m.evaluate(sample(total=100, avail=3))
        self.assertEqual(m.callback_errors, 1)
        self.assertEqual(m.level, "critical")

    def test_a_broken_settings_function_falls_back_to_warn(self):
        m = pressure.Monitor(FakeSampler(), lambda: (_ for _ in ()).throw(RuntimeError()), None, confirm_up=1)
        ev = m.evaluate(sample(total=100, avail=3))
        self.assertEqual(ev["action"], "warn")

    def test_custom_thresholds_from_settings(self):
        m = self.make({"vgpu_pressure_thresholds": {"avail_elevated_pct": 80}}, confirm_up=1)
        self.sampler.add(60)
        m.poll()
        self.assertEqual(m.level, "elevated")

    def test_state_and_recent_events(self):
        m = self.make(confirm_up=1)
        self.sampler.add(3)
        m.poll()
        st = m.state()
        self.assertEqual((st["level"], st["action"], st["running"]), ("critical", "warn", False))
        self.assertEqual(len(m.recent_events()), 1)
        self.assertEqual(m.recent_events(0), [])

    def test_the_monitor_never_stops_anything_itself(self):
        # it only emits: there is no process handle anywhere on the object
        m = self.make(confirm_up=1)
        self.assertFalse(any(hasattr(m, n) for n in ("kill", "terminate", "server", "stop_server")))

    def test_thread_follows_a_live_sampler(self):
        values = iter([70, 70, 5, 5, 5, 5, 5, 5, 5, 5])
        n = [0]

        def fn():
            n[0] += 1
            return sample(total=100, avail=next(values, 5), t=time.time() + n[0] * 1e-3)

        sampler = telemetry.Sampler(0.02, sample_fn=fn)
        got = threading.Event()
        mon = pressure.Monitor(sampler, lambda: {"vgpu_pressure_action": "stop_big_model"},
                               lambda e: got.set() if e["level"] == "critical" else None, poll_s=0.02, confirm_up=2)
        sampler.start()
        self.assertTrue(mon.start())
        self.assertFalse(mon.start())
        try:
            self.assertTrue(got.wait(3.0))
        finally:
            mon.stop()
            sampler.stop()
        self.assertFalse(mon.state()["running"])
        self.assertEqual(mon.recent_events()[-1]["action"], "stop_big_model")


if __name__ == "__main__":
    unittest.main()
