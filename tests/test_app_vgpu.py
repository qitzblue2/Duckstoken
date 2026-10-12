"""The Virtual GPU endpoints through the real HTTP handler: telemetry, planner, compatibility, diagnostics, GPT-OSS
feasibility, storage tiers, benchmarks and the optimizer (with the realistic fake engine from tests/fake_vgpu_engine.py)."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_app_integration as ti  # noqa: E402
import vgpu_kit  # noqa: E402
from duckstoken import app as appmod  # noqa: E402

POSIX = unittest.skipIf(os.name == "nt", "uses a shebang launcher / sparse files")


class VGpuApiBase(ti.Api):
    def setUp(self):
        super().setUp()
        a = appmod.APP
        a.vgpu.jobs.cancel()
        a.vgpu.jobs.wait(10)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = vgpu_kit.make_model(self.tmp.name)

    def wait_job(self, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = self.call("vgpu/job")
            if j["state"] not in ("running",):
                return j
            time.sleep(0.15)
        self.fail("job did not finish: " + str(self.call("vgpu/job")))


class Live(VGpuApiBase):
    def test_telemetry_is_labelled_and_never_invents_gpu_numbers(self):
        t = self.call("vgpu/telemetry?n=5")
        self.assertIn(t["pressure"]["level"], ("ok", "elevated", "high", "critical"))
        ram = t["latest"]["ram"]
        self.assertIn(ram["total_gb"]["status"], ("measured", "unavailable"))
        for g in t["latest"]["gpus"]:  # only present if nvidia-smi exists on the machine running the tests
            self.assertIn(g["util_pct"]["status"], ("measured", "unavailable"))
        self.assertIn("watchdog", t)
        self.assertIn("prewarm", t)

    def test_presets_include_the_owners_pc(self):
        names = {p["name"]: p["label"] for p in self.call("vgpu/presets")["presets"]}
        self.assertIn("duck-user", names)
        self.assertIn("48 GB", names["duck-user"])
        self.assertIn("750", names["duck-user"])

    def test_capabilities_are_honest_about_missing_hooks(self):
        c = self.call("vgpu/capabilities")
        by = {x["id"]: x for x in c["capabilities"]}
        self.assertFalse(by["expert_routing_stats"]["available"])
        self.assertFalse(by["gpu_weight_prefetch"]["available"])
        self.assertTrue(by["layer_placement"]["available"])
        self.assertTrue(by["storage_prefetch"]["available"])
        self.assertFalse(by["dynamic_expert_residency"]["available"])
        self.assertFalse(by["transfer_bandwidth_probe"]["available"])
        self.assertIn("strategy", c)
        self.assertTrue(c["strategy"]["text"])


class Planner(VGpuApiBase):
    def test_plan_for_a_real_gguf_with_whatif_hardware(self):
        r = self.call("vgpu/plan", {"model_path": self.model, "preset": "duck-user", "ctx_size": 4096})
        self.assertEqual(r["what_if"], "duck-user")
        self.assertTrue(r["hardware"]["mock"])
        plan = r["plan"]
        self.assertEqual(plan["status"], "estimated")
        self.assertIn(plan["admission"]["verdict"], ("ok", "tight", "paging", "reject"))
        self.assertEqual(plan["config"]["ctx_size"], 4096)
        self.assertEqual(plan["tiers"]["vram"]["physical_gb"]["value"], 2.0)
        self.assertTrue(plan["assumptions"])
        self.assertIn("kind", r["bottleneck"])

    def test_custom_hardware_overrides_are_validated(self):
        r = self.call("vgpu/plan", {"model_path": self.model, "hardware": {"ram_gb": 8, "vram_mb": 0}})
        self.assertEqual(r["hardware"]["ram_gb"], 8)
        self.assertEqual(r["hardware"]["vram_mb"], 0)
        self.assertEqual(self.call("vgpu/plan", {"model_path": self.model, "hardware": {"ram_gb": "lots"}})["_status"], 400)
        self.assertEqual(self.call("vgpu/plan", {"model_path": self.model, "hardware": {"evil": 1}})["_status"], 400)
        self.assertEqual(self.call("vgpu/plan", {"model_path": self.model, "preset": "nope"})["_status"], 400)

    def test_bad_model_paths_are_rejected_before_any_work(self):
        txt = Path(self.tmp.name) / "notes.gguf"
        txt.write_bytes(b"not a model")
        for bad in ("", "/does/not/exist.gguf", str(Path(self.tmp.name)), str(txt), "../../etc/passwd", "x\x00.gguf"):
            r = self.call("vgpu/plan", {"model_path": bad})
            self.assertEqual(r["_status"], 400, bad)
        other = Path(self.tmp.name) / "a.bin"
        other.write_bytes(b"GGUF....")
        self.assertEqual(self.call("vgpu/plan", {"model_path": str(other)})["_status"], 400)  # wrong suffix

    def test_quant_comparison(self):
        r = self.call("vgpu/quants?model_path=" + ti.urllib.parse.quote(self.model))
        self.assertTrue(r["rows"])
        row = r["rows"][0]
        self.assertIn("quality_note", row)
        self.assertNotIn("quality_score", row)  # no invented quality numbers


class GptOss(VGpuApiBase):
    def test_owners_pc_120b_is_paging_not_comfortable(self):
        r = self.call("vgpu/gptoss", {"model": "gpt-oss-120b", "preset": "duck-user", "ctx": 4096})
        self.assertEqual(r["what_if"], "duck-user")
        self.assertIn(r["verdict"], ("paging", "tight", "not_realistic"))
        self.assertNotEqual(r["verdict"], "comfortable")
        self.assertTrue(r["never_auto_download"])
        self.assertTrue(r["download"]["needs_confirmation"])
        self.assertEqual(len(r["ram_table"]), 7)
        self.assertTrue(any(c["id"] == "architecture" for c in r["checks"]))

    def test_small_machines_are_told_the_truth(self):
        r = self.call("vgpu/gptoss", {"model": "gpt-oss-120b", "hardware": {"ram_gb": 16, "vram_mb": 0}})
        self.assertEqual(r["verdict"], "not_realistic")
        self.assertEqual(r["recommendation"]["action"], "use_20b")

    def test_big_machine_is_comfortable(self):
        r = self.call("vgpu/gptoss", {"model": "gpt-oss-120b", "hardware": {"ram_gb": 128, "vram_mb": 24576, "ram_speed_mts": 5600}})
        self.assertIn(r["verdict"], ("comfortable", "ok"))

    def test_20b_and_validation(self):
        self.assertIn(self.call("vgpu/gptoss", {"model": "gpt-oss-20b", "preset": "duck-user"})["verdict"], ("comfortable", "ok", "tight"))
        self.assertEqual(self.call("vgpu/gptoss", {"model": "llama"})["_status"], 400)
        self.assertEqual(self.call("vgpu/gptoss", {"model": "gpt-oss-20b", "ctx": "big"})["_status"], 400)

    def test_a_wrong_file_is_reported_by_the_checks(self):
        r = self.call("vgpu/gptoss", {"model": "gpt-oss-120b", "preset": "duck-user", "file_path": self.model})
        checks = {c["id"]: c["status"] for c in r["checks"]}
        self.assertEqual(checks["architecture"], "fail")


class Diagnostics(VGpuApiBase):
    def test_report_redacts_secrets_and_lists_compat(self):
        appmod.APP.settings.update({"hf_token": "SECRET-TOKEN-XYZ", "external_api_key": "KEY-12345-ABC", "model_path": self.model})
        d = self.call("vgpu/diagnostics")
        self.assertNotIn("SECRET-TOKEN-XYZ", d["text"])
        self.assertNotIn("KEY-12345-ABC", d["text"])
        self.assertNotIn("SECRET-TOKEN-XYZ", str(d["report"]))
        ids = {i["id"] for i in d["compat"]["items"]}
        self.assertTrue({"cpu", "cuda", "vulkan", "gpt_oss", "mxfp4"} <= ids)
        for i in d["compat"]["items"]:
            self.assertIn(i["status"], ("supported", "unsupported", "experimental", "unknown"))
        self.assertIn("status", d["probe"])

    def test_report_has_a_timestamp_and_notes(self):
        d = self.call("vgpu/diagnostics")
        self.assertIn("generated_at", d["report"])
        self.assertTrue(d["report"]["notes"])

    def test_logs(self):
        for which in ("server", "big", "perf", "app"):
            self.assertIn("lines", self.call("vgpu/logs?which=" + which))
        self.assertEqual(self.call("vgpu/logs?which=../../etc/passwd")["_status"], 400)


class Storage(VGpuApiBase):
    def test_staging_is_off_by_default_and_needs_confirmation(self):
        c = self.call("vgpu/cache")
        self.assertFalse(c["staging"]["enabled"])
        self.assertEqual(self.call("vgpu/cache/stage", {"model_path": self.model})["_status"], 400)           # no confirm
        self.assertEqual(self.call("vgpu/cache/stage", {"model_path": self.model, "confirm": True})["_status"], 400)  # cache off
        self.assertEqual(self.call("vgpu/cache/evict", {"name": "x"})["_status"], 400)

    def test_prewarm_reads_the_file_and_reports(self):
        r = self.call("vgpu/prewarm/start", {"model_path": self.model})
        self.assertIn(r["prewarm"]["state"], ("running", "done", "stopped"))
        for _ in range(100):
            st = self.call("vgpu/cache")["prewarm"]
            if st["state"] != "running":
                break
            time.sleep(0.05)
        self.assertIn(st["state"], ("done", "stopped"))
        self.assertIn("message", st)

    def test_staging_copy_roundtrip(self):
        cache = Path(self.tmp.name) / "cache"
        cache.mkdir()
        appmod.APP.settings.update({"vgpu_cache_dir": str(cache), "vgpu_cache_gb": 5.0, "vgpu_cache_min_free_gb": 1.0, "model_path": self.model})
        c = self.call("vgpu/cache")
        self.assertTrue(c["staging"]["enabled"])
        self.assertIsNotNone(c["stage_plan"])
        self.assertEqual(self.call("vgpu/cache/evict", {"name": "nope", "confirm": True})["_status"], 409)

    def test_metadata_cache_counts_hits(self):
        self.call("vgpu/plan", {"model_path": self.model})
        self.call("vgpu/plan", {"model_path": self.model})
        st = self.call("vgpu/cache")["bounded"]["gguf_header"]
        self.assertGreaterEqual(st["hits"], 1)


class Jobs(VGpuApiBase):
    def fake_engine(self, **extra):
        s = vgpu_kit.fake_settings(self.tmp.name)
        patch = {k: s[k] for k in ("llama_server_path", "ctx_size", "n_gpu_layers", "kv_offload", "extra_args")}
        patch.update(model_path=self.model, vgpu_bench_prompt_tokens=16, vgpu_bench_gen_tokens=8, vgpu_bench_repeats=1, port=ti.free_port())
        patch.update(extra)
        appmod.APP.settings.update(patch)

    def test_requires_confirmation_and_idle_server(self):
        self.fake_engine()
        self.assertEqual(self.call("vgpu/bench/start", {"model_path": self.model})["_status"], 400)
        self.assertEqual(self.call("vgpu/opt/start", {"model_path": self.model})["_status"], 400)
        self.assertEqual(self.call("vgpu/opt/start", {"model_path": self.model, "objective": "fastest"})["_status"], 400)

    @POSIX
    def test_benchmark_end_to_end_with_the_fake_engine(self):
        self.fake_engine()
        j = self.call("vgpu/bench/start", {"model_path": self.model, "confirm": True, "label": "t"})
        self.assertIn(j["state"], ("running", "done"))
        j = self.wait_job()
        self.assertEqual(j["state"], "done", j)
        rec = j["result"]
        self.assertEqual(rec["status"], "ok", rec.get("error"))
        self.assertIn(rec["warm"]["runs"][0]["source"], ("server_timings", "client_side"))
        self.assertEqual(rec["metrics"]["transfer_bandwidth_gbs"]["status"], "unavailable")  # llama-server cannot report this
        lst = self.call("vgpu/bench/list")["benchmarks"]
        self.assertEqual(lst[0]["id"], rec["id"])
        self.assertNotIn("log_tail", lst[0])
        self.assertEqual(self.call("vgpu/bench/get?id=" + rec["id"])["benchmark"]["id"], rec["id"])
        self.assertEqual(self.call("vgpu/bench/delete", {"id": rec["id"]})["_status"], 400)  # needs confirm
        self.assertTrue(self.call("vgpu/bench/delete", {"id": rec["id"], "confirm": True})["deleted"])
        self.assertEqual(self.call("vgpu/bench/get?id=" + rec["id"])["_status"], 404)
        self.assertFalse(appmod.APP.server.status()["running"])  # the benchmark's own server is gone

    @POSIX
    def test_benchmark_refuses_while_the_chat_server_runs_unless_told_to_stop_it(self):
        self.fake_engine()
        self.call("server/start", {})
        for _ in range(100):
            if appmod.APP.server.status()["state"] == "ready":
                break
            time.sleep(0.1)
        r = self.call("vgpu/bench/start", {"model_path": self.model, "confirm": True})
        self.assertEqual(r["_status"], 409)
        self.assertTrue(r.get("needs_stop"))
        r = self.call("vgpu/bench/start", {"model_path": self.model, "confirm": True, "stop_server": True})
        self.assertIn(r["state"], ("running", "done"))
        self.wait_job()

    @POSIX
    def test_optimizer_run_picks_a_measured_winner_and_apply_is_confirmed_and_reversible(self):
        self.fake_engine()
        cands = self.call("vgpu/opt/candidates", {"model_path": self.model, "objective": "balanced"})
        self.assertEqual(cands["candidates"][0]["id"], "cpu-baseline")
        self.assertTrue(cands["candidates"][0]["admitted"])
        self.assertIsNotNone(cands["note"])  # no GPU device with the fake engine -> only CPU variants
        self.call("vgpu/opt/start", {"model_path": self.model, "objective": "balanced", "confirm": True})
        j = self.wait_job(120)
        self.assertEqual(j["state"], "done", j)
        res = j["result"]
        self.assertEqual(res["status"], "ok", res.get("error"))
        self.assertEqual(res["best"]["basis"], "measured")
        self.assertFalse(res["applied"])
        key = res["best"]["profile_key"]
        before = appmod.APP.settings.all()
        self.assertEqual(self.call("vgpu/opt/apply", {"key": key})["_status"], 400)  # no confirm
        self.assertEqual(appmod.APP.settings.all(), before)
        prof = self.call("vgpu/profile?model_path=" + ti.urllib.parse.quote(self.model))
        self.assertIn(prof["status"], ("fresh", "stale"))
        applied = self.call("vgpu/opt/apply", {"key": key, "confirm": True})
        self.assertIn("applied", applied)
        for k, v in applied["applied"].items():
            self.assertEqual(appmod.APP.settings.get(k), v)
        if applied["revertible"]:
            self.assertTrue(self.call("vgpu/opt/revert", {"key": key})["reverted"])
        h = self.call("vgpu/opt/history")
        self.assertTrue(h["history"])

    def test_estimate_only_runs_nothing_and_cannot_be_applied(self):
        self.fake_engine()
        self.call("vgpu/opt/start", {"model_path": self.model, "objective": "max_speed", "estimate_only": True})
        j = self.wait_job()
        self.assertEqual(j["state"], "done", j)
        self.assertEqual(j["result"]["status"], "estimate_only")
        self.assertEqual(j["result"]["best"]["basis"], "estimated")
        self.assertFalse(appmod.APP.server.status()["running"])

    def test_unknown_objective(self):
        self.assertEqual(self.call("vgpu/opt/candidates", {"model_path": self.model, "objective": "x"})["_status"], 400)

    def test_only_one_job_at_a_time(self):
        import threading
        self.fake_engine()
        release = threading.Event()
        appmod.APP.vgpu.jobs.start("block", lambda cancel, progress: release.wait(10))
        try:
            r = self.call("vgpu/bench/start", {"model_path": self.model, "confirm": True})
            self.assertEqual(r["_status"], 409)
            r = self.call("vgpu/opt/start", {"model_path": self.model, "confirm": True})
            self.assertEqual(r["_status"], 409)
            self.assertEqual(self.call("vgpu/job")["state"], "running")
            self.assertIn(self.call("vgpu/job/cancel", {})["state"], ("running", "cancelled"))
        finally:
            release.set()
            appmod.APP.vgpu.jobs.wait(10)


class PrewarmThenStart(VGpuApiBase):
    def test_a_second_start_while_preparing_is_refused_and_stop_really_prevents_the_start(self):
        # review finding: the second Start orphaned the first cancel Event, so Stop did not stop the server from starting later
        import threading
        from unittest import mock
        from duckstoken.vgpu import runtime as rt_mod
        release, started = threading.Event(), []

        def slow_prewarm(path, max_bytes, cancel=None, progress=None, **kw):
            while not cancel.is_set() and not release.wait(0.05):
                pass
            return {"completed": False, "bytes_read": 0, "stopped_because": "cancelled", "mb_per_s": {"value": None}}

        s = vgpu_kit.fake_settings(self.tmp.name)
        appmod.APP.settings.update({"llama_server_path": s["llama_server_path"], "model_path": self.model, "port": ti.free_port(),
                                    "vgpu_prewarm": "before_start"})
        rt = appmod.APP.vgpu
        with mock.patch.object(rt_mod, "run_prewarm", slow_prewarm), \
                mock.patch.object(appmod.APP.server, "start", side_effect=lambda cfg: started.append(1)):
            rt.start_server(appmod.APP.settings.all())
            with self.assertRaises(appmod.ApiError) as cm:
                rt.start_server(appmod.APP.settings.all())
            self.assertEqual(cm.exception.status, 409)
            rt.stop_server()                       # cancels the prewarm that is in flight
            rt._prewarm_thread.join(5)
            self.assertFalse(rt._prewarm_thread.is_alive())
            release.set()
        self.assertEqual(started, [], "the server must not start after an explicit Stop")


class AdmissionGate(VGpuApiBase):
    @POSIX
    def test_impossible_model_is_refused_before_anything_starts_but_can_be_forced(self):
        huge = Path(self.tmp.name) / "huge.gguf"
        vgpu_kit.make_model(self.tmp.name, "huge.gguf")
        with open(huge, "r+b") as f:
            f.truncate(900 * 2**30)  # sparse: 900 GB on a machine with a few GB of RAM
        s = vgpu_kit.fake_settings(self.tmp.name)
        appmod.APP.settings.update({"llama_server_path": s["llama_server_path"], "model_path": str(huge), "port": ti.free_port(), "n_gpu_layers": 0})
        r = self.call("server/start", {})
        self.assertEqual(r["_status"], 409, r)
        self.assertFalse(r["admission"]["can_load"])
        self.assertTrue(r["admission"]["blockers"])
        self.assertFalse(appmod.APP.server.status()["running"])
        kinds = [f["kind"] for f in self.call("vgpu/diagnostics")["failures"]]
        self.assertIn("admission_rejected", kinds)  # remembered for the diagnostic report
        # 'warn' mode lets it through (the fake engine ignores the size)
        appmod.APP.settings.update({"vgpu_admission": "warn"})
        r = self.call("server/start", {})
        self.assertNotIn("_status", r)
        self.call("server/stop", {})

    @POSIX
    def test_gpu_offload_bigger_than_vram_is_fitted_for_that_start_only(self):
        # review finding: the shipped "gpt-oss-120b, small GPU" preset (99 GPU layers) was refused on a 2 GB GTX 750
        from unittest import mock
        from duckstoken import hardware
        from duckstoken.vgpu import hwmodel
        kv = {"general.architecture": "gpt-oss", "general.name": "gpt-oss-120b", "gpt-oss.block_count": 36, "gpt-oss.expert_count": 128,
              "gpt-oss.expert_used_count": 4, "gpt-oss.embedding_length": 2880, "gpt-oss.attention.head_count": 64,
              "gpt-oss.attention.head_count_kv": 8, "gpt-oss.context_length": 131072, "gpt-oss.expert_feed_forward_length": 2880,
              "gpt-oss.attention.sliding_window": 128, "general.file_type": 38}
        big = vgpu_kit.make_model(self.tmp.name, "gpt-oss-120b-MXFP4.gguf", kv)
        with open(big, "r+b") as f:
            f.truncate(int(63.4e9))
        s = vgpu_kit.fake_settings(self.tmp.name)
        appmod.APP.settings.update({"llama_server_path": s["llama_server_path"], "model_path": big, "n_gpu_layers": 99, "n_cpu_moe": 36,
                                    "ctx_size": 4096, "port": ti.free_port()})
        with mock.patch.object(hardware, "get_hardware", return_value=hwmodel.preset_hardware("duck-user")):
            plan = appmod.APP.vgpu.admit(appmod.APP.settings.all())
            self.assertTrue(plan["admission"]["can_load"], plan["admission"])
            fit = plan["autofit"]
            self.assertLess(fit["overrides"]["n_gpu_layers"], 99)
            self.assertEqual(fit["was"]["n_gpu_layers"], 99)
            # the explicit "do not touch my settings" switches
            appmod.APP.settings.update({"vgpu_autofit": False})
            with self.assertRaises(appmod.ApiError) as cm:
                appmod.APP.vgpu.admit(appmod.APP.settings.all())
            self.assertEqual(cm.exception.status, 409)
        self.assertEqual(appmod.APP.settings.get("n_gpu_layers"), 99)  # the saved settings were never changed

    def test_gate_can_be_turned_off_and_small_models_pass(self):
        s = vgpu_kit.fake_settings(self.tmp.name)
        appmod.APP.settings.update({"llama_server_path": s["llama_server_path"], "model_path": self.model, "port": ti.free_port()})
        self.assertIsNotNone(appmod.APP.vgpu.admit(appmod.APP.settings.all()))
        appmod.APP.settings.update({"vgpu_admission": "off"})
        self.assertIsNone(appmod.APP.vgpu.admit(appmod.APP.settings.all()))


if __name__ == "__main__":
    unittest.main()
