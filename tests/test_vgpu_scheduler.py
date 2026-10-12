"""Scheduler: honest capability registry, candidate configs (baseline first, planner-checked), transfer checks."""
import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vgpu_kit as K  # noqa: E402

from duckstoken import settings as settings_mod  # noqa: E402
from duckstoken.vgpu import hwmodel  # noqa: E402
from duckstoken.vgpu import scheduler as S  # noqa: E402

PROBE_CUDA = {"devices": [{"id": "CUDA0", "backend": "CUDA", "name": "Test GPU", "total_mb": 4096, "free_mb": 4000}],
              "backends_usable": ["CUDA"], "status": "ok"}
PROBE_CPU = {"devices": [], "backends_usable": [], "status": "ok"}

DENSE = {"path": "/m/dense.gguf", "remote": True, "name": "dense-7b", "architecture": "llama", "size_bytes": int(4.4e9),
         "size_gb": 4.1, "block_count": 32, "context_length": 8192, "embedding_length": 4096, "head_count": 32,
         "head_count_kv": 8, "expert_count": 0, "expert_used_count": 0, "is_moe": False, "file_type": 15}
MOE = {**DENSE, "path": "/m/moe.gguf", "name": "gpt-oss-20b", "architecture": "gpt-oss", "size_bytes": int(12.1e9),
       "size_gb": 11.27, "block_count": 24, "embedding_length": 2880, "head_count": 64, "expert_count": 32,
       "expert_used_count": 4, "is_moe": True, "file_type": 38}
GPT120 = {**MOE, "name": "gpt-oss-120b", "size_bytes": int(63.4e9), "size_gb": 59.0, "block_count": 36, "expert_count": 128}


def base_settings(**kw):
    s = dict(settings_mod.DEFAULTS)
    s.update(ctx_size=4096, cache_type_k="f16", cache_type_v="f16")
    s.update(kw)
    return s


class TestCapabilities(unittest.TestCase):
    def test_registry_is_complete_and_honest(self):
        ids = [c["id"] for c in S.CAPABILITIES]
        self.assertEqual(len(ids), len(set(ids)))
        for c in S.CAPABILITIES:
            self.assertEqual(set(c), {"id", "label", "available", "how", "limitation"})
            self.assertIsInstance(c["available"], bool)
            self.assertTrue(c["limitation"])
            if c["available"]:
                self.assertTrue(c["how"], c["id"])
        no = {c["id"] for c in S.CAPABILITIES if not c["available"]}
        self.assertEqual(no, {"expert_routing_stats", "dynamic_expert_residency", "gpu_weight_prefetch",
                              "transfer_bandwidth_probe"})
        yes = {c["id"] for c in S.CAPABILITIES if c["available"]}
        for needed in ("layer_placement", "moe_expert_placement", "kv_placement", "mmap_mlock", "op_offload_toggle",
                       "storage_prefetch"):
            self.assertIn(needed, yes)

    def test_lookup(self):
        self.assertFalse(S.capability("gpu_weight_prefetch")["available"])
        self.assertIsNone(S.capability("nope"))
        S.capability("layer_placement")["available"] = False      # a copy: the registry is not mutated
        self.assertTrue(S.capability("layer_placement")["available"])

    def test_module_documents_what_replaces_the_missing_hooks(self):
        doc = S.__doc__
        for word in ("do NOT exist", "static placement", "sequential prefetch", "--n-cpu-moe"):
            self.assertIn(word, doc)


class TestGpuGate(unittest.TestCase):
    def test_cases(self):
        self.assertFalse(S.gpu_gate(K.mock_hw(), PROBE_CUDA)["usable"])                      # no GPU at all
        self.assertIn("no GPU", S.gpu_gate(K.mock_hw(), PROBE_CUDA)["reason"])
        self.assertFalse(S.gpu_gate(K.mock_hw(vram_mb=512), PROBE_CUDA)["usable"])           # under 1 GB
        g = S.gpu_gate(K.mock_hw(vram_mb=4096), None)
        self.assertFalse(g["usable"])
        self.assertIn("not been probed", g["reason"])
        c = S.gpu_gate(K.mock_hw(vram_mb=4096), PROBE_CPU)
        self.assertFalse(c["usable"])
        self.assertIn("CPU-only build", c["reason"])
        ok = S.gpu_gate(K.mock_hw(vram_mb=4096), PROBE_CUDA)
        self.assertTrue(ok["usable"])
        self.assertEqual(ok["backends"], ["CUDA"])
        only_devices = {"devices": [{"backend": "Vulkan"}, {"backend": "CPU"}]}
        self.assertTrue(S.gpu_gate(K.mock_hw(vram_mb=4096), only_devices)["usable"])
        self.assertFalse(S.gpu_gate(K.mock_hw(vram_mb=4096), {"devices": [{"backend": "CPU"}]})["usable"])


class TestCandidates(unittest.TestCase):
    def setUp(self):
        self.planner = K.FakePlanner()
        p = mock.patch.object(S, "_mod", lambda name: self.planner if name == "planner" else None)
        p.start()
        self.addCleanup(p.stop)
        self.hw_gpu = K.mock_hw(vram_mb=4096)
        self.hw_cpu = K.mock_hw()

    def cands(self, model=DENSE, hw=None, objective="max_speed", probe=PROBE_CUDA, **kw):
        return S.candidate_configs(hw or self.hw_gpu, model, kw.pop("base", base_settings()), objective, probe=probe, **kw)

    def ids(self, cands):
        return [c["id"] for c in cands]

    def test_baseline_is_first_for_every_objective_and_is_cpu_only(self):
        for obj in ("max_speed", "max_model_size", "balanced", "unknown-objective"):
            c = self.cands(objective=obj)[0]
            self.assertEqual(c["id"], "cpu-baseline")
            self.assertEqual(c["overrides"]["n_gpu_layers"], 0)
            self.assertIs(c["overrides"]["kv_offload"], False)
            self.assertTrue(c["admitted"])

    def test_item_shape(self):
        for c in self.cands():
            self.assertEqual({"id", "label", "overrides", "plan", "admitted", "reject_reason", "expected", "group"}, set(c))
            self.assertIn("estimate", c["expected"])
            self.assertEqual(c["plan"]["admission"]["can_load"], c["admitted"])
        self.assertEqual(len({c["id"] for c in self.cands()}), len(self.cands()))

    def test_overrides_are_valid_settings_patches(self):
        for model in (DENSE, MOE):
            for c in self.cands(model=model):
                clean = settings_mod.validate_patch(c["overrides"])       # raises on unknown keys / bad values
                self.assertEqual(set(clean), set(c["overrides"]))

    def test_no_gpu_candidates_without_evidence_of_a_usable_device(self):
        for hw, probe in ((self.hw_cpu, PROBE_CUDA), (self.hw_gpu, None), (self.hw_gpu, PROBE_CPU)):
            ids = self.ids(self.cands(hw=hw, probe=probe))
            self.assertEqual(ids[0], "cpu-baseline")
            for bad in ("gpu-", "moe-", "op-offload"):
                self.assertFalse([i for i in ids if i.startswith(bad)], (bad, ids))
            self.assertTrue(all(c["overrides"].get("n_gpu_layers", 0) == 0 for c in self.cands(hw=hw, probe=probe)))

    def test_dense_gpu_steps_up_to_the_planners_maximum(self):
        c = self.cands()
        ids = self.ids(c)
        self.assertEqual([i for i in ids if i.startswith("gpu-") and "kv" not in i], ["gpu-4", "gpu-8", "gpu-12"])
        by = {x["id"]: x for x in c}
        self.assertEqual(by["gpu-12"]["overrides"], {"kv_offload": True, "n_gpu_layers": 12})
        self.assertIn("gpu-14-kv-host", by)                       # planner fits 2 more layers if the KV stays in RAM
        self.assertIs(by["gpu-14-kv-host"]["overrides"]["kv_offload"], False)
        self.assertLess(ids.index("cpu-baseline"), ids.index("gpu-4"))

    def test_moe_variants_keep_all_layers_on_gpu_and_vary_experts(self):
        c = self.cands(model=MOE)
        moe = [x for x in c if x["id"].startswith("moe-cpu-")]
        self.assertEqual([x["overrides"]["n_cpu_moe"] for x in moe], [24, 22, 20])
        self.assertTrue(all(x["overrides"]["n_gpu_layers"] == 25 for x in moe))
        self.assertEqual(self.ids(c)[0], "cpu-baseline")

    def test_moe_when_even_shared_weights_do_not_fit(self):
        self.planner.min_moe = None
        c = self.cands(model=MOE)
        self.assertIn("gpu-12-experts-cpu", self.ids(c))

    def test_op_offload_candidate_uses_the_documented_flag_and_keeps_user_args(self):
        c = {x["id"]: x for x in self.cands(base=base_settings(extra_args="--no-warmup"))}
        self.assertEqual(c["op-offload-off"]["overrides"]["extra_args"], "--no-warmup --no-op-offload")
        self.assertEqual(S.with_extra_flag("--no-op-offload", "--no-op-offload"), "--no-op-offload")
        self.assertEqual(S.with_extra_flag("", "--x"), "--x")

    def test_kv_and_ubatch_modifiers_flip_away_from_the_current_value(self):
        c = {x["id"]: x for x in self.cands(base=base_settings(cache_type_k="q8_0", cache_type_v="q8_0", ubatch_size=256))}
        self.assertIn("kv-f16", c)
        self.assertIn("ubatch-512", c)
        self.assertNotIn("kv-q8_0", c)
        d = {x["id"]: x for x in self.cands(base=base_settings(ubatch_size=512))}
        self.assertIn("kv-q8_0", d)
        self.assertEqual(d["kv-q8_0"]["overrides"]["flash_attn"], "on")        # a quantized V cache needs flash attention
        self.assertIn("ubatch-256", d)

    def test_thread_variants_depend_on_hyperthreading_and_the_current_setting(self):
        ids = self.ids(self.cands(hw=K.mock_hw(cores=4, logical=8), probe=None))
        self.assertIn("threads-phys-logical", ids)
        self.assertIn("threads-logical", ids)
        self.assertNotIn("threads-physical", ids)                              # threads=0 already means "physical"
        no_ht = self.ids(self.cands(hw=K.mock_hw(cores=4, logical=4), probe=None))
        self.assertFalse([i for i in no_ht if i.startswith("threads")])
        explicit = self.ids(self.cands(hw=K.mock_hw(cores=4, logical=8), probe=None, base=base_settings(threads=8, threads_batch=8)))
        self.assertIn("threads-physical", explicit)
        self.assertNotIn("threads-logical", explicit)

    def test_bounded_and_deduplicated(self):
        c = self.cands(max_candidates=3)
        self.assertEqual(len(c), 3)
        self.assertEqual(c[0]["id"], "cpu-baseline")
        self.assertEqual(len(self.cands(max_candidates=0)), 1)                  # never empty: the baseline stays
        full = self.cands(max_candidates=50)
        keys = [S._config_key(S._merged(base_settings(), x["overrides"])) for x in full]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertLessEqual(len(self.cands()), S.MAX_CANDIDATES)

    def test_impossible_candidates_are_returned_flagged_and_never_admitted(self):
        self.planner.reject = lambda cfg: ["Not enough VRAM for 8+ layers."] if (cfg.get("n_gpu_layers") or 0) >= 8 else None
        c = {x["id"]: x for x in self.cands()}
        self.assertTrue(c["gpu-4"]["admitted"])
        for cid in ("gpu-8", "gpu-12"):
            self.assertFalse(c[cid]["admitted"])
            self.assertIn("Not enough VRAM", c[cid]["reject_reason"])
            self.assertEqual(c[cid]["plan"]["admission"]["verdict"], "reject")

    def test_baseline_rejected_means_baseline_not_admitted(self):
        self.planner.reject = lambda cfg: ["Model is larger than RAM and disk."]
        c = self.cands()
        self.assertEqual(c[0]["id"], "cpu-baseline")
        self.assertFalse(any(x["admitted"] for x in c))

    def test_paging_candidates_need_include_unsafe_but_the_baseline_is_exempt(self):
        self.planner.verdict_fn = lambda cfg: "paging"
        c = self.cands()
        self.assertTrue(c[0]["admitted"])
        self.assertTrue(all(not x["admitted"] for x in c[1:]))
        self.assertIn("page about", c[1]["reject_reason"])
        self.assertTrue(all(x["admitted"] for x in self.cands(include_unsafe=True)))

    def test_tight_is_admitted(self):
        self.planner.verdict = "tight"
        self.assertTrue(all(x["admitted"] for x in self.cands()))

    def test_max_model_size_puts_memory_savers_before_offload_experiments(self):
        speed = self.ids(self.cands(objective="max_speed"))
        size = self.ids(self.cands(objective="max_model_size"))
        self.assertLess(size.index("kv-q8_0"), size.index("gpu-4"))
        self.assertLess(speed.index("gpu-4"), speed.index("kv-q8_0"))
        by = {x["id"]: x for x in self.cands(objective="max_model_size")}
        self.assertEqual(by["kv-q8_0"]["overrides"]["n_gpu_layers"], 0)         # savers are tried on the CPU config

    def test_base_settings_are_not_mutated(self):
        base = base_settings()
        before = copy.deepcopy(base)
        S.candidate_configs(self.hw_gpu, DENSE, base, "max_speed", probe=PROBE_CUDA)
        self.assertEqual(base, before)

    def test_planner_receives_only_planner_keys_with_overrides_applied(self):
        self.cands()
        cfgs = [c for c in self.planner.calls]
        self.assertTrue(cfgs)
        for cfg in cfgs:
            self.assertLessEqual(set(cfg), set(S._plan_cfg(settings_mod.DEFAULTS)) | {"swa_full", "parallel"})
        self.assertEqual(cfgs[0]["n_gpu_layers"], 0)

    def test_missing_planner_raises_clearly(self):
        with mock.patch.object(S, "_mod", lambda name: None):
            with self.assertRaises(RuntimeError):
                S.candidate_configs(self.hw_cpu, DENSE, base_settings(), "max_speed")


class TestCandidatesWithTheRealPlanner(unittest.TestCase):
    def setUp(self):
        self.hw = hwmodel.preset_hardware("duck-user")     # i7-8700K, 48 GB DDR4-2133, GTX 750 2 GB, NVMe

    def test_owners_pc_gpt_oss_20b_gets_a_baseline_and_moe_variants(self):
        c = S.candidate_configs(self.hw, MOE, base_settings(), "max_speed", probe=PROBE_CUDA)
        self.assertEqual(c[0]["id"], "cpu-baseline")
        self.assertTrue(c[0]["admitted"], c[0]["reject_reason"])
        moe = [x for x in c if x["id"].startswith("moe-cpu-")]
        self.assertTrue(moe)
        self.assertTrue(all(x["admitted"] for x in moe), [(x["id"], x["reject_reason"]) for x in moe])
        for x in c:   # a candidate is only ever runnable if the planner can load it
            if x["admitted"]:
                self.assertTrue(x["plan"]["admission"]["can_load"], x["id"])
            if not x["plan"]["admission"]["can_load"]:
                self.assertFalse(x["admitted"], x["id"])
                self.assertTrue(x["reject_reason"])

    def test_owners_pc_gpt_oss_120b_pages_so_only_the_baseline_runs_by_default(self):
        c = S.candidate_configs(self.hw, GPT120, base_settings(), "max_speed", probe=PROBE_CUDA)
        self.assertEqual(c[0]["id"], "cpu-baseline")
        self.assertEqual(c[0]["plan"]["admission"]["verdict"], "paging")
        self.assertTrue(c[0]["admitted"])
        self.assertTrue(all(not x["admitted"] for x in c[1:]))
        self.assertTrue(all("page" in x["reject_reason"] for x in c[1:]))
        unsafe = S.candidate_configs(self.hw, GPT120, base_settings(), "max_speed", probe=PROBE_CUDA, include_unsafe=True)
        self.assertTrue(all(x["admitted"] for x in unsafe))

    def test_hdd_and_tiny_ram_machine_rejects_the_big_model_everywhere(self):
        hw = hwmodel.preset_hardware("duck-user-hdd")
        c = S.candidate_configs(hw, GPT120, base_settings(), "max_speed", probe=PROBE_CUDA)
        self.assertFalse(any(x["admitted"] for x in c))
        self.assertTrue(all(x["reject_reason"] for x in c))

    def test_no_gpu_machine_has_only_cpu_candidates(self):
        hw = hwmodel.preset_hardware("no-gpu-8gb")
        c = S.candidate_configs(hw, DENSE, base_settings(), "balanced", probe=PROBE_CPU)
        self.assertTrue(all((x["overrides"].get("n_gpu_layers") or 0) == 0 for x in c))


class TestTransferChecks(unittest.TestCase):
    HW = K.mock_hw(vram_mb=4096)

    def gpu(self, gen, width):
        return {"gpus": [{"name": "x", "pcie_gen": {"value": gen, "status": "measured"}, "pcie_width": {"value": width, "status": "measured"}}]}

    def test_gen3_x16_is_about_15_7_gbs_and_not_slower_than_ddr4(self):
        r = S.host_transfer_check(self.HW, self.gpu(3, 16), host_gbs={"value": 12.0, "status": "measured"})
        self.assertAlmostEqual(r["pcie_gbs"]["value"], 15.76, places=2)
        self.assertEqual(r["pcie_gbs"]["status"], "estimated")
        self.assertEqual(r["host_gbs"]["status"], "measured")
        self.assertIsNone(r["warn"])

    def test_slow_link_warns_with_the_numbers(self):
        r = S.host_transfer_check(self.HW, self.gpu(1, 16), host_gbs=12.0)
        self.assertEqual(r["pcie_gbs"]["value"], 4.0)
        self.assertIn("slower than host memory", r["warn"])
        self.assertIn("4", r["warn"])
        x1 = S.host_transfer_check(self.HW, self.gpu(3, 1), host_gbs=12.0)
        self.assertIsNotNone(x1["warn"])

    def test_per_lane_table(self):
        for gen, expect in ((1, 4.0), (2, 8.0), (3, 15.76), (4, 31.5), (5, 63.0)):
            self.assertAlmostEqual(S.host_transfer_check(self.HW, self.gpu(gen, 16))["pcie_gbs"]["value"], expect, delta=0.1)

    def test_without_a_measurement_it_compares_with_the_theoretical_ram_bandwidth_and_says_so(self):
        r = S.host_transfer_check(self.HW, self.gpu(1, 4))
        self.assertEqual(r["host_gbs"]["status"], "unavailable")
        self.assertEqual(r["host_theoretical_gbs"]["status"], "estimated")
        self.assertIsNotNone(r["warn"])

    def test_input_shapes(self):
        gpu = {"pcie_gen": 3, "pcie_width": 16}
        for probe in (gpu, [gpu], {"gpus": [gpu]}):
            self.assertAlmostEqual(S.host_transfer_check(self.HW, probe)["pcie_gbs"]["value"], 15.76, places=2)

    def test_missing_or_unknown_link_info_is_unavailable_never_guessed(self):
        for probe in (None, {}, {"gpus": []}, [{"pcie_gen": None}], {"gpus": [{"pcie_gen": {"value": None, "status": "unavailable"}}]}):
            r = S.host_transfer_check(self.HW, probe, host_gbs=10)
            self.assertEqual(r["pcie_gbs"]["status"], "unavailable", probe)
            self.assertIsNone(r["warn"])
        self.assertEqual(S.host_transfer_check(self.HW, self.gpu(9, 16))["pcie_gbs"]["status"], "unavailable")

    def test_transfer_dominated_is_unknown_without_a_measurement(self):
        self.assertIsNone(S.transfer_dominated({}))
        self.assertIsNone(S.transfer_dominated(None))
        rec = {"metrics": {"transfer_bandwidth_gbs": {"value": None, "status": "unavailable"}}}
        self.assertIsNone(S.transfer_dominated(rec))
        self.assertIsNone(S.transfer_dominated({"metrics": {"transfer_bandwidth_gbs": {"value": 9.0, "status": "measured"}}}))  # no link to compare

    def test_transfer_dominated_with_measured_numbers(self):
        rec = {"pcie_gbs": 10.0, "metrics": {"transfer_bandwidth_gbs": {"value": 9.0, "status": "measured"}}}
        self.assertTrue(S.transfer_dominated(rec))
        rec["metrics"]["transfer_bandwidth_gbs"]["value"] = 2.0
        self.assertFalse(S.transfer_dominated(rec))


if __name__ == "__main__":
    unittest.main()
