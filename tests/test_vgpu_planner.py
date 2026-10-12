import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gguf_helper import write_gguf

from duckstoken import gguf
from duckstoken.vgpu import hwmodel, memcalc, planner

GIB = 2**30
MIB = 2**20

# Official shapes of the gpt-oss models (model cards; see gptoss.py). 120b file: 63.4 GB decimal.
GPT120 = {"name": "gpt-oss-120b", "architecture": "gpt-oss", "size_bytes": int(63.4e9), "block_count": 36,
          "expert_count": 128, "expert_used_count": 4, "is_moe": True, "embedding_length": 2880, "head_count": 64,
          "head_count_kv": 8, "key_length": 64, "value_length": 64, "expert_feed_forward_length": 2880,
          "feed_forward_length": 2880, "sliding_window": 128, "context_length": 131072, "file_type": 38,
          "file_type_name": "MXFP4_MOE", "vocab_size": 201088, "path": ""}
GPT20 = dict(GPT120, name="gpt-oss-20b", size_bytes=int(12.1e9), block_count=24, expert_count=32)
# Llama-2-7B shape, Q4_K_M
DENSE7B = {"name": "dense-7b", "architecture": "llama", "size_bytes": int(4.08 * GIB), "block_count": 32,
           "embedding_length": 4096, "head_count": 32, "head_count_kv": 32, "feed_forward_length": 11008,
           "context_length": 4096, "file_type": 15, "file_type_name": "Q4_K_M", "is_moe": False, "expert_count": 0,
           "expert_used_count": 0, "vocab_size": 32000}

# Values printed by the compiled llama-server (CPU build, commit 08ffba2, -lv 4) for real GGUF files; see
# duckstoken/vgpu/memcalc.py for how the compute-buffer formula was fitted to them.
QWEN05 = {"embedding_length": 896, "feed_forward_length": 4864, "head_count": 14, "head_count_kv": 2, "block_count": 24}
TINYLLAMA = {"embedding_length": 2048, "feed_forward_length": 5632, "head_count": 32, "head_count_kv": 4, "block_count": 22}
LLAMA1B = {"embedding_length": 2048, "feed_forward_length": 8192, "head_count": 32, "head_count_kv": 8, "block_count": 16}
SMOL360 = {"embedding_length": 960, "feed_forward_length": 2560, "head_count": 15, "head_count_kv": 5, "block_count": 32}
QWEN3_06 = {"embedding_length": 1024, "feed_forward_length": 3072, "head_count": 16, "head_count_kv": 8, "block_count": 28}
REAL_COMPUTE_MIB = [  # (model, ctx, ubatch, flash_attn, compute buffer MiB as printed)
    (QWEN05, 4096, 128, True, 10.32), (QWEN05, 4096, 512, True, 41.27), (QWEN05, 4096, 1024, True, 82.54),
    (QWEN05, 1024, 512, True, 38.27), (QWEN05, 16384, 512, True, 53.27), (QWEN05, 32768, 512, True, 69.27),
    (TINYLLAMA, 4096, 128, True, 14.25), (TINYLLAMA, 4096, 512, True, 57.02), (TINYLLAMA, 16384, 512, True, 69.02),
    (LLAMA1B, 4096, 512, True, 72.02), (LLAMA1B, 4096, 1024, True, 144.04),
    (SMOL360, 4096, 512, True, 28.39), (QWEN3_06, 4096, 512, True, 32.02), (QWEN3_06, 4096, 1024, True, 64.04)]
REAL_COMPUTE_FA_OFF = [(QWEN05, 4096, 512, 131.01), (QWEN05, 16384, 512, 491.01), (TINYLLAMA, 4096, 512, 289.01),
                       (LLAMA1B, 4096, 512, 290.01), (SMOL360, 4096, 512, 140.51), (QWEN3_06, 4096, 512, 158.01)]

REAL_LOG = """\
0.00.584.780 I load_tensors:          AMX model buffer size =   225.48 MiB
0.00.584.820 I load_tensors:   CPU_Mapped model buffer size =   325.02 MiB
0.00.701.087 I llama_context:        CPU  output buffer size =     0.58 MiB
0.00.701.218 I llama_kv_cache:        CPU KV buffer size =    12.75 MiB
0.00.708.224 I llama_kv_cache: size =   12.75 MiB (  2048 cells,  24 layers,  1/1 seqs), K (q8_0):    6.38 MiB, V (q8_0):    6.38 MiB
0.00.712.055 I sched_reserve:        CPU compute buffer size =    39.30 MiB
"""


def plan_for(preset, model, cfg=None, **kw):
    return planner.plan(hwmodel.preset_hardware(preset), model, cfg, **kw)


class BytesPerElementTests(unittest.TestCase):
    def test_ggml_block_sizes(self):
        self.assertEqual(planner.bytes_per_element("f32"), 4)
        self.assertEqual(planner.bytes_per_element("f16"), 2)
        self.assertEqual(planner.bytes_per_element("bf16"), 2)
        self.assertAlmostEqual(planner.bytes_per_element("q8_0"), 34 / 32)
        self.assertAlmostEqual(planner.bytes_per_element("q5_0"), 22 / 32)
        self.assertAlmostEqual(planner.bytes_per_element("q5_1"), 24 / 32)
        self.assertAlmostEqual(planner.bytes_per_element("q4_0"), 18 / 32)
        self.assertAlmostEqual(planner.bytes_per_element("q4_1"), 20 / 32)
        self.assertAlmostEqual(planner.bytes_per_element("IQ4_NL"), 18 / 32)

    def test_unknown_type(self):
        with self.assertRaises(ValueError):
            planner.bytes_per_element("q3_k")
        self.assertEqual(planner.GIB, 2**30)


class KvCacheTests(unittest.TestCase):
    def test_matches_real_llama_server_logs(self):
        # printed "KV buffer size": qwen05 f16 ctx16384 = 192 MiB; tinyllama q8_0 ctx16384 = 187 MiB; qwen05 q8_0 ctx2048 = 12.75
        q = dict(QWEN05, key_length=64)
        self.assertAlmostEqual(planner.kv_cache_bytes(q, 16384, "f16", "f16")[0] / MIB, 192.0, delta=0.01)
        self.assertAlmostEqual(planner.kv_cache_bytes(q, 2048, "q8_0", "q8_0")[0] / MIB, 12.75, delta=0.01)
        t = dict(TINYLLAMA, key_length=64)
        self.assertAlmostEqual(planner.kv_cache_bytes(t, 16384, "q8_0", "q8_0")[0] / MIB, 187.0, delta=0.01)

    def test_gqa_changes_size(self):
        mha = dict(DENSE7B)
        gqa = dict(DENSE7B, head_count_kv=8)
        self.assertAlmostEqual(planner.kv_cache_bytes(mha, 4096, "f16", "f16")[0] / planner.kv_cache_bytes(gqa, 4096, "f16", "f16")[0], 4.0)

    def test_dense_total_and_per_token(self):
        total, per_tok, br = planner.kv_cache_bytes(DENSE7B, 4096, "f16", "f16")
        self.assertEqual(total, 32 * 4096 * 2 * 32 * 128 * 2)  # layers x ctx x (K+V) x kv heads x head dim x 2 bytes
        self.assertEqual(per_tok, 32 * 2 * 32 * 128 * 2)
        self.assertEqual(br["swa_layers"], 0)
        self.assertEqual(br["global_layers"], 32)

    def test_gpt_oss_sliding_window_layers(self):
        total, per_tok, br = planner.kv_cache_bytes(GPT120, 131072, "q8_0", "q8_0", ubatch=512)
        self.assertEqual((br["swa_layers"], br["global_layers"]), (18, 18))
        self.assertEqual(br["cells_swa"], 768)  # pad256(128 + 512)
        cell = 8 * 64 * (34 / 32) * 2
        self.assertEqual(per_tok, 18 * cell)
        self.assertAlmostEqual(total, 18 * cell * 131072 + 18 * cell * 768, delta=1)
        full, _, fbr = planner.kv_cache_bytes(GPT120, 131072, "q8_0", "q8_0", {"swa_full": True})
        self.assertAlmostEqual(full, 36 * cell * 131072, delta=1)
        self.assertGreater(full, total * 1.9)
        self.assertTrue(fbr["swa_full"])

    def test_swa_window_never_exceeds_context(self):
        _, _, br = planner.kv_cache_bytes(GPT120, 256, "f16", "f16")
        self.assertEqual(br["cells_swa"], 256)

    def test_swa_unknown_architecture_is_treated_as_global(self):
        m = dict(GPT120, architecture="mystery-arch")
        _, _, br = planner.kv_cache_bytes(m, 8192, "f16", "f16")
        self.assertEqual(br["swa_layers"], 0)

    def test_context_is_padded_to_256(self):
        a = planner.kv_cache_bytes(DENSE7B, 4097, "f16", "f16")[0]
        b = planner.kv_cache_bytes(DENSE7B, 4352, "f16", "f16")[0]
        self.assertEqual(a, b)

    def test_missing_attention_shape_falls_back_and_says_so(self):
        m = {"block_count": 10, "embedding_length": 2048}
        total, _, br = planner.kv_cache_bytes(m, 1024, "f16", "f16")
        self.assertGreater(total, 0)
        self.assertTrue(br["notes"])
        self.assertEqual(planner.kv_cache_bytes({"size_bytes": 1}, 1024)[0], 0)

    def test_key_value_length_default_from_embedding(self):
        a = planner.kv_cache_bytes({"block_count": 4, "embedding_length": 512, "head_count": 8, "head_count_kv": 2}, 256, "f16", "f16")[0]
        b = planner.kv_cache_bytes({"block_count": 4, "embedding_length": 512, "head_count": 8, "head_count_kv": 2,
                                    "key_length": 64, "value_length": 64}, 256, "f16", "f16")[0]
        self.assertEqual(a, b)

    def test_gguf_summary_dict_is_accepted(self):
        d = Path(tempfile.mkdtemp())
        write_gguf(d / "m.gguf", {"general.architecture": "llama", "llama.block_count": 4, "llama.embedding_length": 256,
                                  "llama.attention.head_count": 4, "llama.attention.head_count_kv": 2}, 100)
        s = gguf.summarize(d / "m.gguf")
        self.assertGreater(planner.kv_cache_bytes(s, 512, "f16", "f16")[0], 0)


class ComputeBufferTests(unittest.TestCase):
    def test_matches_real_llama_server_with_flash_attention(self):
        for m, ctx, ub, fa, real in REAL_COMPUTE_MIB:
            got = planner.compute_buffer_bytes(m, ctx, ub, False, None, fa) / MIB
            self.assertAlmostEqual(got, real, delta=real * 0.01, msg=(m["embedding_length"], ctx, ub))

    def test_without_flash_attention_is_a_bounded_overestimate(self):
        for m, ctx, ub, real in REAL_COMPUTE_FA_OFF:
            got = planner.compute_buffer_bytes(m, ctx, ub, False, None, False) / MIB
            self.assertGreaterEqual(got, real * 0.99)
            self.assertLessEqual(got, real * 1.25)

    def test_scales_with_ubatch_and_flash(self):
        a = planner.compute_buffer_bytes(GPT120, 4096, 256, True)
        b = planner.compute_buffer_bytes(GPT120, 4096, 1024, True)
        self.assertAlmostEqual(b / a, 4.0, delta=0.3)
        self.assertGreater(planner.compute_buffer_bytes(GPT120, 4096, 512, True, None, False),
                           planner.compute_buffer_bytes(GPT120, 4096, 512, True, None, True))

    def test_moe_uses_experts_used_times_expert_width(self):
        moe = planner.compute_buffer_bytes(GPT120, 4096, 512)
        dense_like = planner.compute_buffer_bytes(dict(GPT120, is_moe=False, expert_feed_forward_length=0, feed_forward_length=2880), 4096, 512)
        self.assertGreater(moe, dense_like)

    def test_unknown_model_gives_zero(self):
        self.assertEqual(planner.compute_buffer_bytes({"size_bytes": 5}, 4096, 512), 0)

    def test_ubatch_capped_by_context(self):
        self.assertEqual(planner.compute_buffer_bytes(QWEN05, 128, 4096), planner.compute_buffer_bytes(QWEN05, 128, 128))


class WeightsSplitTests(unittest.TestCase):
    def test_gpt_oss_120b(self):
        ws = planner.weights_split(GPT120)
        self.assertEqual(ws["total_bytes"], int(63.4e9))
        self.assertEqual(ws["shared_bytes"] + ws["expert_bytes"], ws["total_bytes"])
        self.assertEqual(ws["basis"], "dims")
        self.assertGreater(ws["expert_fraction"], 0.94)
        self.assertLess(ws["expert_fraction"], 0.985)
        self.assertLessEqual(ws["expert_bytes"], ws["total_bytes"])
        self.assertAlmostEqual(ws["per_layer_expert_bytes"] * 36, ws["expert_bytes"], delta=36)
        # experts: 36 x 128 x 3 x 2880 x 2880 weights at 4.25 bits = 60.9 GB; attention ~28 MB per layer
        self.assertAlmostEqual(ws["expert_bytes"] / 1e9, 61.2, delta=1.0)
        self.assertGreater(ws["per_layer_shared_bytes"], 20e6)
        self.assertLess(ws["per_layer_shared_bytes"], 60e6)
        self.assertGreater(ws["embedding_bytes"], 1.0e9)
        self.assertLess(ws["embedding_bytes"], 2.0e9)

    def test_gpt_oss_20b_matches_official_size_within_a_percent(self):
        ws = planner.weights_split(GPT20)
        # official 20b file is 12.1 GB: experts 24 x 32 x 3 x 2880^2 x 4.25/8 = 10.2 GB + Q8_0 shared tensors
        self.assertAlmostEqual(ws["expert_bytes"] / 1e9, 10.2, delta=0.5)
        self.assertAlmostEqual((ws["expert_bytes"] + ws["shared_bytes"]) / 1e9, 12.1, delta=0.01)

    def test_dense_has_no_experts(self):
        ws = planner.weights_split(DENSE7B)
        self.assertEqual(ws["expert_bytes"], 0)
        self.assertEqual(ws["shared_bytes"], ws["total_bytes"])
        self.assertAlmostEqual(ws["per_layer_shared_bytes"] * 32 + ws["embedding_bytes"], ws["total_bytes"], delta=32)

    def test_ratio_fallback_without_dims(self):
        m = {"size_bytes": 10 * GIB, "is_moe": True, "expert_count": 64, "expert_used_count": 4, "block_count": 32,
             "file_type": 15}
        ws = planner.weights_split(m)
        self.assertEqual(ws["basis"], "ratio")
        self.assertGreater(ws["expert_fraction"], 0.9)
        self.assertLessEqual(ws["expert_bytes"], ws["total_bytes"])

    def test_tied_embeddings_for_small_models(self):
        small = {"size_bytes": 400 * MIB, "block_count": 24, "embedding_length": 896, "vocab_size": 151936, "file_type": 15}
        ws = planner.weights_split(small)
        self.assertTrue(ws["tied_embeddings"])
        self.assertEqual(ws["output_bytes"], ws["token_embd_bytes"])

    def test_never_exceeds_file_size_for_any_input(self):
        for size in (1, 1000, 10**9):
            ws = planner.weights_split({"size_bytes": size, "block_count": 4, "embedding_length": 4096, "vocab_size": 10**6})
            self.assertLessEqual(ws["embedding_bytes"], size)
            self.assertLessEqual(ws["shared_bytes"], size)
            self.assertGreaterEqual(ws["per_layer_shared_bytes"], 0)

    def test_zero_size(self):
        ws = planner.weights_split({})
        self.assertEqual((ws["total_bytes"], ws["expert_bytes"]), (0, 0))


class ParseBufferLogTests(unittest.TestCase):
    def test_real_log(self):
        r = planner.parse_buffer_log(REAL_LOG.splitlines())
        self.assertEqual(r["model_buffers"], {"AMX": 225.48, "CPU_Mapped": 325.02})
        self.assertEqual(r["kv_buffers"], {"CPU": 12.75})
        self.assertEqual(r["compute_buffers"], {"CPU": 39.3})
        self.assertEqual(r["output_buffers"], {"CPU": 0.58})

    def test_accepts_one_string(self):
        self.assertEqual(planner.parse_buffer_log(REAL_LOG)["kv_buffers"], {"CPU": 12.75})

    def test_gpu_and_several_kv_caches_are_summed(self):
        lines = ["load_tensors:   CPU_Mapped model buffer size = 58123.45 MiB",
                 "load_tensors:        CUDA0 model buffer size =   912.50 MiB",
                 "llama_kv_cache:      CUDA0 KV buffer size =    10.00 MiB",
                 "llama_kv_cache:      CUDA0 KV buffer size =     2.50 MiB",
                 "llama_kv_cache:        CPU KV buffer size =   144.00 MiB",
                 "llama_context:      CUDA0 compute buffer size =   398.00 MiB",
                 "llama_context:  CUDA_Host compute buffer size =    20.01 MiB"]
        r = planner.parse_buffer_log(lines)
        self.assertEqual(r["model_buffers"]["CUDA0"], 912.5)
        self.assertEqual(r["kv_buffers"], {"CUDA0": 12.5, "CPU": 144.0})
        self.assertEqual(r["compute_buffers"], {"CUDA0": 398.0, "CUDA_Host": 20.01})

    def test_old_spellings(self):
        r = planner.parse_buffer_log(["llm_load_tensors:        CPU buffer size = 4096.00 MiB",
                                      "llama_new_context_with_model:        CPU compute buffer size =   70.50 MiB"])
        self.assertEqual(r["model_buffers"], {"CPU": 4096.0})
        self.assertEqual(r["compute_buffers"], {"CPU": 70.5})

    def test_garbage_is_ignored(self):
        r = planner.parse_buffer_log(["", "hello", None, 5, "model buffer size = abc MiB", "x buffer size = 1 GiB"])
        self.assertTrue(all(v == {} for v in r.values()))
        self.assertTrue(all(v == {} for v in planner.parse_buffer_log(None).values()))

    def test_tier_split(self):
        lines = ["load_tensors: CPU_Mapped model buffer size = 2048.00 MiB", "load_tensors: CUDA0 model buffer size = 1024.00 MiB",
                 "llama_context: CUDA_Host compute buffer size = 512.00 MiB", "llama_context: CUDA0 compute buffer size = 256.00 MiB"]
        t = planner.measured_tiers_gb(planner.parse_buffer_log(lines))
        self.assertEqual(t["ram_gb"]["model"], 2.0)
        self.assertEqual(t["vram_gb"]["model"], 1.0)
        self.assertEqual(t["ram_gb"]["compute"], 0.5)
        self.assertEqual(t["vram_gb"]["compute"], 0.25)
        self.assertTrue(memcalc.is_host_device("AMX"))
        self.assertFalse(memcalc.is_host_device("Vulkan0"))


class PlanShapeTests(unittest.TestCase):
    def setUp(self):
        self.r = plan_for("duck-user", GPT120)

    def test_exact_top_level_and_nested_keys(self):
        r = self.r
        self.assertEqual(set(r), {"model", "config", "requirements", "tiers", "placement", "admission", "speed", "assumptions", "status"})
        self.assertEqual(set(r["model"]), {"name", "architecture", "size_gb", "is_moe", "block_count", "expert_count",
                                           "expert_used_count", "quant", "path"})
        self.assertEqual(set(r["config"]), {"ctx_size", "n_gpu_layers", "n_cpu_moe", "kv_offload", "cache_type_k",
                                            "cache_type_v", "ubatch_size", "mmap", "mlock"})
        self.assertEqual(set(r["requirements"]), {"weights_gb", "kv_cache_gb", "compute_buffers_gb", "runtime_overhead_gb",
                                                  "os_reserve_gb", "safety_margin_gb", "total_gb"})
        self.assertEqual(set(r["tiers"]), {"vram", "ram", "storage", "disk_cache"})
        self.assertEqual(set(r["tiers"]["vram"]), {"physical_gb", "budget_gb", "planned_gb", "items"})
        self.assertEqual(set(r["tiers"]["ram"]), {"physical_gb", "available_gb", "budget_gb", "planned_gb", "items"})
        self.assertEqual(set(r["tiers"]["storage"]), {"drive", "kind", "free_gb", "mmap_resident_gb", "spill_gb",
                                                      "spill_fraction", "read_gbs", "model_on_slow_disk"})
        self.assertEqual(r["tiers"]["disk_cache"]["enabled"], False)
        self.assertEqual(set(r["placement"]), {"gpu_layers", "cpu_layers", "moe_experts_on_cpu_layers", "kv_on_gpu",
                                               "weights_vram_gb", "weights_ram_gb", "weights_spill_gb"})
        self.assertEqual(set(r["admission"]), {"verdict", "can_load", "blockers", "warnings", "suggestions", "headroom_gb"})
        self.assertEqual(set(r["speed"]), {"tokens_per_s", "bound_by", "explanation"})
        self.assertEqual(r["status"], "estimated")
        json.dumps(r)  # JSON-serialisable for the HTTP layer

    def test_every_number_is_labelled(self):
        r = self.r
        for k, v in r["requirements"].items():
            self.assertEqual(v["status"], "estimated", k)
        for tier in ("vram", "ram"):
            for k in ("budget_gb", "planned_gb"):
                self.assertEqual(r["tiers"][tier][k]["status"], "estimated")
        for k in ("mmap_resident_gb", "spill_gb", "spill_fraction", "read_gbs"):
            self.assertEqual(r["tiers"]["storage"][k]["status"], "estimated")
        self.assertEqual(r["speed"]["tokens_per_s"]["status"], "estimated")
        self.assertEqual(len(r["speed"]["tokens_per_s"]["value"]), 2)

    def test_what_if_hardware_is_never_called_measured(self):
        self.assertEqual(self.r["tiers"]["ram"]["physical_gb"]["status"], "estimated")
        hw = hwmodel.preset_hardware("duck-user")
        hw.pop("mock")
        real = planner.plan(hw, GPT120)
        self.assertEqual(real["tiers"]["ram"]["physical_gb"]["status"], "measured")
        self.assertEqual(real["tiers"]["ram"]["available_gb"]["status"], "measured")

    def test_total_requirement_is_the_sum_of_its_parts(self):
        q = self.r["requirements"]
        parts = sum(q[k]["value"] for k in ("weights_gb", "kv_cache_gb", "compute_buffers_gb", "runtime_overhead_gb",
                                            "os_reserve_gb", "safety_margin_gb"))
        self.assertAlmostEqual(parts, q["total_gb"]["value"], delta=0.01)

    def test_assumptions_document_every_constant_and_fact(self):
        keys = {a["key"]: a for a in self.r["assumptions"]}
        for k in planner.DEFAULT_ASSUMPTIONS:
            self.assertIn(k, keys)
            self.assertTrue(keys[k]["explain"], k)
        for k in ("os_reserve_gb", "safety_margin_pct", "vram_reserve_mb", "runtime_overhead_gb", "max_spill_fraction",
                  "swa_full", "ubatch_default", "kv_k_type", "kv_v_type", "hdd_spill_tolerance"):
            self.assertIn(k, keys)
        for k in ("model.layers", "model.experts", "model.quantization", "model.weight_split", "model.swa", "unit",
                  "ram.bandwidth_gbs", "storage.read_gbs", "gpu.bandwidth_gbs"):
            self.assertIn(k, keys)
        self.assertEqual(keys["os_reserve_gb"]["value"], 6.0)
        self.assertEqual(keys["model.swa"]["value"]["layers"], 18)

    def test_overridden_assumption_is_flagged(self):
        r = plan_for("duck-user", GPT120, assumptions={"os_reserve_gb": 2.0})
        a = {x["key"]: x for x in r["assumptions"]}["os_reserve_gb"]
        self.assertEqual(a["value"], 2.0)
        self.assertIn("overridden", a["explain"])

    def test_pure_function(self):
        hw = hwmodel.preset_hardware("duck-user")
        model, cfg = copy.deepcopy(GPT120), {"ctx_size": 8192}
        hw0, model0, cfg0 = copy.deepcopy(hw), copy.deepcopy(model), copy.deepcopy(cfg)
        a = planner.plan(hw, model, cfg)
        b = planner.plan(hw, model, cfg)
        self.assertEqual(a, b)
        self.assertEqual((hw, model, cfg), (hw0, model0, cfg0))

    def test_model_block(self):
        m = self.r["model"]
        self.assertEqual((m["name"], m["architecture"], m["quant"], m["block_count"]), ("gpt-oss-120b", "gpt-oss", "MXFP4", 36))
        self.assertAlmostEqual(m["size_gb"], 59.05, delta=0.01)


class OwnersPcTests(unittest.TestCase):
    """The project owner's machine: i7-8700K, 48 GB DDR4-2133, GTX 750 (2 GB), model on NVMe."""

    def test_gpt_oss_120b_pages_from_nvme_about_one_token_per_second(self):
        r = plan_for("duck-user", GPT120)
        adm, st, pl = r["admission"], r["tiers"]["storage"], r["placement"]
        self.assertEqual(adm["verdict"], "paging")
        self.assertTrue(adm["can_load"])
        self.assertEqual(adm["blockers"], [])
        self.assertTrue(any("paged" in w and "NVMe" in w for w in adm["warnings"]))
        self.assertGreater(st["spill_gb"]["value"], 10)
        self.assertGreater(st["spill_fraction"]["value"], 0.2)
        self.assertLess(st["spill_fraction"]["value"], 0.5)
        lo, hi = r["speed"]["tokens_per_s"]["value"]
        self.assertGreater(lo, 0.2)
        self.assertLess(hi, 2.5)
        self.assertEqual(r["speed"]["bound_by"], "storage")
        # the 2 GB card is nearly useless: it can only hold some attention weights
        self.assertLess(pl["weights_vram_gb"], 1.0)
        self.assertLess(pl["weights_vram_gb"] / r["model"]["size_gb"], 0.02)
        self.assertEqual(pl["moe_experts_on_cpu_layers"], 36)
        self.assertGreater(pl["weights_ram_gb"], 57)
        self.assertLess(r["tiers"]["vram"]["planned_gb"]["value"], r["tiers"]["vram"]["budget_gb"]["value"])
        self.assertIn("benchmark", " ".join(adm["warnings"]).lower())

    def test_gpt_oss_120b_numbers_are_internally_consistent(self):
        r = plan_for("duck-user", GPT120)
        st, ram = r["tiers"]["storage"], r["tiers"]["ram"]
        self.assertAlmostEqual(st["mmap_resident_gb"]["value"] + st["spill_gb"]["value"], r["placement"]["weights_ram_gb"], delta=0.01)
        self.assertAlmostEqual(st["spill_gb"]["value"] / r["model"]["size_gb"], st["spill_fraction"]["value"], delta=0.001)
        self.assertLessEqual(ram["planned_gb"]["value"], ram["budget_gb"]["value"])
        self.assertAlmostEqual(r["admission"]["headroom_gb"], -st["spill_gb"]["value"], delta=0.01)

    def test_gpt_oss_20b_fits_with_several_tokens_per_second(self):
        r = plan_for("duck-user", GPT20)
        self.assertEqual(r["admission"]["verdict"], "ok")
        self.assertEqual(r["tiers"]["storage"]["spill_gb"]["value"], 0)
        lo, hi = r["speed"]["tokens_per_s"]["value"]
        self.assertGreater(hi, 4.0)
        self.assertGreater(lo, 1.5)
        self.assertEqual(r["speed"]["bound_by"], "memory_bandwidth")
        self.assertGreater(r["admission"]["headroom_gb"], 20)

    def test_120b_is_much_slower_than_20b(self):
        hi120 = plan_for("duck-user", GPT120)["speed"]["tokens_per_s"]["value"][1]
        hi20 = plan_for("duck-user", GPT20)["speed"]["tokens_per_s"]["value"][1]
        self.assertGreater(hi20, hi120 * 3)

    def test_120b_on_the_hard_drive_is_rejected_unless_forced(self):
        r = plan_for("duck-user-hdd", GPT120)
        self.assertEqual(r["admission"]["verdict"], "reject")
        self.assertFalse(r["admission"]["can_load"])
        self.assertTrue(any("hard drive" in b for b in r["admission"]["blockers"]))
        self.assertTrue(any("SSD" in s for s in r["admission"]["suggestions"]))
        self.assertTrue(r["tiers"]["storage"]["model_on_slow_disk"])
        self.assertLess(r["speed"]["tokens_per_s"]["value"][1], 0.1)
        forced = plan_for("duck-user-hdd", GPT120, force=True)
        self.assertEqual(forced["admission"]["verdict"], "paging")
        self.assertTrue(forced["admission"]["can_load"])
        self.assertTrue(any("Forced" in w for w in forced["admission"]["warnings"]))

    def test_20b_on_hdd_loads_but_warns_about_first_load(self):
        r = plan_for("duck-user-hdd", GPT20)
        self.assertEqual(r["admission"]["verdict"], "ok")
        self.assertTrue(any("first load" in w for w in r["admission"]["warnings"]))

    def test_suggested_placement_for_the_small_card(self):
        hw = hwmodel.preset_hardware("duck-user")
        sp = planner.suggest_placement(hw, GPT120)
        self.assertEqual(sp["n_cpu_moe"], 36)  # no expert fits 2 GB
        self.assertLessEqual(sp["n_gpu_layers"], 37)
        r = planner.plan(hw, GPT120, {k: sp[k] for k in ("n_gpu_layers", "n_cpu_moe")})
        self.assertTrue(r["admission"]["can_load"])
        self.assertTrue(planner.fits_vram(hw, GPT120, {k: sp[k] for k in ("n_gpu_layers", "n_cpu_moe")}))


class VerdictTests(unittest.TestCase):
    def dense(self, gb, **kw):
        return dict(DENSE7B, size_bytes=int(gb * GIB), **kw)

    def test_ok_tight_paging_reject_ladder(self):
        hw = hwmodel.mock_hardware(ram_gb=24, vram_mb=0, gpu_name="", disk_kind="nvme")  # budget = 18 GB
        verdicts = [planner.plan(hw, self.dense(gb), {"ctx_size": 2048})["admission"]["verdict"] for gb in (8, 15, 18, 40)]
        self.assertEqual(verdicts, ["ok", "tight", "paging", "reject"])

    def test_ok_needs_ten_percent_headroom(self):
        hw = hwmodel.mock_hardware(ram_gb=24, vram_mb=0, gpu_name="")
        ok = planner.plan(hw, self.dense(12), {"ctx_size": 2048})
        self.assertEqual(ok["admission"]["verdict"], "ok")
        self.assertGreaterEqual(ok["admission"]["headroom_gb"], 1.8)
        tight = planner.plan(hw, self.dense(15), {"ctx_size": 2048})
        self.assertEqual(tight["admission"]["verdict"], "tight")
        self.assertGreaterEqual(tight["admission"]["headroom_gb"], 0)
        self.assertTrue(tight["admission"]["can_load"])

    def test_paging_over_max_spill_is_rejected(self):
        hw = hwmodel.mock_hardware(ram_gb=24, vram_mb=0, gpu_name="")
        r = planner.plan(hw, self.dense(40), {"ctx_size": 2048})
        self.assertEqual(r["admission"]["verdict"], "reject")
        self.assertTrue(any("limit" in b for b in r["admission"]["blockers"]))
        self.assertEqual(planner.plan(hw, self.dense(40), {"ctx_size": 2048}, force=True)["admission"]["verdict"], "paging")

    def test_max_spill_fraction_is_tunable(self):
        hw = hwmodel.mock_hardware(ram_gb=24, vram_mb=0, gpu_name="")
        r = planner.plan(hw, self.dense(40), {"ctx_size": 2048}, {"max_spill_fraction": 0.9})
        self.assertEqual(r["admission"]["verdict"], "paging")

    def test_mmap_off_or_mlock_on_cannot_page(self):
        hw = hwmodel.mock_hardware(ram_gb=24, vram_mb=0, gpu_name="")
        for cfg in ({"mmap": False}, {"mlock": True}):
            r = planner.plan(hw, self.dense(20), dict(cfg, ctx_size=2048))
            self.assertEqual(r["admission"]["verdict"], "reject", cfg)
            self.assertTrue(any("cannot be paged" in b for b in r["admission"]["blockers"]))

    def test_insufficient_ram_message_and_suggestions_are_concrete(self):
        hw = hwmodel.mock_hardware(ram_gb=16, vram_mb=0, gpu_name="")
        r = planner.plan(hw, self.dense(40), {"ctx_size": 8192})
        text = " ".join(r["admission"]["blockers"])
        self.assertIn("Not enough RAM", text)
        self.assertIn("16 GB installed", text)
        sugg = " ".join(r["admission"]["suggestions"])
        self.assertIn("smaller", sugg)
        self.assertIn("RAM", sugg)

    def test_paging_suggests_how_much_ram_would_fix_it(self):
        r = plan_for("duck-user", GPT120)
        # needs ~62 GB of memory plus the 6 GB OS reserve: 64 GB would still page a little, so 96 GB is the honest answer
        self.assertTrue(any("96 GB of RAM" in s for s in r["admission"]["suggestions"]), r["admission"]["suggestions"])

    def test_tight_and_current_free_ram_warnings(self):
        hw = hwmodel.mock_hardware(ram_gb=24, vram_mb=0, gpu_name="", ram_avail_gb=5)
        r = planner.plan(hw, self.dense(8), {"ctx_size": 2048})
        self.assertEqual(r["admission"]["verdict"], "ok")
        self.assertTrue(any("right now" in w for w in r["admission"]["warnings"]))

    def test_smaller_context_reduces_requirements_and_can_change_verdict(self):
        hw = hwmodel.mock_hardware(ram_gb=24, vram_mb=0, gpu_name="")
        m = dict(self.dense(14), head_count_kv=32)
        big = planner.plan(hw, m, {"ctx_size": 32768, "cache_type_k": "f16", "cache_type_v": "f16"})
        small = planner.plan(hw, m, {"ctx_size": 2048, "cache_type_k": "f16", "cache_type_v": "f16"})
        self.assertGreater(big["requirements"]["kv_cache_gb"]["value"], small["requirements"]["kv_cache_gb"]["value"] * 10)
        self.assertNotEqual(big["admission"]["verdict"], small["admission"]["verdict"])

    def test_os_reserve_assumption_changes_the_outcome(self):
        hw = hwmodel.mock_hardware(ram_gb=24, vram_mb=0, gpu_name="")
        base = planner.plan(hw, self.dense(18), {"ctx_size": 2048})["admission"]["verdict"]
        easy = planner.plan(hw, self.dense(18), {"ctx_size": 2048}, {"os_reserve_gb": 1.0})["admission"]["verdict"]
        self.assertEqual(base, "paging")
        self.assertEqual(easy, "ok")

    def test_unknown_ram_is_a_blocker_not_a_guess(self):
        hw = hwmodel.mock_hardware(vram_mb=0, gpu_name="")
        hw["ram"]["total_gb"] = 0
        r = planner.plan(hw, self.dense(4))
        self.assertEqual(r["admission"]["verdict"], "reject")
        self.assertTrue(any("could not be detected" in b for b in r["admission"]["blockers"]))


class BlockerTests(unittest.TestCase):
    def test_missing_file(self):
        hw = hwmodel.preset_hardware("duck-user")
        r = planner.plan(hw, dict(DENSE7B), model_path=str(Path(tempfile.gettempdir()) / "definitely-not-here" / "m.gguf"))
        self.assertEqual(r["admission"]["verdict"], "reject")
        self.assertTrue(any("not found" in b for b in r["admission"]["blockers"]))

    def test_existing_file_is_fine_and_not_a_disk_space_problem(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.gguf"
            f.write_bytes(b"GGUF")
            hw = hwmodel.mock_hardware(disk_free_gb=0.1)
            r = planner.plan(hw, dict(DENSE7B), model_path=str(f))
            self.assertTrue(r["admission"]["can_load"], r["admission"])

    def test_not_a_gguf(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.safetensors"
            f.write_bytes(b"x")
            r = planner.plan(hwmodel.preset_hardware("duck-user"), dict(DENSE7B), model_path=str(f))
            self.assertEqual(r["admission"]["verdict"], "reject")
            self.assertTrue(any("not a .gguf" in b for b in r["admission"]["blockers"]))
            self.assertTrue(any("convert" in s for s in r["admission"]["suggestions"]))

    def test_not_enough_disk_for_a_download(self):
        hw = hwmodel.mock_hardware(disk_free_gb=30)
        r = planner.plan(hw, dict(GPT120, remote=True))
        self.assertEqual(r["admission"]["verdict"], "reject")
        text = " ".join(r["admission"]["blockers"])
        self.assertIn("free disk space", text)
        self.assertIn("30 GB free", text)
        self.assertTrue(any("smaller quantization" in s or "Free disk" in s for s in r["admission"]["suggestions"]))

    def test_disk_margin_warning(self):
        hw = hwmodel.mock_hardware(disk_free_gb=62)
        r = planner.plan(hw, dict(GPT120, remote=True))
        self.assertFalse(any("free disk space" in b for b in r["admission"]["blockers"]))
        self.assertTrue(any("would remain free" in w for w in r["admission"]["warnings"]))

    def test_each_failure_has_distinct_text(self):
        disk = planner.plan(hwmodel.mock_hardware(disk_free_gb=30), dict(GPT120, remote=True))["admission"]["blockers"]
        ram = planner.plan(hwmodel.mock_hardware(ram_gb=16, vram_mb=0, gpu_name=""), dict(GPT120, remote=True))["admission"]["blockers"]
        vram = planner.plan(hwmodel.mock_hardware(vram_mb=2048), dict(DENSE7B, size_bytes=int(30 * GIB)), {"n_gpu_layers": 99, "kv_offload": True})["admission"]["blockers"]
        self.assertTrue(disk and ram and vram)
        self.assertIn("disk space", disk[0])
        self.assertIn("Not enough RAM", ram[0])
        self.assertIn("Not enough VRAM", vram[0])

    def test_unknown_size(self):
        r = planner.plan(hwmodel.preset_hardware("duck-user"), {"name": "x"})
        self.assertEqual(r["admission"]["verdict"], "reject")
        self.assertTrue(any("size is unknown" in b for b in r["admission"]["blockers"]))

    def test_quantized_v_cache_needs_flash_attention(self):
        r = plan_for("desktop-32gb-rtx3060", DENSE7B, {"flash_attn": "off", "cache_type_v": "q8_0"})
        self.assertEqual(r["admission"]["verdict"], "reject")
        self.assertTrue(any("flash attention" in b for b in r["admission"]["blockers"]))
        ok = plan_for("desktop-32gb-rtx3060", DENSE7B, {"flash_attn": "off", "cache_type_v": "f16"})
        self.assertTrue(ok["admission"]["can_load"])

    def test_unknown_cache_type(self):
        r = plan_for("duck-user", DENSE7B, {"cache_type_k": "q3_k"})
        self.assertEqual(r["admission"]["verdict"], "reject")


class VramTests(unittest.TestCase):
    def test_overcommit_is_a_blocker_not_silently_clamped(self):
        m = dict(DENSE7B, size_bytes=int(30 * GIB))
        r = plan_for("duck-user", m, {"n_gpu_layers": 99, "ctx_size": 2048})
        self.assertEqual(r["config"]["n_gpu_layers"], 99)  # reported as asked
        self.assertEqual(r["admission"]["verdict"], "reject")
        self.assertTrue(any("Not enough VRAM" in b for b in r["admission"]["blockers"]))
        self.assertTrue(any("Lower n_gpu_layers to" in s for s in r["admission"]["suggestions"]))
        self.assertGreater(r["tiers"]["vram"]["planned_gb"]["value"], r["tiers"]["vram"]["budget_gb"]["value"])

    def test_suggested_layer_count_actually_fits(self):
        hw = hwmodel.preset_hardware("duck-user")
        m = dict(DENSE7B, size_bytes=int(30 * GIB))
        best = planner.max_gpu_layers(hw, m, {"ctx_size": 2048})
        self.assertGreater(best, 0)
        self.assertLess(best, 33)
        self.assertTrue(planner.fits_vram(hw, m, {"ctx_size": 2048, "n_gpu_layers": best}))
        self.assertFalse(planner.fits_vram(hw, m, {"ctx_size": 2048, "n_gpu_layers": best + 1}))
        r = planner.plan(hw, m, {"ctx_size": 2048, "n_gpu_layers": best})
        self.assertEqual([b for b in r["admission"]["blockers"] if "VRAM" in b], [])

    def test_max_gpu_layers_is_monotone_in_vram(self):
        m = dict(DENSE7B, size_bytes=int(30 * GIB))
        counts = [planner.max_gpu_layers(hwmodel.mock_hardware(vram_mb=v), m, {"ctx_size": 2048}) for v in (2048, 4096, 8192, 16384, 40960)]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], 33)  # all layers + output

    def test_no_gpu_means_zero_and_a_warning(self):
        hw = hwmodel.preset_hardware("laptop-16gb")
        self.assertEqual(planner.max_gpu_layers(hw, DENSE7B), 0)
        r = planner.plan(hw, DENSE7B, {"n_gpu_layers": 99})
        self.assertEqual(r["config"]["n_gpu_layers"], 0)
        self.assertEqual(r["placement"]["gpu_layers"], 0)
        self.assertTrue(any("no GPU" in w for w in r["admission"]["warnings"]))
        self.assertEqual(r["tiers"]["vram"]["planned_gb"]["value"], 0)

    def test_card_under_one_gb_is_not_used(self):
        hw = hwmodel.mock_hardware(vram_mb=512)
        self.assertEqual(planner.max_gpu_layers(hw, DENSE7B), 0)
        self.assertEqual(planner.plan(hw, DENSE7B, {"n_gpu_layers": 99})["placement"]["gpu_layers"], 0)

    def test_kv_follows_the_layers(self):
        hw = hwmodel.preset_hardware("desktop-32gb-rtx3060")
        on = planner.plan(hw, DENSE7B, {"n_gpu_layers": 99, "kv_offload": True, "ctx_size": 4096})
        off = planner.plan(hw, DENSE7B, {"n_gpu_layers": 99, "kv_offload": False, "ctx_size": 4096})
        self.assertTrue(on["placement"]["kv_on_gpu"])
        self.assertFalse(off["placement"]["kv_on_gpu"])
        self.assertGreater(on["tiers"]["vram"]["planned_gb"]["value"], off["tiers"]["vram"]["planned_gb"]["value"])
        self.assertLess(on["tiers"]["ram"]["planned_gb"]["value"], off["tiers"]["ram"]["planned_gb"]["value"])
        half = planner.plan(hw, DENSE7B, {"n_gpu_layers": 16, "kv_offload": True, "ctx_size": 4096})
        kv_total = half["requirements"]["kv_cache_gb"]["value"]
        in_vram = {i["label"]: i["gb"] for i in half["tiers"]["vram"]["items"]}["KV cache on GPU"]
        self.assertAlmostEqual(in_vram, kv_total / 2, delta=0.01)

    def test_dense_model_fully_on_a_big_gpu(self):
        r = plan_for("desktop-32gb-rtx3060", DENSE7B, {"n_gpu_layers": 99, "ctx_size": 4096})
        self.assertEqual(r["admission"]["verdict"], "ok")
        self.assertEqual(r["placement"]["cpu_layers"], 0)
        self.assertEqual(r["speed"]["bound_by"], "vram")
        self.assertGreater(r["speed"]["tokens_per_s"]["value"][1], 30)
        self.assertLess(r["placement"]["weights_ram_gb"], 1.0)  # only the token embedding stays on the host

    def test_cpu_only_dense_speed_is_bandwidth_bound(self):
        r = plan_for("laptop-16gb", DENSE7B, {"ctx_size": 2048})
        self.assertEqual(r["speed"]["bound_by"], "memory_bandwidth")
        lo, hi = r["speed"]["tokens_per_s"]["value"]
        self.assertTrue(0.5 < lo < hi < 20)

    def test_op_offload_staging_can_be_disabled(self):
        hw = hwmodel.preset_hardware("duck-user")
        on = planner.plan(hw, GPT120, {"n_gpu_layers": 17, "n_cpu_moe": 36})
        off = planner.plan(hw, GPT120, {"n_gpu_layers": 17, "n_cpu_moe": 36, "op_offload": False})
        self.assertGreater(on["tiers"]["vram"]["planned_gb"]["value"], off["tiers"]["vram"]["planned_gb"]["value"] + 0.4)

    def test_moe_blocker_recommends_raising_n_cpu_moe(self):
        r = plan_for("desktop-32gb-rtx3060", GPT120, {"n_gpu_layers": 99, "n_cpu_moe": 0})
        self.assertEqual(r["admission"]["verdict"], "reject")
        self.assertTrue(any("Raise n_cpu_moe" in s for s in r["admission"]["suggestions"]))
        k = planner.min_cpu_moe(hwmodel.preset_hardware("desktop-32gb-rtx3060"), GPT120)
        self.assertTrue(0 < k < 36)
        self.assertTrue(planner.plan(hwmodel.preset_hardware("desktop-32gb-rtx3060"), GPT120,
                                     {"n_gpu_layers": 99, "n_cpu_moe": k})["admission"]["can_load"])
        self.assertFalse(planner.fits_vram(hwmodel.preset_hardware("desktop-32gb-rtx3060"), GPT120, {"n_gpu_layers": 99, "n_cpu_moe": k - 1}))

    def test_min_cpu_moe_edge_cases(self):
        self.assertIsNone(planner.min_cpu_moe(hwmodel.preset_hardware("laptop-16gb"), GPT120))
        self.assertEqual(planner.min_cpu_moe(hwmodel.preset_hardware("desktop-32gb-rtx3060"), DENSE7B), 0)
        self.assertEqual(planner.min_cpu_moe(hwmodel.preset_hardware("workstation-128gb-rtx4090"), GPT20), 0)

    def test_cpu_moe_flag_equals_all_layers(self):
        hw = hwmodel.preset_hardware("duck-user")
        a = planner.plan(hw, GPT120, {"n_gpu_layers": 20, "cpu_moe": True})
        b = planner.plan(hw, GPT120, {"n_gpu_layers": 20, "n_cpu_moe": 36})
        self.assertEqual(a["placement"], b["placement"])
        self.assertEqual(a["config"]["n_cpu_moe"], 36)

    def test_n_cpu_moe_is_ignored_for_dense_models(self):
        r = plan_for("desktop-32gb-rtx3060", DENSE7B, {"n_cpu_moe": 8})
        self.assertEqual(r["placement"]["moe_experts_on_cpu_layers"], 0)
        self.assertTrue(any("not a mixture-of-experts" in w for w in r["admission"]["warnings"]))


class PartialHardwareTests(unittest.TestCase):
    def test_missing_sections_give_blockers_not_exceptions(self):
        for hw in ({}, {"ram": {}}, {"cpu": None, "gpus": None}, None):
            r = planner.plan(hw, GPT20)
            self.assertEqual(r["admission"]["verdict"], "reject", hw)
            self.assertTrue(any("could not be detected" in b for b in r["admission"]["blockers"]))
            json.dumps(r)

    def test_helpers_tolerate_partial_hardware(self):
        self.assertEqual(planner.max_gpu_layers({"ram": {"total_gb": 8}}, DENSE7B), 0)
        self.assertIsNone(planner.min_cpu_moe({}, GPT20))
        self.assertEqual(planner.suggest_placement({}, GPT20)["n_gpu_layers"], 0)

    def test_partial_cpu_info_is_fine(self):
        hw = hwmodel.preset_hardware("duck-user")
        hw["cpu"] = {}
        r = planner.plan(hw, GPT20)
        self.assertTrue(r["admission"]["can_load"])


class AutoFitTests(unittest.TestCase):
    """Settings the caller left out are fitted to the GPU (loudly); settings the caller passed are never touched.

    The tuner now proposes a placement that already fits, so the planner's safety net is exercised with the tuner's
    placement switched off (its old "offload everything" default)."""

    def blind_defaults(self):
        from unittest import mock
        from duckstoken import tuner
        real = tuner.recommend
        return mock.patch.object(tuner, "recommend", lambda hw, model, backend="auto", placement=True: real(hw, model, backend, placement=False))

    def test_default_plan_for_a_midrange_gpu_is_loadable_and_says_what_it_changed(self):
        hw = hwmodel.preset_hardware("desktop-32gb-rtx3060")
        with self.blind_defaults():
            r = planner.plan(hw, GPT120)
        self.assertTrue(r["admission"]["can_load"], r["admission"])
        self.assertTrue(any(w.startswith("Not set by you") and "n_cpu_moe" in w for w in r["admission"]["warnings"]))
        self.assertEqual(r["config"]["n_cpu_moe"], planner.min_cpu_moe(hw, GPT120, {k: v for k, v in r["config"].items() if k == "n_gpu_layers"}))
        self.assertLessEqual(r["tiers"]["vram"]["planned_gb"]["value"], r["tiers"]["vram"]["budget_gb"]["value"])

    def test_workstation_default_plan_is_ok(self):
        r = plan_for("workstation-128gb-rtx4090", GPT120)
        self.assertEqual(r["admission"]["verdict"], "ok")
        self.assertEqual(r["tiers"]["storage"]["spill_gb"]["value"], 0)

    def test_explicit_values_are_never_adjusted(self):
        hw = hwmodel.preset_hardware("desktop-32gb-rtx3060")
        r = planner.plan(hw, GPT120, {"n_gpu_layers": 99, "n_cpu_moe": 5})
        self.assertEqual(r["admission"]["verdict"], "reject")
        self.assertEqual((r["config"]["n_gpu_layers"], r["config"]["n_cpu_moe"]), (99, 5))
        self.assertFalse(any(w.startswith("Not set by you") for w in r["admission"]["warnings"]))

    def test_only_the_missing_one_of_two_is_adjusted(self):
        hw = hwmodel.preset_hardware("desktop-32gb-rtx3060")
        r = planner.plan(hw, GPT120, {"n_gpu_layers": 99})
        self.assertEqual(r["config"]["n_gpu_layers"], 99)
        self.assertGreater(r["config"]["n_cpu_moe"], 5)
        self.assertTrue(r["admission"]["can_load"])

    def test_dense_model_that_overflows_a_small_card_never_overcommits_by_default(self):
        m = dict(DENSE7B, size_bytes=int(30 * GIB))
        r = plan_for("duck-user", m, {"ctx_size": 2048})
        self.assertEqual([b for b in r["admission"]["blockers"] if "VRAM" in b], [])
        self.assertLess(r["config"]["n_gpu_layers"], 33)
        self.assertLessEqual(r["tiers"]["vram"]["planned_gb"]["value"], r["tiers"]["vram"]["budget_gb"]["value"])

    def test_missing_n_gpu_layers_is_lowered_when_the_heuristic_overcommits(self):
        m = dict(DENSE7B, size_bytes=int(1.95 * GIB))  # the old tuner rule kept all layers for models under the VRAM size
        with self.blind_defaults():
            r = plan_for("duck-user", m, {"ctx_size": 2048})
        self.assertEqual([b for b in r["admission"]["blockers"] if "VRAM" in b], [])
        self.assertTrue(any("n_gpu_layers" in w and "Not set by you" in w for w in r["admission"]["warnings"]), r["admission"]["warnings"])

    def test_model_just_too_big_for_the_card_runs_on_the_cpu_by_default(self):
        m = dict(DENSE7B, size_bytes=2 * GIB)
        r = plan_for("duck-user", m)
        self.assertTrue(r["admission"]["can_load"], r["admission"])
        self.assertEqual(r["admission"]["blockers"], [])

    def test_nothing_changes_when_it_already_fits(self):
        r = plan_for("duck-user", GPT20)
        self.assertFalse(any(w.startswith("Not set by you") for w in r["admission"]["warnings"]))


class WorkstationAndLaptopTests(unittest.TestCase):
    def test_workstation_runs_120b_comfortably_with_a_suggested_split(self):
        hw = hwmodel.preset_hardware("workstation-128gb-rtx4090")
        sp = planner.suggest_placement(hw, GPT120)
        self.assertEqual(sp["n_gpu_layers"], 37)
        r = planner.plan(hw, GPT120, {"n_gpu_layers": sp["n_gpu_layers"], "n_cpu_moe": sp["n_cpu_moe"], "ctx_size": 8192})
        self.assertEqual(r["admission"]["verdict"], "ok")
        self.assertEqual(r["tiers"]["storage"]["spill_gb"]["value"], 0)
        self.assertGreater(r["speed"]["tokens_per_s"]["value"][1], 10)

    def test_laptop_16gb_cannot_hold_120b(self):
        r = plan_for("laptop-16gb", GPT120)
        self.assertEqual(r["admission"]["verdict"], "reject")
        self.assertGreater(r["tiers"]["storage"]["spill_fraction"]["value"], 0.8)

    def test_laptop_16gb_20b_is_paging_not_ok(self):
        r = plan_for("laptop-16gb", GPT20)
        self.assertEqual(r["admission"]["verdict"], "paging")

    def test_32gb_desktop_120b_without_the_gpu_is_rejected(self):
        hw = hwmodel.preset_hardware("desktop-32gb-rtx3060", vram_mb=0)
        r = planner.plan(hw, GPT120)
        self.assertEqual(r["admission"]["verdict"], "reject")


class ConfigDefaultsTests(unittest.TestCase):
    def test_explicit_cfg_beats_recommendation(self):
        r = plan_for("duck-user", GPT20, {"ctx_size": 2048, "n_gpu_layers": 0, "n_cpu_moe": 0, "mmap": False, "ubatch_size": 128,
                                          "cache_type_k": "f16", "cache_type_v": "f16", "kv_offload": False, "mlock": False})
        self.assertEqual(r["config"], {"ctx_size": 2048, "n_gpu_layers": 0, "n_cpu_moe": 0, "kv_offload": False,
                                       "cache_type_k": "f16", "cache_type_v": "f16", "ubatch_size": 128, "mmap": False,
                                       "mlock": False})

    def test_missing_keys_come_from_the_tuner_recommendation(self):
        from duckstoken import tuner
        hw = hwmodel.preset_hardware("duck-user")
        tm = dict(GPT120, size_gb=GPT120["size_bytes"] / GIB)
        rec = tuner.recommend(hw, tm)["values"]
        r = planner.plan(hw, GPT120)
        self.assertEqual(r["config"]["ctx_size"], rec["ctx_size"])
        self.assertEqual(r["config"]["n_cpu_moe"], rec["n_cpu_moe"])
        self.assertEqual(r["config"]["ubatch_size"], rec["ubatch_size"])

    def test_context_zero_uses_the_training_context_and_warns(self):
        r = plan_for("duck-user", GPT120, {"ctx_size": 0})
        self.assertEqual(r["config"]["ctx_size"], 131072)
        self.assertTrue(any("training context" in w for w in r["admission"]["warnings"]))
        self.assertGreater(r["requirements"]["kv_cache_gb"]["value"], 1.0)

    def test_context_beyond_training_warns(self):
        r = plan_for("laptop-16gb", DENSE7B, {"ctx_size": 16384})
        self.assertTrue(any("trained for" in w for w in r["admission"]["warnings"]))

    def test_bad_model_dict_does_not_crash(self):
        for m in ({}, {"size_gb": "x"}, {"size_bytes": None, "block_count": "7"}, {"size_gb": 3, "head_count": None}):
            r = plan_for("duck-user", m)
            self.assertIn(r["admission"]["verdict"], ("ok", "tight", "paging", "reject"))
            json.dumps(r)

    def test_resident_estimate(self):
        r = plan_for("duck-user", GPT20)
        want = r["tiers"]["vram"]["planned_gb"]["value"] + r["tiers"]["ram"]["planned_gb"]["value"]
        self.assertAlmostEqual(planner.resident_estimate_gb(r), want, delta=0.01)
        self.assertGreater(planner.resident_estimate_gb(r), 10)
        paging = plan_for("duck-user", GPT120)
        self.assertLess(planner.resident_estimate_gb(paging), paging["model"]["size_gb"])  # paged weights are not resident

    def test_unknown_layer_count_is_cpu_only(self):
        m = dict(DENSE7B, block_count=0)
        r = plan_for("desktop-32gb-rtx3060", m, {"n_gpu_layers": 99})
        self.assertEqual(r["placement"]["gpu_layers"], 0)
        self.assertTrue(any("layer count is unknown" in w for w in r["admission"]["warnings"]))


class GgufSummaryExtraKeysTests(unittest.TestCase):
    def test_additive_keys(self):
        d = Path(tempfile.mkdtemp())
        write_gguf(d / "gpt.gguf", {"general.architecture": "gpt-oss", "general.name": "t", "general.file_type": 38,
                                    "gpt-oss.block_count": 24, "gpt-oss.expert_count": 32, "gpt-oss.expert_used_count": 4,
                                    "gpt-oss.embedding_length": 2880, "gpt-oss.feed_forward_length": 2880,
                                    "gpt-oss.expert_feed_forward_length": 2880, "gpt-oss.attention.key_length": 64,
                                    "gpt-oss.attention.value_length": 64, "gpt-oss.attention.sliding_window": 128}, 100)
        s = gguf.summarize(d / "gpt.gguf")
        self.assertEqual(s["expert_feed_forward_length"], 2880)
        self.assertEqual((s["key_length"], s["value_length"], s["sliding_window"], s["feed_forward_length"]), (64, 64, 128, 2880))
        self.assertEqual(s["file_type_name"], "MXFP4_MOE")
        self.assertEqual(s["architecture"], "gpt-oss")
        r = planner.plan(hwmodel.preset_hardware("duck-user"), s)
        self.assertEqual(r["model"]["quant"], "MXFP4")

    def test_old_keys_unchanged_and_new_keys_default_to_zero(self):
        d = Path(tempfile.mkdtemp())
        write_gguf(d / "m.gguf", {"general.architecture": "llama", "llama.block_count": 4}, 100)
        s = gguf.summarize(d / "m.gguf")
        self.assertEqual(s["block_count"], 4)
        self.assertEqual((s["sliding_window"], s["expert_feed_forward_length"], s["key_length"]), (0, 0, 0))
        self.assertIsNone(s["file_type_name"])
        self.assertFalse(s["is_moe"])


@unittest.skipUnless(os.environ.get("DUCK_REAL_LLAMA") and os.environ.get("DUCK_REAL_MODEL"),
                     "set DUCK_REAL_LLAMA=/path/to/llama-server and DUCK_REAL_MODEL=/path/to.gguf to compare with a real engine")
class RealEngineMemoryTest(unittest.TestCase):
    """Starts the real llama-server on a small GGUF and compares its printed buffer sizes with the planner's estimates."""

    def run_engine(self, extra):
        import socket
        import subprocess
        import time
        exe, model = os.environ["DUCK_REAL_LLAMA"], os.environ["DUCK_REAL_MODEL"]
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = dict(os.environ, LD_LIBRARY_PATH=str(Path(exe).parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
        proc = subprocess.Popen([exe, "-m", model, "--port", str(port), "-np", "1", "-t", "2", "-lv", "4", "--no-warmup"] + extra,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=env,
                                cwd=str(Path(exe).parent))
        lines = []
        try:
            deadline = time.time() + 180
            for line in proc.stdout:
                lines.append(line.rstrip())
                if "compute buffer size" in line and "sched_reserve" in line:
                    break
                if time.time() > deadline:
                    break
        finally:
            proc.kill()
            proc.wait()
            proc.stdout.close()
        return lines

    def test_kv_and_compute_estimates_match_the_real_log(self):
        summary = gguf.summarize(os.environ["DUCK_REAL_MODEL"])
        lines = self.run_engine(["-c", "2048", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-ub", "512"])
        parsed = planner.parse_buffer_log(lines)
        self.assertTrue(parsed["kv_buffers"] and parsed["compute_buffers"] and parsed["model_buffers"], parsed)
        kv_real = sum(parsed["kv_buffers"].values()) * MIB
        cb_real = sum(parsed["compute_buffers"].values()) * MIB
        kv_est = planner.kv_cache_bytes(summary, 2048, "q8_0", "q8_0", ubatch=512)[0]
        cb_est = planner.compute_buffer_bytes(summary, 2048, 512, False, None, True)
        self.assertAlmostEqual(kv_est / kv_real, 1.0, delta=0.02)
        self.assertAlmostEqual(cb_est / cb_real, 1.0, delta=0.05)
        self.assertGreater(sum(parsed["model_buffers"].values()) * MIB, summary["size_bytes"] * 0.5)
        plan = planner.plan(hwmodel.mock_hardware(vram_mb=0, gpu_name=""), summary, {"ctx_size": 2048, "cache_type_k": "q8_0",
                            "cache_type_v": "q8_0", "ubatch_size": 512, "flash_attn": "on"}, model_path=os.environ["DUCK_REAL_MODEL"])
        # plan() reports GiB rounded to 3 decimals (about 1 MiB), so compare at that resolution
        self.assertLess(abs(plan["requirements"]["kv_cache_gb"]["value"] * GIB - kv_real), 1.0 * MIB)


if __name__ == "__main__":
    unittest.main()


class WhatIfDiskIsAuthoritative(unittest.TestCase):
    def test_whatif_hdd_is_used_even_when_the_real_model_sits_on_another_drive_letter(self):
        # review finding: on Windows the real path's drive letter was matched against the mock disk's letter
        import tempfile
        from unittest import mock
        from duckstoken import hardware
        from duckstoken.vgpu import gptoss, hwmodel, planner
        with tempfile.TemporaryDirectory() as d, mock.patch.object(hardware, "_IS_WIN", True):
            path = Path(d) / "m.gguf"
            path.write_bytes(b"GGUF" + b"\0" * 64)
            hw = hwmodel.preset_hardware("duck-user-hdd")           # its drive is E:, the real file is elsewhere
            plan = planner.plan(hw, gptoss.spec_model("gpt-oss-120b"), {"ctx_size": 4096}, None, model_path=str(path))
        st = plan["tiers"]["storage"]
        self.assertEqual(st["kind"], "hdd")
        self.assertEqual(st["drive"], "E:")
        self.assertEqual(plan["admission"]["verdict"], "reject")      # a hard drive cannot page 20 GB


class AutoTuneIsAdmitted(unittest.TestCase):
    def test_tuner_recommendations_load_on_every_preset_machine(self):
        # review finding: n_cpu_moe = 36 - int(vram*1.2) put ~2x the VRAM of gpt-oss-120b on the card
        from duckstoken import tuner
        from duckstoken.vgpu import gptoss, hwmodel, planner
        for preset in ("duck-user", "desktop-32gb-rtx3060", "workstation-128gb-rtx4090"):
            hw = hwmodel.preset_hardware(preset)
            for name in ("gpt-oss-120b", "gpt-oss-20b"):
                m = gptoss.spec_model(name)
                rec = tuner.recommend(hw, m, tuner.resolve_backend(hw, "vulkan"))["values"]
                plan = planner.plan(hw, m, {k: rec[k] for k in planner.CFG_KEYS if k in rec})
                self.assertTrue(plan["admission"]["can_load"], (preset, name, plan["admission"]["blockers"]))
                self.assertFalse(any("VRAM" in b or "GPU" in b for b in plan["admission"]["blockers"]), (preset, name))
