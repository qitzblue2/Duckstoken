import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gguf_helper import write_gguf

from duckstoken import gguf
from duckstoken.vgpu import gptoss, gpucompat, hwmodel, planner

GIB = 2**30


def report(preset="duck-user", model="gpt-oss-120b", **kw):
    hw = kw.pop("hw", None) or hwmodel.preset_hardware(preset)
    return gptoss.feasibility_report(hw, model, **kw)


def check(r, id_):
    return next(c for c in r["checks"] if c["id"] == id_)


def good_summary(model="gpt-oss-120b", **over):
    spec = gptoss.SPECS[model]
    s = {"path": "", "name": model, "architecture": "gpt-oss", "size_bytes": spec["gguf_size_bytes"], "shards": 1,
         "block_count": spec["layers"], "expert_count": spec["experts"], "expert_used_count": 4, "is_moe": True,
         "file_type": 38, "file_type_name": "MXFP4_MOE", "embedding_length": 2880, "head_count": 64, "head_count_kv": 8,
         "context_length": 131072, "sliding_window": 128, "key_length": 64, "value_length": 64,
         "expert_feed_forward_length": 2880, "feed_forward_length": 2880, "vocab_size": 0}
    s.update(over)
    return s


class SpecTests(unittest.TestCase):
    def test_official_numbers(self):
        a, b = gptoss.SPECS["gpt-oss-120b"], gptoss.SPECS["gpt-oss-20b"]
        self.assertEqual((a["layers"], a["experts"], a["experts_used"]), (36, 128, 4))
        self.assertEqual((b["layers"], b["experts"], b["experts_used"]), (24, 32, 4))
        for s in (a, b):
            self.assertEqual((s["hidden_size"], s["heads"], s["kv_heads"], s["head_dim"]), (2880, 64, 8, 64))
            self.assertEqual((s["sliding_window"], s["vocab_size"], s["context_length"]), (128, 201088, 131072))
            self.assertIn("MXFP4", s["quantization"])
            self.assertTrue(s["sources"] and all(u.startswith("https://huggingface.co/") for u in s["sources"]))
        self.assertEqual((a["params_total_b"], a["params_active_b"]), (117, 5.1))
        self.assertEqual((b["params_total_b"], b["params_active_b"]), (21, 3.6))
        self.assertEqual(a["gguf_size_bytes"], 63_400_000_000)
        self.assertEqual(a["gguf_repo"], "ggml-org/gpt-oss-120b-GGUF")
        self.assertEqual(a["gguf_file"], "gpt-oss-120b-MXFP4.gguf")
        self.assertEqual(b["gguf_repo"], "ggml-org/gpt-oss-20b-GGUF")
        self.assertAlmostEqual(a["gguf_size_bytes"] / GIB, 59.0, delta=0.1)

    def test_spec_model_is_a_planner_model(self):
        m = gptoss.spec_model("gpt-oss-120b")
        self.assertTrue(m["remote"])
        self.assertEqual(m["file_type_name"], "MXFP4_MOE")
        ws = planner.weights_split(m)
        self.assertGreater(ws["expert_fraction"], 0.94)

    def test_unknown_model(self):
        with self.assertRaises(KeyError):
            gptoss.feasibility_report(hwmodel.preset_hardware("duck-user"), "gpt-oss-1b")


class OwnersPcTests(unittest.TestCase):
    def test_120b_pages_and_is_about_one_token_per_second(self):
        r = report()
        self.assertEqual(r["verdict"], "paging")
        self.assertFalse(r["in_memory"])
        self.assertIn("1 token", r["speed_class"])
        lo, hi = r["tokens_per_s"]["value"]
        self.assertTrue(0.2 < lo < hi < 2.5, (lo, hi))
        self.assertEqual(r["tokens_per_s"]["status"], "estimated")
        self.assertEqual(r["source"], "official-spec")
        self.assertIn("48 GB", r["explanation"])
        self.assertIn("NVMe", r["explanation"])
        self.assertIn("estimate", r["explanation"])
        self.assertEqual(r["recommendation"]["action"], "use_20b")
        self.assertEqual(r["recommendation"]["alternative_model"], "gpt-oss-20b")

    def test_requirement_breakdown_is_labelled(self):
        r = report()
        rq = r["requirements"]
        self.assertAlmostEqual(rq["weights_gb"]["value"], 59.05, delta=0.05)
        for k, v in rq.items():
            self.assertEqual(v["status"], "estimated", k)
        self.assertIs(r["plan"]["requirements"], rq)
        self.assertEqual(r["status"], "estimated")

    def test_20b_fits_comfortably_with_several_tokens_per_second(self):
        r = report(model="gpt-oss-20b")
        self.assertEqual(r["verdict"], "comfortable")
        self.assertTrue(r["in_memory"])
        self.assertGreater(r["tokens_per_s"]["value"][1], 4)
        self.assertEqual(r["recommendation"]["action"], "run")
        self.assertIsNone(r["alternative"])

    def test_the_120b_report_compares_with_the_20b(self):
        alt = report()["alternative"]
        self.assertEqual(alt["model"], "gpt-oss-20b")
        self.assertTrue(alt["fits"])
        self.assertGreater(alt["tokens_per_s"]["value"][1], 4)

    def test_gpu_section_is_honest_about_the_gtx_750(self):
        g = report()["gpu"]
        self.assertTrue(g["present"])
        self.assertEqual(g["gpus"][0]["architecture"], "Maxwell")
        self.assertLess(g["fraction_of_weights_on_gpu"]["value"], 0.05)
        self.assertEqual(g["fraction_of_weights_on_gpu"]["status"], "estimated")
        text = " ".join(g["can"] + g["cannot"] + g["notes"] + [g["benefit"]])
        self.assertIn("experts", text)
        self.assertIn("benchmark", g["benefit"].lower())
        self.assertIn("CUDA 13", text)
        self.assertIn("below the 4 GB", text)
        self.assertIsNone(g["backend_verified"])

    def test_gpu_backend_verification_comes_from_the_compat_report(self):
        hw = hwmodel.preset_hardware("duck-user")
        verified = gpucompat.compatibility_report(hw, {"status": "ok", "devices": [{"id": "CUDA0", "backend": "CUDA", "name": "x", "total_mb": 2048, "free_mb": 1}], "supports": {}})
        self.assertTrue(report(gpu_report=verified)["gpu"]["backend_verified"])
        unverified = gpucompat.compatibility_report(hw)
        self.assertFalse(report(gpu_report=unverified)["gpu"]["backend_verified"])

    def test_no_gpu_section(self):
        g = report("laptop-16gb")["gpu"]
        self.assertFalse(g["present"])
        self.assertIn("No GPU", g["benefit"])

    def test_never_auto_download_and_download_size_to_confirm(self):
        r = report()
        self.assertTrue(r["never_auto_download"])
        d = r["download"]
        self.assertTrue(d["needs_confirmation"])
        self.assertEqual(d["file"], "gpt-oss-120b-MXFP4.gguf")
        self.assertEqual(d["size_gb"]["status"], "estimated")
        self.assertAlmostEqual(d["size_gb"]["value"], 59.05, delta=0.05)
        self.assertEqual(d["size_bytes"], 63_400_000_000)
        self.assertIn("never downloads", d["note"])

    def test_json_serialisable(self):
        json.dumps(report())
        json.dumps(report("duck-user-hdd"))


class RamTableTests(unittest.TestCase):
    def test_rows_and_current_marker(self):
        rows = report()["ram_table"]
        self.assertEqual([r["ram_gb"] for r in rows], [16, 24, 32, 48, 64, 96, 128])
        self.assertEqual([r["ram_gb"] for r in rows if r["current"]], [48])
        for r in rows:
            self.assertIn(r["outcome"], ("fits", "paging", "reject"))
            self.assertTrue(r["note"])

    def test_progression_for_the_120b(self):
        rows = {r["ram_gb"]: r for r in report()["ram_table"]}
        self.assertEqual([rows[g]["outcome"] for g in (16, 24, 32)], ["reject"] * 3)
        self.assertEqual(rows[48]["outcome"], "paging")
        self.assertEqual(rows[128]["outcome"], "fits")
        order = {"reject": 0, "paging": 1, "fits": 2}
        outcomes = [order[rows[g]["outcome"]] for g in (16, 24, 32, 48, 64, 96, 128)]
        self.assertEqual(outcomes, sorted(outcomes))

    def test_unloadable_rows_have_no_speed_number(self):
        rows = {r["ram_gb"]: r for r in report()["ram_table"]}
        self.assertEqual(rows[16]["tokens_per_s"]["status"], "unavailable")
        self.assertIsNone(rows[16]["tokens_per_s"]["value"])
        self.assertEqual(rows[16]["speed_class"], "not loadable")
        self.assertEqual(rows[48]["tokens_per_s"]["status"], "estimated")

    def test_speed_improves_with_ram(self):
        rows = {r["ram_gb"]: r for r in report()["ram_table"]}
        self.assertGreater(rows[64]["tokens_per_s"]["value"][1], rows[48]["tokens_per_s"]["value"][1])
        self.assertGreater(rows[96]["tokens_per_s"]["value"][1], rows[64]["tokens_per_s"]["value"][1])

    def test_20b_table(self):
        rows = {r["ram_gb"]: r for r in report(model="gpt-oss-20b")["ram_table"]}
        self.assertEqual(rows[16]["outcome"], "paging")  # 11.3 GB weights vs 10 GB budget after the OS reserve
        self.assertEqual(rows[24]["outcome"], "fits")
        self.assertEqual(rows[128]["outcome"], "fits")

    def test_the_table_does_not_modify_the_hardware(self):
        hw = hwmodel.preset_hardware("duck-user")
        before = json.dumps(hw, sort_keys=True)
        gptoss.feasibility_report(hw)
        self.assertEqual(json.dumps(hw, sort_keys=True), before)


class SmallMachinesTests(unittest.TestCase):
    def test_16gb_laptop_is_not_realistic_and_suggests_20b(self):
        r = report("laptop-16gb")
        self.assertEqual(r["verdict"], "not_realistic")
        self.assertFalse(r["in_memory"])
        self.assertIn("not realistic", r["explanation"])
        self.assertIn("does not fit in memory", r["explanation"])
        self.assertEqual(r["recommendation"]["action"], "use_20b")
        self.assertTrue(any("gpt-oss-20b" in s for s in r["recommendation"]["steps"]))
        self.assertEqual(r["speed_class"], "not loadable")
        self.assertEqual(check(r, "ram")["status"], "fail")

    def test_32gb_laptop_is_not_realistic_and_20b_fits(self):
        hw = hwmodel.mock_hardware(ram_gb=32, vram_mb=0, gpu_name="", ram_speed_mts=3200)
        r = gptoss.feasibility_report(hw)
        self.assertEqual(r["verdict"], "not_realistic")
        self.assertEqual(r["recommendation"]["action"], "use_20b")
        self.assertTrue(r["alternative"]["fits"])
        self.assertIn("fits in memory", " ".join(r["recommendation"]["steps"]))

    def test_32gb_desktop_with_a_gpu_pages(self):
        r = report("desktop-32gb-rtx3060")
        self.assertEqual(r["verdict"], "paging")
        self.assertFalse(r["in_memory"])
        self.assertEqual(r["recommendation"]["action"], "use_20b")

    def test_8gb_machine_cannot_run_either(self):
        r = report("no-gpu-8gb")
        self.assertEqual(r["recommendation"]["action"], "add_ram")
        self.assertIsNone(r["recommendation"]["alternative_model"])
        self.assertEqual(report("no-gpu-8gb", "gpt-oss-20b")["verdict"], "not_realistic")


class WorkstationTests(unittest.TestCase):
    def test_128gb_workstation_is_comfortable(self):
        r = report("workstation-128gb-rtx4090")
        self.assertEqual(r["verdict"], "comfortable")
        self.assertTrue(r["in_memory"])
        self.assertEqual(r["recommendation"]["action"], "run")
        self.assertGreater(r["tokens_per_s"]["value"][1], 10)
        self.assertEqual(check(r, "ram")["status"], "ok")
        self.assertGreater(r["gpu"]["fraction_of_weights_on_gpu"]["value"], 0.2)
        self.assertEqual(check(r, "context")["status"], "ok")


class HddTests(unittest.TestCase):
    def test_120b_on_a_hard_drive_says_move_it(self):
        r = report("duck-user-hdd")
        self.assertEqual(r["verdict"], "not_realistic")
        self.assertEqual(r["recommendation"]["action"], "move_to_ssd")
        self.assertTrue(r["if_moved_to_ssd"]["can_load"])
        self.assertIn("SSD", r["recommendation"]["steps"][0])
        self.assertEqual(check(r, "storage_speed")["status"], "warn")
        self.assertIn("hard drive", r["explanation"])

    def test_20b_on_a_hard_drive_is_fine(self):
        r = report("duck-user-hdd", "gpt-oss-20b")
        self.assertEqual(r["verdict"], "comfortable")
        self.assertTrue(r["if_moved_to_ssd"]["can_load"])
        self.assertEqual(r["recommendation"]["action"], "run")

    def test_nvme_has_no_whatif(self):
        self.assertIsNone(report()["if_moved_to_ssd"])


class ChecksTests(unittest.TestCase):
    def test_check_shape_and_unknowns_without_a_file(self):
        r = report()
        for c in r["checks"]:
            self.assertEqual(set(c), {"id", "label", "status", "detail"})
            self.assertIn(c["status"], ("ok", "warn", "fail", "unknown"))
        for id_ in ("architecture", "experts", "quantization", "size", "engine", "file"):
            self.assertEqual(check(r, id_)["status"], "unknown", id_)
        self.assertIn("never started without your confirmation", check(r, "file")["detail"])

    def test_good_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "gpt-oss-120b-MXFP4.gguf"
            f.write_bytes(b"GGUF")
            r = report(file_summary=good_summary(path=str(f)), probe={"supports": {"gpt-oss": True}})
        self.assertEqual(r["source"], "file")
        for id_ in ("architecture", "experts", "quantization", "size", "engine", "file", "disk"):
            self.assertEqual(check(r, id_)["status"], "ok", (id_, check(r, id_)))

    def test_missing_file(self):
        r = report(file_summary=good_summary(path=str(Path(tempfile.gettempdir()) / "nowhere" / "x.gguf")))
        self.assertEqual(check(r, "file")["status"], "fail")
        self.assertIn("not found", check(r, "file")["detail"])
        self.assertEqual(r["verdict"], "not_realistic")
        self.assertTrue(any("not found" in b for b in r["plan"]["admission"]["blockers"]))

    def test_wrong_quantization(self):
        r = report(file_summary=good_summary(file_type=15, file_type_name="Q4_K_M"))
        c = check(r, "quantization")
        self.assertEqual(c["status"], "warn")
        self.assertIn("Q4_K_M", c["detail"])
        self.assertIn("not the official MXFP4", c["detail"])

    def test_wrong_architecture_and_experts(self):
        r = report(file_summary=good_summary(architecture="llama", expert_count=0, is_moe=False, file_type=15, file_type_name="Q4_K_M"))
        self.assertEqual(check(r, "architecture")["status"], "fail")
        self.assertEqual(check(r, "experts")["status"], "unknown")
        r = report(file_summary=good_summary(expert_count=64))
        self.assertEqual(check(r, "experts")["status"], "fail")
        self.assertIn("64", check(r, "experts")["detail"])

    def test_size_tolerance(self):
        official = gptoss.SPECS["gpt-oss-120b"]["gguf_size_bytes"]
        self.assertEqual(check(report(file_summary=good_summary(size_bytes=int(official * 1.04))), "size")["status"], "ok")
        self.assertEqual(check(report(file_summary=good_summary(size_bytes=int(official * 0.96))), "size")["status"], "ok")
        small = check(report(file_summary=good_summary(size_bytes=int(official * 0.8))), "size")
        self.assertEqual(small["status"], "fail")
        self.assertIn("incomplete", small["detail"])
        self.assertEqual(check(report(file_summary=good_summary(size_bytes=int(official * 1.3))), "size")["status"], "warn")
        self.assertEqual(check(report(file_summary=good_summary(size_bytes=0)), "size")["status"], "unknown")

    def test_engine_probe(self):
        self.assertEqual(check(report(probe={"supports": {"gpt-oss": True}}), "engine")["status"], "ok")
        bad = check(report(probe={"supports": {"gpt-oss": False}}), "engine")
        self.assertEqual(bad["status"], "fail")
        self.assertIn("predates", bad["detail"])
        self.assertEqual(check(report(probe={"supports": {"gpt-oss": None}}), "engine")["status"], "unknown")

    def test_free_disk_for_the_download(self):
        self.assertEqual(check(report(hw=hwmodel.mock_hardware(disk_free_gb=200)), "disk")["status"], "ok")
        self.assertEqual(check(report(hw=hwmodel.mock_hardware(disk_free_gb=62)), "disk")["status"], "warn")
        low = check(report(hw=hwmodel.mock_hardware(disk_free_gb=30)), "disk")
        self.assertEqual(low["status"], "fail")
        self.assertIn("30 GB free", low["detail"])
        hw = hwmodel.mock_hardware()
        hw["disks"] = []
        self.assertEqual(check(gptoss.feasibility_report(hw), "disk")["status"], "unknown")

    def test_context_limit(self):
        self.assertEqual(check(report(ctx=131072), "context")["status"], "ok")
        self.assertEqual(check(report(ctx=200000), "context")["status"], "fail")

    def test_ram_check_follows_the_planner(self):
        self.assertEqual(check(report(model="gpt-oss-20b"), "ram")["status"], "ok")
        self.assertEqual(check(report(), "ram")["status"], "warn")
        self.assertEqual(check(report("laptop-16gb"), "ram")["status"], "fail")


class RealFileSummaryTests(unittest.TestCase):
    def test_tiny_gguf_with_gpt_oss_keys_flows_through(self):
        d = Path(tempfile.mkdtemp())
        write_gguf(d / "gpt-oss-120b-MXFP4.gguf", {"general.architecture": "gpt-oss", "general.name": "gpt-oss-120b",
                                                   "general.file_type": 38, "gpt-oss.block_count": 36, "gpt-oss.expert_count": 128,
                                                   "gpt-oss.expert_used_count": 4}, 1000)
        s = gguf.summarize(d / "gpt-oss-120b-MXFP4.gguf")
        r = report(file_summary=s)
        self.assertEqual(check(r, "architecture")["status"], "ok")
        self.assertEqual(check(r, "experts")["status"], "ok")
        self.assertEqual(check(r, "quantization")["status"], "ok")
        self.assertEqual(check(r, "size")["status"], "fail")  # a 1 KB stand-in is far from 63.4 GB
        self.assertEqual(check(r, "file")["status"], "ok")
        self.assertEqual(r["plan"]["model"]["quant"], "MXFP4")

    def test_assumption_overrides_reach_the_plan(self):
        base = report()["plan"]["tiers"]["ram"]["budget_gb"]["value"]
        r = report(assumptions={"os_reserve_gb": 2.0})
        self.assertGreater(r["plan"]["tiers"]["ram"]["budget_gb"]["value"], base + 3.9)
        self.assertTrue(any(a["key"] == "os_reserve_gb" and a["value"] == 2.0 for a in r["assumptions"]))


if __name__ == "__main__":
    unittest.main()
