import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from duckstoken.vgpu import hwmodel, quant  # noqa: E402

GIB = 2**30
# A 7B-class dense model (Llama-2-7B shape) at Q4_K_M.
DENSE7B = {"name": "dense-7b", "architecture": "llama", "size_bytes": int(4.08 * GIB), "block_count": 32,
           "embedding_length": 4096, "head_count": 32, "head_count_kv": 32, "feed_forward_length": 11008,
           "context_length": 4096, "file_type": 15, "file_type_name": "Q4_K_M", "is_moe": False, "expert_count": 0,
           "expert_used_count": 0, "vocab_size": 32000}


class FtypeTests(unittest.TestCase):
    def test_ids_match_llama_h(self):
        # include/llama.h (commit 08ffba2)
        expect = {0: "F32", 1: "F16", 2: "Q4_0", 7: "Q8_0", 10: "Q2_K", 15: "Q4_K_M", 18: "Q6_K", 25: "IQ4_NL", 30: "IQ4_XS",
                  32: "BF16", 38: "MXFP4_MOE"}
        for k, v in expect.items():
            self.assertEqual(quant.FTYPE_NAMES[k], v)
        self.assertNotIn(5, quant.FTYPE_NAMES)  # removed upstream

    def test_ftype_name_handles_junk_and_guessed_flag(self):
        self.assertEqual(quant.ftype_name(15), "Q4_K_M")
        self.assertEqual(quant.ftype_name("15"), "Q4_K_M")
        self.assertEqual(quant.ftype_name(15 | quant.FTYPE_GUESSED), "Q4_K_M")
        self.assertIsNone(quant.ftype_name(None))
        self.assertIsNone(quant.ftype_name("x"))
        self.assertIsNone(quant.ftype_name(999))

    def test_quant_name(self):
        self.assertEqual(quant.quant_name({"file_type": 38}), "MXFP4")
        self.assertEqual(quant.quant_name({"file_type": 15}), "Q4_K_M")
        self.assertEqual(quant.quant_name({"file_type_name": "Q8_0"}), "Q8_0")
        self.assertIsNone(quant.quant_name({"file_type": None}))
        self.assertIsNone(quant.quant_name(None))

    def test_every_ftype_with_bits_has_quality_text(self):
        for name, rec in quant.QUANTS.items():
            self.assertGreater(rec["bits_per_weight"], 1.0, name)
            self.assertTrue(rec["qualitative"], name)
            self.assertTrue(rec["backend_support"]["cpu"], name)
            self.assertNotRegex(rec["qualitative"], r"\d+(\.\d+)?\s*%", name)  # no invented numeric quality claims

    def test_bits_follow_llama_cpp_readme(self):
        self.assertAlmostEqual(quant.bits_per_weight("Q4_K_M"), 4.8944, places=3)
        self.assertAlmostEqual(quant.bits_per_weight("Q8_0"), 8.5008, places=3)
        self.assertAlmostEqual(quant.bits_per_weight("MXFP4_MOE"), 4.25)  # 17 bytes per 32 values
        self.assertEqual(quant.QUANTS["MXFP4"], quant.QUANTS["MXFP4_MOE"])
        self.assertIsNone(quant.bits_per_weight("nope"))
        self.assertEqual(quant.bits_per_weight("nope", 9.0), 9.0)

    def test_ternary_has_no_cuda_support(self):
        self.assertFalse(quant.QUANTS["TQ1_0"]["backend_support"]["cuda"])
        self.assertTrue(quant.QUANTS["Q4_K_M"]["backend_support"]["cuda"])


class ConversionTests(unittest.TestCase):
    def test_gguf_needs_no_conversion(self):
        self.assertFalse(quant.needs_conversion("C:/models/x.GGUF"))

    def test_checkpoints_need_conversion(self):
        for f in ("m.safetensors", "pytorch_model.bin", "m.pt", "weights.ckpt"):
            self.assertTrue(quant.needs_conversion(f), f)

    def test_directories(self):
        with tempfile.TemporaryDirectory() as d:
            hf = Path(d) / "hf"
            hf.mkdir()
            (hf / "config.json").write_text("{}")
            (hf / "model.safetensors").write_bytes(b"x")
            self.assertTrue(quant.needs_conversion(hf))
            gg = Path(d) / "gg"
            gg.mkdir()
            (gg / "a.gguf").write_bytes(b"x")
            self.assertFalse(quant.needs_conversion(gg))

    def test_unknown_format_is_not_claimed_loadable(self):
        self.assertTrue(quant.needs_conversion("mystery.dat"))


class ParameterCountTests(unittest.TestCase):
    def test_from_dict(self):
        params, how = quant.parameter_count(DENSE7B)
        self.assertAlmostEqual(params / 1e9, 4.08 * GIB * 8 / 4.8944 / 1e9, places=3)
        self.assertIn("4.8944", how)

    def test_from_number(self):
        self.assertEqual(quant.parameter_count(7e9), (7e9, "given"))

    def test_unknown(self):
        self.assertEqual(quant.parameter_count({"size_bytes": 5})[0], 0.0)
        self.assertEqual(quant.parameter_count({"file_type": 15})[0], 0.0)
        self.assertEqual(quant.parameter_count(None)[0], 0.0)


class CompareQuantsTests(unittest.TestCase):
    def test_rows_and_ordering(self):
        hw = hwmodel.preset_hardware("laptop-16gb")
        rows = quant.compare_quants(DENSE7B, hw, ctx=4096)
        self.assertEqual([r["quant"] for r in rows], quant.COMPARE_ORDER)
        sizes = [r["file_gb"]["value"] for r in rows]
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        for r in rows:
            self.assertEqual(r["file_gb"]["status"], "estimated")
            self.assertTrue(r["supported_directly"])
            self.assertIn(r["verdict"], ("ok", "tight", "paging", "reject"))
            self.assertEqual(r["fits"], r["verdict"] in ("ok", "tight"))

    def test_current_quant_needs_no_conversion_others_do(self):
        rows = {r["quant"]: r for r in quant.compare_quants(DENSE7B, hwmodel.preset_hardware("laptop-16gb"))}
        self.assertFalse(rows["Q4_K_M"]["conversion_required"])
        self.assertTrue(rows["Q8_0"]["conversion_required"])
        self.assertIn("llama-quantize", rows["Q8_0"]["how_to_get"])
        self.assertAlmostEqual(rows["Q4_K_M"]["file_gb"]["value"], 4.08, delta=0.02)

    def test_f16_does_not_fit_8gb_but_q4_does(self):
        rows = {r["quant"]: r for r in quant.compare_quants(DENSE7B, hwmodel.preset_hardware("no-gpu-8gb"))}
        self.assertFalse(rows["F16"]["fits"])
        self.assertEqual(rows["F16"]["verdict"], "reject")
        self.assertGreater(rows["F16"]["file_gb"]["value"], 12)

    def test_parameter_number_input(self):
        rows = quant.compare_quants(7e9, hwmodel.preset_hardware("laptop-16gb"), names=["Q4_K_M", "Q8_0"])
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["file_gb"]["value"], 7e9 * 4.8944 / 8 / GIB, delta=0.01)
        self.assertTrue(all(r["conversion_required"] for r in rows))  # no current file to compare with

    def test_no_numeric_quality_claims(self):
        for r in quant.compare_quants(DENSE7B, hwmodel.preset_hardware("laptop-16gb")):
            self.assertNotRegex(r["quality_note"], r"\b\d+(\.\d+)?\s*%")

    def test_empty_when_parameters_unknown(self):
        self.assertEqual(quant.compare_quants({"size_bytes": 10}, hwmodel.preset_hardware("laptop-16gb")), [])

    def test_does_not_mutate_input(self):
        before = dict(DENSE7B)
        quant.compare_quants(DENSE7B, hwmodel.preset_hardware("laptop-16gb"))
        self.assertEqual(DENSE7B, before)


if __name__ == "__main__":
    unittest.main()
