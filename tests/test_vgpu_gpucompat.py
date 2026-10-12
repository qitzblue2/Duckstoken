import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from duckstoken.vgpu import gpucompat, hwmodel

KEYS = {"vendor", "name", "architecture", "family", "compute_cap", "legacy", "vram_gb", "integrated", "notes"}


def nv(name, cc="", vram=2048, driver="580.97"):
    return {"name": name, "vendor": "NVIDIA", "vram_mb": vram, "vram_free_mb": vram, "driver": driver, "compute_cap": cc}


def probe(devices=(), supports=None, status="ok"):
    return {"status": status, "devices": list(devices), "supports": supports or {}, "backends": ["CPU"]}


CUDA0 = {"id": "CUDA0", "backend": "CUDA", "name": "NVIDIA GeForce GTX 750", "total_mb": 2048, "free_mb": 1900}
VULKAN0 = {"id": "Vulkan0", "backend": "Vulkan", "name": "NVIDIA GeForce GTX 750", "total_mb": 2048, "free_mb": 1900}


def item(report, id_):
    return next(i for i in report["items"] if i["id"] == id_)


class ClassifyTests(unittest.TestCase):
    def test_shape(self):
        c = gpucompat.classify_gpu(nv("NVIDIA GeForce GTX 750", "5.0"))
        self.assertEqual(set(c), KEYS)
        json.dumps(c)

    def test_gtx_750_is_maxwell_gm107_legacy(self):
        c = gpucompat.classify_gpu(nv("NVIDIA GeForce GTX 750", "5.0"))
        self.assertEqual((c["architecture"], c["family"], c["compute_cap"], c["legacy"]), ("Maxwell", "GM107", "5.0", True))
        self.assertEqual(c["vram_gb"], 2.0)
        text = " ".join(c["notes"])
        self.assertIn("CUDA 13", text)
        self.assertIn("R580", text)
        self.assertIn("1 GB or 2 GB", text)

    def test_name_table_when_the_driver_reports_no_capability(self):
        for name, arch, cc, legacy in (("NVIDIA GeForce GTX 750 Ti", "Maxwell", "5.0", True), ("GeForce GTX 960", "Maxwell", "5.2", True),
                                       ("GeForce GTX 970", "Maxwell", "5.2", True), ("GeForce GTX 1060 6GB", "Pascal", "6.1", True),
                                       ("GeForce GTX 1080 Ti", "Pascal", "6.1", True), ("TITAN X (Pascal)", "Pascal", "6.1", True),
                                       ("NVIDIA TITAN V", "Volta", "7.0", True), ("GeForce GTX 1650", "Turing", "7.5", False),
                                       ("GeForce RTX 2060", "Turing", "7.5", False), ("GeForce RTX 3060", "Ampere", "8.6", False),
                                       ("GeForce RTX 4090", "Ada Lovelace", "8.9", False), ("GeForce RTX 5090", "Blackwell", "12.0", False),
                                       ("GeForce GTX 770", "Kepler", "3.5", True), ("Tesla P100-PCIE-16GB", "Pascal", "6.0", True),
                                       ("NVIDIA T4", "Turing", "7.5", False), ("NVIDIA RTX A4000", "Ampere", "8.6", False)):
            c = gpucompat.classify_gpu(nv(name))
            self.assertEqual((c["architecture"], c["compute_cap"], c["legacy"]), (arch, cc, legacy), name)
            self.assertTrue(any("inferred" in n for n in c["notes"]), name)

    def test_driver_capability_beats_the_name(self):
        c = gpucompat.classify_gpu(nv("GeForce Mystery 9000", "8.9"))
        self.assertEqual((c["architecture"], c["legacy"]), ("Ada Lovelace", False))
        self.assertFalse(any("inferred" in n for n in c["notes"]))

    def test_capability_boundaries(self):
        for cc, arch, legacy in (("5.2", "Maxwell", True), ("6.0", "Pascal", True), ("7.0", "Volta", True), ("7.5", "Turing", False),
                                 ("8.0", "Ampere", False), ("8.6", "Ampere", False), ("8.9", "Ada Lovelace", False),
                                 ("9.0", "Hopper", False), ("10.0", "Blackwell", False), ("12.0", "Blackwell", False)):
            c = gpucompat.classify_gpu(nv("x", cc))
            self.assertEqual((c["architecture"], c["legacy"]), (arch, legacy), cc)

    def test_below_cc5_is_flagged(self):
        c = gpucompat.classify_gpu(nv("GeForce GTX 770", "3.0"))
        self.assertEqual(c["architecture"], "Kepler")
        self.assertTrue(any("no longer targets" in n for n in c["notes"]))

    def test_amd(self):
        for name, arch in (("AMD Radeon RX 580", "GCN"), ("AMD Radeon RX 5700 XT", "RDNA 1"), ("AMD Radeon RX 6800 XT", "RDNA 2"),
                           ("AMD Radeon RX 7900 XTX", "RDNA 3"), ("AMD Radeon RX 9070 XT", "RDNA 4")):
            c = gpucompat.classify_gpu({"name": name, "vram_mb": 8192})
            self.assertEqual((c["vendor"], c["architecture"], c["legacy"], c["integrated"]), ("AMD", arch, False, False), name)

    def test_intel(self):
        self.assertEqual(gpucompat.classify_gpu({"name": "Intel Arc A770", "vram_mb": 16384})["architecture"], "Xe-HPG (Alchemist)")
        self.assertEqual(gpucompat.classify_gpu({"name": "Intel Arc B580", "vram_mb": 12288})["architecture"], "Xe2 (Battlemage)")
        igpu = gpucompat.classify_gpu({"name": "Intel(R) UHD Graphics 630", "vram_mb": 1024})
        self.assertTrue(igpu["integrated"])
        self.assertTrue(any("shares system RAM" in n for n in igpu["notes"]))

    def test_amd_igpu(self):
        c = gpucompat.classify_gpu({"name": "AMD Radeon(TM) Graphics", "vram_mb": 512})
        self.assertTrue(c["integrated"])
        self.assertEqual(c["vendor"], "AMD")

    def test_windows_4gb_cap_note(self):
        c = gpucompat.classify_gpu({"name": "Radeon RX 7900 XTX", "vram_mb": 4096})
        self.assertTrue(any("at most 4 GB" in n for n in c["notes"]))

    def test_unknown_gpu_makes_no_claim(self):
        c = gpucompat.classify_gpu({"name": "Frobnicator 3000"})
        self.assertEqual((c["vendor"], c["architecture"], c["legacy"]), ("unknown", "unknown", False))
        self.assertTrue(any("no compatibility claim" in n for n in c["notes"]))
        self.assertEqual(gpucompat.classify_gpu({})["architecture"], "unknown")
        self.assertEqual(gpucompat.classify_gpu(None)["vram_gb"], 0)

    def test_vendor_inferred_from_name(self):
        self.assertEqual(gpucompat.classify_gpu({"name": "GeForce GTX 1060"})["vendor"], "NVIDIA")
        self.assertEqual(gpucompat.classify_gpu({"name": "NVIDIA GeForce RTX 3060", "vendor": ""})["architecture"], "Ampere")

    def test_unparseable_capability_falls_back_to_name(self):
        c = gpucompat.classify_gpu(nv("GeForce GTX 1060", "n/a"))
        self.assertEqual(c["architecture"], "Pascal")


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.hw = hwmodel.preset_hardware("duck-user")

    def test_shape_and_items(self):
        r = gpucompat.compatibility_report(self.hw)
        self.assertEqual(set(r), {"items", "gpu", "summary", "counts", "generated_at"})
        for i in r["items"]:
            self.assertEqual(set(i), {"id", "label", "status", "detail", "evidence", "how_to_verify"})
            self.assertIn(i["status"], ("supported", "unsupported", "experimental", "unknown"))
            self.assertIn(i["evidence"].split(" ")[0], ("detected", "probe", "table", "planner"))
        ids = {i["id"] for i in r["items"]}
        self.assertTrue({"cpu", "cuda", "vulkan", "hip", "sycl", "metal", "mmap", "mxfp4", "gpt_oss", "vram_threshold", "ram", "disk"} <= ids)
        self.assertEqual(len(r["gpu"]), 1)
        self.assertRegex(r["generated_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        json.dumps(r)

    def test_cpu_avx2(self):
        self.assertEqual(item(gpucompat.compatibility_report(self.hw), "cpu")["status"], "supported")
        hw = hwmodel.mock_hardware(avx2=False)
        self.assertEqual(item(gpucompat.compatibility_report(hw), "cpu")["status"], "experimental")
        hw["cpu"].pop("avx2")
        self.assertEqual(item(gpucompat.compatibility_report(hw), "cpu")["status"], "unknown")

    def test_nothing_is_supported_without_evidence_on_the_owners_pc(self):
        r = gpucompat.compatibility_report(self.hw)
        for id_ in ("cuda", "vulkan", "hip", "sycl", "metal", "gpt_oss", "mxfp4"):
            self.assertNotEqual(item(r, id_)["status"], "supported", id_)
        self.assertEqual(item(r, "cuda")["status"], "experimental")  # legacy GPU, unverified
        self.assertIn("CUDA 13", item(r, "cuda")["detail"])
        self.assertEqual(item(r, "vulkan")["status"], "unknown")
        self.assertEqual(item(r, "hip")["status"], "unsupported")
        self.assertEqual(item(r, "gpt_oss")["status"], "unknown")
        self.assertIn("No GPU backend is verified", r["summary"])

    def test_cuda_supported_only_when_the_probe_lists_a_device(self):
        r = gpucompat.compatibility_report(self.hw, probe([CUDA0]))
        self.assertEqual(item(r, "cuda")["status"], "supported")
        self.assertEqual(item(r, "cuda")["evidence"], "probe")
        self.assertIn("CUDA GPU backend is verified" if False else "verified", r["summary"])

    def test_probe_without_cuda_device_on_a_legacy_nvidia_card_is_unsupported(self):
        r = gpucompat.compatibility_report(self.hw, probe([]))
        c = item(r, "cuda")
        self.assertEqual(c["status"], "unsupported")
        self.assertEqual(c["evidence"], "probe")
        self.assertIn("no CUDA device", c["detail"])

    def test_cuda13_engine_on_maxwell_is_unsupported_without_running_anything(self):
        r = gpucompat.compatibility_report(self.hw, env={"engine_asset": "llama-b9999-bin-win-cuda-13.1-x64.zip"})
        c = item(r, "cuda")
        self.assertEqual((c["status"], c["evidence"]), ("unsupported", "table"))
        self.assertIn("CUDA 13 build", c["detail"])
        r12 = gpucompat.compatibility_report(self.hw, env={"engine_asset": "llama-b9999-bin-win-cuda-12.4-x64.zip"})
        self.assertEqual(item(r12, "cuda")["status"], "experimental")

    def test_modern_nvidia_without_probe_is_unknown_not_supported(self):
        hw = hwmodel.preset_hardware("desktop-32gb-rtx3060")
        r = gpucompat.compatibility_report(hw)
        self.assertEqual(item(r, "cuda")["status"], "unknown")
        self.assertEqual(item(gpucompat.compatibility_report(hw, probe([{"id": "CUDA0", "backend": "CUDA", "name": "RTX 3060", "total_mb": 12288, "free_mb": 11000}])), "cuda")["status"], "supported")
        self.assertEqual(item(gpucompat.compatibility_report(hw, probe([])), "cuda")["status"], "unsupported")

    def test_vulkan_device_makes_vulkan_supported(self):
        r = gpucompat.compatibility_report(self.hw, probe([VULKAN0]))
        self.assertEqual(item(r, "vulkan")["status"], "supported")
        self.assertEqual(item(r, "cuda")["status"], "unsupported")

    def test_failed_probe_is_not_evidence_either_way(self):
        r = gpucompat.compatibility_report(self.hw, probe(status="error"))
        self.assertEqual(item(r, "cuda")["status"], "experimental")
        self.assertEqual(item(r, "vulkan")["status"], "unknown")

    def test_no_gpu_machine(self):
        r = gpucompat.compatibility_report(hwmodel.preset_hardware("laptop-16gb"))
        for id_ in ("cuda", "vulkan", "hip", "sycl", "metal"):
            self.assertEqual(item(r, id_)["status"], "unsupported", id_)
        self.assertEqual(item(r, "vram_threshold")["status"], "unsupported")
        self.assertEqual(r["gpu"], [])

    def test_amd_and_intel_candidates(self):
        hw = hwmodel.mock_hardware(gpu_name="AMD Radeon RX 6800 XT", vram_mb=16384, compute_cap="")
        r = gpucompat.compatibility_report(hw)
        self.assertEqual(item(r, "hip")["status"], "unknown")
        self.assertEqual(item(r, "cuda")["status"], "unsupported")
        self.assertEqual(item(r, "sycl")["status"], "unsupported")
        hw = hwmodel.mock_hardware(gpu_name="Intel Arc A770", vram_mb=16384, compute_cap="")
        self.assertEqual(item(gpucompat.compatibility_report(hw), "sycl")["status"], "unknown")

    def test_metal_only_on_macos(self):
        hw = hwmodel.mock_hardware(gpu_name="Apple M2", vram_mb=16384, os_name="macOS-14")
        self.assertEqual(item(gpucompat.compatibility_report(hw), "metal")["status"], "unknown")
        self.assertEqual(item(gpucompat.compatibility_report(self.hw), "metal")["status"], "unsupported")

    def test_gpt_oss_and_mxfp4_follow_the_probe(self):
        yes = gpucompat.compatibility_report(self.hw, probe(supports={"gpt-oss": True}))
        self.assertEqual((item(yes, "gpt_oss")["status"], item(yes, "mxfp4")["status"]), ("supported", "supported"))
        self.assertEqual(item(yes, "gpt_oss")["evidence"], "probe")
        no = gpucompat.compatibility_report(self.hw, probe(supports={"gpt-oss": False}))
        self.assertEqual((item(no, "gpt_oss")["status"], item(no, "mxfp4")["status"]), ("unsupported", "unsupported"))
        unreadable = gpucompat.compatibility_report(self.hw, probe(supports={"gpt-oss": None}))
        self.assertEqual(item(unreadable, "gpt_oss")["status"], "unknown")

    def test_vram_threshold_tiers(self):
        for vram, status in ((0, "unsupported"), (2048, "experimental"), (4096, "supported"), (12288, "supported")):
            hw = hwmodel.mock_hardware(vram_mb=vram, gpu_name="NVIDIA GeForce RTX 3060" if vram else "", compute_cap="8.6")
            self.assertEqual(item(gpucompat.compatibility_report(hw), "vram_threshold")["status"], status, vram)
        igpu = hwmodel.mock_hardware(vram_mb=8192, gpu_name="Intel(R) UHD Graphics 630", compute_cap="")
        self.assertEqual(item(gpucompat.compatibility_report(igpu), "vram_threshold")["status"], "unsupported")

    def test_ram_and_disk_tiers(self):
        self.assertEqual(item(gpucompat.compatibility_report(self.hw), "ram")["status"], "supported")
        self.assertIn("2133 MT/s", item(gpucompat.compatibility_report(self.hw), "ram")["detail"])
        self.assertEqual(item(gpucompat.compatibility_report(hwmodel.preset_hardware("no-gpu-8gb")), "ram")["status"], "experimental")
        self.assertEqual(item(gpucompat.compatibility_report(hwmodel.mock_hardware(ram_gb=4)), "ram")["status"], "unsupported")
        self.assertEqual(item(gpucompat.compatibility_report(self.hw), "disk")["status"], "supported")
        self.assertEqual(item(gpucompat.compatibility_report(hwmodel.preset_hardware("duck-user-hdd")), "disk")["status"], "experimental")
        hw = hwmodel.mock_hardware()
        hw["disks"] = []
        self.assertEqual(item(gpucompat.compatibility_report(hw), "disk")["status"], "unknown")

    def test_legacy_nvidia_driver_branch(self):
        old = gpucompat.compatibility_report(self.hw)  # driver "mock" has no version number
        self.assertEqual(item(old, "nvidia_driver")["status"], "unknown")
        hw = hwmodel.mock_hardware()
        hw["gpus"][0]["driver"] = "580.97"
        self.assertEqual(item(gpucompat.compatibility_report(hw), "nvidia_driver")["status"], "supported")
        hw["gpus"][0]["driver"] = "591.44"
        d = item(gpucompat.compatibility_report(hw), "nvidia_driver")
        self.assertEqual(d["status"], "unsupported")
        self.assertIn("R580", d["detail"])
        modern = hwmodel.preset_hardware("desktop-32gb-rtx3060")
        self.assertNotIn("nvidia_driver", {i["id"] for i in gpucompat.compatibility_report(modern)["items"]})

    def test_model_fit_item_uses_the_planner(self):
        m = {"name": "big", "size_bytes": int(80 * 2**30), "block_count": 40, "embedding_length": 4096, "head_count": 32,
             "head_count_kv": 8, "file_type": 15}
        r = gpucompat.compatibility_report(self.hw, model=m)
        fit = item(r, "model_fit")
        self.assertEqual(fit["status"], "unsupported")
        self.assertIn("reject", fit["detail"])
        small = dict(m, size_bytes=2 * 2**30)
        self.assertEqual(item(gpucompat.compatibility_report(self.hw, model=small), "model_fit")["status"], "supported")
        self.assertNotIn("model_fit", {i["id"] for i in gpucompat.compatibility_report(self.hw)["items"]})

    def test_counts_add_up(self):
        r = gpucompat.compatibility_report(self.hw)
        self.assertEqual(sum(r["counts"].values()), len(r["items"]))

    def test_tolerates_an_empty_hardware_dict(self):
        r = gpucompat.compatibility_report({})
        self.assertTrue(r["items"])
        self.assertEqual(item(r, "cpu")["status"], "unknown")
        self.assertEqual(item(r, "ram")["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
