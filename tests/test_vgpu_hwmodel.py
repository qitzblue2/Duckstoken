import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from duckstoken import hardware
from duckstoken.vgpu import hwmodel

CPU_KEYS = {"name", "physical_cores", "logical_cores", "avx2", "avx512"}
RAM_KEYS = {"total_gb", "avail_gb", "speed_mts", "sticks", "bandwidth_gbs"}
GPU_KEYS = {"name", "vendor", "vram_mb", "vram_free_mb", "driver", "compute_cap"}
DISK_KEYS = {"drive", "model", "bus", "media", "kind", "size_gb", "free_gb"}


class MockHardwareTests(unittest.TestCase):
    def test_shape_matches_get_hardware(self):
        hw = hwmodel.mock_hardware()
        for k in ("os", "cpu", "ram", "gpus", "disks"):
            self.assertIn(k, hw)
        self.assertTrue(CPU_KEYS <= set(hw["cpu"]))
        self.assertTrue(RAM_KEYS <= set(hw["ram"]))
        self.assertTrue(GPU_KEYS <= set(hw["gpus"][0]))
        self.assertTrue(DISK_KEYS <= set(hw["disks"][0]))

    def test_defaults_are_the_owners_pc(self):
        hw = hwmodel.mock_hardware()
        self.assertEqual(hw["cpu"]["name"], "Intel Core i7-8700K")
        self.assertEqual((hw["cpu"]["physical_cores"], hw["cpu"]["logical_cores"]), (6, 12))
        self.assertEqual(hw["ram"]["total_gb"], 48)
        self.assertEqual(hw["ram"]["speed_mts"], 2133)
        self.assertEqual(hw["gpus"][0]["vram_mb"], 2048)
        self.assertEqual(hw["gpus"][0]["vendor"], "NVIDIA")
        self.assertEqual(hw["gpus"][0]["compute_cap"], "5.0")
        self.assertEqual(hw["disks"][0]["kind"], "nvme")

    def test_bandwidth_follows_hardware_helper(self):
        self.assertEqual(hwmodel.mock_hardware()["ram"]["bandwidth_gbs"], hardware.bandwidth_gbs(2133, 2))
        self.assertEqual(hwmodel.mock_hardware()["ram"]["bandwidth_gbs"], 34.1)
        one = hwmodel.mock_hardware(sticks=1)
        self.assertEqual(one["ram"]["bandwidth_gbs"], hardware.bandwidth_gbs(2133, 1))
        self.assertAlmostEqual(one["ram"]["bandwidth_gbs"] * 2, 34.1, delta=0.2)

    def test_no_gpu(self):
        self.assertEqual(hwmodel.mock_hardware(vram_mb=0)["gpus"], [])

    def test_marked_as_mock_and_available_ram(self):
        hw = hwmodel.mock_hardware(ram_gb=32)
        self.assertTrue(hw["mock"])
        self.assertLess(hw["ram"]["avail_gb"], 32)
        self.assertEqual(hwmodel.mock_hardware(ram_gb=32, ram_avail_gb=4)["ram"]["avail_gb"], 4)

    def test_vendor_detection(self):
        self.assertEqual(hwmodel.mock_hardware(gpu_name="AMD Radeon RX 7900 XT", vram_mb=20480)["gpus"][0]["vendor"], "AMD")
        self.assertEqual(hwmodel.mock_hardware(gpu_name="Intel Arc A770", vram_mb=16384)["gpus"][0]["vendor"], "Intel")
        amd = hwmodel.mock_hardware(gpu_name="AMD Radeon RX 7900 XT", vram_mb=20480)["gpus"][0]
        self.assertEqual(amd["compute_cap"], "")

    def test_disk_kind_sets_media(self):
        self.assertEqual(hwmodel.mock_hardware(disk_kind="hdd")["disks"][0]["media"], "HDD")
        self.assertEqual(hwmodel.mock_hardware(disk_kind="ssd")["disks"][0]["media"], "SSD")


class PresetTests(unittest.TestCase):
    def test_required_presets_exist(self):
        for name in ("duck-user", "duck-user-hdd", "laptop-16gb", "desktop-32gb-rtx3060", "workstation-128gb-rtx4090",
                     "no-gpu-8gb"):
            self.assertIn(name, hwmodel.PRESETS)
            hw = hwmodel.preset_hardware(name)
            self.assertEqual(hw["preset"], name)
            self.assertTrue(RAM_KEYS <= set(hw["ram"]))

    def test_duck_user_hdd_only_differs_by_disk(self):
        a, b = hwmodel.preset_hardware("duck-user"), hwmodel.preset_hardware("duck-user-hdd")
        self.assertEqual(a["ram"], b["ram"])
        self.assertEqual(a["gpus"], b["gpus"])
        self.assertEqual(b["disks"][0]["kind"], "hdd")
        self.assertGreater(b["disks"][0]["size_gb"], 3000)

    def test_preset_facts(self):
        self.assertEqual(hwmodel.preset_hardware("laptop-16gb")["ram"]["total_gb"], 16)
        self.assertEqual(hwmodel.preset_hardware("laptop-16gb")["gpus"], [])
        self.assertEqual(hwmodel.preset_hardware("desktop-32gb-rtx3060")["gpus"][0]["vram_mb"], 12288)
        self.assertEqual(hwmodel.preset_hardware("workstation-128gb-rtx4090")["ram"]["total_gb"], 128)
        self.assertEqual(hwmodel.preset_hardware("no-gpu-8gb")["ram"]["total_gb"], 8)

    def test_overrides_and_isolation(self):
        hw = hwmodel.preset_hardware("duck-user", ram_gb=64, vram_mb=1024)
        self.assertEqual(hw["ram"]["total_gb"], 64)
        self.assertEqual(hw["gpus"][0]["vram_mb"], 1024)
        self.assertEqual(hwmodel.PRESETS["duck-user"]["ram_gb"], 48)  # preset table untouched
        hw["ram"]["total_gb"] = 1
        self.assertEqual(hwmodel.preset_hardware("duck-user")["ram"]["total_gb"], 48)

    def test_unknown_preset(self):
        with self.assertRaises(KeyError):
            hwmodel.preset_hardware("nope")
        self.assertEqual(hwmodel.preset_names()[0], "duck-user")


if __name__ == "__main__":
    unittest.main()
