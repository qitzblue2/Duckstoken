"""What-if hardware: build a `hardware.get_hardware()`-shaped dict for a machine that is not the one we run on.

Used for tests, for "what would this model do on a 16 GB laptop?" in the UI and for the GPT-OSS feasibility table.
The dicts carry ``"mock": True`` and ``"preset"`` so every consumer can label numbers derived from them as estimates
instead of measurements.
"""

from __future__ import annotations

import copy

from .. import hardware


def mock_hardware(ram_gb: float = 48, ram_speed_mts: int = 2133, sticks: int = 2, vram_mb: int = 2048,
                  gpu_name: str = "NVIDIA GeForce GTX 750", compute_cap: str = "5.0", cores: int = 6,
                  logical: int = 12, avx2: bool = True, avx512: bool = False, disk_kind: str = "nvme",
                  disk_free_gb: float = 200, disk_size_gb: float = 256, drive: str = "C:",
                  cpu_name: str = "Intel Core i7-8700K", os_name: str = "Windows-11",
                  ram_avail_gb: float | None = None, preset: str = "") -> dict:
    """A hardware dict in exactly the shape of `hardware.get_hardware()` (plus ``mock``/``preset`` markers).

    ``vram_mb=0`` means no GPU (``gpus == []``). ``ram_avail_gb`` defaults to 85% of ``ram_gb``.
    """
    bandwidth = hardware.bandwidth_gbs(ram_speed_mts, 2 if sticks != 1 else 1)
    gpus = []
    if vram_mb > 0:
        vendor = ("NVIDIA" if "nvidia" in gpu_name.lower() or "geforce" in gpu_name.lower() or "rtx" in gpu_name.lower()
                  else "AMD" if "radeon" in gpu_name.lower() or "amd" in gpu_name.lower()
                  else "Intel" if "intel" in gpu_name.lower() or "arc" in gpu_name.lower() else "")
        gpus.append({"name": gpu_name, "vendor": vendor, "vram_mb": int(vram_mb), "vram_free_mb": int(vram_mb * 0.9),
                     "driver": "mock", "compute_cap": compute_cap if vendor == "NVIDIA" else ""})
    return {
        "os": os_name,
        "cpu": {"name": cpu_name, "physical_cores": int(cores), "logical_cores": int(logical),
                "avx2": bool(avx2), "avx512": bool(avx512)},
        "ram": {"total_gb": float(ram_gb),
                "avail_gb": round(float(ram_avail_gb if ram_avail_gb is not None else ram_gb * 0.85), 1),
                "speed_mts": int(ram_speed_mts), "sticks": int(sticks), "bandwidth_gbs": bandwidth},
        "gpus": gpus,
        "disks": [{"drive": drive, "model": "mock", "bus": "NVMe" if disk_kind == "nvme" else "SATA",
                   "media": "SSD" if disk_kind in ("nvme", "ssd") else "HDD", "kind": disk_kind,
                   "size_gb": float(disk_size_gb), "free_gb": float(disk_free_gb)}],
        "mock": True,
        "preset": preset,
    }


# kwargs for mock_hardware. Facts about "duck-user" come from the project owner: i7-8700K, 48 GB DDR4-2133 in two
# sticks, a GTX 750 and the models on a 256 GB NVMe drive. The GTX 750 shipped with 1 GB or 2 GB; the 2 GB card is assumed.
PRESETS: dict[str, dict] = {
    "duck-user": dict(ram_gb=48, ram_speed_mts=2133, sticks=2, vram_mb=2048, gpu_name="NVIDIA GeForce GTX 750",
                      compute_cap="5.0", cores=6, logical=12, avx2=True, avx512=False, disk_kind="nvme",
                      disk_free_gb=200, disk_size_gb=256, drive="C:", cpu_name="Intel Core i7-8700K"),
    "duck-user-hdd": dict(ram_gb=48, ram_speed_mts=2133, sticks=2, vram_mb=2048, gpu_name="NVIDIA GeForce GTX 750",
                          compute_cap="5.0", cores=6, logical=12, avx2=True, avx512=False, disk_kind="hdd",
                          disk_free_gb=3000, disk_size_gb=4000, drive="E:", cpu_name="Intel Core i7-8700K"),
    "laptop-16gb": dict(ram_gb=16, ram_speed_mts=3200, sticks=2, vram_mb=0, gpu_name="", compute_cap="", cores=4,
                        logical=8, avx2=True, avx512=False, disk_kind="nvme", disk_free_gb=120, disk_size_gb=512,
                        drive="C:", cpu_name="Generic 4-core laptop CPU"),
    "desktop-32gb-rtx3060": dict(ram_gb=32, ram_speed_mts=3200, sticks=2, vram_mb=12288,
                                 gpu_name="NVIDIA GeForce RTX 3060", compute_cap="8.6", cores=8, logical=16,
                                 avx2=True, avx512=False, disk_kind="nvme", disk_free_gb=400, disk_size_gb=1000,
                                 drive="C:", cpu_name="Generic 8-core desktop CPU"),
    "workstation-128gb-rtx4090": dict(ram_gb=128, ram_speed_mts=5600, sticks=4, vram_mb=24576,
                                      gpu_name="NVIDIA GeForce RTX 4090", compute_cap="8.9", cores=16, logical=32,
                                      avx2=True, avx512=True, disk_kind="nvme", disk_free_gb=1500,
                                      disk_size_gb=2000, drive="C:", cpu_name="Generic 16-core workstation CPU"),
    "no-gpu-8gb": dict(ram_gb=8, ram_speed_mts=2666, sticks=1, vram_mb=0, gpu_name="", compute_cap="", cores=4,
                       logical=4, avx2=True, avx512=False, disk_kind="ssd", disk_free_gb=60, disk_size_gb=256,
                       drive="C:", cpu_name="Generic 4-core CPU"),
}


def preset_hardware(name: str, **overrides) -> dict:
    """`mock_hardware` for a named preset, with optional keyword overrides. Unknown names raise KeyError."""
    if name not in PRESETS:
        raise KeyError(f"unknown hardware preset {name!r}; choose one of {', '.join(PRESETS)}")
    kwargs = copy.deepcopy(PRESETS[name])
    kwargs.update(overrides)
    return mock_hardware(preset=name, **kwargs)


def preset_names() -> list[str]:
    return list(PRESETS)
