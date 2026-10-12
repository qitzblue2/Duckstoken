"""Turn hardware + model facts into honest speed estimates and a sensible settings patch.

All numbers here are heuristics. They exist to steer people away from setups that cannot work
(e.g. a 63 GB model paging from a hard drive) - the real tokens/s is shown in the chat footer.
"""

from __future__ import annotations

import threading

from . import hardware

_planning = threading.local()  # set while the memory planner is asked for a placement: it calls recommend() again

OS_RESERVE_GB = 6.0  # Windows + browser + friends

# Effective throughput (GB/s) when llama.cpp mmap-faults weight pages in from each drive type.
DISK_GBS = {"nvme": 0.9, "ssd": 0.25, "hdd": 0.02, "unknown": 0.1}


def _active_gb(model: dict) -> float:
    """Approximate bytes of weights touched to produce one token."""
    size = model["size_gb"]
    if model["is_moe"] and model["expert_count"]:
        frac = max(model["expert_used_count"] / model["expert_count"], 0.02)
        return max(size * frac * 1.5, 0.3)
    return size


def feasibility(hw: dict, model: dict, model_path: str | None = None) -> dict:
    ram = hw["ram"]
    gpus = hw["gpus"]
    vram_gb = max((g["vram_mb"] for g in gpus), default=0) / 1024
    usable_ram = max(ram["total_gb"] - OS_RESERVE_GB, 1)
    usable_vram = max(vram_gb - 0.4, 0) if vram_gb >= 1 else 0
    avail = usable_ram + usable_vram
    size = model["size_gb"]
    active = _active_gb(model)
    bw = ram["bandwidth_gbs"] * 0.6
    warnings: list[str] = []
    tips: list[str] = []

    disk = hardware.disk_for_path(model_path, hw["disks"]) if model_path else None
    kind = (disk or {}).get("kind", "unknown")
    drive = (disk or {}).get("drive", "?")

    fits = size <= avail
    if fits:
        ceiling = bw / active
        tps = (round(ceiling * 0.35, 1), round(ceiling * 0.7, 1))
        verdict = "good" if tps[1] >= 5 else "ok"
        summary = (f"The {size:.0f} GB model fits in memory ({usable_ram:.0f} GB usable RAM"
                   f"{' + GPU' if usable_vram else ''}). Expect roughly {tps[0]}-{tps[1]} tokens/s.")
        if kind == "hdd":
            warnings.append(f"The model is on {drive} (hard drive). First load will be slow (several minutes) "
                            "but it will run from RAM afterwards.")
    else:
        overflow = size - avail
        miss = min(0.9, max(0.05, overflow / size * 0.8))
        disk_gbs = DISK_GBS.get(kind, DISK_GBS["unknown"])
        sec_per_tok = active * miss / disk_gbs + active / bw
        tps_mid = 1 / sec_per_tok
        tps = (round(tps_mid * 0.6, 2), round(tps_mid * 1.4, 2))
        verdict = "slow" if tps[1] >= 0.4 else "unusable"
        summary = (f"The {size:.0f} GB model is {overflow:.0f} GB bigger than your usable memory "
                   f"({avail:.0f} GB), so part of it is read from disk while generating. "
                   f"On a {kind.upper() if kind != 'unknown' else 'this'} drive expect roughly "
                   f"{tps[0]}-{tps[1]} tokens/s.")
        if kind == "hdd":
            warnings.append(f"{drive} is a hard drive - random reads are ~50x slower than an SSD, so replies "
                            "would take minutes per word. Move the model to your SSD (the 256 GB M.2 is big enough).")
        elif kind in ("ssd", "nvme"):
            tips.append("This is the best you can do without more RAM: keep context small and close other apps.")
        if model["is_moe"]:
            tips.append("Mixture-of-Experts models only read a few experts per token, which is why this still works at all.")
        tips.append("A smaller model (e.g. gpt-oss-20b, ~12 GB) will be 10-30x faster on this machine.")

    free = (disk or {}).get("free_gb")
    if free is not None and free < size * 1.05:
        warnings.append(f"Only {free:.0f} GB free on {drive}; the model needs {size:.0f} GB.")
    if hw["cpu"]["avx2"] is False:
        warnings.append("This CPU has no AVX2 - llama.cpp will be very slow.")
    if ram["speed_mts"] and ram["speed_mts"] <= 2200:
        tips.append(f"RAM runs at {ram['speed_mts']} MT/s. Enabling XMP/DOCP in BIOS (if your kit supports it) "
                    "directly raises tokens/s because generation is memory-bandwidth bound.")
    if ram["sticks"] == 1:
        tips.append("Only one RAM stick detected - dual channel would roughly double speed.")
    if 0 < vram_gb < 3:
        tips.append(f"Your GPU has only ~{vram_gb:.0f} GB VRAM, so it helps little. The CPU does most of the work "
                    "(that is normal for this setup).")

    return {"fits_in_memory": fits, "verdict": verdict, "summary": summary, "tps_range": tps,
            "usable_ram_gb": round(usable_ram, 1), "usable_vram_gb": round(usable_vram, 1),
            "model_gb": size, "active_gb_per_token": round(active, 2), "drive": drive, "drive_kind": kind,
            "warnings": warnings, "tips": tips}


def resolve_backend(hw: dict, preference: str = "auto") -> str:
    """Pick cpu / cuda / vulkan. Cards under ~3 GB VRAM are not worth the extra moving parts."""
    if preference != "auto":
        return preference
    gpus = hw["gpus"]
    vram_gb = max((g["vram_mb"] for g in gpus), default=0) / 1024
    if vram_gb < 3:
        return "cpu"
    return "cuda" if any(g.get("vendor") == "NVIDIA" for g in gpus) else "vulkan"


def recommend(hw: dict, model: dict | None, backend: str = "auto", placement: bool = True) -> dict:
    """Return {'values': {...settings patch}, 'notes': [...]}."""
    cpu, ram, gpus = hw["cpu"], hw["ram"], hw["gpus"]
    vram_gb = max((g["vram_mb"] for g in gpus), default=0) / 1024
    notes: list[str] = []
    v: dict = {"threads": cpu["physical_cores"], "threads_batch": cpu["logical_cores"]}
    notes.append(f"Threads: {cpu['physical_cores']} generation / {cpu['logical_cores']} prompt "
                 "(physical cores are best for token generation).")

    use_gpu = backend != "cpu" and vram_gb >= 1
    size = model["size_gb"] if model else 0
    fits_ram = size and size <= max(ram["total_gb"] - OS_RESERVE_GB, 1)

    placement = placement and not getattr(_planning, "busy", False)  # the planner's own defaults must not recurse into it
    ctx_budget = 0
    if model:
        ctx_budget = 4096 if not fits_ram else (8192 if ram["total_gb"] < 32 else 16384)
        if model["context_length"]:
            ctx_budget = min(ctx_budget, model["context_length"])

    if not use_gpu:
        v.update(n_gpu_layers=0, n_cpu_moe=0, kv_offload=False)
        notes.append("No usable GPU backend -> running everything on the CPU.")
    else:
        v.update(n_gpu_layers=99, kv_offload=vram_gb >= 6)
        if model and placement:
            # Ask the memory planner for a layout that really fits this card (it knows the model's real per-layer expert
            # size); a fixed "layers per GB" rule put twice the VRAM of gpt-oss-120b on the card.
            _planning.busy = True
            try:
                from .vgpu import planner  # imported here: planner.py imports this module
                place = planner.suggest_placement(hw, model, {"ctx_size": ctx_budget, "kv_offload": v["kv_offload"]})
                v.update(n_gpu_layers=place["n_gpu_layers"], n_cpu_moe=place["n_cpu_moe"])
                notes.append(place["reason"])
            except Exception:  # noqa: BLE001 - never leave the user with a layout we could not check
                v.update(n_gpu_layers=0, n_cpu_moe=0)
                notes.append("Could not plan the GPU split, so everything runs on the CPU.")
            finally:
                _planning.busy = False

    if model:
        v["ctx_size"] = ctx_budget
        v["mmap"] = True
        v["mlock"] = bool(fits_ram and size < ram["total_gb"] * 0.6)
        if not fits_ram:
            notes.append("Model is larger than free RAM: mmap stays ON and context is kept small.")
    v["flash_attn"] = "on"
    v["cache_type_k"] = "q8_0"
    v["cache_type_v"] = "q8_0"
    v["ubatch_size"] = 256 if vram_gb < 6 else 512
    v["batch_size"] = 512 if vram_gb < 6 else 2048
    notes.append("8-bit KV cache + flash attention to save memory.")
    if model and model["architecture"] and "gpt-oss" in model["architecture"]:
        v["jinja"] = True
        v["reasoning_effort"] = "low" if not fits_ram else "medium"
    return {"values": v, "notes": notes}
