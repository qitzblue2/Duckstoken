"""Which GPU generation is this, and can the llama.cpp build in use really run on it?

Nothing is called "supported" without evidence: a GPU backend is *supported* only when ``llama-server --list-devices``
(the engine probe) lists a device of that backend. Hardware facts alone give *unknown* or *experimental* plus the reason.

Facts encoded (primary sources):
  * llama.cpp ``ggml/src/ggml-cuda/CMakeLists.txt`` (commit 08ffba2): CUDA toolkits older than 13 build PTX for compute
    5.0, 6.1 and 7.0 ("50-virtual 61-virtual 70-virtual", JIT-compiled on first start); toolkit 13 and newer do not.
  * NVIDIA: CUDA 13 dropped offline compilation and library support for Maxwell (5.x), Pascal (6.x) and Volta (7.0);
    CUDA 12.x is the last toolkit that targets them, and the R580 driver branch is the last to support those GPUs
    (R590 and later do not). A CUDA-13 build of llama.cpp therefore cannot run on a GTX 750.
  * The GTX 750 shipped with 1 GB or 2 GB of VRAM only.
"""

from __future__ import annotations

import re
import time

LEGACY_CC = 7.5  # compute capability below this = Maxwell, Pascal, Volta

# NVIDIA name table (used only when the driver does not report a compute capability); first match wins.
# (regex on the lower-case name, architecture, family/chip, compute capability)
_NVIDIA_NAMES = [
    (r"gtx 750 ?ti|gtx 750\b|gtx 745|gtx 840m|gtx 850m|gtx 860m|gm107", "Maxwell", "GM107", "5.0"),
    (r"titan x ?p\b|titan x \(pascal\)", "Pascal", "GP102", "6.1"),
    (r"gtx 9[5-9]0m?\b|titan x\b|quadro m[0-9]|gm20", "Maxwell", "GM20x", "5.2"),
    (r"gtx 10[3-8]0|quadro p|tesla p4\b|tesla p40|gp10[2-9]", "Pascal", "GP10x", "6.1"),
    (r"tesla p100|gp100", "Pascal", "GP100", "6.0"),
    (r"titan v\b|tesla v100|quadro gv100", "Volta", "GV100", "7.0"),
    (r"gtx 16[0-9]0|rtx 20[0-9]0|titan rtx|tesla t4|quadro rtx|\bt[1-4]00\b|\bt4\b", "Turing", "TU1xx", "7.5"),
    (r"rtx 50[0-9]0|rtx pro \d+ blackwell|blackwell|\bb[12]00\b", "Blackwell", "GB20x", "12.0"),
    (r"rtx 40[0-9]0|rtx [2-6]000 ada|\bl4\b|\bl40s?\b|ada generation", "Ada Lovelace", "AD10x", "8.9"),
    (r"rtx 30[0-9]0|rtx a[0-9]{3,4}|\ba(10|16|30|40|100)\b|a100", "Ampere", "GA10x", "8.6"),
    (r"\bh100\b|\bh200\b|gh200", "Hopper", "GH100", "9.0"),
    (r"gtx 6[0-9]0|gtx 7[0-9]0|gtx titan|quadro k|tesla k|gt 7[0-9]0|gtx 6[0-9]0 ti", "Kepler", "GK1xx", "3.5"),
    (r"gtx 4[0-9]0|gtx 5[0-9]0|geforce gt 5|quadro [0-9]{4}\b", "Fermi", "GF1xx", "2.1"),
]
_AMD_NAMES = [
    (r"rx 9[0-9]{3}|radeon ai pro r9", "RDNA 4", "gfx12"), (r"rx 7[0-9]{3}|radeon 7[0-9]{2}m|pro w7[0-9]{3}", "RDNA 3", "gfx11"),
    (r"rx 6[0-9]{3}|pro w6[0-9]{3}|radeon 6[0-9]{2}m", "RDNA 2", "gfx103x"), (r"rx 5[0-9]{3}|pro w5[0-9]{3}", "RDNA 1", "gfx101x"),
    (r"rx vega|vega (56|64)|radeon vii|instinct mi(50|60)", "Vega (GCN 5)", "gfx90x"), (r"instinct mi(1|2|3)[0-9]{2}", "CDNA", "gfx9xx"),
    (r"rx [45][0-9]{2}\b|r9 |r7 |r5 |fury|polaris|radeon pro (wx|580|570)", "GCN", "gfx8xx"),
]
_IGPU = re.compile(r"uhd graphics|hd graphics|iris|intel\(r\) graphics|radeon\(tm\) (vega \d+ )?graphics|radeon graphics|"
                   r"vega \d+ graphics|radeon (6|7|8)\d{2}m\b|radeon 610m|radeon 740m|adreno|apple m\d", re.IGNORECASE)


def _parse_cc(cc) -> float | None:
    try:
        text = str(cc).strip()
        if not text:
            return None
        major, _, minor = text.partition(".")
        return float(f"{int(major)}.{int(minor or 0)}")
    except (TypeError, ValueError):
        return None


def _nvidia_arch(cc: float) -> tuple[str, str]:
    if cc >= 10:
        return "Blackwell", "GB10x/GB20x"
    if cc >= 9:
        return "Hopper", "GH100"
    if cc >= 8.9:
        return "Ada Lovelace", "AD10x"
    if cc >= 8:
        return "Ampere", "GA10x" if cc >= 8.6 else "GA100"
    if cc >= 7.5:
        return "Turing", "TU1xx"
    if cc >= 7:
        return "Volta", "GV100"
    if cc >= 6:
        return "Pascal", "GP10x"
    if cc >= 5:
        return "Maxwell", "GM107" if cc < 5.2 else "GM20x"
    if cc >= 3:
        return "Kepler", "GK1xx"
    return "Fermi", "GF1xx"


def _vendor(gpu: dict) -> str:
    v = (gpu.get("vendor") or "").strip()
    if v:
        return {"nvidia corporation": "NVIDIA"}.get(v.lower(), v)
    n = (gpu.get("name") or "").lower()
    if any(k in n for k in ("nvidia", "geforce", "quadro", "tesla", "rtx", "gtx", "titan")):
        return "NVIDIA"
    if any(k in n for k in ("amd", "radeon", "instinct", "vega")):
        return "AMD"
    if any(k in n for k in ("intel", "arc ", "iris", "uhd")):
        return "Intel"
    if "apple" in n:
        return "Apple"
    return "unknown"


def classify_gpu(gpu: dict) -> dict:
    """Generation and legacy status of one GPU dict (as in ``hardware.get_hardware()["gpus"]``).

    Returns ``{"vendor","name","architecture","family","compute_cap","legacy","vram_gb","integrated","notes"}``. NVIDIA
    cards are classified by the driver's compute capability when present, otherwise by name (and then the capability is
    inferred from the table and the note says so). ``legacy`` is True for NVIDIA compute capability below 7.5.
    """
    gpu = gpu or {}
    name = str(gpu.get("name") or "").strip()
    low = name.lower()
    vendor = _vendor(gpu)
    vram_mb = gpu.get("vram_mb") or 0
    notes: list[str] = []
    out = {"vendor": vendor, "name": name, "architecture": "unknown", "family": "", "compute_cap": "", "legacy": False,
           "vram_gb": round(vram_mb / 1024, 2), "integrated": bool(_IGPU.search(name)), "notes": notes}
    if vendor == "NVIDIA":
        cc = _parse_cc(gpu.get("compute_cap"))
        if cc is not None:
            out["architecture"], out["family"] = _nvidia_arch(cc)
            out["compute_cap"] = f"{cc:g}" if cc != int(cc) else f"{int(cc)}.0"
        else:
            for pat, arch, fam, cc_txt in _NVIDIA_NAMES:
                if re.search(pat, low):
                    out.update(architecture=arch, family=fam, compute_cap=cc_txt)
                    cc = float(cc_txt)
                    notes.append(f"Compute capability {cc_txt} is inferred from the card name; the driver did not report it.")
                    break
        if cc is not None:
            out["legacy"] = cc < LEGACY_CC
            if cc < 5.0:
                notes.append("Below compute capability 5.0: CUDA 12 (the toolkit llama.cpp builds use) no longer targets this GPU.")
            elif out["legacy"]:
                notes.append("Maxwell/Pascal/Volta: CUDA 13 removed support for this generation. Only a CUDA 12.x build of "
                             "llama.cpp can run on it (its PTX is JIT-compiled on first start), and NVIDIA's R580 driver "
                             "branch is the last to support it.")
        if re.search(r"gtx 750(?! ?ti)", low):
            notes.append("The GTX 750 was sold with 1 GB or 2 GB of VRAM.")
    elif vendor == "AMD":
        for pat, arch, fam in _AMD_NAMES:
            if re.search(pat, low):
                out.update(architecture=arch, family=fam)
                break
        else:
            if out["integrated"]:
                out.update(architecture="Radeon integrated graphics", family="APU")
    elif vendor == "Intel":
        if re.search(r"arc.*\bb\d{3}|battlemage", low):
            out.update(architecture="Xe2 (Battlemage)", family="Arc B")
        elif re.search(r"arc|alchemist", low):
            out.update(architecture="Xe-HPG (Alchemist)", family="Arc A")
        elif re.search(r"iris xe|xe graphics", low):
            out.update(architecture="Xe-LP", family="Iris Xe")
        elif out["integrated"]:
            out.update(architecture="Gen9/Gen11 integrated", family="UHD/HD Graphics")
    elif vendor == "Apple":
        out.update(architecture="Apple silicon GPU", family="Metal")
    if out["integrated"]:
        notes.append("Integrated GPU: it shares system RAM with the CPU, so it adds no memory and little speed for "
                     "bandwidth-bound generation.")
    if vendor != "NVIDIA" and vram_mb in (4095, 4096):
        notes.append("Windows reports at most 4 GB for this adapter (32-bit field): the card may have more VRAM than shown.")
    if vendor == "unknown":
        notes.append("Vendor not recognised: no compatibility claim is made.")
    return out


# ------------------------------------------------------------------------------------------------------ report
def _item(id_: str, label: str, status: str, detail: str, evidence: str, how: str = "") -> dict:
    return {"id": id_, "label": label, "status": status, "detail": detail, "evidence": evidence, "how_to_verify": how}


def _driver_major(gpu: dict) -> int | None:
    m = re.match(r"\s*(\d+)", str(gpu.get("driver") or ""))
    return int(m.group(1)) if m else None


def _probe_devices(probe: dict | None, backend: str) -> list[dict]:
    return [d for d in (probe or {}).get("devices", []) if d.get("backend") == backend]


def _gpu_backend_item(backend: str, label: str, candidates: list[dict], classes: list[dict], probe: dict | None,
                      env: dict) -> dict:
    """One GPU backend: supported only with a listed device; otherwise unsupported/experimental/unknown plus reason."""
    id_ = backend.lower()
    verify = ("Run the engine's `llama-server --list-devices`; a " + backend + " device must be listed. Duckstoken's "
              "engine probe does this for you.")
    devs = _probe_devices(probe, backend)
    if devs:
        names = ", ".join(f"{d['name']} ({d['total_mb']} MiB)" for d in devs)
        return _item(id_, label, "supported", f"The engine lists {len(devs)} {backend} device(s): {names}.", "probe", verify)
    legacy = [c for c in classes if c["legacy"] and c["vendor"] == "NVIDIA"]
    asset = str(env.get("engine_asset") or "").lower()
    if not candidates:
        return _item(id_, label, "unsupported", f"No {backend}-capable GPU was detected on this machine.", "detected")
    probed_ok = bool(probe and probe.get("status") in ("ok", "partial") and probe.get("devices") is not None)
    if backend == "CUDA" and legacy:
        gpu = legacy[0]
        base = (f"{gpu['name']} is {gpu['architecture']} (compute capability {gpu['compute_cap']}). CUDA 13 dropped this "
                "generation, so a CUDA 13 build of llama.cpp cannot run on it; a CUDA 12.x build may, by JIT-compiling PTX on "
                "first start (slow once).")
        if "cuda-13" in asset or "cuda13" in asset or str(env.get("cuda_version", "")).startswith("13"):
            return _item(id_, label, "unsupported", base + " The installed engine is a CUDA 13 build.", "table", verify)
        if probed_ok:
            return _item(id_, label, "unsupported", base + " The installed engine's device list has no CUDA device, so this "
                         "build cannot use the card (wrong CUDA version, missing driver or a CPU-only build).", "probe", verify)
        return _item(id_, label, "experimental", base + " Not verified: no engine probe has been run.", "table", verify)
    if probed_ok:
        return _item(id_, label, "unsupported", f"A {backend}-capable GPU is present, but the installed engine lists no {backend} "
                     "device: it is a different build (e.g. CPU-only) or the driver/runtime is missing.", "probe", verify)
    return _item(id_, label, "unknown", f"A {backend}-capable GPU was detected ({candidates[0].get('name', '?')}), but no engine "
                 "probe has been run, so nothing proves this build can use it.", "detected", verify)


def compatibility_report(hw: dict, engine_probe: dict | None = None, model: dict | None = None, env: dict | None = None) -> dict:
    """Honest compatibility items for this machine, optionally against an engine probe and a model.

    ``env`` may carry ``engine_asset`` (the installed engine's release file name, e.g. ``...-cuda-13.1-x64.zip``) and
    ``cuda_version``; they let a known-bad combination be reported as *unsupported* without running anything.
    """
    env = env or {}
    cpu, ram, gpus, disks = hw.get("cpu") or {}, hw.get("ram") or {}, hw.get("gpus") or [], hw.get("disks") or []
    classes = [classify_gpu(g) for g in gpus]
    items: list[dict] = []

    if "avx2" not in cpu:
        items.append(_item("cpu", "CPU inference", "unknown", "The CPU feature flags could not be read.", "detected"))
    elif cpu.get("avx2"):
        extra = " AVX-512 is also present." if cpu.get("avx512") else ""
        items.append(_item("cpu", "CPU inference", "supported", f"{cpu.get('name', 'CPU')} has AVX2; llama.cpp's CPU backend "
                           f"runs at full speed.{extra}", "detected", "Run a benchmark; the CPU baseline is always measured."))
    else:
        items.append(_item("cpu", "CPU inference", "experimental", "No AVX2: llama.cpp falls back to older instruction sets and "
                           "is several times slower.", "detected", "Run a benchmark."))

    nvidia = [g for g in gpus if _vendor(g) == "NVIDIA"]
    amd = [g for g in gpus if _vendor(g) == "AMD"]
    intel = [g for g in gpus if _vendor(g) == "Intel"]
    is_mac = "darwin" in str(hw.get("os", "")).lower() or "macos" in str(hw.get("os", "")).lower()
    items.append(_gpu_backend_item("CUDA", "CUDA (NVIDIA)", nvidia, classes, engine_probe, env))
    items.append(_gpu_backend_item("Vulkan", "Vulkan (any GPU)", gpus, classes, engine_probe, env))
    items.append(_gpu_backend_item("HIP", "HIP / ROCm (AMD)", amd, classes, engine_probe, env))
    items.append(_gpu_backend_item("SYCL", "SYCL (Intel)", intel, classes, engine_probe, env))
    items.append(_gpu_backend_item("Metal", "Metal (Apple)", gpus if is_mac else [], classes, engine_probe, env))

    for g, c in zip(gpus, classes):
        if c["vendor"] == "NVIDIA" and c["legacy"]:
            major = _driver_major(g)
            if major is None:
                st, d = "unknown", "The driver version could not be read."
            elif major >= 590:
                st, d = "unsupported", (f"Driver {g.get('driver')} is newer than R580, the last NVIDIA branch that supports "
                                        "Maxwell, Pascal and Volta.")
            else:
                st, d = "supported", f"Driver {g.get('driver')} is on a branch that still supports {c['architecture']}."
            items.append(_item("nvidia_driver", f"NVIDIA driver for {c['name']}", st, d, "detected",
                               "Run `nvidia-smi`; legacy GPUs need a driver from the R580 branch or older."))
            break

    items.append(_item("mmap", "Memory-mapped model files (mmap)", "supported",
                       "llama.cpp maps GGUF files with CreateFileMapping/mmap on Windows and Linux; weights that do not fit in RAM "
                       "are paged from storage.", "table", "The server log prints `load_mode = mmap` when it is active."))

    supports = (engine_probe or {}).get("supports") or {}
    gpt = supports.get("gpt-oss")
    if engine_probe is None:
        items.append(_item("gpt_oss", "GPT-OSS architecture", "unknown", "No engine probe has been run.", "table",
                           "Probe the engine; it scans the binaries for the `gpt-oss` architecture."))
        items.append(_item("mxfp4", "MXFP4 weight format", "unknown", "No engine probe has been run.", "table"))
    elif gpt is None:
        items.append(_item("gpt_oss", "GPT-OSS architecture", "unknown", "The engine binaries could not be scanned.", "probe"))
        items.append(_item("mxfp4", "MXFP4 weight format", "unknown", "The engine binaries could not be scanned.", "probe"))
    elif gpt:
        items.append(_item("gpt_oss", "GPT-OSS architecture", "supported", "The engine binaries contain the gpt-oss architecture.",
                           "probe", "Load a gpt-oss GGUF; the log prints `arch = gpt-oss`."))
        items.append(_item("mxfp4", "MXFP4 weight format", "supported", "gpt-oss ships in MXFP4, and the engine that knows gpt-oss "
                           "also reads it on the CPU.", "probe"))
    else:
        items.append(_item("gpt_oss", "GPT-OSS architecture", "unsupported", "The engine binaries do not contain the gpt-oss "
                           "architecture: the build predates it.", "probe", "Install a newer llama.cpp build."))
        items.append(_item("mxfp4", "MXFP4 weight format", "unsupported", "The build predates gpt-oss and MXFP4.", "probe"))

    best = max(classes, key=lambda c: c["vram_gb"], default=None)
    if not best or best["vram_gb"] < 1:
        items.append(_item("vram_threshold", "GPU memory worth offloading (4 GB+)", "unsupported",
                           "No GPU with at least 1 GB of VRAM was found; everything runs on the CPU.", "detected"))
    elif best["integrated"] and best["vendor"] != "NVIDIA":
        items.append(_item("vram_threshold", "GPU memory worth offloading (4 GB+)", "unsupported",
                           f"{best['name']} is an integrated GPU that shares system RAM.", "detected"))
    elif best["vram_gb"] >= 4:
        items.append(_item("vram_threshold", "GPU memory worth offloading (4 GB+)", "supported",
                           f"{best['name']} has {best['vram_gb']:g} GB of VRAM; partial offload is worth benchmarking.", "detected",
                           "Compare a GPU-layer run against the CPU baseline in the benchmark."))
    else:
        items.append(_item("vram_threshold", "GPU memory worth offloading (4 GB+)", "experimental",
                           f"{best['name']} has {best['vram_gb']:g} GB of VRAM: below 4 GB only attention/shared weights fit, so "
                           "any benefit must be measured, not assumed.", "detected", "Benchmark n_gpu_layers=0 against the suggested value."))

    total = ram.get("total_gb")
    if not total:
        items.append(_item("ram", "System RAM", "unknown", "The amount of RAM could not be read.", "detected"))
    else:
        bw = ram.get("bandwidth_gbs")
        speed = f" at {ram['speed_mts']} MT/s ({ram.get('sticks', '?')} stick(s), ~{bw} GB/s theoretical)" if ram.get("speed_mts") else ""
        st = "supported" if total >= 16 else "experimental" if total >= 8 else "unsupported"
        items.append(_item("ram", "System RAM", st, f"{total:g} GB installed{speed}. Token generation is limited by this bandwidth.",
                           "detected", "The benchmark reports the measured host memory bandwidth."))

    kinds = {d.get("kind", "unknown") for d in disks}
    if not disks:
        items.append(_item("disk", "Model storage", "unknown", "No drive information was available.", "detected"))
    elif kinds <= {"nvme", "ssd"}:
        names = "/".join({"nvme": "NVMe", "ssd": "SSD"}[k] for k in sorted(kinds))
        items.append(_item("disk", "Model storage", "supported", f"Models can live on a fast {names} drive; paging stays usable.",
                           "detected"))
    elif "hdd" in kinds:
        items.append(_item("disk", "Model storage", "experimental", "A hard drive is present: fine for models that fit in RAM "
                           "(slow first load), unusable for paging. Keep models on an SSD.", "detected"))
    else:
        items.append(_item("disk", "Model storage", "unknown", "The drive type could not be determined.", "detected"))

    if model:
        from . import planner
        pl = planner.plan(hw, model)
        v = pl["admission"]["verdict"]
        st = {"ok": "supported", "tight": "supported", "paging": "experimental", "reject": "unsupported"}[v]
        detail = f"Planner verdict '{v}' for {pl['model']['name'] or 'the selected model'}."
        adm = pl["admission"]
        if adm["blockers"]:
            detail += " " + adm["blockers"][0]
        elif adm["warnings"]:  # the memory warning is the relevant one for a fit verdict
            detail += " " + next((w for w in adm["warnings"] if "RAM" in w or "fits but" in w), adm["warnings"][0])
        items.append(_item("model_fit", "Selected model fits this machine", st, detail, "planner (estimate)",
                           "Run the benchmark to replace the estimate with a measurement."))

    counts = {s: sum(1 for i in items if i["status"] == s) for s in ("supported", "experimental", "unsupported", "unknown")}
    gpu_ok = any(i["status"] == "supported" and i["id"] in ("cuda", "vulkan", "hip", "sycl", "metal") for i in items)
    summary = ("CPU inference " + ("is available. " if any(i["id"] == "cpu" and i["status"] == "supported" for i in items) else
                                    "needs attention. ") +
               ("A GPU backend is verified by the engine probe. " if gpu_ok else
                "No GPU backend is verified"
                + (" (the engine probe found no usable GPU device). " if engine_probe else " (no engine probe has been run). ")) +
               f"{counts['supported']} supported, {counts['experimental']} experimental, {counts['unsupported']} unsupported, "
               f"{counts['unknown']} unknown.")
    return {"items": items, "gpu": classes, "summary": summary, "counts": counts,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
