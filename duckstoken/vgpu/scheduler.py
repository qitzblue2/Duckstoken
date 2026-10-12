"""Candidate configurations to benchmark, and an honest registry of what the "scheduler" can and cannot do.

VirtualGPU does not own the inference engine, it drives llama.cpp through its command line.  That fixes what can be
scheduled:

* Static placement is real and supported: which layers go to the GPU (``-ngl``), which MoE experts stay in RAM
  (``--n-cpu-moe``, ``-ot``), where the KV cache lives and how it is quantized, mmap/mlock, threads, micro-batch size
  and whether batched work may borrow the GPU (``--no-op-offload``).  `candidate_configs` enumerates sensible
  variations of those, checks every one with the planner BEFORE anything is started and never returns a runnable
  candidate the planner rejected.
* Dynamic expert caching, expert-routing statistics, per-expert residency, GPU weight prefetch/double-buffering and a
  host<->GPU transfer-bandwidth probe do NOT exist: llama-server exposes no hooks for them (its scheduler and expert
  routing are internal to ggml, and no per-tensor or per-expert statistics are reported).  They are listed in
  `CAPABILITIES` as unavailable and are not simulated.  What replaces them is static placement plus storage-level
  sequential prefetch (`prewarm.py`: read the model file front to back into the OS page cache before llama.cpp maps it).

The first candidate is always the CPU-only baseline (``n_gpu_layers=0``, KV cache in RAM) so that every GPU
configuration has something measured to be compared against.
"""

from __future__ import annotations

import json
from typing import Any

from .. import settings as settings_mod
from .lazy import module as _mod
from .values import est, measured, num, unavailable

CAPABILITIES: list[dict] = [
    {"id": "layer_placement", "label": "Static layer placement (-ngl)", "available": True,
     "how": "n_gpu_layers sets how many layers live in VRAM; the rest stay in RAM.",
     "limitation": "Static for the whole session; changing it means restarting llama-server."},
    {"id": "moe_expert_placement", "label": "MoE expert placement (--n-cpu-moe, -ot)", "available": True,
     "how": "n_cpu_moe keeps the experts of the first N layers in RAM; override_tensor (-ot) pins tensor patterns.",
     "limitation": "Placement is per layer or per tensor pattern, decided at load time. Not per token or per expert."},
    {"id": "kv_placement", "label": "KV cache placement and quantization", "available": True,
     "how": "kv_offload=false keeps the cache in RAM; cache_type_k/v quantize it (q8_0, q4_0, f16, ...).",
     "limitation": "A quantized V cache needs flash attention. Smaller cache types lose some precision."},
    {"id": "mmap_mlock", "label": "mmap / mlock", "available": True,
     "how": "mmap lets the OS page weights in from storage; mlock pins them in RAM.",
     "limitation": "mlock fails or hurts when the model does not comfortably fit in RAM."},
    {"id": "op_offload_toggle", "label": "Op-offload on/off (--no-op-offload)", "available": True,
     "how": "Controls whether batched work on RAM-resident weights may be copied to the GPU.",
     "limitation": "Only matters with a usable GPU; whether it helps depends on the link, so it is measured, not assumed."},
    {"id": "threads_batch_sizes", "label": "Threads and batch sizes", "available": True,
     "how": "threads, threads_batch, batch_size and ubatch_size are passed to llama-server.",
     "limitation": "Best values depend on the CPU; the optimizer measures a few instead of guessing."},
    {"id": "storage_prefetch", "label": "Storage-level sequential prefetch", "available": True,
     "how": "prewarm.py reads the model file sequentially into the OS page cache before llama.cpp maps it.",
     "limitation": "Helps only when the model fits in free RAM. It does not make a hard drive faster."},
    {"id": "staging_cache", "label": "SSD staging copy of a model on a slow drive", "available": True,
     "how": "diskcache.StagingCache copies the model to a fast drive (after confirmation) so mmap pages come from it.",
     "limitation": "Needs free space for a second copy of the model and a one-time copy."},
    {"id": "expert_routing_stats", "label": "Expert routing / frequency statistics", "available": False,
     "how": "", "limitation": "llama-server does not report which experts a token used, so expert-aware caching cannot be built on top of it."},
    {"id": "dynamic_expert_residency", "label": "Per-expert dynamic residency (hot experts in VRAM)", "available": False,
     "how": "", "limitation": "No engine hook to move individual experts between RAM and VRAM at run time. Static --n-cpu-moe is the substitute."},
    {"id": "gpu_weight_prefetch", "label": "GPU weight prefetch / double-buffering", "available": False,
     "how": "", "limitation": "Internal to ggml and not exposed. Op-offload is the only related switch and is on/off only."},
    {"id": "transfer_bandwidth_probe", "label": "Measured host<->GPU transfer bandwidth", "available": False,
     "how": "", "limitation": "llama-server does not expose it. nvidia-smi gives only the PCIe link generation and width (theoretical)."},
]

PCIE_GBS_PER_LANE = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938}   # after encoding overhead, per direction
MAX_CANDIDATES = 12
CPU_KEYS = ("threads", "threads_batch")


def capability(cap_id: str) -> dict | None:
    return next((dict(c) for c in CAPABILITIES if c["id"] == cap_id), None)


# ------------------------------------------------------------------------------------------------ GPU gate
def gpu_gate(hw: dict, probe: dict | None) -> dict:
    """Whether GPU candidates make sense: a GPU with >= 1 GB VRAM AND evidence (the engine probe) that the engine
    can use a device.  Always says why not."""
    gpus = [g for g in (hw or {}).get("gpus") or [] if (g.get("vram_mb") or 0) >= 1024]
    if not gpus:
        return {"usable": False, "reason": "no GPU with at least 1 GB of VRAM was detected"}
    if not probe:
        return {"usable": False, "reason": "the engine has not been probed, so it is not known whether it can use the GPU"}
    devices = [d for d in probe.get("devices") or [] if str(d.get("backend", "")).upper() not in ("CPU", "BLAS", "")]
    usable = probe.get("backends_usable") or sorted({d["backend"] for d in devices})
    if not usable and not devices:
        return {"usable": False, "reason": "the engine lists no GPU device (a CPU-only build, or the driver is missing); "
                                           "GPU settings would silently run on the CPU"}
    return {"usable": True, "reason": "the engine lists a GPU device", "backends": list(usable), "gpu": gpus[0].get("name")}


# ------------------------------------------------------------------------------------------------ candidates
def _merged(base: dict, overrides: dict) -> dict:
    return {**base, **overrides}


def _config_key(merged: dict) -> str:
    keys = ("ctx_size", "n_gpu_layers", "n_cpu_moe", "cpu_moe", "kv_offload", "cache_type_k", "cache_type_v", "ubatch_size",
            "batch_size", "mmap", "mlock", "flash_attn", "threads", "threads_batch", "extra_args", "override_tensor")
    return json.dumps({k: merged.get(k) for k in keys}, sort_keys=True, default=str)


def with_extra_flag(extra_args: str, flag: str) -> str:
    parts = (extra_args or "").split()
    return " ".join(parts + ([flag] if flag not in parts else []))


def _effective_threads(base: dict, hw: dict) -> tuple[int, int]:
    cpu = hw.get("cpu") or {}
    phys = int(cpu.get("physical_cores") or 1)
    t = int(base.get("threads") or 0) or phys
    return t, int(base.get("threads_batch") or 0) or t


def _plan_cfg(merged: dict) -> dict:
    keys = ("ctx_size", "n_gpu_layers", "n_cpu_moe", "cpu_moe", "kv_offload", "cache_type_k", "cache_type_v",
            "ubatch_size", "mmap", "mlock", "flash_attn", "swa_full", "parallel")
    return {k: merged[k] for k in keys if k in merged}


def _speed_text(plan: dict | None) -> str:
    try:
        lo, hi = plan["speed"]["tokens_per_s"]["value"]
        return f"planner estimate {lo:g}-{hi:g} tokens/s (an estimate; the benchmark measures the real figure)"
    except (TypeError, KeyError, ValueError):
        return "no speed estimate available"


def candidate_configs(hw: dict, model: dict, base_settings: dict, objective: str, assumptions: dict | None = None,
                      include_unsafe: bool = False, *, probe: dict | None = None,
                      max_candidates: int = MAX_CANDIDATES) -> list[dict]:
    """Configurations worth measuring for `model`, conservative first, CPU-only baseline always first.

    Each item: ``{"id", "label", "overrides" (a settings patch), "plan" (planner result), "admitted", "reject_reason",
    "expected", "group"}``.  ``admitted`` is False when the planner cannot load the configuration, or when it would
    page weights from storage and `include_unsafe` is False (the baseline is exempt: it is what the user would run
    anyway).  A candidate with ``admitted=False`` must never be started.  `probe` is the engine probe result; without
    evidence of a usable GPU device no GPU candidates are generated (see `gpu_gate`).
    """
    pl = _mod("planner")
    if pl is None:
        raise RuntimeError("the memory planner is unavailable, so no configuration can be checked")
    base = {**settings_mod.DEFAULTS, **(base_settings or {})}
    cpu = hw.get("cpu") or {}
    phys, logical = int(cpu.get("physical_cores") or 1), int(cpu.get("logical_cores") or 1)
    path = model.get("path")
    gate = gpu_gate(hw, probe)
    seen: set[str] = set()
    cands: list[dict] = []

    def add(cid: str, label: str, overrides: dict, why: str, group: str, baseline: bool = False) -> None:
        merged = _merged(base, overrides)
        key = _config_key(merged)
        if key in seen:
            return
        seen.add(key)
        plan = pl.plan(hw, model, _plan_cfg(merged), assumptions, model_path=path)
        adm = plan["admission"]
        reason = ""
        if not adm["can_load"]:
            reason = "; ".join(adm.get("blockers") or ["the planner rejected this configuration"])
        elif adm["verdict"] == "paging" and not baseline and not include_unsafe:
            spill = num((plan.get("tiers", {}).get("storage") or {}).get("spill_gb"), 0.0)
            reason = f"it would page about {spill:.1f} GB of weights from storage (heavy disk reads); allow unsafe candidates to include it"
        cands.append({"id": cid, "label": label, "overrides": overrides, "plan": plan, "admitted": not reason,
                      "reject_reason": reason, "group": group, "expected": f"{why} - {_speed_text(plan)}"})

    cpu_only = {"n_gpu_layers": 0, "n_cpu_moe": 0, "cpu_moe": False, "kv_offload": False}
    add("cpu-baseline", "CPU only (baseline)", cpu_only, "no GPU involved; every other result is compared with this", "baseline",
        baseline=True)

    base_pair = _effective_threads(base, hw)
    sets = [(phys, phys, "threads-physical", "physical cores for generation and prompt")]
    if logical > phys:   # hyper-threading: also try the logical cores
        sets += [(phys, logical, "threads-phys-logical", "physical cores for generation, all logical for prompt processing"),
                 (logical, logical, "threads-logical", "all logical cores (hyper-threads)")]
    for t, tb, cid, desc in sets:
        if (t, tb) != base_pair:
            add(cid, f"CPU only, {desc}", {**cpu_only, "threads": t, "threads_batch": tb},
                f"same memory as the baseline, different threading ({t}/{tb})", "threads")

    modifiers: list[tuple[str, str, dict, str]] = []
    best_gpu: dict | None = None
    if gate["usable"]:
        n_layers = int(model.get("block_count") or 0)
        gpu_base = {"kv_offload": True}
        if model.get("is_moe"):
            top = n_layers + 1
            k0 = pl.min_cpu_moe(hw, model, {**_plan_cfg(_merged(base, gpu_base)), "n_gpu_layers": top}, assumptions) if hasattr(pl, "min_cpu_moe") else None
            if k0 is None:   # even the shared weights do not all fit: all experts in RAM, as many layers as fit
                m = pl.max_gpu_layers(hw, model, {**_plan_cfg(_merged(base, gpu_base)), "n_cpu_moe": n_layers}, assumptions)
                if m > 0:
                    best_gpu = {**gpu_base, "n_gpu_layers": m, "n_cpu_moe": n_layers}
                    add(f"gpu-{m}-experts-cpu", f"{m} layers on GPU, all experts in RAM", best_gpu,
                        "attention on the GPU, experts stay in RAM", "gpu")
            else:
                steps = sorted({n_layers, (k0 + n_layers) // 2, k0}, reverse=True)
                for k in steps:
                    ov = {**gpu_base, "n_gpu_layers": top, "n_cpu_moe": k}
                    best_gpu = ov
                    add(f"moe-cpu-{k}", f"All layers on GPU, experts of {k} layers in RAM", ov,
                        "shared weights on the GPU, " + ("all" if k == n_layers else f"{k} layers of") + " experts in RAM", "moe")
        else:
            m = pl.max_gpu_layers(hw, model, _plan_cfg(_merged(base, gpu_base)), assumptions)
            if m > 0:
                for ngl in sorted({max(1, round(m * 0.34)), max(1, round(m * 0.67)), m}):
                    ov = {**gpu_base, "n_gpu_layers": ngl}
                    best_gpu = ov
                    add(f"gpu-{ngl}", f"{ngl} layers on GPU", ov, f"{ngl} of {n_layers or '?'} layers in VRAM", "gpu")
                m2 = pl.max_gpu_layers(hw, model, {**_plan_cfg(_merged(base, gpu_base)), "kv_offload": False}, assumptions)
                if m2 > m:
                    ov = {"kv_offload": False, "n_gpu_layers": m2}
                    add(f"gpu-{m2}-kv-host", f"{m2} layers on GPU, KV cache in RAM", ov,
                        "the KV cache stays in RAM so more layers fit in VRAM", "gpu")
        if best_gpu is not None:
            modifiers.append(("op-offload-off", "GPU config with op-offload disabled",
                              {**best_gpu, "extra_args": with_extra_flag(base.get("extra_args", ""), "--no-op-offload")},
                              "batched work no longer borrows the GPU for RAM-resident weights (--no-op-offload)"))
    mod_base = cpu_only if str(objective) == "max_model_size" else (best_gpu or cpu_only)
    cur_k, cur_v = str(base.get("cache_type_k")), str(base.get("cache_type_v"))
    if cur_k == "q8_0" and cur_v == "q8_0":
        modifiers.append(("kv-f16", "KV cache in f16", {**mod_base, "cache_type_k": "f16", "cache_type_v": "f16"},
                          "full-precision KV cache: more memory, slightly better precision"))
    else:
        modifiers.append(("kv-q8_0", "KV cache in q8_0", {**mod_base, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                                                          "flash_attn": "on"},
                          "8-bit KV cache: about half the cache memory"))
    ub = int(base.get("ubatch_size") or 512)
    modifiers.append(("ubatch-256" if ub != 256 else "ubatch-512", f"Micro-batch {256 if ub != 256 else 512}",
                      {**mod_base, "ubatch_size": 256 if ub != 256 else 512},
                      "smaller micro-batch: less compute-buffer memory" if ub != 256 else "larger micro-batch: faster prompt processing"))
    if str(objective) == "max_model_size":
        modifiers.sort(key=lambda m: (m[0] not in ("kv-q8_0", "ubatch-256"), m[0]))   # memory savers first
    for cid, label, ov, why in modifiers:
        add(cid, label, ov, why, "modifier")

    order = {"baseline": 0, "threads": 1, "gpu": 3, "moe": 3, "modifier": 4}
    if str(objective) == "max_model_size":
        order["modifier"] = 2   # memory savers before offload experiments
    cands = sorted(enumerate(cands), key=lambda ic: (order.get(ic[1]["group"], 5), ic[0]))
    out = [c for _, c in cands]
    return out[:max(1, int(max_candidates))]


# ------------------------------------------------------------------------------------------------ transfer checks
def _pcie_of(probe: Any) -> tuple[Any, Any]:
    """(gen, width) from a telemetry sample, a list of GPU dicts or one GPU dict; values may be value-dicts or numbers."""
    gpus = probe.get("gpus") if isinstance(probe, dict) and "gpus" in probe else (probe if isinstance(probe, list) else [probe])
    for g in gpus or []:
        if isinstance(g, dict):
            gen, width = num(g.get("pcie_gen")), num(g.get("pcie_width"))
            if gen is not None and width is not None:
                return gen, width
    return None, None


def host_transfer_check(hw: dict, probe: Any, host_gbs: Any = None) -> dict:
    """Theoretical PCIe bandwidth (from the link generation and width nvidia-smi reports) against the host RAM
    bandwidth.  `probe` is a telemetry sample (``{"gpus": [...]}``), a list of GPU dicts or one GPU dict; `host_gbs`
    is a measured RAM bandwidth (a values dict or number), e.g. the benchmark's ``metrics.host_bandwidth_gbs``.
    Never a measurement of the link itself: llama-server cannot report that."""
    gen, width = _pcie_of(probe)
    host = num(host_gbs)
    out: dict = {"pcie_gbs": unavailable("PCIe link generation/width not reported (needs an NVIDIA GPU and nvidia-smi)", "nvidia-smi"),
                 "host_gbs": measured(round(host, 2), "host memory benchmark", "single-thread copy: a lower bound")
                 if host is not None else unavailable("host memory bandwidth was not measured", "benchmark"),
                 "host_theoretical_gbs": est(num((hw.get("ram") or {}).get("bandwidth_gbs")), "RAM speed and channels"),
                 "warn": None}
    if gen is None or width is None:
        return out
    lane = PCIE_GBS_PER_LANE.get(int(gen))
    if lane is None:
        out["pcie_gbs"] = unavailable(f"unknown PCIe generation {gen}", "nvidia-smi")
        return out
    gbs = round(lane * width, 2)
    out["pcie_gbs"] = est(gbs, "nvidia-smi link gen/width", f"PCIe gen {int(gen)} x{int(width)}, theoretical, one direction; "
                          "the link can run slower at idle")
    ref = host if host is not None else num((hw.get("ram") or {}).get("bandwidth_gbs"))
    if ref and gbs < ref:
        out["warn"] = (f"PCIe ({gbs:g} GB/s) is slower than host memory ({ref:g} GB/s): streaming RAM-resident weights "
                       "through the GPU cannot beat computing them on the CPU. Keep weights either fully on the GPU or fully "
                       "on the CPU, and consider --no-op-offload.")
    return out


def transfer_dominated(bench: dict) -> bool | None:
    """Did host<->GPU transfers dominate this benchmark?  llama-server does not expose transfer volume or time, so
    this is None (unknown) unless the record carries a measured ``transfer_bandwidth_gbs`` AND a theoretical
    ``pcie_gbs`` to compare with; then True when the measured transfer rate is within 20 % of the link's capacity."""
    metrics = (bench or {}).get("metrics") or {}
    tb, link = num(metrics.get("transfer_bandwidth_gbs")), num((bench or {}).get("pcie_gbs") or metrics.get("pcie_gbs"))
    if tb is None or not link:
        return None
    return tb >= 0.8 * link
