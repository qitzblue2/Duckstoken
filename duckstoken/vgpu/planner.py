"""Placement planner: where do the weights, KV cache and compute buffers go, and does the model load at all?

`plan()` is a pure function of (hardware dict, model dict, settings subset, assumptions). It never sums VRAM, RAM and
storage into one number: each tier has its own budget, a GPU overcommit is a blocker for that placement (lower
``n_gpu_layers``) and weights that exceed RAM are reported as storage paging with their own speed penalty.
All memory figures are GiB (2**30), the unit Windows shows; every number returned is labelled ``estimated`` and every
constant it rests on is listed in the returned ``assumptions``.

How llama.cpp places things (src/llama-model.cpp, src/llama-kv-cache*.cpp at commit 08ffba2) and what is modelled:
  * ``-ngl N`` puts the last N repeating layers on the GPU (the output layer too when N > layers); the token embedding
    always stays on the host. ``--n-cpu-moe K`` keeps the expert tensors of the *first* K layers on the host.
  * KV cells of a layer live where the layer lives (unless ``--no-kv-offload``); sliding-window layers keep only
    ``n_swa + ubatch`` cells.
  * ``mmap`` weights are page cache: when RAM is short the OS evicts and re-reads them from storage.
  * ``--op-offload`` (default) copies host-resident weights into the GPU compute buffer for batched work.
"""

from __future__ import annotations

import re
from pathlib import Path

from .. import hardware, settings, tuner
from . import memcalc, quant
from .memcalc import (  # noqa: F401
    ASSUMPTION_DOCS,
    DEFAULT_ASSUMPTIONS,
    GIB,
    bytes_per_element,
    compute_buffer_bytes,
    kv_cache_bytes,
    measured_tiers_gb,
    norm_model,
    parse_buffer_log,
    weights_split,
)
from .values import est, measured, unavailable

# Theoretical GPU memory bandwidth in GB/s (vendor spec sheets); first match wins. Unknown cards fall back to an assumption.
_GPU_BW = [(r"750 ?ti", 86.4), (r"gtx 750", 80.0), (r"gtx 950", 105.7), (r"gtx 960", 112.0), (r"gtx 970", 224.0),
           (r"gtx 980", 224.0), (r"gtx 1050", 112.0), (r"gtx 1060", 192.0), (r"gtx 1070", 256.0), (r"gtx 1080", 320.0),
           (r"gtx 1650", 128.0), (r"gtx 1660 ti", 288.0), (r"gtx 1660 super", 336.0), (r"gtx 1660", 192.0),
           (r"rtx 2060", 336.0), (r"rtx 2070", 448.0), (r"rtx 2080", 448.0), (r"rtx 3050", 224.0),
           (r"rtx 3060 ti", 448.0), (r"rtx 3060", 360.0), (r"rtx 3070", 448.0), (r"rtx 3080", 760.0),
           (r"rtx 3090", 936.0), (r"rtx 4060 ti", 288.0), (r"rtx 4060", 272.0), (r"rtx 4070", 504.0),
           (r"rtx 4080", 717.0), (r"rtx 4090", 1008.0), (r"rtx 5070", 672.0), (r"rtx 5080", 960.0),
           (r"rtx 5090", 1792.0)]
_GPU_BW_FALLBACK = 150.0


def _gpu_bandwidth(gpu: dict | None, a: dict) -> tuple[float, str]:
    if a.get("gpu_bandwidth_gbs"):
        return float(a["gpu_bandwidth_gbs"]), "assumption gpu_bandwidth_gbs"
    name = (gpu or {}).get("name", "").lower()
    for pat, bw in _GPU_BW:
        if re.search(pat, name):
            return bw, "vendor spec (theoretical) for " + (gpu or {}).get("name", "")
    return _GPU_BW_FALLBACK, "assumed: card not in the bandwidth table"


# ---------------------------------------------------------------------------------------------------- settings
_CFG_KEYS = ("ctx_size", "n_gpu_layers", "n_cpu_moe", "cpu_moe", "kv_offload", "cache_type_k", "cache_type_v",
             "ubatch_size", "mmap", "mlock", "flash_attn", "parallel", "op_offload", "swa_full")


CFG_KEYS = _CFG_KEYS  # settings the planner reads; callers can pass just these


def _effective_cfg(hw: dict, model: dict, cfg: dict | None, a: dict) -> dict:
    """cfg > tuner.recommend(hw, model) > planner assumptions > settings.DEFAULTS."""
    base = {k: settings.DEFAULTS[k] for k in _CFG_KEYS if k in settings.DEFAULTS}
    base.update(cache_type_k=a["kv_k_type"], cache_type_v=a["kv_v_type"], ubatch_size=a["ubatch_default"],
                swa_full=a["swa_full"], op_offload=True)
    try:
        tm = {"size_gb": 0, "is_moe": False, "expert_count": 0, "expert_used_count": 0, "block_count": 0,
              "context_length": 0, "architecture": "", **model}
        tm["size_gb"] = tm.get("size_gb") or tm.get("size_bytes", 0) / GIB
        base.update(tuner.recommend(hw, tm)["values"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    base.update({k: v for k, v in (cfg or {}).items() if v is not None})
    return base


def _num(v, default=0):
    try:
        return type(default)(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# -------------------------------------------------------------------------------------------------- placement
def _vram_fits(vram_bytes: float, budget_gb: float, a: dict) -> bool:
    """VRAM check with the same safety margin as RAM (estimate error must not turn into an out-of-memory at load)."""
    return vram_bytes / GIB * (1 + a["safety_margin_pct"] / 100) <= budget_gb + 1e-9


def _place(m: dict, ws: dict, cfg: dict, ngl: int, kvl: dict, cb: int, a: dict, flash_attn: bool) -> dict:
    """Bytes per tier for one n_gpu_layers value. Pure arithmetic; used by plan() and max_gpu_layers()."""
    n = m["block_count"]
    pls, ple = ws["per_layer_shared_bytes"], ws["per_layer_expert_bytes"]
    gpu_layers = 0 if not n else (n if ngl < 0 else min(ngl, n))
    out_gpu = bool(n) and (ngl < 0 or ngl > n)
    first = n - gpu_layers
    k_cpu = 0
    if m["is_moe"] and n:
        k_cpu = n if cfg.get("cpu_moe") else min(max(_num(cfg.get("n_cpu_moe"), 0), 0), n)
    experts_gpu_layers = max(n - max(first, k_cpu), 0) if (gpu_layers and m["is_moe"]) else 0
    w_gpu = gpu_layers * pls + experts_gpu_layers * ple + (ws["output_bytes"] if out_gpu else 0)
    dup = ws["output_bytes"] if (out_gpu and ws["tied_embeddings"]) else 0  # tied lm head: a copy, file keeps its own
    kv_layers = kvl["layer_bytes"]
    kv_total = sum(kv_layers)
    kv_gpu = sum(kv_layers[first:]) if (cfg.get("kv_offload") and gpu_layers) else 0
    on_gpu = bool(gpu_layers or out_gpu)
    cb_gpu = cb if on_gpu else 0
    cb_host = cb * a["host_compute_fraction"] if on_gpu else cb
    staging = 0.0
    if on_gpu and cfg.get("op_offload", True) and (gpu_layers < n or k_cpu > 0) and ws["total_bytes"]:
        unit = ple / 3 if m["is_moe"] else pls / 4
        staging = a["op_offload_staging_tensors"] * unit
    return {"gpu_layers": gpu_layers, "cpu_layers": n - gpu_layers, "output_on_gpu": out_gpu, "k_cpu": k_cpu,
            "w_gpu": w_gpu, "w_ram": ws["total_bytes"] - (w_gpu - dup), "kv_total": kv_total, "kv_gpu": kv_gpu,
            "kv_ram": kv_total - kv_gpu, "cb_gpu": cb_gpu, "cb_host": cb_host, "staging": staging,
            "vram": w_gpu + kv_gpu + cb_gpu + staging}


def _autofit(m: dict, ws: dict, c: dict, given: set, kvl: dict, cb: int, a: dict, flash: bool, budget_gb: float) -> dict:
    """Settings the caller did NOT specify, adjusted so the GPU placement fits: ``{"n_cpu_moe": k, "n_gpu_layers": g}``.

    Values the caller passed are never touched (an overcommit there is a blocker). Only raises ``n_cpu_moe`` (more experts
    stay in RAM) and lowers ``n_gpu_layers``; every change is announced in the plan's warnings.
    """
    n = m["block_count"]
    if not n:
        return {}

    def fits(cc, ngl):
        return _vram_fits(_place(m, ws, cc, ngl, kvl, cb, a, flash)["vram"], budget_gb, a)

    cc, changed = dict(c), {}
    ngl = _num(cc.get("n_gpu_layers"), 0)
    if fits(cc, ngl):
        return {}
    if m["is_moe"] and not ({"n_cpu_moe", "cpu_moe"} & given) and not cc.get("cpu_moe"):
        for k in range(_num(cc.get("n_cpu_moe"), 0) + 1, n + 1):
            if fits({**cc, "n_cpu_moe": k}, ngl):
                return {"n_cpu_moe": k}
        cc["n_cpu_moe"] = n
        changed["n_cpu_moe"] = n
    if "n_gpu_layers" not in given:
        for g in range(min(ngl if ngl >= 0 else n + 1, n + 1), -1, -1):
            if fits(cc, g):
                return {**changed, "n_gpu_layers": g}
    return changed


def _gpu_budget_gb(hw: dict, a: dict) -> tuple[dict | None, float, float]:
    """(gpu, vram_total_gb, usable_budget_gb). Cards under 1 GB are not used (same rule as tuner.py)."""
    gpus = [g for g in hw.get("gpus") or [] if g.get("vram_mb")]
    if not gpus:
        return None, 0.0, 0.0
    gpu = max(gpus, key=lambda g: g["vram_mb"])
    total = gpu["vram_mb"] / 1024
    if total < 1:
        return gpu, total, 0.0
    return gpu, total, max(total - a["vram_reserve_mb"] / 1024, 0.0)


def _context(m: dict, cfg: dict) -> tuple[int, str]:
    ctx = _num(cfg.get("ctx_size"), 0)
    if ctx > 0:
        return ctx, ""
    train = m["context_length"] or 4096
    return train, f"ctx_size=0 makes llama.cpp use the model's full training context ({train} tokens)."


def _norm_hw(hw: dict | None) -> dict:
    """A hardware dict with every section present, so a partial one (detection failed) yields blockers, not KeyErrors."""
    hw = dict(hw or {})
    for key, empty in (("cpu", {}), ("ram", {}), ("gpus", []), ("disks", [])):
        if not isinstance(hw.get(key), type(empty)):
            hw[key] = empty
    return hw


def _build_context(hw: dict, model: dict, cfg: dict | None, assumptions: dict | None):
    """Everything plan() and the helpers share: assumptions, normalised model, effective settings, KV/compute sizes."""
    a = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    m = norm_model(model)
    c = _effective_cfg(hw, model, cfg, a)
    ctx, ctx_note = _context(m, c)
    ub = max(_num(c.get("ubatch_size"), a["ubatch_default"]), 1)
    par = max(_num(c.get("parallel"), 1), 1)
    a_kv = {**a, "swa_full": bool(c.get("swa_full", a["swa_full"]))}
    ktype, vtype = (t if str(t).lower() in memcalc._BPE else "f16" for t in (c["cache_type_k"], c["cache_type_v"]))
    kvl = memcalc.kv_layout(m, ctx, str(ktype), str(vtype), a_kv, par, ub)  # invalid types are blocked in plan()
    flash = str(c.get("flash_attn", "auto")) != "off"
    cb = compute_buffer_bytes(m, ctx, ub, True, a, flash)
    return a, m, c, ctx, ctx_note, ub, kvl, flash, cb


def max_gpu_layers(hw: dict, model: dict, cfg: dict | None = None, assumptions: dict | None = None) -> int:
    """Largest ``n_gpu_layers`` whose VRAM plan fits the card (0 when none does or there is no usable GPU).

    Values above the layer count (output layer on the GPU) are returned as ``block_count + 1``.
    """
    a, m, c, ctx, _, ub, kvl, flash, cb = _build_context(hw, model, cfg, assumptions)
    _, _, budget = _gpu_budget_gb(hw, a)
    n = m["block_count"]
    if budget <= 0 or not n:
        return 0
    ws = weights_split(m, a)
    for ngl in range(n + 1, 0, -1):
        if _vram_fits(_place(m, ws, c, ngl, kvl, cb, a, flash)["vram"], budget, a):
            return ngl
    return 0


def min_cpu_moe(hw: dict, model: dict, cfg: dict | None = None, assumptions: dict | None = None) -> int | None:
    """Smallest ``n_cpu_moe`` (experts of the first K layers stay in RAM) that lets everything else fit the GPU.

    Offloads all layers (attention + the experts of layers >= K). 0 for dense models and for MoE models that fit
    entirely; None when no GPU is usable or even the shared weights do not fit.
    """
    a, m, c, ctx, _, ub, kvl, flash, cb = _build_context(hw, model, cfg, assumptions)
    _, _, budget = _gpu_budget_gb(hw, a)
    n = m["block_count"]
    if budget <= 0 or not n:
        return None
    if not m["is_moe"]:
        return 0
    ws = weights_split(m, a)
    c = {**c, "cpu_moe": False}
    ngl = _num(cfg["n_gpu_layers"], n + 1) if cfg and cfg.get("n_gpu_layers") is not None else n + 1
    for k in range(n + 1):
        if _vram_fits(_place(m, ws, {**c, "n_cpu_moe": k}, ngl, kvl, cb, a, flash)["vram"], budget, a):
            return k
    return None


def suggest_placement(hw: dict, model: dict, cfg: dict | None = None, assumptions: dict | None = None) -> dict:
    """A GPU placement that fits: ``{"n_gpu_layers", "n_cpu_moe", "reason"}`` (a settings patch).

    MoE: all layers on the GPU, as many expert layers there as fit (the rest stay in RAM). Dense: the most layers that
    fit. No usable GPU: ``n_gpu_layers=0``. Whether the GPU actually speeds anything up is for the benchmark to say.
    """
    a, m, c, *_ = _build_context(hw, model, cfg, assumptions)
    _, _, budget = _gpu_budget_gb(hw, a)
    if budget <= 0 or not m["block_count"]:
        return {"n_gpu_layers": 0, "n_cpu_moe": 0, "reason": "No GPU with at least 1 GB of VRAM (or unknown layer count)."}
    n = m["block_count"]
    if m["is_moe"]:
        k = min_cpu_moe(hw, model, cfg, assumptions)
        if k is not None:
            return {"n_gpu_layers": n + 1, "n_cpu_moe": k,
                    "reason": f"All layers on the GPU; experts of the first {k} of {n} layers stay in RAM."}
        best = max_gpu_layers(hw, model, {**(cfg or {}), "n_cpu_moe": n}, assumptions)
        return {"n_gpu_layers": best, "n_cpu_moe": n,
                "reason": f"All experts stay in RAM; only {best} of {n} layers' attention fits the GPU."}
    best = max_gpu_layers(hw, model, cfg, assumptions)
    return {"n_gpu_layers": best, "n_cpu_moe": 0, "reason": f"{best} of {n} layers fit the GPU."}


def fits_vram(hw: dict, model: dict, cfg: dict | None = None, assumptions: dict | None = None) -> bool:
    """True when the configured ``n_gpu_layers`` fits the GPU budget (always True with nothing placed on the GPU)."""
    a, m, c, ctx, _, ub, kvl, flash, cb = _build_context(hw, model, cfg, assumptions)
    _, _, budget = _gpu_budget_gb(hw, a)
    ngl = _num(c.get("n_gpu_layers"), 0) if budget > 0 else 0
    pl = _place(m, weights_split(m, a), c, ngl, kvl, cb, a, flash)
    return _vram_fits(pl["vram"], budget, a)


def resident_estimate_gb(plan_result: dict) -> float:
    """Memory the configuration keeps resident (VRAM + RAM tiers, without the OS reserve and without paged weights)."""
    t = plan_result["tiers"]
    return round(float(t["vram"]["planned_gb"]["value"] or 0) + float(t["ram"]["planned_gb"]["value"] or 0), 2)


# ---------------------------------------------------------------------------------------------------- speed
def _speed(hw: dict, m: dict, ws: dict, pl: dict, kvl: dict, a: dict, gpu: dict | None, kind: str, spill: float) -> dict:
    ram_bw = max(hw["ram"].get("bandwidth_gbs") or 0, 1.0) * a["ram_bw_efficiency"]
    gpu_bw, gpu_src = _gpu_bandwidth(gpu, a)
    n, kfrac = m["block_count"], 1.0
    if m["is_moe"] and m["expert_count"]:
        kfrac = max(m["expert_used_count"], 1) / m["expert_count"]
    pls, ple = ws["per_layer_shared_bytes"], ws["per_layer_expert_bytes"]
    g, k = pl["gpu_layers"], pl["k_cpu"]
    exp_gpu_layers = max(n - max(n - g, k), 0) if (g and m["is_moe"]) else 0
    a_gpu = g * pls + exp_gpu_layers * ple * kfrac + (ws["output_bytes"] if pl["output_on_gpu"] else 0)
    a_total = n * (pls + ple * kfrac) + ws["output_bytes"] if n else ws["total_bytes"] * kfrac
    a_cpu = max(a_total - a_gpu, 0.0)
    kv_read = pl["kv_total"] * a["kv_read_fill_fraction"]
    kv_gpu_read = pl["kv_gpu"] * a["kv_read_fill_fraction"]
    a_gpu += kv_gpu_read
    a_cpu += kv_read - kv_gpu_read
    miss = 0.0
    if spill > 0 and pl["w_ram"] > 0:  # spill is GiB, w_ram bytes
        miss = min(0.9, max(0.05, a["page_cache_skew"] * spill * GIB / pl["w_ram"]))
    disk = tuner.DISK_GBS.get(kind, tuner.DISK_GBS["unknown"])
    t_cpu, t_gpu = a_cpu / GIB / ram_bw, a_gpu / GIB / gpu_bw
    t_miss = a_cpu * miss / GIB / disk
    lo_t = (t_cpu + t_gpu) / a["speed_efficiency_lo"] + t_miss / a["storage_speed_lo"]
    hi_t = (t_cpu + t_gpu) / a["speed_efficiency_hi"] + t_miss / a["storage_speed_hi"]
    total = t_cpu + t_gpu + t_miss
    if total <= 0:  # nothing to read: the model size is unknown, so no number is better than a made-up one
        return {"tokens_per_s": unavailable("model size unknown", "planner"), "bound_by": "unknown",
                "explanation": "Speed cannot be estimated without the model size.", "_active_gb": 0.0, "_active_gpu_gb": 0.0}
    if t_miss > 0.5 * total:
        bound = "storage"
    elif not hw["cpu"].get("avx2", True):
        bound = "compute"
    elif t_gpu > t_cpu:
        bound = "vram"
    else:
        bound = "memory_bandwidth"
    lo, hi = (round(1 / lo_t, 2), round(1 / hi_t, 2)) if lo_t > 0 else (0.0, 0.0)
    parts = [f"Each token reads about {(a_cpu + a_gpu) / GIB:.1f} GB of weights"
             f"{f' (only {kfrac * 100:.0f}% of the experts)' if kfrac < 1 else ''}"
             f" at an effective {ram_bw:.0f} GB/s of RAM bandwidth."]
    if g:
        saved = max(a_gpu / GIB / ram_bw - t_gpu, 0.0)  # seconds per token saved versus reading the same bytes from RAM
        parts.append(f"{a_gpu / GIB:.2f} GB of that is read on the GPU ({gpu_src}), saving about {saved * 1000:.0f} ms of "
                     f"{(total + saved) * 1000:.0f} ms per token ({saved / (total + saved) * 100:.0f}%). Prompt processing is "
                     "compute-bound and may gain more - benchmark it.")
    if miss:
        parts.append(f"About {miss * 100:.0f}% of the CPU-side reads miss RAM and come from {kind or 'unknown'} storage at "
                     f"{disk} GB/s effective ({t_miss / total * 100:.0f}% of the time per token).")
    parts.append("Estimate only - the benchmark measures the real figure.")
    return {"tokens_per_s": est([lo, hi], "planner", "[pessimistic, optimistic] generation tokens/s"),
            "bound_by": bound, "explanation": " ".join(parts), "_active_gb": (a_cpu + a_gpu) / GIB,
            "_active_gpu_gb": a_gpu / GIB}


# ------------------------------------------------------------------------------------------------ advice text
def _ram_kit(need_gb: float) -> int:
    """Smallest common total RAM size (GB) that is at least `need_gb`."""
    return next((k for k in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512) if k >= need_gb), int(need_gb) + 1)


def _smaller_quant_hint(m: dict, target_bytes: float) -> str:
    params, _ = quant.parameter_count({"size_bytes": m["size_bytes"], "file_type_name": m["ftype"]})
    if params <= 0 or target_bytes <= 0:
        return "a smaller model"
    for name in quant.COMPARE_ORDER:
        size = params * quant.QUANTS[name]["bits_per_weight"] / 8
        if size <= target_bytes * 0.97 and size < m["size_bytes"]:
            return f"a smaller quantization (about {name}, ~{size / GIB:.1f} GB) or a smaller model"
    return "a smaller model (no quantization of this one would fit)"


def _kv_hint(m: dict, ctx: int, kvl: dict, need_gb: float) -> str | None:
    kv_gb = sum(kvl["layer_bytes"]) / GIB
    if kv_gb < max(need_gb, 0.2) or ctx < 2048:
        return None
    frac = max(1 - need_gb / kv_gb, 0.0)
    return f"Lower the context from {ctx} to about {max(int(ctx * frac) // 256 * 256, 512)} tokens to free ~{min(need_gb, kv_gb):.1f} GB."


# ------------------------------------------------------------------------------------------------------ plan
def plan(hw: dict, model: dict, cfg: dict | None = None, assumptions: dict | None = None,
         model_path: str | None = None, force: bool = False) -> dict:
    """Plan and admit (or reject) loading `model` on `hw` with the partial settings `cfg`. See the module docstring.

    ``model`` is a `gguf.summarize` dict or a synthetic one; ``remote=True`` marks a model that is not downloaded yet
    (its file is not required to exist, the drive needs room for it). Result keys: model, config, requirements, tiers,
    placement, admission, speed, assumptions, status.
    """
    hw = _norm_hw(hw)
    a, m, c, ctx, ctx_note, ub, kvl, flash, cb = _build_context(hw, model, cfg, assumptions)
    ws = weights_split(m, a)
    n, total = m["block_count"], m["size_bytes"]
    blockers: list[str] = []
    warnings: list[str] = []
    suggestions: list[str] = []
    if ctx_note:
        warnings.append(ctx_note)
    if m["context_length"] and ctx > m["context_length"]:
        warnings.append(f"The context ({ctx}) is longer than the model was trained for ({m['context_length']}); quality drops.")
    for k in ("cache_type_k", "cache_type_v"):
        try:
            bytes_per_element(str(c[k]))
        except ValueError:
            blockers.append(f"Unknown KV cache type {c[k]!r} for {k}.")
    if str(c.get("flash_attn")) == "off" and str(c["cache_type_v"]).lower() not in ("f16", "bf16", "f32"):
        blockers.append("A quantized V cache needs flash attention (llama.cpp refuses to start without it).")
        suggestions.append("Turn flash attention on, or set the V cache type to f16.")

    # ---- storage / file checks
    disks = hw.get("disks") or []
    disk = hardware.disk_for_path(model_path, disks) if model_path else (disks[0] if len(disks) == 1 else None)
    kind, drive = (disk or {}).get("kind", "unknown"), (disk or {}).get("drive", "?")
    free = (disk or {}).get("free_gb")
    size_gb = total / GIB
    remote = bool(model.get("remote"))
    if model_path:
        p = Path(model_path)
        if p.suffix.lower() != ".gguf":
            blockers.append(f"{p.name} is not a .gguf file; llama.cpp only loads GGUF models.")
            suggestions.append("Pick a GGUF file, or convert the checkpoint with llama.cpp's convert_hf_to_gguf.py.")
        elif not p.exists() and not remote:
            blockers.append(f"The model file was not found: {model_path}")
            suggestions.append("Check the path in the Models tab or download the model again.")
    if total <= 0:
        blockers.append("The model size is unknown, so memory cannot be planned.")
    need_disk = remote or (bool(model_path) and not Path(model_path).exists())
    if need_disk and free is not None and total > 0:
        if free < size_gb:
            blockers.append(f"Not enough free disk space on {drive}: {free:.0f} GB free, the model needs {size_gb:.1f} GB.")
            suggestions.append("Free disk space, choose another drive for the models folder, or pick a smaller quantization.")
        elif free < size_gb + a["disk_margin_gb"]:
            warnings.append(f"Only {free - size_gb:.1f} GB would remain free on {drive} after the download.")

    # ---- GPU decision and placement
    gpu, vram_total, vram_budget = _gpu_budget_gb(hw, a)
    ngl_req = _num(c.get("n_gpu_layers"), 0)
    ngl = ngl_req if vram_budget > 0 else 0
    if ngl_req != 0 and vram_budget <= 0:
        warnings.append("GPU layers were requested but no GPU with at least 1 GB of VRAM was found; planned as CPU-only.")
    if ngl_req != 0 and vram_budget > 0 and not n:
        warnings.append("The layer count is unknown, so nothing can be placed on the GPU.")
    if not c.get("cpu_moe") and _num(c.get("n_cpu_moe"), 0) > 0 and not m["is_moe"]:
        warnings.append("n_cpu_moe has no effect: this is not a mixture-of-experts model.")
    pl = _place(m, ws, c, ngl, kvl, cb, a, flash)
    given = {k for k, v in (cfg or {}).items() if v is not None}
    if vram_budget > 0 and not _vram_fits(pl["vram"], vram_budget, a):
        patch = _autofit(m, ws, c, given, kvl, cb, a, flash, vram_budget)
        if patch:
            warnings.append("Not set by you, so adjusted to fit the GPU: " + ", ".join(
                f"{k} {c.get(k)} -> {v}" for k, v in patch.items()) + ".")
            c.update(patch)
            ngl = _num(c.get("n_gpu_layers"), 0)
            pl = _place(m, ws, c, ngl, kvl, cb, a, flash)
    vram_gb = pl["vram"] / GIB
    if pl["vram"] > 0 and not _vram_fits(pl["vram"], vram_budget, a):
        best = max_gpu_layers(hw, model, {**c, "n_gpu_layers": ngl_req}, assumptions)
        kmin = min_cpu_moe(hw, model, c, assumptions) if (m["is_moe"] and pl["w_gpu"] > pl["gpu_layers"] * ws["per_layer_shared_bytes"]
                                                         + (ws["output_bytes"] if pl["output_on_gpu"] else 0)) else None
        blockers.append(f"Not enough VRAM for this placement: it needs {vram_gb:.2f} GB on the GPU "
                        f"(+{a['safety_margin_pct']:g}% safety margin = {vram_gb * (1 + a['safety_margin_pct'] / 100):.2f} GB) "
                        f"but only {vram_budget:.2f} GB are usable ({vram_total:.1f} GB card minus {a['vram_reserve_mb']} MB reserved).")
        if kmin is not None and kmin > pl["k_cpu"]:
            suggestions.append(f"Raise n_cpu_moe to {kmin}: the experts of the first {kmin} layers stay in RAM and the "
                               "attention weights still run on the GPU.")
        else:
            suggestions.append(f"Lower n_gpu_layers to {best} (the largest value that fits)" if best else
                               "Set n_gpu_layers to 0: even one layer does not fit the GPU")
        if c.get("kv_offload") and pl["kv_gpu"]:
            suggestions[-1] += " Or turn off KV offload to keep the cache in RAM."
        if pl["staging"] and c.get("op_offload", True):
            suggestions.append("Pass --no-op-offload to stop llama.cpp copying CPU-resident experts into VRAM "
                               f"({pl['staging'] / GIB:.2f} GB assumed).")
    elif ngl and vram_budget < 3 and m["is_moe"]:
        warnings.append(f"Only {vram_total:.1f} GB of VRAM: the GPU holds attention weights at most, so the speed benefit "
                        "is small. Benchmark GPU layers against 0 before trusting them.")

    # ---- RAM tier
    ram_total = float(hw["ram"].get("total_gb") or 0)
    if ram_total <= 0:
        blockers.append("The amount of installed RAM could not be detected, so memory cannot be planned.")
    ram_budget = max(ram_total - a["os_reserve_gb"], 0.25 * ram_total)
    avail = float(hw["ram"].get("avail_gb") or 0)
    runtime = a["runtime_overhead_gb"]
    kv_ram_gb, cb_host_gb = pl["kv_ram"] / GIB, pl["cb_host"] / GIB
    weights_gb, kv_gb = total / GIB, pl["kv_total"] / GIB
    cb_gb = (pl["cb_gpu"] + pl["cb_host"]) / GIB + pl["staging"] / GIB
    model_related = weights_gb + kv_gb + cb_gb + runtime
    margin = a["safety_margin_pct"] / 100 * model_related
    room = ram_budget - kv_ram_gb - cb_host_gb - runtime - margin
    w_ram = pl["w_ram"] / GIB
    resident = min(w_ram, max(room, 0.0))
    spill = max(w_ram - resident, 0.0)
    headroom = room - w_ram
    spill_fraction = spill / weights_gb if weights_gb else 0.0
    mmap_on, mlock_on = bool(c.get("mmap", True)), bool(c.get("mlock", False))
    kind_txt = {"nvme": "NVMe", "ssd": "SSD", "hdd": "hard drive"}.get(kind, "disk")

    verdict = "ok"
    if spill > 0.005:
        need_ram = w_ram + kv_ram_gb + cb_host_gb + runtime + margin
        msg = (f"Not enough RAM: the model needs about {need_ram:.1f} GB of memory but only {ram_budget:.1f} GB are "
               f"available to it ({ram_total:.0f} GB installed minus {a['os_reserve_gb']:.0f} GB kept for the system and "
               f"other apps); {spill_fraction * 100:.0f}% of the weights ({spill:.1f} GB) would be paged from the {kind_txt}.")
        hints = [_smaller_quant_hint(m, (w_ram - spill) * GIB + pl["w_gpu"]), _kv_hint(m, ctx, kvl, spill)]
        suggestions.extend([h[0].upper() + h[1:] + "." if i == 0 else h for i, h in enumerate(hints) if h])
        ram_sug = f"Installing {_ram_kit(need_ram + a['os_reserve_gb'])} GB of RAM would hold the whole model in memory."
        if not mmap_on or mlock_on:
            verdict = "reject"
            blockers.append(msg + " With mmap off or mlock on the weights cannot be paged, so the model would not load.")
            suggestions.append("Turn mmap on and mlock off to allow paging, or use a smaller model.")
            suggestions.append(ram_sug)
        elif kind == "hdd" and spill_fraction > a["hdd_spill_tolerance"] and not force:
            verdict = "reject"
            blockers.append(msg + " On a hard drive that is unusably slow.")
            suggestions.append("Move the model to an SSD/NVMe drive (paging from SSD is ~12-45x faster than from a hard drive).")
            suggestions.append(ram_sug)
        elif spill_fraction > a["max_spill_fraction"] and not force:
            verdict = "reject"
            blockers.append(msg + f" That is more than the {a['max_spill_fraction'] * 100:.0f}% limit.")
            suggestions.append(ram_sug)
            suggestions.append("Or use a smaller model/quantization that fits in memory.")
        else:
            verdict = "paging"
            warnings.append(msg + (" Forced past the safety limits." if force and (
                spill_fraction > a["max_spill_fraction"] or (kind == "hdd" and spill_fraction > a["hdd_spill_tolerance"])) else
                " It will load but generate slowly."))
            suggestions.append("Close other applications to free RAM; keep the context small.")
            suggestions.append(ram_sug)
            if kind == "hdd":
                suggestions.append("Move the model to an SSD/NVMe drive.")
    elif ram_budget <= 0 or headroom / ram_budget * 100 < a["headroom_ok_pct"]:
        verdict = "tight"
        warnings.append(f"The model fits but leaves only {max(headroom, 0):.1f} GB of the {ram_budget:.1f} GB budget free; "
                        "other applications may push it into paging.")
    if blockers:
        verdict = "reject"
    if avail and verdict in ("ok", "tight") and w_ram + kv_ram_gb + cb_host_gb + runtime > avail:
        warnings.append(f"Only {avail:.1f} GB of RAM is free right now (about {w_ram + kv_ram_gb + cb_host_gb + runtime:.1f} GB "
                        "needed); close other applications before loading.")
    if kind == "hdd" and verdict in ("ok", "tight"):
        warnings.append(f"The model is on {drive} (hard drive): the first load takes minutes, then it runs from RAM.")
    if not mmap_on and verdict != "reject":
        warnings.append("mmap is off: the whole model is read into RAM at start, which is slower.")
    if not hw["cpu"].get("avx2", True):
        warnings.append("This CPU has no AVX2; llama.cpp will be very slow.")

    speed = _speed(hw, m, ws, pl, kvl, a, gpu, kind, spill)
    active_gb, active_gpu_gb = speed.pop("_active_gb"), speed.pop("_active_gpu_gb")
    if verdict == "reject":
        speed["explanation"] = "Not loadable as configured - " + speed["explanation"]

    # ---- assemble
    ram_planned = resident + kv_ram_gb + cb_host_gb + runtime
    mock = bool(hw.get("mock"))
    src = "what-if hardware profile" if mock else "hardware detection"
    phys = (lambda v, s: est(v, s, "what-if profile, not this machine")) if mock else (lambda v, s: measured(v, s))
    placement_cfg = {"ctx_size": ctx, "n_gpu_layers": ngl, "n_cpu_moe": pl["k_cpu"], "kv_offload": bool(c.get("kv_offload")),
                     "cache_type_k": str(c["cache_type_k"]), "cache_type_v": str(c["cache_type_v"]), "ubatch_size": ub,
                     "mmap": mmap_on, "mlock": mlock_on}
    vram_items = [{"label": "weights on GPU", "gb": round(pl["w_gpu"] / GIB, 3)},
                  {"label": "KV cache on GPU", "gb": round(pl["kv_gpu"] / GIB, 3)},
                  {"label": "compute buffer", "gb": round(pl["cb_gpu"] / GIB, 3)},
                  {"label": "op-offload staging", "gb": round(pl["staging"] / GIB, 3)}]
    ram_items = [{"label": "weights resident in RAM (mmap)" if mmap_on else "weights in RAM", "gb": round(resident, 3)},
                 {"label": "KV cache in RAM", "gb": round(kv_ram_gb, 3)},
                 {"label": "compute buffer (host)", "gb": round(cb_host_gb, 3)},
                 {"label": "llama.cpp runtime", "gb": round(runtime, 3)}]
    e = lambda v, note="": est(round(v, 3), "planner", note)
    result = {
        "model": {"name": m["name"] or Path(str(model.get("path") or "")).stem, "architecture": m["architecture"],
                  "size_gb": round(size_gb, 2), "is_moe": m["is_moe"], "block_count": n, "expert_count": m["expert_count"],
                  "expert_used_count": m["expert_used_count"], "quant": m["quant"], "path": model_path or model.get("path")},
        "config": placement_cfg,
        "requirements": {"weights_gb": e(weights_gb, "whole file"), "kv_cache_gb": e(kv_gb), "compute_buffers_gb": e(cb_gb),
                         "runtime_overhead_gb": e(runtime), "os_reserve_gb": e(a["os_reserve_gb"]),
                         "safety_margin_gb": e(margin, f"{a['safety_margin_pct']}% of weights+KV+compute+runtime"),
                         "total_gb": e(model_related + margin + a["os_reserve_gb"])},
        "tiers": {
            "vram": {"physical_gb": phys(round(vram_total, 2), src), "budget_gb": e(vram_budget), "planned_gb": e(vram_gb),
                     "items": vram_items},
            "ram": {"physical_gb": phys(ram_total, src), "available_gb": (est(avail, src, "what-if profile") if mock else
                    measured(avail, "hardware detection")), "budget_gb": e(ram_budget), "planned_gb": e(ram_planned),
                    "items": ram_items},
            "storage": {"drive": drive, "kind": kind, "free_gb": free, "mmap_resident_gb": e(resident),
                        "spill_gb": e(spill), "spill_fraction": est(round(spill_fraction, 4), "planner"),
                        "read_gbs": est(tuner.DISK_GBS.get(kind, tuner.DISK_GBS["unknown"]), "tuner.DISK_GBS",
                                        "effective random-read throughput while paging"),
                        "model_on_slow_disk": kind == "hdd"},
            "disk_cache": {"enabled": False, "note": "optional staging cache, see diskcache.py"}},
        "placement": {"gpu_layers": pl["gpu_layers"], "cpu_layers": pl["cpu_layers"],
                      "moe_experts_on_cpu_layers": pl["k_cpu"], "kv_on_gpu": bool(pl["kv_gpu"]),
                      "weights_vram_gb": round(pl["w_gpu"] / GIB, 3), "weights_ram_gb": round(w_ram, 3),
                      "weights_spill_gb": round(spill, 3)},
        "admission": {"verdict": verdict, "can_load": verdict != "reject", "blockers": blockers, "warnings": warnings,
                      "suggestions": list(dict.fromkeys(suggestions)), "headroom_gb": round(headroom, 2)},
        "speed": speed,
        "assumptions": _assumption_list(a, assumptions, m, ws, kvl, hw, gpu, kind, pl, cb),
        "status": "estimated",
    }
    result["assumptions"].append({"key": "speed.active_gb_per_token", "value": {"total": round(active_gb, 3), "on_gpu": round(active_gpu_gb, 3)},
                                  "explain": "Weight and KV bytes read per generated token (GiB); the basis of the speed estimate."})
    return result


def _assumption_list(a: dict, given: dict | None, m: dict, ws: dict, kvl: dict, hw: dict, gpu: dict | None, kind: str,
                     pl: dict, cb: int) -> list[dict]:
    out = []
    for key, value in a.items():
        note = ASSUMPTION_DOCS.get(key, "")
        if given and key in given and given[key] != DEFAULT_ASSUMPTIONS.get(key):
            note += " (overridden by the caller)"
        out.append({"key": key, "value": value, "explain": note})
    bw, bw_src = _gpu_bandwidth(gpu, a)
    facts = [
        ("unit", "GiB", "Every 'GB' in a plan is 2**30 bytes, the unit Windows shows."),
        ("model.layers", m["block_count"], "block_count from the GGUF header (0 = unknown: nothing can be offloaded)."),
        ("model.experts", f"{m['expert_used_count']} of {m['expert_count']}" if m["is_moe"] else "dense",
         "Experts used per token / experts per layer."),
        ("model.quantization", m["quant"] or "unknown", "general.file_type of the file; sets the bits per weight of the split."),
        ("model.weight_split", {"basis": ws["basis"], "expert_fraction": round(ws["expert_fraction"], 4),
                                "tied_embeddings": ws["tied_embeddings"]},
         "How the file's bytes are divided between experts, per-layer tensors and embeddings (see memcalc.weights_split)."),
        ("model.swa", {"layers": kvl["swa_layers"], "window": m["sliding_window"], "cells_swa": kvl["cells_swa"],
                       "cells_global": kvl["cells_full"]},
         "Sliding-window layers keep n_swa + ubatch cells (llama.cpp llama_kv_cache_iswa)."),
        ("ram.bandwidth_gbs", hw["ram"].get("bandwidth_gbs"), "Theoretical RAM bandwidth (speed x 8 bytes x channels)."),
        ("storage.read_gbs", tuner.DISK_GBS.get(kind, tuner.DISK_GBS["unknown"]), f"tuner.DISK_GBS for a {kind} drive."),
        ("gpu.bandwidth_gbs", bw if gpu else None, bw_src if gpu else "no GPU"),
        ("compute_buffer_mib", round(cb / 2**20, 1), "Estimated llama.cpp compute buffer for the micro-batch."),
    ]
    out += [{"key": k, "value": v, "explain": e_} for k, v, e_ in facts]
    out += [{"key": "kv.note", "value": n, "explain": "Fallback used for the KV estimate."} for n in kvl["notes"]]
    return out
