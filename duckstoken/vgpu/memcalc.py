"""Byte-level memory formulas used by the planner: KV cache, weight split, compute buffers, llama.cpp log parsing.

Pure functions. Every constant lives in `DEFAULT_ASSUMPTIONS` with an explanation in `ASSUMPTION_DOCS`; the planner
copies both into each plan so the user can see (and change) what an estimate rests on.
"""

from __future__ import annotations

import re

from . import quant

GIB = 2**30
MIB = 2**20

DEFAULT_ASSUMPTIONS: dict = {
    "os_reserve_gb": 6.0,
    "safety_margin_pct": 5.0,
    "vram_reserve_mb": 400,
    "runtime_overhead_gb": 0.5,
    "max_spill_fraction": 0.5,
    "swa_full": False,
    "ubatch_default": 512,
    "kv_k_type": "q8_0",
    "kv_v_type": "q8_0",
    "hdd_spill_tolerance": 0.05,
    # --- constants below are not in the contract list but are just as binding on the numbers
    "headroom_ok_pct": 10.0,
    "disk_margin_gb": 5.0,
    "host_compute_fraction": 0.25,
    "op_offload_staging_tensors": 1,
    "kv_read_fill_fraction": 0.5,
    "ram_bw_efficiency": 0.6,
    "speed_efficiency_lo": 0.35,
    "speed_efficiency_hi": 0.7,
    "page_cache_skew": 0.8,
    "storage_speed_lo": 0.6,
    "storage_speed_hi": 1.4,
    "vocab_fallback": 128000,
    "tied_embedding_threshold": 0.35,
    "shared_bits_mxfp4_file": 8.5,
    "gqa_fallback_ratio": 4,
    "kv_pad_cells": 256,
    "compute_floats_per_embd": 5.0,
    "compute_floats_per_ff": 3.0,
    "compute_mask_bytes": 2,
    "gpu_bandwidth_gbs": None,
}

ASSUMPTION_DOCS: dict = {
    "os_reserve_gb": "RAM kept for Windows and other applications. Subtracted from total RAM to get the model's budget.",
    "safety_margin_pct": "Extra margin on the model-related memory (weights + KV + compute + runtime), covering estimate "
                         "error and allocator padding. It is not applied to the OS reserve.",
    "vram_reserve_mb": "VRAM not offered to the model: desktop compositor, driver and CUDA/Vulkan context.",
    "runtime_overhead_gb": "llama-server's own heap, tokenizer, sampler state and thread stacks (estimate).",
    "max_spill_fraction": "Largest fraction of the weights allowed to exceed the RAM+VRAM budget and be paged from "
                          "storage. Above this the plan is rejected.",
    "swa_full": "False = sliding-window layers keep only n_swa+ubatch cells (llama.cpp default); True = --swa-full.",
    "ubatch_default": "Micro-batch used when the settings do not say (llama-server default).",
    "kv_k_type": "KV cache K type used when the settings do not say.",
    "kv_v_type": "KV cache V type used when the settings do not say.",
    "hdd_spill_tolerance": "On a hard drive any paging above this fraction of the weights is rejected (random reads ~50x "
                           "slower than an SSD) unless forced.",
    "headroom_ok_pct": "Verdict 'ok' needs at least this much of the RAM budget left over; below it the plan is 'tight'.",
    "disk_margin_gb": "Free space that should remain on the drive after a download (warning only).",
    "host_compute_fraction": "When layers run on the GPU llama.cpp still keeps a smaller compute buffer in host RAM; "
                             "assumed to be this fraction of the GPU one. Unmeasured: the load log shows the real value.",
    "op_offload_staging_tensors": "With op-offload on (llama.cpp default) CPU-resident weights are copied into the GPU "
                                  "compute buffer for batched work. Assumed number of the largest expert/FFN tensors "
                                  "held at once; a lower bound. --no-op-offload removes it.",
    "kv_read_fill_fraction": "Share of the context assumed filled when estimating how much KV cache is read per token.",
    "ram_bw_efficiency": "Share of theoretical RAM bandwidth that llama.cpp reaches (same constant as tuner.py).",
    "speed_efficiency_lo": "Pessimistic share of that bandwidth actually achieved during generation (tuner.py).",
    "speed_efficiency_hi": "Optimistic share of that bandwidth actually achieved during generation (tuner.py).",
    "page_cache_skew": "Fraction of the uniform-random miss rate assumed for expert reads from storage; <1 because "
                       "routing is skewed towards hot experts (tuner.py uses the same 0.8).",
    "storage_speed_lo": "Storage throughput multiplier for the pessimistic end of the range.",
    "storage_speed_hi": "Storage throughput multiplier for the optimistic end of the range.",
    "vocab_fallback": "Vocabulary size used to size embedding tensors when the file does not state it.",
    "tied_embedding_threshold": "If two untied embedding matrices would be more than this share of the file they are "
                                "assumed tied (one copy).",
    "shared_bits_mxfp4_file": "Bits per weight of attention/embedding tensors in MXFP4_MOE files (Q8_0). Matches the "
                              "official gpt-oss-20b file size within 1%.",
    "gqa_fallback_ratio": "KV heads = attention heads / this when the file states neither (only used for the estimate).",
    "kv_pad_cells": "llama.cpp pads KV cell counts to multiples of this.",
    "compute_floats_per_embd": "Compute-buffer fit: float32 values per ubatch token per hidden unit. Fitted to the "
                               "'compute buffer size' lines of a real CPU llama-server (5 dense models, flash attention "
                               "on, ubatch 128/512/1024): matches within 0.1%. GPU backends were not measurable here.",
    "compute_floats_per_ff": "Compute-buffer fit: float32 values per ubatch token per FFN unit (dense width, or "
                             "experts-used x expert width for MoE; the MoE extension could not be validated here).",
    "compute_mask_bytes": "Bytes per attention-mask cell with flash attention (f16 mask, measured). Without flash "
                          "attention llama.cpp uses an f32 mask plus f32 score matrices for every head; that is "
                          "measured to within +0/-15% at 4k context and exact in slope.",
    "gpu_bandwidth_gbs": "Memory bandwidth of the GPU used for layers placed on it. None = look up by name, else assumed.",
}

# SWA layer pattern per architecture: layer il is sliding-window when il % p < p - 1 (llama.cpp load_swa_pattern; source
# src/models/*.cpp at commit 08ffba2). Architectures with a sliding_window key but not listed are treated as all-global
# (an upper bound on KV size).
SWA_PATTERN = {"gpt-oss": 2, "gemma2": 2, "gemma3": 6, "gemma3n": 5, "cohere2": 4, "exaone4": 4, "exaone-moe": 4,
               "plamo3": 8, "afmoe": 4, "olmo2": 4, "llama4": 4, "gemma-embedding": 6, "mellum": 4}

_BPE = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32,
        "q4_0": 18 / 32, "iq4_nl": 18 / 32}


def bytes_per_element(ggml_type: str) -> float:
    """Bytes per tensor element for a KV-cache type (ggml block sizes: q8_0 = 34 bytes per 32 values, ...)."""
    key = str(ggml_type).lower()
    if key not in _BPE:
        raise ValueError(f"unknown ggml type {ggml_type!r}")
    return _BPE[key]


# ------------------------------------------------------------------------------------------------ model facts
def norm_model(model: dict) -> dict:
    """Canonical numeric view of a model dict (gguf.summarize output or a synthetic one). Missing facts become 0."""
    def i(key, default=0):
        try:
            return int(model.get(key) or default)
        except (TypeError, ValueError):
            return default

    size = i("size_bytes") or int(float(model.get("size_gb") or 0) * GIB)
    heads, kv_heads, embd = i("head_count"), i("head_count_kv"), i("embedding_length")
    experts = i("expert_count")
    key_len = i("key_length") or (embd // heads if heads and embd else 0)
    val_len = i("value_length") or key_len
    return {
        "_norm": True,
        "size_bytes": size, "name": str(model.get("name") or ""), "architecture": str(model.get("architecture") or ""),
        "block_count": i("block_count"), "embedding_length": embd, "head_count": heads, "head_count_kv": kv_heads,
        "key_length": key_len, "value_length": val_len, "expert_count": experts,
        "expert_used_count": i("expert_used_count"), "is_moe": bool(model.get("is_moe", experts > 1)) and experts > 1,
        "ff": i("feed_forward_length"), "ff_exp": i("expert_feed_forward_length"), "sliding_window": i("sliding_window"),
        "vocab_size": i("vocab_size"), "context_length": i("context_length"),
        "quant": quant.quant_name(model), "ftype": model.get("file_type_name") or quant.ftype_name(model.get("file_type")),
    }


def _n(model: dict) -> dict:
    return model if model.get("_norm") else norm_model(model)


def swa_flags(m: dict) -> list[bool]:
    """Per layer: True when the layer uses sliding-window attention."""
    n, p = m["block_count"], SWA_PATTERN.get(m["architecture"], 0)
    if not (n and p and m["sliding_window"] > 0):
        return [False] * n
    return [il % p < p - 1 for il in range(n)]


# --------------------------------------------------------------------------------------------------- KV cache
def kv_layout(m: dict, ctx: int, k_type: str, v_type: str, assumptions: dict, parallel: int = 1,
              ubatch: int = 512) -> dict:
    """Per-layer KV bytes. Returns {"layer_bytes": [..], "per_cell_layer": float, "cells_full", "cells_swa", "notes": []}."""
    a = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    n, pad = m["block_count"], int(a["kv_pad_cells"])
    notes: list[str] = []
    kv_heads = m["head_count_kv"] or (max(m["head_count"] // int(a["gqa_fallback_ratio"]), 1) if m["head_count"] else 0)
    key_len, val_len = m["key_length"], m["value_length"]
    if not kv_heads and m["embedding_length"]:  # nothing about attention in the file at all
        kv_heads, key_len = max(m["embedding_length"] // 128 // int(a["gqa_fallback_ratio"]), 1), 128
        val_len = key_len
        notes.append("Attention shape missing from the file: assumed 128-wide heads and a "
                     f"{a['gqa_fallback_ratio']}:1 grouped-query ratio.")
    per_cell = kv_heads * (key_len * bytes_per_element(k_type) + val_len * bytes_per_element(v_type))
    cells_full = -(-max(int(ctx), 1) // pad) * pad
    flags = swa_flags(m)
    cells_swa = cells_full
    if any(flags) and not a["swa_full"]:
        cells_swa = min(cells_full, -(-(m["sliding_window"] * max(parallel, 1) + int(ubatch)) // pad) * pad)
    layers = [per_cell * (cells_swa if f else cells_full) for f in flags] if n else []
    return {"layer_bytes": layers, "per_cell_layer": per_cell, "cells_full": cells_full, "cells_swa": cells_swa,
            "swa_layers": sum(flags), "notes": notes}


def kv_cache_bytes(model: dict, ctx: int, k_type: str = "q8_0", v_type: str = "q8_0", assumptions: dict | None = None,
                   parallel: int = 1, ubatch: int = 512) -> tuple[int, float, dict]:
    """(total_bytes, per_token_bytes, breakdown).

    GQA-aware (KV heads x head length) and sliding-window aware: layers that use SWA keep only ``n_swa + ubatch`` cells
    (llama.cpp ``llama_kv_cache_iswa``) unless ``swa_full``. ``per_token_bytes`` is the growth per extra context token
    (the global-attention layers); SWA layers do not grow past their window.
    """
    m = _n(model)
    lay = kv_layout(m, ctx, k_type, v_type, assumptions, parallel, ubatch)
    total = int(sum(lay["layer_bytes"]))
    global_layers = len(lay["layer_bytes"]) - lay["swa_layers"]
    breakdown = {"layers": m["block_count"], "global_layers": global_layers, "swa_layers": lay["swa_layers"],
                 "cells_global": lay["cells_full"], "cells_swa": lay["cells_swa"],
                 "bytes_per_cell_per_layer": lay["per_cell_layer"], "k_type": k_type, "v_type": v_type,
                 "swa_full": bool((assumptions or {}).get("swa_full", False)), "notes": lay["notes"]}
    return total, lay["per_cell_layer"] * global_layers, breakdown


# ---------------------------------------------------------------------------------------------------- weights
def weights_split(model: dict, assumptions: dict | None = None) -> dict:
    """Split the file's bytes into experts, per-layer shared tensors and embeddings.

    MoE: expert parameters = layers x experts x 3 x hidden x expert_ff (gate, up, down). The experts' share of the file
    is that parameter count at the file's expert bits-per-weight, against the shared tensors at their own bits-per-weight,
    scaled to the real file size (so the split never exceeds ``size_bytes``). Without those dims the share falls back to
    ``0.97 * E / (E + 1)`` (capped 0.96). Dense models have no expert bytes.
    """
    a = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    m = _n(model)
    total, n, embd, E = m["size_bytes"], m["block_count"], m["embedding_length"], m["expert_count"]
    ftype = m["ftype"]
    b_exp = quant.bits_per_weight(ftype, 0.0) or 4.5
    b_shared = float(a["shared_bits_mxfp4_file"]) if ftype == "MXFP4_MOE" else b_exp
    vocab = m["vocab_size"] or int(a["vocab_fallback"])
    # embeddings: token_embd (always host) + output (lm head); tied models keep a single copy
    one = vocab * embd * b_shared / 8 if embd else 0.0
    tied = bool(one) and total > 0 and 2 * one > a["tied_embedding_threshold"] * total
    emb_total = min(one if tied else 2 * one, 0.6 * total)
    output = emb_total / 2 if not tied else one
    token_embd = emb_total - output if not tied else one
    out = {"total_bytes": total, "tied_embeddings": tied, "basis": "dense"}
    if m["is_moe"] and total > 0:
        if m["ff_exp"] and embd and n:
            p_exp = n * E * 3 * embd * m["ff_exp"]
            # attention + router parameters per layer (approximate; sizes the shared side of the split)
            attn = n * (embd * (m["head_count"] * m["key_length"] + 2 * m["head_count_kv"] * m["key_length"]
                                + m["head_count"] * m["value_length"]) + embd * E) if m["head_count"] else n * 4 * embd * embd
            p_shared = attn + (vocab * embd * (1 if tied else 2))
            e_bits, s_bits = p_exp * b_exp, p_shared * b_shared
            frac = e_bits / (e_bits + s_bits) if (e_bits + s_bits) else 0.0
            out["basis"] = "dims"
        else:
            frac = min(0.96, 0.97 * E / (E + 1))
            out["basis"] = "ratio"
        expert = int(total * frac)
    else:
        expert = 0
    shared = total - expert
    emb_total = min(emb_total, shared)
    layers_shared = max(shared - emb_total, 0)
    out.update(shared_bytes=shared, expert_bytes=expert, embedding_bytes=int(emb_total),
               token_embd_bytes=int(min(token_embd, emb_total)), output_bytes=int(min(output, emb_total)),
               per_layer_shared_bytes=layers_shared / n if n else 0.0, per_layer_expert_bytes=expert / n if n else 0.0,
               expert_fraction=expert / total if total else 0.0)
    return out


# ------------------------------------------------------------------------------------------------ compute buffer
def compute_buffer_bytes(model: dict, ctx: int, ubatch: int, on_gpu: bool = False, assumptions: dict | None = None,
                         flash_attn: bool = True) -> int:
    """Estimated size of llama.cpp's compute (graph scratch) buffer for one micro-batch.

    ``ubatch * 4 B * (5 * hidden + 3 * ffn_width)`` + the attention mask (``ubatch x ctx`` f16 cells) and, when flash
    attention is off, an f32 mask plus f32 score matrices for every head. The constants were fitted to the "compute
    buffer size" lines printed by a real CPU llama-server for five dense models (flash attention on, ubatch 128/512/1024,
    ctx 1k-32k; see the constants' docs). ``ffn_width`` is the dense FFN width, or experts-used x expert width
    for MoE (an extension that could not be validated without an MoE model). GPU buffers use the same formula: llama.cpp
    prints the real figure at load and the benchmark compares it with this estimate. ``on_gpu`` only labels the intent.
    """
    a = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    m = _n(model)
    embd = m["embedding_length"]
    if not embd:
        return 0
    ub = max(1, min(int(ubatch), int(ctx) if ctx else int(ubatch)))
    if m["is_moe"] and m["ff_exp"]:
        ff = max(m["expert_used_count"], 1) * m["ff_exp"] + m["ff"]
    else:
        ff = m["ff"] or int(3.5 * embd)
    total = ub * 4 * (a["compute_floats_per_embd"] * embd + a["compute_floats_per_ff"] * ff)
    cells = -(-max(int(ctx), 1) // int(a["kv_pad_cells"])) * int(a["kv_pad_cells"])
    if flash_attn:
        total += ub * cells * a["compute_mask_bytes"]
    else:  # f32 mask + f32 KQ scores for every head; an upper bound at short context (measured +0..15%)
        total += ub * cells * (4 + 4 * max(m["head_count"], 1))
    return int(total)


# --------------------------------------------------------------------------------------------- log parsing
_BUF = re.compile(r"(\S+)\s+(model|KV|compute|output|RS)\s+buffer size\s*=\s*([\d.]+)\s*MiB")
_OLD_MODEL = re.compile(r"load_tensors:\s+(\S+)\s+buffer size\s*=\s*([\d.]+)\s*MiB")  # builds without the word "model"
_KINDS = {"model": "model_buffers", "KV": "kv_buffers", "compute": "compute_buffers", "output": "output_buffers",
          "RS": "rs_buffers"}


def parse_buffer_log(lines) -> dict:
    """Buffer sizes (MiB, summed per device) from llama.cpp's load log.

    Understands ``load_tensors:   CPU_Mapped model buffer size =  325.02 MiB``, ``llama_kv_cache: CPU KV buffer size =
    12.75 MiB`` (several lines per device are added up: base + SWA caches), ``sched_reserve: CPU compute buffer size =
    39.30 MiB`` and the older ``llama_new_context_with_model`` / ``llm_load_tensors`` spellings, with or without the
    time-stamp prefix. Lines that do not match are ignored. Accepts an iterable of lines or one multi-line string.
    """
    out = {v: {} for v in _KINDS.values()}
    if isinstance(lines, str):
        lines = lines.splitlines()
    for line in lines or []:
        if not isinstance(line, str):
            continue
        mt = _BUF.search(line)
        if mt:
            dev, kind, mib = mt.group(1), mt.group(2), float(mt.group(3))
        else:
            mo = _OLD_MODEL.search(line)
            if not mo:
                continue
            dev, kind, mib = mo.group(1), "model", float(mo.group(2))
        bucket = out[_KINDS[kind]]
        bucket[dev] = round(bucket.get(dev, 0.0) + mib, 2)
    return out


_HOST_PREFIXES = ("CPU", "AMX", "CUDA_Host", "HIP_Host", "Vulkan_Host", "SYCL_Host", "ROCm_Host", "Metal_Mapped")


def is_host_device(name: str) -> bool:
    """True for buffers that live in system RAM (CPU, CPU_Mapped, AMX, pinned *_Host, ...)."""
    return name.startswith(_HOST_PREFIXES)


def measured_tiers_gb(parsed: dict) -> dict:
    """Collapse `parse_buffer_log` output into {"ram_gb": {"model","kv","compute"}, "vram_gb": {...}} (GiB)."""
    res = {"ram_gb": {"model": 0.0, "kv": 0.0, "compute": 0.0}, "vram_gb": {"model": 0.0, "kv": 0.0, "compute": 0.0}}
    for bucket, key in (("model_buffers", "model"), ("kv_buffers", "kv"), ("compute_buffers", "compute")):
        for dev, mib in (parsed.get(bucket) or {}).items():
            res["ram_gb" if is_host_device(dev) else "vram_gb"][key] += mib / 1024
    return {t: {k: round(v, 3) for k, v in d.items()} for t, d in res.items()}
