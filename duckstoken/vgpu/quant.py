"""Quantization table: what each llama.cpp file type costs in bits per weight, and what it does to quality.

Sources (all checked against the local llama.cpp checkout at commit 08ffba2):
  * ftype ids: ``include/llama.h`` (enum llama_ftype).
  * bits per weight: ``tools/quantize/README.md``. For the K- and I-quants these are *whole-file* figures measured on
    Llama-3.1-8B (they include the larger embedding/output tensors), so they predict file size well for ordinary dense
    models. Legacy block formats (Q4_0 ... Q5_1, MXFP4, NVFP4, TQ*) use the exact block layout from ``ggml-common.h``.
  * quality: qualitative text only. No numeric quality score is claimed - perplexity deltas in llama.cpp's README were
    measured on one model and do not transfer to others.

Nothing here converts anything; it only describes formats and estimates sizes.
"""

from __future__ import annotations

from pathlib import Path

from .values import est

GIB = 2**30

# enum llama_ftype (include/llama.h). Ids 4, 5, 6, 33, 34, 35 were removed upstream.
FTYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
    12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
    26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
}
FTYPE_GUESSED = 1024  # "not specified in the model file": llama.cpp ORs this flag in

_NO_DIRECT_NOTE = ("Quality is described qualitatively on purpose: llama.cpp's own perplexity tables are for one model "
                   "and do not transfer.")

# Quality text per family. Ordered roughly from best to worst.
_Q = {
    "ref": "Reference precision (no quantization loss); largest files.",
    "vlow": "Very small quality loss.",
    "low": "Small quality loss.",
    "mid": "Small to moderate quality loss; the usual size/quality sweet spot.",
    "mod": "Moderate quality loss; noticeable on hard reasoning and code.",
    "high": "Large quality loss; small models can become unreliable.",
    "sev": "Severe quality loss unless the model was trained or tuned for this format.",
}


def _q(bpw: float, family: str, quality: str, *, cuda: bool = True, vulkan: bool = True, metal: bool = True,
       sycl: bool = True, note: str = "", source: str = "llama.cpp tools/quantize/README.md") -> dict:
    return {"bits_per_weight": bpw, "family": family, "qualitative": _Q[quality], "quality_level": quality,
            "backend_support": {"cpu": True, "cuda": cuda, "vulkan": vulkan, "metal": metal, "sycl": sycl},
            "note": note, "source": source}


_BLOCK = "exact block layout in ggml-common.h"
# name -> record. Keys are llama-quantize names; MXFP4_MOE is also reachable as "MXFP4".
QUANTS: dict[str, dict] = {
    "F32": _q(32.0, "float", "ref", source="IEEE float32"),
    "F16": _q(16.0005, "float", "ref"),
    "BF16": _q(16.0, "float", "ref", source="bfloat16"),
    "Q8_0": _q(8.5008, "legacy", "vlow"),
    "Q6_K": _q(6.5633, "k-quant", "vlow"),
    "Q5_K_M": _q(5.7036, "k-quant", "low"),
    "Q5_K_S": _q(5.5704, "k-quant", "low"),
    "Q5_1": _q(6.0, "legacy", "low", source=_BLOCK),
    "Q5_0": _q(5.5, "legacy", "low", source=_BLOCK),
    "Q4_K_M": _q(4.8944, "k-quant", "mid", note="Most widely published 4-bit variant."),
    "Q4_K_S": _q(4.6672, "k-quant", "mid"),
    "Q4_1": _q(5.0, "legacy", "mid", source=_BLOCK),
    "Q4_0": _q(4.5, "legacy", "mid", source=_BLOCK),
    "IQ4_NL": _q(4.6818, "i-quant", "mid"),
    "IQ4_XS": _q(4.4597, "i-quant", "mid"),
    "MXFP4_MOE": _q(4.25, "mxfp4", "mid", source=_BLOCK + " (32 values + 1 scale byte = 17 bytes)",
                    note="Native release format of gpt-oss: its MoE weights were post-trained in MXFP4. Other tensors "
                         "(attention, embeddings) stay in higher precision, so the whole file is larger than 4.25 bpw."),
    "NVFP4": _q(4.5, "nvfp4", "mid", source=_BLOCK + " (64 values + 4 scale bytes = 36 bytes)"),
    "Q3_K_L": _q(4.2979, "k-quant", "mod"),
    "Q3_K_M": _q(3.9960, "k-quant", "mod"),
    "Q3_K_S": _q(3.6429, "k-quant", "mod"),
    "IQ3_M": _q(3.7628, "i-quant", "mod"),
    "IQ3_S": _q(3.6606, "i-quant", "mod"),
    "IQ3_XS": _q(3.4977, "i-quant", "mod"),
    "IQ3_XXS": _q(3.2548, "i-quant", "high"),
    "Q2_K": _q(3.1593, "k-quant", "high"),
    "Q2_K_S": _q(2.9697, "k-quant", "high"),
    "IQ2_M": _q(2.9294, "i-quant", "high"),
    "IQ2_S": _q(2.7403, "i-quant", "high"),
    "IQ2_XS": _q(2.5882, "i-quant", "high"),
    "IQ2_XXS": _q(2.3824, "i-quant", "high"),
    "Q2_0": _q(2.25, "legacy", "sev", source="llama-quantize description: 2.25 bpw (group 64)"),
    "IQ1_M": _q(2.1460, "i-quant", "sev"),
    "IQ1_S": _q(2.0042, "i-quant", "sev"),
    "TQ2_0": _q(2.0625, "ternary", "sev", cuda=False, source=_BLOCK + "; no CUDA references found in ggml-cuda",
                note="Ternary format: only useful for models trained as ternary (BitNet-style)."),
    "TQ1_0": _q(1.6875, "ternary", "sev", cuda=False, source=_BLOCK + "; no CUDA references found in ggml-cuda",
                note="Ternary format: only useful for models trained as ternary (BitNet-style)."),
    "Q1_0": _q(1.125, "legacy", "sev", source="llama-quantize description: 1.125 bpw"),
}
QUANTS["MXFP4"] = QUANTS["MXFP4_MOE"]  # display alias used by planner/gpt-oss code

# Order used by compare_quants (smallest file last).
COMPARE_ORDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q3_K_M", "Q2_K", "IQ2_M"]

_CONVERT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".pkl", ".msgpack", ".h5"}


def ftype_name(file_type) -> str | None:
    """Name for a general.file_type value (int or numeric string); None when unknown or not given."""
    try:
        ft = int(file_type)
    except (TypeError, ValueError):
        return None
    return FTYPE_NAMES.get(ft & ~FTYPE_GUESSED)


def display_name(name: str | None) -> str | None:
    """MXFP4_MOE is shown as MXFP4 (the MoE part of the file is MXFP4; see QUANTS note)."""
    return "MXFP4" if name == "MXFP4_MOE" else name


def quant_name(model: dict | None) -> str | None:
    """Display quantization name of a model dict (from `file_type_name`, else `file_type`); None when unknown."""
    if not model:
        return None
    name = model.get("file_type_name") or ftype_name(model.get("file_type"))
    return display_name(str(name)) if name else None


def bits_per_weight(name: str | None, default: float | None = None) -> float | None:
    rec = QUANTS.get(name or "")
    return rec["bits_per_weight"] if rec else default


def needs_conversion(source: str | Path) -> bool:
    """True for checkpoints llama.cpp cannot read directly (Hugging Face safetensors/PyTorch folders); False for GGUF."""
    p = Path(str(source))
    if p.suffix.lower() == ".gguf":
        return False
    if p.suffix.lower() in _CONVERT_SUFFIXES:
        return True
    try:
        if p.is_dir():
            names = {x.name.lower() for x in p.iterdir()}
            if any(n.endswith(".gguf") for n in names):
                return False
            return "config.json" in names or any(Path(n).suffix in _CONVERT_SUFFIXES for n in names)
    except OSError:
        pass
    return True  # unknown format: do not claim it can be loaded directly


def parameter_count(model_or_params) -> tuple[float, str]:
    """(parameters, how) from a model dict (file size / bits per weight of its quant) or a plain number."""
    if isinstance(model_or_params, (int, float)) and not isinstance(model_or_params, bool):
        return float(model_or_params), "given"
    m = model_or_params or {}
    size = m.get("size_bytes") or (m.get("size_gb") or 0) * GIB
    name = m.get("file_type_name") or ftype_name(m.get("file_type"))
    bpw = bits_per_weight(name)
    if not size:
        return 0.0, "unknown: model has no size"
    if bpw is None:
        return 0.0, "unknown: file type not recognised"
    return size * 8 / bpw, f"file size / {bpw} bits per weight ({name})"


def compare_quants(model_or_params, hw: dict, ctx: int = 4096, assumptions: dict | None = None,
                   names: list[str] | None = None) -> list[dict]:
    """The same parameter count in other quantizations: size, memory need and admission verdict on `hw`.

    Everything is an estimate (`file_gb` assumes the uniform bits-per-weight of the table); no quality number is given.
    """
    from . import planner  # late import: planner imports this module for the bits-per-weight table

    params, how = parameter_count(model_or_params)
    base = dict(model_or_params) if isinstance(model_or_params, dict) else {}
    current = quant_name(base)
    out = []
    if params <= 0:
        return out
    for name in names or COMPARE_ORDER:
        rec = QUANTS.get(name)
        if not rec:
            continue
        size_bytes = int(params * rec["bits_per_weight"] / 8)
        synth = dict(base)
        ft = next((k for k, v in FTYPE_NAMES.items() if v == name), None)
        synth.update(size_bytes=size_bytes, size_gb=round(size_bytes / GIB, 2), file_type=ft, file_type_name=name,
                     path="", remote=True)
        synth.setdefault("block_count", 0)
        synth.setdefault("is_moe", False)
        pl = planner.plan(hw, synth, {"ctx_size": int(ctx)}, assumptions)
        adm = pl["admission"]
        same = display_name(name) == current
        out.append({
            "quant": name, "bits_per_weight": rec["bits_per_weight"],
            "file_gb": est(round(size_bytes / GIB, 2), "bits-per-weight table", how),
            "ram_gb": pl["tiers"]["ram"]["planned_gb"], "vram_gb": pl["tiers"]["vram"]["planned_gb"],
            "fits": adm["verdict"] in ("ok", "tight"), "verdict": adm["verdict"],
            "quality_note": rec["qualitative"] + (" " + rec["note"] if rec["note"] else ""),
            "supported_directly": True,  # every format in this table is read natively by llama.cpp on the CPU
            "conversion_required": not same,
            "how_to_get": ("This is the quantization of the current file." if same else
                           f"Download a {name} GGUF of the same model if one is published, or run llama-quantize on an "
                           "F16/BF16 GGUF. Re-quantizing an already quantized file loses more quality."),
            "backend_support": dict(rec["backend_support"]), "status": "estimated",
        })
    return out
