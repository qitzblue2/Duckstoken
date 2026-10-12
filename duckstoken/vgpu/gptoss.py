"""Can this machine run GPT-OSS, honestly? A feasibility report built on the planner, never on wishful thinking.

The specs below come from OpenAI's model configs and the official llama.cpp GGUF repositories (checked October 2026:
``openai/gpt-oss-120b`` and ``-20b`` config.json, and the file listings of ``ggml-org/gpt-oss-120b-GGUF`` and
``ggml-org/gpt-oss-20b-GGUF``). Nothing is downloaded here: the report names the download size so the user can confirm it.
"""

from __future__ import annotations

import copy
from pathlib import Path

from . import gpucompat, planner
from .values import est, unavailable

GIB = 2**30

_COMMON = {
    "architecture": "gpt-oss", "experts_used": 4, "hidden_size": 2880, "expert_ffn": 2880, "heads": 64, "kv_heads": 8,
    "head_dim": 64, "sliding_window": 128, "swa_layers_pattern": "alternating sliding/full (layer 0 sliding)",
    "vocab_size": 201088, "context_length": 131072, "quantization": "MXFP4 (MoE weights only; attention, router and "
    "embeddings stay in higher precision)", "ftype": 38, "ftype_name": "MXFP4_MOE",
}
SPECS: dict[str, dict] = {
    "gpt-oss-120b": {**_COMMON, "name": "gpt-oss-120b", "layers": 36, "experts": 128, "params_total_b": 117,
                     "params_active_b": 5.1, "gguf_repo": "ggml-org/gpt-oss-120b-GGUF", "gguf_file": "gpt-oss-120b-MXFP4.gguf",
                     "gguf_size_bytes": 63_400_000_000,
                     "sources": ["https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json",
                                 "https://huggingface.co/openai/gpt-oss-120b",
                                 "https://huggingface.co/ggml-org/gpt-oss-120b-GGUF"]},
    "gpt-oss-20b": {**_COMMON, "name": "gpt-oss-20b", "layers": 24, "experts": 32, "params_total_b": 21,
                    "params_active_b": 3.6, "gguf_repo": "ggml-org/gpt-oss-20b-GGUF", "gguf_file": "gpt-oss-20b-MXFP4.gguf",
                    "gguf_size_bytes": 12_100_000_000,
                    "sources": ["https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json",
                                "https://huggingface.co/openai/gpt-oss-20b",
                                "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF"]},
}
RAM_TABLE_GB = (16, 24, 32, 48, 64, 96, 128)
SIZE_TOLERANCE = 0.05


def spec_model(name: str) -> dict:
    """A planner model dict built from the official spec (not downloaded yet: ``remote=True``)."""
    s = SPECS[name]
    return {"name": s["name"], "architecture": s["architecture"], "size_bytes": s["gguf_size_bytes"],
            "size_gb": round(s["gguf_size_bytes"] / GIB, 2), "block_count": s["layers"], "expert_count": s["experts"],
            "expert_used_count": s["experts_used"], "is_moe": True, "embedding_length": s["hidden_size"],
            "head_count": s["heads"], "head_count_kv": s["kv_heads"], "key_length": s["head_dim"], "value_length": s["head_dim"],
            "expert_feed_forward_length": s["expert_ffn"], "feed_forward_length": s["expert_ffn"],
            "sliding_window": s["sliding_window"], "context_length": s["context_length"], "vocab_size": s["vocab_size"],
            "file_type": s["ftype"], "file_type_name": s["ftype_name"], "path": "", "remote": True}


def _merge_summary(spec: dict, summary: dict) -> dict:
    """Real file facts win; for a gpt-oss file the spec fills in what the header does not state (gguf.summarize says 0)."""
    if summary.get("architecture") != "gpt-oss":
        return {**summary, "remote": False}  # a different model: plan the file as it is, the architecture check will fail
    m = spec_model(spec["name"])
    m.update({k: v for k, v in summary.items() if v not in (0, None, "", False)})
    m["remote"] = False
    return m


def _what_if(hw: dict, ram_gb: float | None = None, disk_kind: str | None = None) -> dict:
    h = copy.deepcopy(hw)
    h["mock"] = True
    if ram_gb is not None:
        h["ram"]["total_gb"] = float(ram_gb)
        h["ram"]["avail_gb"] = round(ram_gb * 0.85, 1)
    if disk_kind:
        for d in h.get("disks") or []:
            d["kind"] = disk_kind
    return h


def _placement_cfg(hw: dict, model: dict, ctx: int, assumptions: dict | None) -> dict:
    sp = planner.suggest_placement(hw, model, {"ctx_size": ctx}, assumptions)
    return {"ctx_size": ctx, "n_gpu_layers": sp["n_gpu_layers"], "n_cpu_moe": sp["n_cpu_moe"]}


def _speed_class(tps: list | None) -> str:
    if not tps or not tps[1]:
        return "unknown"
    mid = (tps[0] * tps[1]) ** 0.5
    if mid < 0.3:
        return "unusable (well under 1 token/s)"
    if mid < 1.5:
        return "very slow (about 1 token/s)"
    if mid < 4:
        return "slow but usable (a few tokens/s)"
    if mid < 10:
        return "comfortable (5-10 tokens/s)"
    return "fast (10+ tokens/s)"


def _plan(hw, model, ctx, assumptions, path=None):
    cfg = _placement_cfg(hw, model, ctx, assumptions)
    return planner.plan(hw, model, cfg, assumptions, model_path=path)


def _row(hw, model, ctx, assumptions, ram_gb, current) -> dict:
    r = _plan(_what_if(hw, ram_gb), model, ctx, assumptions)
    adm, st = r["admission"], r["tiers"]["storage"]
    outcome = {"ok": "fits", "tight": "fits", "paging": "paging", "reject": "reject"}[adm["verdict"]]
    note = {"fits": "Weights fit in memory.",
            "paging": f"About {st['spill_gb']['value']:.0f} GB would be paged from storage.",
            "reject": adm["blockers"][0] if adm["blockers"] else "Does not fit."}[outcome]
    tps = r["speed"]["tokens_per_s"] if outcome != "reject" else unavailable("not loadable with this much RAM", "planner")
    return {"ram_gb": ram_gb, "current": current, "verdict": adm["verdict"], "outcome": outcome,
            "tokens_per_s": tps, "speed_class": _speed_class(tps["value"]) if outcome != "reject" else "not loadable",
            "spill_gb": st["spill_gb"], "note": note}


def _verdict(plan: dict) -> tuple[str, str]:
    adm = plan["admission"]
    v = adm["verdict"]
    if v == "reject":
        return "not_realistic", "reject"
    if v == "paging":
        return "paging", v
    ram_budget = plan["tiers"]["ram"]["budget_gb"]["value"] or 1
    if v == "ok" and adm["headroom_gb"] >= 0.25 * ram_budget:
        return "comfortable", v
    return ("ok" if v == "ok" else "tight"), v


def _check(id_: str, label: str, status: str, detail: str) -> dict:
    return {"id": id_, "label": label, "status": status, "detail": detail}


def _checks(spec: dict, summary: dict | None, plan: dict, probe: dict | None, ctx: int) -> list[dict]:
    out = []
    official = spec["gguf_size_bytes"]
    if summary is None:
        for id_, label in (("architecture", "File architecture is gpt-oss"), ("experts", "Expert count matches the model"),
                           ("quantization", "File is MXFP4"), ("size", "File size matches the official file")):
            out.append(_check(id_, label, "unknown", "No GGUF file is selected yet."))
    else:
        arch = summary.get("architecture")
        out.append(_check("architecture", "File architecture is gpt-oss", "ok" if arch == "gpt-oss" else "fail",
                          f"The file declares architecture {arch!r}." if arch != "gpt-oss" else "general.architecture = gpt-oss."))
        ec = summary.get("expert_count") or 0
        out.append(_check("experts", "Expert count matches the model", "unknown" if not ec else "ok" if ec == spec["experts"] else "fail",
                          f"The file has {ec} experts per layer; {spec['name']} has {spec['experts']}." if ec else
                          "The file does not state an expert count."))
        ftn = summary.get("file_type_name") or (None if summary.get("file_type") is None else str(summary.get("file_type")))
        if ftn is None:
            out.append(_check("quantization", "File is MXFP4", "unknown", "The file does not state its quantization."))
        elif ftn == "MXFP4_MOE" or str(ftn) == "38":
            out.append(_check("quantization", "File is MXFP4", "ok", "general.file_type = MXFP4_MOE, the format OpenAI released."))
        else:
            out.append(_check("quantization", "File is MXFP4", "warn",
                              f"The file is {ftn}, not the official MXFP4. A re-quantized file can be larger and loses quality "
                              "compared with the original release."))
        size = summary.get("size_bytes") or 0
        if not size:
            out.append(_check("size", "File size matches the official file", "unknown", "The file size is unknown."))
        else:
            ratio = size / official
            if abs(ratio - 1) <= SIZE_TOLERANCE:
                out.append(_check("size", "File size matches the official file", "ok",
                                  f"{size / 1e9:.1f} GB, within {SIZE_TOLERANCE * 100:.0f}% of the official {official / 1e9:.1f} GB."))
            elif ratio < 1:
                out.append(_check("size", "File size matches the official file", "fail",
                                  f"{size / 1e9:.1f} GB is more than {SIZE_TOLERANCE * 100:.0f}% below the official "
                                  f"{official / 1e9:.1f} GB: the download is probably incomplete."))
            else:
                out.append(_check("size", "File size matches the official file", "warn",
                                  f"{size / 1e9:.1f} GB is more than {SIZE_TOLERANCE * 100:.0f}% above the official "
                                  f"{official / 1e9:.1f} GB: probably a different quantization or a merged file."))
    path = (summary or {}).get("path")
    if summary is None:
        out.append(_check("file", "Model file present", "unknown",
                          f"Not downloaded yet. The download is {official / 1e9:.1f} GB and is never started without your confirmation."))
    elif path and not Path(path).exists():
        out.append(_check("file", "Model file present", "fail", f"The file was not found: {path}"))
    else:
        out.append(_check("file", "Model file present", "ok", f"{path or 'File summary supplied'}"))
    supports = (probe or {}).get("supports") or {}
    if probe is None or supports.get("gpt-oss") is None:
        out.append(_check("engine", "Engine knows the gpt-oss architecture", "unknown",
                          "No engine probe result is available." if probe is None else "The engine binaries could not be scanned."))
    else:
        out.append(_check("engine", "Engine knows the gpt-oss architecture", "ok" if supports["gpt-oss"] else "fail",
                          "The engine binaries contain gpt-oss." if supports["gpt-oss"] else
                          "The installed llama.cpp build predates gpt-oss; install a newer one."))
    st = plan["tiers"]["storage"]
    need_gb = spec["gguf_size_bytes"] / GIB
    free = st["free_gb"]
    present = summary is not None
    if present:
        out.append(_check("disk", "Enough free disk", "ok", "The model file is already on disk."))
    elif free is None:
        out.append(_check("disk", "Enough free disk", "unknown", f"Free space on the target drive is unknown; {need_gb:.1f} GB are needed."))
    else:
        margin = planner.DEFAULT_ASSUMPTIONS["disk_margin_gb"]
        status = "ok" if free >= need_gb + margin else "warn" if free >= need_gb else "fail"
        out.append(_check("disk", "Enough free disk", status,
                          f"{free:.0f} GB free on {st['drive']}; the download is {need_gb:.1f} GB (+{margin:g} GB margin)."))
    v = plan["admission"]["verdict"]
    ram_total = plan["tiers"]["ram"]["physical_gb"]["value"]
    out.append(_check("ram", "Enough RAM", {"ok": "ok", "tight": "ok", "paging": "warn", "reject": "fail"}[v],
                      f"{ram_total:g} GB installed; planner verdict '{v}'."))
    kind = st["kind"]
    out.append(_check("storage_speed", "Model drive is fast enough for paging",
                      "ok" if kind in ("nvme", "ssd") else "unknown" if kind == "unknown" else "warn",
                      {"nvme": "NVMe drive.", "ssd": "SATA SSD.", "hdd": "Hard drive: only usable if the whole model fits in RAM.",
                       "unknown": "The drive type is unknown."}[kind]))
    out.append(_check("context", f"Context length within {spec['context_length']} tokens", "ok" if ctx <= spec["context_length"] else "fail",
                      f"Requested {ctx} tokens; the model supports up to {spec['context_length']}."))
    return out


def _gpu_section(hw: dict, plan: dict, spec: dict, gpu_report: dict | None) -> dict:
    gpus = hw.get("gpus") or []
    classes = (gpu_report or {}).get("gpu") or [gpucompat.classify_gpu(g) for g in gpus]
    pl, vram = plan["placement"], plan["tiers"]["vram"]
    total = plan["model"]["size_gb"] or 1
    sec = {"present": bool(gpus), "gpus": classes, "vram_gb": vram["physical_gb"], "layers_on_gpu": pl["gpu_layers"],
           "weights_on_gpu_gb": est(pl["weights_vram_gb"], "planner"), "fraction_of_weights_on_gpu": est(round(pl["weights_vram_gb"] / total, 4), "planner"),
           "can": [], "cannot": [], "notes": [], "benefit": "", "backend_verified": None}
    if gpu_report:
        sec["backend_verified"] = any(i["status"] == "supported" and i["id"] in ("cuda", "vulkan", "hip", "sycl", "metal")
                                      for i in gpu_report.get("items", []))
    if not gpus:
        sec["benefit"] = "No GPU: everything runs on the CPU."
        return sec
    best = max(classes, key=lambda c: c["vram_gb"])
    expert_layer_gb = plan["model"]["size_gb"] * 0.96 / spec["layers"]
    vb = vram["budget_gb"]["value"]
    sec["can"].append(f"Hold the attention/shared weights of {pl['gpu_layers']} of {spec['layers']} layers "
                      f"({pl['weights_vram_gb']:.2f} GB) and compute buffers.")
    sec["cannot"].append(f"Hold the experts: they are {plan['model']['size_gb'] * 0.96:.0f} GB in total (about "
                         f"{expert_layer_gb:.1f} GB per layer) against {vb:.1f} GB of usable VRAM.")
    if pl["moe_experts_on_cpu_layers"] >= spec["layers"]:
        sec["cannot"].append("Run even one layer's experts on the GPU, so the GPU never touches the bulk of the weights.")
    pct = pl["weights_vram_gb"] / total * 100
    sec["benefit"] = (f"Only {pct:.1f}% of the weights fit on the GPU, so token generation (which streams the active experts from "
                      "RAM) barely changes. Prompt processing may gain more. The benefit must be benchmarked, not assumed.")
    if best["vram_gb"] < 4:
        sec["notes"].append(f"{best['name']} has {best['vram_gb']:g} GB of VRAM: below the 4 GB level where partial offload starts to pay off.")
    sec["notes"].extend(best["notes"])
    return sec


def _recommend(model: str, verdict: str, plan: dict, hw: dict, alt: dict | None, if_ssd: dict | None) -> dict:
    kind = plan["tiers"]["storage"]["kind"]
    steps: list[str] = []
    alt_name = "gpt-oss-20b" if model == "gpt-oss-120b" else None
    alt_ok = bool(alt and alt["can_load"])

    def alt_step() -> str:
        how = "fits in memory" if alt["fits"] else f"loads here with some paging (verdict '{alt['verdict']}')"
        return f"Use {alt_name} ({alt['size_gb']:.1f} GB) for everyday work: it {how}, about {_fmt_tps(alt['tokens_per_s'])}."

    if verdict in ("comfortable", "ok"):
        action, title = "run", f"{model} fits this machine."
        steps.append("Load it with the suggested settings and check the real speed in a benchmark.")
    elif verdict == "tight":
        action, title = "free_memory", f"{model} fits, but with little to spare."
        steps += ["Close other applications before loading.", "Keep the context small."]
    elif verdict == "paging":
        action, title = ("use_20b" if alt_ok else "accept_paging"), f"{model} will page from storage and run slowly."
        if kind == "hdd":
            action = "move_to_ssd"
            steps.append("Move the model to an SSD/NVMe drive; paging from a hard drive is unusable.")
        if alt_ok:
            steps.append(alt_step())
            steps.append(f"Keep {model} for the slow background lane where waiting is acceptable.")
        steps.append(f"Installing {_ram_hint(plan)} would remove the paging.")
    else:
        if kind == "hdd" and if_ssd and if_ssd["can_load"]:
            action, title = "move_to_ssd", f"{model} is not usable from a hard drive."
            steps.append(f"Move the model file to an SSD/NVMe drive; there it would load and run at {_speed_class(if_ssd['tps'])}.")
            if alt_ok:
                steps.append(alt_step())
        elif alt_ok:
            action, title = "use_20b", f"{model} is not realistic on this machine."
            steps.append(alt_step())
        else:
            action, title = "add_ram", f"{model} does not fit this machine."
            steps.append("Choose a smaller model or quantization.")
        steps.append(f"Add RAM: {_ram_hint(plan)} would hold the whole model in memory.")
    return {"action": action, "title": title, "steps": steps, "alternative_model": alt_name if alt_ok else None}


def _ram_hint(plan: dict) -> str:
    for s in plan["admission"]["suggestions"]:
        if "GB of RAM" in s:
            return s.split("Installing ")[-1].split(" of RAM")[0] + " of RAM"
    return "more RAM"


def feasibility_report(hw: dict, model: str = "gpt-oss-120b", ctx: int = 4096, file_summary: dict | None = None,
                       probe: dict | None = None, assumptions: dict | None = None, gpu_report: dict | None = None) -> dict:
    """Is GPT-OSS realistic on `hw`? Plain verdict, requirement breakdown, RAM planning table, GPU section, checks.

    `file_summary` is a `gguf.summarize` dict for a downloaded file (omit it to plan from the official spec);
    `probe` an engine probe; `gpu_report` a `gpucompat.compatibility_report`. Everything numeric is a labelled estimate.
    """
    if model not in SPECS:
        raise KeyError(f"unknown model {model!r}; choose one of {', '.join(SPECS)}")
    spec = SPECS[model]
    m = _merge_summary(spec, file_summary) if file_summary else spec_model(model)
    path = (file_summary or {}).get("path") or None
    plan = _plan(hw, m, ctx, assumptions, path)
    verdict, planner_verdict = _verdict(plan)
    rows = [_row(hw, m, ctx, assumptions, g, abs(g - float(hw["ram"].get("total_gb") or 0)) < 0.5) for g in RAM_TABLE_GB]
    alt = None
    if model == "gpt-oss-120b":
        am = spec_model("gpt-oss-20b")
        ap = _plan(hw, am, ctx, assumptions)
        alt = {"model": "gpt-oss-20b", "verdict": ap["admission"]["verdict"], "can_load": ap["admission"]["can_load"],
               "fits": ap["admission"]["verdict"] in ("ok", "tight"),
               "size_gb": ap["model"]["size_gb"], "tokens_per_s": ap["speed"]["tokens_per_s"],
               "speed_class": _speed_class(ap["speed"]["tokens_per_s"]["value"]), "headroom_gb": ap["admission"]["headroom_gb"]}
    if_ssd = None
    kind = plan["tiers"]["storage"]["kind"]
    if kind in ("hdd", "unknown"):
        sp = _plan(_what_if(hw, disk_kind="nvme"), m, ctx, assumptions)
        if_ssd = {"verdict": sp["admission"]["verdict"], "can_load": sp["admission"]["can_load"],
                  "tps": sp["speed"]["tokens_per_s"]["value"], "tokens_per_s": sp["speed"]["tokens_per_s"]}
    tps = plan["speed"]["tokens_per_s"]
    rq = plan["requirements"]
    report = {
        "model": model, "spec": spec, "context": ctx, "source": "file" if file_summary else "official-spec",
        "requirements": rq, "plan": plan, "verdict": verdict, "planner_verdict": planner_verdict,
        "in_memory": plan["tiers"]["storage"]["spill_gb"]["value"] <= 0.005,
        "tokens_per_s": tps, "speed_class": _speed_class(tps["value"]) if plan["admission"]["can_load"] else "not loadable",
        "explanation": _explain(model, spec, plan, verdict, alt, if_ssd, hw), "ram_table": rows,
        "gpu": _gpu_section(hw, plan, spec, gpu_report), "alternative": alt, "if_moved_to_ssd": if_ssd,
        "checks": _checks(spec, file_summary, plan, probe, ctx),
        "never_auto_download": True,
        "download": {"repo": spec["gguf_repo"], "file": spec["gguf_file"], "url": f"https://huggingface.co/{spec['gguf_repo']}",
                     "size_gb": est(round(spec["gguf_size_bytes"] / GIB, 2), "official file listing",
                                    f"{spec['gguf_size_bytes'] / 1e9:.1f} GB decimal; GB here = GiB"),
                     "size_bytes": spec["gguf_size_bytes"], "needs_confirmation": True,
                     "note": "Duckstoken never downloads this model by itself; the user must confirm the size first."},
        "assumptions": plan["assumptions"], "status": "estimated",
    }
    report["recommendation"] = _recommend(model, verdict, plan, hw, alt, if_ssd)
    return report


def _fmt_tps(tps) -> str:
    v = tps["value"]
    return f"{v[0]:g}-{v[1]:g} tokens/s" if v else "an unknown speed"


def _explain(model: str, spec: dict, plan: dict, verdict: str, alt: dict | None, if_ssd: dict | None, hw: dict) -> str:
    rq, st, ram = plan["requirements"], plan["tiers"]["storage"], plan["tiers"]["ram"]
    parts = [f"{model} needs about {rq['weights_gb']['value']:.0f} GB for its weights ({spec['quantization'].split(' ')[0]}, "
             f"{spec['gguf_size_bytes'] / 1e9:.1f} GB file) plus about "
             f"{rq['kv_cache_gb']['value'] + rq['compute_buffers_gb']['value'] + rq['runtime_overhead_gb']['value']:.1f} GB of cache, "
             f"buffers and runtime at {plan['config']['ctx_size']} tokens of context."]
    parts.append(f"This machine has {ram['physical_gb']['value']:g} GB of RAM; after {rq['os_reserve_gb']['value']:g} GB for the system "
                 f"and other applications, {ram['budget_gb']['value']:.0f} GB are available to the model.")
    spill = st["spill_gb"]["value"]
    if verdict == "not_realistic":
        parts.append("The model does not fit in memory and is not realistic here: " + " ".join(plan["admission"]["blockers"][:1]))
    elif spill > 0.005:
        parts.append(f"It does not fit in memory: about {spill:.0f} GB ({st['spill_fraction']['value'] * 100:.0f}% of the weights) "
                     f"would be read from the {({'nvme': 'NVMe drive', 'ssd': 'SSD', 'hdd': 'hard drive'}).get(st['kind'], 'disk')} while generating. "
                     f"It will load, but expect roughly {_fmt_tps(plan['speed']['tokens_per_s'])} (estimate).")
    else:
        parts.append(f"It fits in memory. Expect roughly {_fmt_tps(plan['speed']['tokens_per_s'])} (estimate).")
    if plan["placement"]["gpu_layers"] and plan["tiers"]["vram"]["physical_gb"]["value"] < 4:
        parts.append("The small GPU can only hold some attention weights, so it barely changes the speed.")
    if if_ssd and verdict == "not_realistic" and if_ssd["can_load"]:
        parts.append(f"On an SSD/NVMe it would load and run at about {_fmt_tps(if_ssd['tokens_per_s'])}.")
    if alt and model == "gpt-oss-120b":
        parts.append(f"For comparison gpt-oss-20b ({alt['size_gb']:.1f} GB) is '{alt['verdict']}' here at about "
                     f"{_fmt_tps(alt['tokens_per_s'])}.")
    parts.append("All figures are estimates until the benchmark has measured this machine.")
    return " ".join(parts)
