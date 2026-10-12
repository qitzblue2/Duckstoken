"""Comparing a benchmark with the plan it was admitted under, and two benchmarks with each other."""

from __future__ import annotations

from .values import measured, num


def plan_summary(plan: dict) -> dict:
    req, pl, adm = plan.get("requirements", {}), plan.get("placement", {}), plan.get("admission", {})
    spd = (plan.get("speed") or {}).get("tokens_per_s") or {}
    return {"config": plan.get("config"), "verdict": adm.get("verdict"), "placement": pl,
            "weights_gb": req.get("weights_gb"), "kv_cache_gb": req.get("kv_cache_gb"),
            "compute_buffers_gb": req.get("compute_buffers_gb"), "total_gb": req.get("total_gb"),
            "tokens_per_s": spd, "bound_by": (plan.get("speed") or {}).get("bound_by")}


def compare_plan(estimated: dict, parsed: dict, tiers: dict) -> dict:
    """Planner estimate vs the buffer sizes llama.cpp printed at load, per tier (GiB)."""
    out = {"estimated": estimated, "measured": {}, "delta_gb": {}}
    if not parsed or not any(parsed.values()):
        out["note"] = "the server log did not contain buffer sizes (needs log verbosity 4), so nothing to compare"
        return out
    ram, vram = tiers.get("ram_gb", {}), tiers.get("vram_gb", {})
    src = "llama.cpp load log"
    out["measured"] = {"raw_mib": parsed,
                       "weights_ram_gb": measured(ram.get("model", 0.0), src, "model buffers in host memory (mmap'd file counts in full)"),
                       "weights_vram_gb": measured(vram.get("model", 0.0), src),
                       "kv_cache_gb": measured(round(ram.get("kv", 0.0) + vram.get("kv", 0.0), 3), src),
                       "compute_buffers_gb": measured(round(ram.get("compute", 0.0) + vram.get("compute", 0.0), 3), src)}
    pl = estimated.get("placement") or {}
    pairs = {"weights_ram_gb": num(pl.get("weights_ram_gb")), "weights_vram_gb": num(pl.get("weights_vram_gb")),
             "kv_cache_gb": num(estimated.get("kv_cache_gb")), "compute_buffers_gb": num(estimated.get("compute_buffers_gb"))}
    for key, est_v in pairs.items():
        m = num(out["measured"][key])
        if est_v is not None and m is not None:
            out["delta_gb"][key] = round(m - est_v, 3)
    return out


def compare_benchmarks(a: dict, b: dict, tie: float = 0.03) -> dict:
    """Compare two benchmark records (`b` relative to `a`).  Warns when they are not like for like; ratios are
    ``b / a``; a difference below `tie` (3 %) is reported as a tie because run-to-run noise is of that size."""
    warnings: list[str] = []

    def ident(r):
        m = r.get("model") or {}
        return (m.get("name"), m.get("size_bytes"), m.get("file_type"))

    same_model = ident(a) == ident(b)
    same_spec = (a.get("reproducibility") or {}).get("spec_hash") == (b.get("reproducibility") or {}).get("spec_hash")
    same_config = a.get("config_hash") == b.get("config_hash")
    same_hw = (a.get("hardware") or {}).get("fingerprint") == (b.get("hardware") or {}).get("fingerprint")
    if not same_model:
        warnings.append("different models: the numbers are not comparable like for like")
    if not same_spec:
        warnings.append("different benchmark specs (prompt/generation length, seed or repeats)")
    if not same_hw:
        warnings.append("measured on different hardware")
    for name, r in (("first", a), ("second", b)):
        if r.get("status") != "ok":
            warnings.append(f"the {name} benchmark did not finish ok ({r.get('status')}): {r.get('error', '')}")

    def pick(r, *path):
        cur = r
        for p in path:
            cur = cur.get(p) if isinstance(cur, dict) else None
        return cur

    ratios = {}
    for key, path in (("gen_tps", ("warm", "gen_tps")), ("prompt_tps", ("warm", "prompt_tps")),
                      ("ttft_s", ("warm", "ttft_s")), ("load_s", ("load", "seconds")),
                      ("peak_rss_gb", ("metrics", "peak_rss_gb"))):
        va, vb = pick(a, *path), pick(b, *path)
        na, nb = num(va), num(vb)
        ratios[key] = {"a": va, "b": vb, "ratio": round(nb / na, 3) if na and nb is not None else None}
        if na and nb is not None and va.get("status") != vb.get("status"):
            warnings.append(f"{key}: one value is {va.get('status')} and the other {vb.get('status')}")
    g = ratios["gen_tps"]["ratio"]
    winner = None if g is None else ("tie" if abs(g - 1) <= tie else ("b" if g > 1 else "a"))
    return {"comparable": same_model and same_spec and same_hw and not any("did not finish" in w for w in warnings),
            "same_model": same_model, "same_spec": same_spec, "same_config": same_config, "same_hardware": same_hw,
            "warnings": warnings, "ratios": ratios, "winner": winner,
            "note": "winner is by warm generation speed; ratio = second / first"}
