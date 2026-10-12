"""Pick the configuration for an objective by MEASURING candidates, remember it, and revert safely.

Rules the code enforces (and the tests check):

* A candidate the planner rejected is never started.  The CPU-only baseline is benchmarked first; if it fails, nothing
  else is run.  Before every run the memory pressure is checked: ``high`` skips the candidate, ``critical`` aborts the
  whole optimization.
* A candidate whose benchmark failed, timed out or was cancelled is recorded under ``reverted`` and can never become
  the result.  The optimizer never modifies the settings it is given and never applies anything by itself: the result
  carries ``applied: False``; `apply_best` (called after the user confirms) applies a stored profile and keeps the
  previous values so `revert_profile` can restore them.
* Objectives (all on MEASURED warm generation speed ``gen_tps``; ``resident_gb`` is the measured peak RSS plus peak VRAM
  when known, else the planner's estimate, and says which):
  ``max_speed``       highest gen_tps; configs within 3 % of the best are a tie and the one using less memory wins.
  ``balanced``        highest ``gen_tps / sqrt(resident_gb)``; configs using more than 90 % of the RAM are excluded.
  ``max_model_size``  among the given model files, the largest one with an admitted config whose gen_tps >= ``min_tps``
                      (default 1.0); for that model the config with the smallest resident memory that still meets
                      ``min_tps`` (configs within 2 % of the smallest memory compete on speed).
* `estimate_only` does the same selection on planner estimates without starting a process; everything is labelled
  ``estimated`` and nothing is stored as a learned profile.
"""

from __future__ import annotations

import copy
import math
import time
import uuid
from typing import Callable

from .. import gguf, hardware
from . import benchmark as bench_mod
from . import scheduler
from .lazy import module as _mod
from .store import backend_fingerprint, hardware_fingerprint, profile_key
from .values import est, measured, num, unavailable

OBJECTIVES = ("max_model_size", "max_speed", "balanced")
MIN_TPS_DEFAULT = 1.0
RAM_USE_LIMIT = 0.90
SPEED_TIE = 0.03
MEMORY_TIE = 0.02


class OptimizerError(ValueError):
    """Bad arguments (unknown objective, no models)."""


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------------------------------------ pressure
def basic_pressure_level() -> str | None:
    """Memory pressure from available RAM alone (used when the telemetry/pressure modules are missing).  Same
    thresholds as `pressure.classify_pressure`: critical < 6 % or < 1.5 GB available, high < 12 %."""
    try:
        ram = hardware.ram_info()
        total, avail = float(ram["total_gb"]), float(ram["avail_gb"])
    except Exception:  # noqa: BLE001
        return None
    if total <= 0:
        return None
    pct = avail / total * 100
    return "critical" if pct < 6 or avail < 1.5 else "high" if pct < 12 else "elevated" if pct < 25 else "ok"


def pressure_level() -> str | None:
    pressure, telemetry = _mod("pressure"), _mod("telemetry")
    if pressure is not None and telemetry is not None:
        try:
            return pressure.classify_pressure(telemetry.system_sample()).get("level")
        except Exception:  # noqa: BLE001
            pass
    return basic_pressure_level()


# ------------------------------------------------------------------------------------------------ selection
def backend_id(backend: dict | None) -> str | None:
    """A stable string for the engine build (version, build number, commit), None when nothing is known."""
    b = backend or {}
    parts = [str(b[k]) for k in ("version", "build", "commit") if b.get(k) is not None]
    return "|".join(parts) if parts else None


def resident_of(rec: dict | None, plan: dict | None) -> dict:
    """Memory the configuration keeps resident, GiB: measured (peak RSS + peak VRAM) when known, else the planner."""
    m = (rec or {}).get("metrics") or {}
    rss, vram = num(m.get("peak_rss_gb")), num(m.get("peak_vram_mb"))
    if rss is not None:
        total = rss + (vram / 1024 if vram is not None else 0.0)
        return measured(round(total, 3), "benchmark sampler", "peak RSS" + (" + peak VRAM (device-wide)" if vram is not None else ""))
    pl = _mod("planner")
    if plan is not None and pl is not None:
        try:
            return est(pl.resident_estimate_gb(plan), "planner", "no measurement available")
        except Exception:  # noqa: BLE001
            pass
    return unavailable("neither a measurement nor a plan is available", "optimizer")


def ram_fraction(rec: dict | None, plan: dict | None, hw: dict) -> float | None:
    total = num((hw.get("ram") or {}).get("total_gb"))
    if not total:
        return None
    rss = num(((rec or {}).get("metrics") or {}).get("peak_rss_gb"))
    if rss is None and plan is not None:
        rss = num(((plan.get("tiers") or {}).get("ram") or {}).get("planned_gb"))
    return None if rss is None else rss / total


def _entry(model: dict, cand: dict, rec: dict | None, hw: dict, basis: str) -> dict | None:
    """One selectable (model, candidate) with the numbers the objectives need; None when it has no speed."""
    if basis == "measured":
        gen = num(((rec or {}).get("warm") or {}).get("gen_tps"))
    else:
        rng = ((cand["plan"].get("speed") or {}).get("tokens_per_s") or {}).get("value")
        gen = (sum(rng) / 2) if isinstance(rng, (list, tuple)) and len(rng) == 2 else None
    if gen is None:
        return None
    res = resident_of(rec if basis == "measured" else None, cand["plan"])
    return {"model": model, "cand": cand, "rec": rec, "gen_tps": gen, "resident": res,
            "resident_gb": max(num(res, 0.0) or 0.0, 0.05), "ram_fraction": ram_fraction(rec if basis == "measured" else None, cand["plan"], hw),
            "basis": basis}


def score_of(objective: str, e: dict) -> float:
    """The number the objective maximizes (also shown to the user)."""
    if objective == "balanced":
        return e["gen_tps"] / math.sqrt(e["resident_gb"])
    return e["gen_tps"]


def pick_best(objective: str, entries: list[dict], min_tps: float = MIN_TPS_DEFAULT) -> tuple[dict | None, str]:
    """Choose among entries (see `_entry`); returns (entry or None, human reason)."""
    if not entries:
        return None, "no configuration produced a usable speed"
    if objective == "max_speed":
        top = max(e["gen_tps"] for e in entries)
        tied = [e for e in entries if e["gen_tps"] >= top * (1 - SPEED_TIE)]
        best = min(tied, key=lambda e: (e["resident_gb"], -e["gen_tps"]))
        return best, (f"highest generation speed ({top:.2f} tokens/s)" +
                      (f"; {len(tied)} configs within {SPEED_TIE:.0%}, the one using least memory was taken" if len(tied) > 1 else ""))
    if objective == "balanced":
        ok = [e for e in entries if e["ram_fraction"] is None or e["ram_fraction"] <= RAM_USE_LIMIT]
        if not ok:
            return None, f"every configuration would use more than {RAM_USE_LIMIT:.0%} of the RAM"
        best = max(ok, key=lambda e: (score_of("balanced", e), e["gen_tps"]))
        return best, "best speed per square root of memory used (gen_tps / sqrt(resident GB)); configs above 90 % RAM use excluded"
    if objective == "max_model_size":
        ok = [e for e in entries if e["gen_tps"] >= min_tps]
        if not ok:
            return None, f"no model reached the minimum of {min_tps:g} tokens/s"
        biggest = max(e["model"]["size_bytes"] for e in ok)
        pool = [e for e in ok if e["model"]["size_bytes"] == biggest]
        low = min(e["resident_gb"] for e in pool)
        near = [e for e in pool if e["resident_gb"] <= low * (1 + MEMORY_TIE)]
        best = max(near, key=lambda e: e["gen_tps"])
        return best, (f"largest model that reaches {min_tps:g} tokens/s; smallest memory footprint among its configurations")
    raise OptimizerError(f"unknown objective {objective!r}")


# ------------------------------------------------------------------------------------------------ result rows
def _tps_mid(plan: dict) -> dict:
    rng = ((plan.get("speed") or {}).get("tokens_per_s") or {}).get("value")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        return est(round(sum(rng) / 2, 2), "planner", f"midpoint of the estimated {rng[0]:g}-{rng[1]:g} tokens/s")
    return unavailable("the planner gave no speed estimate", "planner")


def _row(cand: dict, status: str, reason: str = "", rec: dict | None = None, baseline_gen: float | None = None) -> dict:
    gen = ((rec or {}).get("warm") or {}).get("gen_tps") or unavailable(reason or "not benchmarked", "benchmark")
    prompt = ((rec or {}).get("warm") or {}).get("prompt_tps") or unavailable(reason or "not benchmarked", "benchmark")
    cold = ((rec or {}).get("cold") or {}).get("gen_tps") or unavailable(reason or "not benchmarked", "benchmark")
    mid = _tps_mid(cand["plan"])
    g = num(gen)
    adm = cand["plan"].get("admission", {})
    return {"id": cand["id"], "label": cand["label"], "overrides": cand["overrides"], "group": cand.get("group"),
            "admitted": cand["admitted"], "reject_reason": cand["reject_reason"], "verdict": adm.get("verdict"),
            "status": status, "reason": reason, "benchmark_id": (rec or {}).get("id"), "expected": cand.get("expected"),
            "estimated_tps": mid, "measured_tps": gen, "measured_gen_tps": gen, "measured_prompt_tps": prompt,
            "measured_cold_gen_tps": cold,
            "delta": (est(round(g - num(mid), 2), "measured - planner midpoint", "positive = faster than estimated")
                      if g is not None and num(mid) is not None else unavailable("needs a measurement and an estimate", "optimizer")),
            "speedup_vs_baseline": (round(g / baseline_gen, 3) if g is not None and baseline_gen else None),
            "resident_gb": resident_of(rec, cand["plan"]), "warnings": list((rec or {}).get("warnings") or [])}


def _spread(rec: dict | None) -> float | None:
    """Run-to-run spread of the warm generation speed (coefficient of variation), None with fewer than two runs."""
    g = ((rec or {}).get("warm") or {}).get("gen_tps") or {}
    mean, sd = num(g.get("mean")), num(g.get("stdev"))
    return (sd / mean) if mean and sd is not None and (g.get("n") or 0) >= 2 else None


def vs_baseline(best: dict, baseline: dict | None) -> dict | None:
    """How far the winner is ahead of the CPU-only baseline, and whether that gap is larger than the measurement noise
    (twice the larger run-to-run spread, at least the 3 % tie margin).  None when the winner is the baseline or the
    baseline was not measured."""
    if baseline is None or baseline is best:
        return None
    ratio = best["gen_tps"] / baseline["gen_tps"] if baseline["gen_tps"] else None
    if ratio is None:
        return None
    spreads = [x for x in (_spread(best["rec"]), _spread(baseline["rec"])) if x is not None]
    noise = max([SPEED_TIE] + [2 * x for x in spreads])
    return {"ratio": round(ratio, 3), "noise_pct": round(noise * 100, 1), "distinguishable": abs(ratio - 1) > noise,
            "note": ("the gap is larger than the run-to-run spread" if abs(ratio - 1) > noise else
                     "the gap is within the run-to-run spread: treat these configurations as equivalent")}


def _best_block(objective: str, e: dict, reason: str, patch: dict, key: str | None, basis: str, min_tps: float) -> dict:
    cand = e["cand"]
    return {"model_path": e["model"]["path"], "model_name": e["model"]["name"], "candidate_id": cand["id"],
            "label": cand["label"], "patch": patch, "objective": objective, "basis": basis, "reason": reason,
            "score": est(round(score_of(objective, e), 3), "optimizer", {"max_speed": "gen_tps", "balanced": "gen_tps / sqrt(resident GB)",
                                                                           "max_model_size": "gen_tps"}[objective])
            if basis == "estimated" else measured(round(score_of(objective, e), 3), "benchmark",
                                                  {"max_speed": "warm gen_tps", "balanced": "warm gen_tps / sqrt(resident GB)",
                                                   "max_model_size": "warm gen_tps"}[objective]),
            "gen_tps": (est(round(e["gen_tps"], 3), "planner") if basis == "estimated"
                        else ((e["rec"] or {}).get("warm") or {}).get("gen_tps")),
            "prompt_tps": ((e["rec"] or {}).get("warm") or {}).get("prompt_tps") if basis == "measured"
            else unavailable("estimate only", "planner"),
            "resident_gb": e["resident"], "min_tps": min_tps, "profile_key": key,
            "benchmark_id": (e["rec"] or {}).get("id")}


# ------------------------------------------------------------------------------------------------ main entry points
def _validate(objective: str, model_paths) -> list[str]:
    if objective not in OBJECTIVES:
        raise OptimizerError(f"unknown objective {objective!r}; choose one of {', '.join(OBJECTIVES)}")
    paths = [str(p) for p in (model_paths or [])]
    if not paths:
        raise OptimizerError("no model files were given")
    return list(dict.fromkeys(paths))


def _quant_alternatives(model: dict, hw: dict, ctx: int, assumptions: dict | None) -> list:
    quant = _mod("quant")
    if quant is None:
        return []
    try:
        return list(quant.compare_quants(model, hw, ctx, assumptions))
    except Exception:  # noqa: BLE001
        return []


def estimate_only(settings: dict, model_paths, objective: str, hw: dict, *, min_tps: float = MIN_TPS_DEFAULT,
                  probe: dict | None = None, assumptions: dict | None = None, include_unsafe: bool = False,
                  max_candidates: int = 8, store=None, **_ignored) -> dict:
    """The same selection on planner estimates: no process is started, nothing is measured or remembered."""
    paths = _validate(objective, model_paths)
    out = _new_result(objective, "estimate_only", None, min_tps)
    entries, notes = [], out["notes"]
    notes.append("Estimates only: no benchmark was run. These numbers come from the planner's formulas, not from your hardware.")
    for path in paths:
        mrec, cands = _prepare_model(path, settings, objective, hw, assumptions, include_unsafe, probe, max_candidates)
        out["models"].append(mrec)
        for cand in cands:
            mrec["candidates"].append(_row(cand, "estimated" if cand["admitted"] else "rejected", cand["reject_reason"]))
            if cand["admitted"]:
                e = _entry(mrec["_summary"], cand, None, hw, "estimated")
                if e:
                    entries.append(e)
    best, why = pick_best(objective, entries, min_tps)
    out["comparison"] = _comparison(out)
    if best is not None:
        patch = dict(best["cand"]["overrides"], model_path=best["model"]["path"])
        out["best"] = _best_block(objective, best, why, patch, None, "estimated", min_tps)
        _mark_best(out, best)
    else:
        out["notes"].append(why)
    _finish(out, store, None)
    return out


def _new_result(objective: str, status: str, spec, min_tps: float) -> dict:
    return {"id": f"opt-{time.strftime('%Y%m%d%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}", "created": _now_iso(),
            "objective": objective, "status": status, "min_tps": min_tps,
            "spec": spec.normalized().__dict__ if spec is not None else None, "models": [], "best": None, "reverted": [],
            "skipped": [], "rejected": [], "comparison": [], "benchmarks": [], "applied": False, "notes": [], "error": ""}


def _prepare_model(path, settings, objective, hw, assumptions, include_unsafe, probe, max_candidates):
    mrec = {"path": path, "name": path, "size_bytes": 0, "size_gb": 0.0, "error": "", "candidates": [], "baseline_id": None,
            "best_id": None, "quant_alternatives": [], "_summary": None}
    try:
        summary = gguf.summarize(path)
    except (OSError, ValueError) as e:
        mrec["error"] = f"cannot read the model file: {e}"
        return mrec, []
    mrec.update(name=summary["name"], size_bytes=summary["size_bytes"], size_gb=summary["size_gb"], _summary=summary)
    try:
        cands = scheduler.candidate_configs(hw, summary, settings, objective, assumptions, include_unsafe, probe=probe,
                                            max_candidates=max(max_candidates, 1) + 4)
    except Exception as e:  # noqa: BLE001
        mrec["error"] = f"cannot plan candidates: {e}"
        return mrec, []
    mrec["baseline_id"] = cands[0]["id"] if cands else None
    mrec["quant_alternatives"] = _quant_alternatives(summary, hw, int(settings.get("ctx_size") or 4096), assumptions)
    return mrec, cands


def _mark_best(out: dict, e: dict) -> None:
    for m in out["models"]:
        if m["path"] == e["model"]["path"]:
            m["best_id"] = e["cand"]["id"]


def _comparison(out: dict) -> list[dict]:
    return [{k: row[k] for k in ("id", "estimated_tps", "measured_tps", "measured_gen_tps", "delta", "status")} | {"candidate": row["id"], "model": m["name"]}
            for m in out["models"] for row in m["candidates"]]


def _finish(out: dict, store, perflog) -> None:
    for m in out["models"]:
        m.pop("_summary", None)
    if store is not None:
        try:
            store.add_history({"id": out["id"], "created": out["created"], "objective": out["objective"], "status": out["status"],
                               "models": [m["name"] for m in out["models"]], "benchmarks": out["benchmarks"],
                               "best": ({k: out["best"][k] for k in ("model_name", "candidate_id", "label", "patch", "basis", "reason",
                                                                      "profile_key")} if out["best"] else None),
                               "reverted": len(out["reverted"]), "estimate_only": out["status"] == "estimate_only"})
        except Exception as e:  # noqa: BLE001
            out["notes"].append(f"could not save the history record: {e}")
    if perflog is not None:
        try:
            perflog.log("optimizer_done", optimizer=out["id"], status=out["status"], objective=out["objective"],
                        best=(out["best"] or {}).get("candidate_id"))
        except Exception:  # noqa: BLE001
            pass


def run_optimization(settings: dict, model_paths, objective: str, hw: dict, *, spec=None, cancel=None, progress: Callable | None = None,
                     store=None, perflog=None, allow_run: bool = True, apply_cb: Callable | None = None, max_candidates: int = 8,
                     min_tps: float = MIN_TPS_DEFAULT, probe: dict | None = None, assumptions: dict | None = None,
                     include_unsafe: bool = False, work_dir=None, bench_fn: Callable | None = None,
                     pressure_fn: Callable[[], str | None] | None = None) -> dict:
    """Benchmark the candidates for each model and choose per `objective`.  See the module docstring for the rules.

    Returns ``{"id", "status": ok|no_result|cancelled|aborted|estimate_only, "best": {..., "patch"} | None, "reverted": [...],
    "skipped": [...], "models": [...], "comparison": [...], "applied": False, ...}``.  `settings` is never modified.
    `apply_cb(patch, profile)` is only for callers that already obtained the user's consent; by default nothing is applied.
    """
    paths = _validate(objective, model_paths)
    if not allow_run:
        return estimate_only(settings, paths, objective, hw, min_tps=min_tps, probe=probe, assumptions=assumptions,
                             include_unsafe=include_unsafe, max_candidates=max_candidates, store=store)
    settings = copy.deepcopy(settings)
    spec = spec or bench_mod.BenchSpec()
    run = bench_fn or bench_mod.run_benchmark
    level_fn = pressure_fn or pressure_level
    out = _new_result(objective, "ok", spec, min_tps)
    prepared = [_prepare_model(p, settings, objective, hw, assumptions, include_unsafe, probe, max_candidates) for p in paths]
    total = sum(min(max_candidates, sum(1 for c in cands if c["admitted"])) for _, cands in prepared) or 1
    done = 0
    entries: list[dict] = []
    stop: str | None = None
    for mrec, cands in prepared:
        out["models"].append(mrec)
        if mrec["error"]:
            out["rejected"].append({"model": mrec["name"], "reason": mrec["error"]})
            continue
        baseline_gen, baseline_failed, ran = None, False, 0
        for cand in cands:
            if stop:
                mrec["candidates"].append(_row(cand, "skipped", f"not run: {stop}"))
                continue
            if not cand["admitted"]:
                mrec["candidates"].append(_row(cand, "rejected", cand["reject_reason"]))
                out["rejected"].append({"model": mrec["name"], "candidate": cand["id"], "reason": cand["reject_reason"]})
                continue
            if ran >= max_candidates:
                mrec["candidates"].append(_row(cand, "skipped", f"not run: the limit of {max_candidates} candidates per model was reached"))
                continue
            if baseline_failed:
                mrec["candidates"].append(_row(cand, "skipped", "not run: the baseline configuration failed"))
                out["skipped"].append({"model": mrec["name"], "candidate": cand["id"], "reason": "the baseline configuration failed"})
                continue
            if cancel is not None and cancel.is_set():
                stop = "cancelled by the user"
                out["status"] = "cancelled"
                mrec["candidates"].append(_row(cand, "skipped", f"not run: {stop}"))
                continue
            level = level_fn()
            if level == "critical":
                stop = "memory pressure is critical"
                out["status"] = "aborted"
                out["error"] = "Aborted: the PC is critically short of memory. Close other programs and try again."
                mrec["candidates"].append(_row(cand, "skipped", f"not run: {stop}"))
                continue
            if level == "high":
                out["skipped"].append({"model": mrec["name"], "candidate": cand["id"], "reason": "memory pressure is high"})
                mrec["candidates"].append(_row(cand, "skipped", "not run: memory pressure is high"))
                continue
            done += 1
            ran += 1
            if progress:
                progress(done, total, f"{mrec['name']}: {cand['label']} ({done}/{total})")
            rec = run(_merge(settings, cand["overrides"]), mrec["path"], label=f"{objective}: {cand['id']}", hw=hw, spec=spec,
                      cancel=cancel, perflog=perflog, assumptions=assumptions, work_dir=work_dir)
            out["benchmarks"].append(rec.get("id"))
            if store is not None:
                try:
                    store.add_benchmark(rec)
                except Exception as e:  # noqa: BLE001
                    out["notes"].append(f"could not save benchmark {rec.get('id')}: {e}")
            if rec.get("status") != "ok":
                reason = rec.get("error") or rec.get("status")
                out["reverted"].append({"model": mrec["name"], "candidate": cand["id"], "status": rec.get("status"), "reason": reason,
                                        "overrides": cand["overrides"], "benchmark_id": rec.get("id")})
                mrec["candidates"].append(_row(cand, rec.get("status", "failed"), reason, rec))
                if rec.get("status") == "cancelled":
                    stop = "cancelled by the user"
                    out["status"] = "cancelled"
                elif cand["id"] == mrec["baseline_id"]:
                    baseline_failed = True
                continue
            if cand["id"] == mrec["baseline_id"]:
                baseline_gen = num(rec["warm"]["gen_tps"])
            mrec["candidates"].append(_row(cand, "ok", "", rec, baseline_gen))
            e = _entry(mrec["_summary"], cand, rec, hw, "measured")
            if e:
                entries.append(e)
                e["rec_backend"] = rec.get("backend")
    out["comparison"] = _comparison(out)
    if out["status"] in ("ok",):
        best, why = pick_best(objective, entries, min_tps)
        if best is None:
            out["status"] = "no_result"
            out["notes"].append(why)
        else:
            _conclude(out, best, why, objective, hw, spec, store, min_tps, apply_cb, entries)
    elif entries:
        out["notes"].append("The run did not finish, so no configuration was chosen. Measured results are kept in the benchmark list.")
    _finish(out, store, perflog)
    return out


def _merge(settings: dict, overrides: dict) -> dict:
    merged = dict(settings)
    merged.update(overrides)
    return merged


def _conclude(out: dict, best: dict, why: str, objective: str, hw: dict, spec, store, min_tps: float, apply_cb,
              entries: list[dict]) -> None:
    rec, model = best["rec"], best["model"]
    key = profile_key(hw, model, backend_id(rec.get("backend")))
    patch = dict(best["cand"]["overrides"], model_path=model["path"])
    out["best"] = _best_block(objective, best, why, patch, key, "measured", min_tps)
    base_id = next((m["baseline_id"] for m in out["models"] if m["path"] == model["path"]), None)
    baseline = next((e for e in entries if e["model"]["path"] == model["path"] and e["cand"]["id"] == base_id), None)
    out["best"]["vs_baseline"] = vs_baseline(best, baseline)
    if out["best"]["vs_baseline"] and not out["best"]["vs_baseline"]["distinguishable"]:
        out["notes"].append(f"The chosen configuration is only {out['best']['vs_baseline']['ratio']:.2f}x the baseline, "
                            f"which is within the measurement noise (about {out['best']['vs_baseline']['noise_pct']:.0f} %).")
    _mark_best(out, best)
    profile = {"model": {k: model.get(k) for k in ("path", "name", "size_bytes", "file_type")}, "objective": objective, "patch": patch,
               "candidate_id": best["cand"]["id"], "label": best["cand"]["label"], "reason": why, "min_tps": min_tps,
               "measured": {"gen_tps": rec["warm"]["gen_tps"], "prompt_tps": rec["warm"]["prompt_tps"],
                            "cold_gen_tps": rec["cold"]["gen_tps"], "resident_gb": best["resident"],
                            "warm_runs": len(rec["warm"]["runs"])},
               "score": out["best"]["score"], "benchmark_id": rec.get("id"), "config_hash": rec.get("config_hash"),
               "spec_hash": (rec.get("reproducibility") or {}).get("spec_hash"), "hardware_fingerprint": hardware_fingerprint(hw),
               "backend_fingerprint": backend_fingerprint(backend_id(rec.get("backend"))), "backend": rec.get("backend"),
               "vs_baseline": out["best"].get("vs_baseline"), "estimated_only": False, "applied": False, "previous": None}
    if store is not None:
        try:
            saved = store.save_profile(key, profile)
            out["best"]["profile_saved"] = True
            profile = saved
        except Exception as e:  # noqa: BLE001
            out["best"]["profile_saved"] = False
            out["notes"].append(f"could not save the learned profile: {e}")
    if apply_cb is not None:
        try:
            out["applied"] = bool(apply_cb(patch, profile))
        except Exception as e:  # noqa: BLE001
            out["applied"] = False
            out["notes"].append(f"applying the result failed and nothing was changed: {e}")


# ------------------------------------------------------------------------------------------------ apply / revert
def apply_best(store, key: str, settings_updater: Callable[[dict], object], current: dict | None = None) -> dict:
    """Apply the stored profile `key` through `settings_updater(patch)` (call only after the user confirmed).

    `current` is the settings dict as it is now; the values the patch replaces are stored as ``previous`` so
    `revert_profile` can restore them.  Without `current` the profile is still applied but cannot be reverted.
    Profiles from an estimate are refused.  If the updater raises nothing is recorded as applied."""
    profile = store.get_profile(key)
    if profile is None:
        raise OptimizerError(f"no learned profile {key!r}")
    if profile.get("estimated_only") or not profile.get("patch"):
        raise OptimizerError("this profile is only an estimate and was never measured; it cannot be applied")
    patch = dict(profile["patch"])
    previous = profile.get("previous") if profile.get("applied") and profile.get("previous") else None
    if previous is None and current is not None:
        previous = {k: copy.deepcopy(current[k]) for k in patch if k in current}
    settings_updater(patch)
    store.save_profile(key, {**profile, "applied": True, "applied_at": _now_iso(), "previous": previous})
    return {"applied": True, "patch": patch, "previous": previous, "revertible": previous is not None,
            "note": None if previous is not None else "the previous values were not given, so this cannot be reverted automatically"}


def revert_profile(store, key: str, settings_updater: Callable[[dict], object]) -> dict:
    """Restore the values that were replaced by `apply_best`."""
    profile = store.get_profile(key)
    if profile is None:
        raise OptimizerError(f"no learned profile {key!r}")
    previous = profile.get("previous")
    if not profile.get("applied") or not previous:
        return {"reverted": False, "error": "nothing to revert: this profile was not applied, or its previous values are unknown"}
    settings_updater(dict(previous))
    store.save_profile(key, {**profile, "applied": False, "previous": None, "reverted_at": _now_iso()})
    return {"reverted": True, "restored": dict(previous)}
