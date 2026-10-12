"""HTTP endpoints for the Virtual GPU pages: telemetry, memory planner, compatibility and diagnostics, storage tiers
(prewarm, staging cache), benchmarks and the optimizer. Mixed into the request handler in app.py.

Long operations (benchmark, optimization, staging copy) run on the single-slot `JobRunner`, so two of them can never
overlap and fight for memory. Anything that loads a model needs an explicit `confirm`.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import __version__, hardware, paths
from .errors import ApiError
from .vgpu import diagnostics, gptoss, hwmodel, optimizer, pressure, quant, scheduler
from .vgpu.benchmark import BenchSpec, run_benchmark
from .vgpu.runner import Busy
from .vgpu.runtime import assumptions_from_settings, settings_cfg, strategy_text

APP = None

WHAT_IF_KEYS = {"ram_gb": (float, 1, 4096), "ram_speed_mts": (int, 400, 12000), "sticks": (int, 1, 16), "vram_mb": (int, 0, 262144),
                "cores": (int, 1, 256), "logical": (int, 1, 512), "disk_free_gb": (float, 0, 1e6), "disk_size_gb": (float, 1, 1e6)}
WHAT_IF_STR = {"disk_kind": ("nvme", "ssd", "hdd", "unknown"), "gpu_name": None, "compute_cap": None, "cpu_name": None}
CFG_ALLOWED = {"ctx_size": (int, 0, 1 << 21), "n_gpu_layers": (int, -1, 999), "n_cpu_moe": (int, 0, 999), "ubatch_size": (int, 1, 65536)}


def bind(app) -> None:
    global APP
    APP = app


def post(fn):
    fn._method = "POST"
    return fn


# ------------------------------------------------------------------------------------------------ validation helpers
def gguf_path(value, field: str = "model_path") -> Path:
    """A user-supplied model path: must be an existing .gguf file whose header says GGUF. Imported models are untrusted
    input - this only ever reads the first four bytes."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ApiError(f"'{field}' is required")
    p = Path(value).expanduser()
    try:
        p = p.resolve(strict=True)
    except (OSError, RuntimeError) as e:
        raise ApiError("Model file not found") from e
    if not p.is_file() or p.suffix.lower() != ".gguf":
        raise ApiError("That is not a .gguf model file")
    try:
        with open(p, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ApiError("That file is not a GGUF model (bad header)")
    except OSError as e:
        raise ApiError(f"Cannot read the model file: {e.strerror or e}") from e
    return p


def current_model_path(b_or_q: dict, field: str = "model_path") -> Path:
    return gguf_path(b_or_q.get(field) or APP.settings.get("model_path"), field)


def what_if(b: dict) -> tuple[str | None, dict]:
    """(preset, hardware overrides) from a request, validated against a whitelist."""
    preset = b.get("preset") or None
    if preset is not None and preset not in hwmodel.PRESETS:
        raise ApiError("Unknown hardware preset")
    over: dict = {}
    for k, v in (b.get("hardware") or {}).items():
        if k in WHAT_IF_KEYS:
            typ, lo, hi = WHAT_IF_KEYS[k]
            try:
                over[k] = max(lo, min(hi, typ(v)))
            except (TypeError, ValueError) as e:
                raise ApiError(f"Hardware value '{k}' must be a number") from e
        elif k in WHAT_IF_STR:
            opts = WHAT_IF_STR[k]
            if not isinstance(v, str) or (opts and v not in opts) or len(v) > 80:
                raise ApiError(f"Hardware value '{k}' is not valid")
            over[k] = v
        elif k in ("avx2", "avx512"):
            over[k] = bool(v)
        else:
            raise ApiError(f"Unknown hardware field '{k}'")
    if over and preset is None:
        preset = "duck-user"  # custom machine = the owner's PC with the given parts changed
    if over.get("vram_mb") == 0:
        over.setdefault("gpu_name", "")
        over.setdefault("compute_cap", "")
    return preset, over


def cfg_from(b: dict) -> dict:
    out = {}
    for k, v in (b.get("cfg") or {}).items():
        if k in CFG_ALLOWED:
            typ, lo, hi = CFG_ALLOWED[k]
            try:
                out[k] = max(lo, min(hi, typ(v)))
            except (TypeError, ValueError) as e:
                raise ApiError(f"'{k}' must be a number") from e
    if b.get("ctx_size") not in (None, ""):
        try:
            out["ctx_size"] = max(0, min(1 << 21, int(b["ctx_size"])))
        except (TypeError, ValueError) as e:
            raise ApiError("'ctx_size' must be a number") from e
    return out


def _int(q: dict, name: str, default: int, lo: int = 0, hi: int = 10**6) -> int:
    try:
        return max(lo, min(hi, int(q.get(name, default))))
    except (TypeError, ValueError):
        return default


def bench_spec(s: dict) -> BenchSpec:
    return BenchSpec(prompt_tokens=int(s["vgpu_bench_prompt_tokens"]), gen_tokens=int(s["vgpu_bench_gen_tokens"]),
                     repeats=int(s["vgpu_bench_repeats"]))


def _preset_label(name: str, kw: dict) -> str:
    gpu = f"{kw['gpu_name'].replace('NVIDIA GeForce ', '')} {kw['vram_mb'] // 1024 or kw['vram_mb']}{'GB' if kw['vram_mb'] >= 1024 else 'MB'}" if kw.get("vram_mb") else "no GPU"
    return f"{kw.get('cpu_name', '').replace('Intel Core ', '')} · {kw['ram_gb']:g} GB RAM · {gpu} · {kw['disk_kind'].upper()}"


def _trim_record(rec: dict) -> dict:
    """Benchmark list rows without the bulky fields."""
    return {k: v for k, v in rec.items() if k not in ("log_tail", "samples", "hardware", "comparison", "config")}


def _need_idle(b: dict) -> None:
    """A benchmark loads the model itself; two copies of a big model must never coexist."""
    rt = APP.vgpu
    if rt.jobs.active():
        raise ApiError("Another benchmark or copy is already running.", 409)
    if rt.prewarm_state.get("state") == "running":
        raise ApiError("The model is being read into the file cache; wait for it to finish or cancel it.", 409)
    running = APP.server.status().get("running") or APP.big.status().get("running")
    if running:
        if not b.get("stop_server"):
            raise ApiError("Stop the model server first - a second copy of the model would not fit in memory.", 409, needs_stop=True)
        rt.stop_server()
        APP.big.stop()


class VGpuApi:
    # ------------------------------------------------------------------ live telemetry
    def api_vgpu_telemetry(self, q, b):
        return APP.vgpu.telemetry(_int(q, "n", 90, 1, 600))

    def api_vgpu_presets(self, q, b):
        return {"presets": [{"name": n, "label": _preset_label(n, kw)} for n, kw in hwmodel.PRESETS.items()]}

    # ------------------------------------------------------------------ planner
    @post
    def api_vgpu_plan(self, q, b):
        path = current_model_path(b)
        model = APP.model_info(str(path))
        preset, over = what_if(b)
        s = APP.settings.all()
        # default: "what would Duckstoken do with this model" (unset knobs are auto-fitted); use_settings = strict view of your settings
        cfg = settings_cfg(s) if b.get("use_settings") else {}
        cfg.update(cfg_from(b))
        plan, wi, hw = APP.vgpu.plan(model, cfg, preset, over, str(path), force=bool(b.get("force")))
        window = APP.vgpu.sampler.summary(30) if not wi else None
        bottleneck = pressure.diagnose_bottleneck(plan=plan, window=window, hw=hw)
        if not plan["admission"]["can_load"]:
            APP.vgpu.record_failure("estimate_rejected", "; ".join(plan["admission"]["blockers"][:2]), model=model.get("name"), what_if=wi)
        return {"plan": plan, "what_if": wi, "bottleneck": bottleneck,
                "hardware": {"ram_gb": hw["ram"]["total_gb"], "vram_mb": max((g["vram_mb"] for g in hw["gpus"]), default=0), "mock": bool(hw.get("mock"))}}

    def api_vgpu_quants(self, q, b):
        path = current_model_path(q)
        model = APP.model_info(str(path))
        hw = hardware.get_hardware()
        return {"model": model["name"], "current": quant.quant_name(model),
                "rows": quant.compare_quants(model, hw, _int(q, "ctx", int(APP.settings.get("ctx_size") or 4096), 256, 1 << 20),
                                             assumptions_from_settings(APP.settings.all()))}

    def api_vgpu_capabilities(self, q, b):
        s, pr = APP.settings.all(), APP.vgpu.probe(refresh=q.get("refresh") == "1")
        hw = hardware.get_hardware()
        plan = None
        try:
            if q.get("model_path") or s.get("model_path"):
                path = current_model_path(q)
                plan, _, _ = APP.vgpu.plan(APP.model_info(str(path)), settings_cfg(s), model_path=str(path))
        except ApiError:
            plan = None
        return {"capabilities": scheduler.CAPABILITIES, "probe": pr, "gpu_gate": scheduler.gpu_gate(hw, pr),
                "strategy": {"text": strategy_text(s, hw, pr, plan)}}

    # ------------------------------------------------------------------ compatibility / diagnostics
    def api_vgpu_compat(self, q, b):
        hw = hardware.get_hardware(refresh=q.get("refresh") == "1")
        return APP.vgpu.compat(hw)

    def api_vgpu_diagnostics(self, q, b):
        refresh = q.get("refresh") == "1"
        hw = hardware.get_hardware(refresh=refresh)
        pr = APP.vgpu.probe(refresh=refresh)
        s = APP.settings.all()
        model = None
        try:
            if s.get("model_path") and Path(s["model_path"]).is_file():
                model = APP.model_info(s["model_path"])
        except ApiError:
            model = None
        logs = {"model server": APP.server.logs()["lines"][-200:], "big model": APP.big.logs()["lines"][-100:]}
        ctx = {"settings": s, "hardware": hw, "engine_probe": pr, "model": model, "app_version": __version__, "perflog": APP.vgpu.perflog,
               "logs": logs, "telemetry": APP.vgpu.latest(), "compat": APP.vgpu.compat(hw, model, pr)}
        report = diagnostics.build_report(ctx)
        return {"report": report, "text": diagnostics.to_text(report), "compat": report["compat"], "probe": pr,
                "failures": report["failures"]}

    def api_vgpu_logs(self, q, b):
        which, n = q.get("which", "server"), _int(q, "n", 300, 1, 2000)
        if which in ("server", "big"):
            src = APP.server if which == "server" else APP.big
            return {"lines": src.logs()["lines"][-n:]}
        if which == "perf":
            import json
            return {"lines": [json.dumps(e, ensure_ascii=False) for e in APP.vgpu.perflog.tail(n)]}
        if which == "app":
            f = paths.data_dir() / "duckstoken.log"
            try:
                with open(f, "rb") as fh:
                    fh.seek(0, os.SEEK_END)
                    fh.seek(max(0, fh.tell() - 200_000))
                    text = fh.read().decode("utf-8", "replace")
                return {"lines": diagnostics.redact_text(text).splitlines()[-n:]}
            except OSError:
                return {"lines": []}
        raise ApiError("Unknown log")

    @post
    def api_vgpu_gptoss(self, q, b):
        name = b.get("model", "gpt-oss-120b")
        if name not in gptoss.SPECS:
            raise ApiError("Choose gpt-oss-120b or gpt-oss-20b")
        try:
            ctx = max(256, min(131072, int(b.get("ctx", 4096))))
        except (TypeError, ValueError) as e:
            raise ApiError("'ctx' must be a number") from e
        preset, over = what_if(b)
        hw, wi = APP.vgpu.hardware(preset, over)
        file_summary = None
        if b.get("file_path"):
            file_summary = APP.model_info(str(gguf_path(b["file_path"], "file_path")))
        pr = APP.vgpu.probe()
        report = gptoss.feasibility_report(hw, name, ctx, file_summary, pr, assumptions_from_settings(APP.settings.all()),
                                           APP.vgpu.compat(hw, None, pr))
        report["what_if"] = wi
        APP.vgpu.perflog.log("gptoss_report", model=name, verdict=report.get("verdict"), what_if=wi)
        return report

    # ------------------------------------------------------------------ storage tiers
    def api_vgpu_cache(self, q, b):
        rt, st = APP.vgpu, APP.vgpu.staging()
        out = {"prewarm": rt.prewarm_state, "bounded": {"gguf_header": APP.meta_cache.stats()},
               "staging": {"enabled": False}, "stage_plan": None}
        if st is not None:
            out["staging"] = {"enabled": True, "dir": str(st.dir), "budget_gb": st.budget_gb if hasattr(st, "budget_gb") else None,
                              "usage": st.usage(), "items": st.list()}
            out["staging"]["budget_gb"] = out["staging"]["usage"].get("budget_gb")
            try:
                path = current_model_path(q)
                out["stage_plan"] = st.plan_stage(str(path), hardware.get_hardware())
            except ApiError:
                pass
        return out

    @post
    def api_vgpu_cache_stage(self, q, b):
        if b.get("confirm") is not True:
            raise ApiError("Copying a model writes many gigabytes. Confirm to continue.")
        st = APP.vgpu.staging()
        if st is None:
            raise ApiError("The staging cache is off. Set a folder and a size limit in Settings → Virtual GPU & memory manager.")
        path, hw = str(current_model_path(b)), hardware.get_hardware()

        def work(cancel, progress):
            res = st.stage(path, confirm=True, progress=lambda done, total, msg="": progress(done, total or 1, msg or "Copying to the fast drive"),
                           cancel=cancel, hw=hw)
            if not res.get("ok"):
                APP.vgpu.record_failure("stage_failed", res.get("error", "copy failed"), code=res.get("code"))
                raise RuntimeError(res.get("error", "copy failed"))
            return res

        try:
            APP.vgpu.jobs.start("stage", work)
        except Busy as e:
            raise ApiError(str(e), 409) from e
        return APP.vgpu.jobs.status()

    @post
    def api_vgpu_cache_evict(self, q, b):
        if b.get("confirm") is not True:
            raise ApiError("Confirm to remove the staged copy.")
        st = APP.vgpu.staging()
        if st is None:
            raise ApiError("The staging cache is off.")
        name = b.get("name")
        if not isinstance(name, str) or not name:
            raise ApiError("'name' is required")
        res = st.evict(name)
        if not res.get("ok"):
            raise ApiError({"not_found": "No such staged copy.", "in_use": "That copy is in use right now."}.get(res.get("code"), "Could not remove it."), 409)
        return res

    @post
    def api_vgpu_prewarm_start(self, q, b):
        APP.vgpu.prewarm_start(str(current_model_path(b)))
        return {"prewarm": APP.vgpu.prewarm_state}

    @post
    def api_vgpu_prewarm_cancel(self, q, b):
        APP.vgpu.prewarm_cancel()
        return {"prewarm": APP.vgpu.prewarm_state}

    # ------------------------------------------------------------------ optimizer & benchmarks
    @post
    def api_vgpu_opt_candidates(self, q, b):
        objective = b.get("objective", APP.settings.get("vgpu_objective"))
        if objective not in optimizer.OBJECTIVES:
            raise ApiError("Unknown goal")
        path = current_model_path(b)
        model = APP.model_info(str(path))
        hw, s, pr = hardware.get_hardware(), APP.settings.all(), APP.vgpu.probe()
        cands = scheduler.candidate_configs(hw, model, {**s, "model_path": str(path)}, objective, assumptions_from_settings(s), probe=pr)
        gate = scheduler.gpu_gate(hw, pr)
        rows = [{"id": c["id"], "label": c["label"], "overrides": c["overrides"], "admitted": c["admitted"], "reject_reason": c["reject_reason"],
                 "expected": c.get("expected"), "group": c.get("group"), "verdict": c["plan"]["admission"]["verdict"],
                 "tokens_per_s": c["plan"]["speed"]["tokens_per_s"]} for c in cands]
        return {"candidates": rows, "note": None if gate.get("usable") else "No GPU candidates: " + str(gate.get("reason") or "the engine lists no usable GPU device") + " Only CPU variants are tried."}

    @post
    def api_vgpu_opt_start(self, q, b):
        objective = b.get("objective", APP.settings.get("vgpu_objective"))
        if objective not in optimizer.OBJECTIVES:
            raise ApiError("Unknown goal")
        estimate = bool(b.get("estimate_only"))
        if not estimate and b.get("confirm") is not True:
            raise ApiError("The optimizer loads the model several times. Confirm to continue.")
        if b.get("all_models"):
            models = [m["path"] for m in APP.installed_models() if m["kind"] == "model"][:8]
            if not models:
                raise ApiError("No installed models found.")
        else:
            models = [str(current_model_path(b))]
        if not estimate:
            _need_idle(b)
        s, hw, pr, rt = APP.settings.all(), hardware.get_hardware(), APP.vgpu.probe(), APP.vgpu

        def work(cancel, progress):
            res = optimizer.run_optimization(s, models, objective, hw, spec=bench_spec(s), cancel=cancel, progress=progress, store=rt.store,
                                             perflog=rt.perflog, allow_run=not estimate, min_tps=float(s["vgpu_min_tps"]), probe=pr,
                                             assumptions=assumptions_from_settings(s))
            if res.get("status") in ("no_result", "aborted"):
                rt.record_failure("optimizer_" + res["status"], res.get("error") or "; ".join(res.get("notes") or []) or "no result")
            return res

        try:
            rt.jobs.start("optimization", work)
        except Busy as e:
            raise ApiError(str(e), 409) from e
        return rt.jobs.status()

    @post
    def api_vgpu_bench_start(self, q, b):
        if b.get("confirm") is not True:
            raise ApiError("A benchmark loads the model. Confirm to continue.")
        path = str(current_model_path(b))
        _need_idle(b)
        s, hw, rt = APP.settings.all(), hardware.get_hardware(), APP.vgpu
        label = str(b.get("label") or "manual")[:60]

        def work(cancel, progress):
            rec = run_benchmark({**s, "model_path": path}, path, label, hw=hw, spec=bench_spec(s), cancel=cancel, progress=progress,
                                perflog=rt.perflog, assumptions=assumptions_from_settings(s))
            rec = rt.store.add_benchmark(rec)
            if rec.get("status") not in ("ok", "cancelled"):
                rt.record_failure("benchmark_" + rec.get("status", "failed"), rec.get("error") or "; ".join(rec.get("reasons") or []) or "benchmark did not complete")
            return rec

        try:
            rt.jobs.start("benchmark", work)
        except Busy as e:
            raise ApiError(str(e), 409) from e
        return rt.jobs.status()

    def api_vgpu_job(self, q, b):
        return APP.vgpu.jobs.status()

    @post
    def api_vgpu_job_cancel(self, q, b):
        APP.vgpu.jobs.cancel()
        return APP.vgpu.jobs.status()

    def api_vgpu_bench_list(self, q, b):
        return {"benchmarks": [_trim_record(r) for r in APP.vgpu.store.benchmarks(limit=_int(q, "limit", 30, 1, 200), model=q.get("model") or None)]}

    def api_vgpu_bench_get(self, q, b):
        rec = APP.vgpu.store.get_benchmark(q.get("id", ""))
        if rec is None:
            raise ApiError("No such benchmark", 404)
        return {"benchmark": rec}

    @post
    def api_vgpu_bench_delete(self, q, b):
        if b.get("confirm") is not True:
            raise ApiError("Confirm to delete this saved result.")
        return {"deleted": APP.vgpu.store.delete_benchmark(str(b.get("id", "")))}

    def api_vgpu_opt_history(self, q, b):
        return {"history": APP.vgpu.store.history(limit=_int(q, "limit", 15, 1, 100)),
                "profiles": {k: {kk: vv for kk, vv in p.items() if kk in ("objective", "saved", "applied", "estimated_only", "model_name")}
                             for k, p in APP.vgpu.store.profiles().items()}}

    def api_vgpu_profile(self, q, b):
        path = current_model_path(q)
        model, hw, pr = APP.model_info(str(path)), hardware.get_hardware(), APP.vgpu.probe()
        key = APP.vgpu.store.profile_key(hw, model, optimizer.backend_id(pr))
        status = APP.vgpu.store.profile_status(key)
        profile = APP.vgpu.store.get_profile(key)
        if profile is None and status == "stale":
            older = APP.vgpu.store.stale_profiles(key)
            profile = older[-1] if older else None
        return {"key": key, "status": status, "profile": profile}

    @post
    def api_vgpu_opt_apply(self, q, b):
        if b.get("confirm") is not True:
            raise ApiError("Confirm to change your settings.")
        key = str(b.get("key", ""))
        try:
            res = optimizer.apply_best(APP.vgpu.store, key, APP.apply_patch, current=APP.settings.all())
        except optimizer.OptimizerError as e:
            raise ApiError(str(e)) from e
        APP.vgpu.perflog.log("optimizer_apply", key=key, patch=res["patch"])
        return {"applied": res["patch"], "revertible": res["revertible"], "note": res.get("note")}

    @post
    def api_vgpu_opt_revert(self, q, b):
        try:
            res = optimizer.revert_profile(APP.vgpu.store, str(b.get("key", "")), APP.apply_patch)
        except optimizer.OptimizerError as e:
            raise ApiError(str(e)) from e
        if not res.get("reverted"):
            raise ApiError(res.get("error", "Nothing to undo."))
        return res

