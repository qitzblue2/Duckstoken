"""Glue between the app and the VirtualGPU modules: owns the long-lived pieces (sampler, pressure monitor, store, job
runner, caches, crash watchdog), turns the user's settings into planner assumptions, gates model loads, and runs the
storage-level prefetch (prewarm) and the optional staging copy.

Every optional piece degrades to "unavailable" instead of failing the app.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .. import hardware, paths, server_proc
from ..errors import ApiError
from . import diagnostics, gpucompat, planner, pressure, probe, telemetry
from .diskcache import BoundedCache, StagingCache
from .perflog import PerfLog
from .prewarm import prewarm as run_prewarm
from .recovery import CrashWatchdog
from .runner import JobRunner
from .store import VGpuStore
from .values import num

GIB = 2**30


def assumptions_from_settings(s: dict) -> dict:
    """User-tunable planner assumptions (Settings → Virtual GPU & memory manager)."""
    a = dict(planner.DEFAULT_ASSUMPTIONS)
    a.update(os_reserve_gb=float(s.get("vgpu_os_reserve_gb", a["os_reserve_gb"])),
             safety_margin_pct=float(s.get("vgpu_safety_margin_pct", a["safety_margin_pct"])),
             vram_reserve_mb=int(s.get("vgpu_vram_reserve_mb", a["vram_reserve_mb"])),
             max_spill_fraction=int(s.get("vgpu_max_spill_pct", a["max_spill_fraction"] * 100)) / 100.0)
    return a


def settings_cfg(s: dict) -> dict:
    """The part of the settings the planner reads (all of it: this is the strict, 'what you really set' view)."""
    return {k: s[k] for k in planner.CFG_KEYS if k in s}


class VGpuRuntime:
    def __init__(self, app):
        self.app = app
        self.perflog = PerfLog(paths.perf_log_file())
        self.store = VGpuStore(paths.vgpu_store_dir())
        self.jobs = JobRunner()
        self.sampler = telemetry.Sampler(interval_s=2.0, maxlen=600, pid_fn=self._server_pid)
        self.monitor = pressure.Monitor(self.sampler, app.settings.all, on_event=self._on_pressure)
        self.watchdog = CrashWatchdog(app.server, app.settings.all, perflog=self.perflog, on_event=self._on_crash)
        self.pressure_events: list[dict] = []
        self.last_autofit: dict | None = None  # set when the last start lowered the GPU split to fit VRAM
        self.last_start_plan: dict | None = None
        self.prewarm_state: dict = {"state": "idle", "message": ""}
        self._prewarm_cancel = threading.Event()
        self._prewarm_thread: threading.Thread | None = None
        self._staging: StagingCache | None = None
        self._staging_key: tuple | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        s = self.app.settings.all()
        if s.get("vgpu_monitor"):
            self.sampler.interval_s = max(1.0, float(s.get("vgpu_monitor_interval", 2)))
            self.sampler.start()
            self.monitor.start()
        self.watchdog.start()
        staging = self.staging()
        if staging is not None:
            try:
                staging.cleanup()  # partial copies left by a crash
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        for part in (self.watchdog, self.monitor, self.sampler):
            try:
                part.stop()
            except Exception:  # noqa: BLE001
                pass
        self._prewarm_cancel.set()
        self.jobs.cancel()

    def settings_changed(self, patch: dict) -> None:
        """Apply a settings patch to the running monitor (toggle, interval)."""
        if "vgpu_monitor" in patch or "vgpu_monitor_interval" in patch:
            s = self.app.settings.all()
            self.sampler.interval_s = max(1.0, float(s.get("vgpu_monitor_interval", 2)))
            if s.get("vgpu_monitor"):
                self.sampler.start()
                self.monitor.start()
            else:
                self.monitor.stop()
                self.sampler.stop()

    def _server_pid(self):
        try:
            proc = getattr(self.app.server, "proc", None)
            return proc.pid if proc is not None and proc.poll() is None else None
        except Exception:  # noqa: BLE001
            return None

    # --------------------------------------------------------------- events
    def record_failure(self, kind: str, message: str, **context) -> None:
        diagnostics.record_failure(kind, message, context or None)
        self.perflog.log("failure", kind=kind, message=message)

    def _on_pressure(self, event: dict) -> None:
        with self._lock:
            self.pressure_events.append({k: v for k, v in event.items() if k != "sample"})
            del self.pressure_events[:-30]
        self.perflog.log("pressure", level=event.get("level"), previous=event.get("previous"), action=event.get("action"))
        action = event.get("action")
        if event.get("level") == "critical" and action in ("stop_big_model", "stop_all"):
            self.app.big.stop()  # frees the big model first; the queue can restart it later
            if action == "stop_all":
                self.stop_server()
            self.record_failure("memory_pressure", f"Critical memory pressure: {action.replace('_', ' ')} performed.")

    def _on_crash(self, event: dict) -> None:
        self.record_failure("server_crash", f"{event['crash']['why']} ({event['action'].replace('_', ' ')})", crash=event["crash"])

    # ---------------------------------------------------------- engine facts
    def engine_exe(self) -> Path | None:
        return server_proc.find_llama_server(self.app.settings.all())

    def probe(self, refresh: bool = False) -> dict:
        exe = self.engine_exe()
        if exe is None:
            return {"path": None, "exists": False, "errors": ["No llama.cpp engine is installed or bundled yet."], "status": "missing",
                    "backends": [], "devices": [], "supports": {}}
        return probe.probe_engine(exe) if refresh else probe.probe_engine_cached(exe)

    def backend_label(self, pr: dict | None = None) -> str | None:
        from .optimizer import backend_id
        return backend_id(pr if pr is not None else self.probe())

    # ------------------------------------------------------------- planning
    def hardware(self, preset: str | None = None, overrides: dict | None = None) -> tuple[dict, str | None]:
        """Detected hardware, or a hypothetical machine for what-if planning."""
        if preset or overrides:
            from .hwmodel import preset_hardware
            return preset_hardware(preset or "duck-user", **(overrides or {})), preset or "custom"
        return hardware.get_hardware(), None

    def plan(self, model: dict, cfg: dict | None = None, preset: str | None = None, overrides: dict | None = None,
             model_path: str | None = None, force: bool = False) -> tuple[dict, str | None, dict]:
        hw, what_if = self.hardware(preset, overrides)
        a = assumptions_from_settings(self.app.settings.all())
        return planner.plan(hw, model, cfg, a, model_path=model_path, force=force), what_if, hw

    def compat(self, hw: dict, model: dict | None = None, probe_result: dict | None = None) -> dict:
        return gpucompat.compatibility_report(hw, probe_result if probe_result is not None else self.probe(), model)

    def admit(self, s: dict, force: bool = False) -> dict | None:
        """The load gate. Returns the plan (or None when checking is off / not applicable); raises ApiError(409) with the
        admission details when the planner says the model cannot load and the user has not forced it.

        When the *only* problem is GPU offload larger than the card (typical: a preset saying "offload everything" on a 2 GB
        GPU) and `vgpu_autofit` is on, the plan carries ``plan["autofit"]`` = the lower n_gpu_layers / n_cpu_moe that do fit;
        `start_server` uses them for that start only."""
        mode = s.get("vgpu_admission", "enforce")
        path = s.get("model_path", "")
        if mode == "off" or s.get("engine") != "llamacpp" or not path or not Path(path).is_file():
            return None
        try:
            model = self.app.model_info(path)
            cfg = settings_cfg(s)
            plan, what_if, hw = self.plan(model, cfg, model_path=path, force=force)
            if not plan["admission"]["can_load"] and s.get("vgpu_autofit", True) and not force:
                fixed = self._autofit(hw, model, cfg, plan, path)
                if fixed is not None:
                    plan = fixed
        except ApiError:
            raise
        except Exception as e:  # noqa: BLE001 - the planner must never be the reason a model cannot start
            self.perflog.log("admission_error", error=f"{e.__class__.__name__}: {e}")
            return None
        adm = plan["admission"]
        self.perflog.log("admission", model=model.get("name"), verdict=adm["verdict"], can_load=adm["can_load"], mode=mode,
                         blockers=adm["blockers"][:3], autofit=(plan.get("autofit") or {}).get("overrides"))
        if not adm["can_load"] and mode == "enforce" and not force:
            msg = "Duckstoken does not expect this model to load on this PC with the current settings: " + "; ".join(adm["blockers"][:2])
            self.record_failure("admission_rejected", msg, model=model.get("name"), verdict=adm["verdict"])
            raise ApiError(msg, 409, admission=adm)
        return plan

    def _autofit(self, hw: dict, model: dict, cfg: dict, plan: dict, path: str) -> dict | None:
        """A plan that loads after lowering only the GPU split, or None (then the original blockers stand)."""
        if int(cfg.get("n_gpu_layers") or 0) == 0:
            return None
        place = planner.suggest_placement(hw, model, cfg, assumptions_from_settings(self.app.settings.all()))
        over = {"n_gpu_layers": place["n_gpu_layers"], "n_cpu_moe": place["n_cpu_moe"]}
        if over["n_gpu_layers"] > int(cfg.get("n_gpu_layers") or 0) and over["n_cpu_moe"] >= int(cfg.get("n_cpu_moe") or 0):
            return None  # this would offload more than the user asked for: never do that automatically
        fixed, _, _ = self.plan(model, {**cfg, **over}, model_path=path)
        if not fixed["admission"]["can_load"]:
            return None
        fixed["autofit"] = {"overrides": over, "reason": place["reason"],
                            "was": {"n_gpu_layers": cfg.get("n_gpu_layers"), "n_cpu_moe": cfg.get("n_cpu_moe")},
                            "message": f"The GPU cannot hold that much: using {over['n_gpu_layers']} GPU layers and {over['n_cpu_moe']} "
                                       f"expert layers in RAM for this start ({place['reason']}). Your saved settings are unchanged."}
        return fixed

    # ------------------------------------------------------- server control
    def start_server(self, s: dict, force: bool = False) -> None:
        """Gate, optionally substitute a staged copy of the model, optionally prewarm the page cache, then start."""
        plan = self.admit(s, force)
        s = dict(s)
        self.last_autofit = None
        if plan and plan.get("autofit"):
            s.update(plan["autofit"]["overrides"])
            self.last_autofit = plan["autofit"]
        self.last_start_plan = plan
        if self._prewarm_thread and self._prewarm_thread.is_alive():
            raise ApiError("The model is already being prepared to start. Press Stop to cancel it first.", 409)
        try:
            staging = self.staging()
            staged = staging.lookup(s["model_path"]) if staging is not None and s.get("model_path") else None
            if staged:
                s["model_path"] = staged
        except Exception as e:  # noqa: BLE001 - the staging cache is an optional accelerator, never a reason not to start
            self.perflog.log("staging_lookup_failed", error=f"{e.__class__.__name__}: {e}")
        self.watchdog.note_start(s)
        if s.get("vgpu_prewarm") == "before_start" and s.get("engine") == "llamacpp" and s.get("model_path"):
            self._prewarm_cancel = threading.Event()
            cancel = self._prewarm_cancel
            self.prewarm_state = {"state": "running", "message": "Reading the model into the file cache…", "pct": 0}

            def work():
                self._prewarm_run(s["model_path"], s, cancel)
                if cancel.is_set():
                    return  # the user pressed Stop meanwhile
                try:
                    self.app.server.start(s)
                except Exception as e:  # noqa: BLE001
                    self.prewarm_state["error"] = str(e)
                    self.record_failure("start_failed", str(e))

            self._prewarm_thread = threading.Thread(target=work, daemon=True, name="prewarm-then-start")
            self._prewarm_thread.start()
            return
        self.app.server.start(s)

    def stop_server(self) -> None:
        self._prewarm_cancel.set()
        self.watchdog.note_stop()
        self.app.server.stop()

    # -------------------------------------------------------------- prewarm
    def _prewarm_run(self, path: str, s: dict, cancel: threading.Event) -> dict:
        limit_gb = float(s.get("vgpu_prewarm_max_gb", 0) or 0)

        def progress(done, limit, msg=""):
            self.prewarm_state.update(state="running", pct=int(100 * done / limit) if limit else 0,
                                      message=f"{msg or 'Reading the model into the file cache'} ({done / GIB:.1f} of {limit / GIB:.1f} GB)")

        try:
            res = run_prewarm(path, int(limit_gb * GIB) if limit_gb > 0 else None, cancel=cancel, progress=progress)
        except Exception as e:  # noqa: BLE001
            res = {"completed": False, "stopped_because": "error", "error": str(e), "bytes_read": 0}
        speed = num(res.get("mb_per_s"))
        why = res.get("stopped_because")
        self.prewarm_state = {"state": "done" if res.get("completed") else "stopped", "result": res,
                              "message": (f"Read {res.get('bytes_read', 0) / GIB:.1f} GB" + (f" at {speed:.0f} MB/s" if speed else "")
                                          + ("" if res.get("completed") else f" — stopped: {why}"))}
        self.perflog.log("prewarm", bytes_read=res.get("bytes_read"), completed=res.get("completed"), stopped_because=why, mb_per_s=speed)
        return res

    def prewarm_start(self, path: str) -> None:
        if self.jobs.active():
            raise ApiError("A benchmark or copy is running; wait for it to finish.", 409)
        if self._prewarm_thread and self._prewarm_thread.is_alive():
            raise ApiError("Already prewarming.", 409)
        s = self.app.settings.all()
        self._prewarm_cancel = threading.Event()
        cancel = self._prewarm_cancel
        self.prewarm_state = {"state": "running", "message": "Starting…", "pct": 0}
        self._prewarm_thread = threading.Thread(target=self._prewarm_run, args=(path, s, cancel), daemon=True, name="prewarm")
        self._prewarm_thread.start()

    def prewarm_cancel(self) -> None:
        self._prewarm_cancel.set()

    # -------------------------------------------------------------- staging
    def staging(self) -> StagingCache | None:
        s = self.app.settings.all()
        d, gb = s.get("vgpu_cache_dir", ""), float(s.get("vgpu_cache_gb", 0) or 0)
        if not d or gb <= 0:
            return None
        key = (d, gb, float(s.get("vgpu_cache_min_free_gb", 10)))
        with self._lock:
            if self._staging is None or self._staging_key != key:
                self._staging = StagingCache(Path(d), gb, min_free_gb=key[2])
                self._staging_key = key
            return self._staging

    # ------------------------------------------------------------ telemetry
    def latest(self) -> dict | None:
        """The newest sample; takes one on demand when monitoring is off or the buffer is stale."""
        s = self.sampler.latest()
        if s is None or time.time() - s.get("t", 0) > 6:
            s = self.sampler.sample_once()
        return s

    def telemetry(self, n: int = 90) -> dict:
        latest = self.latest() or {}
        hist = []
        for x in self.sampler.history(n):
            gpus = x.get("gpus") or []
            g0 = gpus[0] if gpus else {}
            hist.append({"t": x.get("t"), "ram_used_pct": num((x.get("ram") or {}).get("used_pct")),
                         "cpu_pct": num((x.get("cpu") or {}).get("util_pct")), "gpu_pct": num(g0.get("util_pct")),
                         "vram_used_mb": num(g0.get("vram_used_mb")), "rss_gb": num((x.get("process") or {}).get("rss_gb"))})
        cfg = self.app.settings.all()
        return {"enabled": bool(cfg.get("vgpu_monitor")), "interval": cfg.get("vgpu_monitor_interval"),
                "latest": latest, "history": hist, "pressure": pressure.classify_pressure(latest) if latest else {"level": "ok", "reasons": []},
                "monitor": self.monitor.state(), "events": list(self.pressure_events)[-10:],
                "watchdog": self.watchdog.status(), "last_request": self.app.last_request or None,
                "prewarm": self.prewarm_state}


def strategy_text(s: dict, hw: dict, pr: dict, plan: dict | None = None) -> str:
    """One honest paragraph on what Duckstoken is currently doing with the memory tiers."""
    gpus = hw.get("gpus") or []
    vram = max((g.get("vram_mb", 0) for g in gpus), default=0) / 1024
    devices = [d for d in (pr.get("devices") or []) if d.get("backend") not in (None, "CPU")]
    parts = []
    if not devices:
        parts.append("Inference runs on the CPU: the engine lists no usable GPU device" + (f" (a {vram:.0f} GB GPU was detected but is not used)." if vram else "."))
    else:
        ngl = int(s.get("n_gpu_layers", 0))
        parts.append(f"GPU offload is {'on' if ngl else 'off'} ({ngl} layers) on {devices[0].get('name', 'the GPU')}.")
    if plan:
        pl = plan.get("placement") or {}
        if (plan.get("model") or {}).get("is_moe"):
            parts.append(f"MoE experts of {pl.get('moe_experts_on_cpu_layers', 0)} layers stay in system RAM.")
        if (pl.get("weights_spill_gb") or 0) > 0.05:
            parts.append(f"About {pl['weights_spill_gb']:.1f} GB of weights page from storage through memory mapping.")
    parts.append("Weights are memory-mapped." if s.get("mmap", True) else "Weights are loaded fully into RAM (mmap off).")
    if s.get("mlock"):
        parts.append("Pages are locked in RAM (mlock).")
    parts.append("Storage prefetch: the model is read sequentially into the file cache before start." if s.get("vgpu_prewarm") == "before_start"
                 else "Storage prefetch is off.")
    if float(s.get("vgpu_cache_gb", 0) or 0) > 0 and s.get("vgpu_cache_dir"):
        parts.append("A bounded staging cache on a fast drive is enabled.")
    return " ".join(parts)
