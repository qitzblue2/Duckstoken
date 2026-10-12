"""Glue between the app and the VirtualGPU modules: owns the long-lived pieces (sampler, monitor, store, job runner,
caches, watchdog) and turns the user's settings into planner assumptions.

Everything here is constructed lazily and every optional piece degrades to "unavailable" instead of failing the app.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .. import hardware, paths
from . import planner
from .diskcache import BoundedCache, StagingCache
from .recovery import CrashWatchdog
from .runner import JobRunner
from .store import VGpuStore


def assumptions_from_settings(s: dict) -> dict:
    """User-tunable planner assumptions (Settings → Virtual GPU & memory manager)."""
    a = dict(planner.DEFAULT_ASSUMPTIONS)
    a.update(os_reserve_gb=float(s.get("vgpu_os_reserve_gb", a["os_reserve_gb"])),
             safety_margin_pct=float(s.get("vgpu_safety_margin_pct", a["safety_margin_pct"])),
             vram_reserve_mb=int(s.get("vgpu_vram_reserve_mb", a["vram_reserve_mb"])),
             max_spill_fraction=int(s.get("vgpu_max_spill_pct", a["max_spill_fraction"] * 100)) / 100.0)
    return a


class VGpuRuntime:
    def __init__(self, app):
        from . import pressure, telemetry  # imported here so this module loads before they exist (work in progress)
        from .perflog import PerfLog
        self.app = app
        self.perflog = PerfLog(paths.perf_log_file())
        self.store = VGpuStore(paths.vgpu_store_dir())
        self.jobs = JobRunner()
        self.sampler = telemetry.Sampler(interval_s=2.0, maxlen=600, pid_fn=self._server_pid)
        self.monitor = pressure.Monitor(self.sampler, app.settings.all, on_event=self._on_pressure)
        self.watchdog = CrashWatchdog(app.server, app.settings.all, perflog=self.perflog, on_event=self._on_crash)
        self.meta_cache = BoundedCache(max_bytes=8 << 20, max_items=64)  # parsed GGUF headers
        self.failures: list[dict] = []
        self.pressure_events: list[dict] = []
        self.prewarm_state: dict = {"state": "idle", "message": ""}
        self._prewarm_cancel = threading.Event()
        self._staging: StagingCache | None = None
        self._staging_key: tuple | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        s = self.app.settings.all()
        if s.get("vgpu_monitor"):
            self.sampler.interval_s = max(1.0, float(s.get("vgpu_monitor_interval", 2)))
            self.sampler.start()
            self.monitor.start() if hasattr(self.monitor, "start") else None
        self.watchdog.start()

    def stop(self) -> None:
        for part in (self.watchdog, self.monitor, self.sampler):
            try:
                part.stop()
            except Exception:  # noqa: BLE001
                pass
        self._prewarm_cancel.set()
        self.jobs.cancel()

    def _server_pid(self):
        try:
            proc = getattr(self.app.server, "proc", None)
            return proc.pid if proc is not None and proc.poll() is None else None
        except Exception:  # noqa: BLE001
            return None

    # --------------------------------------------------------------- events
    def record_failure(self, kind: str, message: str, **context) -> None:
        with self._lock:
            self.failures.append({"t": time.time(), "kind": kind, "message": message, "context": context})
            del self.failures[:-50]
        self.perflog.log("failure", kind=kind, message=message)

    def _on_pressure(self, event: dict) -> None:
        with self._lock:
            self.pressure_events.append({k: v for k, v in event.items() if k != "sample"})
            del self.pressure_events[:-30]
        self.perflog.log("pressure", level=event.get("level"), previous=event.get("previous"), action=event.get("action"))
        action = event.get("action")
        if event.get("level") == "critical" and action in ("stop_big_model", "stop_all"):
            self.app.big.stop()  # frees the big model first; it can always be restarted from the queue
            if action == "stop_all":
                self.app.server.stop()
            self.record_failure("memory_pressure", f"Critical memory pressure: {action.replace('_', ' ')} performed.")

    def _on_crash(self, event: dict) -> None:
        self.record_failure("server_crash", f"{event['crash']['why']} ({event['action'].replace('_', ' ')})", crash=event["crash"])

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
        s = self.app.settings.all()
        base = {k: s[k] for k in ("ctx_size", "n_gpu_layers", "n_cpu_moe", "cpu_moe", "kv_offload", "cache_type_k", "cache_type_v",
                                  "ubatch_size", "mmap", "mlock", "flash_attn") if k in s}
        base.update({k: v for k, v in (cfg or {}).items() if v is not None})
        plan = planner.plan(hw, model, base, assumptions_from_settings(s), model_path=model_path, force=force)
        return plan, what_if, hw

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
