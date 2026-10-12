"""Crash and failed-load recovery for the model server.

The watchdog never touches a server the user stopped. When the server dies on its own (OOM kill, driver reset, bad
allocation) it classifies the failure from the log, and - only if `server_auto_restart` is on - restarts it with a safer
configuration. The overrides are applied to that one start; the user's saved settings are never rewritten.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable

LIVE = ("starting", "loading", "ready")
STABLE_SECONDS = 60  # a restart that stays ready this long resets the crash counter

_VRAM_OOM = re.compile(r"cudamalloc failed|out of memory|failed to allocate .*(cuda|vulkan|hip|device)|"
                       r"vk::.*outofdevicememory|errorOutOfDeviceMemory|unable to allocate .*(cuda|vulkan)|"
                       r"ggml_backend_cuda_buffer_type_alloc_buffer", re.I)
_RAM_OOM = re.compile(r"unable to allocate|failed to allocate|bad_alloc|cannot allocate memory|"
                      r"not enough memory|std::bad_alloc|insufficient memory|\bkilled\b", re.I)
_DEVICE = re.compile(r"cuda|vulkan|\bhip\b|\bgpu\b|device", re.I)
# Specific causes only: llama.cpp wraps *every* failed load in "error loading model / failed to load model", including plain
# out-of-memory, so those generic lines must not decide anything.
_BAD_MODEL = re.compile(r"unknown model architecture|invalid magic|tensor .* data is not within the file|unsupported (tensor|gguf)|"
                        r"unknown gguf|gguf_init_from_file.*failed|wrong magic|not a gguf|unsupported .*quantization", re.I)


def classify_crash(lines: list[str], exit_code: int | None = None) -> dict:
    """What most likely went wrong, from the tail of the server log. Always one of the kinds below; `unknown` is honest."""
    tail = lines[-200:]
    text = "\n".join(tail)
    kind, why = "unknown", "The server stopped without saying why. See the log."
    # Out-of-memory first: it is the commonest reason a big model fails to load, and it is fixable by restarting smaller.
    if any(_VRAM_OOM.search(ln) and _DEVICE.search(ln) for ln in tail):
        kind, why = "vram_oom", "The graphics card ran out of memory."
    elif _RAM_OOM.search(text) or exit_code in (137, -9):
        kind, why = "ram_oom", "The PC ran out of memory (or Windows/Linux killed the server to protect itself)."
    elif _BAD_MODEL.search(text):
        kind, why = "bad_model", "The engine could not read this model (unsupported architecture or a damaged file). Restarting will not help."
    return {"kind": kind, "why": why, "exit_code": exit_code}


def safer_overrides(attempt: int, crash_kind: str, settings: dict) -> dict:
    """Settings overrides for restart number `attempt` (1-based). Attempt 1 retries unchanged unless the cause is clear."""
    ctx = int(settings.get("ctx_size") or 8192) or 8192
    ngl = int(settings.get("n_gpu_layers", 0))
    out: dict = {}
    if crash_kind == "bad_model":
        return out
    if crash_kind == "vram_oom" or attempt >= 2:
        if ngl != 0:
            out["n_gpu_layers"] = 0 if attempt >= 3 else max(ngl // 2, 0) if ngl > 1 else 0
        out["kv_offload"] = False
    if crash_kind == "ram_oom" or attempt >= 2:
        out["ctx_size"] = max(2048, ctx // 2)
        out["mlock"] = False
        out["cache_type_k"] = out["cache_type_v"] = "q8_0"
        out["flash_attn"] = "on"
    return out


class CrashWatchdog:
    """Polls `server.status()`; on a crash of a live server decides whether and how to restart it."""

    def __init__(self, server, settings_fn: Callable[[], dict], perflog=None, on_event: Callable[[dict], None] | None = None,
                 poll_s: float = 1.0, clock: Callable[[], float] = time.time):
        self.server, self.settings_fn, self.perflog, self.on_event = server, settings_fn, perflog, on_event
        self.poll_s, self.clock = poll_s, clock
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._prev = "stopped"
        self._base: dict | None = None  # settings of the start we are guarding
        self._ready_since = 0.0
        self.restarts = 0
        self.events: list[dict] = []

    # --------------------------------------------------------------- lifecycle
    def note_start(self, settings: dict) -> None:
        """The user (or autostart) started the server with `settings`: guard it. A fresh start resets the counter."""
        with self._lock:
            self._base, self.restarts, self._ready_since = dict(settings), 0, 0.0

    def note_stop(self) -> None:
        """The user stopped the server on purpose: never restart it."""
        with self._lock:
            self._base, self._prev = None, "stopped"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="crash-watchdog")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t is not threading.current_thread():
            t.join(timeout=3)

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - a watchdog bug must never take the app down
                self._emit({"type": "watchdog_error", "error": f"{e.__class__.__name__}: {e}"})

    # -------------------------------------------------------------------- core
    def tick(self) -> dict | None:
        """One poll; returns the event emitted (crash handling), if any. Public so tests drive it deterministically."""
        st = self.server.status()
        now, state = self.clock(), st.get("state", "stopped")
        with self._lock:
            prev, base = self._prev, self._base
            self._prev = state
            if state == "ready":
                self._ready_since = self._ready_since or now
                if self.restarts and now - self._ready_since >= STABLE_SECONDS:
                    self.restarts = 0
            else:
                self._ready_since = 0.0
            if not (prev in LIVE and state == "error" and base is not None):
                return None
            settings = self.settings_fn()
            lines = self.server.logs()["lines"] if hasattr(self.server, "logs") else []
            crash = classify_crash(lines, _exit_code(st.get("error", "")))
            enabled = bool(settings.get("server_auto_restart"))
            limit = int(settings.get("server_restart_max", 2))
            event = {"type": "crash", "t": now, "crash": crash, "error": st.get("error", ""), "restarts": self.restarts,
                     "action": "none"}
            if not enabled:
                event["action"] = "none"
            elif crash["kind"] == "bad_model":
                event["action"] = "gave_up"
                event["why"] = crash["why"]
            elif self.restarts >= limit:
                event["action"] = "gave_up"
                event["why"] = f"Reached the limit of {limit} automatic restarts."
            else:
                self.restarts += 1
                over = safer_overrides(self.restarts, crash["kind"], base)
                try:
                    self.server.start({**base, **over})
                    event.update(action="restarted", attempt=self.restarts, overrides=over)
                except Exception as e:  # noqa: BLE001
                    event.update(action="restart_failed", error=f"{e}", overrides=over)
        self._emit(event)
        return event

    def _emit(self, event: dict) -> None:
        with self._lock:
            self.events.append(event)
            del self.events[:-50]
        if self.perflog:
            try:
                self.perflog.log("server_" + event["type"], **{k: v for k, v in event.items() if k != "type"})
            except Exception:  # noqa: BLE001
                pass
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:  # noqa: BLE001
                pass

    def status(self) -> dict:
        with self._lock:
            return {"guarding": self._base is not None, "restarts": self.restarts, "events": list(self.events[-10:])}


def _exit_code(error: str) -> int | None:
    m = re.search(r"code (-?\d+)", error or "")
    return int(m.group(1)) if m else None
