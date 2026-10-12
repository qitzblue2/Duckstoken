"""Background sampler: a bounded ring buffer of `telemetry.system_sample` readings (and one process), plus window statistics.

Daemon thread, safe to start and stop repeatedly. Statistics are value dicts; when no sample in the window carried a
number the statistic is ``unavailable`` rather than 0.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque

from . import telemetry as _t
from .values import measured, num, unavailable

_na = unavailable
def _series(samples: list[dict], getter) -> list[float]:
    out = []
    for s in samples:
        try:
            v = num(getter(s))
        except (KeyError, IndexError, TypeError):
            v = None
        if v is not None:
            out.append(float(v))
    return out


def _stat(vals: list[float], how: str, source: str, nd: int = 2, note: str = ""):
    if not vals:
        return _na("no sample in the window had this value", source)
    f = {"min": min, "max": max, "mean": lambda v: sum(v) / len(v)}[how]
    return measured(round(f(vals), nd), source, note or f"{how} over the window")


class Sampler:
    """Background sampler with a bounded ring buffer (daemon thread; safe to start and stop repeatedly).

    ``pid_fn()`` returns the pid to watch (or None). ``sample_fn``/``process_fn`` are injectable for tests.
    """

    def __init__(self, interval_s: float = 1.0, maxlen: int = 600, pid_fn=None, *, gpu: bool = True, sample_fn=None,
                 process_fn=None, runner=None):
        self.interval_s = max(0.02, float(interval_s))
        self._buf: deque = deque(maxlen=max(1, int(maxlen)))
        self._pid_fn, self._gpu = pid_fn, gpu
        self._sample_fn = sample_fn or (lambda: _t.system_sample(runner or subprocess.run, gpu=gpu))
        self._process_fn = process_fn or (lambda pid: _t.process_sample(pid))
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.errors = 0

    @property
    def running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def sample_once(self) -> dict:
        """Take one sample now, append it to the buffer and return it. Never raises."""
        try:
            s = self._sample_fn()
        except Exception:
            self.errors += 1
            s = {"t": time.time(), "sources": {}, "error": "sample failed"}
        try:
            pid = self._pid_fn() if self._pid_fn else None
            if pid:
                s["process"] = self._process_fn(pid)
                s["pid"] = pid
        except Exception:
            self.errors += 1
        with self._lock:
            self._buf.append(s)
        return s

    def _run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self.sample_once()
            if stop.wait(self.interval_s):
                break

    def start(self) -> bool:
        """Start sampling; False when already running."""
        with self._lock:
            if self.running:
                return False
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(self._stop,), daemon=True, name="vgpu-sampler")
            self._thread.start()
            return True

    def stop(self, timeout: float = 3.0) -> None:
        with self._lock:
            t, ev = self._thread, self._stop
        ev.set()
        if t and t is not threading.current_thread():
            t.join(timeout)
        with self._lock:
            if self._thread is t and not (t and t.is_alive()):
                self._thread = None

    def latest(self) -> dict | None:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def history(self, n: int | None = None) -> list[dict]:
        with self._lock:
            items = list(self._buf)
        if n is None:
            return items
        n = max(int(n), 0)
        return items[-n:] if n else []

    def summary(self, window_s: float = 30.0) -> dict:
        """Statistics over the samples of the last `window_s` seconds (all value dicts; unavailable when nothing was read)."""
        with self._lock:
            items = list(self._buf)
        now = items[-1]["t"] if items else time.time()
        win = [s for s in items if now - s.get("t", 0) <= window_s]
        out = {"n": len(win), "window_s": window_s, "t_first": win[0]["t"] if win else None, "t_last": win[-1]["t"] if win else None}
        src = "sampler"
        out["ram_avail_gb"] = {"min": _stat(_series(win, lambda s: s["ram"]["avail_gb"]), "min", src),
                               "mean": _stat(_series(win, lambda s: s["ram"]["avail_gb"]), "mean", src)}
        out["ram_used_pct"] = {"max": _stat(_series(win, lambda s: s["ram"]["used_pct"]), "max", src, 1),
                               "mean": _stat(_series(win, lambda s: s["ram"]["used_pct"]), "mean", src, 1)}
        out["swap_used_gb"] = {"max": _stat(_series(win, lambda s: s["swap"]["used_gb"]), "max", src)}
        out["cpu_util_pct"] = {"mean": _stat(_series(win, lambda s: s["cpu"]["util_pct"]), "mean", src, 1),
                               "max": _stat(_series(win, lambda s: s["cpu"]["util_pct"]), "max", src, 1)}
        gpu_util = _series(win, lambda s: s["gpus"][0]["util_pct"])
        out["gpu_util_pct"] = {"mean": _stat(gpu_util, "mean", src, 1), "max": _stat(gpu_util, "max", src, 1)}
        out["vram_used_mb"] = {"max": _stat(_series(win, lambda s: s["gpus"][0]["vram_used_mb"]), "max", src, 0)}
        out["gpu_temp_c"] = {"max": _stat(_series(win, lambda s: s["gpus"][0]["temp_c"]), "max", src, 0)}
        out["gpu_sm_clock_mhz"] = {"min": _stat(_series(win, lambda s: s["gpus"][0]["sm_clock_mhz"]), "min", src, 0),
                                   "max": _stat(_series(win, lambda s: s["gpus"][0]["sm_clock_mhz"]), "max", src, 0)}
        out["rss_gb"] = {"max": _stat(_series(win, lambda s: s["process"]["rss_gb"]), "max", src, 3)}
        out["major_fault_rate_per_s"] = self._rate(win, "major_page_faults")
        out["page_fault_rate_per_s"] = self._rate(win, "page_faults")
        out["io_read_mb_per_s"] = self._rate(win, "io_read_gb", scale=1024.0)
        return out

    @staticmethod
    def _rate(win: list[dict], key: str, scale: float = 1.0) -> dict:
        pts = [(s["t"], num(s["process"][key])) for s in win if "process" in s and num(s["process"].get(key)) is not None]
        if len(pts) < 2 or pts[-1][0] <= pts[0][0]:
            return _na("needs at least two samples of this counter", "sampler")
        rate = (pts[-1][1] - pts[0][1]) * scale / (pts[-1][0] - pts[0][0])
        return measured(round(max(rate, 0.0), 3), "sampler", "change of the process counter over the window")
