"""Sampling and aggregation for the benchmark: peak RSS, page-fault deltas, CPU/GPU utilisation, host bandwidth.

Everything is a values dict: a metric that could not be read is ``unavailable`` with the reason, never 0.
"""

from __future__ import annotations

import time
from typing import Callable

from .lazy import module as _mod
from .values import measured, num, unavailable


def make_sampler(pid: int, interval_s: float, sampler_fn: Callable[[int], dict] | None = None):
    """A `telemetry.Sampler` watching `pid` (None when telemetry is unavailable).

    `sampler_fn(pid) -> {"process": ..., "system": ...}` replaces the real telemetry readings (tests, other platforms).
    """
    tel = _mod("telemetry")
    if tel is None:
        return None
    try:
        if sampler_fn is None:
            return tel.Sampler(interval_s=interval_s, maxlen=4000, pid_fn=lambda: pid)

        def sample_fn() -> dict:
            got = sampler_fn(pid) or {}
            flat = dict(got.get("system") or {})
            if got.get("process") is not None:
                flat["process"] = got["process"]
            flat.setdefault("t", time.time())
            return flat

        return tel.Sampler(interval_s=interval_s, maxlen=4000, sample_fn=sample_fn)
    except Exception:  # noqa: BLE001
        return None


_HOST_BW: dict = {"t": 0.0, "value": None}
_HOST_BW_TTL = 600.0


def host_bandwidth() -> dict | None:
    """Host RAM copy bandwidth (a values dict), measured once per ten minutes: it is a property of the machine, and
    measuring before the server starts keeps the measurement from disturbing a run."""
    if _HOST_BW["value"] is not None and time.time() - _HOST_BW["t"] < _HOST_BW_TTL:
        return _HOST_BW["value"]
    tel = _mod("telemetry")
    if tel is None:
        return None
    try:
        value = tel.measure_host_bandwidth()
    except Exception:  # noqa: BLE001
        return None
    _HOST_BW.update(t=time.time(), value=value)
    return value


def _leaf(sample: dict, *path):
    cur = sample
    for p in path:
        cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur


def aggregate_samples(samples: list[dict], host_bw: dict | None = None) -> dict:
    """Peak/mean/delta metrics from `telemetry.Sampler` readings (``{"t", "process", "cpu", "gpus", ...}`` dicts),
    each a values dict: unavailable, with the reason, when it could not be read.  Nothing is ever filled with 0."""
    src = "benchmark sampler"

    def series(*path):
        return [(s.get("t", 0.0), v) for s in samples if (v := num(_leaf(s, *path))) is not None]

    def why(*path):
        for s in samples:
            leaf = _leaf(s, *path)
            if isinstance(leaf, dict) and leaf.get("note"):
                return leaf["note"]
        return "telemetry sampler produced no data" if not samples else "not reported on this machine"

    rss = series("process", "rss_gb") + series("process", "peak_rss_gb")
    out = {"peak_rss_gb": measured(round(max(v for _, v in rss), 3), src, "resident set of the server process (includes mmap'd model pages)")
           if rss else unavailable(why("process", "rss_gb"), src)}

    def delta(name, digits=0, note=""):
        pts = series("process", name)
        return (measured(round(max(pts[-1][1] - pts[0][1], 0), digits) if digits else int(max(pts[-1][1] - pts[0][1], 0)), src, note)
                if len(pts) >= 2 else unavailable(why("process", name), src))

    out["page_faults"] = delta("page_faults", note="minor + major page faults during the benchmark")
    out["major_page_faults"] = delta("major_page_faults", note="faults that needed storage I/O")
    out["io_read_gb"] = delta("io_read_gb", 4, "bytes the process fetched from storage")
    cpu_t = series("process", "cpu_seconds")
    out["process_cpu_cores"] = (measured(round(max(cpu_t[-1][1] - cpu_t[0][1], 0.0) / max(cpu_t[-1][0] - cpu_t[0][0], 1e-6), 2), src,
                                         "average cores kept busy by the server process")
                                if len(cpu_t) >= 2 else unavailable(why("process", "cpu_seconds"), src))
    cpu = series("cpu", "util_pct")
    out["cpu_util_pct"] = (measured(round(sum(v for _, v in cpu) / len(cpu), 1), src, "whole-machine CPU utilisation, mean")
                           if cpu else unavailable(why("cpu", "util_pct"), src))
    gutil, gmem = [], []
    for s in samples:
        gpus = s.get("gpus") or []
        u = [x for g in gpus if (x := num(g.get("util_pct"))) is not None]
        m = [x for g in gpus if (x := num(g.get("vram_used_mb"))) is not None]
        if u:
            gutil.append(max(u))
        if m:
            gmem.append(max(m))
    nogpu = next((v.get("note") for s in samples for v in [(s.get("sources") or {}).get("gpu")]
                  if isinstance(v, dict) and v.get("note")), "no GPU telemetry (nvidia-smi not found or no NVIDIA GPU)")
    out["gpu_util_pct"] = (measured(round(sum(gutil) / len(gutil), 1), src, "busiest GPU, mean")
                           if gutil else unavailable(nogpu, src))
    out["peak_vram_mb"] = (measured(round(max(gmem)), src, "device-wide (nvidia-smi memory.used), includes other applications")
                           if gmem else unavailable(nogpu, src))
    out["host_bandwidth_gbs"] = host_bw or unavailable("not measured", "telemetry.measure_host_bandwidth")
    out["transfer_bandwidth_gbs"] = unavailable(
        "llama-server does not expose host<->GPU transfer bandwidth; PCIe link speed (nvidia-smi) is theoretical only",
        "llama-server")
    out["samples"] = len(samples)
    return out
