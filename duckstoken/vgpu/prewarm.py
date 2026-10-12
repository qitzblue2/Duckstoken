"""Sequential page-cache prewarm: the one kind of "prefetch" that exists at the storage tier.

llama.cpp memory-maps the model, so the first touch of every weight page is a page fault served from the drive in
whatever order the compute graph walks the tensors.  Reading the file front to back first lets the drive (and the OS
read-ahead) work sequentially and leaves the pages in the OS page cache, so the later faults are served from RAM.

What this is not: it does not make a hard drive faster, it only moves the sequential read earlier, and it only helps
when the model fits in free RAM next to everything else.  Therefore the read is bounded by ``max_bytes`` and by the
RAM that is really available (minus a reserve), re-checked while reading, and it stops on memory pressure or when the
caller cancels.  Reading more than fits would just evict the pages it read first.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

from .. import gguf, hardware
from .values import measured, unavailable

MIB = 2**20
GIB = 2**30
DEFAULT_RESERVE_GB = 4.0   # RAM that must stay available for the OS, the app and the model server's own buffers
NOTE = ("Prewarm reads the file once, front to back, so the OS page cache holds it before llama.cpp maps it. "
        "It does not make a slow drive faster and only helps when the model fits in free RAM. If part of the file "
        "was already cached, the speed shown reflects memory, not the drive.")


def _expand(path_or_paths) -> list[Path]:
    items = [path_or_paths] if isinstance(path_or_paths, (str, os.PathLike)) else list(path_or_paths or [])
    out: list[Path] = []
    for item in items:
        for p in gguf.shard_files(Path(str(item))):
            if p not in out:
                out.append(p)
    return out


def _default_avail_gb() -> float | None:
    gb = hardware.ram_info().get("avail_gb")
    return float(gb) if gb else None


def _default_pressure() -> str | None:
    """Memory pressure level from the telemetry modules; None when they are unavailable."""
    try:
        from . import pressure, telemetry
        return pressure.classify_pressure(telemetry.system_sample()).get("level")
    except Exception:  # noqa: BLE001 - monitoring must never break a prewarm
        return None


def prewarm(path_or_paths, max_bytes: int, cancel=None, progress: Callable | None = None,
            avail_ram_fn: Callable[[], float | None] | None = None, reserve_gb: float = DEFAULT_RESERVE_GB, *,
            pressure_fn: Callable[[], str | None] | None = None, chunk_bytes: int = 8 * MIB,
            check_every_bytes: int = 256 * MIB) -> dict:
    """Read the GGUF file(s) sequentially into the OS page cache.

    `avail_ram_fn()` returns available RAM in GB (default: the OS figure); `pressure_fn()` returns a pressure level
    string ('ok'|'elevated'|'high'|'critical') or None.  Returns ``{"bytes_read", "seconds", "mb_per_s", "completed",
    "stopped_because", ...}``; ``stopped_because`` is None when everything was read, else one of ``budget`` (max_bytes
    reached), ``ram_headroom`` (all the free RAM minus the reserve was used), ``cancelled``, ``memory_pressure``,
    ``no_ram_headroom``, ``ram_unknown``, ``no_files``, ``read_error``.  Never raises for I/O problems.
    """
    t0 = time.perf_counter()
    avail_fn = avail_ram_fn or _default_avail_gb
    pressure_fn = pressure_fn or _default_pressure
    chunk = max(4096, min(int(chunk_bytes), 16 * MIB))
    files = [p for p in _expand(path_or_paths) if p.is_file()]
    result = {"bytes_read": 0, "seconds": 0.0, "mb_per_s": unavailable("nothing was read", "wall clock"),
              "completed": False, "stopped_because": None, "requested_bytes": 0, "limit_bytes": 0,
              "files": [], "note": NOTE}

    def finish(reason: str | None, completed: bool = False) -> dict:
        secs = time.perf_counter() - t0
        n = result["bytes_read"]
        result.update(seconds=round(secs, 3), completed=completed, stopped_because=reason)
        if n and secs > 0:
            result["mb_per_s"] = measured(round(n / 1e6 / secs, 1), "wall clock", "includes reading from the OS cache")
        return result

    if not files:
        return finish("no_files")
    sizes = [f.stat().st_size for f in files]
    total = sum(sizes)
    result["requested_bytes"] = total
    result["files"] = [{"path": str(f), "size": s, "bytes_read": 0} for f, s in zip(files, sizes)]

    avail = avail_fn()
    if not avail or avail <= 0:
        return finish("ram_unknown")                      # cannot prove it fits: read nothing
    ram_limit = int(max(0.0, avail - float(reserve_gb)) * GIB)
    limit = min(int(max_bytes), ram_limit, total)
    result["limit_bytes"] = limit
    bound_by = "budget" if int(max_bytes) <= ram_limit else "ram_headroom"
    if limit <= 0:
        return finish("no_ram_headroom" if ram_limit <= 0 else "budget")

    buf = bytearray(chunk)
    view = memoryview(buf)
    read_total, next_check, last_report = 0, check_every_bytes, 0.0
    fadvise = getattr(os, "posix_fadvise", None)
    reason: str | None = None
    try:
        for index, path in enumerate(files):
            if read_total >= limit or reason:
                break
            with open(path, "rb", buffering=0) as f:
                fd = f.fileno()
                if fadvise is not None:
                    try:
                        fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                    except OSError:
                        pass
                offset = 0
                while offset < sizes[index] and read_total < limit:
                    if cancel is not None and cancel.is_set():
                        reason = "cancelled"
                        break
                    if read_total >= next_check:
                        next_check += check_every_bytes
                        left = avail_fn()
                        if (left is not None and left < reserve_gb) or pressure_fn() in ("high", "critical"):
                            reason = "memory_pressure"
                            break
                    want = min(chunk, sizes[index] - offset, limit - read_total)
                    if fadvise is not None:
                        try:
                            fadvise(fd, offset, min(want * 2, limit - read_total + want), os.POSIX_FADV_WILLNEED)
                        except OSError:
                            pass
                    n = f.readinto(view[:want])
                    if not n:
                        break                              # the file shrank under us
                    offset += n
                    read_total += n
                    result["bytes_read"] = read_total
                    result["files"][index]["bytes_read"] = offset
                    now = time.perf_counter()
                    if progress and now - last_report >= 0.25:
                        last_report = now
                        progress(read_total, limit, f"Reading {path.name}")
    except OSError as e:
        result["error"] = str(e)
        return finish("read_error")
    if progress:
        progress(read_total, limit, "Done reading" if not reason else "Stopped")
    if reason:
        return finish(reason)
    if read_total >= total:
        return finish(None, completed=True)
    return finish(bound_by if read_total >= limit else "read_error")
