"""Live measurements: host memory and CPU, NVIDIA GPUs, one process, plus a bounded ring-buffer sampler.

Every number is a value dict (`values.py`); the background `Sampler` is in sampler.py (also reachable as `telemetry.Sampler`). A source that is missing or fails gives ``unavailable(reason)`` - never 0
and never a guess - and nothing in here raises. Parsing is separated from reading so the formats are unit-tested.

Sources
  Linux    /proc/meminfo, /proc/stat, /proc/pressure/memory (kernel PSI; "pressure_pct"), /proc/<pid>/{status,stat,io}
  Windows  GlobalMemoryStatusEx, GetPerformanceInfo (commit), GetSystemTimes, GetProcessMemoryInfo, GetProcessIoCounters,
           GetProcessTimes, Toolhelp thread snapshot (ctypes; the Windows code paths could not be run in the Linux
           development sandbox and are written defensively)
  GPU      ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits``. AMD and Intel GPUs have no equivalent tool that
           ships with every driver, so they are reported as unavailable rather than guessed.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .values import measured, unavailable

_IS_WIN = os.name == "nt"
_NO_WINDOW = 0x08000000 if _IS_WIN else 0
GIB = 2**30
MIB = 2**20

_lock = threading.Lock()
_cpu_prev: dict = {}     # last CPU counters, for utilisation between two calls
_win_prev: dict = {}


def _reset_state() -> None:
    """Forget the previous CPU counters (tests)."""
    with _lock:
        _cpu_prev.clear()
        _win_prev.clear()


def _na(reason: str, source: str = "") -> dict:
    return unavailable(reason, source)


def _safe(fn, what: str, source: str = ""):
    """Run `fn`; any exception becomes an unavailable value (telemetry must never raise)."""
    try:
        return fn()
    except Exception as e:
        return _na(f"{what} failed: {type(e).__name__}: {e}", source)


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


# ------------------------------------------------------------------------------------------------- Linux parsers
def parse_meminfo(text: str) -> dict:
    """/proc/meminfo -> {key: bytes}."""
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key.strip()] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1)
    return out


def parse_proc_stat(text: str) -> dict:
    """/proc/stat -> {"total": jiffies, "idle": jiffies, "cores": [(total, idle), ...]}."""
    cores, agg = [], None
    for line in text.splitlines():
        if not line.startswith("cpu"):
            continue
        f = line.split()
        try:
            vals = [int(x) for x in f[1:9]]
        except ValueError:
            continue
        vals += [0] * (8 - len(vals))
        total, idle = sum(vals[:8]), vals[3] + vals[4]  # idle + iowait
        if f[0] == "cpu":
            agg = (total, idle)
        else:
            cores.append((total, idle))
    if agg is None:
        raise ValueError("no aggregate cpu line")
    return {"total": agg[0], "idle": agg[1], "cores": cores}


def parse_psi(text: str) -> float | None:
    """/proc/pressure/memory 'some avg10' (percent of the last 10 s some task stalled on memory)."""
    for line in text.splitlines():
        if line.startswith("some"):
            for tok in line.split():
                if tok.startswith("avg10="):
                    try:
                        return float(tok[6:])
                    except ValueError:
                        return None
    return None


def parse_pid_stat(text: str) -> dict:
    """/proc/<pid>/stat -> minflt, majflt, utime, stime (clock ticks), num_threads. The command name may contain spaces."""
    rest = text[text.rindex(")") + 2:].split()  # fields from #3 (state) on
    return {"minflt": int(rest[7]), "majflt": int(rest[9]), "utime": int(rest[11]), "stime": int(rest[12]),
            "threads": int(rest[17])}


def parse_pid_status(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if key in ("VmRSS", "VmHWM") and parts:
            out[key] = int(parts[0]) * 1024
        elif key == "Threads" and parts:
            out["Threads"] = int(parts[0])
    return out


def parse_pid_io(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if rest.strip().isdigit():
            out[key.strip()] = int(rest)
    return out


def _cpu_util(total: int, idle: int, cores: list, source: str) -> dict:
    """Utilisation since the previous call (module state); the first call has nothing to compare with."""
    with _lock:
        prev = dict(_cpu_prev)
        _cpu_prev.update(total=total, idle=idle, cores=cores)
    if not prev:
        return {"util_pct": _na("first sample: needs a second reading to compute a rate", source),
                "per_core_pct": _na("first sample", source)}
    dt, di = total - prev["total"], idle - prev["idle"]
    if dt <= 0:
        return {"util_pct": _na("no CPU time elapsed between samples", source), "per_core_pct": _na("no time elapsed", source)}
    util = measured(round(100.0 * (dt - di) / dt, 1), source, "all logical CPUs, since the previous sample")
    per_core = _na("per-core data not available", source)
    pc, cc = prev.get("cores") or [], cores
    if pc and len(pc) == len(cc):
        vals = []
        for (t0, i0), (t1, i1) in zip(pc, cc):
            d = t1 - t0
            vals.append(round(100.0 * (d - (i1 - i0)) / d, 1) if d > 0 else 0.0)
        per_core = measured(vals, source, "per logical CPU, since the previous sample")
    return {"util_pct": util, "per_core_pct": per_core}


# ------------------------------------------------------------------------------------------------ Windows readers
def _win_memory() -> dict:
    import ctypes

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMSTAT()
    st.dwLength = ctypes.sizeof(MEMSTAT)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        raise OSError("GlobalMemoryStatusEx failed")
    out = {"total": st.ullTotalPhys, "avail": st.ullAvailPhys, "commit_limit": st.ullTotalPageFile,
           "commit_avail": st.ullAvailPageFile}

    class PERF(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong)] + [(n, ctypes.c_size_t) for n in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache", "KernelTotal",
            "KernelPaged", "KernelNonpaged", "PageSize")] + [("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong),
                                                              ("ThreadCount", ctypes.c_ulong)]

    pi = PERF()
    pi.cb = ctypes.sizeof(PERF)
    if ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        out["commit_used"], out["commit_limit"] = pi.CommitTotal * pi.PageSize, pi.CommitLimit * pi.PageSize
    return out


def _win_times() -> tuple[int, int]:
    """(busy, total) in 100 ns units since boot."""
    import ctypes
    from ctypes import wintypes

    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        raise OSError("GetSystemTimes failed")
    ft = lambda f: (f.dwHighDateTime << 32) | f.dwLowDateTime
    total = ft(kernel) + ft(user)  # kernel time includes idle time
    return total - ft(idle), total


# ---------------------------------------------------------------------------------------------------- host sample
def _ram_swap_linux() -> tuple[dict, dict, dict]:
    src = "/proc/meminfo"
    mi = parse_meminfo(_read_text(src))
    total, avail = mi["MemTotal"], mi.get("MemAvailable", mi.get("MemFree", 0))
    ram = {"total_gb": measured(round(total / GIB, 2), src), "avail_gb": measured(round(avail / GIB, 2), src),
           "used_pct": measured(round(100.0 * (1 - avail / total), 1), src, "100 - MemAvailable/MemTotal")}
    try:
        psi = parse_psi(_read_text("/proc/pressure/memory"))
        ram["pressure_pct"] = (measured(psi, "/proc/pressure/memory", "share of the last 10 s in which a task stalled on memory")
                               if psi is not None else _na("pressure file not understood", "/proc/pressure/memory"))
    except OSError:
        ram["pressure_pct"] = _na("kernel pressure-stall information (PSI) is not available", "/proc/pressure/memory")
    st, sf = mi.get("SwapTotal"), mi.get("SwapFree")
    if st is None or sf is None:
        swap = {"total_gb": _na("no swap information", src), "used_gb": _na("no swap information", src), "used_pct": _na("no swap information", src)}
    else:
        used = st - sf
        swap = {"total_gb": measured(round(st / GIB, 2), src), "used_gb": measured(round(used / GIB, 2), src),
                "used_pct": measured(round(100.0 * used / st, 1), src) if st else measured(0.0, src, "no swap configured")}
    cl, ca = mi.get("CommitLimit"), mi.get("Committed_AS")
    commit = ({"used_gb": measured(round(ca / GIB, 2), src, "Committed_AS (can exceed the limit with overcommit)"),
               "limit_gb": measured(round(cl / GIB, 2), src), "used_pct": measured(round(100.0 * ca / cl, 1), src) if cl else _na("no limit", src)}
              if cl is not None and ca is not None else
              {k: _na("no commit information", src) for k in ("used_gb", "limit_gb", "used_pct")})
    return ram, swap, commit


def _ram_swap_windows() -> tuple[dict, dict, dict]:
    src = "GlobalMemoryStatusEx"
    w = _win_memory()
    total, avail = w["total"], w["avail"]
    ram = {"total_gb": measured(round(total / GIB, 2), src), "avail_gb": measured(round(avail / GIB, 2), src),
           "used_pct": measured(round(100.0 * (1 - avail / total), 1), src),
           "pressure_pct": _na("Windows has no memory stall counter; read avail_gb and the commit charge instead", src)}
    swap = {k: _na("Windows reports the commit charge (see commit), not swap use", src) for k in ("total_gb", "used_gb", "used_pct")}
    limit = w.get("commit_limit")
    used = w.get("commit_used", (limit - w["commit_avail"]) if limit else None)
    commit = ({"used_gb": measured(round(used / GIB, 2), "GetPerformanceInfo"), "limit_gb": measured(round(limit / GIB, 2), "GetPerformanceInfo"),
               "used_pct": measured(round(100.0 * used / limit, 1), "GetPerformanceInfo")}
              if limit and used is not None else {k: _na("no commit information", src) for k in ("used_gb", "limit_gb", "used_pct")})
    return ram, swap, commit


def _cpu_sample() -> tuple[dict, dict]:
    """(cpu dict, source value)."""
    if _IS_WIN:
        busy, total = _win_times()
        out = _cpu_util(total, total - busy, [], "GetSystemTimes")
        out["per_core_pct"] = _na("per-core utilisation is not read on Windows", "GetSystemTimes")
        return out, measured("GetSystemTimes", "GetSystemTimes")
    st = parse_proc_stat(_read_text("/proc/stat"))
    return _cpu_util(st["total"], st["idle"], st["cores"], "/proc/stat"), measured("/proc/stat", "/proc/stat")


# --------------------------------------------------------------------------------------------------------- GPU
GPU_FIELDS = ["index", "name", "utilization.gpu", "utilization.memory", "memory.used", "memory.total", "temperature.gpu",
              "power.draw", "clocks.sm", "pcie.link.gen.current", "pcie.link.width.current"]
_GPU_CORE_FIELDS = GPU_FIELDS[:7]  # fallback for old nvidia-smi builds that reject the newer fields
_NA_TOKENS = {"", "n/a", "[n/a]", "[not supported]", "not supported", "[unknown error]", "[insufficient permissions]"}


def _find_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if _IS_WIN:
        for p in (r"C:\Windows\System32\nvidia-smi.exe", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
            if os.path.isfile(p):
                return p
    return None


def parse_nvidia_smi(text: str, fields: list[str] | None = None) -> list[dict]:
    """CSV rows from nvidia-smi -> GPU dicts of value dicts. '[N/A]' and '[Not Supported]' become unavailable."""
    fields = fields or GPU_FIELDS
    src = "nvidia-smi"
    gpus = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row or len(row) < 2:
            continue
        rec = dict(zip(fields, [c.strip() for c in row]))

        def val(key, cast=float, nd=None, note="", rec=rec):
            raw = rec.get(key)
            if raw is None:
                return _na(f"{key} not requested", src)
            if raw.lower() in _NA_TOKENS:
                return _na(f"{key} is not reported by this GPU/driver ({raw or 'empty'})", src)
            try:
                v = cast(raw)
            except ValueError:
                return _na(f"{key} not understood: {raw!r}", src)
            return measured(round(v, nd) if nd is not None else v, src, note)

        try:
            idx = int(rec.get("index", len(gpus)))
        except ValueError:
            idx = len(gpus)
        gpus.append({"index": idx, "name": measured(rec.get("name", ""), src),
                     "util_pct": val("utilization.gpu"), "mem_util_pct": val("utilization.memory", note="memory controller busy time"),
                     "vram_used_mb": val("memory.used"), "vram_total_mb": val("memory.total"),
                     "temp_c": val("temperature.gpu"), "power_w": val("power.draw", nd=1),
                     "sm_clock_mhz": val("clocks.sm"), "pcie_gen": val("pcie.link.gen.current", int),
                     "pcie_width": val("pcie.link.width.current", int)})
    return gpus


def gpu_sample(runner=subprocess.run, timeout: float = 5.0) -> tuple[list[dict], dict]:
    """(gpus, source). ``gpus == []`` plus an unavailable source when nvidia-smi is missing or fails."""
    exe = _find_nvidia_smi()
    if not exe:
        return [], _na("nvidia-smi not found", "nvidia-smi")
    last_err = ""
    for fields in (GPU_FIELDS, _GPU_CORE_FIELDS):
        cmd = [exe, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
        kw = {"capture_output": True, "text": True, "timeout": timeout}
        if _IS_WIN:
            kw["creationflags"] = _NO_WINDOW
        try:
            r = runner(cmd, **kw)
        except subprocess.TimeoutExpired:
            return [], _na(f"nvidia-smi timed out after {timeout:g} s", "nvidia-smi")
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            return [], _na(f"nvidia-smi could not run: {e}", "nvidia-smi")
        out = getattr(r, "stdout", "") or ""
        if getattr(r, "returncode", 1) == 0 and out.strip():
            gpus = parse_nvidia_smi(out, fields)
            if gpus:
                return gpus, measured("nvidia-smi", "nvidia-smi")
        last_err = ((getattr(r, "stderr", "") or out).strip().splitlines() or ["no output"])[0][:200]
    return [], _na(f"nvidia-smi gave no usable data: {last_err}", "nvidia-smi")


def system_sample(runner=subprocess.run, gpu: bool = True) -> dict:
    """One reading of host RAM/swap/commit, CPU and (optionally) NVIDIA GPUs. Never raises.

    CPU utilisation is the change since the *previous call in this process*, so the first call reports it unavailable.
    """
    fail = lambda e: _na(f"{type(e).__name__}: {e}")
    sample = {"t": time.time(), "sources": {}}
    try:
        ram, swap, commit = _ram_swap_windows() if _IS_WIN else _ram_swap_linux()
        sample["sources"]["ram"] = measured("GlobalMemoryStatusEx" if _IS_WIN else "/proc/meminfo")
    except Exception as e:
        keys = ("total_gb", "avail_gb", "used_pct", "pressure_pct")
        ram = {k: fail(e) for k in keys}
        swap = {k: fail(e) for k in ("total_gb", "used_gb", "used_pct")}
        commit = {k: fail(e) for k in ("used_gb", "limit_gb", "used_pct")}
        sample["sources"]["ram"] = _na(f"memory counters unreadable: {type(e).__name__}: {e}")
    sample.update(ram=ram, swap=swap, commit=commit)
    try:
        cpu, src = _cpu_sample()
        sample["sources"]["cpu"] = src
    except Exception as e:
        cpu = {"util_pct": fail(e), "per_core_pct": fail(e)}
        sample["sources"]["cpu"] = _na(f"CPU counters unreadable: {type(e).__name__}: {e}")
    sample["cpu"] = cpu
    if gpu:
        try:
            gpus, gsrc = gpu_sample(runner)
        except Exception as e:
            gpus, gsrc = [], _na(f"GPU query failed: {type(e).__name__}: {e}", "nvidia-smi")
    else:
        gpus, gsrc = [], _na("GPU polling disabled for this sample", "nvidia-smi")
    sample["gpus"], sample["sources"]["gpu"] = gpus, gsrc
    sample["sources"]["gpu_other"] = _na("AMD and Intel GPUs have no vendor-neutral utilisation tool; not sampled", "")
    return sample


# ------------------------------------------------------------------------------------------------ process sample
_PROC_KEYS = ("rss_gb", "peak_rss_gb", "page_faults", "major_page_faults", "cpu_seconds", "io_read_gb", "io_write_gb", "threads")


def _proc_unavailable(reason: str) -> dict:
    return {k: _na(reason, "process") for k in _PROC_KEYS}


def _process_linux(pid: int) -> dict:
    base = f"/proc/{pid}"
    out = {}
    try:
        st = parse_pid_status(_read_text(f"{base}/status"))
        out["rss_gb"] = measured(round(st["VmRSS"] / GIB, 4), f"{base}/status", "resident set (includes file-backed mmap pages)")
        out["peak_rss_gb"] = measured(round(st["VmHWM"] / GIB, 4), f"{base}/status", "VmHWM")
        out["threads"] = measured(st["Threads"], f"{base}/status")
    except (OSError, KeyError, ValueError) as e:
        if isinstance(e, (FileNotFoundError, ProcessLookupError)):
            return _proc_unavailable("process not found")
        for k in ("rss_gb", "peak_rss_gb", "threads"):
            out[k] = _na(f"/proc status unreadable: {type(e).__name__}", base)
    try:
        ps = parse_pid_stat(_read_text(f"{base}/stat"))
        tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        out["page_faults"] = measured(ps["minflt"] + ps["majflt"], f"{base}/stat", "minor + major faults since start")
        out["major_page_faults"] = measured(ps["majflt"], f"{base}/stat", "faults that needed storage I/O")
        out["cpu_seconds"] = measured(round((ps["utime"] + ps["stime"]) / tck, 3), f"{base}/stat", "user + system")
    except (OSError, ValueError, IndexError) as e:
        for k in ("page_faults", "major_page_faults", "cpu_seconds"):
            out[k] = _na(f"/proc stat unreadable: {type(e).__name__}", base)
    try:
        io_ = parse_pid_io(_read_text(f"{base}/io"))
        out["io_read_gb"] = measured(round(io_["read_bytes"] / GIB, 4), f"{base}/io", "bytes fetched from the storage layer")
        out["io_write_gb"] = measured(round(io_["write_bytes"] / GIB, 4), f"{base}/io")
    except (OSError, KeyError, ValueError):
        for k in ("io_read_gb", "io_write_gb"):
            out[k] = _na("/proc/<pid>/io is not readable for this process (needs the same user or root)", f"{base}/io")
    return out


def _process_windows(pid: int) -> dict:
    import ctypes
    from ctypes import wintypes

    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    # explicit signatures: without them ctypes passes handles as 32-bit ints
    k32.OpenProcess.restype, k32.OpenProcess.argtypes = wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.GetProcessIoCounters.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.c_void_p] * 4
    h = k32.OpenProcess(0x0410, False, pid) or k32.OpenProcess(0x1000, False, pid)  # QUERY_INFORMATION|VM_READ, else LIMITED
    if not h:
        return _proc_unavailable("process not found or access denied")
    out = {}
    try:
        # A handle to a process that has exited stays valid while anyone holds it (our own Popen object does), and the memory
        # counters then still answer: only STILL_ACTIVE (259) means there is something to measure.
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        code = wintypes.DWORD()
        if k32.GetExitCodeProcess(h, ctypes.byref(code)) and code.value != 259:
            return _proc_unavailable("the process has exited")
        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            src = "GetProcessMemoryInfo"
            out["rss_gb"] = measured(round(pmc.WorkingSetSize / GIB, 4), src, "working set (includes mapped model pages)")
            out["peak_rss_gb"] = measured(round(pmc.PeakWorkingSetSize / GIB, 4), src)
            out["page_faults"] = measured(int(pmc.PageFaultCount), src, "hard + soft faults since start")
        else:
            for k in ("rss_gb", "peak_rss_gb", "page_faults"):
                out[k] = _na("GetProcessMemoryInfo failed", "GetProcessMemoryInfo")
        out["major_page_faults"] = _na("Windows counts hard and soft faults together", "GetProcessMemoryInfo")

        class IOC(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

        ioc = IOC()
        if k32.GetProcessIoCounters(h, ctypes.byref(ioc)):
            note = "bytes requested through ReadFile/WriteFile; mmap page-ins are not counted"
            out["io_read_gb"] = measured(round(ioc.ReadBytes / GIB, 4), "GetProcessIoCounters", note)
            out["io_write_gb"] = measured(round(ioc.WriteBytes / GIB, 4), "GetProcessIoCounters", note)
        else:
            out["io_read_gb"] = out["io_write_gb"] = _na("GetProcessIoCounters failed", "GetProcessIoCounters")

        c, e, kt, ut = (wintypes.FILETIME() for _ in range(4))
        if k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut)):
            ft = lambda f: (f.dwHighDateTime << 32) | f.dwLowDateTime
            out["cpu_seconds"] = measured(round((ft(kt) + ft(ut)) / 1e7, 3), "GetProcessTimes", "kernel + user")
        else:
            out["cpu_seconds"] = _na("GetProcessTimes failed", "GetProcessTimes")
    finally:
        k32.CloseHandle(h)
    out["threads"] = _win_thread_count(pid)
    return out


def _win_thread_count(pid: int) -> dict:
    import ctypes
    from ctypes import wintypes

    class TE(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ThreadID", wintypes.DWORD),
                    ("th32OwnerProcessID", wintypes.DWORD), ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD)]

    k32 = ctypes.windll.kernel32
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.Thread32First.argtypes = k32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    snap = k32.CreateToolhelp32Snapshot(0x4, 0)
    if not snap or snap == wintypes.HANDLE(-1).value:
        return _na("thread snapshot failed", "Toolhelp32")
    try:
        te = TE()
        te.dwSize = ctypes.sizeof(TE)
        n, ok = 0, k32.Thread32First(snap, ctypes.byref(te))
        while ok:
            n += te.th32OwnerProcessID == pid
            ok = k32.Thread32Next(snap, ctypes.byref(te))
        return measured(n, "Toolhelp32")
    finally:
        k32.CloseHandle(snap)


def process_sample(pid) -> dict:
    """Counters of one process: rss_gb, peak_rss_gb, page_faults, major_page_faults, cpu_seconds, io_read_gb, io_write_gb, threads.

    Each is a value dict; anything unreadable (process gone, no permission) is unavailable with the reason. Never raises.
    """
    try:
        pid = int(pid)
        if pid <= 0:
            return _proc_unavailable("no process id")
        out = _process_windows(pid) if _IS_WIN else _process_linux(pid)
    except Exception as e:
        return _proc_unavailable(f"process counters failed: {type(e).__name__}: {e}")
    for k in _PROC_KEYS:
        out.setdefault(k, _na("not measured", "process"))
    return out


# ----------------------------------------------------------------------------------------------- bandwidth probe
def measure_host_bandwidth(size_mb: int = 64, repeats: int = 3) -> dict:
    """Single-thread RAM copy rate in GB/s (best of `repeats`), a *lower bound* on usable memory bandwidth.

    Copies `size_mb` MiB between two bytearrays (a memcpy). The bus carries the data twice (read + write), and llama.cpp
    reads with many threads, so the multi-threaded read bandwidth is higher than this figure.
    """
    src_name = "bytearray memcpy"
    try:
        size = int(max(32, min(int(size_mb), 512))) * MIB  # >= 32 MiB so the copy does not fit in the CPU cache
        a, b = bytearray(size), bytearray(size)
        a[::4096] = b"\x01" * len(a[::4096])  # touch every page so the OS has really mapped them
        b[::4096] = b"\x01" * len(b[::4096])
        va, vb = memoryview(a), memoryview(b)
        best = 0.0
        for _ in range(max(1, int(repeats))):
            t0 = time.perf_counter()
            vb[:] = va
            dt = time.perf_counter() - t0
            if dt > 0:
                best = max(best, size / dt / 1e9)
        del va, vb, a, b
        if best <= 0:
            return _na("timer resolution too coarse", src_name)
        return measured(round(best, 2), src_name, "single-thread copy rate (GB/s); lower bound of multi-threaded read bandwidth")
    except (MemoryError, OverflowError, ValueError, TypeError) as e:
        return _na(f"could not allocate the test buffers: {type(e).__name__}", src_name)


def __getattr__(name: str):
    """`telemetry.Sampler` lives in sampler.py (which imports this module); resolved lazily to avoid a circular import."""
    if name == "Sampler":
        from .sampler import Sampler
        return Sampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
