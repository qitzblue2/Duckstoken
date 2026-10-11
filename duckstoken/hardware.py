"""Detect CPU, RAM, GPU and drives (Windows first, Linux good enough for development)."""

from __future__ import annotations

import json
import ntpath
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

_IS_WIN = os.name == "nt"
_NO_WINDOW = 0x08000000 if _IS_WIN else 0
_cache: dict = {}
_cache_lock = threading.Lock()


def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           creationflags=_NO_WINDOW)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _ps_json(script: str, timeout: int = 20):
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout)
    out = out.strip()
    if not out:
        return None
    try:
        data = json.loads(out)
    except ValueError:
        return None
    return data if isinstance(data, list) else [data]


# ------------------------------------------------------------------------ RAM
def ram_info() -> dict:
    total = avail = 0
    if _IS_WIN:
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMSTAT()
        st.dwLength = ctypes.sizeof(MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            total, avail = st.ullTotalPhys, st.ullAvailPhys
    else:
        try:
            kv = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, _, v = line.partition(":")
                kv[k] = int(v.split()[0]) * 1024
            total, avail = kv["MemTotal"], kv.get("MemAvailable", kv.get("MemFree", 0))
        except (OSError, ValueError, KeyError):
            pass
    return {"total_gb": round(total / 2**30, 1), "avail_gb": round(avail / 2**30, 1)}


def ram_modules() -> dict:
    """Speed / count of installed sticks (Windows). Used to estimate memory bandwidth."""
    speed, sticks = 0, 0
    if _IS_WIN:
        rows = _ps_json("Get-CimInstance Win32_PhysicalMemory | "
                        "Select-Object Speed,ConfiguredClockSpeed,Capacity | ConvertTo-Json")
        for r in rows or []:
            sticks += 1
            speed = max(speed, int(r.get("ConfiguredClockSpeed") or r.get("Speed") or 0))
    return {"speed_mts": speed, "sticks": sticks}


# ------------------------------------------------------------------------ CPU
def cpu_info() -> dict:
    name, physical, logical = "", 0, os.cpu_count() or 1
    avx2 = avx512 = False
    if _IS_WIN:
        rows = _ps_json("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
                        "NumberOfLogicalProcessors | ConvertTo-Json")
        if rows:
            name = rows[0].get("Name", "").strip()
            physical = sum(int(r.get("NumberOfCores") or 0) for r in rows)
            logical = sum(int(r.get("NumberOfLogicalProcessors") or 0) for r in rows) or logical
        try:
            import ctypes
            f = ctypes.windll.kernel32.IsProcessorFeaturePresent
            avx2, avx512 = bool(f(40)), bool(f(41))
        except Exception:  # noqa: BLE001
            pass
    else:
        try:
            txt = Path("/proc/cpuinfo").read_text()
            m = re.search(r"model name\s*:\s*(.+)", txt)
            name = m.group(1).strip() if m else ""
            flags = re.search(r"flags\s*:\s*(.+)", txt)
            fl = set(flags.group(1).split()) if flags else set()
            avx2, avx512 = "avx2" in fl, "avx512f" in fl
            cores = {(a, b) for a, b in re.findall(r"physical id\s*:\s*(\d+)[\s\S]*?core id\s*:\s*(\d+)", txt)}
            physical = len(cores)
        except OSError:
            pass
    return {"name": name or "Unknown CPU", "physical_cores": physical or max(1, logical // 2),
            "logical_cores": logical, "avx2": avx2, "avx512": avx512}


# ------------------------------------------------------------------------ GPU
def gpu_info() -> list[dict]:
    gpus = []
    smi = shutil.which("nvidia-smi")
    if smi:
        out = _run([smi, "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
                    "--format=csv,noheader,nounits"])
        for line in out.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                try:
                    gpus.append({"name": p[0], "vendor": "NVIDIA", "vram_mb": int(float(p[1])),
                                 "vram_free_mb": int(float(p[2])),
                                 "driver": p[3] if len(p) > 3 else "",
                                 "compute_cap": p[4] if len(p) > 4 else ""})
                except ValueError:
                    pass
    if not gpus and _IS_WIN:
        rows = _ps_json("Get-CimInstance Win32_VideoController | "
                        "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json")
        for r in rows or []:
            ram = int(r.get("AdapterRAM") or 0)
            # AdapterRAM is a 32-bit field: it reads 4 GB for any card >= 4 GB. Treat as "at least".
            gpus.append({"name": r.get("Name", "GPU"), "vendor": "", "vram_mb": ram // 2**20,
                         "vram_free_mb": ram // 2**20, "driver": r.get("DriverVersion", ""),
                         "compute_cap": ""})
    return gpus


# ---------------------------------------------------------------------- disks
def disk_info() -> list[dict]:
    disks: list[dict] = []
    if _IS_WIN:
        script = (
            "$pds = Get-PhysicalDisk; "
            "Get-Partition | Where-Object DriveLetter | ForEach-Object { "
            "$p=$_; $pd = $pds | Where-Object { [int]$_.DeviceId -eq $p.DiskNumber } | Select-Object -First 1; "
            "$v = Get-Volume -DriveLetter $p.DriveLetter -ErrorAction SilentlyContinue; "
            "[pscustomobject]@{Drive=\"$($p.DriveLetter):\"; Media=\"$($pd.MediaType)\"; Bus=\"$($pd.BusType)\"; "
            "Model=\"$($pd.FriendlyName)\"; SizeGB=[math]::Round($v.Size/1GB,1); "
            "FreeGB=[math]::Round($v.SizeRemaining/1GB,1)} } | ConvertTo-Json")
        for r in _ps_json(script, 30) or []:
            media, bus = str(r.get("Media") or ""), str(r.get("Bus") or "")
            disks.append({"drive": r.get("Drive"), "model": r.get("Model") or "", "bus": bus,
                          "media": media, "kind": classify_disk(media, bus),
                          "size_gb": r.get("SizeGB") or 0, "free_gb": r.get("FreeGB") or 0})
    if _IS_WIN and not disks:  # PowerShell storage cmdlets unavailable: fall back to plain drive info
        import string
        for c in string.ascii_uppercase:
            root = f"{c}:\\"
            if os.path.exists(root):
                try:
                    u = shutil.disk_usage(root)
                except OSError:
                    continue
                disks.append({"drive": f"{c}:", "model": "", "bus": "", "media": "", "kind": "unknown",
                              "size_gb": round(u.total / 2**30, 1), "free_gb": round(u.free / 2**30, 1)})
    elif not _IS_WIN:
        seen = set()
        for mount in ("/", str(Path.home())):
            try:
                u = shutil.disk_usage(mount)
            except OSError:
                continue
            if (u.total, u.free) in seen:
                continue
            seen.add((u.total, u.free))
            disks.append({"drive": mount, "model": "", "bus": "", "media": "", "kind": "unknown",
                          "size_gb": round(u.total / 2**30, 1), "free_gb": round(u.free / 2**30, 1)})
    return disks


def classify_disk(media: str, bus: str) -> str:
    """'nvme' | 'ssd' | 'hdd' | 'unknown'"""
    media, bus = media.lower(), bus.lower()
    if "nvme" in bus:
        return "nvme"
    if "ssd" in media:
        return "ssd"
    if "hdd" in media or "hard" in media or "rotat" in media:
        return "hdd"
    return "unknown"


def disk_for_path(path: str, disks: list[dict]) -> dict | None:
    if _IS_WIN:
        drive = ntpath.splitdrive(os.path.abspath(path) if os.name == "nt" else path)[0].upper()
        for d in disks:
            if (d.get("drive") or "").upper() == drive:
                return d
        return None
    return disks[0] if disks else None


# ----------------------------------------------------------------------- all
def bandwidth_gbs(speed_mts: int, channels: int = 2) -> float:
    """Theoretical RAM bandwidth. Unknown speed -> assume DDR4-2400 dual channel."""
    return round((speed_mts or 2400) * 8 * channels / 1000, 1)


def get_hardware(refresh: bool = False) -> dict:
    with _cache_lock:
        slow = _cache.get("slow")
        if refresh or not slow or time.time() - slow["t"] > 300:
            mods = ram_modules()
            slow = {"t": time.time(), "cpu": cpu_info(), "mods": mods, "disks": disk_info(),
                    "gpus": gpu_info()}
            _cache["slow"] = slow
        ram = ram_info()
    mods = slow["mods"]
    ram.update(mods)
    ram["bandwidth_gbs"] = bandwidth_gbs(mods["speed_mts"], 2 if mods["sticks"] != 1 else 1)
    return {"os": platform.platform(), "cpu": slow["cpu"], "ram": ram,
            "gpus": slow["gpus"], "disks": slow["disks"]}
