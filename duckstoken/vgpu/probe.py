"""Ask a llama-server binary what it can do, without loading a model.

`probe_engine` runs ``llama-server --version`` and ``llama-server --list-devices`` (the supported ways to read the build
and the GPUs the build can actually use) and parses the text leniently. `scan_arch_support` looks for architecture names
inside the engine binaries: llama.cpp's architecture table (``LLM_ARCH_NAMES`` in src/llama-arch.cpp) stores each name
as a NUL-terminated string literal, so the string being present in ``llama.dll`` / ``libllama.so`` means the architecture
is compiled in. That is evidence, not proof (a name could also occur in unrelated text), and it is reported as such.

Output formats, verified against a real build (commit 08ffba2, Linux CPU; both ``--version`` lines go to *stderr*,
``--list-devices`` to stdout)::

    version: 0.6.0-dev (build 1, commit 08ffba2)          # older builds: "version: 6123 (abc1234)"
    built with GNU 13.3.0 for Linux x86_64
    Available devices:
      CUDA0: NVIDIA GeForce RTX 3060 (12288 MiB, 11000 MiB free)      # "  (none)" when only the CPU backend exists
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path

_IS_WIN = os.name == "nt"
_NO_WINDOW = 0x08000000 if _IS_WIN else 0

DEFAULT_ARCHS = ("gpt-oss", "qwen3moe", "qwen2", "llama", "gemma3", "phi3", "deepseek2")
_CHUNK = 4 * 1024 * 1024
_MAX_SCAN_BYTES = 512 * 1024 * 1024  # a single file bigger than this is skipped (GPU kernel libraries, not llama.dll)

_VERSION_NEW = re.compile(r"version:\s*(\S+)\s*\(\s*build\s+(\d+)\s*,\s*commit\s+([0-9a-fA-F]+)\s*\)")
_VERSION_OLD = re.compile(r"version:\s*(\d+)\s*\(\s*([0-9a-fA-F]+)\s*\)")
_DEVICE = re.compile(r"^\s*([A-Za-z_][\w\-\[\]:.]*?)\s*:\s*(.+?)\s*\(\s*(\d+)\s*MiB\s*,\s*(\d+)\s*MiB\s*free\s*\)\s*$")
_BUILT_WITH = re.compile(r"built with\s+(.+?)\s+for\s+(.+)", re.IGNORECASE)
# device-id prefix -> backend name (ggml_backend_dev_name: CUDA0, Vulkan0, SYCL0, MTL0, ROCm0, ...)
_BACKEND_PREFIX = [("cuda", "CUDA"), ("rocm", "HIP"), ("hip", "HIP"), ("musa", "MUSA"), ("vulkan", "Vulkan"), ("sycl", "SYCL"),
                   ("mtl", "Metal"), ("metal", "Metal"), ("gpuopencl", "OpenCL"), ("opencl", "OpenCL"), ("cann", "CANN"),
                   ("rpc", "RPC"), ("htp", "Hexagon"), ("webgpu", "WebGPU"), ("blas", "BLAS"), ("cpu", "CPU")]
# sibling library file -> backend that was compiled into the build
_LIB_BACKENDS = [("ggml-cuda", "CUDA"), ("ggml-hip", "HIP"), ("ggml-musa", "MUSA"), ("ggml-vulkan", "Vulkan"),
                 ("ggml-sycl", "SYCL"), ("ggml-opencl", "OpenCL"), ("ggml-metal", "Metal"), ("ggml-cann", "CANN"),
                 ("ggml-rpc", "RPC"), ("ggml-blas", "BLAS")]


def backend_of(device_id: str) -> str:
    low = device_id.lower()
    for prefix, name in _BACKEND_PREFIX:
        if low.startswith(prefix):
            return name
    m = re.match(r"[A-Za-z]+", device_id)
    return m.group(0) if m else "unknown"


def parse_version(text: str) -> dict:
    """{"version","build","commit","compiler","target"}; every field None when the text does not say."""
    out = {"version": None, "build": None, "commit": None, "compiler": None, "target": None}
    m = _VERSION_NEW.search(text or "")
    if m:
        out.update(version=m.group(1), build=int(m.group(2)), commit=m.group(3))
    else:
        m = _VERSION_OLD.search(text or "")
        if m:
            out.update(version=m.group(1), build=int(m.group(1)), commit=m.group(2))
    b = _BUILT_WITH.search(text or "")
    if b:
        out.update(compiler=b.group(1).strip(), target=b.group(2).strip())
    return out


def parse_devices(text: str) -> list[dict]:
    """Devices from ``--list-devices`` output: ``[{"id","backend","name","total_mb","free_mb"}]`` (empty for "(none)")."""
    devices = []
    for line in (text or "").splitlines():
        m = _DEVICE.match(line)
        if not m:
            continue
        dev_id = m.group(1)
        devices.append({"id": dev_id, "backend": backend_of(dev_id), "name": m.group(2), "total_mb": int(m.group(3)),
                        "free_mb": int(m.group(4))})
    return devices


def _join(*parts: str) -> str:
    return "\n".join(p.strip() for p in parts if p and p.strip())[:2000]


def _compiled_backends(exe: Path) -> list[str]:
    found = []
    try:
        names = [p.name.lower() for p in exe.parent.iterdir()]
    except OSError:
        return found
    for needle, backend in _LIB_BACKENDS:
        if any(needle in n for n in names) and backend not in found:
            found.append(backend)
    return found


# ------------------------------------------------------------------------------------------------- arch scan
_scan_cache: dict = {}
_scan_lock = threading.Lock()


def _candidate_files(exe: Path) -> list[Path]:
    """The exe and sibling libraries that can contain llama.cpp's architecture table (name contains 'llama')."""
    files, seen = [], set()
    cands = [exe]
    try:
        cands += sorted(p for p in exe.parent.iterdir() if p.is_file() and p != exe and "llama" in p.name.lower()
                        and (p.suffix.lower() in (".dll", ".so", ".dylib") or ".so." in p.name.lower()))
    except OSError:
        pass
    for p in cands:
        try:
            real = os.path.realpath(p)
            if real in seen or not os.path.isfile(real) or os.path.getsize(real) > _MAX_SCAN_BYTES:
                continue
        except OSError:
            continue
        seen.add(real)
        files.append(Path(real))
    return files


def _scan_file(path: Path, needles: dict[str, bytes], found: set[str]) -> bool:
    """Search one file in 4 MB chunks (with overlap so a name split by a boundary is still seen). False if unreadable."""
    overlap = max((len(n) for n in needles.values()), default=1)
    tail = b""
    try:
        with open(path, "rb") as f:
            while len(found) < len(needles):
                chunk = f.read(_CHUNK)
                if not chunk:
                    break
                data = tail + chunk
                for name, needle in needles.items():
                    if name not in found and needle in data:
                        found.add(name)
                tail = data[-overlap:]
    except OSError:
        return False
    return True


def scan_arch_support(exe_path: str | Path, names: tuple[str, ...] | list[str] = DEFAULT_ARCHS) -> dict:
    """``{name: True|False|None}``: is the architecture name present in the engine binaries? None = nothing readable.

    Looks in the executable and sibling ``*llama*`` libraries (``llama.dll``, ``libllama.so``). Results are cached by
    (path, mtime, size) of every scanned file, so repeated calls are free until the engine is replaced.
    """
    exe = Path(exe_path)
    files = _candidate_files(exe)
    if not files:
        return {n: None for n in names}
    try:
        key = tuple((str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in files)
    except OSError:
        return {n: None for n in names}
    with _scan_lock:
        hit = _scan_cache.get(key)
        if hit is not None:
            known = hit
        else:
            known = {}
            if len(_scan_cache) > 16:
                _scan_cache.clear()
            _scan_cache[key] = known
    todo = {n: n.encode() + b"\x00" for n in names if n not in known}
    if todo:
        found: set[str] = set()
        readable = [_scan_file(f, todo, found) for f in files]
        if not any(readable):
            return {n: None for n in names}
        with _scan_lock:
            for n in todo:
                known[n] = n in found
    return {n: known.get(n) for n in names}


# -------------------------------------------------------------------------------------------------- probing
def _run(runner, cmd: list[str], timeout: float, cwd: str) -> tuple[str, str, int | None, str]:
    """(stdout, stderr, returncode, error)."""
    env = dict(os.environ)
    if not _IS_WIN:  # shared libraries next to the binary (cmake builds without an install step)
        env["LD_LIBRARY_PATH"] = cwd + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    kwargs = {"capture_output": True, "text": True, "timeout": timeout, "cwd": cwd, "env": env, "errors": "replace"}
    if _IS_WIN:
        kwargs["creationflags"] = _NO_WINDOW
    try:
        r = runner(cmd, **kwargs)
    except subprocess.TimeoutExpired:
        return "", "", None, f"{Path(cmd[0]).name} {' '.join(cmd[1:])} timed out after {timeout:g} s"
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return "", "", None, f"could not run {Path(cmd[0]).name} {' '.join(cmd[1:])}: {e}"
    out, err = getattr(r, "stdout", "") or "", getattr(r, "stderr", "") or ""
    if isinstance(out, bytes):
        out = out.decode("utf-8", "replace")
    if isinstance(err, bytes):
        err = err.decode("utf-8", "replace")
    return out, err, getattr(r, "returncode", None), ""


def probe_engine(exe_path: str | Path, runner=subprocess.run, timeout: float = 20, archs=DEFAULT_ARCHS) -> dict:
    """Capabilities of a llama-server binary. Never raises; problems are listed in ``errors``.

    ``status``: ``missing`` (no such file), ``ok`` (version parsed and devices listed), ``partial`` (one of the two
    commands failed) or ``error`` (neither worked). ``backends`` are the backends compiled into the build (from the
    exe and sibling libraries plus any listed device); ``backends_usable`` are those with at least one device the
    engine can really use. ``supports`` maps architecture name -> True/False/None (see `scan_arch_support`).
    """
    exe = Path(str(exe_path)) if exe_path else Path("")
    res = {"path": str(exe_path or ""), "exists": False, "version": None, "build": None, "commit": None, "backends": [],
           "backends_usable": [], "devices": [], "supports": {n: None for n in archs}, "raw_version": "", "raw_devices": "",
           "compiler": None, "target": None, "errors": [], "status": "missing", "probed_at": time.time()}
    try:
        res["exists"] = bool(exe_path) and exe.is_file()
    except OSError:
        res["exists"] = False
    if not res["exists"]:
        res["errors"].append(f"engine not found: {exe_path or '(no path given)'}")
        return res
    cwd = str(exe.resolve().parent)
    vout, verr, vcode, vexc = _run(runner, [str(exe), "--version"], timeout, cwd)
    dout, derr, dcode, dexc = _run(runner, [str(exe), "--list-devices"], timeout, cwd)
    res["raw_version"], res["raw_devices"] = _join(vout, verr), _join(dout, derr)
    ver = parse_version(vout + "\n" + verr)
    res.update({k: ver[k] for k in ("version", "build", "commit", "compiler", "target")})
    if vexc:
        res["errors"].append(vexc)
    elif ver["version"] is None:
        res["errors"].append("--version printed nothing recognisable" + (f" (exit code {vcode})" if vcode else ""))
    if dexc:
        res["errors"].append(dexc)
    else:
        res["devices"] = parse_devices(dout + "\n" + derr)
        if "available devices" not in (dout + derr).lower() and not res["devices"]:
            res["errors"].append("--list-devices output not recognised" + (f" (exit code {dcode})" if dcode else ""))
    usable = []
    for d in res["devices"]:
        if d["backend"] not in usable and d["backend"] not in ("CPU", "BLAS"):
            usable.append(d["backend"])
    compiled = _compiled_backends(exe)
    res["backends_usable"] = usable
    res["backends"] = ["CPU"] + [b for b in dict.fromkeys(compiled + usable) if b != "CPU"]
    res["supports"] = scan_arch_support(exe, archs)
    version_ok, devices_ok = ver["version"] is not None, not dexc and bool(res["devices"] or "available devices" in (dout + derr).lower())
    res["status"] = "ok" if version_ok and devices_ok else "partial" if version_ok or devices_ok else "error"
    return res


_probe_cache: dict = {}


def probe_engine_cached(exe_path: str | Path, ttl: float = 300.0, **kw) -> dict:
    """`probe_engine` with a cache keyed by (path, mtime, size); entries expire after `ttl` seconds."""
    try:
        st = Path(str(exe_path)).stat()
        key = (str(exe_path), st.st_mtime_ns, st.st_size)
    except OSError:
        return probe_engine(exe_path, **kw)
    with _scan_lock:
        hit = _probe_cache.get(key)
        if hit and time.time() - hit["probed_at"] < ttl:
            return hit
    res = probe_engine(exe_path, **kw)
    with _scan_lock:
        if len(_probe_cache) > 8:
            _probe_cache.clear()
        _probe_cache[key] = res
    return res
