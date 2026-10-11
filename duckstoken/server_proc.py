"""Starts, watches and stops llama-server."""

from __future__ import annotations

import collections
import json
import os
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import paths
from .settings import SCHEMA

_PRIORITY = {"idle": 0x40, "below_normal": 0x4000, "normal": 0x20, "above_normal": 0x8000, "high": 0x80}


_JOB = None


def _job_object():
    """Windows job that kills llama-server if Duckstoken dies (e.g. its console window is closed)."""
    global _JOB
    if _JOB is not None or os.name != "nt":
        return _JOB
    import ctypes
    from ctypes import wintypes

    k = ctypes.windll.kernel32
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class Extended(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("Io", IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k.CreateJobObjectW(None, None)
    info = Extended()
    info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not job or not k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        return None
    _JOB = (job, k)
    return _JOB


def _bind_lifetime(proc: subprocess.Popen) -> None:
    try:
        j = _job_object()
        if j:
            j[1].AssignProcessToJobObject(j[0], int(proc._handle))  # noqa: SLF001
    except Exception:  # noqa: BLE001 - best effort
        pass


def connect_host(host: str) -> str:
    return "127.0.0.1" if host in ("0.0.0.0", "", "::") else host


def split_args(text: str) -> list[str]:
    """Split a command-line string. Quotes group words but are removed; backslashes stay literal (Windows paths)."""
    out = []
    for tok in shlex.split(text, posix=False):
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
            tok = tok[1:-1]
        out.append(tok)
    return out


def build_args(s: dict) -> list[str]:
    """Settings -> llama-server argv (without the executable)."""
    args: list[str] = []
    for spec in SCHEMA:
        flag = spec.get("flag")
        if not flag:
            continue
        val = s.get(spec["key"], spec["default"])
        if spec["type"] == "bool":
            if bool(val) == spec.get("when", True):
                args += flag.split()
            continue
        if val in ("", None) or val in spec.get("skip_if", []):
            continue
        if spec["key"] == "ctx_size" and val == 0:
            args += [flag, "0"]
            continue
        args += [flag, str(val)]
    if s.get("lora_path"):
        scale = float(s.get("lora_scale", 1.0))
        args += ["--lora", s["lora_path"]] if scale == 1.0 else ["--lora-scaled", s["lora_path"], str(scale)]
    extra = (s.get("extra_args") or "").strip()
    if extra:
        args += split_args(extra)
    return args


class ServerManager:
    def __init__(self, log_file: Path | None = None):
        self.log_file = log_file or paths.server_log_file()
        self.proc: subprocess.Popen | None = None
        self.state = "stopped"  # stopped starting loading ready error
        self.error = ""
        self.started_at = 0.0
        self.host, self.port, self.api_key = "127.0.0.1", 8080, ""
        self.cmdline: list[str] = []
        self._log: collections.deque[str] = collections.deque(maxlen=5000)
        self._log_base = 0  # index of first retained line
        self._lock = threading.RLock()

    # logging ------------------------------------------------------------
    def _append(self, line: str) -> None:
        with self._lock:
            if len(self._log) == self._log.maxlen:
                self._log_base += 1
            self._log.append(line)
        try:
            with open(self.log_file, "a", encoding="utf-8", errors="replace") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def logs(self, since: int = 0) -> dict:
        with self._lock:
            start = max(since - self._log_base, 0)
            lines = list(self._log)[start:]
            return {"lines": lines, "next": self._log_base + len(self._log)}

    # lifecycle ----------------------------------------------------------
    def start(self, s: dict) -> None:
        with self._lock:
            if self.proc and self.proc.poll() is None:
                raise RuntimeError("The model server is already running")
            exe, model = s.get("llama_server_path", ""), s.get("model_path", "")
            if not exe or not Path(exe).is_file():
                raise RuntimeError("llama-server is not set up yet. Open Models -> Engine and install it.")
            if not model or not Path(model).is_file():
                raise RuntimeError("No model selected. Pick a .gguf file in the Models tab.")
            self.host = connect_host(s["host"])
            self.port, self.api_key = int(s["port"]), s.get("api_key", "")
            self.cmdline = [exe] + build_args(s)
            flags = 0
            if os.name == "nt":
                flags = 0x08000000 | _PRIORITY.get(s.get("process_priority", "normal"), 0x20)
            self._append(f"$ {subprocess.list2cmdline(self.cmdline)}")
            try:
                self.log_file.write_text("", encoding="utf-8")
            except OSError:
                pass
            self.error = ""
            self.state = "starting"
            self.started_at = time.time()
            try:
                self.proc = subprocess.Popen(self.cmdline, cwd=str(Path(exe).parent), stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, errors="replace",
                                             creationflags=flags)
            except OSError as e:
                self.state, self.error = "error", str(e)
                raise RuntimeError(f"Could not launch llama-server: {e}") from e
            proc = self.proc
            _bind_lifetime(proc)
        threading.Thread(target=self._pump, args=(proc,), daemon=True, name="llama-log").start()
        threading.Thread(target=self._watch_ready, args=(proc,), daemon=True, name="llama-ready").start()

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout
        for line in proc.stdout:
            self._append(line.rstrip())
        code = proc.wait()
        proc.stdout.close()
        with self._lock:
            if self.proc is proc:
                if self.state != "stopped":
                    self.state = "error" if code not in (0, None) else "stopped"
                    if self.state == "error":
                        self.error = f"llama-server exited with code {code}. See the log below."
                self._append(f"[duckstoken] llama-server exited (code {code})")

    def _watch_ready(self, proc: subprocess.Popen) -> None:
        url = f"http://{self.host}:{self.port}/health"
        while proc.poll() is None and self.proc is proc:
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        with self._lock:
                            if self.proc is proc:
                                self.state = "ready"
                        return
            except urllib.error.HTTPError as e:
                if e.code == 503 and self.state == "starting":
                    self.state = "loading"
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1.0)

    def stop(self) -> None:
        with self._lock:
            proc, self.proc = self.proc, None
            self.state = "stopped"
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()

    def status(self) -> dict:
        with self._lock:
            running = bool(self.proc and self.proc.poll() is None)
            if not running and self.state in ("starting", "loading", "ready"):
                self.state = "stopped"
            return {"state": self.state, "error": self.error, "running": running,
                    "uptime": int(time.time() - self.started_at) if running else 0,
                    "url": f"http://{self.host}:{self.port}", "cmdline": subprocess.list2cmdline(self.cmdline)}

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def upstream_headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h
