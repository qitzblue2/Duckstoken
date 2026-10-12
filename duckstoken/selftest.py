"""`Duckstoken.exe --selftest <file>`: proves the packaged app contains what it needs, without opening a window.

CI runs it against the frozen portable exe and the installed app. The report is written to the file (a windowed exe has
no console) and the exit code says whether every *required* part of a frozen build is present. From source the same checks
run, but missing optional packages are only reported.
"""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
import subprocess
import sys

from . import __version__, bundle


def _web_assets() -> str:
    index = bundle.resource_root() / "duckstoken" / "web" / "index.html"
    if not index.is_file():
        raise FileNotFoundError(f"web UI missing: {index}")
    return str(index)


def _webview() -> str:
    wv = importlib.import_module("webview")
    if not hasattr(wv, "create_window"):
        raise ImportError("the 'webview' module was imported but has no create_window (only its data files were packaged)")
    return f"pywebview {getattr(wv, '__version__', '?')}"


def _webview_backend() -> str:
    if os.name != "nt":
        return "not applicable on this OS"
    errors = []
    for name in ("webview.platforms.edgechromium", "webview.platforms.winforms"):
        try:
            importlib.import_module(name)
            return f"{name} imports (needs pythonnet/clr)"
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e.__class__.__name__}: {e}")
    raise ImportError("; ".join(errors))


def _tray() -> str:
    importlib.import_module("PIL.Image")
    pystray = importlib.import_module("pystray")
    if not hasattr(pystray, "Icon"):
        raise ImportError("pystray has no Icon")
    return "pystray + Pillow"


def _numpy() -> str:
    return "numpy " + importlib.import_module("numpy").__version__


def _engine() -> str:
    exe = bundle.bundled_llama_server()
    if exe is None:
        raise FileNotFoundError("the bundled llama-server is missing")
    flags = 0x08000000 if os.name == "nt" else 0
    r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60, creationflags=flags, cwd=str(exe.parent))
    text = (r.stdout + r.stderr).strip().splitlines()
    if r.returncode != 0:
        raise RuntimeError(f"llama-server --version exited with {r.returncode}: {' '.join(text[-2:])}")
    return text[0] if text else "llama-server ran"


def _fts5() -> str:
    c = sqlite3.connect(":memory:")
    try:
        c.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    finally:
        c.close()
    return "sqlite FTS5 available"


# (name, function, required in a frozen build)
CHECKS = [
    ("web ui files", _web_assets, True),
    ("pywebview module", _webview, True),
    ("pywebview windows backend", _webview_backend, False),
    ("tray (pystray + Pillow)", _tray, os.name == "nt"),   # pystray needs a display server: not available on headless Linux
    ("numpy (built-in engine)", _numpy, True),
    ("bundled llama.cpp engine", _engine, True),
    ("sqlite FTS5 (memory search)", _fts5, False),
]


def run(out_path: str | None = None) -> int:
    frozen = bundle.frozen()
    results, failed = [], False
    for name, fn, required in CHECKS:
        try:
            detail, ok = fn(), True
        except Exception as e:  # noqa: BLE001 - report every failure, do not stop at the first
            detail, ok = f"{e.__class__.__name__}: {e}", False
        fatal = (not ok) and required and frozen
        failed = failed or fatal
        results.append({"check": name, "ok": ok, "required": required, "detail": detail})
    report = {"version": __version__, "frozen": frozen, "python": sys.version.split()[0], "ok": not failed, "checks": results}
    text = json.dumps(report, indent=2)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    if sys.stdout is not None:
        try:
            print(text)
        except Exception:  # noqa: BLE001
            pass
    return 0 if not failed else 1
