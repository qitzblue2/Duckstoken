"""Locating files that ship inside the app (PyInstaller bundle) or next to the source checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Folder holding bundled data: PyInstaller's extraction dir, or the repository root in development."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def app_dir() -> Path:
    """Folder containing the running exe (or the repository root in development)."""
    return Path(sys.executable).resolve().parent if frozen() else Path(__file__).resolve().parent.parent


def bundled_engine_dir() -> Path | None:
    """The llama.cpp build shipped inside the app, if any."""
    for base in (resource_root(), app_dir()):
        d = base / "engine"
        if d.is_dir() and any(d.glob("llama-server*")):
            return d
    return None


def server_exe_name() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def bundled_llama_server() -> Path | None:
    d = bundled_engine_dir()
    if d is None:
        return None
    p = d / server_exe_name()
    return p if p.is_file() else None
