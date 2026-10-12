"""Where Duckstoken keeps its files."""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Settings, logs, llama.cpp builds. Override with DUCKSTOKEN_HOME."""
    override = os.environ.get("DUCKSTOKEN_HOME")
    if override:
        p = Path(override)
    elif os.name == "nt":
        p = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Duckstoken"
    else:
        p = Path.home() / ".duckstoken"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_models_dir() -> Path:
    return Path.home() / "Duckstoken" / "models"


def default_llama_dir() -> Path:
    return data_dir() / "engines"


def settings_file() -> Path:
    return data_dir() / "settings.json"


def profiles_file() -> Path:
    return data_dir() / "profiles.json"


def audit_file() -> Path:
    return data_dir() / "tool_audit.jsonl"


def server_log_file() -> Path:
    return data_dir() / "llama-server.log"


def learning_db_file() -> Path:
    return data_dir() / "learning.db"


def big_server_log_file() -> Path:
    return data_dir() / "llama-server-big.log"


def vgpu_store_dir() -> Path:
    """Benchmarks, optimizer history and learned VirtualGPU profiles."""
    p = data_dir() / "vgpu"
    p.mkdir(parents=True, exist_ok=True)
    return p


def staging_dir() -> Path:
    """SSD staging cache for models that live on a slow drive (see vgpu/diskcache.py)."""
    p = data_dir() / "staging"
    p.mkdir(parents=True, exist_ok=True)
    return p


def vgpu_dir() -> Path:
    """VirtualGPU working files that are not benchmark records (performance log, probe cache)."""
    p = data_dir() / "vgpu"
    p.mkdir(parents=True, exist_ok=True)
    return p


def perf_log_file() -> Path:
    return vgpu_dir() / "perf.jsonl"
