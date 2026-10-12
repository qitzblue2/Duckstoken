"""Bounded, crash-tolerant persistence for VirtualGPU: benchmark records, optimizer history and learned profiles.

Layout inside one directory::

    benchmarks.jsonl          one benchmark record per line (newest last)
    optimizer_history.jsonl   one optimizer-run summary per line
    profiles.json             {profile_key: profile}  (best known config per hardware + model + engine)

Every write goes to a temp file first and is moved over the real file with ``os.replace``, so a crash or a full disk
never leaves a half-written file behind.  Files are capped (`max_records` lines, oldest dropped) and a corrupted
line is skipped on read instead of failing the whole store.

Profile keys are ``<model>-<hardware>-<engine>``, three short hashes joined by dashes.  Because the parts are
separate, `profile_status` can tell "never tuned" (``missing``) from "tuned, but on other hardware / another engine
build / a re-downloaded model file" (``stale``) from a profile that still applies (``fresh``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

MAX_RECORD_BYTES = 256 * 1024     # one serialized record; bigger ones lose their bulky optional fields first
_BULKY_KEYS = ("log_tail", "samples", "raw")
_KEY_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}$")


class StoreError(RuntimeError):
    """The store could not read or write its files (permissions, full disk, ...)."""


# ------------------------------------------------------------------------------------------------ fingerprints
def _digest(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def hardware_fingerprint(hw: dict) -> str:
    """Stable hash of CPU name + cores, RAM size (rounded to whole GB) and GPU names + VRAM."""
    hw = hw or {}
    cpu, ram = hw.get("cpu") or {}, hw.get("ram") or {}
    gpus = sorted((str(g.get("name", "")).strip(), int(g.get("vram_mb") or 0)) for g in hw.get("gpus") or [])
    return _digest(str(cpu.get("name", "")).strip(), int(cpu.get("physical_cores") or 0),
                   int(cpu.get("logical_cores") or 0), int(round(float(ram.get("total_gb") or 0))), gpus)


def model_fingerprint(model: dict) -> str:
    """Stable hash of what makes a model file what it is: name, size and quantization type (not its path)."""
    model = model or {}
    return _digest(str(model.get("name", "")), int(model.get("size_bytes") or 0), model.get("file_type"))


def backend_fingerprint(backend_version) -> str:
    """Stable hash of the engine build (a version string or build number; None when unknown)."""
    return _digest(None if backend_version is None else str(backend_version))


def profile_key(hw: dict, model_summary: dict, backend_version) -> str:
    return "-".join((model_fingerprint(model_summary), hardware_fingerprint(hw), backend_fingerprint(backend_version)))


def _split_key(key: str) -> tuple[str, str, str] | None:
    if not isinstance(key, str) or not _KEY_RE.match(key):
        return None
    a, b, c = key.split("-")
    return a, b, c


# ------------------------------------------------------------------------------------------------ file helpers
def atomic_write_text(path: Path, text: str, fsync: bool = True) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        for attempt in range(5):  # Windows: a virus scanner or indexer may hold the target for a moment
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except OSError as e:
        raise StoreError(f"cannot write {path.name}: {e}") from e
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _dump(rec: dict) -> str:
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=str)


def _shrink(rec: dict) -> str:
    """Serialize `rec`, dropping bulky optional fields if the record is over `MAX_RECORD_BYTES`."""
    line = _dump(rec)
    if len(line.encode("utf-8")) <= MAX_RECORD_BYTES:
        return line
    slim = {k: v for k, v in rec.items() if k not in _BULKY_KEYS}
    slim["truncated"] = True
    line = _dump(slim)
    if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError("record is too large to store")
    return line


def _new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class VGpuStore:
    def __init__(self, dir, max_records: int = 500, fsync: bool = True):
        self.dir = Path(dir)
        self.max_records = max(1, int(max_records))
        self.fsync = bool(fsync)  # tests turn it off: fsync is slow on busy disks, the atomic rename stays
        self._lock = threading.RLock()
        self.corrupt_lines = 0  # corrupted lines seen by the latest read of each file (informational)

    # ---------------------------------------------------------------- paths
    @property
    def benchmarks_path(self) -> Path:
        return self.dir / "benchmarks.jsonl"

    @property
    def history_path(self) -> Path:
        return self.dir / "optimizer_history.jsonl"

    @property
    def profiles_path(self) -> Path:
        return self.dir / "profiles.json"

    def _ensure_dir(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise StoreError(f"cannot create {self.dir}: {e}") from e

    # ---------------------------------------------------------------- jsonl
    def _read_lines(self, path: Path) -> list[dict]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return []
        except OSError as e:
            raise StoreError(f"cannot read {path.name}: {e}") from e
        out, bad = [], 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(rec, dict):
                out.append(rec)
            else:
                bad += 1
        self.corrupt_lines = bad
        return out

    def _append(self, path: Path, rec: dict, prefix: str) -> dict:
        rec = dict(rec)
        rec.setdefault("id", _new_id(prefix))
        rec.setdefault("created", _now_iso())
        line = _shrink(rec)
        with self._lock:
            self._ensure_dir()
            records = self._read_lines(path)
            records = [r for r in records if r.get("id") != rec["id"]]  # re-adding an id replaces it
            lines = [_dump(r) for r in records] + [line]
            lines = lines[-self.max_records:]
            atomic_write_text(path, "\n".join(lines) + "\n", self.fsync)
        return json.loads(line)

    # ---------------------------------------------------------------- benchmarks
    def add_benchmark(self, rec: dict) -> dict:
        """Store a benchmark record (an `id` and `created` stamp are added when missing); returns the stored copy."""
        return self._append(self.benchmarks_path, rec, "bench")

    def benchmarks(self, limit: int = 50, model: str | None = None, config_hash: str | None = None) -> list[dict]:
        """Newest first.  `model` matches the model name or path; `config_hash` the benchmark's config hash."""
        with self._lock:
            records = self._read_lines(self.benchmarks_path)
        out = []
        if int(limit) <= 0:
            return out
        for r in reversed(records):
            if model is not None:
                m = r.get("model") or {}
                if model not in (m.get("name"), m.get("path")):
                    continue
            if config_hash is not None and r.get("config_hash") != config_hash:
                continue
            out.append(r)
            if len(out) >= int(limit):
                break
        return out

    def get_benchmark(self, bench_id: str) -> dict | None:
        with self._lock:
            for r in self._read_lines(self.benchmarks_path):
                if r.get("id") == bench_id:
                    return r
        return None

    def delete_benchmark(self, bench_id: str) -> bool:
        with self._lock:
            records = self._read_lines(self.benchmarks_path)
            kept = [r for r in records if r.get("id") != bench_id]
            if len(kept) == len(records):
                return False
            atomic_write_text(self.benchmarks_path, "".join(_dump(r) + "\n" for r in kept), self.fsync)
        return True

    # ---------------------------------------------------------------- optimizer history
    def add_history(self, rec: dict) -> dict:
        return self._append(self.history_path, rec, "opt")

    def history(self, limit: int = 50) -> list[dict]:
        with self._lock:
            records = self._read_lines(self.history_path)
        return list(reversed(records))[:max(0, int(limit))]

    # ---------------------------------------------------------------- profiles
    def _load_profiles(self) -> dict:
        try:
            data = json.loads(self.profiles_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            return {}  # unreadable / corrupted: behave as "nothing learned yet"; the next save rewrites it
        return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}

    def _save_profiles(self, profiles: dict) -> None:
        if len(profiles) > self.max_records:
            newest = sorted(profiles.items(), key=lambda kv: str(kv[1].get("saved", "")))[-self.max_records:]
            profiles = dict(newest)
        self._ensure_dir()
        atomic_write_text(self.profiles_path, json.dumps(profiles, ensure_ascii=False, indent=1, default=str), self.fsync)

    def save_profile(self, key: str, profile: dict) -> dict:
        prof = dict(profile)
        prof["key"] = key
        prof["saved"] = _now_iso()
        with self._lock:
            profiles = self._load_profiles()
            profiles[key] = prof
            self._save_profiles(profiles)
        return prof

    def get_profile(self, key: str) -> dict | None:
        with self._lock:
            return self._load_profiles().get(key)

    def profiles(self) -> dict:
        with self._lock:
            return self._load_profiles()

    def delete_profile(self, key: str) -> bool:
        with self._lock:
            profiles = self._load_profiles()
            if key not in profiles:
                return False
            del profiles[key]
            self._save_profiles(profiles)
        return True

    def profile_status(self, key: str) -> str:
        """``fresh``: a profile for exactly this hardware + model + engine.  ``stale``: this model was tuned before but
        on other hardware or another engine build.  ``missing``: nothing learned for this model."""
        profiles = self.profiles()
        if key in profiles:
            return "fresh"
        parts = _split_key(key)
        if parts and any((_split_key(k) or ("",))[0] == parts[0] for k in profiles):
            return "stale"
        return "missing"

    def stale_profiles(self, key: str) -> list[dict]:
        """Profiles for the same model as `key` whose hardware or engine fingerprint differs."""
        parts = _split_key(key)
        if not parts:
            return []
        return [p for k, p in self.profiles().items() if k != key and (_split_key(k) or ("",))[0] == parts[0]]

    def stats(self) -> dict:
        with self._lock:
            return {"dir": str(self.dir), "max_records": self.max_records,
                    "benchmarks": len(self._read_lines(self.benchmarks_path)),
                    "history": len(self._read_lines(self.history_path)),
                    "profiles": len(self._load_profiles())}
