"""The optional SSD staging tier: copy a model from a slow drive to a fast one so llama.cpp's mmap page faults are served
by the fast drive (see `diskcache.py` for the module overview; `StagingCache` is re-exported from there).

It never starts on its own (the caller must pass ``confirm=True``), copies through ``<file>.part`` + atomic rename,
removes partial files when it fails or is cancelled, and only ever deletes files it created itself: everything it owns is
listed in ``manifest.json`` and deletion is limited to those paths inside its own directory.  Originals are never
modified or removed.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from .. import gguf, hardware
from .store import StoreError, atomic_write_text
from .values import measured, unavailable

GIB = 2**30
MIB = 2**20
_DISK_FULL = (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC))   # "no space left" / quota exceeded


# -------------------------------------------------------------------------------------------------- StagingCache
def _fail(code: str, error: str, **extra) -> dict:
    return {"ok": False, "code": code, "error": error, **extra}


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")[:40] or "model"


class StagingCache:
    MANIFEST = "manifest.json"

    def __init__(self, dir, budget_gb: float, min_free_gb: float = 10.0, *,
                 free_fn: Callable[[], int] | None = None, chunk_bytes: int = 8 * MIB):
        self.dir = Path(dir)
        self.budget_bytes = max(0, int(float(budget_gb) * GIB))
        self.min_free_bytes = max(0, int(float(min_free_gb) * GIB))
        self._free_fn = free_fn
        self.chunk_bytes = max(4096, int(chunk_bytes))
        self._lock = threading.RLock()          # manifest access
        self._stage_lock = threading.Lock()     # one copy at a time

    # ------------------------------------------------------------------ manifest
    @property
    def manifest_path(self) -> Path:
        return self.dir / self.MANIFEST

    def _load(self) -> dict:
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        entries = data.get("entries") if isinstance(data.get("entries"), dict) else {}
        pending = data.get("pending") if isinstance(data.get("pending"), list) else []
        return {"version": 1, "entries": {k: v for k, v in entries.items() if isinstance(v, dict)},
                "pending": [p for p in pending if isinstance(p, str)]}

    def _save(self, manifest: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.manifest_path, json.dumps(manifest, indent=1), fsync=False)

    def _owned_path(self, rel: str) -> Path | None:
        """`rel` ('<entry>/<file>') as an absolute path, only if it is a plain two-part name inside our directory."""
        parts = Path(rel.replace("\\", "/")).parts
        if len(parts) != 2 or any(p in ("..", ".", "") or "/" in p or "\\" in p or ":" in p for p in parts):
            return None
        target = (self.dir / parts[0] / parts[1])
        try:
            if not target.resolve().is_relative_to(self.dir.resolve()):
                return None
        except (OSError, ValueError):
            return None
        return target

    def _entry_files(self, name: str, entry: dict) -> list[tuple[Path, int]]:
        out = []
        for f in entry.get("files") or []:
            if isinstance(f, dict) and isinstance(f.get("name"), str):
                p = self._owned_path(f"{name}/{f['name']}")
                if p is not None:
                    out.append((p, int(f.get("size") or 0)))
        return out

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _norm(src) -> str:
        return os.path.normcase(os.path.realpath(str(src)))

    def _inside_cache(self, path) -> bool:
        try:
            return Path(os.path.realpath(str(path))).is_relative_to(Path(os.path.realpath(self.dir)))
        except (OSError, ValueError):
            return False

    def _entry_name(self, src: str) -> str:
        return f"{_safe_name(Path(src).stem)}-{hashlib.sha1(src.encode('utf-8', 'replace')).hexdigest()[:8]}"

    def _free_bytes(self) -> int | None:
        if self._free_fn:
            try:
                return int(self._free_fn())
            except Exception:  # noqa: BLE001
                return None
        probe = self.dir
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            return shutil.disk_usage(probe).free
        except OSError:
            return None

    @staticmethod
    def _disk_for(path: str, hw: dict) -> dict | None:
        """The drive entry of `hw["disks"]` that contains `path`: the longest matching drive letter or mount point."""
        disks = (hw or {}).get("disks") or []
        norm, best, best_len = os.path.normcase(os.path.realpath(path)), None, -1
        for d in disks:
            mount = str(d.get("drive") or "")
            if not mount:
                continue
            m = os.path.normcase(mount)
            if len(m) == 2 and m[1] == ":":                 # a bare Windows drive letter such as "D:"
                hit = norm.startswith(m)
            else:
                m = os.path.normcase(os.path.realpath(mount)).rstrip("\\/") or os.sep
                hit = m == os.sep or norm == m or norm.startswith(m + os.sep)
            if hit and len(m) > best_len:
                best, best_len = d, len(m)
        return best

    def _current(self, name: str, entry: dict) -> bool:
        """The staged files exist with the recorded sizes and the original is unchanged."""
        try:
            st = os.stat(entry["src"])
        except (OSError, KeyError):
            return False
        if st.st_size != entry.get("src_size") or st.st_mtime_ns != entry.get("src_mtime_ns"):
            return False
        files = self._entry_files(name, entry)
        if not files:
            return False
        for p, size in files:
            try:
                if p.stat().st_size != size:
                    return False
            except OSError:
                return False
        return True

    def _source_files(self, src: str) -> list[Path]:
        p = Path(src)
        return gguf.shard_files(p) if p.is_file() else [p]

    def _lru_plan(self, manifest: dict, needed: int, keep: str | None = None) -> list[str] | None:
        """Names to evict (least recently used first) so that `needed` more bytes fit the budget; None if impossible."""
        if needed > self.budget_bytes:
            return None
        used = sum(int(e.get("size") or 0) for e in manifest["entries"].values())
        victims: list[str] = []
        for name, e in sorted(manifest["entries"].items(), key=lambda kv: kv[1].get("last_used") or 0):
            if used + needed <= self.budget_bytes:
                break
            if name == keep:
                continue
            victims.append(name)
            used -= int(e.get("size") or 0)
        return victims if used + needed <= self.budget_bytes else None

    # ------------------------------------------------------------------ queries
    def lookup(self, src, touch: bool = True) -> str | None:
        """Path of the staged copy of `src` if it is still current (and mark it used), else None."""
        key = self._norm(src)
        name = self._entry_name(key)
        with self._lock:
            manifest = self._load()
            entry = manifest["entries"].get(name)
            if not entry or entry.get("src") != key or not self._current(name, entry):
                return None
            if touch:
                entry["last_used"] = time.time()
                try:
                    self._save(manifest)
                except (OSError, StoreError):
                    pass
            files = self._entry_files(name, entry)
            return str(files[0][0]) if files else None

    def list(self) -> list[dict]:
        with self._lock:
            manifest = self._load()
            out = []
            for name, e in manifest["entries"].items():
                files = self._entry_files(name, e)
                out.append({"name": name, "src": e.get("src"), "size": int(e.get("size") or 0),
                            "staged_at": e.get("staged_at"), "last_used": e.get("last_used"),
                            "path": str(files[0][0]) if files else None, "current": self._current(name, e)})
            return sorted(out, key=lambda r: r.get("last_used") or 0, reverse=True)

    def usage(self) -> dict:
        with self._lock:
            manifest = self._load()
            used = sum(int(e.get("size") or 0) for e in manifest["entries"].values())
            known = set(manifest["entries"]) | {self.MANIFEST} | {r.split("/")[0] for r in manifest["pending"]}
            try:
                unmanaged = sorted(p.name for p in self.dir.iterdir() if p.name not in known and not p.name.endswith(".tmp"))
            except OSError:
                unmanaged = []
        free = self._free_bytes()
        return {"dir": str(self.dir), "budget_bytes": self.budget_bytes, "budget_gb": round(self.budget_bytes / GIB, 2),
                "used_bytes": used, "free_budget_bytes": max(0, self.budget_bytes - used),
                "entries": len(manifest["entries"]), "pending": list(manifest["pending"]),
                "unmanaged": unmanaged,   # found in our directory but not created by us: reported, never deleted
                "disk_free_gb": (measured(round(free / GIB, 2), "shutil.disk_usage") if free is not None
                                 else unavailable("cannot read free space", "shutil.disk_usage"))}

    def plan_stage(self, src, hw: dict) -> dict:
        """Is staging `src` worthwhile?  Nothing is copied.  Always explains why not."""
        base = {"worthwhile": False, "reason": "", "src": str(src), "size_bytes": 0, "size_gb": 0.0,
                "src_kind": "unknown", "cache_kind": "unknown", "needs_gb": 0.0, "would_evict": [],
                "already_staged": False, "staged_path": None}
        p = Path(str(src))
        if not p.is_file():
            return {**base, "reason": f"The model file does not exist: {src}"}
        files = self._source_files(str(p))
        if not all(f.is_file() for f in files):
            return {**base, "reason": "A shard of the split model is missing."}
        size = sum(f.stat().st_size for f in files)
        base.update(size_bytes=size, size_gb=round(size / GIB, 2))
        staged = self.lookup(p, touch=False)
        if staged:
            return {**base, "already_staged": True, "staged_path": staged,
                    "reason": "Already staged and up to date; llama.cpp can use the copy on the fast drive."}
        src_disk, cache_disk = self._disk_for(str(p), hw), self._disk_for(str(self.dir), hw)
        sk, ck = (src_disk or {}).get("kind", "unknown"), (cache_disk or {}).get("kind", "unknown")
        base.update(src_kind=sk, cache_kind=ck)
        if sk not in ("hdd", "unknown"):
            return {**base, "reason": f"The model is already on a fast drive ({sk.upper()}); copying it gains nothing."}
        if ck not in ("ssd", "nvme"):
            return {**base, "reason": "The cache folder is not on a drive known to be an SSD/NVMe, so a copy would not be faster."}
        if self._inside_cache(p):
            return {**base, "reason": "The model already lives inside the staging folder."}
        with self._lock:
            manifest = self._load()
            stale = manifest["entries"].get(self._entry_name(self._norm(p)))
            victims = self._lru_plan(manifest, size, keep=None)
        if victims is None:
            return {**base, "reason": f"The model ({size / GIB:.1f} GB) is larger than the staging budget "
                                      f"({self.budget_bytes / GIB:.1f} GB). Raise the budget or leave it on its drive."}
        free = self._free_bytes()
        need = size + self.min_free_bytes
        base.update(needs_gb=round(need / GIB, 2), would_evict=victims)
        if free is None:
            return {**base, "reason": "Cannot read the free space of the cache drive."}
        if free < need:
            return {**base, "reason": f"Not enough free space on the cache drive: {free / GIB:.1f} GB free, "
                                      f"{need / GIB:.1f} GB needed ({size / GIB:.1f} GB model + "
                                      f"{self.min_free_bytes / GIB:.0f} GB kept free)."}
        why = "The model is on a slow drive and the cache drive is fast"
        if stale:
            why += "; an outdated staged copy will be replaced"
        return {**base, "worthwhile": True,
                "reason": why + f". Copying {size / GIB:.1f} GB takes about as long as reading it once."}

    # ------------------------------------------------------------------ removal
    def _remove_entry(self, manifest: dict, name: str) -> tuple[bool, int, str]:
        """Delete the files the manifest lists for `name` (nothing else).  Returns (ok, freed_bytes, error)."""
        entry = manifest["entries"].get(name)
        if entry is None:
            return False, 0, f"{name!r} is not a staged copy created by this cache"
        freed = 0
        for p, size in self._entry_files(name, entry):
            try:
                p.unlink()
                freed += size
            except FileNotFoundError:
                pass
            except OSError as e:     # e.g. Windows: the model is still mapped by a running llama-server
                return False, freed, f"cannot delete {p.name}: {e}"
        dirp = self._owned_dir(name)
        if dirp is not None:
            try:
                dirp.rmdir()         # only succeeds when empty: never recursive
            except OSError:
                pass
        del manifest["entries"][name]
        return True, freed, ""

    def _owned_dir(self, name: str) -> Path | None:
        if not name or any(c in name for c in "/\\:") or name in (".", ".."):
            return None
        return self.dir / name

    def evict(self, name: str) -> dict:
        with self._lock:
            manifest = self._load()
            known = name in manifest["entries"]
            ok, freed, error = self._remove_entry(manifest, name)
            if known:
                try:
                    self._save(manifest)
                except (OSError, StoreError) as e:
                    return _fail("io_error", f"could not update the manifest: {e}")
        if ok:
            return {"ok": True, "freed_bytes": freed}
        return _fail("in_use" if known else "not_found", error)

    def _cleanup_partials(self, manifest: dict) -> int:
        """Delete partial copies this cache registered (never other files).  Returns how many were removed."""
        removed = 0
        for rel in list(manifest["pending"]):
            p = self._owned_path(rel)
            if p is not None:
                try:
                    p.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
                except OSError:
                    continue              # keep it registered, try again next time
                try:
                    p.parent.rmdir()
                except OSError:
                    pass
            manifest["pending"].remove(rel)
        return removed

    def cleanup(self) -> dict:
        """Remove leftover partial copies from a crashed or cancelled run."""
        with self._lock:
            manifest = self._load()
            n = self._cleanup_partials(manifest)
            try:
                self._save(manifest)
            except (OSError, StoreError):
                pass
        return {"removed": n}

    # ------------------------------------------------------------------ staging
    def _copy_file(self, src: Path, dst: Path, done: list[int], total: int, progress, cancel) -> str | None:
        """Copy in chunks; returns None on success or a failure code ('cancelled', 'disk_full', 'io_error')."""
        try:
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                while True:
                    if cancel is not None and cancel.is_set():
                        return "cancelled"
                    chunk = fi.read(self.chunk_bytes)
                    if not chunk:
                        break
                    fo.write(chunk)
                    done[0] += len(chunk)
                    if progress:
                        progress(done[0], total, f"Copying {src.name}")
                fo.flush()
                os.fsync(fo.fileno())
        except OSError as e:
            return "disk_full" if e.errno in _DISK_FULL else "io_error"
        return None

    @staticmethod
    def _edges(path: Path, size: int) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(MIB))
            if size > 2 * MIB:
                f.seek(size - MIB)
                h.update(f.read(MIB))
        return h.hexdigest()

    def stage(self, src, confirm: bool = False, progress: Callable | None = None, cancel=None, hw: dict | None = None,
              verify_edges: bool = True) -> dict:
        """Copy `src` (all shards) into the cache.  Never raises for expected failures: returns ``{"ok": False,
        "code", "error"}``.  Refuses unless ``confirm=True``.  With `hw` it also refuses when `plan_stage` says it is
        not worthwhile."""
        if confirm is not True:
            return _fail("confirm_required", "Staging copies the whole model onto the cache drive. "
                         "Nothing was copied; confirm to proceed.")
        if not self._stage_lock.acquire(blocking=False):
            return _fail("busy", "Another staging copy is already running.")
        try:
            return self._stage_locked(src, progress, cancel, hw, verify_edges)
        except StoreError as e:                  # the manifest could not be written (disk full, read-only, locked)
            return _fail("io_error", str(e))
        finally:
            self._stage_lock.release()

    def _stage_locked(self, src, progress, cancel, hw, verify_edges) -> dict:
        copied: list[Path] = []     # final files created by this attempt
        parts: list[Path] = []      # .part files created by this attempt
        p = Path(str(src))
        if not p.is_file():
            return _fail("source_missing", f"The model file does not exist: {src}")
        files = self._source_files(str(p))
        if not all(f.is_file() for f in files):
            return _fail("source_missing", "A shard of the split model is missing.")
        key = self._norm(p)
        if self._inside_cache(p):
            return _fail("source_in_cache", "The model is already inside the staging folder.")
        sizes = [f.stat().st_size for f in files]
        total = sum(sizes)
        before = [(f.stat().st_size, f.stat().st_mtime_ns) for f in files]

        existing = self.lookup(p)
        if existing:
            return {"ok": True, "already_staged": True, "path": existing, "size_bytes": total, "seconds": 0.0}
        if hw is not None:
            plan = self.plan_stage(p, hw)
            if not plan["worthwhile"]:
                return _fail("not_worthwhile", plan["reason"], plan=plan)

        name = self._entry_name(key)
        with self._lock:
            manifest = self._load()
            self._cleanup_partials(manifest)
            if name in manifest["entries"]:          # an outdated copy of this very model
                ok, _, err = self._remove_entry(manifest, name)
                self._save(manifest)
                if not ok:
                    return _fail("evict_failed", f"Cannot replace the outdated staged copy: {err}")
            victims = self._lru_plan(manifest, total)
            if victims is None:
                return _fail("over_budget", f"The model ({total / GIB:.1f} GB) is larger than the staging budget "
                                            f"({self.budget_bytes / GIB:.1f} GB).")
            for v in victims:
                ok, _, err = self._remove_entry(manifest, v)
                if not ok:
                    self._save(manifest)
                    return _fail("evict_failed", f"Cannot free space in the cache: {err}")
            free = self._free_bytes()
            if free is not None and free < total + self.min_free_bytes:
                self._save(manifest)
                return _fail("disk_full", f"Not enough free space on the cache drive: {free / GIB:.1f} GB free, "
                                          f"{(total + self.min_free_bytes) / GIB:.1f} GB needed.")
            edir = self._owned_dir(name)
            edir.mkdir(parents=True, exist_ok=True)
            parts[:] = [edir / (f.name + ".part") for f in files]
            manifest["pending"] = [f"{name}/{pp.name}" for pp in parts]    # registered BEFORE the first byte is written
            self._save(manifest)

        t0, done = time.perf_counter(), [0]
        failure: tuple[str, str] | None = None
        try:
            for f, part in zip(files, parts):
                code = self._copy_file(f, part, done, total, progress, cancel)
                if code:
                    failure = (code, {"cancelled": "Cancelled; the partial copy was removed.",
                                      "disk_full": "The cache drive ran out of space; the partial copy was removed.",
                                      "io_error": "Reading or writing failed; the partial copy was removed."}[code])
                    break
            if failure is None:
                for f, part, size in zip(files, parts, sizes):
                    if part.stat().st_size != size:
                        failure = ("verify_failed", f"The copy of {f.name} has the wrong size.")
                        break
                    if verify_edges and self._edges(f, size) != self._edges(part, size):
                        failure = ("verify_failed", f"The copy of {f.name} does not match the original.")
                        break
            if failure is None:
                after = [(f.stat().st_size, f.stat().st_mtime_ns) for f in files]
                if after != before:
                    failure = ("source_changed", "The original model changed while it was being copied.")
            if failure is None:
                for part in parts:
                    final = part.with_name(part.name[:-len(".part")])
                    os.replace(part, final)
                    copied.append(final)
        except OSError as e:
            failure = ("disk_full" if e.errno in _DISK_FULL else "io_error", f"Staging failed: {e}")
        except BaseException:
            self._abort(name, parts, copied)
            raise
        seconds = time.perf_counter() - t0
        if failure is not None:
            self._abort(name, parts, copied)
            return _fail(failure[0], failure[1], cleaned=True)
        now = time.time()
        entry = {"name": name, "src": key, "src_size": before[0][0], "src_mtime_ns": before[0][1], "size": total,
                 "files": [{"name": f.name, "size": s} for f, s in zip(files, sizes)], "staged_at": now,
                 "last_used": now, "verified": {"size": True, "edges_sha256": bool(verify_edges)},
                 "shard_stats": [{"size": a, "mtime_ns": b} for a, b in before]}
        with self._lock:
            manifest = self._load()
            manifest["entries"][name] = entry
            manifest["pending"] = [x for x in manifest["pending"] if x not in {f"{name}/{pp.name}" for pp in parts}]
            self._save(manifest)
        return {"ok": True, "already_staged": False, "path": str(copied[0]), "name": name, "size_bytes": total,
                "seconds": round(seconds, 2),
                "mb_per_s": measured(round(total / 1e6 / seconds, 1), "wall clock", "includes reading the original")
                if seconds > 0 else unavailable("copy too fast to time", "wall clock"),
                "verified": entry["verified"]}

    def _abort(self, name: str, parts: list[Path], finals: list[Path]) -> None:
        """Remove what this attempt created (partial and renamed files), then forget the pending registration."""
        for path in [*parts, *finals]:
            try:
                path.unlink()
            except OSError:
                pass
        edir = self._owned_dir(name)
        if edir is not None:
            try:
                edir.rmdir()
            except OSError:
                pass
        with self._lock:
            manifest = self._load()
            wanted = {f"{name}/{pp.name}" for pp in parts}
            manifest["pending"] = [x for x in manifest["pending"] if x not in wanted]
            try:
                self._save(manifest)
            except (OSError, StoreError):
                pass
