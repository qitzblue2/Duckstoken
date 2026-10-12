"""Bounded structured performance log (JSON lines) that can never take the app down.

One record per line: ``{"t": <unix seconds>, "event": "<name>", ...fields}``. Values are cleaned (strings truncated,
non-finite floats dropped, containers capped) so a runaway field cannot grow the file; the file rotates by size
(``path`` -> ``path.1`` -> ... -> ``path.<backups>``) and the oldest backup is deleted. The file is opened for each write and
closed again so Windows can always delete or move it. Any I/O failure (disk full, permissions) drops the record and counts it.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

MAX_STR = 500
MAX_ITEMS = 50
MAX_DEPTH = 4
MAX_LINE = 16_000


def _clean(v, depth: int = 0):
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, str):
        return v if len(v) <= MAX_STR else v[:MAX_STR] + "...[truncated]"
    if depth >= MAX_DEPTH:
        return _clean(str(v), depth)
    if isinstance(v, dict):
        return {str(k)[:100]: _clean(x, depth + 1) for k, x in list(v.items())[:MAX_ITEMS]}
    if isinstance(v, (list, tuple, set, frozenset)):
        items = list(v)
        out = [_clean(x, depth + 1) for x in items[:MAX_ITEMS]]
        if len(items) > MAX_ITEMS:
            out.append(f"...{len(items) - MAX_ITEMS} more")
        return out
    if isinstance(v, bytes):
        return _clean(v.decode("utf-8", "replace"), depth)
    return _clean(str(v), depth)


class PerfLog:
    def __init__(self, path: str | Path, max_bytes: int = 2_000_000, backups: int = 3):
        self.path = Path(path)
        self.max_bytes = max(1024, int(max_bytes))
        self.backups = max(0, int(backups))
        self._lock = threading.Lock()
        self.written = 0
        self.dropped = 0
        self.rotations = 0
        self.last_error = ""

    # ------------------------------------------------------------------ writing
    def _line(self, event: str, fields: dict) -> str:
        rec = {"t": round(time.time(), 3), "event": _clean(str(event)[:100])}
        for k, v in fields.items():
            rec[k if k not in ("t", "event") else "field_" + k] = _clean(v)
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        if len(line) > MAX_LINE:
            rec = {"t": rec["t"], "event": rec["event"], "_truncated": True, "fields": sorted(fields)[:MAX_ITEMS]}
            line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        return line

    def _backup(self, i: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{i}")

    def _rotate(self) -> None:
        if self.backups <= 0:
            self.path.unlink(missing_ok=True)
        else:
            self._backup(self.backups).unlink(missing_ok=True)
            for i in range(self.backups - 1, 0, -1):
                if self._backup(i).exists():
                    os.replace(self._backup(i), self._backup(i + 1))
            os.replace(self.path, self._backup(1))
        self.rotations += 1

    def log(self, event: str, **fields) -> bool:
        """Append one record; returns False (and counts it in ``dropped``) when it could not be written. Never raises."""
        try:
            line = self._line(event, fields) + "\n"
            data = line.encode("utf-8", "replace")
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    size = self.path.stat().st_size
                except FileNotFoundError:
                    size = 0
                if size and size + len(data) > self.max_bytes:
                    self._rotate()
                with open(self.path, "ab") as f:
                    f.write(data)
                self.written += 1
            return True
        except Exception as e:
            self.dropped += 1
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            return False

    # ------------------------------------------------------------------ reading
    def _files(self) -> list[Path]:
        """Oldest first: highest backup number ... backup 1, then the live file."""
        files = [self._backup(i) for i in range(self.backups, 0, -1)] + [self.path]
        return [f for f in files if f.exists()]

    @staticmethod
    def _records(path: Path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue  # a torn or corrupt line is skipped, not fatal
                    if isinstance(rec, dict):
                        yield rec
        except OSError:
            return

    def tail(self, n: int = 100) -> list[dict]:
        """The newest `n` records, oldest of them first."""
        n = max(int(n), 0)
        if not n:
            return []
        with self._lock:
            out: list[dict] = []
            for f in reversed(self._files()):
                recs = list(self._records(f))
                out = recs + out
                if len(out) >= n:
                    break
        return out[-n:]

    def read_since(self, ts: float, limit: int = 5000) -> list[dict]:
        """Records newer than `ts` (unix seconds), oldest first, at most `limit` (the newest ones)."""
        with self._lock:
            out = [r for f in self._files() for r in self._records(f) if isinstance(r.get("t"), (int, float)) and r["t"] > ts]
        return out[-max(int(limit), 0):] if limit else []

    def stats(self) -> dict:
        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        return {"path": str(self.path), "size_bytes": size, "max_bytes": self.max_bytes, "backups": self.backups,
                "written": self.written, "dropped": self.dropped, "rotations": self.rotations, "last_error": self.last_error}

    def clear(self) -> None:
        with self._lock:
            for f in self._files():
                try:
                    f.unlink()
                except OSError:
                    pass
