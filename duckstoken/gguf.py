"""Minimal GGUF header reader: enough to learn layer/expert counts without loading the model."""

from __future__ import annotations

import re
import struct
from pathlib import Path

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
            12: "<d"}
_SHARD = re.compile(r"^(?P<base>.*)-(?P<n>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I)


class GGUFError(ValueError):
    pass


def _read(f, fmt):
    size = struct.calcsize(fmt)
    buf = f.read(size)
    if len(buf) != size:
        raise GGUFError("Unexpected end of file")
    return struct.unpack(fmt, buf)[0]


def _read_str(f) -> str:
    n = _read(f, "<Q")
    if n > 1 << 24:
        raise GGUFError("Corrupt string length")
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, t, keep=True):
    if t in _SCALARS:
        return _read(f, _SCALARS[t])
    if t == 8:
        return _read_str(f)
    if t == 9:
        et = _read(f, "<I")
        n = _read(f, "<Q")
        if et in _SCALARS and not keep:
            f.seek(struct.calcsize(_SCALARS[et]) * n, 1)
            return None
        return [_read_value(f, et, keep) for _ in range(n)] if keep else [_read_value(f, et, False) for _ in range(n)]
    raise GGUFError(f"Unknown GGUF value type {t}")


def read_metadata(path: str | Path, stop_prefix: str = "tokenizer.") -> dict:
    """Return scalar/string metadata, stopping before the (huge) tokenizer arrays."""
    meta: dict = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise GGUFError("Not a GGUF file")
        version = _read(f, "<I")
        if version < 2:
            raise GGUFError(f"GGUF v{version} is too old")
        meta["_tensor_count"] = _read(f, "<Q")
        kv_count = _read(f, "<Q")
        for _ in range(kv_count):
            key = _read_str(f)
            if key.startswith(stop_prefix):
                break
            t = _read(f, "<I")
            val = _read_value(f, t, keep=False)
            if val is not None and not isinstance(val, list):
                meta[key] = val
    return meta


def shard_files(path: str | Path) -> list[Path]:
    """All files making up a (possibly split) model, in order."""
    p = Path(path)
    m = _SHARD.match(p.name)
    if not m:
        return [p]
    total = int(m["total"])
    files = [p.with_name(f"{m['base']}-{i:05d}-of-{m['total']}.gguf") for i in range(1, total + 1)]
    return [x for x in files if x.exists()] or [p]


def summarize(path: str | Path) -> dict:
    p = Path(path)
    meta = read_metadata(p)
    arch = str(meta.get("general.architecture", ""))
    files = shard_files(p)
    size = sum(x.stat().st_size for x in files)

    def g(suffix, default=0):
        return meta.get(f"{arch}.{suffix}", default)

    experts = int(g("expert_count"))
    return {
        "path": str(p),
        "name": meta.get("general.name") or p.stem,
        "architecture": arch,
        "size_bytes": size,
        "size_gb": round(size / 2**30, 2),
        "shards": len(files),
        "block_count": int(g("block_count")),
        "context_length": int(g("context_length")),
        "embedding_length": int(g("embedding_length")),
        "head_count": int(g("attention.head_count")),
        "head_count_kv": int(g("attention.head_count_kv")),
        "expert_count": experts,
        "expert_used_count": int(g("expert_used_count")),
        "is_moe": experts > 1,
        "file_type": meta.get("general.file_type"),
    }
