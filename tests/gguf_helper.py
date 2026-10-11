"""Builds tiny synthetic GGUF files for tests."""
import struct


def _s(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kvs: dict, pad: int = 0):
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs) + 1)
    for k, v in kvs.items():
        out += _s(k)
        if isinstance(v, str):
            out += struct.pack("<I", 8) + _s(v)
        elif isinstance(v, float):
            out += struct.pack("<I", 6) + struct.pack("<f", v)
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", v)
    out += _s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", 2)
    out += _s("a") + _s("b")
    with open(path, "wb") as f:
        f.write(out + b"\0" * pad)
