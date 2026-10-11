"""GGUF reader for the built-in engine: parses the header and metadata, memory-maps the tensor data and dequantizes
tensors to NumPy arrays. Parsing needs no NumPy; only `GGUFFile.load` does."""

from __future__ import annotations

import gc
import mmap
import struct
from pathlib import Path
from typing import Callable, NamedTuple

try:
    import numpy as np
except ImportError:  # the engine package reports itself unavailable instead
    np = None

MAX_STRING = 1 << 26        # bytes; a chat template or one token is far smaller
MAX_ENTRIES = 1 << 20       # metadata keys / tensors we are willing to parse
MAX_ARRAY_DEPTH = 2
_CHUNK_ELEMENTS = 1 << 22   # dequantize this many elements at a time to bound temporary memory


class UnsupportedModel(Exception):
    """The file or model cannot be run by the built-in engine; the message says why in plain words."""


class TensorInfo(NamedTuple):
    name: str
    shape: tuple[int, ...]   # NumPy / row-major order: the reverse of the GGUF dims
    ggml_type: int
    offset: int              # absolute offset of the tensor data in the file
    n_bytes: int             # 0 for a type this reader cannot size

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def type_name(self) -> str:
        return type_name(self.ggml_type)


# ------------------------------------------------------------------------------------------------ dequantizers
# Each takes `blocks`, a (n_blocks, block_bytes) uint8 array, and returns a (n_blocks, block_elements) float32 array.

def _f16(blocks, at: int):
    """The little-endian float16 stored at byte `at` of every block, as a float32 column."""
    return np.ascontiguousarray(blocks[:, at:at + 2]).view("<f2").astype(np.float32)


def _nibbles(qs, out) -> None:
    """Fill `out` (n, 2*w) from `qs` (n, w): the low nibbles first, then the high nibbles."""
    w = qs.shape[1]
    out[:, :w] = qs & 15
    out[:, w:] = qs >> 4


def _q4_0(b):
    out = np.empty((len(b), 32), np.float32)
    _nibbles(b[:, 2:], out)
    out -= 8
    out *= _f16(b, 0)
    return out


def _q4_1(b):
    out = np.empty((len(b), 32), np.float32)
    _nibbles(b[:, 4:], out)
    out *= _f16(b, 0)
    out += _f16(b, 2)
    return out


def _q5_high_bits(b):
    """The fifth bit of the 32 values of a Q5_0/Q5_1 block, already shifted into place (0 or 16)."""
    return np.unpackbits(np.ascontiguousarray(b[:, :4]), axis=1, bitorder="little") << 4


def _q5_0(b):
    out = np.empty((len(b), 32), np.float32)
    _nibbles(b[:, 6:], out)
    out += _q5_high_bits(b[:, 2:])
    out -= 16
    out *= _f16(b, 0)
    return out


def _q5_1(b):
    out = np.empty((len(b), 32), np.float32)
    _nibbles(b[:, 8:], out)
    out += _q5_high_bits(b[:, 4:])
    out *= _f16(b, 0)
    out += _f16(b, 2)
    return out


def _q8_0(b):
    out = b[:, 2:].view(np.int8).astype(np.float32)
    out *= _f16(b, 0)
    return out


def _scale_min_k4(sc):
    """Unpack the 6-bit scales and mins of a Q4_K/Q5_K block from its 12 packed bytes -> two (n, 8) uint8 arrays."""
    scales = np.empty((len(sc), 8), np.uint8)
    mins = np.empty((len(sc), 8), np.uint8)
    scales[:, :4] = sc[:, :4] & 63
    mins[:, :4] = sc[:, 4:8] & 63
    scales[:, 4:] = (sc[:, 8:] & 15) | ((sc[:, :4] >> 6) << 4)
    mins[:, 4:] = (sc[:, 8:] >> 4) | ((sc[:, 4:8] >> 6) << 4)
    return scales, mins


def _q4_k(b):
    n = len(b)
    scales, mins = _scale_min_k4(b[:, 4:16])
    d = _f16(b, 0) * scales
    m = _f16(b, 2) * mins
    qs = b[:, 16:].reshape(n, 4, 32)
    out = np.empty((n, 8, 32), np.float32)
    out[:, 0::2] = qs & 15
    out[:, 1::2] = qs >> 4
    out *= d[:, :, None]
    out -= m[:, :, None]
    return out.reshape(n, 256)


def _q5_k(b):
    n = len(b)
    scales, mins = _scale_min_k4(b[:, 4:16])
    d = _f16(b, 0) * scales
    m = _f16(b, 2) * mins
    qh = b[:, 16:48]
    qs = b[:, 48:].reshape(n, 4, 32)
    out = np.empty((n, 8, 32), np.float32)
    out[:, 0::2] = qs & 15
    out[:, 1::2] = qs >> 4
    out += ((qh[:, None, :] >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1) << 4
    out *= d[:, :, None]
    out -= m[:, :, None]
    return out.reshape(n, 256)


def _q6_k(b):
    n = len(b)
    ql = b[:, :128].reshape(n, 2, 2, 32)
    qh = b[:, 128:192].reshape(n, 2, 32)
    sc = b[:, 192:208].view(np.int8).astype(np.float32) * _f16(b, 208)
    out = np.empty((n, 2, 4, 32), np.float32)
    out[:, :, 0] = (ql[:, :, 0] & 15) | ((qh & 3) << 4)
    out[:, :, 1] = (ql[:, :, 1] & 15) | (((qh >> 2) & 3) << 4)
    out[:, :, 2] = (ql[:, :, 0] >> 4) | (((qh >> 4) & 3) << 4)
    out[:, :, 3] = (ql[:, :, 1] >> 4) | ((qh >> 6) << 4)
    out -= 32
    out = out.reshape(n, 16, 16)
    out *= sc[:, :, None]
    return out.reshape(n, 256)


def _two_bit_groups(qs):
    """(n, 64) packed 2-bit values -> (n, 2, 4, 32): [half][shift step][position] as ordered in a Q2_K/Q3_K block."""
    q = qs.reshape(len(qs), 2, 1, 32)
    return (q >> (2 * np.arange(4, dtype=np.uint8))[None, None, :, None]) & 3


def _q2_k(b):
    n = len(b)
    sc = b[:, :16]
    d = _f16(b, 80) * (sc & 15)
    m = _f16(b, 82) * (sc >> 4)
    out = _two_bit_groups(b[:, 16:80]).astype(np.float32).reshape(n, 16, 16)
    out *= d[:, :, None]
    out -= m[:, :, None]
    return out.reshape(n, 256)


def _q3_k(b):
    n = len(b)
    hmask = b[:, :32]
    raw = b[:, 96:108]
    low, top = raw[:, :8], raw[:, 8:]
    scales = np.empty((n, 16), np.uint8)
    scales[:, 0:4] = (low[:, 0:4] & 15) | ((top & 3) << 4)
    scales[:, 4:8] = (low[:, 4:8] & 15) | (((top >> 2) & 3) << 4)
    scales[:, 8:12] = (low[:, 0:4] >> 4) | (((top >> 4) & 3) << 4)
    scales[:, 12:16] = (low[:, 4:8] >> 4) | ((top >> 6) << 4)
    d = (scales.astype(np.float32) - 32) * _f16(b, 108)
    high = (hmask[:, None, None, :] >> np.arange(8, dtype=np.uint8).reshape(2, 4)[None, :, :, None]) & 1
    out = _two_bit_groups(b[:, 32:96]).astype(np.float32)
    out += high * 4
    out -= 4
    out = out.reshape(n, 16, 16)
    out *= d[:, :, None]
    return out.reshape(n, 256)


_IQ4_NL_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _iq4_nl(b):
    table = np.array(_IQ4_NL_VALUES, np.float32)
    out = np.empty((len(b), 32), np.float32)
    qs = b[:, 2:]
    out[:, :16] = table[qs & 15]
    out[:, 16:] = table[qs >> 4]
    out *= _f16(b, 0)
    return out


def _iq4_xs(b):
    n = len(b)
    table = np.array(_IQ4_NL_VALUES, np.float32)
    high = np.ascontiguousarray(b[:, 2:4]).view("<u2")
    low = b[:, 4:8]
    ls = np.empty((n, 8), np.int16)
    ls[:, 0::2] = low & 15
    ls[:, 1::2] = low >> 4
    ls += ((high >> (2 * np.arange(8, dtype=np.uint16))[None, :]) & 3).astype(np.int16) << 4
    dl = (ls - 32) * _f16(b, 0)
    qs = b[:, 8:].reshape(n, 8, 16)
    out = np.empty((n, 8, 32), np.float32)
    out[:, :, :16] = table[qs & 15]
    out[:, :, 16:] = table[qs >> 4]
    out *= dl[:, :, None]
    return out.reshape(n, 256)


def _raw_float(dtype: str) -> Callable:
    def dequantize(b):
        return np.ascontiguousarray(b).view(dtype).astype(np.float32)
    return dequantize


def _bf16(b):
    return (np.ascontiguousarray(b).view("<u2").astype(np.uint32) << 16).view(np.float32)


class _GType(NamedTuple):
    name: str
    block: int                  # elements per block
    size: int                   # bytes per block
    dequantize: Callable | None


# ggml_type id -> layout. Types without a dequantizer are listed so that errors can name them.
_TYPES: dict[int, _GType] = {
    0: _GType("F32", 1, 4, _raw_float("<f4")),
    1: _GType("F16", 1, 2, _raw_float("<f2")),
    2: _GType("Q4_0", 32, 18, _q4_0),
    3: _GType("Q4_1", 32, 20, _q4_1),
    6: _GType("Q5_0", 32, 22, _q5_0),
    7: _GType("Q5_1", 32, 24, _q5_1),
    8: _GType("Q8_0", 32, 34, _q8_0),
    9: _GType("Q8_1", 32, 36, None),
    10: _GType("Q2_K", 256, 84, _q2_k),
    11: _GType("Q3_K", 256, 110, _q3_k),
    12: _GType("Q4_K", 256, 144, _q4_k),
    13: _GType("Q5_K", 256, 176, _q5_k),
    14: _GType("Q6_K", 256, 210, _q6_k),
    15: _GType("Q8_K", 256, 292, None),
    20: _GType("IQ4_NL", 32, 18, _iq4_nl),
    23: _GType("IQ4_XS", 256, 136, _iq4_xs),
    28: _GType("F64", 1, 8, _raw_float("<f8")),
    30: _GType("BF16", 1, 2, _bf16),
}
_UNSIZED_NAMES = {16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 21: "IQ3_S", 22: "IQ2_S", 24: "I8", 25: "I16",
                  26: "I32", 27: "I64", 29: "IQ1_M", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}


def type_name(ggml_type: int) -> str:
    t = _TYPES.get(ggml_type)
    return t.name if t else _UNSIZED_NAMES.get(ggml_type, f"type {ggml_type}")


def is_supported(ggml_type: int) -> bool:
    """True when `GGUFFile.load` can dequantize tensors of this ggml type."""
    t = _TYPES.get(ggml_type)
    return bool(t and t.dequantize)


SUPPORTED_TYPES = [t.name for t in _TYPES.values() if t.dequantize]

# ------------------------------------------------------------------------------------------------------ parsing
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_U32, _U64 = struct.Struct("<I"), struct.Struct("<Q")
_TRUNCATED = "the file is truncated or corrupt"


class _Reader:
    """Sequential little-endian reader over the memory-mapped header with bounds checks on every read."""

    def __init__(self, buf):
        self.buf, self.pos, self.end = buf, 0, len(buf)

    def take(self, n: int) -> int:
        """Reserve the next n bytes and return where they start."""
        start = self.pos
        if n < 0 or start + n > self.end:
            raise UnsupportedModel(f"Not a valid GGUF file: {_TRUNCATED}.")
        self.pos = start + n
        return start

    def u32(self) -> int:
        return _U32.unpack_from(self.buf, self.take(4))[0]

    def u64(self) -> int:
        return _U64.unpack_from(self.buf, self.take(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > MAX_STRING:
            raise UnsupportedModel("Not a valid GGUF file: an absurdly long string.")
        start = self.take(n)
        return str(self.buf[start:start + n], "utf-8", "replace")

    def value(self, vtype: int, depth: int = 0):
        code = _SCALARS.get(vtype)
        if code:
            return struct.unpack_from("<" + code, self.buf, self.take(struct.calcsize(code)))[0]
        if vtype == 8:
            return self.string()
        if vtype == 9:
            if depth >= MAX_ARRAY_DEPTH:
                raise UnsupportedModel("Not a valid GGUF file: arrays nested too deeply.")
            etype, n = self.u32(), self.u64()
            if etype in _SCALARS:
                fmt = f"<{n}{_SCALARS[etype]}"
                return list(struct.unpack_from(fmt, self.buf, self.take(struct.calcsize(fmt))))
            if etype == 8:
                if n * 8 > self.end - self.pos:
                    raise UnsupportedModel(f"Not a valid GGUF file: {_TRUNCATED}.")
                return [self.string() for _ in range(n)]
            if n > self.end - self.pos:
                raise UnsupportedModel(f"Not a valid GGUF file: {_TRUNCATED}.")
            return [self.value(etype, depth + 1) for _ in range(n)]
        raise UnsupportedModel(f"Not a valid GGUF file: unknown metadata value type {vtype}.")


class GGUFFile:
    """An open GGUF file: `meta` holds all metadata (including the tokenizer arrays), `tensors` maps names to `TensorInfo`.

    Tensor data stays memory-mapped on disk; `load` copies and dequantizes one tensor at a time. Close the file (or use it as a
    context manager) when done so that it can be deleted on Windows.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.meta: dict = {}
        self.tensors: dict[str, TensorInfo] = {}
        self._mm = None
        try:
            with open(self.path, "rb") as f:
                self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError) as e:
            raise UnsupportedModel(f"Cannot open the model file {self.path.name}: {e}") from e
        try:
            self._parse()
        except Exception:
            self.close()
            raise

    # ----------------------------------------------------------------------------------------------- header
    def _parse(self) -> None:
        r = _Reader(self._mm)
        if r.end < 24 or self._mm[:4] != b"GGUF":
            raise UnsupportedModel(f"{self.path.name} is not a GGUF file.")
        r.pos = 4
        version = r.u32()
        if version not in (2, 3):
            hint = " (big-endian files are not supported)" if int.from_bytes(struct.pack("<I", version), "big") in (2, 3) else ""
            raise UnsupportedModel(f"Unsupported GGUF version {version}{hint}.")
        n_tensors, n_kv = r.u64(), r.u64()
        if n_tensors > MAX_ENTRIES or n_kv > MAX_ENTRIES:
            raise UnsupportedModel("Not a valid GGUF file: implausible tensor or metadata count.")
        for _ in range(n_kv):
            key = r.string()
            if key in self.meta:
                raise UnsupportedModel(f"Not a valid GGUF file: duplicate metadata key {key!r}.")
            self.meta[key] = r.value(r.u32())
        alignment = self.meta.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
            raise UnsupportedModel("Not a valid GGUF file: bad tensor alignment.")
        if self.meta.get("split.count", 1) != 1:
            raise UnsupportedModel("Split (multi-file) GGUF models are not supported; merge them or use the llama.cpp engine.")
        raw = []
        for _ in range(n_tensors):
            name, n_dims = r.string(), r.u32()
            if not 1 <= n_dims <= 4:
                raise UnsupportedModel(f"Not a valid GGUF file: tensor {name!r} has {n_dims} dimensions.")
            dims = tuple(r.u64() for _ in range(n_dims))
            raw.append((name, dims, r.u32(), r.u64()))
        data_start = -(-r.pos // alignment) * alignment
        for name, dims, gtype, rel in raw:
            if name in self.tensors:
                raise UnsupportedModel(f"Not a valid GGUF file: duplicate tensor {name!r}.")
            self.tensors[name] = self._tensor_info(name, dims, gtype, data_start + rel)

    def _tensor_info(self, name: str, dims: tuple[int, ...], gtype: int, offset: int) -> TensorInfo:
        if not all(0 < d < 1 << 40 for d in dims):
            raise UnsupportedModel(f"Not a valid GGUF file: tensor {name!r} has invalid dimensions {dims}.")
        n_bytes, layout = 0, _TYPES.get(gtype)
        if layout:
            if dims[0] % layout.block:
                raise UnsupportedModel(f"Not a valid GGUF file: tensor {name!r} ({layout.name}) has a row length of "
                                       f"{dims[0]}, not a multiple of {layout.block}.")
            n = 1
            for d in dims:
                n *= d
            n_bytes = n // layout.block * layout.size
        if offset + n_bytes > len(self._mm):
            raise UnsupportedModel(f"Not a valid GGUF file: tensor {name!r} lies beyond the end of the file ({_TRUNCATED}).")
        return TensorInfo(name, dims[::-1], gtype, offset, n_bytes)

    # ---------------------------------------------------------------------------------------------- tensors
    def load(self, name: str, dtype="float32", out=None):
        """Read tensor `name` as a dequantized C-contiguous array (float32 or float16), optionally into the preallocated `out`."""
        if np is None:
            raise UnsupportedModel("NumPy is not installed.")
        if self._mm is None:
            raise ValueError("the GGUF file is closed")
        info = self.tensors.get(name)
        if info is None:
            raise UnsupportedModel(f"The model has no tensor named {name!r}.")
        layout = _TYPES.get(info.ggml_type)
        if layout is None or layout.dequantize is None:
            raise UnsupportedModel(f"Tensor {name!r} uses the {info.type_name} format, which the built-in engine cannot read. "
                                   f"Supported: {', '.join(SUPPORTED_TYPES)}. Use a Q8_0 or Q4_K_M file, or the llama.cpp engine.")
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
            raise ValueError("dtype must be float32 or float16")
        if out is None:
            out = np.empty(info.shape, dtype)
        elif out.shape != info.shape or out.dtype != dtype or not out.flags.c_contiguous:
            raise ValueError(f"`out` must be a C-contiguous {dtype} array of shape {info.shape}")
        flat = out.reshape(-1)
        raw = np.frombuffer(self._mm, np.uint8, info.n_bytes, info.offset)
        blocks = raw.reshape(-1, layout.size)
        step = max(1, _CHUNK_ELEMENTS // layout.block)
        chunk = None
        try:
            for i in range(0, len(blocks), step):
                chunk = layout.dequantize(blocks[i:i + step])
                flat[i * layout.block:(i + len(chunk)) * layout.block] = chunk.reshape(-1)
        finally:
            del raw, blocks, chunk
        return out

    def close(self) -> None:
        mm, self._mm = self._mm, None
        if mm is None:
            return
        try:
            mm.close()
        except BufferError:  # a NumPy view of the mapping is still being released by the garbage collector
            gc.collect()
            mm.close()

    @property
    def closed(self) -> bool:
        return self._mm is None

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
