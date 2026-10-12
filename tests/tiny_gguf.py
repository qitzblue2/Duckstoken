"""Writes tiny but VALID GGUF files for the built-in engine tests: random weights, a byte-level BPE or SentencePiece tokenizer,
and reference quantizers (a simple min/max encoder for every format the engine reads, so the dequantizers get real data).

    model = write_tiny_llama_gguf(path, n_layer=2, quant="q8_0", arch="qwen2")
    model.path, model.meta, model.tensors (the float32 weights before quantization), model.tokens
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ggml_type ids, elements per block and bytes per block
TYPE_ID = {"f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7, "q8_0": 8, "q2_k": 10, "q3_k": 11, "q4_k": 12,
           "q5_k": 13, "q6_k": 14, "iq4_nl": 20, "iq4_xs": 23, "bf16": 30}
BLOCK = {"f32": (1, 4), "f16": (1, 2), "bf16": (1, 2), "q4_0": (32, 18), "q4_1": (32, 20), "q5_0": (32, 22), "q5_1": (32, 24),
         "q8_0": (32, 34), "iq4_nl": (32, 18), "q2_k": (256, 84), "q3_k": (256, 110), "q4_k": (256, 144), "q5_k": (256, 176),
         "q6_k": (256, 210), "iq4_xs": (256, 136)}
IQ4_NL_TABLE = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
SEPARATOR = "\u2581"  # the SentencePiece word-boundary mark


# ------------------------------------------------------------------------------------------------ quantizers
def _put_f16(blocks: np.ndarray, at: int, values: np.ndarray) -> np.ndarray:
    """Store `values` (n,) as float16 at byte `at` of every block and return them rounded to float16, as float32."""
    h = np.ascontiguousarray(values.astype("<f2"))
    blocks[:, at:at + 2] = h.view(np.uint8).reshape(-1, 2)
    return h.astype(np.float32)


def _inverse(x: np.ndarray) -> np.ndarray:
    return np.divide(1.0, x, out=np.zeros_like(x), where=x != 0)


def _signed_max(x: np.ndarray) -> np.ndarray:
    """The element of largest magnitude along the last axis, with its sign."""
    return np.take_along_axis(x, np.abs(x).argmax(-1)[..., None], -1)[..., 0]


def _q8_0(x):
    x = x.reshape(-1, 32)
    blocks = np.empty((len(x), 34), np.uint8)
    d = _put_f16(blocks, 0, np.abs(x).max(1) / 127)
    blocks[:, 2:] = np.clip(np.rint(x * _inverse(d)[:, None]), -127, 127).astype(np.int8).view(np.uint8)
    return blocks


def _q4_0(x):
    x = x.reshape(-1, 32)
    blocks = np.empty((len(x), 18), np.uint8)
    d = _put_f16(blocks, 0, _signed_max(x) / -8)
    q = np.clip(np.floor(x * _inverse(d)[:, None] + 8.5), 0, 15).astype(np.uint8)
    blocks[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return blocks


def _q4_1(x):
    x = x.reshape(-1, 32)
    blocks = np.empty((len(x), 20), np.uint8)
    lo = _put_f16(blocks, 2, x.min(1))
    d = _put_f16(blocks, 0, (x.max(1) - lo) / 15)
    q = np.clip(np.floor((x - lo[:, None]) * _inverse(d)[:, None] + 0.5), 0, 15).astype(np.uint8)
    blocks[:, 4:] = q[:, :16] | (q[:, 16:] << 4)
    return blocks


def _q5_0(x):
    x = x.reshape(-1, 32)
    blocks = np.empty((len(x), 22), np.uint8)
    d = _put_f16(blocks, 0, _signed_max(x) / -16)
    q = np.clip(np.floor(x * _inverse(d)[:, None] + 16.5), 0, 31).astype(np.uint8)
    blocks[:, 2:6] = np.packbits(q >> 4, axis=1, bitorder="little")
    blocks[:, 6:] = (q[:, :16] & 15) | ((q[:, 16:] & 15) << 4)
    return blocks


def _q5_1(x):
    x = x.reshape(-1, 32)
    blocks = np.empty((len(x), 24), np.uint8)
    lo = _put_f16(blocks, 2, x.min(1))
    d = _put_f16(blocks, 0, (x.max(1) - lo) / 31)
    q = np.clip(np.floor((x - lo[:, None]) * _inverse(d)[:, None] + 0.5), 0, 31).astype(np.uint8)
    blocks[:, 4:8] = np.packbits(q >> 4, axis=1, bitorder="little")
    blocks[:, 8:] = (q[:, :16] & 15) | ((q[:, 16:] & 15) << 4)
    return blocks


def _k_min_max(x, n_sub: int, qmax: int, smax: int):
    """Shared by Q2_K/Q4_K/Q5_K: y = d*sc*q - dmin*mn per sub-block, with q in 0..qmax and 0 <= sc, mn <= smax.
    Returns (d, dmin, sc, mn, q)."""
    xs = x.reshape(len(x), n_sub, -1)
    low = np.minimum(xs.min(2), 0)
    scale, minimum = (xs.max(2) - low) / qmax, -low
    d, dmin = scale.max(1) / smax, minimum.max(1) / smax
    d16, dmin16 = d.astype("<f2").astype(np.float32), dmin.astype("<f2").astype(np.float32)
    sc = np.clip(np.rint(scale * _inverse(d16)[:, None]), 0, smax).astype(np.uint8)
    mn = np.clip(np.rint(minimum * _inverse(dmin16)[:, None]), 0, smax).astype(np.uint8)
    eff_scale, eff_min = d16[:, None] * sc, dmin16[:, None] * mn
    q = np.clip(np.rint((xs + eff_min[:, :, None]) * _inverse(eff_scale)[:, :, None]), 0, qmax).astype(np.uint8)
    return d, dmin, sc, mn, q.reshape(len(x), 256)


def _pack_scales_k4(sc, mn):
    """The 12-byte layout of 8 six-bit scales and 8 six-bit mins used by Q4_K and Q5_K."""
    out = np.empty((len(sc), 12), np.uint8)
    out[:, :4] = sc[:, :4] | ((sc[:, 4:] >> 4) << 6)
    out[:, 4:8] = mn[:, :4] | ((mn[:, 4:] >> 4) << 6)
    out[:, 8:] = (sc[:, 4:] & 15) | ((mn[:, 4:] & 15) << 4)
    return out


def _q4_k(x):
    x = x.reshape(-1, 256)
    blocks = np.empty((len(x), 144), np.uint8)
    d, dmin, sc, mn, q = _k_min_max(x, 8, 15, 63)
    _put_f16(blocks, 0, d)
    _put_f16(blocks, 2, dmin)
    blocks[:, 4:16] = _pack_scales_k4(sc, mn)
    q = q.reshape(-1, 4, 2, 32)
    blocks[:, 16:] = (q[:, :, 0] | (q[:, :, 1] << 4)).reshape(-1, 128)
    return blocks


def _q5_k(x):
    x = x.reshape(-1, 256)
    blocks = np.empty((len(x), 176), np.uint8)
    d, dmin, sc, mn, q = _k_min_max(x, 8, 31, 63)
    _put_f16(blocks, 0, d)
    _put_f16(blocks, 2, dmin)
    blocks[:, 4:16] = _pack_scales_k4(sc, mn)
    q = q.reshape(-1, 8, 32)
    blocks[:, 16:48] = np.packbits(q >> 4, axis=1, bitorder="little").reshape(-1, 32)
    low = (q & 15).reshape(-1, 4, 2, 32)
    blocks[:, 48:] = (low[:, :, 0] | (low[:, :, 1] << 4)).reshape(-1, 128)
    return blocks


def _pack_2bit(q):
    """q (n, 256) values 0..3 -> (n, 64) bytes: element 128*h + 32*j + l sits in byte 32*h + l at bit 2*j."""
    q = q.reshape(-1, 2, 4, 32).astype(np.uint8)
    return (q[:, :, 0] | (q[:, :, 1] << 2) | (q[:, :, 2] << 4) | (q[:, :, 3] << 6)).reshape(-1, 64)


def _q2_k(x):
    x = x.reshape(-1, 256)
    blocks = np.empty((len(x), 84), np.uint8)
    d, dmin, sc, mn, q = _k_min_max(x, 16, 3, 15)
    blocks[:, :16] = sc | (mn << 4)
    blocks[:, 16:80] = _pack_2bit(q)
    _put_f16(blocks, 80, d)
    _put_f16(blocks, 82, dmin)
    return blocks


def _symmetric_levels(x, n_sub: int, extreme: int, smax: int):
    """Shared by Q3_K/Q6_K: y = d*sc*q with signed integer scales. The element of largest magnitude maps to -extreme."""
    xs = x.reshape(len(x), n_sub, -1)
    scale = _signed_max(xs) / -extreme
    d = np.abs(scale).max(1) / smax
    d16 = d.astype("<f2").astype(np.float32)
    sc = np.clip(np.rint(scale * _inverse(d16)[:, None]), -smax - 1, smax)
    return xs, d, d16, sc


def _q3_k(x):
    x = x.reshape(-1, 256)
    blocks = np.empty((len(x), 110), np.uint8)
    xs, d, d16, sc = _symmetric_levels(x, 16, 4, 31)
    eff = d16[:, None] * sc
    q = np.clip(np.rint(xs * _inverse(eff)[:, :, None]), -4, 3).astype(np.int8) + 4  # 0..7: low two bits + high bit
    q = q.reshape(-1, 2, 4, 32).astype(np.uint8)
    blocks[:, :32] = np.packbits((q >> 2).reshape(-1, 8, 32), axis=1, bitorder="little").reshape(-1, 32)
    blocks[:, 32:96] = _pack_2bit((q & 3).reshape(-1, 256))
    s = (sc.astype(np.int16) + 32).astype(np.uint8)
    blocks[:, 96:100] = (s[:, 0:4] & 15) | ((s[:, 8:12] & 15) << 4)
    blocks[:, 100:104] = (s[:, 4:8] & 15) | ((s[:, 12:16] & 15) << 4)
    blocks[:, 104:108] = (s[:, 0:4] >> 4) | ((s[:, 4:8] >> 4) << 2) | ((s[:, 8:12] >> 4) << 4) | ((s[:, 12:16] >> 4) << 6)
    _put_f16(blocks, 108, d)
    return blocks


def _q6_k(x):
    x = x.reshape(-1, 256)
    blocks = np.empty((len(x), 210), np.uint8)
    xs, d, d16, sc = _symmetric_levels(x, 16, 32, 127)
    eff = d16[:, None] * sc
    q = (np.clip(np.rint(xs * _inverse(eff)[:, :, None]), -32, 31) + 32).astype(np.uint8).reshape(-1, 2, 4, 32)
    blocks[:, :128] = np.concatenate([(q[:, :, 0] & 15) | ((q[:, :, 2] & 15) << 4),
                                      (q[:, :, 1] & 15) | ((q[:, :, 3] & 15) << 4)], axis=2).reshape(-1, 128)
    blocks[:, 128:192] = ((q[:, :, 0] >> 4) | ((q[:, :, 1] >> 4) << 2) | ((q[:, :, 2] >> 4) << 4)
                          | ((q[:, :, 3] >> 4) << 6)).reshape(-1, 64)
    blocks[:, 192:208] = sc.astype(np.int8).view(np.uint8)
    _put_f16(blocks, 208, d)
    return blocks


def _nearest_iq4(values: np.ndarray) -> np.ndarray:
    return np.abs(values[..., None] - IQ4_NL_TABLE).argmin(-1).astype(np.uint8)


def _iq4_nl(x):
    x = x.reshape(-1, 32)
    blocks = np.empty((len(x), 18), np.uint8)
    d = _put_f16(blocks, 0, np.abs(x).max(1) / 127)
    q = _nearest_iq4(x * _inverse(d)[:, None])
    blocks[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return blocks


def _iq4_xs(x):
    x = x.reshape(-1, 256)
    blocks = np.empty((len(x), 136), np.uint8)
    xs = x.reshape(-1, 8, 32)
    want = np.abs(xs).max(2) / 127
    d = _put_f16(blocks, 0, want.max(1) / 31)
    ls = np.clip(np.rint(want * _inverse(d)[:, None]) + 32, 0, 63).astype(np.uint16)
    eff = d[:, None] * (ls.astype(np.float32) - 32)
    q = _nearest_iq4(xs * _inverse(eff)[:, :, None])
    blocks[:, 8:] = (q[:, :, :16] | (q[:, :, 16:] << 4)).reshape(-1, 128)
    blocks[:, 2:4] = np.ascontiguousarray(
        np.bitwise_or.reduce((ls >> 4) << (2 * np.arange(8, dtype=np.uint16)), axis=1).astype("<u2")).view(np.uint8).reshape(-1, 2)
    blocks[:, 4:8] = (ls[:, 0::2] & 15) | ((ls[:, 1::2] & 15) << 4)
    return blocks


def _f32(x):
    return np.ascontiguousarray(x.astype("<f4").reshape(-1, 1)).view(np.uint8)


def _f16(x):
    return np.ascontiguousarray(x.astype("<f2").reshape(-1, 1)).view(np.uint8)


def _bf16(x):
    bits = np.ascontiguousarray(x.astype("<f4")).view("<u4")
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16  # round to nearest even
    return np.ascontiguousarray(rounded.astype("<u2").reshape(-1, 1)).view(np.uint8)


_QUANTIZERS = {"f32": _f32, "f16": _f16, "bf16": _bf16, "q8_0": _q8_0, "q4_0": _q4_0, "q4_1": _q4_1, "q5_0": _q5_0,
               "q5_1": _q5_1, "q2_k": _q2_k, "q3_k": _q3_k, "q4_k": _q4_k, "q5_k": _q5_k, "q6_k": _q6_k, "iq4_nl": _iq4_nl,
               "iq4_xs": _iq4_xs}


def quantize(x: np.ndarray, qtype: str) -> bytes:
    """Encode a float array whose last axis is a multiple of the block size; the bytes follow the array's row-major order."""
    x = np.asarray(x, np.float32)
    elements, _ = BLOCK[qtype]
    if x.shape[-1] % elements:
        raise ValueError(f"row length {x.shape[-1]} is not a multiple of {elements} for {qtype}")
    return np.ascontiguousarray(_QUANTIZERS[qtype](x.reshape(-1, x.shape[-1]))).tobytes()


# ------------------------------------------------------------------------------------------------ GGUF writer
def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _scalar_type(v) -> tuple[int, str]:
    if isinstance(v, bool):
        return 7, "?"
    if isinstance(v, float):
        return 6, "f"
    if isinstance(v, int):
        return (4, "I") if 0 <= v < 1 << 32 else (11, "q")
    raise TypeError(f"unsupported metadata value {v!r}")


def _value(v) -> tuple[int, bytes]:
    if isinstance(v, str):
        return 8, _s(v)
    if isinstance(v, (list, tuple)):
        if v and isinstance(v[0], str):
            et, body = 8, b"".join(_s(x) for x in v)
        elif v and isinstance(v[0], int) and not isinstance(v[0], bool):
            et, fmt = 5, "i"  # integer arrays (token types) are int32 like gguf-py writes them
            body = struct.pack(f"<{len(v)}{fmt}", *v)
        else:
            et, fmt = _scalar_type(v[0]) if v else (4, "I")
            body = struct.pack(f"<{len(v)}{fmt}", *v)
        return 9, struct.pack("<IQ", et, len(v)) + body
    t, fmt = _scalar_type(v)
    return t, struct.pack("<" + fmt, v)


def write_gguf(path, meta: dict, tensors: list[tuple[str, tuple[int, ...], int, bytes]], alignment: int = 32,
               version: int = 3) -> None:
    """tensors: (name, numpy shape, ggml type id, raw bytes). The GGUF dims are the numpy shape reversed."""
    head = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(meta))
    for key, v in meta.items():
        t, body = _value(v)
        head += _s(key) + struct.pack("<I", t) + body
    offsets, pos = [], 0
    for _, _, _, raw in tensors:
        offsets.append(pos)
        pos += -(-len(raw) // alignment) * alignment
    for (name, shape, gtype, _), off in zip(tensors, offsets):
        head += _s(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape[::-1])
        head += struct.pack("<IQ", gtype, off)
    head += b"\0" * (-len(head) % alignment)
    with open(path, "wb") as f:
        f.write(head)
        for _, _, _, raw in tensors:
            f.write(raw + b"\0" * (-len(raw) % alignment))


# ------------------------------------------------------------------------------------------------ tokenizers
CORPUS = ("the quick brown fox jumps over the lazy dog. Hello, world! The theory of the thing: in the end there is "
          "nothing in there. 123 456 7890 hello hello world 2024-05-17 it's don't we're they'll I'd you've she's "
          "line one\nline two\n\n\tindented   spaces and trailing  \n").split(" ")


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable character table."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    table, extra = {b: chr(b) for b in keep}, 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + extra)
            extra += 1
    return table


def _train_merges(words: list[list[str]], n_merges: int) -> list[tuple[str, str]]:
    merges = []
    for _ in range(n_merges):
        counts: dict[tuple[str, str], int] = {}
        for w in words:
            for a, b in zip(w, w[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], p))
        merges.append(best)
        for w in words:
            i = 0
            while i < len(w) - 1:
                if (w[i], w[i + 1]) == best:
                    w[i:i + 2] = [w[i] + w[i + 1]]
                else:
                    i += 1
    return merges


def bpe_tokenizer_meta(pre: str = "qwen2", n_merges: int = 48, chat_template: str | None = None) -> dict:
    """Metadata of a tiny byte-level BPE tokenizer: 256 byte tokens, trained merges, chatml control tokens, <think> markers."""
    table = bytes_to_unicode()
    words = [[table[b] for b in (" " + w).encode()] for w in CORPUS if w]
    merges = _train_merges(words, n_merges)
    tokens = [table[b] for b in range(256)] + [a + b for a, b in merges]
    types = [1] * len(tokens)
    specials = [("<|endoftext|>", 3), ("<|im_start|>", 3), ("<|im_end|>", 3), ("<think>", 4), ("</think>", 4)]
    ids = {text: len(tokens) + i for i, (text, _) in enumerate(specials)}
    tokens += [t for t, _ in specials]
    types += [k for _, k in specials]
    meta = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": pre, "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.token_type": types, "tokenizer.ggml.merges": [f"{a} {b}" for a, b in merges],
            "tokenizer.ggml.bos_token_id": ids["<|endoftext|>"], "tokenizer.ggml.eos_token_id": ids["<|im_end|>"],
            "tokenizer.ggml.padding_token_id": ids["<|endoftext|>"], "tokenizer.ggml.add_bos_token": False}
    if chat_template is not None:
        meta["tokenizer.chat_template"] = chat_template
    return meta


def spm_tokenizer_meta(n_merges: int = 40, chat_template: str | None = None, add_space_prefix: bool = True) -> dict:
    """Metadata of a tiny SentencePiece (llama-style) tokenizer: <unk>/<s>/</s>, byte fallback tokens, ASCII, trained pieces."""
    chars = [chr(c) for c in range(33, 127)]
    words = [[SEPARATOR] + list(w) for w in CORPUS if w]
    merges = _train_merges(words, n_merges)
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)] + [SEPARATOR] + chars
    types = [2, 3, 3] + [6] * 256 + [1] * (1 + len(chars))
    scores = [0.0] * len(tokens)
    for i, (a, b) in enumerate(merges):
        tokens.append(a + b)
        types.append(1)
        scores.append(-float(i + 1))
    meta = {"tokenizer.ggml.model": "llama", "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.token_type": types,
            "tokenizer.ggml.scores": scores, "tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.unknown_token_id": 0, "tokenizer.ggml.add_bos_token": True,
            "tokenizer.ggml.add_eos_token": False, "tokenizer.ggml.add_space_prefix": add_space_prefix}
    if chat_template is not None:
        meta["tokenizer.chat_template"] = chat_template
    return meta


# ------------------------------------------------------------------------------------------------ tiny model
@dataclass
class TinyModel:
    path: Path
    meta: dict
    tensors: dict[str, np.ndarray]  # float32 weights as generated, before quantization
    tokens: list[str]


def write_tiny_llama_gguf(path, *, n_layer: int = 2, n_embd: int = 32, n_head: int = 4, n_kv_head: int = 2, n_ff: int = 64,
                          head_dim: int | None = None, vocab: dict | None = None, tokenizer: str = "bpe", quant: str = "f32",
                          seed: int = 0, arch: str = "llama", tie_embeddings: bool = False, rope_base: float = 10000.0,
                          rope_freqs: np.ndarray | None = None, context_length: int = 256, chat_template: str | None = None,
                          name: str = "tiny", extra_meta: dict | None = None, mutate=None) -> TinyModel:
    """Write a valid decoder-only GGUF with random weights.

    `arch` is "llama", "qwen2" (q/k/v biases) or "qwen3" (q/k RMS norms); `tokenizer` is "bpe" or "spm" unless `vocab` (tokenizer
    metadata, e.g. from bpe_tokenizer_meta) is given. 2-D weights are stored as `quant`; tensors whose rows do not fit the block
    size fall back to q8_0, as llama.cpp does for K-quants. `extra_meta` adds or overrides metadata; `mutate(weights)` may edit
    the {tensor name: float32 array} dict before it is written (to build broken or unusual files).
    """
    rng = np.random.default_rng(seed)
    tok_meta = vocab if vocab is not None else (spm_tokenizer_meta(chat_template=chat_template) if tokenizer == "spm"
                                                else bpe_tokenizer_meta(chat_template=chat_template))
    tokens = tok_meta["tokenizer.ggml.tokens"]
    n_vocab, hd = len(tokens), head_dim or n_embd // n_head
    q_dim, kv_dim = n_head * hd, n_kv_head * hd

    def weight(rows, cols):
        return (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)

    def norm(n):
        return (1 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    w: dict[str, np.ndarray] = {"token_embd.weight": (0.5 * rng.standard_normal((n_vocab, n_embd))).astype(np.float32),
                                "output_norm.weight": norm(n_embd)}
    if not tie_embeddings:
        w["output.weight"] = weight(n_vocab, n_embd)
    if rope_freqs is not None:
        w["rope_freqs.weight"] = np.asarray(rope_freqs, np.float32)
    for i in range(n_layer):
        b = f"blk.{i}."
        w[b + "attn_norm.weight"] = norm(n_embd)
        for proj, rows in (("q", q_dim), ("k", kv_dim), ("v", kv_dim)):
            w[b + f"attn_{proj}.weight"] = weight(rows, n_embd)
        if arch == "qwen2":
            w[b + "attn_q.bias"], w[b + "attn_k.bias"], w[b + "attn_v.bias"] = (
                (0.2 * rng.standard_normal(n)).astype(np.float32) for n in (q_dim, kv_dim, kv_dim))
        if arch == "qwen3":
            w[b + "attn_q_norm.weight"], w[b + "attn_k_norm.weight"] = norm(hd), norm(hd)
        w[b + "attn_output.weight"] = weight(n_embd, q_dim)
        w[b + "ffn_norm.weight"] = norm(n_embd)
        w[b + "ffn_gate.weight"], w[b + "ffn_up.weight"] = weight(n_ff, n_embd), weight(n_ff, n_embd)
        w[b + "ffn_down.weight"] = weight(n_embd, n_ff)

    if mutate is not None:
        mutate(w)

    meta = {"general.architecture": arch, "general.name": name, "general.file_type": 0,
            f"{arch}.context_length": context_length, f"{arch}.embedding_length": n_embd, f"{arch}.block_count": n_layer,
            f"{arch}.feed_forward_length": n_ff, f"{arch}.attention.head_count": n_head,
            f"{arch}.attention.head_count_kv": n_kv_head, f"{arch}.attention.layer_norm_rms_epsilon": 1e-5,
            f"{arch}.rope.freq_base": float(rope_base), f"{arch}.vocab_size": n_vocab}
    if arch == "qwen3":
        meta[f"{arch}.attention.key_length"] = hd
        meta[f"{arch}.attention.value_length"] = hd
    else:
        meta[f"{arch}.rope.dimension_count"] = hd
    meta.update(tok_meta)
    meta.update(extra_meta or {})

    tensors = []
    for tname, arr in w.items():
        qtype = quant if arr.ndim == 2 and tname != "rope_freqs.weight" else "f32"
        if arr.ndim == 2 and arr.shape[1] % BLOCK[qtype][0]:
            qtype = "q8_0"
        tensors.append((tname, arr.shape, TYPE_ID[qtype], quantize(arr, qtype)))
    write_gguf(path, meta, tensors)
    return TinyModel(Path(path), meta, w, tokens)
