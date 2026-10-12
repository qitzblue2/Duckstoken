"""A small transformer inference engine in NumPy for dense decoder-only models read from GGUF files.

Runs Llama-style ("llama": Llama 2/3, TinyLlama, Mistral, SmolLM2), "qwen2" and "qwen3" models: RMSNorm, rotary embeddings
(adjacent-pair or NeoX pairing), grouped-query attention, SwiGLU, optional q/k/v biases and q/k norms, tied embeddings and a
KV cache. Weights are dequantized to float32 (or bfloat16 when RAM is short) when the model is loaded.
It is far slower than llama.cpp and is meant for models up to a few billion parameters.
"""

from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the engine package reports itself unavailable instead
    np = None

from .gguf_weights import SUPPORTED_TYPES, GGUFFile, UnsupportedModel, is_supported, type_name
from .tokenizer import Tokenizer

LLAMA_CPP_HINT = "Use the llama.cpp engine for this model."
MAX_PARAMS = 4_500_000_000
RAM_FRACTION = 0.6        # of the available RAM the model may take
BATCH = 256               # prompt tokens processed per forward pass (measured fastest of 32..512)
KV_FLOAT32_LIMIT = 256 << 20
WIDEN_ELEMENTS = 1 << 18  # bfloat16 weights are widened to float32 this many elements at a time

# arch -> (NeoX rope pairing instead of adjacent pairs, q/k/v biases, q/k norms)
_ARCHITECTURES = {"llama": (False, False, False), "qwen2": (True, True, False), "qwen3": (True, False, True)}
SUPPORTED_ARCHITECTURES = tuple(_ARCHITECTURES)
_MOE_HINTS = ("moe", "mixtral", "gpt-oss", "grok", "dbrx", "arctic", "jamba")


def _no_progress(fraction: float, message: str) -> None:
    pass


class ContextFull(ValueError):
    """The tokens would not fit in the session's context window."""


@dataclass(frozen=True)
class _Config:
    arch: str
    n_layer: int
    n_embd: int
    n_head: int
    n_kv_head: int
    head_dim: int
    n_ff: int
    n_vocab: int
    eps: float
    rope_dim: int
    rope_base: float
    rope_scale: float
    neox: bool
    qkv_bias: bool
    qk_norm: bool
    context_length: int

    @property
    def q_dim(self) -> int:
        return self.n_head * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_head * self.head_dim


def _count(meta: dict, key: str, default=None, limit: int = 1 << 20) -> int:
    """A positive integer metadata entry (bounded, so that a corrupt file cannot make us build absurd structures)."""
    v = meta.get(key, default)
    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= limit:
        raise UnsupportedModel(f"The model file lacks a valid {key!r} entry.")
    return v


def _read_config(meta: dict, tensors: dict) -> _Config:
    arch = str(meta.get("general.architecture", ""))
    if arch not in _ARCHITECTURES:
        moe = any(h in arch for h in _MOE_HINTS) or meta.get(f"{arch}.expert_count", 0)
        what = "a mixture-of-experts model, which" if moe else f"the '{arch or 'unknown'}' architecture, which"
        raise UnsupportedModel(f"This is {what} the built-in engine cannot run (it supports dense "
                               f"{', '.join(_ARCHITECTURES)} models up to a few billion parameters). {LLAMA_CPP_HINT}")
    if meta.get(f"{arch}.expert_count", 0):
        raise UnsupportedModel(f"This is a mixture-of-experts model, which the built-in engine cannot run. {LLAMA_CPP_HINT}")
    neox, qkv_bias, qk_norm = _ARCHITECTURES[arch]
    n_embd, n_head = _count(meta, f"{arch}.embedding_length"), _count(meta, f"{arch}.attention.head_count")
    n_kv_head = _count(meta, f"{arch}.attention.head_count_kv", n_head)
    head_dim = _count(meta, f"{arch}.attention.key_length", n_embd // n_head)
    if n_head % n_kv_head or head_dim % 2 or meta.get(f"{arch}.attention.value_length", head_dim) != head_dim:
        raise UnsupportedModel("This model's attention layout is not one the built-in engine supports.")
    rope_dim = _count(meta, f"{arch}.rope.dimension_count", head_dim)
    if rope_dim > head_dim or rope_dim % 2:
        raise UnsupportedModel("This model's rotary embedding layout is not one the built-in engine supports.")
    scaling = meta.get(f"{arch}.rope.scaling.type", "none")
    scale = 1.0
    if scaling == "linear":
        scale = float(meta.get(f"{arch}.rope.scaling.factor", 1.0))
        if not scale >= 1:
            raise UnsupportedModel("This model has an invalid RoPE scaling factor.")
    elif scaling not in ("none", ""):
        raise UnsupportedModel(f"This model uses {scaling!r} RoPE scaling, which the built-in engine does not implement. {LLAMA_CPP_HINT}")
    embd = tensors.get("token_embd.weight")
    if embd is None:
        raise UnsupportedModel("The model file has no token embedding table.")
    return _Config(arch=arch, n_layer=_count(meta, f"{arch}.block_count", limit=1024), n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head,
                   head_dim=head_dim, n_ff=_count(meta, f"{arch}.feed_forward_length"), n_vocab=embd.shape[0],
                   eps=float(meta.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)), rope_dim=rope_dim,
                   rope_base=float(meta.get(f"{arch}.rope.freq_base", 10000.0)), rope_scale=scale, neox=neox,
                   qkv_bias=qkv_bias, qk_norm=qk_norm, context_length=_count(meta, f"{arch}.context_length", limit=1 << 30))


def _tensor_names(cfg: _Config) -> dict[str, tuple[int, ...]]:
    """Every tensor the model needs and the shape it must have."""
    shapes = {"token_embd.weight": (cfg.n_vocab, cfg.n_embd), "output_norm.weight": (cfg.n_embd,)}
    for i in range(cfg.n_layer):
        b = f"blk.{i}."
        shapes.update({b + "attn_norm.weight": (cfg.n_embd,), b + "attn_q.weight": (cfg.q_dim, cfg.n_embd),
                       b + "attn_k.weight": (cfg.kv_dim, cfg.n_embd), b + "attn_v.weight": (cfg.kv_dim, cfg.n_embd),
                       b + "attn_output.weight": (cfg.n_embd, cfg.q_dim), b + "ffn_norm.weight": (cfg.n_embd,),
                       b + "ffn_gate.weight": (cfg.n_ff, cfg.n_embd), b + "ffn_up.weight": (cfg.n_ff, cfg.n_embd),
                       b + "ffn_down.weight": (cfg.n_embd, cfg.n_ff)})
        if cfg.qkv_bias:
            shapes.update({b + "attn_q.bias": (cfg.q_dim,), b + "attn_k.bias": (cfg.kv_dim,), b + "attn_v.bias": (cfg.kv_dim,)})
        if cfg.qk_norm:
            shapes.update({b + "attn_q_norm.weight": (cfg.head_dim,), b + "attn_k_norm.weight": (cfg.head_dim,)})
    return shapes


def _free_ram_bytes() -> int:
    try:
        from ...hardware import ram_info
        return int(ram_info()["avail_gb"] * 2**30)
    except Exception:  # noqa: BLE001 - unknown RAM: do not block loading
        return 0


_scratch = threading.local()


def _widen_buffer(n: int):
    """A float32 scratch array whose low 16 bits are zero. Writing a bfloat16 value into the high half of each element
    (little-endian) turns it into the float32 it stands for, far faster than a float16 -> float32 conversion."""
    buf = getattr(_scratch, "buf", None)
    if buf is None or buf.size < n:
        buf = _scratch.buf = np.zeros(n, np.float32)
    return buf


class _Matrix:
    """A weight matrix (n_out, n_in) as float32, or as bfloat16 (uint16 bit patterns) widened a block of rows at a time."""

    def __init__(self, w):
        self.w = w
        self._rows = max(1, WIDEN_ELEMENTS // w.shape[1])

    def __call__(self, x):
        w = self.w
        if w.dtype == np.float32:
            return x @ w.T
        n_out, n_in = w.shape
        out = np.empty((x.shape[0], n_out), np.float32)
        buf = _widen_buffer(self._rows * n_in)
        halves = buf.view(np.uint16)
        for r in range(0, n_out, self._rows):
            m = min(self._rows, n_out - r)
            halves[:2 * m * n_in].reshape(m, n_in, 2)[:, :, 1] = w[r:r + m]
            out[:, r:r + m] = x @ buf[:m * n_in].reshape(m, n_in).T
        return out

    def rows(self, ids):
        """Rows `ids` of the matrix as float32."""
        picked = self.w[ids]
        return picked if picked.dtype == np.float32 else (picked.astype(np.uint32) << 16).view(np.float32)


@dataclass
class _Layer:
    attn_norm: "np.ndarray"
    wqkv: _Matrix
    bqkv: "np.ndarray | None"
    q_norm: "np.ndarray | None"
    k_norm: "np.ndarray | None"
    wo: _Matrix
    ffn_norm: "np.ndarray"
    wgu: _Matrix
    wd: _Matrix


def _rms_norm(x, weight, eps: float):
    return x * (1.0 / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)) * weight


def _plan(gf: GGUFFile):
    """Everything that can be checked without reading the weights; raises UnsupportedModel with the reason if the model cannot run.
    Returns (config, tied, n_params, tensor shapes, tokenizer)."""
    cfg = _read_config(gf.meta, gf.tensors)
    tied = "output.weight" not in gf.tensors or gf.same_data("output.weight", "token_embd.weight")
    n_params = sum(t.n_elements for n, t in gf.tensors.items() if not (tied and n == "output.weight"))
    if n_params > MAX_PARAMS:
        raise UnsupportedModel(f"This model has {n_params / 1e9:.1f} billion parameters, too many for the built-in engine "
                               f"(limit {MAX_PARAMS / 1e9:.1f} billion). {LLAMA_CPP_HINT}")
    shapes = _tensor_names(cfg)
    LoadedModel._check_tensors(gf, shapes)  # noqa: SLF001
    tokenizer = Tokenizer.from_gguf(gf.meta)
    if tokenizer.vocab_size > cfg.n_vocab:
        raise UnsupportedModel("The tokenizer has more tokens than the model's embedding table.")
    return cfg, tied, n_params, shapes, tokenizer


def _resident_params(cfg: _Config, tied: bool, n_params: int) -> int:
    return n_params - (0 if tied else cfg.n_vocab * cfg.n_embd)  # a separate embedding table stays on disk


def inspect_model(path) -> dict:
    """Quickly check whether the built-in engine can run a GGUF file, without reading its weights.

    Returns {"name", "arch", "n_params", "context_length", "vocab_size", "formats", "chat_format", "ram_gb": {"float32", "bfloat16"}}
    (weights only, without the KV cache) or raises UnsupportedModel with a plain-language reason.
    """
    with GGUFFile(path) as gf:
        cfg, tied, n_params, _, tokenizer = _plan(gf)
        resident = _resident_params(cfg, tied, n_params)
        return {"name": str(gf.meta.get("general.name") or Path(path).stem), "arch": cfg.arch, "n_params": n_params,
                "context_length": cfg.context_length, "vocab_size": cfg.n_vocab,
                "formats": sorted({t.type_name for t in gf.tensors.values()}), "chat_format": tokenizer.chat_format,
                "ram_gb": {"float32": round(resident * 4 / 2**30, 2), "bfloat16": round(resident * 2 / 2**30, 2)}}


class LoadedModel:
    """A model in memory. Create sessions with `new_session()`; the model itself is read-only and can be shared."""

    name: str
    arch: str
    n_params: int
    context_length: int
    max_ctx: int
    vocab_size: int
    tokenizer: Tokenizer
    weight_dtype: str
    kv_dtype: str

    def __init__(self, path, max_ctx: int = 2048, n_threads: int = 0, progress=_no_progress, *,
                 weight_dtype: str = "auto", kv_dtype: str = "auto"):
        if np is None:
            raise UnsupportedModel("NumPy is not installed, so the built-in engine is unavailable.")
        if weight_dtype not in ("auto", "float32", "bfloat16") or kv_dtype not in ("auto", "float32", "float16"):
            raise ValueError("weight_dtype must be 'auto', 'float32' or 'bfloat16' and kv_dtype 'auto', 'float32' or 'float16'")
        if sys.byteorder != "little":
            raise UnsupportedModel("The built-in engine needs a little-endian computer.")
        if not isinstance(max_ctx, int) or max_ctx < 8:
            raise ValueError("max_ctx must be an integer of at least 8")
        self._limits = None
        self._gf = None
        self.layers = None
        gf = GGUFFile(path)
        try:
            self.cfg, self.tied, self.n_params, shapes, self.tokenizer = _plan(gf)
            self.name = str(gf.meta.get("general.name") or Path(path).stem)
            self.arch = self.cfg.arch
            self.context_length = self.cfg.context_length
            self.max_ctx = min(max_ctx, self.cfg.context_length)
            self.vocab_size = self.cfg.n_vocab
            self._choose_dtypes(weight_dtype, kv_dtype)
            if n_threads > 0:
                self._limit_threads(n_threads)
            self._load_weights(gf, shapes, progress)
        except BaseException:
            gf.close()
            self.close()
            raise
        if self.tied:
            gf.close()
        else:
            self._gf = gf  # the embedding table is read row by row from the file instead of being kept in RAM
        progress(1.0, "Model ready")

    @classmethod
    def load(cls, path, max_ctx: int = 2048, n_threads: int = 0, progress=_no_progress, *,
             weight_dtype: str = "auto", kv_dtype: str = "auto") -> "LoadedModel":
        """Read a GGUF model. Raises UnsupportedModel (with a plain-language reason) for anything it cannot run.

        `weight_dtype` is "auto", "float32" or "bfloat16" and `kv_dtype` "auto", "float32" or "float16". "auto" keeps float32
        unless the RAM is short: bfloat16 weights take half the memory and run several times slower.
        """
        return cls(path, max_ctx, n_threads, progress, weight_dtype=weight_dtype, kv_dtype=kv_dtype)

    # ------------------------------------------------------------------------------------------------ loading
    @staticmethod
    def _check_tensors(gf: GGUFFile, shapes: dict) -> None:
        for name, shape in shapes.items():
            info = gf.tensors.get(name)
            if info is None:
                raise UnsupportedModel(f"The model file is missing the tensor {name!r}; it is not a standard "
                                       f"{gf.meta.get('general.architecture')} model.")
            if info.shape != shape:
                raise UnsupportedModel(f"Tensor {name!r} has shape {info.shape}, expected {shape}.")
            if not is_supported(info.ggml_type):
                raise UnsupportedModel(f"The model uses the {type_name(info.ggml_type)} format (tensor {name!r}), which the built-in engine "
                                       f"cannot read. It reads {', '.join(SUPPORTED_TYPES)} files. Download a Q8_0 or Q4_K_M version, "
                                       f"or use the llama.cpp engine.")
        out = gf.tensors.get("output.weight")
        if out is not None and (out.shape != shapes["token_embd.weight"] or not is_supported(out.ggml_type)):
            raise UnsupportedModel("The model's output projection has an unsupported shape or format.")
        # tensors we would silently ignore mean this is a variant of the architecture that needs more than we implement
        unknown = [n for n in gf.tensors if n not in shapes and n not in ("output.weight", "rope_freqs.weight")
                   and not n.endswith(".attn_rot_embd.weight")]
        if unknown:
            raise UnsupportedModel(f"This model has parts the built-in engine does not implement ({', '.join(unknown[:3])}"
                                   f"{', …' if len(unknown) > 3 else ''}), so it would answer wrongly. {LLAMA_CPP_HINT}")

    def _choose_dtypes(self, weight_dtype: str, kv_dtype: str) -> None:
        cfg = self.cfg
        kv_elements = 2 * cfg.n_layer * cfg.kv_dim * self.max_ctx
        self.kv_dtype = kv_dtype if kv_dtype != "auto" else ("float32" if kv_elements * 4 <= KV_FLOAT32_LIMIT else "float16")
        kv_bytes = kv_elements * (4 if self.kv_dtype == "float32" else 2)
        resident = _resident_params(cfg, self.tied, self.n_params)
        free = _free_ram_bytes()
        budget = free * RAM_FRACTION
        if weight_dtype == "auto":
            if not free or resident * 4 + kv_bytes <= budget:
                weight_dtype = "float32"
            elif resident * 2 + kv_bytes <= budget:
                weight_dtype = "bfloat16"
            else:
                raise UnsupportedModel(f"This model needs about {(resident * 2 + kv_bytes) / 2**30:.1f} GB of RAM even in half precision, "
                                       f"but only {free / 2**30:.1f} GB is free. Close other programs, pick a smaller or more compressed "
                                       f"model, or use the llama.cpp engine.")
        self.weight_dtype = weight_dtype

    def _limit_threads(self, n_threads: int) -> None:
        try:
            from threadpoolctl import threadpool_limits
            self._limits = threadpool_limits(limits=n_threads, user_api="blas")
        except Exception:  # noqa: BLE001 - without threadpoolctl NumPy keeps its own default
            self._limits = None

    def _load_weights(self, gf: GGUFFile, shapes: dict, progress) -> None:
        cfg, wdt = self.cfg, self.weight_dtype
        stored = np.dtype(np.float32 if wdt == "float32" else np.uint16)
        total, done = len(shapes), 0

        def step(name: str) -> None:
            nonlocal done
            done += 1
            progress(min(done / total, 0.99), f"Loading {name}")

        def fused(*names: str) -> "np.ndarray":
            rows = [shapes[n][0] for n in names]
            out = np.empty((sum(rows), shapes[names[0]][1]), stored)
            at = 0
            for n, r in zip(names, rows):
                gf.load(n, wdt, out=out[at:at + r])
                at += r
                step(n)
            return out

        def vector(name: str) -> "np.ndarray":
            step(name)
            return gf.load(name)

        head_name = "token_embd.weight" if self.tied else "output.weight"  # tied: one matrix is both the embedding and the head
        head = np.empty(shapes["token_embd.weight"], stored)
        gf.load(head_name, wdt, out=head)
        step(head_name)
        self.head = _Matrix(head)
        self.output_norm = vector("output_norm.weight")
        self.layers = []
        for i in range(cfg.n_layer):
            b = f"blk.{i}."
            bias = (np.concatenate([vector(b + f"attn_{x}.bias") for x in "qkv"]) if cfg.qkv_bias else None)
            self.layers.append(_Layer(
                attn_norm=vector(b + "attn_norm.weight"),
                wqkv=_Matrix(fused(b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight")), bqkv=bias,
                q_norm=vector(b + "attn_q_norm.weight") if cfg.qk_norm else None,
                k_norm=vector(b + "attn_k_norm.weight") if cfg.qk_norm else None,
                wo=_Matrix(fused(b + "attn_output.weight")), ffn_norm=vector(b + "ffn_norm.weight"),
                wgu=_Matrix(fused(b + "ffn_gate.weight", b + "ffn_up.weight")), wd=_Matrix(fused(b + "ffn_down.weight"))))
        self._rope_tables(gf)

    def _rope_tables(self, gf: GGUFFile) -> None:
        """cos/sin of every position for each rotated pair; llama.cpp's per-frequency `rope_freqs.weight` (Llama 3.x scaling)
        divides the base frequencies."""
        cfg = self.cfg
        half = cfg.rope_dim // 2
        inv_freq = cfg.rope_base ** (-2.0 * np.arange(half, dtype=np.float64) / cfg.rope_dim)
        if "rope_freqs.weight" in gf.tensors:
            factors = gf.load("rope_freqs.weight")
            if factors.shape != (half,) or not np.all(factors > 0):
                raise UnsupportedModel("The model's rope_freqs tensor does not match its rotary embedding size.")
            inv_freq = inv_freq / factors
        angles = np.outer(np.arange(self.max_ctx, dtype=np.float64) / cfg.rope_scale, inv_freq)
        self.rope_cos, self.rope_sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

    def close(self) -> None:
        """Drop the weights, close the model file and undo the thread limit set at load time."""
        limits, self._limits = getattr(self, "_limits", None), None
        if limits is not None:
            limits.restore_original_limits()
        gf, self._gf = getattr(self, "_gf", None), None
        if gf is not None:
            gf.close()
        for attr in ("head", "layers", "rope_cos", "rope_sin"):
            setattr(self, attr, None)

    def new_session(self) -> "Session":
        return Session(self)

    # ---------------------------------------------------------------------------------------------- forward pass
    def _rope(self, x, cos, sin) -> None:
        """Rotate x (T, heads, head_dim) in place by the angles of its T positions."""
        cfg = self.cfg
        r, half = cfg.rope_dim, cfg.rope_dim // 2
        c, s = cos[:, None, :], sin[:, None, :]
        a, b = (x[..., :half], x[..., half:r]) if cfg.neox else (x[..., 0:r:2], x[..., 1:r:2])
        rotated_a, rotated_b = a * c - b * s, b * c + a * s
        a[...], b[...] = rotated_a, rotated_b

    def _attention(self, session: "Session", layer: int, q, p0: int, mask):
        """q: (T, n_head, head_dim) already rotated and scaled; `mask` hides future keys (None for a single token).
        Returns (T, n_head * head_dim)."""
        cfg = self.cfg
        t, g = q.shape[0], cfg.n_head // cfg.n_kv_head
        end = p0 + t
        k, v = session.k[layer, :, :end], session.v[layer, :, :end]
        if k.dtype != np.float32:
            k, v = k.astype(np.float32), v.astype(np.float32)
        qg = q.reshape(t, cfg.n_kv_head, g, cfg.head_dim).transpose(1, 2, 0, 3).reshape(cfg.n_kv_head, g * t, cfg.head_dim)
        scores = np.matmul(qg, k.transpose(0, 2, 1))
        if mask is not None:
            scores.reshape(cfg.n_kv_head, g, t, end)[...] += mask
        scores -= scores.max(axis=-1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=-1, keepdims=True)
        out = np.matmul(scores, v)
        return out.reshape(cfg.n_kv_head, g, t, cfg.head_dim).transpose(2, 0, 1, 3).reshape(t, cfg.q_dim)

    def _forward(self, session: "Session", tokens, want_logits: bool):
        cfg, t, p0 = self.cfg, len(tokens), session.n_past
        x = self.head.rows(tokens) if self.tied else self._gf.load_rows("token_embd.weight", tokens)
        cos, sin = self.rope_cos[p0:p0 + t], self.rope_sin[p0:p0 + t]
        scale = 1.0 / math.sqrt(cfg.head_dim)
        # a query may only see keys up to and including itself
        mask = np.triu(np.full((t, p0 + t), -np.inf, np.float32), p0 + 1) if t > 1 else None
        with np.errstate(over="ignore", under="ignore"):
            for i, ly in enumerate(self.layers):
                qkv = ly.wqkv(_rms_norm(x, ly.attn_norm, cfg.eps))
                if ly.bqkv is not None:
                    qkv += ly.bqkv
                q = qkv[:, :cfg.q_dim].reshape(t, cfg.n_head, cfg.head_dim)
                k = qkv[:, cfg.q_dim:cfg.q_dim + cfg.kv_dim].reshape(t, cfg.n_kv_head, cfg.head_dim)
                v = qkv[:, cfg.q_dim + cfg.kv_dim:].reshape(t, cfg.n_kv_head, cfg.head_dim)
                if ly.q_norm is not None:
                    q, k = _rms_norm(q, ly.q_norm, cfg.eps), _rms_norm(k, ly.k_norm, cfg.eps)
                self._rope(q, cos, sin)
                self._rope(k, cos, sin)
                q = q * scale
                session.k[i, :, p0:p0 + t] = k.transpose(1, 0, 2)
                session.v[i, :, p0:p0 + t] = v.transpose(1, 0, 2)
                x += ly.wo(self._attention(session, i, q, p0, mask))
                gu = ly.wgu(_rms_norm(x, ly.ffn_norm, cfg.eps))
                gate, up = gu[:, :cfg.n_ff], gu[:, cfg.n_ff:]
                x += ly.wd((gate / (1.0 + np.exp(-gate))) * up)
            if not want_logits:
                return None
            return self.head(_rms_norm(x[-1:], self.output_norm, cfg.eps))[0]


class Session:
    """One conversation's KV cache. Not thread-safe: feed it from one thread at a time."""

    def __init__(self, model: LoadedModel):
        self.model = model
        cfg = model.cfg
        shape = (cfg.n_layer, cfg.n_kv_head, model.max_ctx, cfg.head_dim)
        self.k = np.zeros(shape, model.kv_dtype)  # untouched pages cost no memory until a position is used
        self.v = np.zeros(shape, model.kv_dtype)
        self.n_past = 0

    def feed(self, token_ids) -> "np.ndarray":
        """Append tokens to the context and return the float32 logits [vocab] for the last one."""
        model = self.model
        if model.layers is None:
            raise ValueError("the model has been closed")
        try:
            tokens = np.asarray(token_ids, dtype=np.int64)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("feed() needs a list of integer token ids") from None
        if tokens.ndim != 1 or tokens.size == 0:
            raise ValueError("feed() needs a non-empty list of token ids")
        if tokens.min() < 0 or tokens.max() >= model.vocab_size:
            raise ValueError(f"token ids must be in 0..{model.vocab_size - 1}")
        if self.n_past + tokens.size > model.max_ctx:
            raise ContextFull(f"{self.n_past + tokens.size} tokens do not fit in the {model.max_ctx}-token context window")
        logits = None
        for start in range(0, tokens.size, BATCH):
            chunk = tokens[start:start + BATCH]
            logits = model._forward(self, chunk, start + chunk.size == tokens.size)  # noqa: SLF001
            self.n_past += chunk.size
        if model.tokenizer.vocab_size < model.vocab_size:
            logits[model.tokenizer.vocab_size:] = -np.inf  # padding rows of the embedding table are not real tokens
        return logits

    def truncate(self, n: int) -> None:
        """Forget everything after the first n tokens, so a new prompt sharing that prefix can reuse the cache."""
        if not 0 <= n <= self.n_past:
            raise ValueError(f"cannot truncate to {n}: the session holds {self.n_past} tokens")
        self.n_past = n

    def reset(self) -> None:
        self.n_past = 0
