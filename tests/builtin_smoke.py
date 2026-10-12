#!/usr/bin/env python3
"""Smoke test of the built-in engine on a real GGUF file: loads it, decodes greedily and reports speed and memory.

    python tests/builtin_smoke.py path/to/model.gguf [--tokens 48] [--ctx 2048] [--threads 0] [--weights auto|float32|bfloat16]

Run it by hand on a small instruct model (for example Qwen2.5-0.5B-Instruct Q8_0 or Q4_K_M). The text should be coherent
("The capital of France is" -> " Paris ..."); that is the proof that weights, RoPE and tokenizer are right. It is not part of the
unit-test suite because it needs a model download.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckstoken.engines.builtin.gguf_weights import UnsupportedModel  # noqa: E402
from duckstoken.engines.builtin.model import LoadedModel  # noqa: E402
from duckstoken.engines.builtin.sampler import Sampler  # noqa: E402


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process in MB (includes the memory-mapped model file), or None if unknown."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except ImportError:
        pass
    try:  # Windows
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 2**20 if ok else None
    except (AttributeError, OSError):
        return None


def generate(model: LoadedModel, prompt_ids: list[int], max_tokens: int) -> tuple[str, dict]:
    """Greedy decoding. Returns the text and {prompt_n, prefill_tps, new_n, decode_tps}."""
    tok = model.tokenizer
    session, sampler, stream = model.new_session(), Sampler(temperature=0), tok.stream_decoder()
    session.feed(prompt_ids[:1])  # warm-up: the first pass pays for thread start-up and page faults
    session.reset()
    start = time.perf_counter()
    logits = session.feed(prompt_ids)
    prefill = time.perf_counter() - start
    history, text, produced = list(prompt_ids), [], 0
    start = time.perf_counter()
    while produced < max_tokens and session.n_past < model.max_ctx:
        token = sampler.sample(logits, history)
        if token in tok.eos_ids:
            break
        history.append(token)
        produced += 1
        text.append(stream.push(token))
        logits = session.feed([token])
    decode = time.perf_counter() - start
    text.append(stream.flush())
    return "".join(text), {"prompt_n": len(prompt_ids), "prefill_tps": len(prompt_ids) / prefill if prefill else 0.0,
                           "new_n": produced, "decode_tps": produced / decode if decode else 0.0}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # a legacy Windows console cannot show every character the model writes
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--tokens", type=int, default=48, help="tokens to generate per prompt")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=0, help="BLAS threads (0 = library default; needs the threadpoolctl package)")
    ap.add_argument("--weights", default="auto", choices=("auto", "float32", "bfloat16"))
    ap.add_argument("--prompt", action="append", help="extra raw prompt to complete (repeatable)")
    args = ap.parse_args(argv)

    start = time.perf_counter()
    shown = [-1]

    def progress(fraction: float, message: str) -> None:
        if int(fraction * 10) > shown[0]:
            shown[0] = int(fraction * 10)
            print(f"  loading {fraction:4.0%}", flush=True)

    try:
        model = LoadedModel.load(args.gguf, max_ctx=args.ctx, n_threads=args.threads, weight_dtype=args.weights, progress=progress)
    except UnsupportedModel as e:
        print(f"Cannot run this model: {e}")
        return 2
    print(f"{model.name}: {model.arch}, {model.n_params / 1e6:.0f}M parameters, {model.cfg.n_layer} layers, vocab {model.vocab_size}, "
          f"weights {model.weight_dtype}, KV cache {model.kv_dtype}, context {model.max_ctx} "
          f"(loaded in {time.perf_counter() - start:.1f}s)")
    tok = model.tokenizer
    jobs = [("completion", p) for p in ["The capital of France is"] + (args.prompt or [])]
    jobs.append(("chat", tok.apply_chat_template([{"role": "user", "content": "Name three primary colors, then say hello in French."}])))
    status = 0
    for kind, prompt in jobs:
        ids = tok.encode(prompt, add_bos=tok.add_bos, parse_special=kind == "chat")
        text, stats = generate(model, ids, args.tokens)
        print(f"\n[{kind}] {prompt!r}\n  -> {text!r}")
        print(f"  prefill {stats['prompt_n']} tokens at {stats['prefill_tps']:.1f} tok/s, "
              f"decode {stats['new_n']} tokens at {stats['decode_tps']:.2f} tok/s")
        status |= not text.strip()
    peak = peak_memory_mb()
    print(f"\npeak memory: {peak:.0f} MB" if peak else "\npeak memory: unknown")
    model.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
