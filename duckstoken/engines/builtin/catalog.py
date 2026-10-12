"""Small instruct models that the built-in engine can run, as files on Hugging Face.

Every entry was downloaded and run through the engine; the file names and sizes are what the Hugging Face API reports for the
repository. Speeds are what a 4-core CPU managed with float32 weights. The RAM figures are the free memory the engine wants for those
weights: with less it keeps them in half precision, which halves the memory and runs about five times slower.
"""

from __future__ import annotations

_GIB = 2**30

# (label, repo, file, size in bytes, parameters in billions, note)
_MODELS = (
    ("Qwen2.5 0.5B Instruct (Q8_0)", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", 675710816, 0.49,
     "The fastest and lightest choice: about 18 tokens/s and 3 GB of free RAM. Good for short answers and simple tasks."),
    ("Qwen3 0.6B (Q8_0)", "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", 639446688, 0.6,
     "Thinks before it answers, so replies start later. About 14 tokens/s and 4 GB of free RAM."),
    ("Llama 3.2 1B Instruct (Q4_K_M)", "bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", 807694464, 1.24,
     "Meta's small chat model. About 9 tokens/s and 8 GB of free RAM."),
    ("Qwen2.5 1.5B Instruct (Q4_K_M)", "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf", 1117320736, 1.78,
     "A clear step up from the 0.5B model. About 7 tokens/s and 10 GB of free RAM."),
    ("SmolLM2 1.7B Instruct (Q4_K_M)", "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF", "smollm2-1.7b-instruct-q4_k_m.gguf", 1055609536, 1.71,
     "Hugging Face's compact assistant. About 6 tokens/s and 11 GB of free RAM."),
    ("Qwen2.5 3B Instruct (Q4_K_M)", "Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf", 2104932768, 3.4,
     "The best answers here, but slow: about 0.5 tokens/s unless 20 GB of RAM are free, and it needs 10 GB even then."),
)


def recommended_models() -> list[dict]:
    """The catalog as [{"label", "repo", "file", "size_gb", "size_bytes", "params_b", "note"}]; `size_gb` is in GiB."""
    return [{"label": label, "repo": repo, "file": file, "size_gb": round(size / _GIB, 4), "size_bytes": size, "params_b": params,
             "note": note} for label, repo, file, size, params, note in _MODELS]
