# Hardware and backend compatibility

Statuses use the same words as the in-app **Virtual GPU → Diagnostics → Compatibility** table:

* **Supported** — evidence on this machine (a probe, a measurement) or an unambiguous platform fact.
* **Experimental** — expected to work from documentation, but not validated here; the app tells you how to verify.
* **Unsupported** — will not work (or only through a different backend).
* **Unknown** — cannot be determined without testing on that hardware.

Nothing is marked *Supported* without evidence. Where the project owner's own PC (i7-8700K, 48 GB DDR4-2133,
GTX 750, NVMe/HDD) is concerned, "validated on hardware" is stated explicitly — see the bottom of this page for
what has and has not been run on real machines.

## Inference backends (llama.cpp, bundled)

| Backend | Hardware | Status | Notes |
|---|---|---|---|
| CPU (AVX2) | any x86-64 since ~2013 (Haswell+) | **Supported** | Default and fallback. The bundled Windows build selects the best CPU variant at run time. |
| CPU (no AVX2) | Sandy/Ivy Bridge, old Atom | **Experimental** | Runs through a slower variant; expect a fraction of the speed. |
| CUDA 12.x build | NVIDIA Turing (RTX 20) and newer | **Supported** (when the probe lists the device) | Compute ≥ 7.5. |
| CUDA 12.x build | NVIDIA Maxwell / Pascal / Volta (GTX 750, 9xx, 10xx, Titan V) | **Experimental** | llama.cpp's official CUDA 12 builds ship PTX for compute 5.0/6.1/7.0 (`50-virtual 61-virtual 70-virtual`), which the driver JIT-compiles on first start (slow first launch, cached afterwards). The driver must be R580 or older — R580 is the last branch that supports these GPUs. |
| CUDA 13 build | Maxwell / Pascal / Volta | **Unsupported** | CUDA 13 removed offline compilation and library support for compute < 7.5; llama.cpp's CUDA 13 builds omit those architectures. |
| Vulkan | NVIDIA Maxwell+ (Vulkan 1.1+), AMD GCN+, Intel Arc/Xe | **Experimental** | Works on cards CUDA dropped, usually slower than CUDA; tiny VRAM (1–2 GB) rarely beats the CPU. |
| HIP / ROCm | recent AMD (RDNA2+) | **Unknown** | Needs a ROCm build of llama.cpp (not bundled). |
| SYCL | Intel Arc / Xe | **Unknown** | Needs a SYCL build (not bundled). |
| Metal | Apple silicon | **Unsupported** here | Windows app. |
| External engine | Ollama, LM Studio, any OpenAI-compatible server | **Supported** | Duckstoken talks to it over HTTP; it cannot free that server's memory. |
| Built-in NumPy engine | any x86-64 with NumPy | **Supported** for small dense models | From-scratch, CPU only. About 0.3–3 B parameter dense models; **not** for 20B/120B. |

A GPU backend is shown as *Supported* only when `llama-server --list-devices` actually lists a device of that backend. Everything
else is *Experimental* or *Unknown* with an explanation.

## NVIDIA generations

| Generation | Compute | Examples | CUDA toolkit | Driver | Verdict for LLMs |
|---|---|---|---|---|---|
| Maxwell | 5.0 / 5.2 | GTX 750, 750 Ti, 950–980 | ≤ 12.x | ≤ R580 | Legacy. GTX 750 has only 1–2 GB VRAM, so expect CPU-first inference with little or no GPU benefit. |
| Pascal | 6.0 / 6.1 | GTX 10xx, P100 | ≤ 12.x | ≤ R580 | Legacy but useful with ≥ 6 GB. |
| Volta | 7.0 | Titan V, V100 | ≤ 12.x | ≤ R580 | Legacy. |
| Turing | 7.5 | RTX 20xx, GTX 16xx | 12.x, 13 | current | Fully supported. |
| Ampere / Ada | 8.x | RTX 30xx / 40xx | 12.x, 13 | current | Fully supported. |
| Blackwell | 10.x / 12.x | RTX 50xx | ≥ 12.8 | current | Fully supported. |

## Model formats

| Format | Status | Notes |
|---|---|---|
| GGUF (any llama.cpp quantization) | **Supported** | The only format Duckstoken loads; imports are read-only and validated (header parsing is bounded). |
| MXFP4 MoE (gpt-oss) | **Supported** by builds that contain the `gpt-oss` architecture | Checked by scanning the engine binaries for the architecture string; older builds fail with "unknown model architecture". |
| Safetensors / PyTorch checkpoints | **Unsupported** directly | Needs conversion with llama.cpp's `convert_hf_to_gguf.py` — Duckstoken never converts silently. |
| Sharded GGUF (`-00001-of-0000N`) | **Supported** | All shards must be present. |

## Memory configurations (gpt-oss-120b MXFP4, ~59 GiB file, context 4096)

| System RAM | Result | Why |
|---|---|---|
| 16–32 GB | **Not realistic** | The weights alone are about twice your RAM; paging them from disk makes every token wait on the drive. Use gpt-oss-20b (~12 GB). |
| 48 GB | **Works, slowly** (≈ 1 token/s class on NVMe) | The model exceeds RAM by ~20 GB, so part of it is paged from disk; MoE helps because only 4 of 128 experts are read per token. A hard drive is not usable. |
| 64 GB | **Borderline** | Weights fit, but the OS, KV cache and buffers leave little headroom. |
| 96–128 GB | **Comfortable** | Everything resident; speed is limited by RAM bandwidth (dual-channel DDR4-2133 ≈ 34 GB/s peak). |

These rows are planner output (labelled *estimated* in the app). They are explanations of the arithmetic, not benchmark results.
Run **Virtual GPU → Optimizer** for measurements on your own PC.

## What has been validated, and where

| Item | Validated how |
|---|---|
| Planner, admission, quant tables, GPT-OSS feasibility maths | Unit tests with mocked hardware (clearly named `mock_*`); formulas cross-checked against the model cards. |
| Benchmark and optimizer logic | Unit tests against a fake server **and** an opt-in real-engine test with a CPU `llama-server` built from source on a 4-vCPU Linux sandbox. |
| Windows telemetry (ctypes calls), Windows installer, native window, tray | CI builds and smoke-starts on `windows-latest`; **not** exercised on a physical PC. |
| NVIDIA / Vulkan offload, GTX 750 behaviour | **Not validated on real GPU hardware** — no GPU is available in the development sandbox. The app reports what the driver and engine expose and benchmarks on your machine. |
| gpt-oss-120b actually loading | **Not run** (needs ~60 GB RAM and a 63 GB download); only the arithmetic and file/engine checks are implemented. |
