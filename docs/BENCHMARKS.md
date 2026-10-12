# Benchmark reports

**Read this first.** Every number below was measured in the development sandbox — a shared 4-vCPU Intel Xeon @ 2.1 GHz
(AVX2 + AVX-512) Linux VM with 15.7 GB of RAM and **no GPU** — running a CPU-only `llama-server` built from llama.cpp
commit `08ffba2`. It is **not** the project owner's i7-8700K / GTX 750 PC, and the VM is shared with other work, so run-to-run
noise is large (the same configuration measured 17–36 tokens/s on different runs). These tables document *how the
benchmark works and that it produces real numbers*; they say nothing about your machine. Run the benchmark on your PC.

## What the benchmark measures

`duckstoken/vgpu/benchmark.py` starts its **own** `llama-server` (never the chat server) on a free port with the exact
settings under test and records:

| Field | How it is obtained |
|---|---|
| Model load time | process start → `/health` OK |
| Time to first token | client-side clock on a streamed request |
| Prompt-processing speed (tokens/s) | llama-server `timings.prompt_per_second` (else client side, labelled) |
| Generation speed (tokens/s) | llama-server `timings.predicted_per_second` (else client side, labelled) |
| Cold vs warm | cold = first request after process start; warm = mean ± stdev of N further requests (one extra warm-up is discarded). The OS page cache is **not** purged, and the record says so. |
| Peak RAM | `VmHWM` / peak working set of the server process |
| Peak VRAM, GPU utilization | `nvidia-smi` when present, otherwise *Unavailable* |
| CPU utilization, page faults, disk reads | OS counters for the system / server process |
| GPU↔host transfer bandwidth | *Unavailable* — llama-server does not expose it |
| Planner estimate vs measured buffers | parsed from the server's load log (`-lv 4`) and compared in GiB |

Fixed for reproducibility: the same prompt (sha256 recorded), `temperature = 0`, fixed seed, `cache_prompt = false`,
one slot, identical settings hash. Failures, timeouts and cancellations are recorded as such; nothing is invented.

## Results from the sandbox (CPU only, `-ngl 0`, context 2048, 512-token prompt, 64 generated tokens, 3 warm runs)

| Model | Load | Cold: prompt / generation | Warm: prompt (t/s) | Warm: generation (t/s) | TTFT (warm) | Peak RSS | Planner estimate (t/s) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_K_M (0.46 GiB) | 1.1 s | 187.6 / 22.7 | 238.2 ± 20.6 | 24.1 ± 1.8 | 2.2 s | 0.74 GB | 20.0–39.9 (measured inside the range) |
| Llama-3.2-1B-Instruct Q4_K_M (0.75 GiB) | 2.0 s | 265.7 / 15.4 | 270.4 ± 3.6 | 15.8 ± 0.7 | 2.0 s (cold) | 1.71 GB | 10.3–20.6 (measured inside the range) |

A second run through the web UI (**Virtual GPU → Optimizer → Benchmark**, 128-token prompt, 32 generated, 2 warm runs) on
the 0.5B model: cold 213 / 28.0 t/s, warm 195 / 27.2 ± 1.9 t/s, load 1.3 s, peak RAM 0.7 GB, CPU 96 %, GPU *Unavailable*.

### What the comparison taught us (kept as limitations)

* On this AVX-512/AMX CPU llama.cpp holds a *repacked* copy of Q4 weights next to the mapped file: measured weights were
  **1.50 GiB for a 0.75 GiB file** (Llama-3.2-1B). The planner's RAM estimate for CPU-only runs can therefore be up to ~2× low
  on such CPUs; the benchmark shows the delta instead of hiding it. The i7-8700K has neither AVX-512 nor AMX.
* Buffer sizes appear in the server log only at verbosity 4, so the benchmark adds `-lv 4` unless you set a verbosity yourself.
* llama-server defaults to 4 slots; the benchmark forces one so the figures are not split.

### Optimizer on the 0.5B model (256-token prompt, 32 generated, 2 warm runs)

| Candidate | Measured generation (t/s) |
|---|---|
| CPU-only baseline | 27.6 |
| KV cache q8_0 (**chosen**) | 33.4 (× 1.21) |
| Micro-batch 256 | 25.5 |

The same baseline configuration measured between 17.2 and 36.0 t/s in other runs in this shared VM, which is why the
optimizer records standard deviation and run count, treats results within 3 % as ties (preferring the leaner
configuration), and reports `vs_baseline.distinguishable` — whether the winning margin exceeds twice the run-to-run spread.
Use at least 3 repeats on a quiet machine.

## Reproduce

```bash
# build the engine (CPU)
git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server -j 4
# run the opt-in real-engine tests against any small GGUF
DUCK_REAL_LLAMA=$PWD/build/bin/llama-server DUCK_REAL_MODEL=/path/to/model.gguf \
  python -m unittest tests.test_vgpu_benchmark.TestRealEngine
```

In the app: **Virtual GPU → Optimizer** (a confirmation explains what will be loaded and stopped first).

## Not yet measured anywhere

* Any run with a GPU (CUDA/Vulkan), `--no-op-offload`, peak VRAM, PCIe behaviour, GTX 750.
* Any MoE model and gpt-oss (needs ≥ 12 GB for the 20B, ≥ 59 GB for the 120B).
* Windows: process launch, telemetry counters and drive detection are exercised by unit tests and CI smoke tests only.
