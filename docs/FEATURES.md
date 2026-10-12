# Feature status

A control existing in the UI is **not** counted as a finished feature. Each row says what is implemented, how it was
validated, and what is missing. Status words: **Implemented & tested** (automated tests exercise it), **Implemented,
unvalidated on target** (written and unit-tested, but never run on the real target — e.g. Windows hardware or a GPU),
**Experimental**, **Unavailable** (not possible with the current engine/hardware, shown as such in the app).

The development sandbox is a 4-vCPU Linux VM without a GPU or Windows. Windows-specific code paths are covered by
unit tests with injected fakes and by GitHub Actions on `windows-latest` (build + smoke start), not by a physical PC.

## VirtualGPU memory manager (`duckstoken/vgpu/`, **Virtual GPU** page)

| Capability (from the project spec) | Status | Where | Notes |
|---|---|---|---|
| Hardware discovery (CPU, RAM, GPU, VRAM, drivers, drives) | Implemented & tested (Linux); Windows paths unvalidated on a PC | `hardware.py`, `vgpu/telemetry.py`, `vgpu/probe.py` | Several sources per fact; anything unreadable is `unavailable`, never `0`. |
| Compatibility report: supported / unsupported / experimental / unknown | Implemented & tested | `vgpu/gpucompat.py`, Diagnostics tab | Nothing is “supported” without evidence (probe of the real engine binary or an unambiguous platform fact). |
| Legacy NVIDIA handling (GTX 750 = Maxwell, CUDA 12 PTX only, R580 driver limit) | Implemented & tested; **not observed on a real card** | `vgpu/gpucompat.py` | Facts verified in llama.cpp's CMake and NVIDIA's deprecation notices. |
| Memory tiers shown separately (VRAM / RAM / mmap storage / disk cache) | Implemented & tested | Memory tab | Tiers are never summed into a “virtual VRAM”. |
| Memory-budget estimation + model admission checks | Implemented & tested | `vgpu/planner.py`, `/api/server/start` gate | Reject, tight, paging, ok; every constant is listed under “Assumptions”; user can force. |
| Disk-space validation | Implemented & tested | planner (downloads/remote models), staging cache | |
| Runtime memory monitoring + pressure detection | Implemented & tested (Linux PSI only where the OS provides it) | `vgpu/sampler.py`, `vgpu/pressure.py` | Levels ok/elevated/high/critical with debouncing; action on critical is a setting (warn / stop big model / stop all). |
| Model unload, failure recovery | Implemented & tested | `vgpu/recovery.py`, `server_proc.py` | Crash watchdog restarts with a safer configuration (opt-in), classifies VRAM/RAM OOM and bad models. |
| Bounded caches, adaptive eviction, hit/miss stats | Implemented & tested | `vgpu/diskcache.py` (`BoundedCache`) | Used for parsed GGUF headers; thrashing detection. |
| Optional disk-backed cache (staging copy on a fast drive) | Implemented & tested with tmp dirs; HDD→SSD **unvalidated on real drives** | `vgpu/staging.py` | Bounded, confirm-required, atomic, only ever deletes files in its own manifest. |
| Predictive prefetch | **Storage level only**: sequential page-cache prewarm | `vgpu/prewarm.py` | Bounded by free RAM and pressure; does not make an HDD fast. |
| GPU weight prefetch / double buffering | **Unavailable** | `vgpu/scheduler.py` capability table | llama.cpp does not expose it. |
| Expert-aware caching (routing statistics, per-expert residency) | **Unavailable** | capability table | `llama-server` exposes no routing data. Replaced by static placement: `--n-cpu-moe`, `-ot`. |
| GPU↔host transfer-bandwidth measurement | **Unavailable** (theoretical PCIe from `nvidia-smi` only) | `vgpu/scheduler.py` | Shown as `Unavailable` in benchmarks. |
| Host memory bandwidth | Implemented (single-thread copy, a lower bound) | `vgpu/telemetry.py` | |
| Bottleneck explanation | Implemented & tested; heuristic thresholds | `vgpu/pressure.py` | Measured signals first; estimate-only fallback has low confidence and says so. |
| Quantization comparison (size, RAM, VRAM, quality *notes*, conversion) | Implemented & tested | `vgpu/quant.py`, Memory tab | No invented quality scores; sizes are estimates from bits/weight. |
| Reproducible benchmark (cold vs warm, PP vs generation speed, TTFT, peak RAM/VRAM, page faults, CPU/GPU use) | Implemented & tested against a fake engine **and a real CPU llama-server** | `vgpu/benchmark.py` | GPU metrics only where `nvidia-smi` exists. |
| Automatic optimizer (max model size / max speed / balanced), baseline first, safe revert | Implemented & tested (fake engine + real CPU engine) | `vgpu/optimizer.py`, `vgpu/scheduler.py` | GPU candidates are only generated when the engine probe lists a usable GPU device. |
| Persisted results, profiles re-evaluated when hardware/model/engine changes | Implemented & tested | `vgpu/store.py` | fresh / stale / missing. |
| Structured performance log (bounded) | Implemented & tested | `vgpu/perflog.py` | Local only. |
| Copyable diagnostic report with secrets redacted | Implemented & tested | `vgpu/diagnostics.py` | |
| GPT-OSS 120B/20B feasibility report | Implemented & tested (arithmetic); **model never loaded here** | `vgpu/gptoss.py` | See [GPT_OSS_120B.md](GPT_OSS_120B.md). Never auto-downloads. |
| Real request telemetry (time to first token, prompt/generation speed) | Implemented & tested | `vgpu/meter.py`, Dashboard | From llama-server `timings`, else client-side (labelled experimental). |

## The rest of the app

| Feature | Status | Notes |
|---|---|---|
| Launcher with ~160 settings, presets, profiles, auto-tune | Implemented & tested | Settings schema drives both CLI flags and UI. |
| Streaming chat, tools, history, export | Implemented & tested | |
| PC access (guarded tools) | Implemented & tested | Off by default; allow/deny folders, confirmations, audit log. |
| Learning: memory, feedback, web research, big-model queue, training-data export | Implemented & tested | Weights never change locally; see README for honest limits. |
| Engines: bundled llama.cpp, external OpenAI-compatible (Ollama/LM Studio), built-in NumPy engine | Implemented & tested | Built-in engine: small dense models only (≈3B and below), far slower than llama.cpp. |
| Native window, tray, autostart | Implemented; unit-tested with fakes; **window/tray not run on a physical Windows PC** | pywebview + pystray; browser fallback. |
| Installer + portable exe + SHA-256 sums | Built and smoke-tested by GitHub Actions on Windows | Unsigned: Windows SmartScreen will warn. |
| Auto-update with verified download | Implemented & tested against a fake GitHub | Size + SHA-256 checked before install. |

## Known limits, stated plainly

* Software cannot turn RAM into VRAM or add GPU compute. The tiers are coordinated, not merged.
* A GTX 750 (1–2 GB) will rarely beat the CPU; the Optimizer exists to measure that instead of assuming it.
* gpt-oss-120b on 48 GB of RAM pages ~20 GB from storage: expect about 1 token/s from an NVMe SSD, and no usable speed from a hard drive.
* Benchmarks in this repository come from a shared 4-vCPU sandbox (see [BENCHMARKS.md](BENCHMARKS.md)), not from the project owner's PC.
