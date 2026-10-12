# Duckstoken architecture

Duckstoken is a local app that runs large language models on ordinary Windows PCs and keeps getting more useful the
longer you use it. It is a single Python process (standard library only, plus optional NumPy/Pillow/pywebview/pystray
for the built-in engine and the native window) that serves a web UI on `127.0.0.1` and supervises a separate
inference-engine process.

> **Stack note.** An early design proposal for the memory manager ("VirtualGPU") suggested Tauri + React + Rust.
> Duckstoken deliberately stays on Python + vanilla JS: one dependency-free codebase that the project owner can run
> from source, freeze with PyInstaller into a single `.exe`, and extend without a Node/Rust toolchain. Everything
> that needs a native shell (window, tray, installer, autostart) is isolated in `duckstoken/shell.py` and
> `installer/`, so a Tauri shell could replace it later without touching the engine, memory or UI layers.

## Process model

```
 Duckstoken.exe  (one Python process)                         separate OS processes
 ┌──────────────────────────────────────────────┐
 │ native window / tray (shell.py)              │
 │ HTTP server 127.0.0.1:8765 (app.py)          │          ┌────────────────────────┐
 │   token + Host check + JSON content-type     │  OpenAI  │ llama-server (bundled) │ ← chat model
 │   web UI (duckstoken/web/*)                  ├─────────►│ llama-server (bundled) │ ← optional big model
 │ engines/  llamacpp | external | builtin      │   HTTP   │ Ollama / LM Studio …   │ ← external engine
 │ learning/ memory, feedback, dataset, queue   │          └────────────────────────┘
 │ vgpu/     planner, telemetry, benchmark …    │
 └──────────────────────────────────────────────┘
```

* The UI never talks to an engine directly; `app.py` proxies chat (`/api/chat`) so it can add memory, web context and
  telemetry, and so the per-launch token protects the API from other web pages.
* The model server is a child process bound to a Windows Job Object: if Duckstoken dies, the OS kills it too.
* Engines are interchangeable behind `EngineSwitch` (`engines/__init__.py`): `llamacpp` (bundled `llama-server`),
  `external` (any OpenAI-compatible server) and `builtin` (a from-scratch NumPy transformer for small models).

## Layers

| Layer | Modules | Notes |
|---|---|---|
| Shell | `shell.py`, `updater.py`, `installer/duckstoken.iss`, `tools/` | native window (WebView2), tray, autostart, verified self-update, installer |
| HTTP/API | `app.py`, `api_learning.py`, `api_system.py`, `api_vgpu.py` | one `api_<path>` method per endpoint; POST endpoints are marked with `@post` |
| Engines | `server_proc.py`, `engines/` | process lifecycle, argument building from the settings schema, health, logs |
| Settings | `settings.py` | one schema drives the CLI flags *and* the settings UI |
| Learning | `memory.py`, `feedback.py`, `websearch.py`, `escalation.py`, `dataset.py`, `context.py`, `db.py` | memory, feedback, big-model queue, training data, web grounding |
| Hardware & memory | `hardware.py`, `gguf.py`, `tuner.py`, **`vgpu/`** | discovery, model metadata, placement planning, telemetry, benchmarking |
| UI | `web/*.js`, `web/*.css` | no framework, no build step |

## VirtualGPU: the memory/performance layer (`duckstoken/vgpu/`)

VirtualGPU coordinates the memory tiers of the machine **without pretending they are one pool**:

| Tier | What lives there | How Duckstoken influences it | What it cannot do |
|---|---|---|---|
| 1 GPU VRAM | layers/tensors offloaded with `-ngl`, KV cache if `kv_offload` | planner picks `n_gpu_layers`, `--n-cpu-moe`, `-ot`, KV type | cannot enlarge VRAM or make a GTX 750 run modern kernels |
| 2 System RAM | everything not in VRAM; compute buffers; page cache | budgets (`os_reserve`, safety margin), `mlock`, admission checks | cannot be "fused" with VRAM; PCIe is the only bridge |
| 3 Memory-mapped storage | weights beyond RAM, paged on demand by the OS | `--mmap`, spill estimate, sequential **prewarm** into the page cache | page faults are 10–1000× slower than RAM (HDD worst) |
| 4 Disk-backed cache | an optional *staging copy* of a model on a fast drive | `StagingCache` (bounded, atomic, only ever deletes its own files) | does not speed up a model that already sits on an SSD |

```
 hardware.get_hardware() ─┐
 gguf.summarize(model) ───┼─► planner.plan() ─► {tiers, placement, requirements, admission, speed, assumptions}
 settings / assumptions ──┘            │
                                       ├─► admission gate in /api/server/start (enforce | warn | off)
                                       ├─► scheduler.candidate_configs()  (CPU-only baseline first)
                                       └─► optimizer.run_optimization() ──► benchmark.run_benchmark() ─► store (profiles, history)

 telemetry.Sampler ─► pressure.Monitor ─► (warn | stop big model | stop all)         recovery.CrashWatchdog ─► safer restart
 probe.probe_engine ─► gpucompat.compatibility_report ─► diagnostics.build_report
```

Honesty rules enforced by the code (see `vgpu/values.py`): every number carries a status — `measured`, `estimated`,
`experimental` or `unavailable` — and an unavailable source yields `unavailable`, never `0` or a guess. Benchmarks
record the exact settings, engine build, hardware fingerprint and prompt hash, and never invent a result.

### Backend capabilities (what llama.cpp does and does not expose)

`vgpu/scheduler.py` keeps an explicit capability registry. Static placement (`-ngl`, `--n-cpu-moe`, `-ot`), KV
placement/quantization, mmap/mlock and the op-offload switch are available. **Not available** in `llama-server`:
expert routing/frequency statistics, per-expert dynamic residency, host↔GPU prefetch or double-buffering controls,
and a transfer-bandwidth probe. The "expert-aware cache" and "predictive prefetch" ideas are therefore *not*
implemented as fake features; their supported substitutes are static MoE placement and storage-level sequential
prewarm. If a future llama.cpp exposes these hooks, add them to the registry and the scheduler.

## Security model

* API bound to loopback; random per-launch token required in `X-Duck-Token`; `Host` header must be loopback;
  mutating endpoints require `Content-Type: application/json` (blocks cross-site form posts).
* No telemetry leaves the machine. Network use is explicit: Hugging Face downloads (confirmed by the user), web search
  (opt-in), update checks (GitHub API only; verified SHA-256 before any install).
* Model files are untrusted input: GGUF headers are parsed with bounds, paths are validated, nothing from a model file
  is executed. Subprocesses use argv lists, never shell strings.
* PC-access tools are off by default, gated separately from web tools, with allow/deny folders and an audit log.
* Deleting or overwriting user data (models, staged copies, profiles) always needs an explicit confirmation, and the
  staging cache only ever removes files recorded in its own manifest.

## Data locations (`%APPDATA%\Duckstoken` or `~/.duckstoken`, override with `DUCKSTOKEN_HOME`)

`settings.json`, `profiles.json`, `learning.db` (SQLite: memory, exchanges, queue, dataset), `llama-server*.log`,
`tool_audit.jsonl`, `vgpu/` (benchmarks, optimizer history, learned profiles, performance log, staging cache),
`update_state.json`, `updates/`.
