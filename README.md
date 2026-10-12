# 🦆 Duckstoken

**Run big local LLMs — like `gpt-oss-120b` — on a modest Windows PC, and keep them useful.**
Duckstoken is a Windows app (installer or single portable `.exe`, with [llama.cpp](https://github.com/ggml-org/llama.cpp)
bundled inside) that tunes ~160 settings for your hardware, checks *before* loading whether a model can really run,
starts the model server, measures what your PC actually achieves, and gives you a chat UI — with optional, heavily-guarded
tools that let the AI work on your PC. You can also point it at Ollama / LM Studio instead, or use its own small built-in engine.

> **Honest by design.** Software cannot turn RAM into VRAM or invent GPU compute. Duckstoken coordinates the memory tiers
> you actually have (VRAM, RAM, memory-mapped storage, an optional fast-drive cache), says what it measured versus what it
> estimated, and says "Unavailable" when it cannot know. See [docs/FEATURES.md](docs/FEATURES.md) for what is implemented, tested,
> experimental or impossible.

## What it does

| | |
|---|---|
| **Hardware scan** | CPU cores/AVX2, RAM size *and speed*, GPU/VRAM, and **which drives are SSD / NVMe / HDD** |
| **"Can my PC run this?"** | Reads the model's GGUF header and tells you honestly how fast (or slow) it will be — and warns when a model would be unusable on your drive |
| **One-click auto-tune** | Picks threads, context, MoE-expert placement (`--n-cpu-moe`), KV-cache quantization, batch sizes… for *your* machine |
| **Downloader** | Hugging Face GGUFs (resumable, multi-part) and the latest official llama.cpp Windows build (CPU / CUDA / Vulkan) |
| **160+ settings** | Every important llama-server option + sampling + reasoning effort, with presets, saved profiles, search, import/export, per-setting reset |
| **Chat UI** | Streaming, gpt-oss reasoning view, tokens/s readout, tool calling, chat history, export |
| **PC access (opt-in)** | List/read/find/write files, run PowerShell commands, fetch URLs, system info — per-tool switches, folder allow-list or *Full PC access*, approval prompts, protected folders, audit log |

## Virtual GPU: memory manager, optimizer and diagnostics

The **Virtual GPU** page (Memory · Optimizer · Diagnostics · GPT-OSS 120B) is Duckstoken's memory and performance layer:

| | |
|---|---|
| **Memory tiers** | VRAM, system RAM, memory-mapped storage and an optional disk cache are shown *separately* — 8 GB of VRAM plus 32 GB of RAM is never presented as 40 GB of VRAM. Live RAM/CPU/GPU/process telemetry with honest `measured` / `estimated` / `unavailable` labels. |
| **“Will it fit?” planner** | Weights, KV cache, compute buffers, OS reserve and a safety margin for any model and context, on your hardware or a **what-if machine** (16 GB laptop, your own PC with more RAM…). Every assumption is listed. Admission verdicts: fits · tight · pages from disk · cannot load — and **▶ Start refuses an impossible load** (you can still force it). |
| **Optimizer + benchmarks** | Measures a CPU-only baseline and a few safe variants (KV type, micro-batch, and GPU offload only if the engine reports a usable GPU) with the same prompt and seed; prompt-processing and generation speed are reported separately, cold vs warm, plus peak RAM/VRAM and page faults. Pick *largest usable model*, *max speed* or *balanced*; nothing is applied until you confirm, and it can be undone. Results are saved and flagged **stale** when your hardware, model or engine changes. |
| **Storage tier** | *Prewarm* reads the model sequentially into the OS file cache (bounded by free RAM); an optional *staging cache* copies a model from a slow drive to your SSD after you confirm. |
| **Crash recovery** | Opt-in: if the model server dies (out of VRAM/RAM) it restarts with a safer configuration. Critical memory pressure can unload the big model or stop the server instead of freezing the PC. |
| **Diagnostics** | Hardware, a compatibility table (supported / unsupported / experimental / unknown — nothing is “supported” without evidence), engine probe, failed estimates, logs and a copyable report with secrets redacted. |
| **GPT-OSS 120B report** | Whether gpt-oss-120b/20b is realistic on your RAM, drive and GPU, with a RAM planning table (16 → 128 GB). It never downloads the 63 GB file without confirmation. |

What llama.cpp does **not** expose — expert routing statistics, per-expert residency, GPU weight prefetch, a transfer-bandwidth probe — is
listed as unavailable in the app rather than faked; the substitutes are static MoE placement (`--n-cpu-moe`) and storage-level prewarm.
Docs: [architecture](docs/ARCHITECTURE.md) · [compatibility matrix](docs/COMPATIBILITY.md) · [feature status](docs/FEATURES.md) ·
[benchmark report](docs/BENCHMARKS.md) · [GPT-OSS 120B feasibility](docs/GPT_OSS_120B.md).

## Learning: make the AI better for *you*

A **Learning** tab collects everything that can make a local model more useful over time. Weights never change on your PC —
Duckstoken improves the *context* the model sees and gathers *training data* you can use for a real fine-tune.

| Feature | What it does |
|---|---|
| **Long-term memory** | Facts/preferences (typed, `remember that…`, suggested after a chat, or saved by the AI with your OK) are searched (SQLite FTS5) and the relevant ones are added to every question. Teach it documents or web pages too. |
| **Feedback loop** | 👍 / 👎 / ✏️ on any answer. Liked and corrected answers are shown to the model as examples for similar questions; 👎 notes become reminders. |
| **Web research** | Optional. DuckDuckGo / Wikipedia / SearXNG / Brave. The AI can call `web_search` / `web_read`, or Duckstoken can search before answering fresh-fact questions and show the sources. SSRF-protected. |
| **Refine 🔁** | Critique-and-rewrite passes with the same model. |
| **Smart prompt** | A short built-in system prompt tuned for small models. |
| **Big-model queue 🐘** | Send a question to your 120B to answer in the background (manual, immediate, or overnight window). It swaps with the everyday model if RAM can't hold both, and can "teacher-review" your recent answers while you sleep. |
| **Training-data sandbox 🎓** | Liked / corrected / big-model answers become reviewable SFT and preference examples. Approve, edit, redact, then export a zip with a LoRA training script for a rented GPU. Load the resulting adapter via *Settings → LoRA adapter*. |

**Honest expectations:** this makes a small model noticeably better at *your* tasks, style and facts — and fresher via web
search — but it will not turn a 20B model into a 120B in general. Model size still sets the ceiling.
Everything is stored locally in `%APPDATA%\Duckstoken\learning.db`; nothing leaves your PC unless you turn on web research
(queries go to the chosen provider) or export and upload training data yourself.

## Install (Windows 10/11)

Download from the [Releases](../../releases) page — pick one:

* **`Duckstoken-Setup.exe`** — the installer. No admin rights needed; adds a Start-menu entry (optional desktop icon and
  "start with Windows"), opens in its own window and keeps the model running from the system tray.
* **`Duckstoken.exe`** — portable single file. Double-click it; nothing is installed.

Both contain everything: a CPU build of the llama.cpp engine is **bundled**, so there is nothing else to download.
(Windows SmartScreen may warn because the app is unsigned → *More info → Run anyway*.)
Updates are checked from GitHub and verified against a published SHA-256 before they are applied.

First run: **Models → download** `gpt-oss-20b` (11 GB, fast) or `gpt-oss-120b` (59 GB, needs a fast SSD) →
**Dashboard → ⚡ Auto-tune → ▶ Start** → **Chat**.

### Engines — you are not tied to llama.cpp

| Engine | What it is | Use it when |
|---|---|---|
| **llama.cpp (bundled)** | The default. Fast native engine; the only one here that can run 59 GB models from RAM + SSD. Optional CUDA/Vulkan builds can be installed from the app. | Almost always |
| **External server** | Talks to Ollama, LM Studio, vLLM or any OpenAI-compatible server you already run. | You already use one of those |
| **Built-in (pure Python + NumPy)** | Duckstoken's own inference engine for small GGUF models (≈3B and below). No other program involved. | You want zero dependencies; small models only. It is far slower than llama.cpp and cannot run gpt-oss. |

From source: `python -m duckstoken` (Python 3.10+; the standard library is enough, `numpy` enables the built-in engine,
`pywebview` + `pystray` + `pillow` enable the native window and tray) or double-click `run_windows.bat`.

## Running `gpt-oss-120b` on 48 GB RAM, an i7-8700K and a GTX 750

It works — slowly — **if the model lives on an SSD** (the numbers below are the planner's estimates; see
[docs/GPT_OSS_120B.md](docs/GPT_OSS_120B.md) for the arithmetic and **Virtual GPU → GPT-OSS 120B** for your own machine):

* The 120B model is ≈ 59 GiB; after the 6 GB Duckstoken keeps free for Windows you have ~42 GB of usable RAM, so ≈ 20 GB (a third of the
  weights) is read from disk *while generating*. llama.cpp memory-maps the file, and the model is a Mixture-of-Experts that only touches
  4 of 128 experts per token, which is why this works at all.
* **Store it on the 256 GB M.2.** Expect roughly **0.6–1.4 tokens/s** on NVMe, less on a SATA SSD.
  On a **hard drive it is unusable** (minutes per word) — the planner refuses it unless you force the load.
* `gpt-oss-20b` (≈ 11 GiB) fits entirely in RAM → about **4.7–9.4 tokens/s** (estimate) on that CPU. Best for daily use.
* The GTX 750 (1–2 GB VRAM, Maxwell) can hold only the attention weights and helps very little; auto-tune uses the CPU. If you want to try
  the GPU anyway, install a CUDA **12.x** build (CUDA 13 builds dropped Maxwell) or Vulkan, and let the Optimizer measure whether it helps.
* Turn on **XMP/DOCP** in BIOS and use both memory channels: generation speed is memory-bandwidth bound.
* 64 GB of RAM would nearly remove the paging; 96–128 GB keeps the whole model resident.

> These are estimates from published model facts and a memory-bandwidth model, not measurements of your PC. The Dashboard shows the
> measured speed of your last reply, and the Optimizer benchmarks properly.

## PC access — read this before enabling

Everything is **off by default**. Under **PC access** you choose which tools exist, which folders they may touch
(or *Full PC access*), and which actions need your approval. Writes and commands **ask first by default** (the dialog shows the
exact text that would be written, and never offers "always allow" for writes, commands or secrets), web tools ask first by default,
system folders are write-protected, and every call is appended to `tool_audit.jsonl`.

Two protections you cannot switch off: Duckstoken's own data folder (it holds your settings and API keys) is unreadable and unwritable
for the AI, and anything that looks like credentials — `.ssh`, `.aws`, browser profiles, `.env`, key files — needs an approval every time,
whatever the "ask first" switches say. Importing a settings file or loading a profile that grants PC access, turns off approvals or changes
the program Duckstoken launches asks you to confirm, and exported settings never include API keys.

Be aware: text the model reads (a web page, a document, a repo) can contain instructions that try to make it misuse tools
(prompt injection). Commands are **not sandboxed** — they run with your Windows permissions. Keep the confirmations on.
See [docs/SECURITY_REVIEW.md](docs/SECURITY_REVIEW.md) for what an independent review of this code found.

## Security of the app itself

* The UI/API listens on `127.0.0.1` only, requires a per-launch secret token, rejects foreign `Host` headers (DNS-rebinding)
  and non-JSON POSTs (cross-site forms).
* Downloads are restricted to Hugging Face / GitHub hosts; archives are extracted with zip-slip protection.
* If you change the model server's bind address, **set an API key**.

## Build it yourself

```bat
build_windows.bat
```
builds the portable exe (engine bundled). The GitHub Action (`release.yml`) additionally builds the Inno Setup installer,
smoke-tests both (the installed app is launched and its UI fetched), writes `SHA256SUMS.txt`, and publishes: pushes to `main`
update the rolling **Latest build** release; pushing a tag such as `v0.2.0` publishes a versioned release.

## Development

```bash
python -m unittest discover -s tests -v   # ~1600 tests: backend, learning, engines, Virtual GPU, HTTP API, fake and (opt-in) real llama-server
python -m duckstoken --no-browser         # then open the printed URL
```

Layout: `duckstoken/settings.py` (schema = single source of truth for CLI flags *and* the UI),
`hardware.py`, `gguf.py`, `tuner.py`, `downloads.py`, `server_proc.py`, `tools.py`, `app.py`, `vgpu/` (planner, telemetry, benchmark, optimizer, caches), `engines/`, `web/`.
Design notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

MIT licensed. Not affiliated with OpenAI or the llama.cpp project.
