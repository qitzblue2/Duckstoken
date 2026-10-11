# 🦆 Duckstoken

**Run big local LLMs — like `gpt-oss-120b` — on a modest Windows PC.**
Duckstoken is a one-file Windows app (no installer, no dependencies) that downloads
[llama.cpp](https://github.com/ggml-org/llama.cpp) and your model, **tunes ~85 settings for your hardware**,
starts the model server, and gives you a chat UI — with optional, heavily-guarded tools that let the AI work on your PC.

![dashboard](docs/dashboard.png)

## What it does

| | |
|---|---|
| **Hardware scan** | CPU cores/AVX2, RAM size *and speed*, GPU/VRAM, and **which drives are SSD / NVMe / HDD** |
| **"Can my PC run this?"** | Reads the model's GGUF header and tells you honestly how fast (or slow) it will be — and warns when a model would be unusable on your drive |
| **One-click auto-tune** | Picks threads, context, MoE-expert placement (`--n-cpu-moe`), KV-cache quantization, batch sizes… for *your* machine |
| **Downloader** | Hugging Face GGUFs (resumable, multi-part) and the latest official llama.cpp Windows build (CPU / CUDA / Vulkan) |
| **85+ settings** | Every important llama-server option + sampling + reasoning effort, with presets, saved profiles, search, import/export, per-setting reset |
| **Chat UI** | Streaming, gpt-oss reasoning view, tokens/s readout, tool calling, chat history, export |
| **PC access (opt-in)** | List/read/find/write files, run PowerShell commands, fetch URLs, system info — per-tool switches, folder allow-list or *Full PC access*, approval prompts, protected folders, audit log |

## Quick start (Windows 10/11)

1. Download `Duckstoken.exe` from the [Releases](../../releases) page and run it. (SmartScreen may warn: it's an unsigned open-source app → *More info → Run anyway*.)
2. **Models → Engine → Find latest build → Install.**
3. **Models → download** `gpt-oss-20b` (11 GB, fast) or `gpt-oss-120b` (59 GB, needs a fast SSD).
4. **Dashboard → ⚡ Auto-tune → ▶ Start**, then open **Chat**.

From source: `python -m duckstoken` (Python 3.10+, standard library only) or double-click `run_windows.bat`.

## Running `gpt-oss-120b` on 48 GB RAM, an i7-8700K and a GTX 750

It works — slowly — **if the model lives on an SSD**:

* The 120B model is ~59 GB; you have ~42 GB of usable RAM, so ~15 GB is read from disk *while generating*
  (llama.cpp memory-maps the file, and the model is a Mixture-of-Experts that only touches ~4 of 128 experts per token).
* **Store it on the 256 GB M.2.** It fits (59 GB). Expect roughly **0.5–2 tokens/s** on NVMe, less on a SATA SSD.
  On a **hard drive it is unusable** (minutes per word) — Duckstoken detects this and warns you.
* `gpt-oss-20b` (11 GB) fits entirely in RAM → **~4–7 tokens/s** on that CPU. Best for daily use.
* The GTX 750 (1–2 GB VRAM, Maxwell) helps very little; auto-tune uses the CPU backend by default. If you want to try the GPU anyway, pick the Vulkan or CUDA 12.x build.
* Turn on **XMP/DOCP** in BIOS and use both memory channels: generation speed is memory-bandwidth bound.

> The speed numbers are estimates from a simple model of RAM/disk bandwidth. The chat footer shows your **real** tokens/s.

## PC access — read this before enabling

Everything is **off by default**. Under **PC access** you choose which tools exist, which folders they may touch
(or *Full PC access*), and which actions need your approval. Writes and commands **ask first by default**,
system folders are write-protected, and every call is appended to `tool_audit.jsonl`.

Be aware: text the model reads (a web page, a document, a repo) can contain instructions that try to make it misuse tools
(prompt injection). Commands are **not sandboxed** — they run with your Windows permissions. Keep the confirmations on.

## Security of the app itself

* The UI/API listens on `127.0.0.1` only, requires a per-launch secret token, rejects foreign `Host` headers (DNS-rebinding)
  and non-JSON POSTs (cross-site forms).
* Downloads are restricted to Hugging Face / GitHub hosts; archives are extracted with zip-slip protection.
* If you change the model server's bind address, **set an API key**.

## Build the exe yourself

```bat
build_windows.bat
```
or push a tag (`git tag v0.1.0 && git push --tags`) and the GitHub Action builds, smoke-tests and publishes `Duckstoken.exe`.

## Development

```bash
python -m unittest discover -s tests -v   # backend, tools, security, fake llama-server lifecycle
python -m duckstoken --no-browser         # then open the printed URL
```

Layout: `duckstoken/settings.py` (schema = single source of truth for CLI flags *and* the UI),
`hardware.py`, `gguf.py`, `tuner.py`, `downloads.py`, `server_proc.py`, `tools.py`, `app.py`, `web/`.

MIT licensed. Not affiliated with OpenAI or the llama.cpp project.
