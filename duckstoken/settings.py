"""Settings schema (single source of truth), persistence, presets and profiles.

Every setting is declared once in SCHEMA. The web UI renders itself from this list and
`server_proc.build_args` turns the entries that carry a `flag` into llama-server arguments.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from . import paths

GROUPS = [
    ("launcher", "Launcher"),
    ("server", "Server (llama.cpp)"),
    ("memory", "Memory & offload"),
    ("cpu", "CPU & performance"),
    ("sampling", "Sampling (applies live)"),
    ("reasoning", "Reasoning & chat"),
    ("pc", "PC access (tools)"),
    ("learning", "Learning & memory"),
    ("web", "Web & research"),
    ("big", "Big model (escalation)"),
    ("vgpu", "Virtual GPU & memory manager"),
    ("engine", "Engine"),
    ("app", "App & updates"),
]

CACHE_TYPES = ["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl", "bf16", "f32"]

if os.name == "nt":
    _BLOCKED = [
        r"C:\Windows",
        r"C:\Program Files",
        r"C:\Program Files (x86)",
        r"C:\ProgramData\Microsoft",
        r"C:\$Recycle.Bin",
        r"C:\System Volume Information",
    ]
else:
    _BLOCKED = ["/etc", "/boot", "/sys", "/proc", "/dev", "/usr", "/bin", "/sbin", "/lib"]


def _s(key, label, group, typ, default, help="", **extra) -> dict:
    d = {"key": key, "label": label, "group": group, "type": typ, "default": default, "help": help}
    d.update(extra)
    return d


SCHEMA: list[dict] = [
    # ---------------------------------------------------------------- launcher
    _s("models_dir", "Models folder", "launcher", "path", "",
       "Where downloaded models live. Put this on your fastest drive (SSD/NVMe!). Empty = ~/Duckstoken/models.",
       kind="dir"),
    _s("llama_dir", "Engines folder", "launcher", "path", "",
       "Where llama.cpp builds are installed. Empty = Duckstoken data folder.", kind="dir"),
    _s("backend_preference", "Preferred engine backend", "launcher", "select", "auto",
       "Which llama.cpp build the installer picks. 'auto' picks CUDA/Vulkan if a GPU with enough VRAM is found, otherwise CPU.",
       options=["auto", "cpu", "cuda", "vulkan"]),
    _s("hf_token", "Hugging Face token", "launcher", "password", "",
       "Optional. Only needed for gated / private repos."),
    _s("open_browser", "Open the UI in my browser on start", "launcher", "bool", True),
    _s("autostart_server", "Start the model server when Duckstoken starts", "launcher", "bool", False),
    _s("stop_server_on_exit", "Stop the model server when Duckstoken exits", "launcher", "bool", True),
    _s("process_priority", "Model server process priority (Windows)", "launcher", "select", "normal",
       "Higher priority can help speed slightly but may make the PC feel sluggish while the model runs.",
       options=["idle", "below_normal", "normal", "above_normal", "high"]),
    _s("theme", "Theme", "launcher", "select", "dark", options=["dark", "light"]),
    _s("hardware_poll_seconds", "Dashboard refresh (seconds)", "launcher", "int", 5, min=2, max=120),

    # ------------------------------------------------------------------ server
    _s("llama_server_path", "llama-server executable", "server", "path", "",
       "Path to llama-server(.exe). The Models tab can download and set this for you.", kind="file"),
    _s("model_path", "Model file (.gguf)", "server", "path", "",
       "For split models pick the first shard (…-00001-of-0000N.gguf).", kind="file", flag="-m"),
    _s("host", "Bind address", "server", "str", "127.0.0.1",
       "Keep 127.0.0.1 unless you really want other devices on your network to use the model.", flag="--host"),
    _s("port", "Server port", "server", "int", 8080, min=1, max=65535, flag="--port"),
    _s("api_key", "API key", "server", "password", "",
       "If set, clients must send it. STRONGLY recommended if you change the bind address.", flag="--api-key"),
    _s("parallel", "Parallel slots", "server", "int", 1, min=1, max=64, flag="-np",
       help="Concurrent requests. Each slot gets ctx/slots of the context, so keep 1 on small machines."),
    _s("cont_batching", "Continuous batching", "server", "bool", True, flag="--no-cont-batching", when=False),
    _s("jinja", "Use the model's chat template (--jinja)", "server", "bool", True,
       "Required for tool calling and gpt-oss reasoning channels.", flag="--jinja", when=True),
    _s("reasoning_format", "Reasoning format", "server", "select", "auto",
       "How llama-server returns the model's thinking.", options=["auto", "none", "deepseek"],
       flag="--reasoning-format", skip_if=["auto"]),
    _s("chat_template", "Chat template override", "server", "str", "", flag="--chat-template", advanced=True),
    _s("server_timeout", "Request timeout (seconds)", "server", "int", 3600, min=10, max=86400,
       flag="--timeout", advanced=True),
    _s("builtin_webui", "Enable llama.cpp's own web UI", "server", "bool", True,
       "Off = API only (Duckstoken's chat still works).", flag="--no-webui", when=False, advanced=True),
    _s("metrics", "Prometheus /metrics endpoint", "server", "bool", False, flag="--metrics", when=True,
       advanced=True),
    _s("verbose", "Verbose server log", "server", "bool", False, flag="-v", when=True, advanced=True),
    _s("extra_args", "Extra llama-server arguments", "server", "text", "",
       "Raw arguments appended as-is, e.g.  --draft-max 8 --no-warmup", advanced=True),

    # ------------------------------------------------------------------ memory
    _s("ctx_size", "Context size (tokens)", "memory", "int", 8192, min=0, max=1048576, flag="-c",
       help="Bigger = more RAM for the KV cache. 0 = model default (can be huge!)."),
    _s("n_gpu_layers", "GPU layers", "memory", "int", 99, min=-1, max=999, flag="-ngl",
       help="99 = offload everything that fits. With a tiny GPU use a small number or 0."),
    _s("n_cpu_moe", "MoE layers whose experts stay on CPU", "memory", "int", 0, min=0, max=999,
       flag="--n-cpu-moe", skip_if=[0],
       help="THE key setting for gpt-oss-120b on a small GPU: keeps the huge expert weights in system RAM. Set to the number of layers (36 for 120b, 24 for 20b)."),
    _s("cpu_moe", "All MoE experts on CPU", "memory", "bool", False, flag="--cpu-moe", when=True),
    _s("mmap", "Memory-map the model (mmap)", "memory", "bool", True,
       "Lets the OS page weights in from disk on demand. REQUIRED if the model is larger than your RAM.",
       flag="--no-mmap", when=False),
    _s("mlock", "Lock model in RAM (mlock)", "memory", "bool", False,
       "Prevents Windows from swapping model pages out. Only enable if the model fits comfortably.",
       flag="--mlock", when=True),
    _s("flash_attn", "Flash attention", "memory", "select", "auto", options=["auto", "on", "off"],
       flag="-fa", skip_if=["auto"], help="Needed for quantized V cache. 'auto' lets llama.cpp decide."),
    _s("cache_type_k", "KV cache type (K)", "memory", "select", "f16", options=CACHE_TYPES, flag="-ctk",
       skip_if=["f16"], help="q8_0 halves K-cache memory with tiny quality loss."),
    _s("cache_type_v", "KV cache type (V)", "memory", "select", "f16", options=CACHE_TYPES, flag="-ctv",
       skip_if=["f16"], help="Quantized V cache needs flash attention on."),
    _s("kv_offload", "Keep KV cache on GPU", "memory", "bool", True, flag="--no-kv-offload", when=False,
       help="Turn off to keep the KV cache in system RAM (frees VRAM on tiny GPUs)."),
    _s("swa_full", "Full-size sliding-window cache", "memory", "bool", False, flag="--swa-full", when=True,
       advanced=True),
    _s("op_offload", "Offload big batch operations to the GPU", "memory", "bool", True, flag="--no-op-offload",
       when=False, advanced=True,
       help="While reading your prompt, llama.cpp can copy CPU-resident weights over PCIe to the GPU. On a slow link or a "
            "tiny GPU this can be slower than staying on the CPU - the Optimizer measures both."),
    _s("cache_reuse", "Prompt cache reuse (min chunk)", "memory", "int", 0, min=0, max=65536,
       flag="--cache-reuse", skip_if=[0], advanced=True),
    _s("main_gpu", "Main GPU index", "memory", "int", 0, min=0, max=16, flag="-mg", skip_if=[0],
       advanced=True),
    _s("tensor_split", "Tensor split (multi-GPU)", "memory", "str", "", flag="-ts", advanced=True),
    _s("override_tensor", "Override tensor placement", "memory", "str", "", flag="-ot", advanced=True,
       help='Advanced: e.g. ".ffn_.*_exps.=CPU" keeps all experts in RAM.'),

    # --------------------------------------------------------------------- cpu
    _s("threads", "Generation threads", "cpu", "int", 0, min=0, max=256, flag="-t", skip_if=[0],
       help="0 = let llama.cpp choose. Best is usually your physical core count (6 on an i7-8700K)."),
    _s("threads_batch", "Prompt-processing threads", "cpu", "int", 0, min=0, max=256, flag="-tb",
       skip_if=[0], help="0 = same as generation threads. Logical core count (12 on an 8700K) often helps."),
    _s("batch_size", "Batch size", "cpu", "int", 2048, min=1, max=65536, flag="-b", advanced=True),
    _s("ubatch_size", "Micro-batch size", "cpu", "int", 512, min=1, max=65536, flag="-ub",
       help="Lower (128-256) reduces peak memory on small GPUs."),
    _s("cpu_strict", "Strict CPU placement", "cpu", "bool", False, flag="--cpu-strict 1", when=True,
       advanced=True),
    _s("prio", "llama.cpp thread priority", "cpu", "int", 0, min=-1, max=3, flag="--prio", skip_if=[0],
       advanced=True),
    _s("poll", "Polling level (0-100)", "cpu", "int", 50, min=0, max=100, flag="--poll", skip_if=[50],
       advanced=True),
    _s("numa", "NUMA mode", "cpu", "select", "none", options=["none", "distribute", "isolate", "numactl"],
       flag="--numa", skip_if=["none"], advanced=True),

    # ---------------------------------------------------------------- sampling
    _s("temperature", "Temperature", "sampling", "float", 1.0, min=0, max=5, live=True,
       help="gpt-oss recommends 1.0."),
    _s("top_p", "Top-p", "sampling", "float", 1.0, min=0, max=1, live=True),
    _s("top_k", "Top-k (0 = off)", "sampling", "int", 0, min=0, max=1000, live=True),
    _s("min_p", "Min-p", "sampling", "float", 0.0, min=0, max=1, live=True),
    _s("repeat_penalty", "Repeat penalty", "sampling", "float", 1.0, min=0.5, max=2, live=True),
    _s("presence_penalty", "Presence penalty", "sampling", "float", 0.0, min=-2, max=2, live=True),
    _s("frequency_penalty", "Frequency penalty", "sampling", "float", 0.0, min=-2, max=2, live=True),
    _s("mirostat", "Mirostat", "sampling", "select", "0", options=["0", "1", "2"], live=True, advanced=True),
    _s("mirostat_tau", "Mirostat tau", "sampling", "float", 5.0, min=0, max=20, live=True, advanced=True),
    _s("mirostat_eta", "Mirostat eta", "sampling", "float", 0.1, min=0, max=1, live=True, advanced=True),
    _s("seed", "Seed (-1 = random)", "sampling", "int", -1, min=-1, max=2**31 - 1, live=True),
    _s("max_tokens", "Max tokens per reply (-1 = unlimited)", "sampling", "int", -1, min=-1, max=1048576,
       live=True),

    # --------------------------------------------------------------- reasoning
    _s("reasoning_effort", "gpt-oss reasoning effort", "reasoning", "select", "medium",
       "low = fastest. high = best answers but many more tokens (slow on CPU).",
       options=["low", "medium", "high"], live=True),
    _s("show_reasoning", "Show the model's reasoning in chat", "reasoning", "bool", True, live=True),
    _s("stream", "Stream replies token by token", "reasoning", "bool", True, live=True),
    _s("system_prompt", "System prompt", "reasoning", "text", "", live=True),
    _s("max_tool_rounds", "Max tool-call rounds per message", "reasoning", "int", 8, min=1, max=50,
       live=True),

    # ---------------------------------------------------------------------- pc
    _s("tools_enabled", "Let the model use my PC (master switch)", "pc", "bool", False,
       "When off, the model can only chat. When on, it can call the enabled tools below.", live=True),
    _s("full_pc_access", "Full PC access (every drive and folder)", "pc", "bool", False,
       "Off = tools only reach the allowed folders below.", live=True, danger=True),
    _s("allowed_roots", "Allowed folders (one per line)", "pc", "list", [],
       "Used when full PC access is off. Empty = your home folder.", live=True),
    _s("protect_system_paths", "Never write to system folders", "pc", "bool", True,
       "Blocks writes under the protected list below even with full PC access.", live=True),
    _s("blocked_paths", "Protected folders (one per line)", "pc", "list", _BLOCKED, live=True),
    _s("tool_list_dir", "Tool: list folder", "pc", "bool", True, live=True),
    _s("tool_read_file", "Tool: read file", "pc", "bool", True, live=True),
    _s("tool_find_files", "Tool: find files", "pc", "bool", True, live=True),
    _s("tool_system_info", "Tool: system info", "pc", "bool", True, live=True),
    _s("tool_write_file", "Tool: write / edit file", "pc", "bool", True, live=True),
    _s("tool_run_command", "Tool: run command (PowerShell / shell)", "pc", "bool", False,
       "Commands are NOT sandboxed - they run as you.", live=True, danger=True),
    _s("tool_fetch_url", "Tool: fetch web page", "pc", "bool", False, live=True),
    _s("confirm_reads", "Ask me before reading", "pc", "bool", False, live=True),
    _s("confirm_writes", "Ask me before writing", "pc", "bool", True, live=True),
    _s("confirm_commands", "Ask me before running commands", "pc", "bool", True, live=True, danger=True),
    _s("confirm_fetch", "Ask me before fetching URLs", "pc", "bool", True, live=True),
    _s("max_read_bytes", "Max bytes returned per file read", "pc", "int", 200_000, min=1000, max=50_000_000,
       live=True),
    _s("max_output_chars", "Max characters of tool output sent to the model", "pc", "int", 20_000,
       min=500, max=1_000_000, live=True),
    _s("command_timeout", "Command timeout (seconds)", "pc", "int", 60, min=1, max=3600, live=True),
    _s("audit_log", "Write every tool call to the audit log", "pc", "bool", True, live=True),

    # ---------------------------------------------------------------- learning
    _s("memory_enabled", "Long-term memory", "learning", "bool", True,
       "Remembers facts/preferences you teach it and quietly adds the relevant ones to each chat. Stored only on this PC.",
       live=True),
    _s("memory_inject_items", "Max memories added per message", "learning", "int", 6, min=1, max=30, live=True),
    _s("memory_inject_chars", "Max characters of memory added per message", "learning", "int", 1800,
       min=200, max=20000, live=True, help="Everything injected eats context and slows prompt processing."),
    _s("memory_auto_extract", "Suggest memories after a chat", "learning", "select", "ask",
       "ask = propose what to remember and let you pick; auto = save without asking; off = only when you say 'remember…'.",
       options=["off", "ask", "auto"], live=True),
    _s("memory_extract_max", "Max memories suggested per chat", "learning", "int", 5, min=1, max=20, live=True),
    _s("tool_memory", "Let the AI save/search memory itself (tools)", "learning", "bool", True, live=True),
    _s("confirm_memory_save", "Ask me before the AI saves a memory", "learning", "bool", True,
       "Recommended: web pages and documents can try to plant false 'memories'.", live=True),
    _s("exchange_retention_days", "Forget unrated logged answers after (days, 0 = keep)", "learning", "int", 0,
       "Answers you rated, corrected or approved for training are kept; the rest of the log is deleted after this many days.",
       min=0, max=3650),
    _s("record_exchanges", "Keep a log of questions and answers (for ratings & training data)", "learning", "bool",
       True, "Local only. Needed for 👍/👎, corrections and the training-data sandbox.", live=True),
    _s("examples_enabled", "Show the AI examples of answers you liked", "learning", "bool", True,
       "Similar past 👍 / corrected answers are added to the prompt as examples.", live=True),
    _s("examples_max", "Max examples per message", "learning", "int", 2, min=1, max=6, live=True),
    _s("examples_chars", "Max characters of examples per message", "learning", "int", 1500, min=200, max=12000,
       live=True),
    _s("lessons_enabled", "Apply lessons from 👎 notes", "learning", "bool", True,
       "When you downvote with a note, similar future questions get the note as a reminder.", live=True),
    _s("smart_prompt", "Built-in 'think carefully' prompt", "learning", "select", "careful",
       "A short system prompt tuned for small models (step by step, admit uncertainty, use tools/search).",
       options=["off", "concise", "careful", "tutor"], live=True),
    _s("reflect_passes", "Refine passes (🔁 button)", "learning", "int", 1, min=1, max=3, live=True,
       help="Each pass = critique + rewrite by the same model. Slower but often better."),
    _s("redact_pii", "Redact emails, keys, phone numbers, Windows user names in exported training data", "learning",
       "bool", True, live=True),

    # --------------------------------------------------------------------- web
    _s("web_grounding", "Web research", "web", "select", "off",
       "off = never goes online. tool = the AI may call web_search / web_read. auto = also searches before answering "
       "questions that look like they need fresh facts. Your query is sent to the search provider.",
       options=["off", "tool", "auto"], live=True),
    _s("web_provider", "Search provider", "web", "select", "duckduckgo",
       "duckduckgo and wikipedia need no key. searxng needs your own instance URL. brave needs an API key.",
       options=["duckduckgo", "wikipedia", "searxng", "brave"], live=True),
    _s("searxng_url", "SearXNG instance URL", "web", "str", "", live=True),
    _s("brave_api_key", "Brave Search API key", "web", "password", "", live=True),
    _s("web_max_results", "Search results used", "web", "int", 5, min=1, max=10, live=True),
    _s("web_fetch_pages", "Pages read in full per search", "web", "int", 2, min=0, max=5, live=True),
    _s("web_page_chars", "Max characters taken from each page", "web", "int", 3000, min=300, max=30000, live=True),
    _s("web_cache_minutes", "Cache search results (minutes)", "web", "int", 30, min=0, max=1440, live=True),
    _s("web_timeout", "Web request timeout (seconds)", "web", "int", 15, min=3, max=120, live=True),
    _s("web_save_to_memory", "Save useful pages to memory as 'web' notes", "web", "bool", False, live=True),
    _s("confirm_web", "Ask me before the AI uses web tools", "web", "bool", True, live=True,
       help="On by default: a web page can try to talk the model into sending your files out through a web tool. "
            "Turning this off means the AI can fetch URLs (and put data in them) without asking."),
    _s("web_allow_private", "Allow web tools to reach private/local addresses", "web", "bool", False,
       "Leave off: it stops a web page from steering the AI into your home network.", live=True, danger=True,
       advanced=True),

    # --------------------------------------------------------------------- big
    _s("big_model_path", "Big model (.gguf)", "big", "path", "",
       "e.g. gpt-oss-120b. Used for escalations, overnight review and teacher answers.", kind="file"),
    _s("big_run_mode", "When to run queued big-model jobs", "big", "select", "manual",
       "manual = only when you press 'Run queue now'. immediate = as soon as queued. overnight = only inside the night window.",
       options=["manual", "immediate", "overnight"], live=True),
    _s("big_swap_models", "Stop the everyday model while the big one runs", "big", "bool", True,
       "Needed when both do not fit in RAM together (e.g. 20B + 120B on 48 GB). The everyday model is restarted after.",
       live=True),
    _s("big_port", "Big model server port", "big", "int", 8081, min=1, max=65535),
    _s("big_ctx_size", "Big model context size", "big", "int", 4096, min=512, max=131072),
    _s("big_n_cpu_moe", "Big model MoE layers on CPU (0 = auto)", "big", "int", 0, min=0, max=999),
    _s("big_reasoning_effort", "Big model reasoning effort", "big", "select", "high", options=["low", "medium", "high"],
       live=True),
    _s("big_max_tokens", "Max tokens per big-model answer", "big", "int", 3000, min=64, max=65536, live=True),
    _s("big_idle_stop_minutes", "Stop the big model after being idle (minutes)", "big", "int", 3, min=0, max=240,
       live=True),
    _s("auto_escalate_downvote", "Queue a big-model answer when I 👎 something", "big", "bool", False, live=True),
    _s("auto_escalate_hard", "Queue a big-model answer when a question looks hard or the AI sounds unsure", "big",
       "bool", False, live=True),
    _s("overnight_start", "Night window starts (HH:MM)", "big", "str", "01:00", live=True),
    _s("overnight_end", "Night window ends (HH:MM)", "big", "str", "07:00", live=True),
    _s("overnight_review_count", "Teacher-review this many recent answers per night", "big", "int", 0, min=0,
       max=500, live=True, help="0 = off. The big model re-answers your recent questions; the answers become training data."),
    _s("queue_max", "Max queued jobs", "big", "int", 100, min=1, max=1000, live=True),

    # ---------------------------------------------------------------- lora
    _s("lora_path", "LoRA adapter (.gguf)", "server", "path", "",
       "Optional. A fine-tune trained from your exported data (see Learning → Training data).", kind="file",
       advanced=True),
    _s("lora_scale", "LoRA strength", "server", "float", 1.0, min=0, max=4, advanced=True),

    # -------------------------------------------------------------------- vgpu
    _s("vgpu_monitor", "Monitor memory, CPU and GPU while running", "vgpu", "bool", True,
       "Samples real telemetry (RAM, swap, CPU, NVIDIA GPU, the model process) for the Memory page and the Dashboard."),
    _s("vgpu_monitor_interval", "Monitor interval (seconds)", "vgpu", "int", 2, min=1, max=60, advanced=True),
    _s("vgpu_pressure_action", "When memory pressure becomes critical", "vgpu", "select", "warn",
       "warn = only tell me. stop_big_model = unload the big escalation model. stop_all = also stop the chat model "
       "(prevents the whole PC from freezing).", options=["warn", "stop_big_model", "stop_all"]),
    _s("vgpu_admission", "Check a model before loading it", "vgpu", "select", "enforce",
       "enforce = refuse loads that cannot work (you can still force). warn = show the problems but load anyway. off = no check.",
       options=["enforce", "warn", "off"]),
    _s("vgpu_autofit", "Reduce GPU offload automatically when it would not fit in VRAM", "vgpu", "bool", True,
       "If the only thing stopping a model from loading is GPU offload bigger than your card's VRAM, Duckstoken lowers "
       "n_gpu_layers / n_cpu_moe for that start only and tells you. Your saved settings are not changed."),
    _s("vgpu_os_reserve_gb", "Memory kept free for Windows and other apps (GB)", "vgpu", "float", 6.0,
       "The planner never hands this memory to the model.", min=0, max=128),
    _s("vgpu_safety_margin_pct", "Safety margin on the estimate (%)", "vgpu", "float", 5.0, min=0, max=50),
    _s("vgpu_vram_reserve_mb", "VRAM kept free for the desktop and driver (MB)", "vgpu", "int", 400, min=0, max=8192),
    _s("vgpu_max_spill_pct", "Most of the model allowed to page from disk (%)", "vgpu", "int", 50,
       "Past this the model would spend its time reading the drive. Hard drives tolerate almost none.",
       min=0, max=100, advanced=True),
    _s("vgpu_objective", "Optimizer goal", "vgpu", "select", "balanced",
       "max_model_size = biggest model that still runs usably. max_speed = fastest measured tokens/s. balanced = speed per GB of memory.",
       options=["max_model_size", "max_speed", "balanced"]),
    _s("vgpu_min_tps", "Slowest acceptable speed (tokens/s)", "vgpu", "float", 1.0,
       "Used by the 'max model size' goal.", min=0.1, max=100),
    _s("vgpu_bench_prompt_tokens", "Benchmark prompt length (tokens)", "vgpu", "int", 512, min=16, max=32768, advanced=True),
    _s("vgpu_bench_gen_tokens", "Benchmark generated tokens", "vgpu", "int", 64, min=8, max=2048, advanced=True),
    _s("vgpu_bench_repeats", "Benchmark warm runs", "vgpu", "int", 3, min=1, max=10, advanced=True),
    _s("vgpu_perf_log", "Write a performance log", "vgpu", "bool", True,
       "Structured JSON lines in the Duckstoken data folder (size-limited, local only, never uploaded)."),
    _s("vgpu_cache_dir", "Fast-drive staging cache folder", "vgpu", "path", "",
       "Optional. A folder on your SSD: Duckstoken can copy a model from a slow hard drive here (after you confirm) so paging reads from the SSD.",
       kind="dir"),
    _s("vgpu_cache_gb", "Staging cache size limit (GB, 0 = off)", "vgpu", "float", 0.0, min=0, max=100000),
    _s("vgpu_cache_min_free_gb", "Keep this much free on the cache drive (GB)", "vgpu", "float", 10.0, min=1, max=10000,
       advanced=True),
    _s("vgpu_prewarm", "Read the model into the OS file cache before starting", "vgpu", "select", "off",
       "Sequential read converts slow random disk faults into one fast pass. Only helps if the model fits in free RAM.",
       options=["off", "before_start"]),
    _s("vgpu_prewarm_max_gb", "Prewarm at most (GB, 0 = as much as fits)", "vgpu", "float", 0.0, min=0, max=100000,
       advanced=True),
    _s("server_auto_restart", "Restart the model server if it crashes", "vgpu", "bool", False,
       "Retries with a safer configuration (smaller context, less GPU) and gives up after the limit below."),
    _s("server_restart_max", "Crash restarts before giving up", "vgpu", "int", 2, min=1, max=10, advanced=True),

    # ------------------------------------------------------------------ engine
    _s("engine", "Inference engine", "engine", "select", "llamacpp",
       "llamacpp = the bundled llama.cpp engine (fast, runs huge models). external = Ollama / LM Studio / any "
       "OpenAI-compatible server you already run. builtin = Duckstoken's own pure-Python engine (small models only).",
       options=["llamacpp", "external", "builtin"]),
    _s("external_preset", "External server", "engine", "select", "ollama", options=["ollama", "lmstudio", "custom"],
       help="Where the external engine listens. Ollama: 11434, LM Studio: 1234."),
    _s("external_url", "Custom server URL", "engine", "str", "http://127.0.0.1:8000",
       "Used when the preset is 'custom'. Must speak the OpenAI /v1 API."),
    _s("external_api_key", "External API key", "engine", "password", "", "Only if the server requires one."),
    _s("external_model", "External model name", "engine", "str", "",
       "Leave empty to use the first model the server reports."),
    _s("builtin_threads", "Built-in engine threads (0 = auto)", "engine", "int", 0, min=0, max=256, advanced=True),

    # --------------------------------------------------------------------- app
    _s("window_mode", "Open Duckstoken in", "app", "select", "native",
       "native = its own window (needs WebView2, preinstalled on Windows 11). browser = your default web browser.",
       options=["native", "browser"]),
    _s("minimize_to_tray", "Keep running in the system tray when the window is closed", "app", "bool", True,
       "The model server keeps running. Right-click the tray icon to quit."),
    _s("start_minimized", "Start hidden in the tray", "app", "bool", False),
    _s("launch_at_startup", "Start Duckstoken when I sign in to Windows", "app", "bool", False),
    _s("update_channel", "Update channel", "app", "select", "stable",
       "stable = tagged releases. latest = every build of the main branch. off = never check.",
       options=["off", "stable", "latest"]),
    _s("update_auto_install", "Install updates automatically on exit", "app", "bool", False,
       "Off: you are asked first. Updates are verified against a published SHA-256 before being applied."),
]

SCHEMA_BY_KEY = {s["key"]: s for s in SCHEMA}
DEFAULTS: dict[str, Any] = {s["key"]: s["default"] for s in SCHEMA}


# ------------------------------------------------------------------ validation
def coerce(spec: dict, value: Any) -> Any:
    """Coerce `value` to the setting's type and clamp it; raises ValueError if hopeless."""
    t = spec["type"]
    if t == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if t == "int":
        v = int(float(value))
    elif t == "float":
        v = float(value)
    elif t == "select":
        v = str(value)
        if v not in spec["options"]:
            raise ValueError(f"{spec['key']}: '{v}' is not one of {spec['options']}")
        return v
    elif t == "list":
        if isinstance(value, str):
            value = value.splitlines()
        return [str(x).strip() for x in value if str(x).strip()]
    else:  # str, text, path, password
        return "" if value is None else str(value).strip() if t != "text" else str(value)
    if "min" in spec:
        v = max(spec["min"], v)
    if "max" in spec:
        v = min(spec["max"], v)
    return v


# Settings that decide what runs on the PC or where data goes. Importing a settings file or loading a profile that changes
# any of them (or any "pc" tool switch, or any danger-flagged switch away from its default) must be confirmed by the user:
# otherwise a settings file shared as "my best settings" could switch off every approval prompt.
PRIVILEGED = {"llama_server_path", "extra_args", "lora_path", "llama_dir", "models_dir", "searxng_url", "external_url", "host",
              "big_model_path", "vgpu_cache_dir", "update_channel", "update_auto_install", "launch_at_startup"}


def risky_changes(current: dict, patch: dict) -> list[dict]:
    """The entries of `patch` that need an explicit confirmation before they are applied: [{key, label, from, to}]."""
    out = []
    for k, v in patch.items():
        spec = SCHEMA_BY_KEY.get(k)
        if spec is None:
            continue
        try:
            new = coerce(spec, v)
        except ValueError:
            continue
        old = current.get(k, spec["default"])
        if new == old:
            continue
        away_from_default = new != spec["default"]
        if k in PRIVILEGED or spec["group"] == "pc" and away_from_default or spec.get("danger") and away_from_default:
            out.append({"key": k, "label": spec["label"], "from": "" if spec["type"] == "password" else old,
                        "to": "(hidden)" if spec["type"] == "password" else new})
    return out


def validate_patch(patch: dict) -> dict:
    out = {}
    for k, v in patch.items():
        spec = SCHEMA_BY_KEY.get(k)
        if spec is None:
            raise ValueError(f"Unknown setting: {k}")
        out[k] = coerce(spec, v)
    return out


# ----------------------------------------------------------------- persistence
class SettingsStore:
    def __init__(self, path: Path | None = None):
        self.path = path or paths.settings_file()
        self._lock = threading.RLock()
        self._values = dict(DEFAULTS)
        self.load()

    def load(self) -> None:
        with self._lock:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raw = {}
            for k, v in raw.items():
                spec = SCHEMA_BY_KEY.get(k)
                if spec:
                    try:
                        self._values[k] = coerce(spec, v)
                    except (ValueError, TypeError):
                        pass

    def save(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._values, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values[key]

    def all(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._values))

    def update(self, patch: dict) -> dict:
        clean = validate_patch(patch)
        with self._lock:
            self._values.update(clean)
            self.save()
        return clean

    def reset(self, keys: list[str] | None = None) -> None:
        with self._lock:
            for k in keys or list(DEFAULTS):
                if k in DEFAULTS:
                    self._values[k] = json.loads(json.dumps(DEFAULTS[k]))
            self.save()

    # resolved directories --------------------------------------------------
    def models_dir(self) -> Path:
        p = self.get("models_dir")
        return Path(p) if p else paths.default_models_dir()

    def engines_dir(self) -> Path:
        p = self.get("llama_dir")
        return Path(p) if p else paths.default_llama_dir()


# --------------------------------------------------------------------- presets
PRESETS: dict[str, dict] = {
    "gpt-oss-120b - big model, small GPU (recommended)": {
        "description": "Experts stay in RAM, attention on GPU. Small context, mmap on, 8-bit KV cache.",
        "values": {"n_cpu_moe": 36, "n_gpu_layers": 99, "ctx_size": 4096, "mmap": True, "mlock": False,
                   "flash_attn": "on", "cache_type_k": "q8_0", "cache_type_v": "q8_0", "ubatch_size": 256,
                   "batch_size": 512, "reasoning_effort": "low", "parallel": 1},
    },
    "gpt-oss-20b - fits in RAM": {
        "description": "Comfortable on 16-32 GB RAM; fastest option for daily use on old hardware.",
        "values": {"n_cpu_moe": 24, "n_gpu_layers": 99, "ctx_size": 16384, "mmap": True, "mlock": False,
                   "flash_attn": "on", "cache_type_k": "q8_0", "cache_type_v": "q8_0", "ubatch_size": 256,
                   "batch_size": 1024, "reasoning_effort": "medium", "parallel": 1},
    },
    "CPU only (no usable GPU)": {
        "description": "Ignore the GPU completely.",
        "values": {"n_gpu_layers": 0, "n_cpu_moe": 0, "cpu_moe": False, "kv_offload": False,
                   "flash_attn": "on", "ubatch_size": 512},
    },
    "Everything fits in RAM - max speed": {
        "description": "For when the model is smaller than free RAM. Locks pages so nothing is swapped out.",
        "values": {"mmap": True, "mlock": True, "ctx_size": 8192, "batch_size": 2048, "ubatch_size": 512},
    },
    "Low memory emergency": {
        "description": "Shrinks context and caches to stop the PC from swapping to disk.",
        "values": {"ctx_size": 2048, "cache_type_k": "q4_0", "cache_type_v": "q4_0", "flash_attn": "on",
                   "batch_size": 256, "ubatch_size": 128, "mlock": False},
    },
    "Long context": {
        "description": "32k context with a quantized KV cache. Needs plenty of spare RAM.",
        "values": {"ctx_size": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on"},
    },
    "Creative writing": {
        "description": "Looser sampling.",
        "values": {"temperature": 1.15, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.05},
    },
    "Precise / coding": {
        "description": "Tighter sampling, higher reasoning effort.",
        "values": {"temperature": 0.6, "top_p": 0.9, "reasoning_effort": "high"},
    },
}


class ProfileStore:
    """User-named snapshots of the whole settings object."""

    def __init__(self, path: Path | None = None):
        self.path = path or paths.profiles_file()
        self._lock = threading.Lock()

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._read())

    def get(self, name: str) -> dict | None:
        with self._lock:
            return self._read().get(name)

    def save(self, name: str, values: dict) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Profile needs a name")
        with self._lock:
            data = self._read()
            data[name] = values
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def delete(self, name: str) -> None:
        with self._lock:
            data = self._read()
            data.pop(name, None)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def public_schema() -> list[dict]:
    return SCHEMA
