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
