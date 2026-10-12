"""Shared helpers for the VirtualGPU runtime tests: tiny GGUF models, fake-engine settings, a recording server
manager (to prove no process leaks) and a deterministic stand-in for the planner."""
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fake_vgpu_engine  # noqa: E402
from gguf_helper import write_gguf  # noqa: E402

from duckstoken import settings as settings_mod  # noqa: E402
from duckstoken.server_proc import ServerManager  # noqa: E402

DENSE = {"general.architecture": "llama", "general.name": "tiny-dense", "llama.block_count": 4,
         "llama.context_length": 2048, "llama.embedding_length": 64, "llama.attention.head_count": 4,
         "llama.attention.head_count_kv": 2, "general.file_type": 15}


def make_model(directory, name="tiny.gguf", kvs=None, pad=256 * 1024) -> str:
    path = Path(directory) / name
    write_gguf(path, dict(kvs or DENSE), pad=pad)
    return str(path)


def fake_settings(directory, extra="", **overrides) -> dict:
    """A full settings dict that launches the fake engine; `extra` becomes extra_args (``--fake-*`` knobs)."""
    exe = fake_vgpu_engine.make_exe(directory)
    s = dict(settings_mod.DEFAULTS)
    s.update(llama_server_path=str(exe), ctx_size=2048, n_gpu_layers=0, kv_offload=False, extra_args=extra,
             parallel=1)
    s.update(overrides)
    return s


def mock_hw(ram_gb=16.0, vram_mb=0, disk_kind="nvme", disk_free_gb=200.0, cores=4, logical=8) -> dict:
    return {"os": "test", "cpu": {"name": "Test CPU", "physical_cores": cores, "logical_cores": logical,
                                   "avx2": True, "avx512": False},
            "ram": {"total_gb": ram_gb, "avail_gb": ram_gb * 0.8, "speed_mts": 3200, "sticks": 2, "bandwidth_gbs": 51.2},
            "gpus": ([{"name": "Test GPU", "vendor": "NVIDIA", "vram_mb": vram_mb, "vram_free_mb": vram_mb,
                       "driver": "1", "compute_cap": "8.6"}] if vram_mb else []),
            "disks": [{"drive": "C:", "model": "", "bus": "", "media": "", "kind": disk_kind,
                       "size_gb": 500.0, "free_gb": disk_free_gb}]}


class RecordingServer(ServerManager):
    """A ServerManager that remembers every process it started so tests can prove none is left running."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.procs = []

    def start(self, s):
        super().start(s)
        self.procs.append(self.proc)


class ServerFactory:
    """Hands out RecordingServers and keeps them all, across several benchmark runs."""

    def __init__(self):
        self.servers = []

    def __call__(self, log_file):
        srv = RecordingServer(log_file=log_file)
        self.servers.append(srv)
        return srv

    @property
    def procs(self):
        return [p for s in self.servers for p in s.procs]

    def alive(self):
        return [p.pid for p in self.procs if p.poll() is None]


def pid_exists(pid: int) -> bool:
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def thread_names():
    return sorted(t.name for t in threading.enumerate() if t.name.startswith(("bench-", "vgpu-job", "vgpu-sampler")))


# --------------------------------------------------------------------------------------------- fake planner
def _est(v, note=""):
    return {"value": v, "status": "estimated", "source": "FakePlanner", "note": note}


class FakePlanner:
    """Deterministic stand-in for `vgpu.planner` (unit isolation).  `reject(cfg) -> list[str] | None` decides
    admission per config; `speed(cfg) -> (lo, hi)` and `resident(cfg) -> GiB` shape the estimates."""

    def __init__(self, reject=None, speed=None, resident=None, verdict="ok", max_ngl=12, min_moe=20, verdict_fn=None):
        self.reject, self.speed_fn, self.resident_fn, self.verdict, self.max_ngl = reject, speed, resident, verdict, max_ngl
        self.min_moe, self.verdict_fn = min_moe, verdict_fn
        self.calls = []

    def plan(self, hw, model, cfg=None, assumptions=None, model_path=None, force=False):
        cfg = dict(cfg or {})
        self.calls.append(cfg)
        blockers = list(self.reject(cfg) or []) if self.reject else []
        lo, hi = self.speed_fn(cfg) if self.speed_fn else (4.0, 8.0)
        gb = self.resident_fn(cfg) if self.resident_fn else 3.0 + 0.1 * (cfg.get("n_gpu_layers") or 0)
        ngl = int(cfg.get("n_gpu_layers") or 0)
        return {"model": {"name": model.get("name"), "path": model.get("path")}, "config": cfg,
                "requirements": {"weights_gb": _est(gb), "kv_cache_gb": _est(0.1), "compute_buffers_gb": _est(0.05),
                                 "runtime_overhead_gb": _est(0.5), "os_reserve_gb": _est(6.0),
                                 "safety_margin_gb": _est(0.1), "total_gb": _est(gb + 6.6)},
                "tiers": {"vram": {"planned_gb": _est(0.1 * ngl)}, "ram": {"planned_gb": _est(gb - 0.1 * ngl)}},
                "placement": {"gpu_layers": ngl, "cpu_layers": 4 - min(4, ngl), "kv_on_gpu": bool(cfg.get("kv_offload")),
                              "weights_vram_gb": 0.1 * ngl, "weights_ram_gb": gb - 0.1 * ngl, "weights_spill_gb": 0.0},
                "admission": {"verdict": "reject" if blockers else (self.verdict_fn(cfg) if self.verdict_fn else self.verdict),
                              "can_load": not blockers,
                              "blockers": blockers, "warnings": [], "suggestions": [], "headroom_gb": 5.0},
                "speed": {"tokens_per_s": _est([lo, hi]), "bound_by": "memory_bandwidth", "explanation": "fake"},
                "assumptions": [], "status": "estimated"}

    def max_gpu_layers(self, hw, model, cfg=None, assumptions=None):
        if cfg and cfg.get("kv_offload") is False:
            return self.max_ngl + 2           # keeping the KV cache in RAM frees room for more layers
        return self.max_ngl

    def min_cpu_moe(self, hw, model, cfg=None, assumptions=None):
        return self.min_moe

    def resident_estimate_gb(self, plan):
        return round(plan["tiers"]["vram"]["planned_gb"]["value"] + plan["tiers"]["ram"]["planned_gb"]["value"], 2)

    def parse_buffer_log(self, lines):
        from duckstoken.vgpu import memcalc
        return memcalc.parse_buffer_log(lines)

    def measured_tiers_gb(self, parsed):
        from duckstoken.vgpu import memcalc
        return memcalc.measured_tiers_gb(parsed)


def child_pids() -> list:
    """PIDs of live child processes of this interpreter (Linux only; [] elsewhere)."""
    if not os.path.isdir("/proc"):
        return []
    me, out = os.getpid(), []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                data = f.read()
            fields = data[data.rindex(")") + 2:].split()
            state, ppid = fields[0], int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == me and state != "Z":
            out.append(int(name))
    return out


class FakeProbe:
    """Canned engine probe (the real one spawns the executable twice)."""

    def __init__(self, version="9999", build=9999, devices=None, status="ok"):
        self.result = {"path": "", "exists": True, "version": version, "build": build, "commit": "abc1234", "backends": ["CPU"],
                       "backends_usable": [], "devices": devices or [], "supports": {}, "errors": [], "status": status}
        self.calls = 0

    def probe_engine(self, path, **kw):
        self.calls += 1
        return {**self.result, "path": str(path)}
