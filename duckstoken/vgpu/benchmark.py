"""Reproducible benchmark of one configuration on one model.

`run_benchmark` starts its OWN llama-server (separate log, free port, 127.0.0.1, no API key) from a full settings dict,
so it never touches the chat server.  The sequence is:

    engine probe -> admission by the planner (rejected: no process is started) -> start -> wait healthy
    -> one COLD run (the first request after load) -> optional discarded warm-up -> `repeats` WARM runs -> stop

Every run uses the same deterministic prompt (a fixed sentence repeated, sha256 recorded), temperature 0, a fixed seed
and ``cache_prompt=false`` so the prompt is processed every time.  Prompt-processing and generation speeds are separate
numbers; cold and warm are separate; llama-server's own ``timings`` are used when sent (``measured``) and client-side
timestamps otherwise (``estimated``).  Peak RSS, CPU/GPU utilisation, page faults and I/O come from a sampler thread.
The page cache is NOT purged between benchmarks: "cold" means the first request after the process started.

The server is always stopped in a ``finally`` block, also on cancel, timeout, crash or exception, and the result says
what happened instead of raising.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .. import gguf, hardware, settings as settings_mod
from ..server_proc import ServerManager, find_llama_server, terminate_tree
from . import bench_client as bc
from .bench_client import summarize_runs  # noqa: F401 - part of this module's API
from .bench_compare import compare_benchmarks, compare_plan, plan_summary  # noqa: F401 - re-exported
from .bench_metrics import aggregate_samples, host_bandwidth, make_sampler  # noqa: F401 - re-exported
from .lazy import module as _mod
from .store import hardware_fingerprint
from .values import measured, unavailable

# Settings that change speed or memory; their canonical JSON is the config hash.  Everything else (ports, API key,
# web UI, template, log verbosity) does not affect what is being measured.
NON_PERF_KEYS = {"host", "port", "api_key", "model_path", "verbose", "server_timeout", "builtin_webui", "metrics",
                 "jinja", "reasoning_format", "chat_template"}
PERF_KEYS = [s["key"] for s in settings_mod.SCHEMA if s.get("flag") and s["key"] not in NON_PERF_KEYS] + \
    ["extra_args", "lora_path", "lora_scale"]
PLAN_KEYS = ("ctx_size", "n_gpu_layers", "n_cpu_moe", "cpu_moe", "kv_offload", "cache_type_k", "cache_type_v",
             "ubatch_size", "mmap", "mlock", "flash_attn", "swa_full", "parallel")
VERBOSITY_FLAGS = {"-lv", "--verbosity", "--log-verbosity", "-v", "--verbose", "--log-verbose"}
LOG_TAIL = 40
NOTE_REPRO = ("page cache not purged between runs; cold = first request after process start; "
              "prompt re-processed every run (cache_prompt=false)")


@dataclass
class BenchSpec:
    """What to measure.  ``warmup`` adds one extra request after the cold run that is discarded, so the warm runs
    reflect steady state; the cold run itself is always recorded separately."""
    prompt_tokens: int = 512
    gen_tokens: int = 64
    repeats: int = 3
    warmup: bool = True
    seed: int = 1234
    temperature: float = 0.0
    ctx: int | None = None
    load_timeout_s: float = 900
    request_timeout_s: float = 600
    sample_interval_s: float = 0.5

    def normalized(self) -> "BenchSpec":
        return BenchSpec(max(8, int(self.prompt_tokens)), max(2, int(self.gen_tokens)), max(1, int(self.repeats)),
                         bool(self.warmup), int(self.seed), float(self.temperature),
                         int(self.ctx) if self.ctx else None, max(1.0, float(self.load_timeout_s)),
                         max(1.0, float(self.request_timeout_s)), max(0.05, float(self.sample_interval_s)))

    def spec_hash(self) -> str:
        """Hash of the fields that change the numbers (not the timeouts or the sampling interval)."""
        keep = {k: v for k, v in asdict(self).items() if k not in ("load_timeout_s", "request_timeout_s", "sample_interval_s")}
        return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


class _Stop(Exception):
    """Internal: unwind to the cleanup code with a final status."""

    def __init__(self, status: str, error: str = ""):
        super().__init__(error)
        self.status, self.error = status, error


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------------------------------------ the benchmark
class _Bench:
    def __init__(self, settings, model_path, label, hw, spec, server, cancel, progress, perflog, sampler_fn,
                 assumptions, work_dir, stop_grace_s):
        self.eff = {**settings_mod.DEFAULTS, **(settings or {})}
        self.model_path, self.label = str(model_path), label
        self.hw = hw                 # None: detected in prepare() (so a failure there is recorded, not raised)
        self.spec = (spec or BenchSpec()).normalized()
        self.server, self.cancel = server, cancel or threading.Event()
        self._progress, self.perflog = progress, perflog
        self.sampler_fn = sampler_fn
        self.assumptions, self.work_dir, self.grace = assumptions, work_dir, stop_grace_s
        self.id = f"bench-{time.strftime('%Y%m%d%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"
        self.proc = None            # the process THIS benchmark started; only that one is ever stopped
        self.step = 0
        self.total = 6 + (1 if self.spec.warmup else 0) + self.spec.repeats
        self.endpoint: str | None = None
        self.planner = _mod("planner")
        self.prompt, self.prompt_sha = bc.build_prompt(self.spec.prompt_tokens)
        self.tmp: Path | None = None
        self.rec = self._blank()

    def _blank(self) -> dict:
        na = unavailable("not measured", "benchmark")
        run = {k: na for k in ("ttft_s", "prompt_tokens", "prompt_tps", "gen_tokens", "gen_tps", "total_s")}
        stat = {**na, "mean": None, "stdev": None, "min": None, "max": None, "n": 0}
        return {"id": self.id, "created": _now_iso(), "label": self.label, "status": "failed", "error": "", "reasons": [],
                "warnings": [], "hardware": {}, "model": {"path": self.model_path}, "backend": {},
                "config": {}, "config_hash": "", "spec": asdict(self.spec), "load": {"seconds": na},
                "cold": run, "warm": {"runs": [], "ttft_s": stat, "prompt_tps": stat, "gen_tps": stat},
                "metrics": {}, "comparison": {}, "bottleneck": {}, "log_tail": [],
                "reproducibility": {"seed": self.spec.seed, "prompt_sha256": self.prompt_sha, "config_hash": "",
                                    "spec_hash": self.spec.spec_hash(), "note": NOTE_REPRO}}

    # ------------------------------------------------------------------ small helpers
    def say(self, message: str) -> None:
        self.step += 1
        if self._progress:
            try:
                self._progress(self.step, self.total, message)
            except Exception:  # noqa: BLE001
                pass

    def log(self, event: str, **fields) -> None:
        if self.perflog:
            try:
                self.perflog.log(event, benchmark=self.id, **fields)
            except Exception:  # noqa: BLE001
                pass

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise _Stop("cancelled", "cancelled by the user")

    def log_lines(self) -> list[str]:
        try:
            return list(self.server.logs()["lines"])
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------ stages
    def run(self) -> dict:
        rec = self.rec
        started = time.perf_counter()
        self.log("benchmark_start", model=self.model_path, label=self.label)
        owns_tmp = self.work_dir is None
        try:
            self.tmp = Path(tempfile.mkdtemp(prefix="duckstoken-bench-")) if owns_tmp else Path(self.work_dir)
            self.tmp.mkdir(parents=True, exist_ok=True)
            self.prepare()
            self.launch_and_measure()
            rec["status"] = "ok"
        except _Stop as stop:
            rec["status"], rec["error"] = stop.status, stop.error
            if stop.status == "rejected" and not rec["reasons"]:
                rec["reasons"] = [stop.error]
        except Exception as e:  # noqa: BLE001 - recorded, never raised
            rec["status"], rec["error"] = "failed", f"{type(e).__name__}: {e}"
        finally:
            self.shutdown()
            if self.tmp is not None:
                if owns_tmp:
                    shutil.rmtree(self.tmp, ignore_errors=True)
                else:
                    (self.tmp / f"{self.id}.log").unlink(missing_ok=True)
        rec["reproducibility"]["config_hash"] = rec["config_hash"]
        rec["elapsed_s"] = round(time.perf_counter() - started, 2)
        self.log("benchmark_done", status=rec["status"], error=rec["error"], seconds=rec["elapsed_s"])
        return rec

    def prepare(self) -> None:
        rec, eff, spec = self.rec, self.eff, self.spec
        if self.hw is None:
            self.hw = hardware.get_hardware()
        self.say("Reading the model header")
        try:
            model = gguf.summarize(self.model_path)
        except (OSError, ValueError) as e:
            raise _Stop("rejected", f"cannot read the model file: {e}") from e
        quant = _mod("quant")
        try:
            qname = quant.quant_name(model) if quant else None
        except Exception:  # noqa: BLE001
            qname = None
        st = os.stat(self.model_path)
        rec["model"] = {"path": self.model_path, "name": model["name"], "size_bytes": model["size_bytes"],
                        "mtime": st.st_mtime, "architecture": model["architecture"], "file_type": model["file_type"],
                        "quant": qname}
        rec["hardware"] = {**{k: self.hw.get(k) for k in ("os", "cpu", "ram", "gpus", "disks")},
                           "fingerprint": hardware_fingerprint(self.hw)}
        self.model = model
        # the benchmark prompt must fit one slot; the context is raised (and the record says so) rather than failing
        needed = math.ceil(spec.prompt_tokens * 1.25) + spec.gen_tokens + 64
        ctx = int(eff.get("ctx_size") or 0)
        if ctx or spec.ctx:   # ctx_size 0 means "the model's own maximum": leave that to the planner
            new_ctx = max(spec.ctx or ctx, needed)
            if new_ctx != ctx:
                rec["warnings"].append(f"context size set to {new_ctx} (settings had {ctx}) for this benchmark")
                eff["ctx_size"] = new_ctx
        eff["parallel"] = 1
        self.say("Checking the engine")
        exe = find_llama_server(eff)
        if exe is None:
            raise _Stop("failed", "the llama.cpp engine is not installed")
        self.probe(exe)
        self.say("Planning memory")
        self.admit()
        user_extra = eff.get("extra_args", "")
        cfg = {k: eff.get(k) for k in PERF_KEYS}
        rec["config"] = {**cfg, "cmdline": ""}
        rec["config_hash"] = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
        if not (set(user_extra.split()) & VERBOSITY_FLAGS) and not eff.get("verbose"):
            eff["extra_args"] = (user_extra + " -lv 4").strip()   # level 4 prints the buffer-size lines we compare with the plan
        eff["host"], eff["api_key"], eff["model_path"] = "127.0.0.1", "", self.model_path
        eff["builtin_webui"] = False
        self.exe = exe

    def probe(self, exe: Path) -> None:
        pm = _mod("probe")
        info: dict = {}
        if pm is not None:
            try:
                info = (getattr(pm, "probe_engine_cached", None) or pm.probe_engine)(str(exe))
            except Exception as e:  # noqa: BLE001
                info = {"errors": [str(e)], "status": "error"}
        self.rec["backend"] = {"engine": "llama.cpp", "path": str(exe), "version": info.get("version"),
                               "build": info.get("build"), "commit": info.get("commit"),
                               "backends": info.get("backends") or [], "devices": info.get("devices") or [],
                               "probe": info.get("status", "unavailable"), "probe_errors": info.get("errors") or []}

    def admit(self) -> None:
        if self.planner is None:
            raise _Stop("failed", "the memory planner is unavailable, so the configuration cannot be checked before loading")
        cfg = {k: self.eff[k] for k in PLAN_KEYS if k in self.eff}
        plan = self.planner.plan(self.hw, self.model, cfg, self.assumptions, model_path=self.model_path)
        self.plan = plan
        adm = plan.get("admission", {})
        self.rec["comparison"] = {"estimated": plan_summary(plan)}
        self.rec["warnings"] += list(adm.get("warnings") or [])
        if not adm.get("can_load", False):
            self.rec["reasons"] = list(adm.get("blockers") or ["the planner rejected this configuration"])
            raise _Stop("rejected", "; ".join(self.rec["reasons"]))

    def launch_and_measure(self) -> None:
        rec, spec, eff = self.rec, self.spec, self.eff
        host_bw = host_bandwidth()
        self.check_cancel()
        eff["port"] = bc.free_port()
        if self.server is None:
            self.server = ServerManager(log_file=self.tmp / f"{self.id}.log")
        self.base = f"http://127.0.0.1:{eff['port']}"
        self.say("Starting the model server")
        t0 = time.perf_counter()
        try:
            self.server.start(eff)
        except RuntimeError as e:
            raise _Stop("failed", str(e)) from e
        self.proc = self.server.proc
        rec["config"]["cmdline"] = " ".join(getattr(self.server, "cmdline", []) or [])
        self.wait_ready()
        rec["load"] = {"seconds": measured(round(time.perf_counter() - t0, 2), "wall clock",
                                           "process start until /health answered 200")}
        self.sampler = make_sampler(self.proc.pid, spec.sample_interval_s, self.sampler_fn)
        if self.sampler is not None:
            self.sampler.start()
        try:
            self.say("Cold run (first request after load)")
            rec["cold"] = self.one_run(0, "cold")
            if spec.warmup:
                self.say("Warm-up run (not counted)")
                self.one_run(-1, "warmup")
            warm = []
            for i in range(1, spec.repeats + 1):
                self.say(f"Warm run {i}/{spec.repeats}")
                warm.append(self.one_run(i, "warm"))
                rec["warm"] = self.warm_stats(warm)    # keep partial results if a later run is cancelled
        finally:
            samples = self.stop_sampler()
            rec["metrics"] = aggregate_samples(samples, host_bw)
        self.say("Collecting results")
        self.finish_analysis()

    def stop_sampler(self) -> list[dict]:
        """Stop the sampler, take one last reading (counters need an end point) and return all readings."""
        sampler = getattr(self, "sampler", None)
        if sampler is None:
            return []
        try:
            sampler.stop()
            sampler.sample_once()
            return sampler.history()
        except Exception:  # noqa: BLE001
            return []

    def wait_ready(self) -> None:
        began = time.perf_counter()
        deadline = began + self.spec.load_timeout_s
        while True:
            self.check_cancel()
            if self.proc.poll() is not None:
                raise _Stop("failed", f"llama-server exited while loading (code {self.proc.returncode}); see the log tail")
            if bc.http_get(self.base, "/health") == 200:    # 503 while loading, None while not listening yet
                return
            if time.perf_counter() > deadline:
                raise _Stop("timeout", f"the model did not finish loading within {int(self.spec.load_timeout_s)} s")
            self.cancel.wait(0.05 if time.perf_counter() - began < 2 else 0.25)

    def one_run(self, index: int, phase: str) -> dict:
        self.check_cancel()
        spec = self.spec
        raw = self.request()
        if raw["outcome"] == "cancelled":
            raise _Stop("cancelled", "cancelled by the user")
        if raw["outcome"] == "timeout":
            raise _Stop("timeout", f"a request took longer than {int(spec.request_timeout_s)} s")
        if self.proc.poll() is not None and raw["outcome"] != "ok":
            raise _Stop("failed", f"llama-server exited during the {phase} run (code {self.proc.returncode})")
        if raw["outcome"] != "ok":
            raise _Stop("failed", raw["error"] or f"the {phase} request failed")
        if raw["chunks"] == 0:
            raise _Stop("failed", f"the server returned no tokens in the {phase} run")
        run = bc.run_metrics(raw, index, phase, spec.prompt_tokens)
        if run["cached_tokens"]:
            self.rec["warnings"].append(f"{phase} run {index}: the server reused {run['cached_tokens']} cached prompt tokens; "
                                        "prompt speed covers only the tokens it processed")
        return run

    def request(self) -> dict:
        spec = self.spec
        endpoints = [self.endpoint] if self.endpoint else [bc.COMPLETION, bc.CHAT]
        raw: dict = {}
        for ep in endpoints:
            payload = bc.completion_payload(ep, self.prompt, spec.gen_tokens, spec.seed, spec.temperature)
            raw = bc.stream_request(self.base, ep, payload, self.cancel, spec.request_timeout_s)
            retry = (raw["outcome"] == "unsupported"
                     or (raw["outcome"] == "error" and not raw["chunks"] and "Disconnected" in raw["error"]))
            if retry and ep == bc.COMPLETION and not self.endpoint:
                continue   # an older server without /completion: use the chat endpoint for every run
            self.endpoint = ep
            break
        return raw

    def warm_stats(self, runs: list[dict]) -> dict:
        return {"runs": runs, "ttft_s": bc.stat_val(runs, "ttft_s", 4), "prompt_tps": bc.stat_val(runs, "prompt_tps", 2),
                "gen_tps": bc.stat_val(runs, "gen_tps", 3)}

    def finish_analysis(self) -> None:
        rec, pl = self.rec, self.planner
        lines = self.log_lines()
        rec["log_tail"] = lines[-LOG_TAIL:]
        try:
            parsed = pl.parse_buffer_log(lines)
            tiers = pl.measured_tiers_gb(parsed)
        except Exception:  # noqa: BLE001
            parsed, tiers = {}, {}
        rec["comparison"] = compare_plan(rec["comparison"].get("estimated", {}), parsed, tiers)
        diag = _mod("pressure")
        try:
            window = self.sampler.summary(window_s=1e9) if getattr(self, "sampler", None) is not None else {}
        except Exception:  # noqa: BLE001
            window = {}
        try:
            rec["bottleneck"] = diag.diagnose_bottleneck(plan=self.plan, window=window, bench=rec, hw=self.hw)
        except Exception as e:  # noqa: BLE001
            rec["bottleneck"] = {"kind": "unknown", "confidence": "low", "evidence": [], "suggestions": [],
                                 "explanation": f"bottleneck diagnosis unavailable: {e}"}

    # ------------------------------------------------------------------ cleanup
    def shutdown(self) -> None:
        """Stop the server for sure: terminate, escalate to kill, and wait for its log pump to drain."""
        sampler = getattr(self, "sampler", None)
        if sampler is not None and getattr(sampler, "running", False):
            sampler.stop()
        server = self.server
        if server is None:
            return
        proc = self.proc
        if proc is None:
            return                  # nothing was started here (rejected, cancelled early, or the start failed)
        if proc.poll() is None:
            try:
                terminate_tree(proc, self.grace)   # the whole tree: on Windows a launcher script would otherwise outlive us
            except Exception:  # noqa: BLE001
                pass
        try:
            server.stop()
        except Exception:  # noqa: BLE001
            pass
        deadline = time.monotonic() + 3
        while proc.stdout is not None and not proc.stdout.closed and time.monotonic() < deadline:
            time.sleep(0.02)   # the log pump closes stdout after the last line: nothing writes the log afterwards
        if self.rec["status"] != "ok" or not self.rec["log_tail"]:
            self.rec["log_tail"] = self.log_lines()[-LOG_TAIL:]


def run_benchmark(settings: dict, model_path, label: str = "", *, hw: dict | None = None, spec: BenchSpec | None = None,
                  server: ServerManager | None = None, cancel: threading.Event | None = None,
                  progress: Callable | None = None, perflog=None, sampler_fn: Callable[[int], dict] | None = None,
                  assumptions: dict | None = None, work_dir: str | Path | None = None,
                  stop_grace_s: float = 8.0) -> dict:
    """Benchmark `model_path` with the full `settings` dict.  Always returns a record (see the module docstring);
    ``status`` is ``ok | rejected | failed | timeout | cancelled`` and ``error``/``reasons`` say why.  The server is
    stopped before this returns.  ``progress(step, total, message)`` and ``cancel.is_set()`` follow `JobRunner`."""
    try:
        bench = _Bench(settings, model_path, label, hw, spec, server, cancel, progress, perflog, sampler_fn, assumptions,
                       work_dir, stop_grace_s)
    except Exception as e:  # noqa: BLE001 - e.g. a spec with non-numeric fields: still a record, never an exception
        bench = _Bench(settings, model_path, label, hw, None, server, cancel, progress, perflog, sampler_fn, assumptions,
                       work_dir, stop_grace_s)
        bench.rec.update(status="failed", error=f"invalid benchmark arguments: {type(e).__name__}: {e}")
        return bench.rec
    return bench.run()


