"""Memory pressure levels, bottleneck diagnosis and a monitor that reports (never performs) protective actions.

`classify_pressure` reads a `telemetry.system_sample`. `diagnose_bottleneck` names the limiting resource from *measured*
signals first (major page faults, CPU and GPU utilisation, clocks, achieved vs available memory bandwidth) and falls back
to the planner's estimate only when nothing was measured - then confidence is "low" and the text says it is an estimate.
`Monitor` watches a `Sampler`, de-bounces level changes and emits events; the caller decides what to do with the action.
"""

from __future__ import annotations

import collections
import threading
import time

from .values import num

LEVELS = ("ok", "elevated", "high", "critical")
ACTIONS = ("warn", "stop_big_model", "stop_all")

DEFAULT_THRESHOLDS = {
    "avail_elevated_pct": 25.0,   # available RAM below this share of total -> elevated
    "avail_high_pct": 12.0,
    "avail_critical_pct": 6.0,
    "avail_critical_gb": 1.5,     # or below this many GB -> critical
    "swap_elevated_pct": 50.0,    # swap in use (only when the machine has at least 1 GB of swap)
    "swap_high_pct": 80.0,
    "commit_elevated_pct": 90.0,  # Windows commit charge against its limit
    "commit_high_pct": 97.0,
    "psi_elevated": 10.0,         # Linux: % of the last 10 s some task stalled waiting for memory
    "psi_high": 25.0,
    "psi_critical": 50.0,
}


def _rank(level: str) -> int:
    return LEVELS.index(level)


def classify_pressure(sample: dict, thresholds: dict | None = None) -> dict:
    """Memory pressure of one system sample.

    Defaults (see `DEFAULT_THRESHOLDS`): available RAM < 25% of total is elevated, < 12% high, < 6% or < 1.5 GB critical.
    Swap use, the Windows commit charge and Linux memory-stall time can raise the level but never lower it. When the
    counters could not be read the level is "ok" with ``known=False`` - absence of data is not evidence of safety.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    ram = (sample or {}).get("ram") or {}
    total, avail = num(ram.get("total_gb")), num(ram.get("avail_gb"))
    reasons: list[str] = []
    level = "ok"
    known = total is not None and avail is not None and total > 0
    pct = None

    def raise_to(new: str, why: str) -> None:
        nonlocal level
        reasons.append(why)
        if _rank(new) > _rank(level):
            level = new

    if known:
        pct = round(100.0 * avail / total, 1)
        if pct < th["avail_critical_pct"] or avail < th["avail_critical_gb"]:
            raise_to("critical", f"Only {avail:.1f} GB ({pct:.0f}%) of RAM is available.")
        elif pct < th["avail_high_pct"]:
            raise_to("high", f"Only {avail:.1f} GB ({pct:.0f}%) of RAM is available.")
        elif pct < th["avail_elevated_pct"]:
            raise_to("elevated", f"{avail:.1f} GB ({pct:.0f}%) of RAM is available.")
    else:
        reasons.append("Memory counters could not be read; no pressure can be judged.")
    swap = (sample or {}).get("swap") or {}
    sp, st = num(swap.get("used_pct")), num(swap.get("total_gb"))
    if sp is not None and st is not None and st >= 1:
        if sp >= th["swap_high_pct"]:
            raise_to("high", f"Swap is {sp:.0f}% full.")
        elif sp >= th["swap_elevated_pct"]:
            raise_to("elevated", f"Swap is {sp:.0f}% full.")
    commit = (sample or {}).get("commit") or {}
    cp = num(commit.get("used_pct"))
    csrc = str((commit.get("used_pct") or {}).get("source", "")) if isinstance(commit.get("used_pct"), dict) else ""
    if cp is not None and not csrc.startswith("/proc"):  # Linux overcommit makes Committed_AS > limit normal
        if cp >= th["commit_high_pct"]:
            raise_to("high", f"The commit charge is {cp:.0f}% of its limit; allocations may start to fail.")
        elif cp >= th["commit_elevated_pct"]:
            raise_to("elevated", f"The commit charge is {cp:.0f}% of its limit.")
    psi = num(ram.get("pressure_pct"))
    if psi is not None:
        if psi >= th["psi_critical"]:
            raise_to("critical", f"Tasks were stalled on memory {psi:.0f}% of the last 10 s.")
        elif psi >= th["psi_high"]:
            raise_to("high", f"Tasks were stalled on memory {psi:.0f}% of the last 10 s.")
        elif psi >= th["psi_elevated"]:
            raise_to("elevated", f"Tasks were stalled on memory {psi:.0f}% of the last 10 s.")
    if known and level == "ok":
        reasons.append(f"{avail:.1f} GB ({pct:.0f}%) of RAM is available.")
    return {"level": level, "avail_gb": avail, "avail_pct": pct, "known": known, "reasons": reasons}


# --------------------------------------------------------------------------------------------- bottleneck
_SUGGESTIONS = {
    "storage_paging": [
        "The model does not fit in RAM: use a smaller model or quantization, or add RAM.",
        "Lower the context size and close other applications to leave more RAM for the model.",
        "Keep the model on an SSD/NVMe drive; a hard drive is unusably slow for paging.",
        "A sequential page-cache prewarm helps only if the model fits in the page cache."],
    "memory_bandwidth": [
        "Token generation reads the active weights from RAM every token: bandwidth is the limit.",
        "Enable XMP/DOCP in the BIOS and use matched sticks in dual channel; faster RAM raises tokens/s directly.",
        "A smaller quantization reads fewer bytes per token."],
    "compute": [
        "The CPU is saturated: use a build with AVX2/AVX-512, make sure threads equal the physical core count.",
        "Prompt processing is compute-bound; offloading to a GPU helps it most."],
    "vram_capacity": [
        "The GPU memory is full: lower n_gpu_layers, keep the KV cache in RAM, or add --no-op-offload.",
        "Use a smaller context or a quantized KV cache."],
    "pcie_transfer": [
        "Data is moving between GPU and RAM every layer: compare against n_gpu_layers=0 and against fully offloading.",
        "Partial offload only pays off when the GPU part is large; the transfer rate cannot be measured through llama-server."],
    "thermal": [
        "The GPU is hot or its clock dropped: check cooling and the power limit, then re-run the benchmark."],
    "cpu_threads": [
        "Fewer threads than physical cores are in use: set threads to the physical core count.",
        "If CPU use is low and speed is poor, check process priority and background load."],
    "unknown": [
        "Run a benchmark with the sampler enabled so the limiting resource can be measured."],
}


def _dig(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _first(*vals):
    for v in vals:
        n = num(v)
        if n is not None:
            return n
    return None


def diagnose_bottleneck(plan: dict | None = None, window: dict | None = None, bench: dict | None = None,
                        hw: dict | None = None) -> dict:
    """Which resource limits generation speed? ``{"kind","confidence","explanation","evidence","suggestions","basis"}``.

    ``window`` is `Sampler.summary()`, ``bench`` a benchmark record, ``plan`` a `planner.plan` result and ``hw`` the
    hardware dict. Measured signals decide; the planner's estimate is used only when nothing was measured. Malformed
    inputs give ``kind="unknown"`` instead of raising.
    """
    plan, window, bench, hw = (x if isinstance(x, dict) else None for x in (plan, window, bench, hw))
    try:
        return _diagnose(plan, window, bench, hw)
    except Exception as e:
        return {"kind": "unknown", "confidence": "low", "basis": "none",
                "explanation": f"The diagnosis could not be computed from the data given ({type(e).__name__}).",
                "evidence": [], "suggestions": list(_SUGGESTIONS["unknown"])}


def _diagnose(plan, window, bench, hw) -> dict:
    evidence: list[str] = []
    votes: dict[str, int] = collections.Counter()
    notes: list[str] = []
    w, b = window or {}, bench or {}
    metrics = b.get("metrics") or {}
    hw = hw or b.get("hardware") or {}
    phys = _first(_dig(hw, "cpu", "physical_cores"))

    # ---- measured: storage paging
    major = _first(_dig(w, "major_fault_rate_per_s"))
    io_rate = _first(_dig(w, "io_read_mb_per_s"))
    avail_min = _first(_dig(w, "ram_avail_gb", "min"))
    swap_max = _first(_dig(w, "swap_used_gb", "max"))
    if major is not None and major >= 20:
        votes["storage_paging"] += 2
        evidence.append(f"{major:.0f} major page faults per second (measured): the process keeps reading weights from storage.")
    if io_rate is not None and io_rate >= 50:
        votes["storage_paging"] += 1
        evidence.append(f"The process reads {io_rate:.0f} MB/s from storage while running (measured).")
    if swap_max is not None and swap_max >= 1 and avail_min is not None and avail_min < 1.5:
        votes["storage_paging"] += 1
        evidence.append(f"Swap holds {swap_max:.1f} GB and RAM fell to {avail_min:.1f} GB available (measured).")
    size_gb = _first(_dig(b, "model", "size_bytes"))
    size_gb = size_gb / 2**30 if size_gb else _first(_dig(plan, "model", "size_gb"))
    io_total = _first(metrics.get("io_read_gb"))
    if io_total is not None and size_gb and io_total >= 1.25 * size_gb:
        votes["storage_paging"] += 1
        evidence.append(f"The run read {io_total:.1f} GB from storage, {io_total / size_gb:.1f}x the model size (measured): "
                        "pages were evicted and read again.")

    # ---- measured: GPU memory, thermals, utilisation
    cfg = b.get("config") or {}
    ngl = _first(cfg.get("n_gpu_layers"), _dig(plan, "config", "n_gpu_layers"))
    gpu_layers = _first(_dig(plan, "placement", "gpu_layers"))
    offloaded = bool(hw.get("gpus")) and ((gpu_layers or 0) > 0 or (ngl or 0) != 0)
    vram_used = _first(_dig(w, "vram_used_mb", "max"), metrics.get("peak_vram_mb"))
    gpus = hw.get("gpus") or []
    vram_total = _first(gpus[0].get("vram_mb")) if gpus else None
    if vram_used is not None and vram_total and vram_used >= 0.95 * vram_total and offloaded:
        votes["vram_capacity"] += 2
        evidence.append(f"GPU memory peaked at {vram_used:.0f} of {vram_total:.0f} MiB (measured): no headroom left.")
    temp, clk_min, clk_max = _first(_dig(w, "gpu_temp_c", "max")), _first(_dig(w, "gpu_sm_clock_mhz", "min")), _first(_dig(w, "gpu_sm_clock_mhz", "max"))
    if temp is not None and temp >= 83:
        votes["thermal"] += 2
        evidence.append(f"The GPU reached {temp:.0f} degC (measured).")
    if clk_min and clk_max and clk_max > 0 and clk_min < 0.75 * clk_max:
        votes["thermal"] += 1
        evidence.append(f"The GPU clock dropped from {clk_max:.0f} to {clk_min:.0f} MHz during the run (measured).")
    cpu = _first(_dig(w, "cpu_util_pct", "mean"), metrics.get("cpu_util_pct"))
    gpu_util = _first(_dig(w, "gpu_util_pct", "mean"), metrics.get("gpu_util_pct"))
    if cpu is not None:
        evidence.append(f"Mean CPU utilisation {cpu:.0f}% (measured).")
    if gpu_util is not None:
        evidence.append(f"Mean GPU utilisation {gpu_util:.0f}% (measured).")

    # ---- measured: achieved vs available memory bandwidth
    tps = _first(_dig(b, "warm", "gen_tps", "mean"), _dig(b, "warm", "gen_tps"))
    active = next((_first(a.get("value", {}).get("total")) for a in (plan or {}).get("assumptions", [])
                   if a.get("key") == "speed.active_gb_per_token" and isinstance(a.get("value"), dict)), None)
    host_bw = _first(metrics.get("host_bandwidth_gbs"))
    theo_bw = _first(_dig(hw, "ram", "bandwidth_gbs"))
    if tps and active and not votes.get("storage_paging"):
        eff = tps * active
        ref = max(host_bw or 0, 0)
        ref_name = "measured single-thread copy rate"
        if theo_bw and (not ref or theo_bw > ref):
            ref, ref_name = theo_bw, "theoretical RAM bandwidth"
        if ref:
            ratio = eff / ref
            evidence.append(f"Generation reads about {eff:.1f} GB/s of weights ({tps:.2f} tokens/s x {active:.2f} GB) = "
                            f"{ratio * 100:.0f}% of the {ref_name} ({ref:.1f} GB/s) (measured speed, estimated bytes).")
            if ratio >= 0.4:
                votes["memory_bandwidth"] += 3  # near the ceiling: nothing else can be the limit
            elif ratio >= 0.15:
                votes["memory_bandwidth"] += 1
            elif cpu is not None and cpu >= 90:
                votes["compute"] += 1
    if cpu is not None and cpu < 60 and phys and (cfg.get("threads") or 0) and cfg["threads"] < phys:
        votes["cpu_threads"] += 2
        evidence.append(f"Only {cfg['threads']} threads for {phys:.0f} physical cores with {cpu:.0f}% CPU use (measured).")
    if offloaded and gpu_util is not None and gpu_util < 30 and cpu is not None and cpu >= 70:
        votes["pcie_transfer"] += 1
        evidence.append("The GPU is mostly idle while the CPU is busy although layers are offloaded: the work waits on the "
                        "CPU side or on transfers (transfer bandwidth itself cannot be measured).")

    if votes:
        kind, score = max(votes.items(), key=lambda kv: kv[1])
        confidence = "high" if score >= 3 else "medium" if score == 2 else "low"
        explanation = {
            "storage_paging": "The model is being paged in from storage while generating; RAM is too small for the weights in use.",
            "memory_bandwidth": "Generation is limited by how fast RAM can deliver the active weights.",
            "compute": "The CPU is saturated and memory bandwidth is not the limit.",
            "vram_capacity": "The GPU memory is full, which limits how much can be offloaded.",
            "pcie_transfer": "The GPU is waiting on the CPU side or on host-GPU transfers.",
            "thermal": "The GPU is running hot or throttling.",
            "cpu_threads": "The CPU is under-used: too few threads are configured.",
        }[kind]
        return {"kind": kind, "confidence": confidence, "explanation": explanation, "evidence": evidence,
                "suggestions": list(_SUGGESTIONS[kind]), "basis": "measured"}

    # ---- fallback: the planner's estimate
    bound = _dig(plan, "speed", "bound_by")
    mapped = {"storage": "storage_paging", "memory_bandwidth": "memory_bandwidth", "vram": "memory_bandwidth",
              "compute": "compute"}.get(bound)
    if mapped:
        spill = _first(_dig(plan, "tiers", "storage", "spill_gb", "value"), _dig(plan, "tiers", "storage", "spill_gb"))
        notes.append(f"Planner estimate: speed is bound by {bound}" + (f", with {spill:.1f} GB expected to page from storage." if mapped == "storage_paging" and spill else "."))
        return {"kind": mapped, "confidence": "low",
                "explanation": "No measurements were available, so this comes from the planner's estimate alone and may be wrong. "
                               + _EST_TEXT[mapped],
                "evidence": (evidence or []) + notes + ["Run a benchmark with the sampler to confirm."],
                "suggestions": list(_SUGGESTIONS[mapped]), "basis": "estimated"}
    return {"kind": "unknown", "confidence": "low",
            "explanation": "Not enough information to name a bottleneck: no measurements and no plan were provided.",
            "evidence": evidence, "suggestions": list(_SUGGESTIONS["unknown"]), "basis": "none"}


_EST_TEXT = {"storage_paging": "The weights are expected not to fit in RAM.",
             "memory_bandwidth": "Generation is expected to be limited by memory bandwidth.",
             "compute": "Compute is expected to be the limit."}


# ----------------------------------------------------------------------------------------------------- monitor
class Monitor:
    """Watches a `Sampler`, turns samples into pressure levels and emits events when the level really changes.

    Level changes are de-bounced: a new level must be seen in ``confirm_up`` (rising) or ``confirm_down`` (falling)
    consecutive readable samples. Events are ``{"type": "pressure", "level", "previous", "sample", "action", "reasons",
    "t"}``. ``action`` is ``"none"`` below high, ``"warn"`` at high, and at critical the configured
    ``vgpu_pressure_action`` (warn, stop_big_model or stop_all). The monitor never stops anything itself.
    """

    def __init__(self, sampler, settings_fn=None, on_event=None, *, poll_s: float = 1.0, confirm_up: int = 2,
                 confirm_down: int = 3, max_events: int = 100):
        self.sampler, self._settings_fn, self._on_event = sampler, settings_fn or (dict), on_event
        self.poll_s, self.confirm_up, self.confirm_down = max(0.02, poll_s), max(1, confirm_up), max(1, confirm_down)
        self.level, self.since = "ok", time.time()
        self.blind_samples = 0
        self.callback_errors = 0
        self._pending: tuple[str, int] | None = None
        self._last_t = 0.0
        self._events: collections.deque = collections.deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- settings
    def _settings(self) -> dict:
        try:
            return self._settings_fn() or {}
        except Exception:
            return {}

    def action_for(self, level: str) -> str:
        if level == "critical":
            act = str(self._settings().get("vgpu_pressure_action", "warn"))
            return act if act in ACTIONS else "warn"
        return "warn" if level == "high" else "none"

    # ---- evaluation
    def evaluate(self, sample: dict) -> dict | None:
        """Feed one sample; returns the emitted event when the confirmed level changed, else None."""
        th = self._settings().get("vgpu_pressure_thresholds")
        c = classify_pressure(sample, th if isinstance(th, dict) else None)
        if not c["known"]:
            self.blind_samples += 1
            return None
        new = c["level"]
        event = None
        with self._lock:
            if new == self.level:
                self._pending = None
                return None
            need = self.confirm_up if _rank(new) > _rank(self.level) else self.confirm_down
            count = self._pending[1] + 1 if self._pending and self._pending[0] == new else 1
            if count < need:
                self._pending = (new, count)
                return None
            previous, self.level, self.since, self._pending = self.level, new, time.time(), None
            event = {"type": "pressure", "level": new, "previous": previous, "sample": sample, "action": self.action_for(new),
                     "reasons": c["reasons"], "t": time.time()}
            self._events.append(event)
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                self.callback_errors += 1
        return event

    def poll(self) -> list[dict]:
        """Evaluate every sampler sample newer than the last one seen; returns the events emitted."""
        out = []
        for s in self.sampler.history():
            t = s.get("t", 0)
            if t <= self._last_t:
                continue
            self._last_t = t
            ev = self.evaluate(s)
            if ev:
                out.append(ev)
        return out

    def recent_events(self, n: int = 20) -> list[dict]:
        with self._lock:
            return list(self._events)[-max(n, 0):] if n else []

    def state(self) -> dict:
        with self._lock:
            return {"level": self.level, "since": self.since, "pending": self._pending, "action": self.action_for(self.level),
                    "blind_samples": self.blind_samples, "running": bool(self._thread and self._thread.is_alive())}

    # ---- thread
    def _run(self, stop: threading.Event) -> None:
        while not stop.wait(self.poll_s):
            try:
                self.poll()
            except Exception:
                self.callback_errors += 1

    def start(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(self._stop,), daemon=True, name="vgpu-monitor")
            self._thread.start()
            return True

    def stop(self, timeout: float = 3.0) -> None:
        with self._lock:
            t, ev = self._thread, self._stop
        ev.set()
        if t and t is not threading.current_thread():
            t.join(timeout)
