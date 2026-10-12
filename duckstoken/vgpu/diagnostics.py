"""A copyable diagnostic report: everything needed to understand a performance or memory problem, with secrets removed.

`build_report(app_ctx)` gathers hardware, the compatibility report, the engine probe, settings, the planner's verdict for
the selected model, recorded failures, the recent performance log and log tails. Each section is built on its own, so one
failing source never stops the others. `to_text(report)` renders plain text for pasting into a bug report.

``app_ctx`` keys (all optional): ``hardware``, ``settings``, ``engine_probe``, ``model`` (a `gguf.summarize` dict),
``plan`` (precomputed), ``compat`` (precomputed), ``perflog`` (a `PerfLog` or a path), ``logs`` (name -> text or lines),
``telemetry`` (a system sample or `Sampler.summary()`), ``app_version``, ``extra`` (any dict).
"""

from __future__ import annotations

import collections
import json
import os
import platform
import re
import threading
import time
from pathlib import Path

from .. import hardware
from . import gpucompat, perflog, planner

REDACTED = "***REDACTED***"
MAX_LOG_LINES = 200
MAX_VALUE_CHARS = 400

# Setting/field names that hold credentials. Written to catch api_key, hf_token, brave_api_key, password ... but not
# tokens_per_s, max_tokens or key_length, which are numbers about the model.
_SECRET_KEY = re.compile(r"(password|passwd|passphrase|secret|credential|cookie|bearer|private_key|apikey|"
                         r"(^|_)(api_?key|key|token|auth|authorization|authkey)$)", re.IGNORECASE)
_keep = lambda m: m.group(1) + m.group(2) + REDACTED
_SECRET_VALUE = [
    (re.compile(r"(?i)\b(bearer)(\s+)[A-Za-z0-9._\-~+/=]{8,}"), _keep),
    (re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"), lambda m: REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"), lambda m: REDACTED),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), lambda m: REDACTED),
    # key=value / key: value in logs and URLs
    (re.compile(r"(?i)\b(api[_-]?key|token|password|secret|authorization)(\s*[=:]\s*)(?!\*\*\*)[^\s,;&\"']{4,}"), _keep),
    # command lines: --api-key VALUE, --hf-token=VALUE
    (re.compile(r"(?i)(--[a-z\-]*(?:api-key|token|password|secret)[a-z\-]*)(\s+|=)(?!--|\*\*\*)[^\s\"']{4,}"), _keep),
]
_USER_PATHS = [
    re.compile(r"(?i)([A-Za-z]:[\\/]+Users[\\/]+)([^\\/\"']+?)(?=[\\/\"']|$)"),  # Windows user names may contain spaces
    re.compile(r"(/home/)([^/\s\"']+)"),
    re.compile(r"(/Users/)([^/\s\"']+)"),
]

_failures: collections.deque = collections.deque(maxlen=50)
_failures_lock = threading.Lock()


def is_secret_key(name) -> bool:
    return bool(_SECRET_KEY.search(str(name)))


def redact_text(text: str, secrets: tuple | list = ()) -> str:
    """Remove known secret values, credential-looking strings and user names inside paths from free text."""
    if not isinstance(text, str):
        return text
    for sec in secrets:
        if isinstance(sec, str) and len(sec) >= 4:
            text = text.replace(sec, REDACTED)
    for pat, repl in _SECRET_VALUE:
        text = pat.sub(repl, text)
    for pat in _USER_PATHS:
        text = pat.sub(lambda m: m.group(1) + "<user>", text)
    return text


def _secret_values(obj, found=None) -> list[str]:
    found = [] if found is None else found
    if isinstance(obj, dict):
        for k, v in obj.items():
            if is_secret_key(k) and isinstance(v, str) and len(v) >= 4:
                found.append(v)
            else:
                _secret_values(v, found)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _secret_values(v, found)
    return found


def redact(obj, secrets: tuple | list | None = None, _depth: int = 0):
    """A copy of `obj` with secrets removed: values under credential-like keys, known secret values anywhere in text,
    bearer tokens / API keys in strings, and user names inside file paths. Numbers and structure are preserved."""
    secrets = list(secrets) if secrets is not None else _secret_values(obj)
    if _depth > 12:
        return "..."
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if is_secret_key(k) and not isinstance(v, (dict, list, tuple)):
                out[k] = REDACTED if v not in (None, "", [], {}) else v
            else:
                out[k] = redact(v, secrets, _depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(v, secrets, _depth + 1) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj, secrets)
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    return redact_text(str(obj), secrets)


# ------------------------------------------------------------------------------------------------ failures
def record_failure(kind: str, message: str, context: dict | None = None) -> None:
    """Remember a failed estimate, admission or benchmark for the next report (ring buffer of the last 50). Never raises."""
    try:
        rec = {"t": time.time(), "kind": str(kind)[:60], "message": str(message)[:MAX_VALUE_CHARS * 2],
               "context": perflog._clean(context or {})}
        with _failures_lock:
            _failures.append(rec)
    except Exception:
        pass


def failures(n: int = 50) -> list[dict]:
    with _failures_lock:
        return list(_failures)[-max(int(n), 0):] if n else []


def clear_failures() -> None:
    with _failures_lock:
        _failures.clear()


# ---------------------------------------------------------------------------------------------------- report
def _section(fn, what: str):
    try:
        return fn()
    except Exception as e:
        return {"error": f"{what} could not be collected: {type(e).__name__}: {e}"}


def _tail_lines(v, n: int = MAX_LOG_LINES) -> list[str]:
    if isinstance(v, str):
        v = v.splitlines()
    return [str(x)[:MAX_VALUE_CHARS * 2] for x in list(v or [])[-n:]]


def _perf_tail(src, n: int = 50) -> list[dict]:
    if src is None:
        return []
    log = src if hasattr(src, "tail") else perflog.PerfLog(Path(str(src)))
    return log.tail(n)


def build_report(app_ctx: dict | None = None) -> dict:
    """Collect the diagnostic report (see the module docstring for `app_ctx`). Secrets are redacted before it is returned."""
    ctx = app_ctx or {}
    settings = ctx.get("settings") or {}
    hw = ctx.get("hardware")
    if hw is None and ctx.get("collect_hardware", False):
        hw = _section(hardware.get_hardware, "hardware")
    model = ctx.get("model")
    probe = ctx.get("engine_probe")

    def compat():
        if ctx.get("compat") is not None:
            return ctx["compat"]
        return gpucompat.compatibility_report(hw, probe, model) if hw else {"error": "no hardware information supplied"}

    def plan():
        if ctx.get("plan") is not None:
            return ctx["plan"]
        if not (hw and model):
            return None
        cfg = {k: settings[k] for k in planner.CFG_KEYS if k in settings}
        return planner.plan(hw, model, cfg, model_path=model.get("path") or None)

    report = {
        "title": "Duckstoken VirtualGPU diagnostic report",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "app": {"version": ctx.get("app_version"), "os": (hw or {}).get("os") if isinstance(hw, dict) else None,
                "python": platform.python_version(), "pid": os.getpid()},
        "hardware": hw,
        "compat": _section(compat, "compatibility report"),
        "engine": probe,
        "settings": settings,
        "model": model,
        "plan": _section(plan, "memory plan"),
        "failures": failures(),
        "perflog": _section(lambda: _perf_tail(ctx.get("perflog")), "performance log"),
        "telemetry": ctx.get("telemetry"),
        "logs": {name: _tail_lines(lines) for name, lines in (ctx.get("logs") or {}).items()},
        "extra": ctx.get("extra"),
        "notes": ["Values marked 'estimated' are calculations, not measurements; 'unavailable' means the source could not be read.",
                  "Secrets (API keys, tokens, passwords) and user names inside paths were removed from this report."],
    }
    return redact(report, _secret_values(settings) + _secret_values(ctx.get("extra") or {}))


# ------------------------------------------------------------------------------------------------------- text
def _v(x):
    """Plain text for a value dict or scalar."""
    if isinstance(x, dict) and "status" in x and "value" in x:
        if x["status"] == "unavailable" or x["value"] is None:
            return f"unavailable ({x.get('note') or 'no reason given'})"
        return f"{x['value']} [{x['status']}]"
    return json.dumps(x, default=str) if isinstance(x, (dict, list)) else str(x)


def _short(x, n: int = MAX_VALUE_CHARS) -> str:
    t = _v(x).replace("\n", " ")
    return t if len(t) <= n else t[:n] + "..."


def _hw_text(hw) -> list[str]:
    if not isinstance(hw, dict) or "error" in hw:
        return [f"  {hw}"]
    cpu, ram = hw.get("cpu", {}), hw.get("ram", {})
    out = [f"  OS: {hw.get('os')}", f"  CPU: {cpu.get('name')} ({cpu.get('physical_cores')}C/{cpu.get('logical_cores')}T, "
           f"AVX2={cpu.get('avx2')}, AVX512={cpu.get('avx512')})",
           f"  RAM: {ram.get('total_gb')} GB total, {ram.get('avail_gb')} GB free, {ram.get('speed_mts')} MT/s x {ram.get('sticks')} "
           f"stick(s), ~{ram.get('bandwidth_gbs')} GB/s theoretical"]
    for g in hw.get("gpus") or []:
        out.append(f"  GPU: {g.get('name')} ({g.get('vram_mb')} MiB, driver {g.get('driver')}, compute {g.get('compute_cap') or '?'})")
    if not hw.get("gpus"):
        out.append("  GPU: none detected")
    for d in hw.get("disks") or []:
        out.append(f"  Disk {d.get('drive')}: {d.get('kind')} {d.get('model') or ''} {d.get('free_gb')} GB free of {d.get('size_gb')}")
    if hw.get("mock"):
        out.append(f"  (what-if hardware profile: {hw.get('preset') or 'custom'})")
    return out


def to_text(report: dict) -> str:
    """Plain-text rendering of `build_report` output, sectioned for pasting into an issue."""
    L: list[str] = [report.get("title", "Diagnostic report"), f"Generated: {report.get('generated_at')}"]
    app = report.get("app") or {}
    L.append(f"App version: {app.get('version') or 'unknown'}   Python {app.get('python')}")
    L.append("")
    L.append("== Hardware ==")
    L += _hw_text(report.get("hardware"))

    comp = report.get("compat")
    L += ["", "== Compatibility =="]
    if isinstance(comp, dict) and "items" in comp:
        L.append(f"  {comp.get('summary', '')}")
        for i in comp["items"]:
            L.append(f"  [{i['status']:12s}] {i['label']} ({i['evidence']}): {i['detail']}")
        for g in comp.get("gpu", []):
            L.append(f"  GPU {g['name']}: {g['architecture']}, compute {g['compute_cap'] or '?'}, legacy={g['legacy']}")
    else:
        L.append(f"  {comp}")

    eng = report.get("engine")
    L += ["", "== Engine =="]
    if isinstance(eng, dict):
        L.append(f"  {eng.get('path')} exists={eng.get('exists')} status={eng.get('status')} version={eng.get('version')} "
                 f"build={eng.get('build')} commit={eng.get('commit')}")
        L.append(f"  backends={eng.get('backends')} usable={eng.get('backends_usable')} supports={eng.get('supports')}")
        for d in eng.get("devices") or []:
            L.append(f"  device {d['id']}: {d['name']} ({d['total_mb']} MiB, {d['free_mb']} MiB free)")
        for e in eng.get("errors") or []:
            L.append(f"  probe error: {e}")
    else:
        L.append("  not probed")

    L += ["", "== Settings =="]
    for k in sorted(report.get("settings") or {}):
        L.append(f"  {k} = {_short(report['settings'][k], 200)}")

    L += ["", "== Selected model =="]
    m = report.get("model")
    if isinstance(m, dict):
        keys = ("path", "name", "architecture", "size_gb", "block_count", "expert_count", "expert_used_count", "context_length",
                "file_type", "file_type_name")
        L.append("  " + ", ".join(f"{k}={m.get(k)}" for k in keys if k in m))
    else:
        L.append("  none selected")

    L += ["", "== Memory plan =="]
    p = report.get("plan")
    if isinstance(p, dict) and "admission" in p:
        adm, t = p["admission"], p["tiers"]
        L.append(f"  verdict={adm['verdict']} can_load={adm['can_load']} headroom={adm['headroom_gb']} GB")
        L.append(f"  config: {p['config']}")
        L.append(f"  VRAM planned {_short(t['vram']['planned_gb'])} of budget {_short(t['vram']['budget_gb'])} (card {_short(t['vram']['physical_gb'])})")
        L.append(f"  RAM planned {_short(t['ram']['planned_gb'])} of budget {_short(t['ram']['budget_gb'])} (installed {_short(t['ram']['physical_gb'])}, "
                 f"free now {_short(t['ram']['available_gb'])})")
        L.append(f"  storage: {t['storage']['kind']} {t['storage']['drive']}, spill {_short(t['storage']['spill_gb'])}")
        L.append(f"  speed: {_short(p['speed']['tokens_per_s'])} t/s, bound by {p['speed']['bound_by']}")
        for label in ("blockers", "warnings", "suggestions"):
            for x in adm[label]:
                L.append(f"  {label[:-1]}: {x}")
    elif p is None:
        L.append("  no model selected")
    else:
        L.append(f"  {p}")

    L += ["", f"== Recorded failures ({len(report.get('failures') or [])}) =="]
    for f in report.get("failures") or []:
        stamp = time.strftime("%H:%M:%S", time.gmtime(f.get("t", 0)))
        L.append(f"  {stamp} [{f.get('kind')}] {f.get('message')}" + (f"  {json.dumps(f['context'], default=str)}" if f.get("context") else ""))

    L += ["", "== Recent performance log =="]
    pl = report.get("perflog")
    if isinstance(pl, list):
        for r in pl[-25:]:
            rest = {k: v for k, v in r.items() if k not in ("t", "event")}
            L.append(f"  {time.strftime('%H:%M:%S', time.gmtime(r.get('t', 0)))} {r.get('event')} {json.dumps(rest, default=str)[:300]}")
        if not pl:
            L.append("  (empty)")
    else:
        L.append(f"  {pl}")

    tel = report.get("telemetry")
    if tel:
        L += ["", "== Telemetry =="]
        L.append("  " + _short(tel, 1500))

    for name, lines in (report.get("logs") or {}).items():
        L += ["", f"== Log: {name} (last {len(lines)} lines) =="]
        L += [f"  {x}" for x in lines]

    if report.get("extra"):
        L += ["", "== Extra =="]
        L.append("  " + _short(report["extra"], 1500))
    L += [""] + [f"Note: {n}" for n in report.get("notes", [])]
    return "\n".join(L)
