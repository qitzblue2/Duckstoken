"""Local HTTP server: serves the web UI and a JSON API. Stdlib only."""

from __future__ import annotations

import json
import mimetypes
import os
import secrets
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__, api_learning, api_system, api_vgpu, bundle, context, downloads, gguf, hardware, paths, settings as st, tuner, updater, websearch
from .api_learning import LearningApi, post
from .api_system import SystemApi, UpdateJob
from .api_vgpu import VGpuApi
from .dataset import DatasetStore
from .db import Database
from .engines import ENGINE_LABELS, EngineSwitch, load_engines
from .errors import ApiError
from .escalation import EscalationWorker, QueueStore
from .feedback import ExchangeStore
from .memory import MemoryStore
from .server_proc import ServerManager
from .tools import TOOLS, ToolRunner
from .vgpu.diskcache import BoundedCache
from .vgpu.meter import RequestMeter
from .vgpu.runtime import VGpuRuntime

WEB = Path(__file__).parent / "web"
HF_QUICK = [
    {"label": "gpt-oss-120b (~59 GB)", "repo": "ggml-org/gpt-oss-120b-GGUF"},
    {"label": "gpt-oss-20b (~11 GB)", "repo": "ggml-org/gpt-oss-20b-GGUF"},
]


class App:
    def __init__(self):
        self.settings = st.SettingsStore()
        self.profiles = st.ProfileStore()
        self.engines = load_engines()
        self.server = EngineSwitch(self.engines, self.settings.all)  # whichever engine the settings select
        self.big = ServerManager(log_file=paths.big_server_log_file())  # the big model is always llama.cpp
        self.jobs = downloads.JobManager()
        self.db = Database()
        self.memory = MemoryStore(self.db)
        self.exchanges = ExchangeStore(self.db, self.memory)
        self.queue = QueueStore(self.db)
        self.dataset = DatasetStore(self.db, self.exchanges)
        self.worker = EscalationWorker(self.db, self.settings.all, self.server, self.big, self.exchanges, self.queue)
        self.tools = ToolRunner(self.settings.all, memory=self.memory, web=websearch)
        self.updater = updater.UpdateChecker(self.settings.all, paths.data_dir())
        self.update_job = UpdateJob()
        self.last_request: dict = {}  # speed of the latest chat reply (measured), shown on the Dashboard
        self.meta_cache = BoundedCache(max_bytes=4 << 20, max_items=64)  # parsed GGUF headers (bounded, with hit/miss stats)
        self.vgpu = VGpuRuntime(self)
        api_learning.bind(self)
        api_system.bind(self)
        api_vgpu.bind(self)
        self.token = secrets.token_urlsafe(24)
        self.httpd: ThreadingHTTPServer | None = None
        self.port = 0

    # ----------------------------------------------------------- helpers
    def model_info(self, path: str) -> dict:
        p = Path(path)
        if not p.is_file():
            raise ApiError("Model file not found")
        key = f"{p}|{p.stat().st_mtime_ns}"
        info = self.meta_cache.get(key)
        if info is None:
            try:
                info = gguf.summarize(p)
            except gguf.GGUFError as e:
                raise ApiError(f"Not a usable GGUF model: {e}") from e
            self.meta_cache.put(key, info, size=4096)
        return info

    def installed_models(self) -> list[dict]:
        root = self.settings.models_dir()
        out = []
        if root.is_dir():
            for f in sorted(root.rglob("*.gguf")):
                m = downloads._SHARD.match(f.name)
                if m and int(m["n"]) != 1:
                    continue
                files = gguf.shard_files(f)
                size = sum(x.stat().st_size for x in files)
                low = f.name.lower()
                out.append({"path": str(f), "name": f.name, "size_gb": round(size / 2**30, 2),
                            "shards": len(files),
                            "kind": "draft" if ("eagle" in low or "draft" in low) else
                                    "vision" if "mmproj" in low else "model"})
        return out

    def apply_patch(self, patch: dict) -> dict:
        try:
            applied = self.settings.update(patch)
        except ValueError as e:
            raise ApiError(str(e)) from e
        if any(k.startswith("vgpu_") for k in applied):
            self.vgpu.settings_changed(applied)
        return applied

    def chat_payload(self, body: dict) -> dict:
        s = self.settings.all()
        msgs, info = context.augment(list(body.get("messages") or []), s, self.memory, self.exchanges, websearch)
        p = {"model": self.server.model_name(), "messages": msgs, "stream": bool(s["stream"]),
             "temperature": s["temperature"], "top_p": s["top_p"], "top_k": s["top_k"], "min_p": s["min_p"],
             "repeat_penalty": s["repeat_penalty"], "presence_penalty": s["presence_penalty"],
             "frequency_penalty": s["frequency_penalty"],
             "reasoning_effort": s["reasoning_effort"],
             "chat_template_kwargs": {"reasoning_effort": s["reasoning_effort"]}}
        if int(s["mirostat"]):
            p.update(mirostat=int(s["mirostat"]), mirostat_tau=s["mirostat_tau"], mirostat_eta=s["mirostat_eta"])
        if s["seed"] >= 0:
            p["seed"] = s["seed"]
        if s["max_tokens"] > 0:
            p["max_tokens"] = s["max_tokens"]
        if body.get("use_tools", True):
            defs = self.tools.definitions()
            if defs:
                p["tools"] = defs
        p["_duck_context"] = info
        return p

    def note_request(self, meter: RequestMeter) -> None:
        """Remember how fast the latest reply was (measured from the real stream) and log it."""
        try:
            r = meter.result(self.settings.get("engine"))
            if r["ttft_s"]["value"] is None:
                return
            self.last_request = r
            if self.vgpu is not None and self.settings.get("vgpu_perf_log"):
                self.vgpu.perflog.log("chat_request", **{k: (v["value"] if isinstance(v, dict) else v) for k, v in r.items()})
        except Exception:  # noqa: BLE001 - metering must never break a chat
            pass

    def disk_benchmark(self, path: str, mb: int = 256) -> dict:
        d = Path(path)
        if not d.is_dir():
            raise ApiError("Pick a folder on the drive you want to test")
        f = d / ".duckstoken_bench.tmp"
        chunk = os.urandom(1 << 20)
        try:
            t = time.time()
            with open(f, "wb") as fh:
                for _ in range(mb):
                    fh.write(chunk)
                fh.flush()
                os.fsync(fh.fileno())
            w = mb / (time.time() - t)
            try:
                fd = os.open(f, os.O_RDONLY)
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except OSError:
                pass
            t = time.time()
            with open(f, "rb") as fh:
                while fh.read(1 << 20):
                    pass
            r = mb / (time.time() - t)
        finally:
            f.unlink(missing_ok=True)
        return {"write_mb_s": round(w), "read_mb_s": round(r),
                "note": "Read speed may be inflated by the Windows file cache; treat it as an upper bound."}

    def shutdown(self) -> None:
        self.worker.stop()
        self.vgpu.stop()
        api_system.apply_on_exit(self)
        if self.settings.get("stop_server_on_exit"):
            self.server.stop()
            self.big.stop()
        if self.httpd:
            threading.Thread(target=self.httpd.shutdown, daemon=True).start()


APP = App()


def _browse(path: str, want_files: bool, ext: str) -> dict:
    drives = []
    if os.name == "nt":
        import string
        drives = [f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
    if not path:
        path = str(Path.home())
    p = Path(path)
    if not p.exists():
        p = Path.home()
    if p.is_file():
        p = p.parent
    dirs, files = [], []
    try:
        for e in sorted(os.scandir(p), key=lambda e: e.name.lower()):
            try:
                if e.is_dir():
                    dirs.append(e.name)
                elif want_files and (not ext or e.name.lower().endswith(ext)):
                    files.append({"name": e.name, "size": e.stat().st_size})
            except OSError:
                continue
    except OSError as e:
        raise ApiError(f"Cannot open folder: {e}") from e
    parent = str(p.parent) if p.parent != p else ""
    return {"path": str(p), "parent": parent, "dirs": dirs, "files": files, "drives": drives}


# ============================================================ HTTP handler
class Handler(VGpuApi, SystemApi, LearningApi, BaseHTTPRequestHandler):
    server_version = "Duckstoken/" + __version__
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):  # quiet
        pass

    # --- plumbing
    def _send(self, status: int, body: bytes, ctype: str = "application/json", extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                         "connect-src 'self'; frame-ancestors 'none'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200):
        self._send(status, json.dumps(obj).encode())

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 50_000_000:
            raise ApiError("Request too large", 413)
        if n == 0:
            return {}
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            raise ApiError("Content-Type must be application/json", 415)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError as e:
            raise ApiError("Invalid JSON") from e

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _dispatch(self, method: str):
        if not self._host_ok():
            return self._send(403, b"Forbidden host", "text/plain")
        url = urllib.parse.urlparse(self.path)
        path, q = url.path, {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        try:
            if method == "GET" and not path.startswith("/api/"):
                return self._static(path)
            if not path.startswith("/api/"):
                raise ApiError("Not found", 404)
            if not secrets.compare_digest(self.headers.get("X-Duck-Token") or "", APP.token):
                raise ApiError("Missing or wrong session token - reload the page.", 401)
            name = "api_" + path[5:].strip("/").replace("/", "_").replace("-", "_")
            fn = getattr(self, name, None)
            if fn is None or getattr(fn, "_method", "GET") != method:
                raise ApiError("Unknown endpoint", 404)
            body = self._body() if method == "POST" else {}
            result = fn(q, body)
            if result is not None:
                self._json(result)
        except ApiError as e:
            self._json({"error": str(e), **e.extra}, e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{e.__class__.__name__}: {e}"}, 500)

    def _static(self, path: str):
        if path in ("/", "/index.html"):
            html = (WEB / "index.html").read_text(encoding="utf-8")
            html = html.replace("__TOKEN__", APP.token).replace("__VERSION__", __version__)
            return self._send(200, html.encode(), "text/html; charset=utf-8")
        rel = path.lstrip("/")
        if rel.startswith("static/"):
            f = (WEB / rel[7:]).resolve()
            if WEB.resolve() in f.parents and f.is_file():
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                return self._send(200, f.read_bytes(), ctype)
        raise ApiError("Not found", 404)

    # ============================================================ API
    # state ------------------------------------------------------------
    def api_status(self, q, b):
        return {"version": __version__, "engine": APP.settings.get("engine"), "server": APP.server.status(),
                "jobs": APP.jobs.list(), "prewarm": APP.vgpu.prewarm_state,
                "model_path": APP.settings.get("model_path")}

    def api_hardware(self, q, b):
        hw = hardware.get_hardware(refresh=q.get("refresh") == "1")
        hw["backend_suggestion"] = tuner.resolve_backend(hw, APP.settings.get("backend_preference"))
        return hw

    def api_schema(self, q, b):
        return {"groups": st.GROUPS, "schema": st.SCHEMA, "presets": {
            k: v["description"] for k, v in st.PRESETS.items()}, "hf_quick": HF_QUICK,
            "tools": {n: {"description": t["description"]} for n, t in TOOLS.items()}}

    # settings ---------------------------------------------------------
    def api_settings(self, q, b):
        return {"values": APP.settings.all(), "profiles": APP.profiles.names()}

    def api_settings_update(self, q, b):
        wanted = (b.get("values") or {}).get("engine")
        if wanted and wanted != APP.settings.get("engine"):
            APP.server.stop()  # never leave the previous engine running unseen
        applied = APP.apply_patch(b.get("values") or {})
        if "launch_at_startup" in applied:
            from . import shell
            if not shell.set_autostart(bool(applied["launch_at_startup"])) and applied["launch_at_startup"] and os.name == "nt":
                APP.settings.update({"launch_at_startup": False})
                raise ApiError("Windows would not let Duckstoken add itself to the startup list.")
        return {"applied": applied, "values": APP.settings.all()}

    def api_settings_reset(self, q, b):
        keys = b.get("keys")
        APP.settings.reset(keys if keys else None)
        return {"values": APP.settings.all()}

    def api_presets_apply(self, q, b):
        preset = st.PRESETS.get(b.get("name", ""))
        if not preset:
            raise ApiError("Unknown preset")
        APP.apply_patch(preset["values"])
        return {"values": APP.settings.all()}

    def api_profiles_save(self, q, b):
        try:
            APP.profiles.save(b.get("name", ""), APP.settings.all())
        except ValueError as e:
            raise ApiError(str(e)) from e
        return {"profiles": APP.profiles.names()}

    def api_profiles_load(self, q, b):
        vals = APP.profiles.get(b.get("name", ""))
        if vals is None:
            raise ApiError("No such profile")
        APP.apply_patch({k: v for k, v in vals.items() if k in st.SCHEMA_BY_KEY})
        return {"values": APP.settings.all()}

    def api_profiles_delete(self, q, b):
        APP.profiles.delete(b.get("name", ""))
        return {"profiles": APP.profiles.names()}

    def api_settings_import(self, q, b):
        vals = {k: v for k, v in (b.get("values") or {}).items() if k in st.SCHEMA_BY_KEY}
        APP.apply_patch(vals)
        return {"values": APP.settings.all()}

    # models / tuning ---------------------------------------------------
    def api_models(self, q, b):
        d = APP.settings.models_dir()
        return {"dir": str(d), "models": APP.installed_models(),
                "engines": downloads.installed_engines(APP.settings.engines_dir())}

    def api_models_inspect(self, q, b):
        path = b.get("path") or APP.settings.get("model_path")
        info = APP.model_info(path)
        hw = hardware.get_hardware()
        return {"info": info, "feasibility": tuner.feasibility(hw, info, path)}

    def api_models_delete(self, q, b):
        p = Path(b.get("path", "")).resolve()
        root = APP.settings.models_dir().resolve()
        if root not in p.parents or p.suffix.lower() != ".gguf":
            raise ApiError("Only .gguf files inside the models folder can be deleted")
        for f in gguf.shard_files(p):
            f.unlink(missing_ok=True)
        return {"models": APP.installed_models()}

    def api_tune(self, q, b):
        hw = hardware.get_hardware()
        path = b.get("path") or APP.settings.get("model_path")
        info = APP.model_info(path) if path and Path(path).is_file() else None
        backend = tuner.resolve_backend(hw, b.get("backend") or APP.settings.get("backend_preference"))
        rec = tuner.recommend(hw, info, backend)
        if b.get("apply"):
            APP.apply_patch(rec["values"])
        rec["backend"] = backend
        if info:
            rec["feasibility"] = tuner.feasibility(hw, info, path)
        return rec

    # downloads ---------------------------------------------------------
    def api_hf_files(self, q, b):
        try:
            return {"groups": downloads.list_hf_models(q.get("repo", "").strip(), APP.settings.get("hf_token"))}
        except ValueError as e:
            raise ApiError(str(e)) from e
        except urllib.error.HTTPError as e:
            raise ApiError(f"Hugging Face said HTTP {e.code} (wrong repo name, or it needs a token?)") from e
        except urllib.error.URLError as e:
            raise ApiError(f"Cannot reach Hugging Face: {e.reason}") from e

    def api_hf_download(self, q, b):
        files = b.get("files") or []
        if not files:
            raise ApiError("Nothing selected")
        dest = APP.settings.models_dir()
        dest.mkdir(parents=True, exist_ok=True)
        try:
            job = downloads.start_hf_download(APP.jobs, b.get("repo", ""), files, dest, APP.settings.get("hf_token"))
        except ValueError as e:
            raise ApiError(str(e)) from e
        return {"job": job.to_dict()}

    def api_llama_releases(self, q, b):
        hw = hardware.get_hardware()
        backend = tuner.resolve_backend(hw, q.get("backend") or APP.settings.get("backend_preference"))
        try:
            data = downloads.list_llama_assets(backend)
        except urllib.error.URLError as e:
            raise ApiError(f"Cannot reach GitHub: {e}") from e
        data["backend"] = backend
        return data

    def api_llama_install(self, q, b):
        assets = b.get("assets") or []
        if not assets:
            raise ApiError("No engine build selected")

        def done(exe: str):
            APP.settings.update({"llama_server_path": exe})

        job = downloads.start_llama_install(APP.jobs, assets, b.get("tag", "custom"), b.get("backend", "cpu"),
                                            APP.settings.engines_dir(), done)
        return {"job": job.to_dict()}

    def api_jobs_cancel(self, q, b):
        return {"ok": APP.jobs.cancel(b.get("id", ""))}

    # server control ----------------------------------------------------
    def api_server_start(self, q, b):
        try:
            APP.vgpu.start_server(APP.settings.all(), force=bool(b.get("force")))
        except RuntimeError as e:
            raise ApiError(str(e)) from e
        return {"server": APP.server.status()}

    def api_server_stop(self, q, b):
        APP.vgpu.stop_server()
        return {"server": APP.server.status()}

    def api_server_restart(self, q, b):
        APP.vgpu.stop_server()
        time.sleep(0.5)
        return self.api_server_start(q, b)

    # engines -----------------------------------------------------------
    def api_engine_info(self, q, b):
        s = APP.settings.all()
        out = []
        for eid in ("llamacpp", "external", "builtin"):
            eng = APP.engines.get(eid)
            if eng is None:
                info = {"available": False, "problem": "Not included in this build (needs NumPy)."}
            else:
                info = eng.describe(s)
            out.append({"id": eid, "label": ENGINE_LABELS[eid], **info})
        return {"selected": s["engine"], "engines": out, "bundled": bundle.bundled_llama_server() is not None}

    @post
    def api_engine_test(self, q, b):
        return APP.engines["external"].probe(APP.settings.all())

    def api_engine_builtin_catalog(self, q, b):
        if "builtin" not in APP.engines:
            raise ApiError("The built-in engine is not available in this build.")
        from .engines.builtin.catalog import recommended_models
        return {"models": recommended_models()}

    def api_server_logs(self, q, b):
        which = APP.big if q.get("which") == "big" else APP.server
        return which.logs(int(q.get("since") or 0))

    # tools -------------------------------------------------------------
    def api_tools_schema(self, q, b):
        return {"definitions": APP.tools.definitions()}

    def api_tools_call(self, q, b):
        name, args = b.get("name", ""), b.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except ValueError:
                return {"ok": False, "output": "Error: tool arguments were not valid JSON"}
        try:
            needs = APP.tools.needs_approval(name)
        except Exception as e:  # noqa: BLE001 - ToolError
            return {"ok": False, "output": f"Error: {e}"}
        if needs and not b.get("approved"):
            return {"needs_approval": True, "description": APP.tools.describe(name, args)}
        return APP.tools.execute(name, args)

    def api_tools_audit(self, q, b):
        return {"entries": APP.tools.audit_tail()}

    # misc --------------------------------------------------------------
    def api_browse(self, q, b):
        return _browse(q.get("path", ""), q.get("files") == "1", q.get("ext", "").lower())

    def api_disk_benchmark(self, q, b):
        return APP.disk_benchmark(b.get("path", ""))

    def api_open_folder(self, q, b):
        p = Path(b.get("path", ""))
        if not p.is_dir():
            raise ApiError("Folder does not exist")
        if os.name == "nt":
            os.startfile(str(p))  # noqa: S606
        return {"ok": True}

    def api_app_quit(self, q, b):
        self._json({"ok": True})
        APP.shutdown()

    # chat (streaming proxy) ------------------------------------------
    def api_chat(self, q, b):
        status = APP.server.status()
        if status["state"] != "ready":
            raise ApiError("The model server is not ready. Start it on the Dashboard and wait for 'ready'.", 503)
        payload = APP.chat_payload(b)
        duck_context = payload.pop("_duck_context", {})
        req = urllib.request.Request(APP.server.base_url() + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(), headers=APP.server.upstream_headers())
        meter = RequestMeter()
        try:
            up = urllib.request.urlopen(req, timeout=APP.settings.get("server_timeout"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:2000]
            raise ApiError(f"The model engine answered HTTP {e.code}: {detail}", 502) from e
        except (urllib.error.URLError, OSError) as e:
            raise ApiError(f"Cannot reach the model engine: {e}", 502) from e
        with up:
            if not payload["stream"]:
                body = up.read()
                meter.feed_body(body)
                APP.note_request(meter)
                self._send(200, body)
                return None
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(b"data: " + json.dumps({"duck_context": duck_context}).encode() + b"\n\n")
                self.wfile.flush()
                while True:
                    line = up.readline()
                    if not line:
                        break
                    meter.feed(line)
                    self.wfile.write(line)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass  # browser pressed Stop -> closing `up` makes llama-server abort the generation
            finally:
                APP.note_request(meter)
        return None


# mark POST endpoints
for _n in ("settings_update", "settings_reset", "presets_apply", "profiles_save", "profiles_load",
           "profiles_delete", "settings_import", "models_inspect", "models_delete", "tune", "hf_download",
           "llama_install", "jobs_cancel", "server_start", "server_stop", "server_restart", "tools_call",
           "disk_benchmark", "open_folder", "app_quit", "chat"):
    getattr(Handler, "api_" + _n)._method = "POST"


# ============================================================ entry point
def _free_port(preferred: int) -> int:
    for p in range(preferred, preferred + 30):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port found")


def _make_shell(url: str, minimized: bool):
    """The native window + tray, or None when its optional pieces are unavailable."""
    try:
        from .shell import Shell
    except Exception:  # noqa: BLE001 - missing optional dependencies must never stop the app
        return None
    icon = bundle.resource_root() / "duckstoken" / "web" / "icon.png"
    get = (lambda: {**APP.settings.all(), "start_minimized": True}) if minimized else APP.settings.all
    return Shell(url, get, on_quit=APP.shutdown, icon_path=str(icon) if icon.is_file() else None)


def _redirect_stdio() -> None:
    """Windowed builds have no console: send output to a log file so problems can still be diagnosed."""
    for sink in ("stdout", "stderr"):
        if getattr(sys, sink) is None:
            try:
                setattr(sys, sink, open(paths.data_dir() / "duckstoken.log", "a", encoding="utf-8", buffering=1))
            except OSError:
                setattr(sys, sink, open(os.devnull, "w"))


def run(open_browser: bool | None = None, port: int | None = None, headless: bool = False,
        minimized: bool = False) -> None:
    _redirect_stdio()
    APP.port = _free_port(port or 8765)
    ThreadingHTTPServer.daemon_threads = True
    APP.httpd = ThreadingHTTPServer(("127.0.0.1", APP.port), Handler)
    url = f"http://127.0.0.1:{APP.port}/"
    print(f"Duckstoken {__version__} is running at {url}", flush=True)
    serving = threading.Thread(target=APP.httpd.serve_forever, daemon=True, name="http")
    serving.start()
    threading.Thread(target=hardware.get_hardware, daemon=True).start()  # warm the slow probes
    APP.worker.start()
    APP.vgpu.start()
    threading.Thread(target=api_system.background_update_check, args=(APP,), daemon=True, name="update-check").start()
    if APP.settings.get("autostart_server"):
        try:
            APP.vgpu.start_server(APP.settings.all())
        except (RuntimeError, ApiError) as e:
            print("Auto-start skipped:", e, flush=True)
    for name in ("SIGTERM", "SIGBREAK"):  # make `kill` / Ctrl+Break shut the model server down too
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: APP.shutdown())
    try:
        shell = None if headless or open_browser is False else _make_shell(url, minimized)
        if shell is not None:
            shell.run()  # blocks until the user quits from the window or the tray
        else:
            if not headless and (open_browser if open_browser is not None else APP.settings.get("open_browser")):
                threading.Timer(0.6, lambda: webbrowser.open(url)).start()
            while serving.is_alive():
                serving.join(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        APP.shutdown()
        APP.httpd.server_close()
