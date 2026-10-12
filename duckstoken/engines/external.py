"""Use a server that is already running (Ollama, LM Studio, vLLM, …) through its OpenAI-compatible API."""

from __future__ import annotations

import collections
import json
import threading
import time
import urllib.error
import urllib.request

PRESETS = {"ollama": "http://127.0.0.1:11434", "lmstudio": "http://127.0.0.1:1234"}
_HOW = {"ollama": "Is Ollama running? Install it from ollama.com and start it.",
        "lmstudio": "Is LM Studio's local server running? (Developer tab → Start Server)",
        "custom": "Check the URL and that the server is running."}
_SKIP = ("embed", "rerank")


class ExternalEngine:
    engine_id = "external"
    label = "External server"
    swappable = False  # we cannot free another program's memory

    def __init__(self):
        self.state, self.error, self.url, self.key, self.model = "stopped", "", "", "", ""
        self.models: list[str] = []
        self._log: collections.deque[str] = collections.deque(maxlen=500)
        self._base = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.started_at = 0.0

    # ----------------------------------------------------------------- config
    @staticmethod
    def resolve_url(s: dict) -> str:
        preset = s.get("external_preset", "ollama")
        url = PRESETS.get(preset) or s.get("external_url", "")
        url = url.strip().rstrip("/")
        return url[:-3] if url.endswith("/v1") else url

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def _get_models(self, timeout: float = 5) -> list[str]:
        req = urllib.request.Request(self.url + "/v1/models", headers=self._headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        ids = [str(m.get("id")) for m in data.get("data", []) if m.get("id")]
        return [i for i in ids if not any(w in i.lower() for w in _SKIP)] or ids

    def _note(self, line: str) -> None:
        with self._lock:
            if len(self._log) == self._log.maxlen:
                self._base += 1
            self._log.append(line)

    # -------------------------------------------------------------- lifecycle
    def probe(self, s: dict) -> dict:
        """Check connectivity without changing state: {'ok', 'models', 'error'}."""
        self.url, self.key = self.resolve_url(s), s.get("external_api_key", "")
        if not self.url.lower().startswith(("http://", "https://")):
            return {"ok": False, "models": [], "error": "The URL must start with http:// or https://"}
        try:
            return {"ok": True, "models": self._get_models(), "error": ""}
        except urllib.error.HTTPError as e:
            return {"ok": False, "models": [], "error": f"{self.url} answered HTTP {e.code}. "
                    + ("The API key looks wrong." if e.code in (401, 403) else "Is this an OpenAI-compatible server?")}
        except (urllib.error.URLError, OSError, ValueError) as e:
            return {"ok": False, "models": [], "error": f"Cannot reach {self.url} ({getattr(e, 'reason', e)}). "
                    + _HOW.get(s.get("external_preset", "custom"), "")}

    def start(self, s: dict) -> None:
        self.stop()
        self.state, self.error, self.started_at = "starting", "", time.time()
        r = self.probe(s)
        if not r["ok"]:
            self.state, self.error = "error", r["error"]
            self._note("[duckstoken] " + r["error"])
            raise RuntimeError(r["error"])
        self.models = r["models"]
        wanted = (s.get("external_model") or "").strip()
        if wanted:
            self.model = wanted
        elif self.models:
            self.model = self.models[0]
        else:
            self.state, self.error = "error", "The server has no models. Pull or load one first."
            raise RuntimeError(self.error)
        self.state = "ready"
        self._note(f"[duckstoken] connected to {self.url} · model {self.model} · {len(self.models)} model(s) available")
        self._stop.clear()
        threading.Thread(target=self._watch, args=(s,), daemon=True, name="external-watch").start()

    def _watch(self, s: dict) -> None:
        while not self._stop.wait(10):
            ok = self.probe(s)["ok"]
            if self.state == "ready" and not ok:
                self.state, self.error = "error", f"Lost connection to {self.url}"
                self._note("[duckstoken] " + self.error)
            elif self.state == "error" and ok:
                self.state, self.error = "ready", ""

    def stop(self) -> None:
        self._stop.set()
        if self.state != "stopped":
            self._note("[duckstoken] disconnected (the external server keeps running)")
        self.state, self.error = "stopped", ""

    # ------------------------------------------------------------------- info
    def status(self) -> dict:
        running = self.state in ("ready", "starting")
        return {"state": self.state, "error": self.error, "running": running,
                "uptime": int(time.time() - self.started_at) if running else 0, "url": self.url,
                "cmdline": f"{self.url}  (model: {self.model or '—'})"}

    def base_url(self) -> str:
        return self.url

    def upstream_headers(self) -> dict:
        return self._headers()

    def model_name(self) -> str:
        return self.model or "local"

    def logs(self, since: int = 0) -> dict:
        with self._lock:
            return {"lines": list(self._log)[max(since - self._base, 0):], "next": self._base + len(self._log)}

    def describe(self, s: dict) -> dict:
        return {"available": True, "url": self.resolve_url(s), "models": self.models}
