"""Inference engines Duckstoken can talk to. Every engine offers the same small interface:

    start(settings) / stop() / status() / base_url() / upstream_headers() / model_name() / logs(since) / describe(settings)

and speaks the OpenAI chat-completions protocol at base_url(), so the chat proxy and the learning features never
care which one is active. `EngineSwitch` forwards calls to whichever engine the `engine` setting selects.
"""

from __future__ import annotations

from .external import ExternalEngine

ENGINE_LABELS = {"llamacpp": "llama.cpp (bundled)", "external": "External server (Ollama, LM Studio…)",
                 "builtin": "Built-in (pure Python, small models)"}


class EngineSwitch:
    """Stands in for a single engine object; always delegates to the one chosen in the settings."""

    def __init__(self, engines: dict, get_settings):
        self.engines, self._get = engines, get_settings

    @property
    def current(self):
        return self.engines.get(self._get().get("engine", "llamacpp")) or self.engines["llamacpp"]

    def start(self, s: dict) -> None:
        for e in self.engines.values():  # one engine at a time: free the RAM of the others first
            if e is not self.current and e.status()["running"]:
                e.stop()
        self.current.start(s)

    def stop(self) -> None:
        for e in self.engines.values():
            e.stop()

    def __getattr__(self, name):  # status, base_url, upstream_headers, model_name, logs, swappable, …
        if name in ("engines", "_get"):
            raise AttributeError(name)
        return getattr(self.current, name)


def load_engines() -> dict:
    """llama.cpp and external always exist; the built-in engine only when its module and NumPy are present."""
    from ..server_proc import ServerManager

    engines = {"llamacpp": ServerManager(), "external": ExternalEngine()}
    try:
        from .builtin import BuiltinEngine
        engines["builtin"] = BuiltinEngine()
    except Exception:  # noqa: BLE001 - missing NumPy or module: the engine simply isn't offered
        pass
    return engines
