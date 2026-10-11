"""Tiny client for calling a running llama-server (non-streaming) from backend code."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class LLMError(Exception):
    pass


def complete(server, messages: list[dict], *, max_tokens: int = 512, temperature: float = 0.3,
             effort: str = "low", timeout: float = 900, extra: dict | None = None) -> dict:
    """Run one chat completion on `server` (a ServerManager).

    Returns {"content", "reasoning", "tokens", "tps", "model"}; raises LLMError on any failure.
    """
    status = server.status()
    if status["state"] != "ready":
        raise LLMError(f"model server is {status['state']}, not ready")
    payload = {"model": "local", "messages": messages, "stream": False, "temperature": temperature,
               "max_tokens": max_tokens, "chat_template_kwargs": {"reasoning_effort": effort}}
    payload.update(extra or {})
    req = urllib.request.Request(server.base_url() + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers=server.upstream_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        raise LLMError(f"llama-server error {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise LLMError(f"cannot reach llama-server: {e}") from e
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError("unexpected response from llama-server") from e
    t = data.get("timings") or {}
    return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
            "tokens": int(t.get("predicted_n") or (data.get("usage") or {}).get("completion_tokens") or 0),
            "tps": float(t.get("predicted_per_second") or 0), "model": str(data.get("model") or "")}
