"""The built-in engine's OpenAI-compatible HTTP server (standard library only).

Serves GET /health, GET /v1/models and POST /v1/chat/completions on 127.0.0.1. This module checks and parses requests and writes the
JSON or server-sent-event responses; the model work is done by a backend object (see `ApiServer`) that yields the events defined here.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import os
import selectors
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NamedTuple
from urllib.parse import urlsplit

MAX_BODY = 8 << 20
DEFAULT_MAX_TOKENS = 512
MAX_STOP_STRINGS = 16
MAX_STOP_LENGTH = 256
_LOOPBACK = ("127.0.0.1", "localhost", "::1")
# OpenAI field -> (lowest, highest, whole numbers only). The ranges are wider than what the Duckstoken settings allow.
_SAMPLING = {"temperature": (0, 10, False), "top_p": (0, 1, False), "top_k": (0, 1_000_000, True), "min_p": (0, 1, False),
             "repeat_penalty": (0.01, 10, False), "presence_penalty": (-10, 10, False), "frequency_penalty": (-10, 10, False),
             "repeat_last_n": (-1, 1_000_000, True)}


class RequestError(Exception):
    """A request the engine cannot serve: `status` is the HTTP status and `kind` the OpenAI error type."""

    def __init__(self, message: str, status: int = 400, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.kind = status, kind


class Started(NamedTuple):
    """The first event of a reply: the prompt is accepted and about to be processed."""
    prompt_tokens: int
    cached_tokens: int


class Chunk(NamedTuple):
    """A piece of the reply. `kind` is the field of the OpenAI delta it belongs to: "content" or "reasoning_content"."""
    kind: str
    text: str


class Finish(NamedTuple):
    """The last event of a reply."""
    reason: str               # "stop" or "length"
    prompt_tokens: int        # the whole prompt ...
    cached_tokens: int        # ... of which this many were already in the KV cache
    predicted_tokens: int
    prompt_seconds: float
    predicted_seconds: float


@dataclass(frozen=True)
class ChatRequest:
    messages: list            # [{"role": str, "content": str}]
    stream: bool
    max_tokens: int           # -1: as many as the context window allows
    stop: tuple
    sampling: dict            # keyword arguments for Sampler
    enable_thinking: bool | None


# ------------------------------------------------------------------------------------------------ request parsing
def _number(body: dict, key: str, low: float, high: float, whole: bool = False):
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high or (whole and value != int(value)):
        raise RequestError(f"'{key}' must be {'a whole number' if whole else 'a number'} between {low} and {high}.")
    return int(value) if whole else float(value)


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    raise RequestError("A message's 'content' must be text.")


def _messages(value) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise RequestError("'messages' must be a non-empty list.")
    out = []
    for m in value:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise RequestError("Every message must be an object with a 'role'.")
        out.append({"role": m["role"], "content": _text(m.get("content"))})
    return out


def _stop(value) -> tuple[str, ...]:
    if value is None:
        return ()
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or len(items) > MAX_STOP_STRINGS or not all(isinstance(s, str) for s in items):
        raise RequestError(f"'stop' must be a string or a list of at most {MAX_STOP_STRINGS} strings.")
    if any(len(s) > MAX_STOP_LENGTH for s in items):
        raise RequestError(f"Stop strings may be at most {MAX_STOP_LENGTH} characters long.")
    return tuple(dict.fromkeys(s for s in items if s))


def parse_chat_request(body) -> ChatRequest:
    """Validate a /v1/chat/completions body. Fields the engine has no use for (tools, response_format, …) are ignored."""
    if not isinstance(body, dict):
        raise RequestError("The request body must be a JSON object.")
    max_tokens = _number(body, "max_tokens", -1, 1_000_000, whole=True)
    if max_tokens is None:
        max_tokens = _number(body, "max_completion_tokens", -1, 1_000_000, whole=True)
    if max_tokens == 0:
        raise RequestError("'max_tokens' must be at least 1 (or -1 for as many as fit).")
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    sampling = {key: v for key, (low, high, whole) in _SAMPLING.items() if (v := _number(body, key, low, high, whole)) is not None}
    seed = _number(body, "seed", -(2**63), 2**63 - 1, whole=True)
    if seed is not None:
        sampling["seed"] = seed
    kwargs = body.get("chat_template_kwargs")
    thinking = kwargs.get("enable_thinking") if isinstance(kwargs, dict) else None
    return ChatRequest(messages=_messages(body.get("messages")), stream=body.get("stream") is True, max_tokens=max_tokens,
                       stop=_stop(body.get("stop")), sampling=sampling, enable_thinking=thinking if isinstance(thinking, bool) else None)


# ------------------------------------------------------------------------------------------------ responses
def _usage(f: Finish) -> dict:
    return {"prompt_tokens": f.prompt_tokens, "completion_tokens": f.predicted_tokens, "total_tokens": f.prompt_tokens + f.predicted_tokens}


def _timings(f: Finish) -> dict:
    prompt_n = f.prompt_tokens - f.cached_tokens
    return {"prompt_n": prompt_n, "prompt_ms": round(f.prompt_seconds * 1000, 3), "cache_n": f.cached_tokens,
            "prompt_per_second": prompt_n / f.prompt_seconds if f.prompt_seconds > 0 else 0.0,
            "predicted_n": f.predicted_tokens, "predicted_ms": round(f.predicted_seconds * 1000, 3),
            "predicted_per_second": f.predicted_tokens / f.predicted_seconds if f.predicted_seconds > 0 else 0.0}


def _error_body(e: RequestError) -> dict:
    return {"error": {"message": str(e), "type": e.kind, "code": e.status}}


def _host(value: str, origin: bool = False) -> str | None:
    """The host name in a Host header (or, with `origin`, in an Origin header)."""
    try:
        return urlsplit(value if origin else "//" + value).hostname
    except ValueError:
        return None


class _Handler(BaseHTTPRequestHandler):
    server_version = "Duckstoken"
    sys_version = ""
    timeout = 30  # seconds a client may stall while we read from or write to it

    def log_message(self, format, *args):  # noqa: A002 - the engine logs its own one-line summaries instead
        pass

    # --------------------------------------------------------------------------------------------- plumbing
    def do_GET(self):
        self._serve(self._get)

    def do_POST(self):
        self._serve(self._post)

    def _serve(self, route) -> None:
        self._sent = False
        try:
            self._check_access()
            route()
        except RequestError as e:
            self._reply_error(e)
        except OSError:
            pass  # the client went away
        except Exception as e:  # noqa: BLE001 - a bug must not kill the server thread silently
            self.server.backend.log(f"[duckstoken] request failed: {e!r}")
            self._reply_error(RequestError(f"The built-in engine failed: {e}", 500, "server_error"))

    def _reply_error(self, e: RequestError) -> None:
        if not self._sent:
            with contextlib.suppress(OSError):  # the client may be gone already
                self._json(e.status, _error_body(e))

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self._sent = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _check_access(self) -> None:
        """Refuse web pages and rebound DNS names (they can reach 127.0.0.1 too) and, when a key is set, clients without it."""
        host, origin = self.headers.get("Host"), self.headers.get("Origin")
        if (host is not None and _host(host) not in _LOOPBACK) or (origin is not None and _host(origin, origin=True) not in _LOOPBACK):
            raise RequestError("Requests from other web sites are not allowed.", 403, "forbidden")
        key = self.server.backend.api_key
        if key and not (self.command == "GET" and self._path == "/health"):
            header = self.headers.get("Authorization", "")
            given = header[7:].strip() if header[:7].lower() == "bearer " else ""
            if not hmac.compare_digest(given.encode(), key.encode()):
                raise RequestError("Invalid or missing API key.", 401, "authentication_error")

    @property
    def _path(self) -> str:
        return self.path.split("?", 1)[0]

    def _client_gone(self) -> bool:
        """True once the client has closed its end of the connection (it sends nothing after its request, so any readable byte is EOF)."""
        try:
            with selectors.DefaultSelector() as readable:
                readable.register(self.connection, selectors.EVENT_READ)
                if readable.select(0):
                    return not self.connection.recv(1, socket.MSG_PEEK)
        except (OSError, ValueError):
            return True
        return False

    def _read_json(self):
        length = self.headers.get("Content-Length")
        if length is None:
            raise RequestError("The request needs a Content-Length header.", 411)
        if not (length.isascii() and length.isdigit()):
            raise RequestError("The Content-Length header is invalid.")
        if len(length) > 12 or int(length) > MAX_BODY:
            raise RequestError(f"The request is larger than {MAX_BODY >> 20} MB.", 413)
        raw = self.rfile.read(int(length))
        if len(raw) < int(length):
            raise RequestError("The request body ended early.")
        try:
            return json.loads(raw)
        except (ValueError, RecursionError):
            raise RequestError("The request body is not valid JSON.") from None

    # --------------------------------------------------------------------------------------------- routes
    def _get(self) -> None:
        backend = self.server.backend
        if self._path == "/health":
            self._json(200 if backend.ready else 503, {"status": "ok" if backend.ready else "loading"})
        elif self._path == "/v1/models":
            self._require_ready()
            self._json(200, {"object": "list", "data": [{"id": backend.model_id, "object": "model", "created": int(time.time()),
                                                         "owned_by": "duckstoken"}]})
        else:
            raise RequestError("Not found.", 404, "not_found_error")

    def _post(self) -> None:
        if self._path != "/v1/chat/completions":
            raise RequestError("Not found.", 404, "not_found_error")
        self._require_ready()
        backend = self.server.backend
        request = parse_chat_request(self._read_json())
        events = backend.generate(request, self._client_gone)
        try:
            if next(events, None) is None:  # the first event is Started; nothing at all means the client left while waiting its turn
                return
            (self._stream if request.stream else self._complete)(events, backend.model_id)
        finally:
            events.close()  # releases the model if the client left halfway

    def _require_ready(self) -> None:
        if not self.server.backend.ready:
            raise RequestError("The model is still loading.", 503, "unavailable_error")

    def _complete(self, events, model: str) -> None:
        content, reasoning, finish = [], [], None
        for event in events:
            if isinstance(event, Chunk):
                (reasoning if event.kind == "reasoning_content" else content).append(event.text)
            elif isinstance(event, Finish):
                finish = event
        if finish is None:
            return
        message = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        self._json(200, {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
                         "model": model, "choices": [{"index": 0, "message": message, "finish_reason": finish.reason}],
                         "usage": _usage(finish), "timings": _timings(finish)})

    def _event(self, payload) -> None:
        self.wfile.write(b"data: " + (payload if isinstance(payload, bytes) else json.dumps(payload).encode()) + b"\n\n")
        self.wfile.flush()

    def _stream(self, events, model: str) -> None:
        ident, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        def frame(delta: dict, finish=None, **extra) -> dict:
            return {"id": ident, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}

        self._sent = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self._event(frame({"role": "assistant", "content": ""}))
        while True:
            try:
                event = next(events)
            except StopIteration:
                break
            except RequestError as e:
                self._event(_error_body(e))
                break
            except Exception as e:  # noqa: BLE001 - tell the client instead of just dropping the stream
                self.server.backend.log(f"[duckstoken] reply failed: {e!r}")
                self._event(_error_body(RequestError(f"The built-in engine failed: {e}", 500, "server_error")))
                break
            if isinstance(event, Chunk):
                self._event(frame({event.kind: event.text}))
            elif isinstance(event, Finish):
                self._event(frame({}, event.reason, usage=_usage(event), timings=_timings(event)))
        self._event(b"[DONE]")


class ApiServer(ThreadingHTTPServer):
    """Serves one backend on 127.0.0.1. The backend provides `ready` (bool), `model_id` (str), `api_key` (str, "" = none),
    `log(line)` and `generate(request, client_gone)`, a generator of Started, Chunk… and Finish events that may raise RequestError.
    Raises OSError when the port cannot be opened."""

    daemon_threads = True
    block_on_close = False
    allow_reuse_address = os.name != "nt"  # on Windows SO_REUSEADDR would let us share a port that is already in use

    def __init__(self, port: int, backend):
        self.backend = backend
        super().__init__(("127.0.0.1", port), _Handler)

    @property
    def port(self) -> int:
        return self.server_address[1]

    def handle_error(self, request, client_address) -> None:
        error = sys.exc_info()[1]
        if not isinstance(error, OSError):  # a client that disconnects mid-reply is routine
            self.backend.log(f"[duckstoken] server error: {error!r}")
