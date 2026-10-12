"""The built-in engine: loads a GGUF model in the background, serves it over HTTP and generates replies one at a time.

It offers the same interface as the other engines (start, stop, status, base_url, …) and speaks the OpenAI chat-completions
protocol, so the rest of Duckstoken cannot tell it from llama.cpp.
"""

from __future__ import annotations

import bisect
import collections
import contextlib
import re
import threading
import time
from pathlib import Path

from .gguf_weights import SUPPORTED_TYPES, UnsupportedModel
from .model import BATCH, MAX_PARAMS, SUPPORTED_ARCHITECTURES, LoadedModel, inspect_model
from .sampler import Sampler
from .server import ApiServer, ChatRequest, Chunk, Finish, RequestError, Started
from .tokenizer import CONTROL

MIN_CTX = 256
MAX_CTX = 4096            # the KV cache and the prompt processing time grow with the context; a bigger window is not practical
CHARS_PER_TOKEN_LIMIT = 32  # a prompt with more characters per token than this is refused without being tokenized
_ACTIVE = ("starting", "loading", "ready")
_SYSTEM_ROLES = ("system", "developer")


# ------------------------------------------------------------------------------------------------ text processing
class ThinkSplitter:
    """Separates the reasoning a model writes between <think> and </think> from its answer, in a stream of text pieces.

    `thinking` says that the prompt already opened the reasoning (its template ended with "<think>"). Otherwise the reasoning
    starts only if the reply begins with the tag. Whitespace around the tags is dropped.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self, thinking: bool = False):
        self._reasoning = thinking
        self._undecided = not thinking  # nothing but whitespace seen yet, so the reply may still open with <think>
        self._trim = thinking           # drop whitespace at the start of the next piece
        self._held = ""

    def feed(self, text: str) -> list[Chunk]:
        self._held += text
        out: list[Chunk] = []
        while self._held:
            if self._reasoning:
                end = self._held.find(self.CLOSE)
                if end >= 0:
                    self._emit(out, "reasoning_content", self._held[:end])
                    self._held = self._held[end + len(self.CLOSE):]
                    self._reasoning, self._trim = False, True
                    continue
                keep = _partial_suffix(self._held, (self.CLOSE,))
                self._emit(out, "reasoning_content", self._held[:len(self._held) - keep])
                self._held = self._held[len(self._held) - keep:]
                break
            if self._undecided:
                word = self._held.lstrip()
                if word.startswith(self.OPEN):
                    self._held, self._reasoning, self._undecided, self._trim = word[len(self.OPEN):], True, False, True
                    continue
                if not word or self.OPEN.startswith(word):
                    break  # might still turn out to be the opening tag
                self._undecided = False
            self._emit(out, "content", self._held)
            self._held = ""
        return out

    def finish(self) -> list[Chunk]:
        """Whatever was held back because it might have been part of a tag."""
        out: list[Chunk] = []
        self._emit(out, "reasoning_content" if self._reasoning else "content", self._held)
        self._held = ""
        return out

    def _emit(self, out: list[Chunk], kind: str, text: str) -> None:
        if self._trim:
            text = text.lstrip()
            self._trim = not text
        if text:
            out.append(Chunk(kind, text))


class StopMatcher:
    """Cuts a stream of text at the first stop string, holding back text that might still become one."""

    def __init__(self, stops):
        self._stops = tuple(stops)
        self._held = ""
        self.hit = False

    def feed(self, text: str) -> str:
        """The text that is safe to show; sets `hit` when a stop string completed (the text before it is returned)."""
        if self.hit:
            return ""
        if not self._stops:
            return text
        buf = self._held + text
        cuts = [i for i in (buf.find(s) for s in self._stops) if i >= 0]
        if cuts:
            self.hit, self._held = True, ""
            return buf[:min(cuts)]
        keep = _partial_suffix(buf, self._stops)
        self._held = buf[len(buf) - keep:]
        return buf[:len(buf) - keep]

    def flush(self) -> str:
        held, self._held = ("" if self.hit else self._held), ""
        return held


def _partial_suffix(text: str, words) -> int:
    """Length of the longest ending of `text` that is a proper beginning of one of `words`."""
    for n in range(min(max(map(len, words)) - 1, len(text)), 0, -1):
        tail = text[-n:]
        if any(w.startswith(tail) and len(w) > n for w in words):
            return n
    return 0


class _Reply:
    """Turns sampled token ids into the chunks the client sees: decoded text, reasoning split from the answer, stop strings applied."""

    def __init__(self, decoder, splitter: ThinkSplitter, stops: StopMatcher):
        self._decoder, self._splitter, self._stops = decoder, splitter, stops

    @property
    def stopped(self) -> bool:
        return self._stops.hit

    def push(self, token_id: int) -> list[Chunk]:
        return self._route(self._splitter.feed(self._decoder.push(token_id)))

    def finish(self) -> list[Chunk]:
        out = self._route(self._splitter.feed(self._decoder.flush()) + self._splitter.finish())
        tail = self._stops.flush()
        return out + [Chunk("content", tail)] if tail else out

    def _route(self, pieces: list[Chunk]) -> list[Chunk]:
        out = []
        for kind, text in pieces:
            if kind == "content":
                text = self._stops.feed(text)
            if text:
                out.append(Chunk(kind, text))
        return out


# ------------------------------------------------------------------------------------------------ generation
class _ClientGone(Exception):
    """The client disconnected while its request was queued or being processed."""


class Generator:
    """Answers chat requests with one loaded model, one at a time. The KV cache of the previous prompt is kept, so a follow-up turn
    only processes the tokens that are new."""

    def __init__(self, model: LoadedModel, closing: threading.Event, log):
        self.model, self.tokenizer = model, model.tokenizer
        self._closing, self._log = closing, log
        self._lock = threading.Lock()
        self._session = model.new_session()
        self._cached: list[int] = []  # the tokens the session's KV cache holds
        tok = self.tokenizer
        ids = tok.eos_ids | ({tok.bos_id} if tok.bos_id is not None else set())
        markers = {t for t, kind in zip(tok.tokens, tok.types, strict=True) if kind == CONTROL} | {tok.tokens[i] for i in ids}
        markers.discard("")
        self._markers = re.compile("|".join(map(re.escape, sorted(markers, key=len, reverse=True)))) if markers else None

    def close(self) -> None:
        """Release the model once the reply in progress (if any) has noticed that the engine is stopping."""
        with self._lock:
            self._session, self._cached = None, []
            self.model.close()

    def generate(self, request: ChatRequest, client_gone):
        """Yield Started, then Chunk events, then Finish. Raises RequestError for a request that cannot be served; ends quietly when
        `client_gone()` turns true. Close the generator when abandoning it: it holds the model until then."""
        try:
            with self._turn(client_gone):
                yield from self._reply(request, client_gone)
        except _ClientGone:
            return

    @contextlib.contextmanager
    def _turn(self, client_gone):
        while not self._lock.acquire(timeout=0.25):
            self._check(client_gone)
        try:
            self._check(client_gone)
            yield
        finally:
            self._lock.release()

    def _check(self, client_gone) -> None:
        if self._closing.is_set():
            raise RequestError("The built-in engine is shutting down.", 503, "unavailable_error")
        if client_gone():
            raise _ClientGone

    def _reply(self, request: ChatRequest, client_gone):
        try:
            sampler = Sampler(**request.sampling)
        except ValueError as e:
            raise RequestError(str(e)) from None
        limit = self.model.max_ctx
        ids, thinking = self._prompt(request)
        max_new = limit - len(ids) if request.max_tokens < 0 else min(request.max_tokens, limit - len(ids))
        reused = self._reuse(ids)
        yield Started(len(ids), reused)

        began = time.perf_counter()
        logits = None
        for start in range(reused, len(ids), BATCH):
            self._check(client_gone)
            logits = self._feed(ids[start:start + BATCH])
        prompt_seconds = time.perf_counter() - began

        reply = _Reply(self.tokenizer.stream_decoder(), ThinkSplitter(thinking), StopMatcher(request.stop))
        history, produced, reason = list(ids), 0, "length"
        began = time.perf_counter()
        while True:
            token = sampler.sample(logits, history)
            if token in self.tokenizer.eos_ids:
                reason = "stop"
                break
            history.append(token)
            produced += 1
            yield from reply.push(token)
            if reply.stopped:
                reason = "stop"
                break
            if produced >= max_new:
                break
            self._check(client_gone)
            logits = self._feed([token])
        yield from reply.finish()
        predicted_seconds = time.perf_counter() - began
        rate = produced / predicted_seconds if predicted_seconds > 0 else 0.0
        self._log(f"[duckstoken] reply: {len(ids)} prompt tokens ({reused} reused), {produced} generated at {rate:.1f} tok/s ({reason})")
        yield Finish(reason, len(ids), reused, produced, prompt_seconds, predicted_seconds)

    # ------------------------------------------------------------------------------------------------ KV cache
    def _reuse(self, ids: list[int]) -> int:
        """Keep the part of the KV cache that matches the start of `ids`; returns how many tokens that is."""
        common = 0
        for a, b in zip(self._cached, ids, strict=False):
            if a != b:
                break
            common += 1
        common = min(common, len(ids) - 1)  # the last prompt token is always processed: it provides the logits to sample from
        self._session.truncate(common)
        del self._cached[common:]
        return common

    def _feed(self, tokens: list[int]):
        try:
            logits = self._session.feed(tokens)
        except BaseException:
            self._session.reset()  # the cache may be half updated: start over rather than answer from a corrupt one
            self._cached.clear()
            raise
        self._cached.extend(tokens)
        return logits

    # ------------------------------------------------------------------------------------------------ prompt
    def _defang(self, text: str) -> str:
        """Break up the special tokens a message spells out ("<|im_end|>"), so that text from a web page or a user cannot forge
        a turn or end the reply. The chat template's own markers are added afterwards and stay intact."""
        return self._markers.sub(lambda m: m.group()[0] + " " + m.group()[1:], text) if self._markers else text

    def _encode(self, messages: list[dict], enable_thinking) -> tuple[list[int], bool]:
        tok = self.tokenizer
        text = tok.apply_chat_template(messages, enable_thinking=enable_thinking)
        return tok.encode(text, add_bos=tok.add_bos, parse_special=True), text.rstrip().endswith("<think>")

    def _prompt(self, request: ChatRequest) -> tuple[list[int], bool]:
        """The prompt tokens (and whether the template left the model inside <think>). When the conversation is too long the oldest
        messages (never the system ones or the last) are left out; if it still does not fit the request is refused."""
        limit = self.model.max_ctx
        wanted = limit if request.max_tokens < 0 else request.max_tokens
        budget = limit - max(1, min(wanted, limit // 4))  # prompt tokens allowed: part of the window must stay free for the answer
        messages = [{"role": m["role"], "content": self._defang(m["content"])} for m in request.messages]
        starts = _starting_points(messages, limit * CHARS_PER_TOKEN_LIMIT)
        encoded: dict[int, tuple[list[int], bool]] = {}

        def fits(start: int) -> bool:
            if start not in encoded:
                kept = [m for i, m in enumerate(messages) if i >= start or m["role"] in _SYSTEM_ROLES]
                encoded[start] = self._encode(kept, request.enable_thinking)
            return len(encoded[start][0]) <= budget

        if not starts or not fits(starts[-1]):
            raise RequestError(f"The conversation does not fit in the model's {limit}-token context window. Shorten the message or "
                               f"raise the context size in the settings (the built-in engine allows up to {MAX_CTX}).",
                               400, "context_length_exceeded")
        # fewer messages never need more tokens, so the first starting point that fits can be found by bisection
        start = starts[0] if fits(starts[0]) else starts[bisect.bisect_left(starts, True, lo=1, key=fits)]
        if start > 0:
            left_out = sum(1 for m in messages[:start] if m["role"] not in _SYSTEM_ROLES)
            self._log(f"[duckstoken] the conversation is longer than the {limit}-token context: left out the oldest {left_out} message(s)")
        return encoded[start]


def _starting_points(messages: list[dict], max_chars: int) -> list[int]:
    """Indices of the messages the conversation sent to the model may start with, oldest first: the first message, each user message and
    the last message, as long as they and everything after them (plus the system messages) stay within `max_chars` characters."""
    cost = [len(m["content"]) + 16 for m in messages]
    chars = sum(c for c, m in zip(cost, messages, strict=True) if m["role"] in _SYSTEM_ROLES)
    points = []
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] not in _SYSTEM_ROLES:
            chars += cost[i]
        if i in (0, len(messages) - 1) or messages[i]["role"] == "user":
            if chars > max_chars:
                break
            points.append(i)
    return points[::-1]


# ------------------------------------------------------------------------------------------------ engine
class _Cancelled(Exception):
    """Raised inside the model loader to abandon a load that was cancelled."""


class _Run:
    """One start of the engine: its HTTP server, the reply generator once the model is loaded, and the threads. It is the server's backend."""

    def __init__(self, port: int, api_key: str, log):
        self.api_key, self.log = api_key, log
        self.cancel = threading.Event()
        self.generator: Generator | None = None
        self.model_id = ""
        try:
            self.server = ApiServer(port, self)
        except OSError:
            self.server = ApiServer(0, self)  # the wanted port is busy (or not allowed): any free one will do
        self.serving = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True, name="builtin-http")
        self.serving.start()
        self.loading: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        return self.generator is not None and not self.cancel.is_set()

    def generate(self, request: ChatRequest, client_gone):
        generator = self.generator
        if generator is None or self.cancel.is_set():
            raise RequestError("The model is not loaded.", 503, "unavailable_error")
        return generator.generate(request, client_gone)

    def shutdown(self) -> None:
        """Stop serving, abort whatever is running and release the model. Safe to call again and from the loading thread."""
        self.cancel.set()
        self.server.shutdown()
        self.server.server_close()
        self.serving.join(timeout=10)
        if self.loading is not None and self.loading is not threading.current_thread():
            self.loading.join(timeout=60)
        generator, self.generator = self.generator, None
        if generator is not None:
            generator.close()


def _numpy_version() -> str | None:
    try:
        import numpy
    except ImportError:
        return None
    return numpy.__version__


class BuiltinEngine:
    engine_id = "builtin"
    label = "Built-in engine"
    swappable = True  # stopping it really frees the model's RAM

    def __init__(self):
        self.state, self.error = "stopped", ""
        self.started_at = 0.0
        self.port = 0
        self.model_id, self.cmdline = "", ""
        self.api_key = ""
        self.progress = 0.0
        self._log: collections.deque[str] = collections.deque(maxlen=500)
        self._log_base = 0
        self._lock = threading.RLock()
        self._run: _Run | None = None
        self._inspected: tuple | None = None

    # ------------------------------------------------------------------------------------------------ logging
    def _note(self, line: str) -> None:
        with self._lock:
            if len(self._log) == self._log.maxlen:
                self._log_base += 1
            self._log.append(line)

    def logs(self, since: int = 0) -> dict:
        with self._lock:
            return {"lines": list(self._log)[max(since - self._log_base, 0):], "next": self._log_base + len(self._log)}

    # ------------------------------------------------------------------------------------------------ lifecycle
    def _refuse(self, message: str) -> None:
        self.state, self.error = "error", message
        self._note("[duckstoken] " + message)
        raise RuntimeError(message)

    def start(self, s: dict) -> None:
        """Open the server at once and load the model in the background: poll `status()` for starting -> loading -> ready (or error)."""
        with self._lock:
            if self.state in _ACTIVE:
                raise RuntimeError("The model server is already running")
            self.error = ""
            path = str(s.get("model_path") or "")
            if not path:
                self._refuse("No model selected. Pick a .gguf file in the Models tab.")
            if not Path(path).is_file():
                self._refuse(f"The model file was not found: {path}")
            wanted, key = int(s.get("port") or 0), s.get("api_key") or ""
            try:
                run = _Run(wanted, key, self._note)
            except OSError as e:
                self._refuse(f"The built-in engine could not open a local port: {e}")
            ctx = max(MIN_CTX, min(int(s.get("ctx_size") or MAX_CTX), MAX_CTX))
            self._run, self.port, self.api_key = run, run.server.port, key
            self.state, self.started_at, self.progress, self.model_id = "starting", time.time(), 0.0, ""
            self.cmdline = f"built-in engine · {Path(path).name} · context up to {ctx} tokens"
            if wanted and run.server.port != wanted:
                self._note(f"[duckstoken] port {wanted} is in use, serving on port {run.server.port} instead")
            self._note(f"[duckstoken] built-in engine listening on {self.base_url()}")
            run.loading = threading.Thread(target=self._load, args=(run, path, ctx, int(s.get("builtin_threads") or 0)),
                                           daemon=True, name="builtin-load")
            run.loading.start()

    def _load(self, run: _Run, path: str, ctx: int, threads: int) -> None:
        shown = [-1]

        def progress(fraction: float, message: str) -> None:
            if fraction < 1.0 and run.cancel.is_set():  # the final call comes after the model is complete: nothing left to abort
                raise _Cancelled
            with self._lock:
                if self._run is run:
                    self.progress = fraction
            if int(fraction * 4) > shown[0]:
                shown[0] = int(fraction * 4)
                self._note(f"[duckstoken] loading the model: {fraction:.0%}")

        try:
            with self._lock:
                if self._run is not run:
                    return
                self.state = "loading"
            self._note(f"[duckstoken] loading {Path(path).name} …")
            model = LoadedModel.load(path, max_ctx=ctx, n_threads=threads, progress=progress)
            try:
                generator = Generator(model, run.cancel, self._note)
            except BaseException:
                model.close()
                raise
        except _Cancelled:
            return
        except UnsupportedModel as e:
            self._fail(run, str(e))
            return
        except MemoryError:
            self._fail(run, "The computer ran out of memory while loading this model. Close other programs or pick a smaller model.")
            return
        except Exception as e:  # noqa: BLE001 - whatever went wrong, the user gets a message instead of a silent dead thread
            self._fail(run, f"The model could not be loaded: {e}")
            return
        with self._lock:
            current = self._run is run and not run.cancel.is_set()
            if current:
                run.generator, run.model_id = generator, model.name
                self.state, self.model_id, self.progress = "ready", model.name, 1.0
                self.cmdline = (f"built-in engine · {model.name} · {model.arch} {model.n_params / 1e9:.2f}B parameters · "
                                f"{model.weight_dtype} weights · context {model.max_ctx} tokens")
        if current:
            self._note(f"[duckstoken] ready: {self.cmdline}")
        else:
            generator.close()

    def _fail(self, run: _Run, message: str) -> None:
        run.shutdown()  # free the port and whatever the failed load left behind, so that "error" means the engine really is down
        with self._lock:
            if self._run is not run:
                return
            self._run = None
            self.state, self.error = "error", message
        self._note("[duckstoken] " + message)

    def stop(self) -> None:
        with self._lock:
            run, self._run = self._run, None
            was_active = self.state in _ACTIVE
            self.state, self.error, self.progress = "stopped", "", 0.0
        if run is not None:
            run.shutdown()
        if was_active:
            self._note("[duckstoken] built-in engine stopped")

    # ------------------------------------------------------------------------------------------------ info
    def status(self) -> dict:
        with self._lock:
            running = self.state in _ACTIVE
            return {"state": self.state, "error": self.error, "running": running,
                    "uptime": int(time.time() - self.started_at) if running else 0, "url": self.base_url(),
                    "cmdline": self.cmdline, "progress": round(self.progress, 3)}

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}" if self.port else ""

    def upstream_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def model_name(self) -> str:
        return self.model_id or "local"

    def describe(self, s: dict) -> dict:
        """What the settings page shows: whether the engine can run, what it supports and whether the chosen model file is usable."""
        version = _numpy_version()
        info = {"available": version is not None, "numpy": version, "supported": list(SUPPORTED_ARCHITECTURES),
                "formats": list(SUPPORTED_TYPES), "max_params_b": MAX_PARAMS / 1e9,
                "problem": "" if version else "NumPy is not installed, so the built-in engine cannot run."}
        path = s.get("model_path")
        if version and path:
            info.update(self._inspect(str(path)))
        return info

    def _inspect(self, path: str) -> dict:
        """Check a model file without loading its weights; the answer is remembered until the file changes."""
        try:
            stat = Path(path).stat()
        except OSError as e:
            return {"model_problem": f"The model file cannot be read: {e.strerror or e}"}
        key = (path, stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if self._inspected and self._inspected[0] == key:
                return self._inspected[1]
        try:
            found = inspect_model(path)
            params = found.pop("n_params")
            result = {"model": {**found, "params_b": round(params / 1e9, 2)}}
        except UnsupportedModel as e:
            result = {"model_problem": str(e)}
        except (OSError, ValueError) as e:
            result = {"model_problem": f"The model file cannot be read: {e}"}
        with self._lock:
            self._inspected = (key, result)
        return result
