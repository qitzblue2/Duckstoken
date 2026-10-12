#!/usr/bin/env python3
"""Stand-in for llama-server with llama.cpp-shaped output, used by the VirtualGPU benchmark/optimizer tests.

It speaks the parts of the real server the benchmark uses: /health (503 while "loading"), /props, /tokenize,
POST /completion and POST /v1/chat/completions with streaming and a `timings` object in the final chunk, and it prints
load-log lines in the real format (``load_tensors: CPU_Mapped model buffer size = ...``).  Behaviour is steered by
``--fake-*`` flags (passed through the settings' ``extra_args``), everything else on the command line is ignored the way
a real server would accept it.

  --fake-load-s S        seconds until /health turns 200 (default 0.2)
  --fake-hang-load       never becomes healthy
  --fake-exit-code N     print an error and exit with N while "loading" (a load failure)
  --fake-pp-tps X        simulated prompt processing speed (default 20000 tokens/s)
  --fake-tg-tps X        simulated generation speed (default 2000 tokens/s)
  --fake-cold-factor F   the first request is F times slower (default 1)
  --fake-gpu-mib N       report N MiB of model buffer on CUDA0 when -ngl > 0 (and the rest on CPU)
  --fake-chat-only       /completion answers 404 (an older server)
  --fake-no-timings      leave `timings` out of the final chunk
  --fake-status N        answer completions with HTTP N and an error body
  --fake-crash-after N   die abruptly while serving request number N+1
  --fake-ignore-term     ignore SIGTERM (tests the kill escalation)
  --fake-early-eos N     stop after N generated tokens even if more were asked for (ignore_eos is not honoured)
  --fake-cache-prompt    honour `cache_prompt` like llama-server does (a repeated prompt is not re-processed)
"""
import hashlib
import json
import os
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = sys.argv[1:]


def opt(name, default=None, cast=str):
    if name in ARGS:
        i = ARGS.index(name)
        if i + 1 < len(ARGS):
            return cast(ARGS[i + 1])
    return default


def flag(name):
    return name in ARGS


PORT = opt("--port", 8080, int)
MODEL = opt("-m", opt("--model", ""))
CTX = opt("-c", opt("--ctx-size", 4096, int), int)
NGL = opt("-ngl", opt("--n-gpu-layers", 0, int), int)
LOAD_S = opt("--fake-load-s", 0.2, float)
PP_TPS = opt("--fake-pp-tps", 20000.0, float)
TG_TPS = opt("--fake-tg-tps", 2000.0, float)
COLD = opt("--fake-cold-factor", 1.0, float)
GPU_MIB = opt("--fake-gpu-mib", 0, int)
STATUS = opt("--fake-status", 0, int)
CRASH_AFTER = opt("--fake-crash-after", -1, int)
EARLY_EOS = opt("--fake-early-eos", 0, int)
START = time.time()
READY = threading.Event()
STATE = {"requests": 0, "last_prompt": None}
LOCK = threading.Lock()


def say(line):
    print(f"0.00.{int((time.time() - START) * 1000):03d}.000 I {line}", flush=True)


def model_mib():
    try:
        return os.path.getsize(MODEL) / 2**20
    except OSError:
        return 100.0


def load():
    say("srv  llama_server: initializing ...")
    say(f"srv    load_model: loading model '{MODEL}'")
    code = opt("--fake-exit-code", None, int)
    time.sleep(min(LOAD_S, 0.05))
    if code is not None:
        print("error: failed to load model: unable to allocate memory", flush=True)
        os._exit(code)
    time.sleep(max(0.0, LOAD_S - 0.05))
    if flag("--fake-hang-load"):
        say("load_tensors: loading... (this fake never finishes)")
        return
    total = model_mib()
    gpu = min(float(GPU_MIB), total) if NGL else 0.0
    say(f"load_tensors:   CPU_Mapped model buffer size = {total - gpu:9.2f} MiB")
    if gpu:
        say(f"load_tensors:        CUDA0 model buffer size = {gpu:9.2f} MiB")
    kv = CTX * 0.012
    say(f"llama_kv_cache:        CPU KV buffer size = {kv:9.2f} MiB")
    say(f"sched_reserve:        CPU compute buffer size = {39.27:9.2f} MiB")
    say("srv  llama_server: model loaded")
    say(f"srv  llama_server: listening on http://127.0.0.1:{PORT}")
    READY.set()


def count_tokens(text):
    return len(re.findall(r"\w+|[^\w\s]", text))


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            if READY.is_set():
                return self._json(200, {"status": "ok"})
            return self._json(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
        if self.path == "/props":
            return self._json(200, {"total_slots": 1, "build_info": "fake"})
        if self.path == "/v1/models":
            return self._json(200, {"data": [{"id": "fake-model"}]})
        self._json(404, {"error": {"code": 404, "message": "not found"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/tokenize":
            return self._json(200, {"tokens": list(range(count_tokens(body.get("content", ""))))})
        chat = self.path == "/v1/chat/completions"
        if self.path == "/completion" and flag("--fake-chat-only"):
            return self._json(404, {"error": {"code": 404, "message": "File Not Found"}})
        if self.path not in ("/completion", "/v1/chat/completions"):
            return self._json(404, {"error": {"code": 404, "message": "not found"}})
        if not READY.is_set():
            return self._json(503, {"error": {"code": 503, "message": "Loading model"}})
        if STATUS:
            return self._json(STATUS, {"error": {"code": STATUS, "message": "simulated failure"}})
        prompt = (body.get("messages") or [{}])[-1].get("content", "") if chat else body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)
        n_ctx_tokens = count_tokens(prompt)
        want = int(body.get("n_predict", body.get("max_tokens", 16)) or 0)
        if n_ctx_tokens + want >= CTX:
            return self._json(400, {"error": {"code": 400, "type": "exceed_context_size_error",
                                              "message": f"request ({n_ctx_tokens} tokens) exceeds the available context size ({CTX} tokens)"}})
        with LOCK:
            idx = STATE["requests"]
            STATE["requests"] += 1
            digest = hashlib.sha1(prompt.encode()).hexdigest()
            cached = STATE["last_prompt"] == digest and flag("--fake-cache-prompt") and body.get("cache_prompt", True)
            STATE["last_prompt"] = digest
        if CRASH_AFTER >= 0 and idx >= CRASH_AFTER:
            os._exit(3)
        cache_n = n_ctx_tokens - 1 if cached else 0
        prompt_n = n_ctx_tokens - cache_n
        slow = COLD if idx == 0 else 1.0
        pp_s = prompt_n / (PP_TPS / slow)
        tg_s = 1.0 / (TG_TPS / slow)
        if EARLY_EOS and not body.get("ignore_eos"):
            want = min(want, EARLY_EOS)
        elif EARLY_EOS:
            want = min(want, EARLY_EOS * 1000)
        stream = bool(body.get("stream"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
        if stream:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            data = ("data: " + json.dumps(obj) + "\n\n").encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        time.sleep(pp_s)
        seed = int(body.get("seed", 0) or 0)
        for i in range(want):
            time.sleep(tg_s)
            word = f" w{(seed * 7 + i) % 97}"
            if chat:
                send({"choices": [{"finish_reason": None, "index": 0, "delta": {"content": word}}]})
            else:
                send({"index": 0, "content": word, "tokens": [i], "stop": False, "tokens_predicted": i + 1,
                      "tokens_evaluated": n_ctx_tokens})
        timings = {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": pp_s * 1000,
                   "prompt_per_token_ms": (pp_s * 1000 / prompt_n) if prompt_n else 0.0,
                   "prompt_per_second": (prompt_n / pp_s) if pp_s else 0.0,
                   "predicted_n": want, "predicted_ms": want * tg_s * 1000, "predicted_per_token_ms": tg_s * 1000,
                   "predicted_per_second": 1.0 / tg_s}
        extra = {} if flag("--fake-no-timings") else {"timings": timings}
        if chat:
            send({"choices": [{"finish_reason": "length", "index": 0, "delta": {}}], **extra})
            data = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        else:
            send({"index": 0, "content": "", "tokens": [], "stop": True, "tokens_predicted": want,
                  "tokens_evaluated": n_ctx_tokens, "stop_type": "limit", **extra})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main():
    if flag("--version"):
        print("version: 0.6.0-dev (build 9999, commit abc1234)\nbuilt with GNU 13.3.0 for Linux x86_64", file=sys.stderr, flush=True)
        return
    if flag("--list-devices"):
        print("Available devices:\n  (none)", file=sys.stderr, flush=True)
        return
    if flag("--fake-ignore-term"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("fake llama-server args:", " ".join(ARGS), flush=True)
    threading.Thread(target=load, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


def make_exe(directory):
    """A launcher that ServerManager can start like a real llama-server executable; returns its path."""
    from pathlib import Path
    d = Path(directory)
    script = Path(__file__).resolve()
    if os.name == "nt":
        exe = d / "llama-server.cmd"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        exe = d / "llama-server"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        exe.chmod(0o755)
    return exe


if __name__ == "__main__":
    main()
