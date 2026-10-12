"""The client half of the benchmark: deterministic prompt, a cancellable streaming request, and per-run metrics.

Kept apart from `benchmark.py` (process lifecycle + record) so each stays small.  Two honesty rules live here:

* llama-server reports its own ``timings`` (prompt_per_second / predicted_per_second) in the final chunk.  Those are
  used when present and labelled ``measured`` ("llama-server timings").  When the engine does not send them, speeds
  are computed from client-side timestamps and labelled ``estimated`` ("client-side"): that estimate cannot separate
  prompt processing from the first generated token, so it is a lower bound.
* Prompt-processing and generation speed are always two separate numbers, and each run says how many tokens were
  really processed (a server that re-used its prompt cache would otherwise report a flattering prompt speed).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import queue
import socket
import statistics
import threading
import time
from urllib.parse import urlsplit

from .values import ESTIMATED, MEASURED, UNAVAILABLE, est, measured, unavailable

PROMPT_SENTENCE = "The quick brown fox jumps over the lazy dog while the sun sets behind the quiet hills."
TOKENS_PER_SENTENCE = 18   # real tokenizers give 18-20 for this sentence; the measured prompt_n is what gets reported
COMPLETION = "/completion"
CHAT = "/v1/chat/completions"


def build_prompt(prompt_tokens: int) -> tuple[str, str]:
    """Deterministic text of roughly `prompt_tokens` tokens (a fixed sentence repeated) and its sha256."""
    repeats = max(1, math.ceil(max(1, int(prompt_tokens)) / TOKENS_PER_SENTENCE))
    text = " ".join([PROMPT_SENTENCE] * repeats)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def free_port() -> int:
    """A currently free TCP port on 127.0.0.1 (another process could grab it before the server binds; rare)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------------------------- statistics
def summarize_runs(values: list[float]) -> dict:
    """``{mean, stdev, min, max, n}`` of the numbers in `values` (stdev is the sample stdev, 0.0 for one value)."""
    xs = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    if not xs:
        return {"mean": None, "stdev": None, "min": None, "max": None, "n": 0}
    return {"mean": statistics.fmean(xs), "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs), "n": len(xs)}


def stat_val(runs: list[dict], key: str, digits: int = 3) -> dict:
    """Aggregate one metric over run dicts into a values dict whose ``value`` is the mean.

    Extra keys carry ``mean/stdev/min/max/n``.  The status is ``measured`` only if every contributing run was
    measured; any client-side estimate makes the aggregate ``estimated``; no data at all -> ``unavailable``.
    """
    pairs = [(r.get(key) or {}) for r in runs]
    usable = [p for p in pairs if isinstance(p.get("value"), (int, float)) and p.get("status") != UNAVAILABLE]
    if not usable:
        why = next((p.get("note") for p in pairs if p.get("note")), "no run produced this number")
        return {**unavailable(why, "benchmark runs"), "mean": None, "stdev": None, "min": None, "max": None, "n": 0}
    st = summarize_runs([p["value"] for p in usable])
    status = MEASURED if all(p["status"] == MEASURED for p in usable) else ESTIMATED
    rounded = {k: (round(v, digits) if isinstance(v, float) else v) for k, v in st.items()}
    return {"value": rounded["mean"], "status": status, "source": usable[0].get("source", ""),
            "note": f"mean of {st['n']} run(s)" + ("; includes client-side estimates" if status == ESTIMATED else ""),
            **rounded}


# ---------------------------------------------------------------------------------------------- HTTP streaming
def _content_of(obj: dict, chat: bool) -> str:
    if chat:
        choice = (obj.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        return (delta.get("content") or "") + (delta.get("reasoning_content") or "")
    return obj.get("content") or ""


def stream_request(base_url: str, endpoint: str, payload: dict, cancel: threading.Event | None, timeout_s: float,
                   clock=time.perf_counter) -> dict:
    """POST `payload`, read the server-sent events, stop on cancel/timeout.  Never raises.

    Returns ``{"outcome": ok|cancelled|timeout|http_error|error|unsupported, "http_status", "error", "t_sent",
    "t_first", "t_last", "t_end", "chunks", "timings", "final", "text"}`` with times from `clock`.  A reader thread
    does the blocking I/O so cancel and the overall deadline are honoured within ~100 ms, even mid-generation.
    """
    chat = endpoint == CHAT
    parts = urlsplit(base_url)
    out = {"outcome": "error", "http_status": None, "error": "", "t_sent": None, "t_first": None, "t_last": None,
           "t_end": None, "chunks": 0, "timings": None, "final": None, "text": "", "endpoint": endpoint}
    q: queue.Queue = queue.Queue()
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=max(5.0, timeout_s))

    def reader() -> None:
        try:
            body = json.dumps(payload).encode("utf-8")
            conn.request("POST", endpoint, body, {"Content-Type": "application/json", "Accept": "text/event-stream"})
            q.put(("sent", clock(), None))
            resp = conn.getresponse()
            if resp.status != 200:
                q.put(("http", clock(), (resp.status, resp.read(8192))))
                return
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    q.put(("data", clock(), json.loads(data)))
                except ValueError:
                    continue
            q.put(("end", clock(), None))
        except Exception as e:  # noqa: BLE001 - reported through the queue
            q.put(("error", clock(), f"{type(e).__name__}: {e}"))

    thread = threading.Thread(target=reader, name="bench-stream", daemon=True)
    thread.start()
    deadline = clock() + timeout_s
    aborted: str | None = None
    try:
        while True:
            if cancel is not None and cancel.is_set():
                aborted = "cancelled"
                break
            if clock() > deadline:
                aborted = "timeout"
                break
            try:
                kind, t, item = q.get(timeout=0.1)
            except queue.Empty:
                if not thread.is_alive() and q.empty():
                    out["error"] = "connection closed without a response"
                    break
                continue
            if kind == "sent":
                out["t_sent"] = t
            elif kind == "http":
                status, body = item
                out.update(outcome="unsupported" if status in (404, 405, 501) else "http_error", http_status=status,
                           error=_error_text(body, status), t_end=t)
                break
            elif kind == "data":
                text = _content_of(item, chat)
                if text:
                    out["chunks"] += 1
                    out["text"] += text
                    out["t_last"] = t
                    if out["t_first"] is None:
                        out["t_first"] = t
                if item.get("timings"):
                    out["timings"] = item["timings"]
                if (not chat and item.get("stop") is True) or (chat and (item.get("choices") or [{}])[0].get("finish_reason")):
                    out["final"] = item
            elif kind == "end":
                out.update(outcome="ok", t_end=t)
                break
            elif kind == "error":
                out.update(error=item, t_end=t)
                break
    finally:
        if aborted:
            out.update(outcome=aborted, t_end=clock(), error=f"{aborted} while waiting for the server")
        _abort_connection(conn)
        thread.join(2.0)
    if out["outcome"] == "ok" and out["t_sent"] is None:
        out["outcome"] = "error"
    return out


def http_get(base_url: str, path: str, timeout: float = 2.0) -> int | None:
    """Status code of ``GET base_url + path`` or None when nothing answers.  Uses http.client directly so a proxy
    configured in the environment (HTTP_PROXY) can never intercept the local health check."""
    parts = urlsplit(base_url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        resp.read(1024)
        return resp.status
    except (OSError, http.client.HTTPException):
        return None
    finally:
        conn.close()


def _abort_connection(conn: http.client.HTTPConnection) -> None:
    try:
        if conn.sock is not None:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)   # unblocks a reader stuck in recv
            except OSError:
                pass
        conn.close()
    except Exception:  # noqa: BLE001
        pass


def _error_text(body: bytes, status: int) -> str:
    try:
        obj = json.loads(body.decode("utf-8", "replace"))
        err = obj.get("error", obj)
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return f"HTTP {status}: {msg}"
    except (ValueError, AttributeError):
        return f"HTTP {status}: {body[:200].decode('utf-8', 'replace')}"


# ---------------------------------------------------------------------------------------------- one run
def completion_payload(endpoint: str, prompt: str, gen_tokens: int, seed: int, temperature: float) -> dict:
    """Request body: deterministic sampling, no prompt-cache reuse (so every run really processes the prompt)."""
    base = {"stream": True, "temperature": float(temperature), "seed": int(seed), "cache_prompt": False,
            "ignore_eos": True}   # ignore_eos: always generate gen_tokens tokens so runs are comparable
    if endpoint == CHAT:
        return {**base, "messages": [{"role": "user", "content": prompt}], "max_tokens": int(gen_tokens)}
    return {**base, "prompt": prompt, "n_predict": int(gen_tokens)}


def run_metrics(raw: dict, index: int, phase: str, nominal_prompt_tokens: int) -> dict:
    """Turn the raw stream measurements into one run record of honest values."""
    t_sent, t_first, t_last, t_end = raw["t_sent"], raw["t_first"], raw["t_last"], raw["t_end"]
    timings = raw.get("timings") or {}
    final = raw.get("final") or {}
    cs = "client-side stream timing"
    ts = "llama-server timings"
    run = {"index": index, "phase": phase, "endpoint": raw.get("endpoint"), "stop_type": final.get("stop_type"),
           "source": "server_timings" if timings else "client_side", "cached_tokens": int(timings.get("cache_n") or 0)}
    run["ttft_s"] = (measured(round(t_first - t_sent, 4), cs, "request sent until the first generated token arrived")
                     if t_first is not None and t_sent is not None else unavailable("no token was streamed", cs))
    run["total_s"] = (measured(round(t_end - t_sent, 4), cs, "request sent until the stream ended")
                      if t_end is not None and t_sent is not None else unavailable("request did not finish", cs))
    n_gen_client = raw.get("chunks", 0)
    if timings and timings.get("predicted_n") is not None:
        run["prompt_tokens"] = measured(int(timings.get("prompt_n") or 0), ts, "tokens actually processed this run")
        run["gen_tokens"] = measured(int(timings["predicted_n"]), ts)
        pps, gps = timings.get("prompt_per_second"), timings.get("predicted_per_second")
        run["prompt_tps"] = (measured(round(float(pps), 2), ts, "prompt processing only") if pps
                             else unavailable("no prompt tokens were processed", ts))
        run["gen_tps"] = (measured(round(float(gps), 3), ts, "token generation only") if gps
                          else unavailable("fewer than two tokens generated", ts))
    else:
        n_prompt = final.get("tokens_evaluated")
        run["prompt_tokens"] = (measured(int(n_prompt), "llama-server final chunk") if n_prompt
                                else est(int(nominal_prompt_tokens), "benchmark spec", "target size; real token count unknown"))
        run["gen_tokens"] = est(n_gen_client, cs, "streamed chunks, assumed one token each") if n_gen_client else \
            unavailable("no token was streamed", cs)
        ttft = (t_first - t_sent) if (t_first is not None and t_sent is not None) else None
        n_p = run["prompt_tokens"]["value"]
        run["prompt_tps"] = (est(round(n_p / ttft, 2), cs, "prompt tokens / time to first token: a lower bound, "
                                 "includes generating the first token") if ttft and ttft > 0 and n_p
                             else unavailable("engine sent no timings and no first token arrived", cs))
        span = (t_last - t_first) if (t_last is not None and t_first is not None) else 0
        run["gen_tps"] = (est(round((n_gen_client - 1) / span, 3), cs, "streamed chunks after the first / time between them")
                          if n_gen_client >= 2 and span > 0 else unavailable("fewer than two tokens streamed", cs))
    return run
