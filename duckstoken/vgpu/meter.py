"""Measures real chat requests as they stream through the proxy: time to first token, prompt-processing speed and
generation speed. Uses llama-server's own `timings` block when the engine sends one; otherwise falls back to client-side
timestamps and says so (status `experimental`)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from .values import experimental_note, measured, unavailable


class RequestMeter:
    def __init__(self, clock: Callable[[], float] = time.perf_counter, wall: Callable[[], float] = time.time):
        self._clock, self._wall = clock, wall
        self.t0 = clock()
        self.started = wall()
        self.first: float | None = None
        self.last: float | None = None
        self.chunks = 0
        self.timings: dict | None = None
        self.usage: dict | None = None

    def feed(self, line: bytes) -> None:
        """One raw line of the upstream SSE stream (cheap checks first; JSON only when it can matter)."""
        if not line.startswith(b"data:"):
            return
        now = self._clock()
        has_content = b'"content"' in line and b'"content":null' not in line and b'"content":""' not in line
        if has_content or b'"reasoning_content"' in line:
            if self.first is None:
                self.first = now
            self.last = now
            self.chunks += 1
        if b'"timings"' in line or b'"usage"' in line:
            try:
                obj = json.loads(line[5:].strip())
            except ValueError:
                return
            if isinstance(obj.get("timings"), dict):
                self.timings = obj["timings"]
            if isinstance(obj.get("usage"), dict):
                self.usage = obj["usage"]

    def feed_body(self, body: bytes) -> None:
        """A non-streaming JSON reply."""
        now = self._clock()
        self.first = self.last = now
        try:
            obj = json.loads(body)
        except ValueError:
            return
        if isinstance(obj, dict):
            self.timings = obj.get("timings") if isinstance(obj.get("timings"), dict) else None
            self.usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
            self.chunks = int((self.usage or {}).get("completion_tokens") or 0)

    def result(self, engine: str = "") -> dict:
        t = self.timings or {}
        total = (self.last or self._clock()) - self.t0
        ttft = (self.first - self.t0) if self.first is not None else None
        out = {"t": self.started, "engine": engine, "total_s": measured(round(total, 3), "client timestamps"),
               "ttft_s": measured(round(ttft, 3), "client timestamps") if ttft is not None else unavailable("no token arrived")}
        if t.get("predicted_per_second"):
            out["gen_tps"] = measured(round(float(t["predicted_per_second"]), 2), "llama-server timings")
            out["gen_tokens"] = measured(int(t.get("predicted_n", 0)), "llama-server timings")
        elif self.first is not None and self.last is not None and self.chunks > 1 and self.last > self.first:
            out["gen_tps"] = experimental_note((self.chunks - 1) / (self.last - self.first),
                                               "stream chunks / seconds (assumes one token per chunk)")
            out["gen_tokens"] = experimental_note(self.chunks, "stream chunks")
        else:
            out["gen_tps"] = unavailable("not enough streamed tokens to measure")
        if t.get("prompt_per_second"):
            out["prompt_tps"] = measured(round(float(t["prompt_per_second"]), 1), "llama-server timings")
            out["prompt_tokens"] = measured(int(t.get("prompt_n", 0)), "llama-server timings")
        elif self.usage and self.usage.get("prompt_tokens") is not None:
            out["prompt_tokens"] = measured(int(self.usage["prompt_tokens"]), "usage")
            out["prompt_tps"] = unavailable("engine did not report prompt timings")
        else:
            out["prompt_tps"] = unavailable("engine did not report prompt timings")
        return out
