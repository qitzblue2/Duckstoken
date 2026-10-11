#!/usr/bin/env python3
"""Stand-in for llama-server used by the tests: /health + streaming /v1/chat/completions."""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1]) if "--port" in args else 8080
START = time.time()
print("fake llama-server args:", " ".join(args), flush=True)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _j(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._j(503 if time.time() - START < 1.5 else 200, {"status": "ok"})
        self._j(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        msgs = body["messages"]
        last = msgs[-1]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(delta, **extra):
            d = {"choices": [{"index": 0, "delta": delta}], **extra}
            self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
            self.wfile.flush()

        if last["role"] == "user" and "list files" in last["content"] and body.get("tools"):
            emit({"reasoning_content": "I should look at the folder."})
            emit({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                  "function": {"name": "list_dir", "arguments": ""}}]})
            emit({"tool_calls": [{"index": 0, "function": {"arguments": json.dumps({"path": body["_dir"] if "_dir" in body else "/tmp"})}}]})
        else:
            for part in ["thinking ", "hard"]:
                emit({"reasoning_content": part})
            for part in ["Hello ", "from **fake** ", f"model (effort={body.get('chat_template_kwargs', {}).get('reasoning_effort')}, tools={len(body.get('tools') or [])}, msgs={len(msgs)})"]:
                emit({"content": part})
                time.sleep(0.05)
        emit({}, timings={"predicted_n": 5, "predicted_per_second": 4.2, "prompt_per_second": 99.0})
        self.wfile.write(b"data: [DONE]\n\n")


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
