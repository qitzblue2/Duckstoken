"""End-to-end through the real HTTP handler: learning endpoints, chat augmentation, engines."""
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("DUCKSTOKEN_HOME", tempfile.mkdtemp(prefix="duck-int-"))

from gguf_helper import write_gguf  # noqa: E402
from duckstoken import app as appmod  # noqa: E402

POSIX_ONLY = unittest.skipIf(os.name == "nt", "fake llama-server is a shebang script")


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), appmod.Handler)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        appmod.APP.server.stop()

    def setUp(self):
        a = appmod.APP
        a.server.stop()
        a.settings.reset()
        a.memory.clear()
        a.db.execute("DELETE FROM exchanges")
        a.db.execute("DELETE FROM queue")
        a.db.execute("DELETE FROM dataset_items")

    def call(self, path, body=None, raw=False):
        h = {"X-Duck-Token": appmod.APP.token}
        data = None
        if body is not None:
            h["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        req = urllib.request.Request(self.base + "/api/" + path, data=data, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = r.read()
                return payload if raw else json.loads(payload)
        except urllib.error.HTTPError as e:
            return {"_status": e.code, **json.loads(e.read() or b"{}")}


class LearningEndpoints(Api):
    def test_memory_crud_and_preview(self):
        item = self.call("memory/add", {"text": "The user's cat is called Biscuit", "kind": "fact"})["item"]
        self.assertEqual(self.call("memory/list?q=biscuit")["total"], 1)
        pre = self.call("memory/preview?q=what%20is%20my%20cat%20called")
        self.assertIn("Biscuit", pre["text"])
        self.assertEqual(self.call("memory/update", {"id": item["id"], "pinned": True})["item"]["pinned"], True)
        self.assertEqual(self.call("memory/stats")["total"], 1)
        self.assertEqual(self.call("memory/add", {"text": ""})["_status"], 400)
        self.assertTrue(self.call("memory/delete", {"id": item["id"]})["ok"])

    def test_memory_documents_and_import_export(self):
        r = self.call("memory/add_doc", {"title": "Notes", "text": "Duckstoken stores memories locally. " * 80})
        self.assertGreaterEqual(r["chunks"], 2)
        exported = self.call("memory/export")["items"]
        self.assertEqual(self.call("memory/import", {"items": exported})["added"], 0)  # deduped

    def test_feedback_flow_creates_lesson_and_dataset(self):
        rec = self.call("feedback/record", {"chat_id": "c1", "msg_index": 1, "answer": "Paris is in France, obviously.",
                                            "messages": [{"role": "user", "content": "Where is Paris located?"}]})
        xid = rec["id"]
        self.assertEqual(self.call("feedback/rate", {"id": xid, "rating": -1, "note": "Be shorter"})["item"]["rating"], -1)
        self.assertEqual(self.call("memory/list?kind=lesson")["total"], 1)
        self.call("feedback/correct", {"id": xid, "correction": "Paris is in France."})
        self.assertEqual(self.call("dataset/refresh", {})["added"], 2)
        ds = self.call("dataset/list")
        self.assertEqual(ds["total"], 2)
        self.call("dataset/set_status", {"ids": [i["id"] for i in ds["items"]], "status": "approved"})
        blob = self.call("dataset/export?only_approved=1&redact=1", raw=True)
        names = zipfile.ZipFile(io.BytesIO(blob)).namelist()
        self.assertTrue({"sft.jsonl", "pref.jsonl", "manifest.json"} <= set(names))
        self.assertEqual(self.call("feedback/stats")["corrected"], 1)

    def test_feedback_respects_privacy_switch(self):
        appmod.APP.settings.update({"record_exchanges": False})
        r = self.call("feedback/record", {"chat_id": "c", "msg_index": 1, "answer": "x", "messages": []})
        self.assertIsNone(r["id"])
        self.assertEqual(self.call("feedback/stats")["total"], 0)

    def test_queue_requires_big_model_and_validates(self):
        self.assertEqual(self.call("queue/enqueue", {"messages": [{"role": "user", "content": "hi"}]})["_status"], 400)
        appmod.APP.settings.update({"big_model_path": str(ROOT / "README.md")})
        job = self.call("queue/enqueue", {"messages": [{"role": "user", "content": "hard question"}]})["item"]
        self.assertEqual(job["status"], "queued")
        self.assertEqual(self.call("queue/status")["counts"]["queued"], 1)
        self.assertTrue(self.call("queue/cancel", {"id": job["id"]})["ok"])
        self.assertEqual(self.call("queue/get?id=%d" % job["id"])["item"]["status"], "cancelled")

    def test_web_test_without_network_returns_clean_error(self):
        appmod.APP.settings.update({"web_provider": "searxng", "searxng_url": "http://127.0.0.1:1"})
        r = self.call("web/test", {"query": "x"})
        self.assertEqual(r["_status"], 400)
        self.assertNotIn("Traceback", r["error"])


@POSIX_ONLY
class ChatAndEngines(Api):
    def _model(self):
        p = Path(os.environ["DUCKSTOKEN_HOME"]) / "m.gguf"
        write_gguf(p, {"general.architecture": "llama", "llama.block_count": 2}, 100)
        return str(p)

    def _wait_ready(self):
        deadline = time.time() + 20
        while time.time() < deadline and appmod.APP.server.status()["state"] != "ready":
            time.sleep(0.2)
        self.assertEqual(appmod.APP.server.status()["state"], "ready", appmod.APP.server.status())

    def _chat(self, text):
        raw = self.call("chat", {"messages": [{"role": "user", "content": text}]}, raw=True).decode()
        events = [json.loads(l[5:]) for l in raw.splitlines() if l.startswith("data:") and "[DONE]" not in l]
        ctx = next(e["duck_context"] for e in events if "duck_context" in e)
        reply = "".join(e["choices"][0]["delta"].get("content", "") for e in events if e.get("choices"))
        return ctx, reply

    def test_llamacpp_chat_injects_memory_and_reports_context(self):
        port = free_port()
        appmod.APP.settings.update({"llama_server_path": str(ROOT / "tests" / "fake_llama_server.py"),
                                    "model_path": self._model(), "port": port, "smart_prompt": "concise"})
        appmod.APP.memory.add("The user's favourite colour is teal", kind="preference")
        self.call("server/start", {})
        self._wait_ready()
        ctx, reply = self._chat("what is my favourite colour?")
        self.assertEqual(len(ctx["memory_ids"]), 1)
        self.assertIn("teal", reply)             # the memory block reached the model in the system message
        self.assertIn("Answer directly", reply)  # and so did the built-in prompt
        self.assertIn("name=local", reply)
        ctx, _ = self._chat("remember that my dog is called Rex")
        self.assertTrue(ctx["remembered"])
        self.assertEqual(self.call("memory/list?q=Rex")["total"], 1)

    def test_external_engine_end_to_end(self):
        # the fake llama-server doubles as an "Ollama": it answers /v1/models and chat completions
        import subprocess
        port = free_port()
        proc = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_llama_server.py"), "--port", str(port)])
        try:
            appmod.APP.settings.update({"engine": "external", "external_preset": "custom",
                                        "external_url": f"http://127.0.0.1:{port}/v1"})
            deadline = time.time() + 10
            while time.time() < deadline and not self.call("engine/test", {})["ok"]:
                time.sleep(0.2)
            probe = self.call("engine/test", {})
            self.assertTrue(probe["ok"], probe)
            self.assertEqual(probe["models"], ["fake-model"])  # embedding models are hidden
            self.call("server/start", {})
            self._wait_ready()
            self.assertEqual(self.call("status")["engine"], "external")
            _, reply = self._chat("hello")
            self.assertIn("name=fake-model", reply)
            self.call("server/stop", {})
            self.assertEqual(self.call("status")["server"]["state"], "stopped")
        finally:
            proc.terminate()
            proc.wait(10)

    def test_external_engine_reports_unreachable_server(self):
        appmod.APP.settings.update({"engine": "external", "external_preset": "custom",
                                    "external_url": f"http://127.0.0.1:{free_port()}"})
        r = self.call("server/start", {})
        self.assertEqual(r["_status"], 400)
        self.assertIn("Cannot reach", r["error"])

    def test_engine_info_lists_all_engines(self):
        info = self.call("engine/info")
        self.assertEqual([e["id"] for e in info["engines"]], ["llamacpp", "external", "builtin"])
        self.assertEqual(info["selected"], "llamacpp")


if __name__ == "__main__":
    unittest.main()
