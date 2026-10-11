import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

_HOME = tempfile.mkdtemp(prefix="duck-test-")
os.environ["DUCKSTOKEN_HOME"] = _HOME

from gguf_helper import write_gguf  # noqa: E402
from duckstoken import app as appmod, downloads, gguf, hardware, settings as st, tuner  # noqa: E402
from duckstoken.server_proc import build_args  # noqa: E402
from duckstoken.tools import ToolRunner  # noqa: E402


class SettingsTests(unittest.TestCase):
    def test_every_default_validates(self):
        for spec in st.SCHEMA:
            self.assertEqual(st.coerce(spec, spec["default"]), spec["default"], spec["key"])

    def test_keys_unique(self):
        keys = [s["key"] for s in st.SCHEMA]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreater(len(keys), 70)

    def test_clamp_and_select(self):
        self.assertEqual(st.coerce(st.SCHEMA_BY_KEY["port"], 999999), 65535)
        with self.assertRaises(ValueError):
            st.coerce(st.SCHEMA_BY_KEY["flash_attn"], "maybe")
        with self.assertRaises(ValueError):
            st.validate_patch({"nope": 1})

    def test_presets_only_use_known_keys(self):
        for name, p in st.PRESETS.items():
            st.validate_patch(p["values"])

    def test_store_roundtrip(self):
        p = Path(tempfile.mkdtemp()) / "s.json"
        s = st.SettingsStore(p)
        s.update({"ctx_size": 4096, "allowed_roots": "a\nb"})
        s2 = st.SettingsStore(p)
        self.assertEqual(s2.get("ctx_size"), 4096)
        self.assertEqual(s2.get("allowed_roots"), ["a", "b"])
        s2.reset(["ctx_size"])
        self.assertEqual(s2.get("ctx_size"), 8192)

    def test_build_args(self):
        s = dict(st.DEFAULTS, model_path="m.gguf", n_cpu_moe=36, mmap=False, cache_type_k="q8_0",
                 extra_args='--foo "a b"')
        a = build_args(s)
        self.assertIn("--no-mmap", a)
        self.assertEqual(a[a.index("--n-cpu-moe") + 1], "36")
        self.assertEqual(a[a.index("-ctk") + 1], "q8_0")
        self.assertNotIn("-ctv", a)
        self.assertNotIn("--mlock", a)
        self.assertEqual(a[-2:], ["--foo", "a b"])
        self.assertIn("--jinja", a)

    def test_split_args_keeps_windows_paths_and_strips_quotes(self):
        from duckstoken.server_proc import split_args
        self.assertEqual(split_args(r'--lora "C:\My Models\a.gguf" --x C:\y\z -n 3'),
                         ["--lora", r"C:\My Models\a.gguf", "--x", r"C:\y\z", "-n", "3"])
        self.assertEqual(split_args("--a 'b c'"), ["--a", "b c"])


class GGUFTests(unittest.TestCase):
    def test_summary(self):
        d = Path(tempfile.mkdtemp())
        write_gguf(d / "m-00001-of-00002.gguf", {"general.architecture": "gpt-oss", "general.name": "t",
                   "gpt-oss.block_count": 36, "gpt-oss.expert_count": 128, "gpt-oss.expert_used_count": 4}, 500)
        write_gguf(d / "m-00002-of-00002.gguf", {"general.architecture": "gpt-oss"}, 500)
        info = gguf.summarize(d / "m-00001-of-00002.gguf")
        self.assertEqual((info["block_count"], info["is_moe"], info["shards"]), (36, True, 2))

    def test_not_gguf(self):
        f = Path(tempfile.mkdtemp()) / "x.gguf"
        f.write_bytes(b"nope" * 10)
        with self.assertRaises(gguf.GGUFError):
            gguf.read_metadata(f)


HW = {"cpu": {"physical_cores": 6, "logical_cores": 12, "avx2": True, "avx512": False},
      "ram": {"total_gb": 48, "avail_gb": 40, "speed_mts": 2133, "sticks": 2, "bandwidth_gbs": 34.1},
      "gpus": [{"vram_mb": 2048, "vendor": "NVIDIA", "name": "GTX 750"}],
      "disks": [{"drive": "E:", "kind": "hdd", "free_gb": 3000}, {"drive": "C:", "kind": "nvme", "free_gb": 200}]}
M120 = {"size_gb": 59.0, "is_moe": True, "expert_count": 128, "expert_used_count": 4, "block_count": 36,
        "context_length": 131072, "architecture": "gpt-oss"}


class TunerTests(unittest.TestCase):
    def setUp(self):
        self._w = hardware._IS_WIN
        hardware._IS_WIN = True  # so drive-letter paths resolve like on Windows

    def tearDown(self):
        hardware._IS_WIN = self._w

    def test_hdd_vs_nvme(self):
        hdd = tuner.feasibility(HW, M120, "E:\\models\\m.gguf")
        ssd = tuner.feasibility(HW, M120, "C:\\models\\m.gguf")
        self.assertEqual(hdd["verdict"], "unusable")
        self.assertIn(ssd["verdict"], ("slow",))
        self.assertGreater(ssd["tps_range"][0], hdd["tps_range"][1] * 5)
        self.assertTrue(any("SSD" in w for w in hdd["warnings"]))

    def test_small_model_fits(self):
        m = dict(M120, size_gb=11.3, expert_count=32, block_count=24)
        f = tuner.feasibility(HW, m, "E:\\m.gguf")
        self.assertTrue(f["fits_in_memory"])
        self.assertGreater(f["tps_range"][1], 3)

    def test_backend_and_recommend(self):
        self.assertEqual(tuner.resolve_backend(HW), "cpu")  # 2 GB card: not worth it
        rec = tuner.recommend(HW, M120, "cpu")["values"]
        self.assertEqual((rec["threads"], rec["threads_batch"], rec["n_gpu_layers"]), (6, 12, 0))
        self.assertTrue(rec["mmap"])
        rec = tuner.recommend(HW, M120, "vulkan")["values"]
        self.assertEqual(rec["n_cpu_moe"], 36)
        self.assertLess(rec["n_gpu_layers"], 99)  # 2 GB cannot hold the shared weights
        st.validate_patch(rec)


class DownloadTests(unittest.TestCase):
    def test_pick_assets(self):
        A = lambda n: {"name": n, "url": "x", "size": 1}
        w = [A("llama-b1-bin-win-cpu-x64.zip"), A("llama-b1-bin-win-cuda-12.4-x64.zip"),
             A("llama-b1-bin-win-cuda-13.1-x64.zip"), A("cudart-llama-bin-win-cuda-12.4-x64.zip"),
             A("llama-b1-bin-win-vulkan-x64.zip")]
        n = lambda b: [a["name"] for a in downloads.pick_assets(w, b)]
        self.assertEqual(n("cpu"), ["llama-b1-bin-win-cpu-x64.zip"])
        self.assertEqual(n("cuda"), ["llama-b1-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"])
        self.assertEqual(n("vulkan"), ["llama-b1-bin-win-vulkan-x64.zip"])

    def test_repo_validation(self):
        jm = downloads.JobManager()
        for bad in ("../x", "a/../b", "a\\b/c", "justname", "a/b/c"):
            with self.assertRaises(ValueError):
                downloads.start_hf_download(jm, bad, [{"path": "m.gguf", "size": 1}], Path("."), "")

    def test_zip_slip(self):
        import zipfile
        d = Path(tempfile.mkdtemp())
        z = d / "bad.zip"
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("../evil.txt", "x")
        with self.assertRaises(RuntimeError):
            downloads.safe_extract(z, d / "out")

    def test_resume(self):
        import http.server, socketserver
        data = os.urandom(300_000)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                rng = self.headers.get("Range")
                start = int(rng.split("=")[1].rstrip("-")) if rng else 0
                self.send_response(206 if rng else 200)
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                self.wfile.write(data[start:])

        srv = socketserver.TCPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        d = Path(tempfile.mkdtemp())
        (d / "f.bin.part").write_bytes(data[:100_000])
        job = downloads.Job("t", "model")
        downloads.download_file(job, f"http://127.0.0.1:{srv.server_address[1]}/f", d / "f.bin")
        srv.shutdown()
        self.assertEqual((d / "f.bin").read_bytes(), data)


POSIX_ONLY = unittest.skipIf(os.name == "nt", "uses POSIX paths / symlinks / shebang scripts")


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp()).resolve()
        (self.dir / "a.txt").write_text("hello")
        self.s = dict(st.DEFAULTS, tools_enabled=True, allowed_roots=[str(self.dir)], tool_run_command=True)
        self.t = ToolRunner(lambda: self.s)

    def test_disabled_master(self):
        self.s["tools_enabled"] = False
        self.assertEqual(self.t.definitions(), [])
        self.assertFalse(self.t.execute("read_file", {"path": str(self.dir / "a.txt")})["ok"])

    def test_read_and_roots(self):
        self.assertEqual(self.t.execute("read_file", {"path": str(self.dir / "a.txt")})["output"], "hello")
        r = self.t.execute("read_file", {"path": "/etc/hostname"})
        self.assertFalse(r["ok"])
        self.assertIn("outside the allowed", r["output"])
        self.assertFalse(self.t.execute("read_file", {"path": str(self.dir / ".." / "x")})["ok"])

    @POSIX_ONLY
    def test_full_access_and_protected_write(self):
        self.s["full_pc_access"] = True
        self.assertTrue(self.t.execute("read_file", {"path": "/etc/hostname"})["ok"])
        r = self.t.execute("write_file", {"path": "/etc/duck-test", "content": "x"})
        self.assertFalse(r["ok"])
        self.assertIn("protected", r["output"])

    def test_write_find_command(self):
        self.assertTrue(self.t.execute("write_file", {"path": str(self.dir / "n" / "b.md"), "content": "hi"})["ok"])
        self.assertIn("b.md", self.t.execute("find_files", {"root": str(self.dir), "pattern": "*.md"})["output"])
        r = self.t.execute("run_command", {"command": "echo duck", "cwd": str(self.dir)})
        self.assertIn("duck", r["output"])
        self.assertFalse(self.t.execute("run_command", {"command": "echo", "cwd": "/"})["ok"])

    def test_approvals(self):
        self.assertTrue(self.t.needs_approval("write_file"))
        self.assertTrue(self.t.needs_approval("run_command"))
        self.assertFalse(self.t.needs_approval("read_file"))
        self.s["confirm_writes"] = False
        self.assertFalse(self.t.needs_approval("write_file"))

    @POSIX_ONLY
    def test_symlink_escape(self):
        link = self.dir / "link"
        os.symlink("/etc", link)
        self.assertFalse(self.t.execute("list_dir", {"path": str(link)})["ok"])


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from http.server import ThreadingHTTPServer
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), appmod.Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def req(self, path, body=None, token=True, host=None, ctype="application/json"):
        h = {}
        if token:
            h["X-Duck-Token"] = appmod.APP.token
        if host:
            h["Host"] = host
        data = None
        if body is not None:
            h["Content-Type"] = ctype
            data = json.dumps(body).encode()
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, headers=h)
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_security(self):
        self.assertEqual(self.req("/api/status", token=False)[0], 401)
        self.assertEqual(self.req("/api/status", host="evil.com")[0], 403)
        self.assertEqual(self.req("/api/tools/call", {"name": "x"}, ctype="text/plain")[0], 415)
        code, body = self.req("/")
        self.assertIn(appmod.APP.token.encode(), body)
        self.assertEqual(self.req("/static/../settings.py")[0], 404)
        self.assertEqual(self.req("/api/chat")[0], 404)  # GET on a POST endpoint

    def test_settings_api(self):
        code, body = self.req("/api/settings/update", {"values": {"ctx_size": 2048}})
        self.assertEqual(code, 200)
        self.assertEqual(appmod.APP.settings.get("ctx_size"), 2048)
        self.assertEqual(self.req("/api/settings/update", {"values": {"bogus": 1}})[0], 400)
        code, body = self.req("/api/schema")
        self.assertGreater(len(json.loads(body)["schema"]), 70)
        appmod.APP.settings.reset()


@POSIX_ONLY
class ServerLifecycleTests(unittest.TestCase):
    def test_start_chat_stop(self):
        from http.server import ThreadingHTTPServer
        a = appmod.APP
        a.settings.reset()
        a.settings.update({"llama_server_path": str(ROOT / "tests" / "fake_llama_server.py"),
                           "model_path": str(self._model()), "port": self._free_port(), "stop_server_on_exit": True})
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), appmod.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        hdr = {"X-Duck-Token": a.token, "Content-Type": "application/json"}

        def call(path, body=None):
            r = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=hdr)
            return urllib.request.urlopen(r)

        try:
            call("/api/server/start", {})
            deadline = time.time() + 15
            while time.time() < deadline and a.server.status()["state"] != "ready":
                time.sleep(0.3)
            self.assertEqual(a.server.status()["state"], "ready")
            with call("/api/chat", {"messages": [{"role": "user", "content": "hi"}]}) as r:
                text = r.read().decode()
            self.assertIn("Hello ", text)
            self.assertIn("effort=medium", text)
            self.assertIn("[DONE]", text)
            logs = json.load(call("/api/server/logs?since=0"))
            self.assertTrue(any("--n-cpu-moe" in l or "fake llama-server args" in l for l in logs["lines"]))
        finally:
            a.server.stop()
            httpd.shutdown()
        self.assertFalse(a.server.status()["running"])

    @staticmethod
    def _free_port():
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _model(self):
        p = Path(_HOME) / "m.gguf"
        write_gguf(p, {"general.architecture": "llama", "llama.block_count": 2}, 100)
        return p


if __name__ == "__main__":
    unittest.main()
