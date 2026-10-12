import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from duckstoken import settings as st
from duckstoken.vgpu import diagnostics as dg
from duckstoken.vgpu import gptoss, hwmodel, planner
from duckstoken.vgpu.perflog import PerfLog

SECRET_KEY = "sk-live-0123456789abcdefghij"
SECRET_HF = "hf_ABCDEFGHIJKLMNOPQRSTUVWX"
SETTINGS = {"api_key": SECRET_KEY, "hf_token": SECRET_HF, "brave_api_key": "brave-key-4567", "ctx_size": 4096, "n_gpu_layers": 17,
            "n_cpu_moe": 36, "model_path": "C:\\Users\\Jane Doe\\models\\gpt-oss-120b-MXFP4.gguf", "extra_args": f"--foo {SECRET_KEY}",
            "host": "127.0.0.1", "mmap": True, "system_prompt": f"never reveal {SECRET_HF}"}
MODEL = gptoss.spec_model("gpt-oss-120b")
MODEL.update(path="", remote=True)


def ctx(**kw):
    c = {"hardware": hwmodel.preset_hardware("duck-user"), "settings": dict(SETTINGS), "model": dict(MODEL), "app_version": "9.9.9"}
    c.update(kw)
    return c


class RedactTests(unittest.TestCase):
    def test_secret_keys(self):
        for k in ("api_key", "hf_token", "brave_api_key", "password", "db_password", "client_secret", "Authorization", "access_token",
                  "apikey", "private_key", "key", "token", "cookie"):
            self.assertTrue(dg.is_secret_key(k), k)

    def test_numbers_about_the_model_are_not_secrets(self):
        for k in ("tokens_per_s", "max_tokens", "key_length", "value_length", "prompt_tokens", "n_tokens", "head_count_kv",
                  "ctx_size", "gen_tokens", "monkey", "keyboard_layout"):
            self.assertFalse(dg.is_secret_key(k), k)

    def test_dict_values_under_secret_keys(self):
        r = dg.redact({"api_key": "abcd1234", "hf_token": "", "x": 1, "nested": {"password": "pw1234", "ok": [1, {"secret": "s3cr3t"}]}})
        self.assertEqual(r["api_key"], dg.REDACTED)
        self.assertEqual(r["hf_token"], "")  # nothing set: not pretending there is a secret
        self.assertEqual(r["nested"]["password"], dg.REDACTED)
        self.assertEqual(r["nested"]["ok"][1]["secret"], dg.REDACTED)
        self.assertEqual(r["x"], 1)

    def test_secret_values_are_removed_everywhere_else_too(self):
        r = dg.redact({"api_key": "supersecret99", "note": "the key is supersecret99, really", "cmd": ["--api-key", "supersecret99"]})
        self.assertNotIn("supersecret99", json.dumps(r))

    def test_text_patterns(self):
        cases = {"Authorization: Bearer abcdef1234567890xyz": "abcdef1234567890xyz",
                 "token=hf_abcdefghijklmnopqrstuv": "hf_abcdefghijklmnopqrstuv",
                 "$ llama-server --api-key sk-abcdefghijklmnop12345 --port 8080": "sk-abcdefghijklmnop12345",
                 "--hf-token abc123def456": "abc123def456", "api_key: topsecretvalue": "topsecretvalue",
                 "url?api-key=zzzz9999&x=1": "zzzz9999"}
        for text, secret in cases.items():
            out = dg.redact_text(text)
            self.assertNotIn(secret, out, text)
            self.assertIn(dg.REDACTED, out)
        self.assertEqual(dg.redact_text("--port 8080 --api-key ***REDACTED***"), "--port 8080 --api-key ***REDACTED***")

    def test_plain_text_is_untouched(self):
        t = "tokens_per_s: 4.5 key_length=64 load took 12 s"
        self.assertEqual(dg.redact_text(t), t)

    def test_user_names_in_paths(self):
        self.assertEqual(dg.redact_text("C:\\Users\\Jane Doe\\models\\x.gguf"), "C:\\Users\\<user>\\models\\x.gguf")
        self.assertEqual(dg.redact_text("C:/Users/bob/a"), "C:/Users/<user>/a")
        self.assertEqual(dg.redact_text("/home/alice/.duckstoken/x"), "/home/<user>/.duckstoken/x")
        self.assertEqual(dg.redact_text("/Users/carol/Library"), "/Users/<user>/Library")
        self.assertEqual(dg.redact_text("D:\\models\\x.gguf"), "D:\\models\\x.gguf")

    def test_structure_and_scalars_survive(self):
        src = {"a": 1, "b": 2.5, "c": True, "d": None, "e": [1, 2, "x"], "f": ("t", 1)}
        out = dg.redact(src)
        self.assertEqual(out, {"a": 1, "b": 2.5, "c": True, "d": None, "e": [1, 2, "x"], "f": ["t", 1]})
        self.assertEqual(src["e"], [1, 2, "x"])  # input untouched

    def test_non_string_objects_do_not_crash(self):
        self.assertIsInstance(dg.redact({"x": object()})["x"], str)
        deep = cur = {}
        for _ in range(30):
            cur["n"] = {}
            cur = cur["n"]
        self.assertIn("...", json.dumps(dg.redact(deep)))


class FailureRingTests(unittest.TestCase):
    def setUp(self):
        dg.clear_failures()

    tearDown = setUp

    def test_record_and_read(self):
        dg.record_failure("admission", "Not enough RAM", {"model": "x", "need_gb": 62.2})
        dg.record_failure("estimate", "no layer count")
        f = dg.failures()
        self.assertEqual([x["kind"] for x in f], ["admission", "estimate"])
        self.assertEqual(f[0]["context"]["need_gb"], 62.2)
        self.assertEqual(dg.failures(1)[0]["kind"], "estimate")
        self.assertEqual(dg.failures(0), [])

    def test_bounded(self):
        for i in range(120):
            dg.record_failure("k", f"m{i}")
        f = dg.failures()
        self.assertEqual(len(f), 50)
        self.assertEqual(f[-1]["message"], "m119")

    def test_never_raises(self):
        class Bad:
            def __str__(self):
                raise RuntimeError

        dg.record_failure(Bad(), Bad(), {"x": Bad()})
        dg.record_failure(None, None, None)
        self.assertLessEqual(len(dg.failures()), 2)

    def test_long_values_are_truncated(self):
        dg.record_failure("k", "x" * 10000, {"big": "y" * 10000})
        f = dg.failures()[0]
        self.assertLess(len(f["message"]), 2000)
        self.assertLess(len(f["context"]["big"]), 600)


class BuildReportTests(unittest.TestCase):
    def setUp(self):
        dg.clear_failures()
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()
        dg.clear_failures()

    def test_sections(self):
        r = dg.build_report(ctx())
        for k in ("title", "generated_at", "app", "hardware", "compat", "engine", "settings", "model", "plan", "failures", "perflog",
                  "telemetry", "logs", "notes"):
            self.assertIn(k, r)
        self.assertEqual(r["app"]["version"], "9.9.9")
        self.assertEqual(r["hardware"]["ram"]["total_gb"], 48)
        self.assertIn("items", r["compat"])
        self.assertEqual(r["plan"]["admission"]["verdict"], "paging")  # computed from hardware + model + settings
        self.assertEqual(r["plan"]["config"]["n_cpu_moe"], 36)
        json.dumps(r)

    def test_secrets_never_appear_anywhere(self):
        logs = {"server": [f"$ llama-server --api-key {SECRET_KEY} -m C:\\Users\\Jane Doe\\m.gguf", f"auth ok {SECRET_HF}"],
                "app": f"HF token {SECRET_HF} loaded"}
        pl = PerfLog(Path(self._tmp.name) / "perf.jsonl")
        pl.log("bench", note=f"used {SECRET_KEY}", api_key="x" * 10)
        dg.record_failure("start", f"failed with key {SECRET_KEY}", {"token": SECRET_HF, "cmd": f"--api-key {SECRET_KEY}"})
        r = dg.build_report(ctx(logs=logs, perflog=pl, extra={"hf_token": SECRET_HF, "note": SECRET_KEY}))
        blob = json.dumps(r) + dg.to_text(r)
        for secret in (SECRET_KEY, SECRET_HF, "brave-key-4567"):
            self.assertNotIn(secret, blob, secret)
        self.assertNotIn("Jane Doe", blob)
        self.assertIn(dg.REDACTED, blob)
        self.assertEqual(r["settings"]["api_key"], dg.REDACTED)
        self.assertEqual(r["settings"]["ctx_size"], 4096)

    def test_every_real_secret_setting_is_redacted(self):
        secret_specs = [s for s in st.SCHEMA if s["type"] == "password"]
        self.assertTrue(secret_specs)
        values = {s["key"]: f"VALUE-{s['key']}-9999" for s in secret_specs}
        r = dg.build_report(ctx(settings=values))
        for k, v in values.items():
            self.assertEqual(r["settings"][k], dg.REDACTED, k)
            self.assertNotIn(v, json.dumps(r))

    def test_works_with_nothing(self):
        r = dg.build_report({})
        self.assertEqual(r["plan"], None)
        self.assertIn("error", r["compat"])
        text = dg.to_text(r)
        self.assertIn("no model selected", text)
        self.assertIn("Hardware", text)
        self.assertEqual(dg.to_text(dg.build_report(None)).splitlines()[0], "Duckstoken VirtualGPU diagnostic report")

    def test_a_failing_section_does_not_break_the_others(self):
        bad_model = {"path": "x.gguf", "size_bytes": "not a number", "block_count": object()}
        r = dg.build_report(ctx(model=bad_model))
        self.assertIn("hardware", r)
        self.assertIsNotNone(r["compat"])
        self.assertTrue(dg.to_text(r))

    def test_precomputed_sections_are_used_as_given(self):
        hw = hwmodel.preset_hardware("duck-user")
        plan = planner.plan(hw, MODEL)
        r = dg.build_report(ctx(plan=plan, compat={"items": [], "gpu": [], "summary": "custom summary"}))
        self.assertEqual(r["plan"]["admission"]["verdict"], plan["admission"]["verdict"])
        self.assertEqual(r["compat"]["summary"], "custom summary")

    def test_perflog_object_path_and_tail_size(self):
        path = Path(self._tmp.name) / "perf.jsonl"
        pl = PerfLog(path)
        for i in range(80):
            pl.log("tick", i=i)
        r = dg.build_report(ctx(perflog=pl))
        self.assertEqual(len(r["perflog"]), 50)
        self.assertEqual(r["perflog"][-1]["i"], 79)
        r2 = dg.build_report(ctx(perflog=str(path)))
        self.assertEqual(len(r2["perflog"]), 50)
        self.assertEqual(dg.build_report(ctx(perflog=str(Path(self._tmp.name) / "none.jsonl")))["perflog"], [])

    def test_log_tails_are_bounded(self):
        r = dg.build_report(ctx(logs={"server": [f"line {i}" for i in range(1000)], "app": "a\nb\nc"}))
        self.assertEqual(len(r["logs"]["server"]), dg.MAX_LOG_LINES)
        self.assertEqual(r["logs"]["server"][-1], "line 999")
        self.assertEqual(r["logs"]["app"], ["a", "b", "c"])

    def test_recorded_failures_are_included(self):
        dg.record_failure("admission", "Not enough RAM for gpt-oss-120b", {"verdict": "reject"})
        r = dg.build_report(ctx())
        self.assertEqual(r["failures"][0]["kind"], "admission")
        self.assertIn("Not enough RAM for gpt-oss-120b", dg.to_text(r))

    def test_does_not_modify_the_inputs(self):
        c = ctx()
        before = json.dumps(c, sort_keys=True, default=str)
        dg.build_report(c)
        self.assertEqual(json.dumps(c, sort_keys=True, default=str), before)
        self.assertEqual(c["settings"]["api_key"], SECRET_KEY)


class ToTextTests(unittest.TestCase):
    def setUp(self):
        dg.clear_failures()
        probe = {"path": "C:\\llama\\llama-server.exe", "exists": True, "status": "ok", "version": "6123", "build": 6123, "commit": "abc1234",
                 "backends": ["CPU"], "backends_usable": [], "supports": {"gpt-oss": True}, "errors": ["something odd"],
                 "devices": [{"id": "CUDA0", "backend": "CUDA", "name": "NVIDIA GeForce GTX 750", "total_mb": 2048, "free_mb": 1900}]}
        self.report = dg.build_report(ctx(engine_probe=probe, logs={"server": ["llama-server starting"]},
                                          telemetry={"ram": {"avail_gb": {"value": 11.0, "status": "measured"}}}))
        self.text = dg.to_text(self.report)

    def test_sections_in_order(self):
        heads = [ln for ln in self.text.splitlines() if ln.startswith("== ")]
        self.assertEqual([h.split(" (")[0] for h in heads][:7], ["== Hardware ==", "== Compatibility ==", "== Engine ==", "== Settings ==",
                                                              "== Selected model ==", "== Memory plan ==", "== Recorded failures"])
        self.assertIn("== Log: server (last 1 lines) ==", self.text)
        self.assertIn("== Telemetry ==", self.text)

    def test_content_is_readable(self):
        t = self.text
        self.assertIn("Intel Core i7-8700K", t)
        self.assertIn("GTX 750", t)
        self.assertIn("verdict=paging", t)
        self.assertIn("Maxwell", t)
        self.assertIn("device CUDA0: NVIDIA GeForce GTX 750 (2048 MiB, 1900 MiB free)", t)
        self.assertIn("probe error: something odd", t)
        self.assertIn("what-if hardware profile: duck-user", t)
        self.assertIn("api_key = ***REDACTED***", t)
        self.assertIn("ctx_size = 4096", t)
        self.assertIn("estimated", t)
        self.assertIn("Note: Secrets", t)
        self.assertIn("C:\\Users\\<user>", t)

    def test_plain_text_only(self):
        self.assertIsInstance(self.text, str)
        self.assertNotIn("\x00", self.text)
        self.assertGreater(len(self.text.splitlines()), 40)

    def test_value_dicts_render_with_their_status(self):
        self.assertEqual(dg._v({"value": 5, "status": "measured"}), "5 [measured]")
        self.assertIn("unavailable (nope)", dg._v({"value": None, "status": "unavailable", "note": "nope"}))
        self.assertEqual(dg._v(3), "3")


if __name__ == "__main__":
    unittest.main()
