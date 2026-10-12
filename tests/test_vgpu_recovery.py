"""Crash classification and the watchdog's restart policy, driven deterministically with a fake server."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckstoken.vgpu import recovery  # noqa: E402
from duckstoken.vgpu.recovery import CrashWatchdog, classify_crash, safer_overrides  # noqa: E402


class FakeServer:
    swappable = True

    def __init__(self):
        self.state, self.error, self.lines, self.started = "stopped", "", [], []
        self.fail_start = False

    def status(self):
        return {"state": self.state, "error": self.error, "running": self.state in ("starting", "loading", "ready")}

    def logs(self, since=0):
        return {"lines": list(self.lines), "next": len(self.lines)}

    def start(self, s):
        if self.fail_start:
            raise RuntimeError("no engine")
        self.started.append(dict(s))
        self.state, self.error = "starting", ""


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Classify(unittest.TestCase):
    def test_vram_oom(self):
        r = classify_crash(["ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1900 MiB on device 0: cudaMalloc failed: out of memory"])
        self.assertEqual(r["kind"], "vram_oom")

    def test_ram_oom_from_exit_code_and_text(self):
        self.assertEqual(classify_crash([], 137)["kind"], "ram_oom")
        self.assertEqual(classify_crash(["terminate called after throwing an instance of 'std::bad_alloc'"])["kind"], "ram_oom")

    def test_ram_oom_not_confused_by_cuda_listing(self):
        lines = ["ggml_cuda_init: found 1 CUDA devices:", "  Device 0: NVIDIA GeForce GTX 750", "llama_model_load: unable to allocate 60000 MiB"]
        self.assertEqual(classify_crash(lines)["kind"], "ram_oom")

    def test_load_time_oom_is_not_blamed_on_the_model(self):
        # real llama.cpp tails: every failed load ends with the generic "failed to load model" wrapper lines
        cuda = ["ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1900 MiB on device 0: cudaMalloc failed: out of memory",
                "alloc_tensor_range: failed to allocate CUDA0 buffer of size 1992294400",
                "llama_model_load: error loading model: unable to allocate CUDA0 buffer",
                "llama_model_load_from_file_impl: failed to load model",
                "srv    load_model: failed to load model, 'gpt-oss-120b.gguf'"]
        self.assertEqual(classify_crash(cuda)["kind"], "vram_oom")
        cpu = ["llama_model_load: error loading model: unable to allocate CPU buffer",
               "llama_model_load_from_file_impl: failed to load model"]
        self.assertEqual(classify_crash(cpu)["kind"], "ram_oom")

    def test_generic_load_failure_without_a_cause_is_unknown_not_bad_model(self):
        self.assertEqual(classify_crash(["llama_model_load_from_file_impl: failed to load model"])["kind"], "unknown")

    def test_bad_model_and_unknown(self):
        self.assertEqual(classify_crash(["llama_model_load: error loading model: unknown model architecture: 'foo'"])["kind"], "bad_model")
        self.assertEqual(classify_crash(["gguf_init_from_file: invalid magic characters"])["kind"], "bad_model")
        self.assertEqual(classify_crash(["hello"])["kind"], "unknown")


class SaferOverrides(unittest.TestCase):
    def test_first_attempt_unchanged_when_cause_unknown(self):
        self.assertEqual(safer_overrides(1, "unknown", {"ctx_size": 8192, "n_gpu_layers": 99}), {})

    def test_vram_oom_drops_gpu_layers_and_kv_offload(self):
        o = safer_overrides(1, "vram_oom", {"ctx_size": 8192, "n_gpu_layers": 20})
        self.assertEqual(o["n_gpu_layers"], 10)
        self.assertFalse(o["kv_offload"])
        self.assertEqual(safer_overrides(3, "vram_oom", {"n_gpu_layers": 20})["n_gpu_layers"], 0)

    def test_ram_oom_halves_context_with_floor(self):
        self.assertEqual(safer_overrides(1, "ram_oom", {"ctx_size": 8192})["ctx_size"], 4096)
        self.assertEqual(safer_overrides(1, "ram_oom", {"ctx_size": 3000})["ctx_size"], 2048)

    def test_bad_model_never_changes_anything(self):
        self.assertEqual(safer_overrides(2, "bad_model", {"ctx_size": 8192}), {})


class Watchdog(unittest.TestCase):
    def make(self, **settings):
        self.srv, self.clock = FakeServer(), Clock()
        self.settings = {"server_auto_restart": True, "server_restart_max": 2, **settings}
        self.events = []
        self.wd = CrashWatchdog(self.srv, lambda: self.settings, on_event=self.events.append, clock=self.clock)
        self.base = {"ctx_size": 8192, "n_gpu_layers": 10, "model_path": "m.gguf"}
        self.wd.note_start(self.base)
        self.srv.state = "ready"
        self.wd.tick()

    def crash(self, *lines, error="llama-server exited with code 1."):
        self.srv.state, self.srv.error, self.srv.lines = "error", error, list(lines)
        return self.wd.tick()

    def test_restarts_with_safer_settings_and_keeps_saved_settings(self):
        self.make()
        ev = self.crash("cudaMalloc failed: out of memory on CUDA device")
        self.assertEqual(ev["action"], "restarted")
        self.assertEqual(self.srv.started[-1]["n_gpu_layers"], 5)
        self.assertEqual(self.base["n_gpu_layers"], 10)  # the caller's dict is untouched

    def test_user_stop_is_never_restarted(self):
        self.make()
        self.wd.note_stop()
        self.srv.state = "stopped"
        self.assertIsNone(self.wd.tick())
        self.assertEqual(self.srv.started, [])

    def test_disabled_setting_only_reports(self):
        self.make(server_auto_restart=False)
        ev = self.crash("boom")
        self.assertEqual(ev["action"], "none")
        self.assertEqual(self.srv.started, [])

    def test_gives_up_after_limit(self):
        self.make(server_restart_max=2)
        for _ in range(2):
            self.assertEqual(self.crash("std::bad_alloc")["action"], "restarted")
            self.srv.state = "ready"
            self.wd.tick()
        ev = self.crash("std::bad_alloc")
        self.assertEqual(ev["action"], "gave_up")
        self.assertEqual(len(self.srv.started), 2)

    def test_stable_run_resets_the_counter(self):
        self.make()
        self.crash("std::bad_alloc")
        self.srv.state = "ready"
        self.wd.tick()
        self.clock.t += recovery.STABLE_SECONDS + 1
        self.wd.tick()
        self.assertEqual(self.wd.restarts, 0)

    def test_bad_model_gives_up_immediately(self):
        self.make()
        ev = self.crash("error loading model: unknown model architecture: 'x'")
        self.assertEqual(ev["action"], "gave_up")
        self.assertEqual(self.srv.started, [])

    def test_failed_restart_is_reported_not_raised(self):
        self.make()
        self.srv.fail_start = True
        ev = self.crash("std::bad_alloc")
        self.assertEqual(ev["action"], "restart_failed")

    def test_thread_starts_and_stops(self):
        self.make()
        self.wd.poll_s = 0.01
        self.wd.start()
        self.wd.start()  # idempotent
        self.wd.stop()
        self.assertFalse(self.wd._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
