import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from duckstoken.vgpu import probe

# Captured from the real binary built from llama.cpp commit 08ffba2 (Linux, CPU only): --version prints to stderr.
REAL_VERSION = "version: 0.6.0-dev (build 1, commit 08ffba2)\nbuilt with GNU 13.3.0 for Linux x86_64\n"
REAL_DEVICES = "Available devices:\n  (none)\n"
# Format from common/arg.cpp print_device_list: "  %s: %s (%zu MiB, %zu MiB free)"
CUDA_DEVICES = ("Available devices:\n  CUDA0: NVIDIA GeForce RTX 3060 (12288 MiB, 11034 MiB free)\n"
                "  CUDA1: NVIDIA GeForce GTX 750 (2048 MiB, 1900 MiB free)\n")
VULKAN_DEVICES = "Available devices:\n  Vulkan0: Intel(R) UHD Graphics 630 (8192 MiB, 7000 MiB free)\n"
OLD_VERSION = "version: 6123 (abc1234)\nbuilt with MSVC 19.44.35213.0 for x64\n"


class Result:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


class FakeRunner:
    """Stands in for subprocess.run: maps the flag to a canned result, records the calls."""

    def __init__(self, version=None, devices=None, raises=None):
        self.version = version if version is not None else Result(stderr=REAL_VERSION)
        self.devices = devices if devices is not None else Result(stdout=REAL_DEVICES)
        self.raises, self.calls = raises or {}, []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        flag = cmd[1]
        if flag in self.raises:
            raise self.raises[flag]
        return self.version if flag == "--version" else self.devices


class ProbeBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.exe = self.dir / "llama-server"
        self.exe.write_bytes(b"\x7fELF stub")
        probe._scan_cache.clear()
        probe._probe_cache.clear()

    def tearDown(self):
        self._tmp.cleanup()


class ParseTests(unittest.TestCase):
    def test_new_version_format(self):
        v = probe.parse_version(REAL_VERSION)
        self.assertEqual((v["version"], v["build"], v["commit"]), ("0.6.0-dev", 1, "08ffba2"))
        self.assertEqual((v["compiler"], v["target"]), ("GNU 13.3.0", "Linux x86_64"))

    def test_old_version_format(self):
        v = probe.parse_version(OLD_VERSION)
        self.assertEqual((v["version"], v["build"], v["commit"]), ("6123", 6123, "abc1234"))
        self.assertEqual(v["target"], "x64")

    def test_garbage_version(self):
        v = probe.parse_version("segmentation fault")
        self.assertTrue(all(x is None for x in v.values()))
        self.assertIsNone(probe.parse_version(None)["version"])

    def test_devices(self):
        d = probe.parse_devices(CUDA_DEVICES)
        self.assertEqual([x["id"] for x in d], ["CUDA0", "CUDA1"])
        self.assertEqual(d[0], {"id": "CUDA0", "backend": "CUDA", "name": "NVIDIA GeForce RTX 3060", "total_mb": 12288,
                                "free_mb": 11034})
        self.assertEqual(probe.parse_devices(VULKAN_DEVICES)[0]["backend"], "Vulkan")
        self.assertEqual(probe.parse_devices(REAL_DEVICES), [])
        self.assertEqual(probe.parse_devices("junk\n  broken: line"), [])
        self.assertEqual(probe.parse_devices(None), [])

    def test_name_with_parentheses_and_colon(self):
        d = probe.parse_devices("  CUDA0: NVIDIA RTX A2000 (Laptop GPU) (4096 MiB, 3000 MiB free)")
        self.assertEqual(d[0]["name"], "NVIDIA RTX A2000 (Laptop GPU)")
        self.assertEqual(d[0]["total_mb"], 4096)

    def test_backend_of(self):
        for dev, backend in (("CUDA0", "CUDA"), ("ROCm1", "HIP"), ("HIP0", "HIP"), ("Vulkan0", "Vulkan"), ("SYCL0", "SYCL"),
                             ("MTL0", "Metal"), ("GPUOpenCL", "OpenCL"), ("RPC[host:50052]", "RPC"), ("CPU", "CPU"),
                             ("Weird9", "Weird"), ("", "unknown")):
            self.assertEqual(probe.backend_of(dev), backend, dev)


class ProbeEngineTests(ProbeBase):
    def test_real_cpu_build_output(self):
        r = probe.probe_engine(self.exe, FakeRunner())
        self.assertTrue(r["exists"])
        self.assertEqual(r["status"], "ok")
        self.assertEqual((r["version"], r["build"], r["commit"]), ("0.6.0-dev", 1, "08ffba2"))
        self.assertEqual(r["backends"], ["CPU"])
        self.assertEqual(r["backends_usable"], [])
        self.assertEqual(r["devices"], [])
        self.assertEqual(r["errors"], [])
        self.assertIn("version: 0.6.0-dev", r["raw_version"])
        self.assertIn("(none)", r["raw_devices"])

    def test_calls_use_list_argv_and_no_shell(self):
        run = FakeRunner()
        probe.probe_engine(self.exe, run, timeout=7)
        self.assertEqual([c[0][1] for c in run.calls], ["--version", "--list-devices"])
        for cmd, kw in run.calls:
            self.assertIsInstance(cmd, list)
            self.assertNotIn("shell", kw)
            self.assertEqual(kw["timeout"], 7)
            self.assertEqual(kw["cwd"], str(self.dir.resolve()))
        if os.name == "nt":
            self.assertEqual(run.calls[0][1]["creationflags"], 0x08000000)

    def test_cuda_devices_make_cuda_usable(self):
        r = probe.probe_engine(self.exe, FakeRunner(devices=Result(stdout=CUDA_DEVICES)))
        self.assertEqual(r["backends_usable"], ["CUDA"])
        self.assertEqual(r["backends"], ["CPU", "CUDA"])
        self.assertEqual(len(r["devices"]), 2)

    def test_compiled_but_unusable_backend_is_not_usable(self):
        (self.dir / "ggml-cuda.dll").write_bytes(b"x")
        (self.dir / "ggml-vulkan.dll").write_bytes(b"x")
        r = probe.probe_engine(self.exe, FakeRunner())
        self.assertEqual(r["backends"], ["CPU", "CUDA", "Vulkan"])
        self.assertEqual(r["backends_usable"], [])  # a CUDA build on a machine whose driver cannot run it lists no device

    def test_devices_reported_on_stderr_are_found(self):
        r = probe.probe_engine(self.exe, FakeRunner(devices=Result(stderr=VULKAN_DEVICES)))
        self.assertEqual(r["backends_usable"], ["Vulkan"])

    def test_old_build(self):
        r = probe.probe_engine(self.exe, FakeRunner(version=Result(stdout=OLD_VERSION)))
        self.assertEqual((r["version"], r["build"]), ("6123", 6123))

    def test_missing_executable(self):
        run = FakeRunner()
        r = probe.probe_engine(self.dir / "nope.exe", run)
        self.assertFalse(r["exists"])
        self.assertEqual(r["status"], "missing")
        self.assertTrue(r["errors"])
        self.assertEqual(run.calls, [])  # nothing is executed
        self.assertEqual(probe.probe_engine("", run)["status"], "missing")
        self.assertEqual(probe.probe_engine(None, run)["status"], "missing")

    def test_timeout_and_oserror_are_reported_not_raised(self):
        r = probe.probe_engine(self.exe, FakeRunner(raises={"--version": subprocess.TimeoutExpired("x", 1)}))
        self.assertTrue(any("timed out" in e for e in r["errors"]))
        self.assertIsNone(r["version"])
        self.assertEqual(r["status"], "partial")
        r = probe.probe_engine(self.exe, FakeRunner(raises={"--version": OSError("bad exe"), "--list-devices": OSError("bad exe")}))
        self.assertEqual(r["status"], "error")
        self.assertEqual(len(r["errors"]), 2)

    def test_unrecognised_output(self):
        r = probe.probe_engine(self.exe, FakeRunner(version=Result(stdout="hello", returncode=3),
                                                    devices=Result(stderr="CUDA error: no driver", returncode=1)))
        self.assertEqual(r["status"], "error")
        self.assertTrue(any("--version" in e for e in r["errors"]))
        self.assertTrue(any("--list-devices" in e for e in r["errors"]))
        self.assertIsNone(r["version"])

    def test_bytes_output_is_decoded(self):
        r = probe.probe_engine(self.exe, FakeRunner(version=Result(stderr=REAL_VERSION.encode()),
                                                    devices=Result(stdout=CUDA_DEVICES.encode())))
        self.assertEqual(r["version"], "0.6.0-dev")
        self.assertEqual(len(r["devices"]), 2)

    def test_supports_come_from_the_binaries(self):
        (self.dir / "libllama.so").write_bytes(b"\0gpt-oss\0qwen3moe\0")
        r = probe.probe_engine(self.exe, FakeRunner(), archs=("gpt-oss", "qwen3moe", "mamba"))
        self.assertEqual(r["supports"], {"gpt-oss": True, "qwen3moe": True, "mamba": False})

    def test_cached_probe_runs_once(self):
        run = FakeRunner()
        a = probe.probe_engine_cached(self.exe, runner=run)
        b = probe.probe_engine_cached(self.exe, runner=run)
        self.assertIs(a, b)
        self.assertEqual(len(run.calls), 2)  # one probe = two commands
        self.exe.write_bytes(b"\x7fELF a different, larger stub")
        probe.probe_engine_cached(self.exe, runner=run)
        self.assertEqual(len(run.calls), 4)

    def test_cached_probe_expires(self):
        run = FakeRunner()
        probe.probe_engine_cached(self.exe, ttl=0.0, runner=run)
        probe.probe_engine_cached(self.exe, ttl=0.0, runner=run)
        self.assertEqual(len(run.calls), 4)

    def test_json_serialisable(self):
        import json
        json.dumps(probe.probe_engine(self.exe, FakeRunner(devices=Result(stdout=CUDA_DEVICES))))


class ScanArchTests(ProbeBase):
    def test_finds_names_in_exe_and_llama_library(self):
        self.exe.write_bytes(b"xx\0llama\0yy")
        (self.dir / "libllama.so.0").write_bytes(b"\0gpt-oss\0")
        (self.dir / "llama.dll").write_bytes(b"\0qwen3moe\0")
        r = probe.scan_arch_support(self.exe, ("llama", "gpt-oss", "qwen3moe", "mamba"))
        self.assertEqual(r, {"llama": True, "gpt-oss": True, "qwen3moe": True, "mamba": False})

    def test_name_split_across_chunk_boundary_is_found(self):
        old = probe._CHUNK
        probe._CHUNK = 16
        try:
            for offset in range(9):
                probe._scan_cache.clear()
                (self.dir / "libllama.so").write_bytes(b"A" * (10 + offset) + b"gpt-oss\0" + b"B" * 40)
                self.assertTrue(probe.scan_arch_support(self.exe, ("gpt-oss",))["gpt-oss"], offset)
        finally:
            probe._CHUNK = old

    def test_nul_terminator_avoids_prefix_false_positives(self):
        (self.dir / "libllama.so").write_bytes(b"\0gpt-oss-extended-thing\0")
        self.assertFalse(probe.scan_arch_support(self.exe, ("gpt-oss",))["gpt-oss"])

    def test_unrelated_libraries_are_not_scanned(self):
        (self.dir / "libmtmd.so").write_bytes(b"\0gemma3\0")  # mtmd names projector types, not architectures
        (self.dir / "ggml-cuda.dll").write_bytes(b"\0gemma3\0")
        self.assertFalse(probe.scan_arch_support(self.exe, ("gemma3",))["gemma3"])

    def test_unreadable_gives_none_not_false(self):
        r = probe.scan_arch_support(self.dir / "missing" / "llama-server", ("gpt-oss",))
        self.assertEqual(r, {"gpt-oss": None})
        self.exe.unlink()
        self.assertEqual(probe.scan_arch_support(self.exe, ("gpt-oss",)), {"gpt-oss": None})

    def test_cache_by_path_mtime_size(self):
        (self.dir / "libllama.so").write_bytes(b"\0gpt-oss\0")
        self.assertTrue(probe.scan_arch_support(self.exe, ("gpt-oss",))["gpt-oss"])
        self.assertEqual(len(probe._scan_cache), 1)
        calls = []
        orig = probe._scan_file
        probe._scan_file = lambda *a, **k: calls.append(1) or orig(*a, **k)
        try:
            self.assertTrue(probe.scan_arch_support(self.exe, ("gpt-oss",))["gpt-oss"])
            self.assertEqual(calls, [])  # served from cache
            (self.dir / "libllama.so").write_bytes(b"\0nothing here at all\0")  # replaced engine
            os.utime(self.dir / "libllama.so", (time.time() + 10, time.time() + 10))
            self.assertFalse(probe.scan_arch_support(self.exe, ("gpt-oss",))["gpt-oss"])
            self.assertTrue(calls)
        finally:
            probe._scan_file = orig

    def test_new_names_are_scanned_on_demand(self):
        (self.dir / "libllama.so").write_bytes(b"\0gpt-oss\0phi3\0")
        self.assertEqual(probe.scan_arch_support(self.exe, ("gpt-oss",)), {"gpt-oss": True})
        self.assertEqual(probe.scan_arch_support(self.exe, ("gpt-oss", "phi3", "mamba")), {"gpt-oss": True, "phi3": True, "mamba": False})

    def test_symlinked_duplicates_are_scanned_once(self):
        real = self.dir / "libllama.so.0.6.0"
        real.write_bytes(b"\0gpt-oss\0")
        try:
            (self.dir / "libllama.so").symlink_to(real)
            (self.dir / "libllama.so.0").symlink_to(real)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        files = probe._candidate_files(self.exe)
        self.assertEqual(len([f for f in files if f.name == "libllama.so.0.6.0"]), 1)
        self.assertEqual(len(files), 2)  # the stub exe + the one real library


@unittest.skipUnless(os.environ.get("DUCK_REAL_LLAMA"), "set DUCK_REAL_LLAMA=/path/to/llama-server for the real-engine probe")
class RealEngineProbeTest(unittest.TestCase):
    def test_real_binary(self):
        r = probe.probe_engine(os.environ["DUCK_REAL_LLAMA"])
        self.assertTrue(r["exists"])
        self.assertEqual(r["status"], "ok", r["errors"])
        self.assertIsNotNone(r["version"])
        self.assertIsNotNone(r["build"])
        self.assertIn("CPU", r["backends"])
        self.assertTrue(r["supports"]["llama"])


if __name__ == "__main__":
    unittest.main()
