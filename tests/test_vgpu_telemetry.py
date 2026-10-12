import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from duckstoken.vgpu import telemetry as t
from duckstoken.vgpu.values import is_available

MEMINFO = """MemTotal:       49283072 kB
MemFree:         1234567 kB
MemAvailable:   12320768 kB
SwapTotal:       8388608 kB
SwapFree:        6291456 kB
CommitLimit:    33000000 kB
Committed_AS:   16500000 kB
HugePages_Total:       0
"""
PROC_STAT_1 = "cpu  1000 0 500 8000 500 0 0 0 0 0\ncpu0 500 0 250 4000 250 0 0 0 0 0\ncpu1 500 0 250 4000 250 0 0 0 0 0\nintr 1 2 3\n"
PROC_STAT_2 = "cpu  1600 0 700 8200 500 0 0 0 0 0\ncpu0 900 0 350 4100 250 0 0 0 0 0\ncpu1 700 0 350 4100 250 0 0 0 0 0\n"
PSI = "some avg10=12.50 avg60=3.00 avg300=1.00 total=123456\nfull avg10=2.00 avg60=0.50 avg300=0.10 total=1234\n"
PID_STAT = "4242 (llama server (x)) S 1 4242 4242 0 -1 4194560 120000 0 3500 0 900 300 0 0 20 0 17 0 100 1 1 18446744073709551615 0 0 0"
PID_STATUS = "Name:\tllama-server\nVmHWM:\t 2048000 kB\nVmRSS:\t 1024000 kB\nThreads:\t17\n"
PID_IO = "rchar: 1\nwchar: 2\nsyscr: 3\nsyscw: 4\nread_bytes: 1073741824\nwrite_bytes: 2147483648\ncancelled_write_bytes: 0\n"
SMI_FULL = ("0, NVIDIA GeForce GTX 750, 3, 1, 612, 2048, 45, [N/A], 1020, 3, 16\n"
            "1, NVIDIA GeForce RTX 3060, 99, 55, 8123, 12288, 71, 150.25, 1777, 4, 16\n")


class Result:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


class ParserTests(unittest.TestCase):
    def test_meminfo(self):
        mi = t.parse_meminfo(MEMINFO)
        self.assertEqual(mi["MemTotal"], 49283072 * 1024)
        self.assertEqual(mi["HugePages_Total"], 0)
        self.assertNotIn("Nope", mi)

    def test_proc_stat(self):
        st = t.parse_proc_stat(PROC_STAT_1)
        self.assertEqual((st["total"], st["idle"]), (10000, 8500))
        self.assertEqual(len(st["cores"]), 2)
        with self.assertRaises(ValueError):
            t.parse_proc_stat("garbage")

    def test_psi(self):
        self.assertEqual(t.parse_psi(PSI), 12.5)
        self.assertIsNone(t.parse_psi("nothing"))

    def test_pid_stat_with_spaces_and_parens_in_the_command_name(self):
        ps = t.parse_pid_stat(PID_STAT)
        self.assertEqual((ps["minflt"], ps["majflt"], ps["utime"], ps["stime"], ps["threads"]), (120000, 3500, 900, 300, 17))

    def test_pid_status_and_io(self):
        st = t.parse_pid_status(PID_STATUS)
        self.assertEqual((st["VmRSS"], st["VmHWM"], st["Threads"]), (1024000 * 1024, 2048000 * 1024, 17))
        self.assertEqual(t.parse_pid_io(PID_IO)["read_bytes"], 1073741824)

    def test_nvidia_smi_full_row(self):
        gpus = t.parse_nvidia_smi(SMI_FULL)
        self.assertEqual(len(gpus), 2)
        g = gpus[0]
        self.assertEqual(g["index"], 0)
        self.assertEqual(g["name"]["value"], "NVIDIA GeForce GTX 750")
        self.assertEqual(g["util_pct"]["value"], 3.0)
        self.assertEqual((g["vram_used_mb"]["value"], g["vram_total_mb"]["value"]), (612.0, 2048.0))
        self.assertEqual(g["temp_c"]["value"], 45.0)
        self.assertEqual((g["pcie_gen"]["value"], g["pcie_width"]["value"]), (3, 16))
        self.assertEqual(gpus[1]["power_w"]["value"], 150.2)  # rounded to one decimal

    def test_not_available_markers_are_unavailable_never_zero(self):
        g = t.parse_nvidia_smi(SMI_FULL)[0]
        self.assertEqual(g["power_w"]["status"], "unavailable")
        self.assertIsNone(g["power_w"]["value"])
        self.assertIn("N/A", g["power_w"]["note"])
        row = "0, Some GPU, [Not Supported], [N/A], [Not Supported], 2048, [Unknown Error], N/A, , [N/A], [N/A]"
        g = t.parse_nvidia_smi(row)[0]
        for k in ("util_pct", "mem_util_pct", "vram_used_mb", "temp_c", "power_w", "sm_clock_mhz", "pcie_gen", "pcie_width"):
            self.assertEqual(g[k]["status"], "unavailable", k)
            self.assertIsNone(g[k]["value"], k)
        self.assertEqual(g["vram_total_mb"]["value"], 2048.0)

    def test_garbage_rows_do_not_raise(self):
        self.assertEqual(t.parse_nvidia_smi(""), [])
        self.assertEqual(t.parse_nvidia_smi("just one cell"), [])
        g = t.parse_nvidia_smi("x, GPU, abc, 1, 2, 3, 4, 5, 6, 7, 8")[0]
        self.assertEqual(g["util_pct"]["status"], "unavailable")
        self.assertEqual(g["index"], 0)

    def test_old_nvidia_smi_with_fewer_fields(self):
        g = t.parse_nvidia_smi("0, GTX, 5, 2, 100, 2048, 40", t.GPU_FIELDS[:7])[0]
        self.assertEqual(g["temp_c"]["value"], 40.0)
        self.assertEqual(g["pcie_gen"]["status"], "unavailable")
        self.assertIn("not requested", g["pcie_gen"]["note"])


class GpuSampleTests(unittest.TestCase):
    def setUp(self):
        self._find = t._find_nvidia_smi
        t._find_nvidia_smi = lambda: "/usr/bin/nvidia-smi"

    def tearDown(self):
        t._find_nvidia_smi = self._find

    def test_ok(self):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            return Result(SMI_FULL)

        gpus, src = t.gpu_sample(run)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(src["status"], "measured")
        self.assertIsInstance(calls[0], list)
        self.assertIn("--format=csv,noheader,nounits", calls[0])
        self.assertTrue(calls[0][1].startswith("--query-gpu=index,name,utilization.gpu"))

    def test_absent_nvidia_smi(self):
        t._find_nvidia_smi = lambda: None
        gpus, src = t.gpu_sample(lambda *a, **k: self.fail("must not run"))
        self.assertEqual(gpus, [])
        self.assertEqual(src["status"], "unavailable")
        self.assertEqual(src["note"], "nvidia-smi not found")

    def test_falls_back_to_core_fields(self):
        seen = []

        def run(cmd, **kw):
            seen.append(cmd[1])
            if len(seen) == 1:
                return Result("", "Field \"pcie.link.gen.current\" is not a valid field to query.", 2)
            return Result("0, GTX 750, 3, 1, 612, 2048, 45\n")

        gpus, src = t.gpu_sample(run)
        self.assertEqual(len(seen), 2)
        self.assertEqual(gpus[0]["temp_c"]["value"], 45.0)
        self.assertEqual(gpus[0]["pcie_gen"]["status"], "unavailable")
        self.assertEqual(src["status"], "measured")

    def test_failures_are_unavailable_not_raised(self):
        for run, text in ((lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("x", 1)), "timed out"),
                          (lambda *a, **k: (_ for _ in ()).throw(OSError("boom")), "could not run"),
                          (lambda *a, **k: Result("", "NVIDIA-SMI has failed because it couldn't communicate with the driver", 9), "no usable data"),
                          (lambda *a, **k: Result("", "", 0), "no usable data")):
            gpus, src = t.gpu_sample(run)
            self.assertEqual(gpus, [])
            self.assertEqual(src["status"], "unavailable")
            self.assertIn(text, src["note"])


@unittest.skipIf(os.name == "nt", "uses /proc")
class SystemSampleTests(unittest.TestCase):
    def setUp(self):
        t._reset_state()
        self._read = t._read_text
        self.files = {"/proc/meminfo": MEMINFO, "/proc/stat": PROC_STAT_1, "/proc/pressure/memory": PSI}
        t._read_text = lambda p: self.files[p] if p in self.files else (_ for _ in ()).throw(FileNotFoundError(p))
        self._find = t._find_nvidia_smi
        t._find_nvidia_smi = lambda: None

    def tearDown(self):
        t._read_text = self._read
        t._find_nvidia_smi = self._find

    def test_shape_and_values(self):
        s = t.system_sample(gpu=False)
        self.assertEqual({"t", "ram", "swap", "commit", "cpu", "gpus", "sources"}, set(s))
        self.assertEqual(set(s["ram"]), {"total_gb", "avail_gb", "used_pct", "pressure_pct"})
        self.assertAlmostEqual(s["ram"]["total_gb"]["value"], 47.0, delta=0.01)
        self.assertAlmostEqual(s["ram"]["avail_gb"]["value"], 11.75, delta=0.01)
        self.assertAlmostEqual(s["ram"]["used_pct"]["value"], 75.0, delta=0.1)
        self.assertEqual(s["ram"]["pressure_pct"]["value"], 12.5)
        self.assertAlmostEqual(s["swap"]["used_gb"]["value"], 2.0, delta=0.01)
        self.assertAlmostEqual(s["swap"]["used_pct"]["value"], 25.0, delta=0.1)
        self.assertAlmostEqual(s["commit"]["used_pct"]["value"], 50.0, delta=0.1)
        for tier in ("ram", "swap", "commit", "cpu"):
            for k, v in s[tier].items():
                self.assertIn("status", v, (tier, k))
        self.assertEqual(s["sources"]["ram"]["status"], "measured")

    def test_cpu_needs_two_samples(self):
        first = t.system_sample(gpu=False)
        self.assertEqual(first["cpu"]["util_pct"]["status"], "unavailable")
        self.assertIsNone(first["cpu"]["util_pct"]["value"])
        self.assertIn("first sample", first["cpu"]["util_pct"]["note"])
        self.files["/proc/stat"] = PROC_STAT_2
        second = t.system_sample(gpu=False)
        # 1000 jiffies elapsed (10000 -> 11000), 200 of them idle (8500 -> 8700): 80% busy
        self.assertAlmostEqual(second["cpu"]["util_pct"]["value"], 80.0, delta=0.1)
        self.assertEqual(second["cpu"]["util_pct"]["status"], "measured")
        self.assertEqual(second["cpu"]["per_core_pct"]["value"], [83.3, 75.0])

    def test_no_time_elapsed_is_unavailable(self):
        t.system_sample(gpu=False)
        again = t.system_sample(gpu=False)
        self.assertEqual(again["cpu"]["util_pct"]["status"], "unavailable")

    def test_missing_psi_and_swap(self):
        del self.files["/proc/pressure/memory"]
        self.files["/proc/meminfo"] = "MemTotal: 1048576 kB\nMemAvailable: 524288 kB\n"
        s = t.system_sample(gpu=False)
        self.assertEqual(s["ram"]["pressure_pct"]["status"], "unavailable")
        self.assertEqual(s["swap"]["used_gb"]["status"], "unavailable")
        self.assertEqual(s["commit"]["used_pct"]["status"], "unavailable")
        self.assertEqual(s["ram"]["used_pct"]["value"], 50.0)

    def test_unreadable_proc_never_raises(self):
        self.files.clear()
        s = t.system_sample(gpu=False)
        self.assertEqual(s["ram"]["total_gb"]["status"], "unavailable")
        self.assertEqual(s["cpu"]["util_pct"]["status"], "unavailable")
        self.assertEqual(s["sources"]["ram"]["status"], "unavailable")
        self.assertFalse(is_available(s["swap"]["used_gb"]))

    def test_corrupt_files_never_raise(self):
        self.files.update({"/proc/meminfo": "\x00garbage", "/proc/stat": "cpu x y z"})
        s = t.system_sample(gpu=False)
        self.assertEqual(s["ram"]["total_gb"]["status"], "unavailable")
        self.assertEqual(s["cpu"]["util_pct"]["status"], "unavailable")

    def test_no_nvidia_smi_reports_the_reason(self):
        s = t.system_sample()
        self.assertEqual(s["gpus"], [])
        self.assertEqual(s["sources"]["gpu"]["status"], "unavailable")
        self.assertEqual(s["sources"]["gpu"]["note"], "nvidia-smi not found")
        self.assertEqual(s["sources"]["gpu_other"]["status"], "unavailable")

    def test_gpu_rows_flow_into_the_sample(self):
        t._find_nvidia_smi = lambda: "nvidia-smi"
        s = t.system_sample(lambda *a, **k: Result(SMI_FULL))
        self.assertEqual(len(s["gpus"]), 2)
        self.assertEqual(s["gpus"][1]["vram_total_mb"]["value"], 12288.0)

    def test_a_crashing_runner_cannot_break_the_sample(self):
        t._find_nvidia_smi = lambda: "nvidia-smi"
        s = t.system_sample(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("weird")))
        self.assertEqual(s["gpus"], [])
        self.assertEqual(s["sources"]["gpu"]["status"], "unavailable")


class WindowsBranchTests(unittest.TestCase):
    """The Win32 calls cannot run here; forcing the branch proves failures degrade to 'unavailable', not exceptions."""

    @unittest.skipIf(os.name == "nt", "on a real Windows machine windll exists; see RealWindowsTests")
    def test_windows_branch_without_windll_never_raises(self):
        old = t._IS_WIN
        t._IS_WIN = True
        try:
            t._reset_state()
            s = t.system_sample(gpu=False)
            self.assertEqual(s["ram"]["total_gb"]["status"], "unavailable")
            self.assertEqual(s["cpu"]["util_pct"]["status"], "unavailable")
            p = t.process_sample(os.getpid())
            self.assertEqual(set(p), set(t._PROC_KEYS))
            self.assertTrue(all(v["status"] == "unavailable" for v in p.values()))
        finally:
            t._IS_WIN = old


class ProcessSampleTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "uses /proc")
    def test_own_process(self):
        p = t.process_sample(os.getpid())
        self.assertEqual(set(p), set(t._PROC_KEYS))
        self.assertGreater(p["rss_gb"]["value"], 0)
        self.assertGreaterEqual(p["peak_rss_gb"]["value"], p["rss_gb"]["value"])
        self.assertGreater(p["page_faults"]["value"], 0)
        self.assertGreaterEqual(p["page_faults"]["value"], p["major_page_faults"]["value"])
        self.assertGreaterEqual(p["threads"]["value"], 1)
        self.assertEqual(p["rss_gb"]["status"], "measured")

    @unittest.skipIf(os.name == "nt", "uses /proc")
    def test_canned_proc_files(self):
        files = {"/proc/4242/status": PID_STATUS, "/proc/4242/stat": PID_STAT, "/proc/4242/io": PID_IO}
        old = t._read_text
        t._read_text = lambda p: files[p]
        try:
            p = t.process_sample(4242)
        finally:
            t._read_text = old
        self.assertAlmostEqual(p["rss_gb"]["value"], 1024000 * 1024 / 2**30, places=3)
        self.assertEqual(p["page_faults"]["value"], 123500)
        self.assertEqual(p["major_page_faults"]["value"], 3500)
        self.assertEqual(p["threads"]["value"], 17)
        self.assertEqual(p["io_read_gb"]["value"], 1.0)
        self.assertEqual(p["io_write_gb"]["value"], 2.0)
        self.assertGreater(p["cpu_seconds"]["value"], 0)

    @unittest.skipIf(os.name == "nt", "uses /proc")
    def test_unreadable_io_is_unavailable_but_the_rest_survives(self):
        files = {"/proc/4242/status": PID_STATUS, "/proc/4242/stat": PID_STAT}

        def read(p):
            if p in files:
                return files[p]
            raise PermissionError(p)

        old = t._read_text
        t._read_text = read
        try:
            p = t.process_sample(4242)
        finally:
            t._read_text = old
        self.assertEqual(p["io_read_gb"]["status"], "unavailable")
        self.assertIn("not readable", p["io_read_gb"]["note"])
        self.assertEqual(p["rss_gb"]["status"], "measured")

    def test_missing_or_invalid_pid(self):
        for pid in (99999999, 0, -3, None, "abc"):
            p = t.process_sample(pid)
            self.assertEqual(set(p), set(t._PROC_KEYS))
            self.assertTrue(all(v["status"] == "unavailable" and v["value"] is None for v in p.values()), pid)

    def test_child_process_is_measured_then_gone(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            time.sleep(0.2)
            if os.name != "nt":
                self.assertEqual(t.process_sample(child.pid)["rss_gb"]["status"], "measured")
        finally:
            child.kill()
            child.wait()
        self.assertEqual(t.process_sample(child.pid)["rss_gb"]["status"], "unavailable")


@unittest.skipUnless(os.name == "nt", "calls the real Win32 API")
class RealWindowsTests(unittest.TestCase):
    """Runs only on Windows (CI's windows-latest): the ctypes struct layouts and signatures are exercised for real."""

    def test_memory_and_process_counters_are_measured(self):
        t._reset_state()
        s = t.system_sample(gpu=False)
        self.assertEqual(s["ram"]["total_gb"]["status"], "measured")
        self.assertGreater(s["ram"]["total_gb"]["value"], 1)
        self.assertEqual(s["ram"]["avail_gb"]["status"], "measured")
        self.assertLessEqual(s["ram"]["avail_gb"]["value"], s["ram"]["total_gb"]["value"])
        p = t.process_sample(os.getpid())
        self.assertEqual(p["rss_gb"]["status"], "measured")
        self.assertGreater(p["rss_gb"]["value"], 0)
        self.assertGreaterEqual(p["peak_rss_gb"]["value"], p["rss_gb"]["value"])
        self.assertEqual(p["cpu_seconds"]["status"], "measured")
        self.assertEqual(p["io_read_gb"]["status"], "measured")

    def test_cpu_utilisation_needs_two_samples_then_is_measured(self):
        t._reset_state()
        first = t.system_sample(gpu=False)
        self.assertEqual(first["cpu"]["util_pct"]["status"], "unavailable")   # one reading is not a rate
        x = 0
        for i in range(2_000_000):
            x += i
        second = t.system_sample(gpu=False)
        self.assertEqual(second["cpu"]["util_pct"]["status"], "measured")
        self.assertTrue(0 <= second["cpu"]["util_pct"]["value"] <= 100)


class BandwidthTests(unittest.TestCase):
    def test_measured_and_labelled_as_a_lower_bound(self):
        v = t.measure_host_bandwidth(size_mb=32, repeats=2)
        self.assertEqual(v["status"], "measured")
        self.assertGreater(v["value"], 0.2)
        self.assertEqual(v["source"], "bytearray memcpy")
        self.assertIn("lower bound", v["note"])

    def test_bad_arguments_do_not_raise(self):
        self.assertEqual(t.measure_host_bandwidth(size_mb=-5, repeats=0)["status"], "measured")  # clamped to 32 MiB, one repeat
        self.assertEqual(t.measure_host_bandwidth(size_mb="x")["status"], "unavailable")

    def test_allocation_failure_is_unavailable(self):
        orig = t.bytearray if hasattr(t, "bytearray") else None
        t.bytearray = lambda n: (_ for _ in ()).throw(MemoryError())
        try:
            v = t.measure_host_bandwidth(32, 1)
        finally:
            if orig is None:
                del t.bytearray
        self.assertEqual(v["status"], "unavailable")
        self.assertIn("allocate", v["note"])


def fake_sample(i, avail=10.0, cpu=50.0, gpu_util=None, t0=1000.0):
    s = {"t": t0 + i, "ram": {"avail_gb": {"value": avail, "status": "measured"}, "used_pct": {"value": 100 - avail, "status": "measured"}},
         "swap": {"used_gb": {"value": 0.0, "status": "measured"}}, "cpu": {"util_pct": {"value": cpu, "status": "measured"}},
         "gpus": [], "sources": {}}
    if gpu_util is not None:
        s["gpus"] = [{"util_pct": {"value": gpu_util, "status": "measured"}, "vram_used_mb": {"value": 500, "status": "measured"},
                      "temp_c": {"value": 60, "status": "measured"}, "sm_clock_mhz": {"value": 1000, "status": "measured"}}]
    return s


class SamplerTests(unittest.TestCase):
    def test_sample_once_and_ring_buffer_bound(self):
        n = [0]

        def fn():
            n[0] += 1
            return fake_sample(n[0])

        s = t.Sampler(0.05, maxlen=5, sample_fn=fn)
        for _ in range(12):
            s.sample_once()
        self.assertEqual(len(s.history()), 5)
        self.assertEqual(s.latest()["t"], 1012.0)
        self.assertEqual([x["t"] for x in s.history(2)], [1011.0, 1012.0])
        self.assertEqual(s.history(0), [])
        self.assertIsNone(t.Sampler(0.05).latest())

    def test_start_stop_join_and_restart(self):
        s = t.Sampler(0.02, maxlen=50, sample_fn=lambda: fake_sample(0, t0=time.time()))
        self.assertTrue(s.start())
        self.assertFalse(s.start())  # already running
        deadline = time.time() + 3
        while len(s.history()) < 3 and time.time() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(len(s.history()), 3)
        thread = s._thread
        s.stop()
        self.assertFalse(thread.is_alive())
        self.assertFalse(s.running)
        n = len(s.history())
        time.sleep(0.1)
        self.assertEqual(len(s.history()), n)  # really stopped
        s.stop()  # idempotent
        self.assertTrue(s.start())  # restartable
        time.sleep(0.1)
        s.stop()
        self.assertGreater(len(s.history()), n)
        self.assertTrue(thread.daemon)

    def test_repeated_start_stop_leaks_no_threads(self):
        before = threading.active_count()
        s = t.Sampler(0.02, sample_fn=lambda: fake_sample(0))
        for _ in range(5):
            s.start()
            s.stop()
        self.assertLessEqual(threading.active_count(), before)

    def test_a_failing_sample_function_does_not_kill_the_thread(self):
        calls = []

        def fn():
            calls.append(1)
            raise RuntimeError("sensor exploded")

        s = t.Sampler(0.02, sample_fn=fn)
        s.start()
        time.sleep(0.15)
        self.assertTrue(s.running)
        s.stop()
        self.assertGreater(len(calls), 2)
        self.assertGreater(s.errors, 2)
        self.assertTrue(s.history())  # an explicit error record, not silence

    def test_pid_fn_adds_process_samples(self):
        s = t.Sampler(0.05, pid_fn=lambda: 77, sample_fn=lambda: fake_sample(0),
                      process_fn=lambda pid: {"rss_gb": {"value": 3.5, "status": "measured"}, "pid_seen": pid})
        got = s.sample_once()
        self.assertEqual(got["pid"], 77)
        self.assertEqual(got["process"]["rss_gb"]["value"], 3.5)
        none = t.Sampler(0.05, pid_fn=lambda: None, sample_fn=lambda: fake_sample(0)).sample_once()
        self.assertNotIn("process", none)
        boom = t.Sampler(0.05, pid_fn=lambda: (_ for _ in ()).throw(ValueError()), sample_fn=lambda: fake_sample(0))
        self.assertIn("t", boom.sample_once())

    def test_summary_statistics(self):
        s = t.Sampler(0.05, sample_fn=dict)
        for i, (avail, cpu, g) in enumerate([(12.0, 40.0, 10.0), (8.0, 100.0, 50.0), (10.0, 70.0, 30.0)]):
            s._buf.append(fake_sample(i, avail, cpu, g))
        sm = s.summary(window_s=30)
        self.assertEqual(sm["n"], 3)
        self.assertEqual(sm["ram_avail_gb"]["min"]["value"], 8.0)
        self.assertEqual(sm["ram_avail_gb"]["mean"]["value"], 10.0)
        self.assertEqual(sm["cpu_util_pct"]["max"]["value"], 100.0)
        self.assertEqual(sm["cpu_util_pct"]["mean"]["value"], 70.0)
        self.assertEqual(sm["gpu_util_pct"]["mean"]["value"], 30.0)
        self.assertEqual(sm["vram_used_mb"]["max"]["value"], 500.0)
        self.assertEqual(sm["ram_avail_gb"]["min"]["status"], "measured")
        # nothing measured for these: unavailable, not zero
        self.assertEqual(sm["rss_gb"]["max"]["status"], "unavailable")
        self.assertEqual(sm["major_fault_rate_per_s"]["status"], "unavailable")

    def test_summary_window_and_empty_buffer(self):
        s = t.Sampler(0.05, sample_fn=dict)
        empty = s.summary(10)
        self.assertEqual(empty["n"], 0)
        self.assertEqual(empty["cpu_util_pct"]["mean"]["status"], "unavailable")
        for i in range(10):
            s._buf.append(fake_sample(i * 10, avail=float(i)))
        self.assertEqual(s.summary(window_s=25)["n"], 3)  # samples 20, 10 and 0 s before the newest one

    def test_summary_rates_from_process_counters(self):
        s = t.Sampler(0.05, sample_fn=dict)
        for i in range(5):
            x = fake_sample(i)
            x["process"] = {"major_page_faults": {"value": 100 * i, "status": "measured"},
                            "page_faults": {"value": 1000 * i, "status": "measured"},
                            "io_read_gb": {"value": 0.5 * i, "status": "measured"},
                            "rss_gb": {"value": 1.0 + i, "status": "measured"}}
            s._buf.append(x)
        sm = s.summary(30)
        self.assertEqual(sm["major_fault_rate_per_s"]["value"], 100.0)
        self.assertEqual(sm["page_fault_rate_per_s"]["value"], 1000.0)
        self.assertEqual(sm["io_read_mb_per_s"]["value"], 512.0)
        self.assertEqual(sm["rss_gb"]["max"]["value"], 5.0)

    def test_real_system_sampler_end_to_end(self):
        s = t.Sampler(0.05, maxlen=20, gpu=False, pid_fn=os.getpid)
        s.start()
        deadline = time.time() + 3
        while len(s.history()) < 4 and time.time() < deadline:
            time.sleep(0.02)
        s.stop()
        self.assertGreaterEqual(len(s.history()), 3)
        latest = s.latest()
        self.assertIn("ram", latest)
        self.assertEqual(latest["pid"], os.getpid())
        sm = s.summary(30)
        self.assertGreaterEqual(sm["n"], 3)


if __name__ == "__main__":
    unittest.main()
