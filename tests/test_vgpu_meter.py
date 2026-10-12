import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckstoken.vgpu.meter import RequestMeter  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def sse(obj):
    return b"data: " + json.dumps(obj).encode() + b"\n"


class Meter(unittest.TestCase):
    def test_uses_server_timings_when_present(self):
        c = Clock()
        m = RequestMeter(clock=c, wall=lambda: 5.0)
        c.t += 0.8
        m.feed(sse({"choices": [{"delta": {"content": "Hi"}}]}))
        c.t += 1.0
        m.feed(sse({"choices": [{"delta": {"content": "!"}}], "timings": {"prompt_n": 10, "prompt_per_second": 55.5, "predicted_n": 2, "predicted_per_second": 7.25}}))
        r = m.result("llamacpp")
        self.assertEqual(r["ttft_s"]["value"], 0.8)
        self.assertEqual(r["gen_tps"]["value"], 7.25)
        self.assertEqual(r["gen_tps"]["status"], "measured")
        self.assertEqual(r["prompt_tps"]["value"], 55.5)
        self.assertEqual(r["gen_tokens"]["value"], 2)

    def test_falls_back_to_chunks_and_labels_it(self):
        c = Clock()
        m = RequestMeter(clock=c)
        for _ in range(5):
            c.t += 0.5
            m.feed(sse({"choices": [{"delta": {"content": "x"}}]}))
        r = m.result()
        self.assertEqual(r["gen_tps"]["status"], "experimental")
        self.assertAlmostEqual(r["gen_tps"]["value"], 2.0, places=2)
        self.assertEqual(r["prompt_tps"]["status"], "unavailable")

    def test_no_tokens_is_unavailable_not_zero(self):
        m = RequestMeter(clock=Clock())
        m.feed(b": keep-alive\n")
        m.feed(b"data: [DONE]\n")
        r = m.result()
        self.assertIsNone(r["ttft_s"]["value"])
        self.assertEqual(r["gen_tps"]["status"], "unavailable")

    def test_non_stream_body(self):
        c = Clock()
        m = RequestMeter(clock=c)
        c.t += 2
        m.feed_body(json.dumps({"usage": {"prompt_tokens": 7, "completion_tokens": 9}, "timings": {"predicted_per_second": 4.0, "predicted_n": 9}}).encode())
        r = m.result()
        self.assertEqual(r["gen_tps"]["value"], 4.0)
        self.assertEqual(r["prompt_tokens"]["value"], 7)

    def test_garbage_lines_never_raise(self):
        m = RequestMeter(clock=Clock())
        m.feed(b'data: {"timings": not json}\n')
        m.feed_body(b"\xff\xfe")
        m.result()


if __name__ == "__main__":
    unittest.main()
