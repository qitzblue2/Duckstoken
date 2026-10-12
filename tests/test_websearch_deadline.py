"""Review finding: the fetch deadline was only checked between reads, so a server that trickles one byte at a time could hold a
chat turn forever (the socket timeout restarts on every byte)."""
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckstoken import websearch  # noqa: E402
from duckstoken.websearch import WebError  # noqa: E402

CFG = {"web_timeout": 1, "web_allow_private": True, "web_page_chars": 3000}


class Trickle:
    """A server that answers like a web page but sends one byte every `gap` seconds."""

    def __init__(self, phase: str, gap: float = 0.25, total: float = 20.0):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.phase, self.gap, self.total = phase, gap, total
        self.stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            conn.recv(4096)
            if self.phase == "headers":
                payload = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nX-Pad: " + b"a" * 400
            else:
                payload = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n<html><body>" + b"x" * 400
            end = time.monotonic() + self.total
            for i in range(len(payload)):
                if self.stop.is_set() or time.monotonic() > end:
                    break
                conn.sendall(payload[i:i + 1])
                time.sleep(self.gap)
        except OSError:
            pass
        finally:
            conn.close()

    def close(self):
        self.stop.set()
        self.sock.close()


class Deadline(unittest.TestCase):
    def check(self, phase):
        srv = Trickle(phase)
        self.addCleanup(srv.close)
        t0 = time.monotonic()
        with self.assertRaisesRegex(WebError, "too slow"):
            websearch.fetch_page(f"http://127.0.0.1:{srv.port}/", CFG)
        took = time.monotonic() - t0
        self.assertLess(took, 5.0, f"{phase}: held the caller for {took:.1f}s (deadline is 2 s)")

    def test_body_trickle_is_cut_off_at_the_deadline(self):
        self.check("body")

    def test_header_trickle_is_cut_off_at_the_deadline(self):
        self.check("headers")


if __name__ == "__main__":
    unittest.main()
