"""The HTTP server runs each request on a new thread; their SQLite connections must not pile up."""
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duckstoken.db import Database  # noqa: E402


class ConnectionLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Database(Path(self.tmp.name) / "t.db")
        self.addCleanup(self.db.close)

    def burst(self, n):
        def one():
            self.db.execute("SELECT count(*) FROM memories").fetchone()
        threads = [threading.Thread(target=one) for _ in range(n)]
        for t in threads:
            t.start()
            t.join()  # one after another, like sequential HTTP requests

    def test_connections_of_finished_threads_are_closed(self):
        self.burst(200)
        self.burst(1)  # the next new thread prunes the dead ones
        self.assertLess(len(self.db._connections), 5)

    def test_live_threads_keep_their_connection(self):
        ready, done = threading.Event(), threading.Event()
        seen = []

        def worker():
            seen.append(self.db.conn())
            ready.set()
            done.wait(5)
            seen.append(self.db.conn())

        t = threading.Thread(target=worker)
        t.start()
        ready.wait(5)
        self.burst(3)  # pruning must not close the live worker's connection
        done.set()
        t.join()
        self.assertIs(seen[0], seen[1])
        seen[0].execute("SELECT 1")  # still usable

    def test_close_closes_everything(self):
        self.burst(3)
        self.db.close()
        self.assertEqual(self.db._connections, [])


if __name__ == "__main__":
    unittest.main()
