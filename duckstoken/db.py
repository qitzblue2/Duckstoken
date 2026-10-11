"""One SQLite database for everything Duckstoken learns (memories, exchanges, queue, dataset).

Stores take a `Database` so tests can point them at a temp file. Connections are per-thread.
FTS5 is used for memory search when this Python's SQLite has it; `has_fts` says whether it does.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from . import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories(
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'fact',      -- fact preference project lesson note web doc
  source TEXT NOT NULL DEFAULT 'user',    -- user auto model feedback web import
  url TEXT NOT NULL DEFAULT '',
  title TEXT NOT NULL DEFAULT '',
  pinned INTEGER NOT NULL DEFAULT 0,
  enabled INTEGER NOT NULL DEFAULT 1,
  uses INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL,
  updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS memories_kind ON memories(kind);

CREATE TABLE IF NOT EXISTS exchanges(
  id INTEGER PRIMARY KEY,
  chat_id TEXT NOT NULL,
  msg_index INTEGER NOT NULL,
  created REAL NOT NULL,
  prompt TEXT NOT NULL DEFAULT '',        -- last user message text
  messages_json TEXT NOT NULL DEFAULT '[]', -- conversation up to (not including) the answer
  answer TEXT NOT NULL DEFAULT '',
  reasoning TEXT NOT NULL DEFAULT '',
  model TEXT NOT NULL DEFAULT '',
  effort TEXT NOT NULL DEFAULT '',
  tokens INTEGER NOT NULL DEFAULT 0,
  tps REAL NOT NULL DEFAULT 0,
  rating INTEGER NOT NULL DEFAULT 0,      -- 1 liked, -1 disliked, 0 none
  note TEXT NOT NULL DEFAULT '',
  correction TEXT NOT NULL DEFAULT '',
  teacher_answer TEXT NOT NULL DEFAULT '',
  teacher_model TEXT NOT NULL DEFAULT '',
  used_tools INTEGER NOT NULL DEFAULT 0,
  web_sources TEXT NOT NULL DEFAULT '[]', -- JSON list of {title,url}
  UNIQUE(chat_id, msg_index)
);
CREATE INDEX IF NOT EXISTS exchanges_rating ON exchanges(rating);

CREATE TABLE IF NOT EXISTS queue(
  id INTEGER PRIMARY KEY,
  created REAL NOT NULL,
  started REAL NOT NULL DEFAULT 0,
  finished REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'queued',  -- queued running done error cancelled
  reason TEXT NOT NULL DEFAULT 'manual',  -- manual downvote hard review
  chat_id TEXT NOT NULL DEFAULT '',
  exchange_id INTEGER,
  messages_json TEXT NOT NULL DEFAULT '[]',
  answer TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '',
  model TEXT NOT NULL DEFAULT '',
  tokens INTEGER NOT NULL DEFAULT 0,
  tps REAL NOT NULL DEFAULT 0,
  priority INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS queue_status ON queue(status);

CREATE TABLE IF NOT EXISTS dataset_items(
  id INTEGER PRIMARY KEY,
  exchange_id INTEGER NOT NULL,
  kind TEXT NOT NULL,                     -- sft pref
  messages_json TEXT NOT NULL,            -- prompt messages (no answer)
  chosen TEXT NOT NULL,
  rejected TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL,                   -- liked corrected teacher
  status TEXT NOT NULL DEFAULT 'pending', -- pending approved rejected
  edited INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL,
  UNIQUE(exchange_id, kind)
);
CREATE INDEX IF NOT EXISTS dataset_status ON dataset_items(status);
"""

FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, title, tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, text, title) VALUES (new.id, new.text, new.title);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF text, title ON memories BEGIN
  UPDATE memories_fts SET text = new.text, title = new.title WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  DELETE FROM memories_fts WHERE rowid = old.id;
END;
"""


class Database:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else paths.learning_db_file()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self.has_fts = False
        self._init()

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)  # autocommit; we use explicit tx
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        return c

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = self._local.c = self._connect()
        return c

    def _init(self) -> None:
        c = self.conn()
        c.executescript(SCHEMA)
        try:
            c.executescript(FTS)
            self.has_fts = True
            # Rebuild if the FTS table was created after memories already existed.
            n_fts = c.execute("SELECT count(*) FROM memories_fts").fetchone()[0]
            n_mem = c.execute("SELECT count(*) FROM memories").fetchone()[0]
            if n_fts != n_mem:
                c.execute("DELETE FROM memories_fts")
                c.execute("INSERT INTO memories_fts(rowid, text, title) SELECT id, text, title FROM memories")
        except sqlite3.OperationalError:
            self.has_fts = False

    # -------------------------------------------------------------- helpers
    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._write_lock:
            return self.conn().execute(sql, params)

    def query(self, sql: str, params: tuple | list = ()) -> list[dict]:
        return [dict(r) for r in self.conn().execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple | list = ()) -> dict | None:
        r = self.conn().execute(sql, params).fetchone()
        return dict(r) if r else None

    def scalar(self, sql: str, params: tuple | list = (), default=0):
        r = self.conn().execute(sql, params).fetchone()
        return r[0] if r and r[0] is not None else default

    @contextmanager
    def transaction(self):
        with self._write_lock:
            c = self.conn()
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
            except BaseException:
                c.execute("ROLLBACK")
                raise
            else:
                c.execute("COMMIT")
