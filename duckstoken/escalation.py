"""The big-model lane: a slow, smart model answers queued questions in the background.

Jobs wait in the `queue` table (`QueueStore`). `EscalationWorker` runs them one at a time on a second
llama-server (on `big_port`), optionally stopping the everyday model while the big one needs the RAM, and stores
every answer on the job and as the matching exchange's "teacher answer" (training data).
"""

from __future__ import annotations

import datetime
import json
import re
import threading
import time

from . import gguf, llm
from .db import Database

STATUSES = ("queued", "running", "done", "error", "cancelled")
REASONS = ("manual", "downvote", "hard", "review")
ROLES = ("system", "user", "assistant", "tool")
MAX_TEXT = 200_000               # one message / one answer
MAX_JOB_CHARS = 400_000          # the stored conversation; far more than any big-model context window holds
MAX_NAME = 200                   # chat id, model label
MAX_ERROR = 2000
MAX_LIST = 500                   # rows per list() call
MAX_REVIEW = 500                 # exchanges re-queued by one schedule_review() call
POLL_SECONDS = 5.0               # how often the worker looks at the queue when nobody wakes it
LOAD_POLL_SECONDS = 2.0
LOAD_TIMEOUT_SECONDS = 45 * 60   # a 120B model read from a slow disk really can take this long
TEACHER_TEMPERATURE = 0.6
ACTIVE_STATES = ("starting", "loading", "ready")

_JOB_COLUMNS = ("id, created, started, finished, status, reason, chat_id, exchange_id, answer, error, model, tokens, "
                "tps, priority")


# ----------------------------------------------------------------------------------------------- text helpers
def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def _content_text(content) -> str:
    """Plain text of a message's content, which may be a string or a list of parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type", "text") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "\n".join(texts)


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _as_id(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 0 <= number < 2**62 else None


def _clean_messages(messages) -> list[dict]:
    """The conversation as plain OpenAI-style messages: known roles, text-only content, tool plumbing kept."""
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages must be a list")
    cleaned = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            continue
        item = {"role": m["role"], "content": _clip(_content_text(m.get("content")), MAX_TEXT)}
        if m["role"] == "assistant" and isinstance(m.get("tool_calls"), list) and m["tool_calls"]:
            item["tool_calls"] = m["tool_calls"]
        if m["role"] == "tool" and isinstance(m.get("tool_call_id"), str):
            item["tool_call_id"] = m["tool_call_id"]
        cleaned.append(item)
    return cleaned


def _bounded_messages(messages) -> str:
    """JSON for a job's conversation: at most MAX_JOB_CHARS, oldest messages dropped first, last question kept."""
    cleaned = _clean_messages(messages)
    asked = [i for i, m in enumerate(cleaned) if m["role"] == "user" and m["content"].strip()]
    if not asked:
        raise ValueError("A big-model job needs at least one user message")
    last_question = asked[-1]
    sizes = [len(_dumps(m)) + 1 for m in cleaned]
    total, start = sum(sizes) + 2, 0
    while start < last_question and (total > MAX_JOB_CHARS or cleaned[start]["role"] == "tool"):
        total -= sizes[start]
        start += 1
    if total > MAX_JOB_CHARS:
        raise ValueError("The conversation is too long to send to the big model")
    return _dumps(cleaned[start:])


# ------------------------------------------------------------------------------------------------- the queue
class QueueFull(ValueError):
    """The queue already holds as many waiting jobs as allowed."""


class QueueStore:
    def __init__(self, db: Database):
        self.db = db

    def enqueue(self, messages, chat_id="", exchange_id=None, reason="manual", priority=0, max_queued=100) -> dict:
        """Queue a question for the big model. `messages` is the conversation up to (excluding) the answer."""
        if reason not in REASONS:
            raise ValueError(f"reason must be one of: {', '.join(REASONS)}")
        payload = _bounded_messages(messages)
        if exchange_id is not None:
            exchange_id = _as_id(exchange_id)
            if exchange_id is None:
                raise ValueError("exchange_id must be a whole number")
        priority = max(-100, min(100, int(priority)))
        max_queued = max(1, int(max_queued))
        with self.db.transaction():
            if exchange_id is not None:
                same = self.db.one(f"SELECT {_JOB_COLUMNS}, messages_json FROM queue "
                                   "WHERE status = 'queued' AND exchange_id = ? AND messages_json = ?",
                                   (exchange_id, payload))
                if same:
                    return same
            if self.db.scalar("SELECT count(*) FROM queue WHERE status = 'queued'") >= max_queued:
                raise QueueFull(f"The big-model queue is full ({max_queued} waiting). Clear some or raise the limit.")
            cur = self.db.execute("""INSERT INTO queue(created, status, reason, chat_id, exchange_id, messages_json,
                                                       priority)
                                     VALUES (?, 'queued', ?, ?, ?, ?, ?)""",
                                  (time.time(), reason, _clip(str(chat_id or ""), MAX_NAME), exchange_id, payload,
                                   priority))
            return self.db.one(f"SELECT {_JOB_COLUMNS}, messages_json FROM queue WHERE id = ?", (cur.lastrowid,))

    def get(self, id) -> dict | None:
        id = _as_id(id)
        if id is None:
            return None
        return self.db.one(f"SELECT {_JOB_COLUMNS}, messages_json FROM queue WHERE id = ?", (id,))

    def list(self, status=None, limit=100, include_messages=True) -> list[dict]:
        """Newest first. `messages_json` (the conversation, as JSON text) is left out when include_messages is False."""
        columns = _JOB_COLUMNS + (", messages_json" if include_messages else "")
        where, params = ("WHERE status = ?", [str(status)]) if status else ("", [])
        params.append(max(1, min(MAX_LIST, int(limit))))
        return self.db.query(f"SELECT {columns} FROM queue {where} ORDER BY id DESC LIMIT ?", params)

    def next_queued(self) -> dict | None:
        """The job to run next: highest priority first, then the oldest."""
        return self.db.one(f"SELECT {_JOB_COLUMNS}, messages_json FROM queue WHERE status = 'queued' "
                           "ORDER BY priority DESC, created ASC, id ASC LIMIT 1")

    def mark_running(self, id) -> bool:
        """Claim a queued job. False when it is no longer queued (for example because it was just cancelled)."""
        return self.db.execute("UPDATE queue SET status = 'running', started = ?, error = '' "
                               "WHERE id = ? AND status = 'queued'", (time.time(), id)).rowcount > 0

    def finish(self, id, answer, tokens=0, tps=0.0, model="") -> None:
        self.db.execute("""UPDATE queue SET status = 'done', finished = ?, answer = ?, error = '', tokens = ?, tps = ?,
                           model = ? WHERE id = ?""",
                        (time.time(), _clip(str(answer), MAX_TEXT), max(0, int(tokens or 0)),
                         max(0.0, float(tps or 0)), _clip(str(model or ""), MAX_NAME), id))

    def fail(self, id, error) -> None:
        self.db.execute("UPDATE queue SET status = 'error', finished = ?, error = ? WHERE id = ?",
                        (time.time(), _clip(str(error), MAX_ERROR), id))

    def requeue_running(self, id=None) -> int:
        """Put interrupted jobs back in the queue: one job, or (no id) every job still marked running."""
        where, params = ("AND id = ?", (id,)) if id is not None else ("", ())
        return self.db.execute(f"UPDATE queue SET status = 'queued', started = 0 WHERE status = 'running' {where}",
                               params).rowcount

    def cancel(self, id) -> bool:
        """Cancel a waiting job. A job that is already running cannot be stopped from here."""
        id = _as_id(id)
        return id is not None and self.db.execute("UPDATE queue SET status = 'cancelled', finished = ? "
                                                  "WHERE id = ? AND status = 'queued'", (time.time(), id)).rowcount > 0

    def clear_finished(self) -> int:
        return self.db.execute("DELETE FROM queue WHERE status IN ('done', 'error', 'cancelled')").rowcount

    def counts(self) -> dict:
        counts = dict.fromkeys(STATUSES, 0)
        for row in self.db.query("SELECT status, count(*) AS n FROM queue GROUP BY status"):
            counts[row["status"]] = row["n"]
        return counts


# -------------------------------------------------------------------------------------------- night window
def _parse_hhmm(value) -> datetime.time | None:
    m = re.fullmatch(r"\s*([0-9]{1,2}):([0-9]{2})\s*", value) if isinstance(value, str) else None
    if not m or int(m[1]) > 23 or int(m[2]) > 59:
        return None
    return datetime.time(int(m[1]), int(m[2]))


def in_window(now: datetime.time, start: str, end: str) -> bool:
    """Is `now` inside the daily window [start, end)? It may wrap past midnight; bad input or start == end -> False."""
    first, last = _parse_hhmm(start), _parse_hhmm(end)
    if first is None or last is None or first == last:
        return False
    if isinstance(now, datetime.datetime):
        now = now.time()
    now = now.replace(tzinfo=None)
    return first <= now < last if first < last else now >= first or now < last


def _night_of(now: datetime.datetime, start: str, end: str) -> datetime.date:
    """The calendar date on which the night window containing `now` began."""
    first, last = _parse_hhmm(start), _parse_hhmm(end)
    if first > last and now.time().replace(tzinfo=None) < last:
        return now.date() - datetime.timedelta(days=1)
    return now.date()


# ---------------------------------------------------------------------------------------- server settings
def _moe_layers(model_path: str) -> int:
    """Layer count of a Mixture-of-Experts GGUF (so all its experts can stay on the CPU), else 0."""
    if not model_path:
        return 0
    try:
        info = gguf.summarize(model_path)
    except (OSError, ValueError):
        return 0
    return info["block_count"] if info["is_moe"] else 0


def big_settings(s: dict) -> dict:
    """The settings dict for the big model's llama-server: everyday settings with the big_* ones laid over them."""
    path = s.get("big_model_path", "")
    out = dict(s)
    out.update(model_path=path, port=int(s.get("big_port") or 8081), ctx_size=int(s.get("big_ctx_size") or 4096),
               api_key="", host="127.0.0.1", parallel=1, mlock=False,
               n_cpu_moe=int(s.get("big_n_cpu_moe") or 0) or _moe_layers(path),
               reasoning_effort=s.get("big_reasoning_effort", "high"), lora_path="")
    return out


# ------------------------------------------------------------------------------------------- heuristics
_HARD_ANALYSIS = (r"prove (?:that|the|this|why|how|by|it)|proofs? (?:that|by)|proof of (?!concept)|"
                  r"derive (?:the|an?|this|that|it)|derivation|theorem|lemma|np-?hard|time complexity|"
                  r"space complexity|big-?o|trade[- ]?offs?|root cause|race condition|deadlock|memory leak|"
                  r"segfault|segmentation fault|stack ?trace|traceback|differential equation|eigenvalues?|"
                  r"closed form|optimi[sz]e (?:this|the|my)")
_HARD_DESIGN = r"design (?:an?|the) (?:[\w-]+ ){0,3}(?:architecture|system|schema|algorithm)|architecture for"
_STRONG = re.compile(rf"\b(?:{_HARD_ANALYSIS}|{_HARD_DESIGN})\b", re.I)
_PROBLEM = re.compile(r"\b(?:errors?|exceptions?|bugs?|crash(?:es|ed|ing)?|fail(?:s|ed|ing)?|debug|"
                      r"doesn'?t work|not working|broken)\b", re.I)
_ANALYSIS = re.compile(r"\b(?:compare|versus|pros and cons|advantages and disadvantages|step[- ]by[- ]step|in depth|"
                       r"in detail|thorough(?:ly)?|comprehensive)\b", re.I)
_MATH = re.compile(r"\b(?:solve|calculate|equations?|integral|derivative|probability|matrix|matrices)\b|"
                   r"\d\s*[\^*]\s*\d", re.I)
_CODE = re.compile(r"```|^\s*(?:def |class |function |import |#include|public static )", re.M)
_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+\S", re.M)
_MIN_WORDS = 4
_LONG_PROMPT = 500
_SCAN_CHARS = 20_000


def _last_user_text(messages_or_text) -> str:
    if isinstance(messages_or_text, str):
        return messages_or_text
    if isinstance(messages_or_text, (list, tuple)):
        for m in reversed(messages_or_text):
            if isinstance(m, dict) and m.get("role") == "user":
                return _content_text(m.get("content"))
    return ""


def hard_prompt(messages_or_text) -> bool:
    """Does this question look like one the everyday model will probably get wrong?

    `messages_or_text` is the question text or a conversation (its last user message is judged). A single strong cue
    (proofs, derivations, trade-offs, tracebacks, architecture design ...) is enough; weaker cues must add up to two.
    """
    text = _last_user_text(messages_or_text).strip()[:_SCAN_CHARS]
    if len(text.split()) < _MIN_WORDS:
        return False
    if _STRONG.search(text):
        return True
    signals = (len(text) >= _LONG_PROMPT, text.count("?") >= 3, len(_LIST_ITEM.findall(text)) >= 3,
               bool(_CODE.search(text)), bool(_PROBLEM.search(text)), bool(_ANALYSIS.search(text)),
               bool(_MATH.search(text)))
    return sum(signals) >= 2


_UNSURE_PHRASES = (
    "i'm not sure", "i am not sure", "i'm not certain", "i am not certain", "i'm not confident", "i am not confident",
    "i'm unsure", "i am unsure", "not 100% sure", "i don't know", "i do not know", "i have no idea",
    "i cannot verify", "i can't verify", "i cannot confirm", "i can't confirm", "i may be wrong", "i might be wrong",
    "i could be wrong", "i don't have enough information", "i do not have enough information",
    "i'm unable to", "i am unable to", "i can't help with", "i cannot help with", "i can't assist", "i cannot assist")
_AS_AN_AI = re.compile(r"\bas an ai\b[^.!?\n]{0,40}\b(?:can'?t|cannot|can not|unable|don'?t|do not)\b")
_SORRY_CANT = re.compile(r"\b(?:sorry|apologi[sz]e)\b[^.!?\n]{0,20}\b(?:can'?t|cannot|unable|won'?t)\b")
_SHORT_REFUSAL = re.compile(r"^(?:i (?:can'?t|cannot|can not|am unable|am not able)|i'm (?:unable|not able)|"
                            r"unable to)\b")
_SHORT_ANSWER = 160
_LONG_ANSWER = 1200
_EDGE_CHARS = 400
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})


def unsure_answer(answer: str) -> bool:
    """Does the answer admit it does not know, cannot verify, or refuse? (Empty answers are not judged.)"""
    text = " ".join(str(answer or "").translate(_APOSTROPHES).casefold().split())
    if not text:
        return False
    if len(text) > _LONG_ANSWER:  # hedges that matter show up where the answer starts or ends
        text = f"{text[:_EDGE_CHARS]} {text[-_EDGE_CHARS:]}"
    if any(phrase in text for phrase in _UNSURE_PHRASES):
        return True
    if _AS_AN_AI.search(text) or _SORRY_CANT.search(text):
        return True
    return len(text) <= _SHORT_ANSWER and bool(_SHORT_REFUSAL.match(text))


# ---------------------------------------------------------------------------------------------- the worker
def _model_label(*names) -> str:
    """A short display name for a model: the first non-empty file name, without folders or the .gguf suffix."""
    for name in names:
        base = re.split(r"[\\/]", str(name or "").strip())[-1]
        if base.lower().endswith(".gguf"):
            base = base[:-5]
        if base:
            return _clip(base, MAX_NAME)
    return "big model"


def _answer_timeout(max_tokens: int) -> float:
    """Seconds to wait for one answer: slow CPU generation (about 0.25 token/s at worst), but never absurd."""
    return float(min(6 * 3600, max(1800, 4 * max_tokens)))


class EscalationWorker:
    """Background thread that works through the queue on the big model and keeps the two servers tidy."""

    def __init__(self, db: Database, get_settings, small_server, big_server, exchanges, queue: QueueStore,
                 complete=llm.complete, clock=datetime.datetime.now, sleep=time.sleep):
        self.db, self.get_settings = db, get_settings
        self.small_server, self.big_server = small_server, big_server
        self.exchanges, self.queue = exchanges, queue
        self.complete, self.clock, self.sleep = complete, clock, sleep
        self._lock = threading.Lock()  # guards the status fields below
        self._busy = threading.Lock()  # held while a job runs or the servers are being shuffled
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase, self._current_job, self._last_error = "idle", None, ""
        self._run_now = False
        self._small_was_stopped = False
        self._idle_since: datetime.datetime | None = None
        self._reviewed_night: datetime.date | None = None

    # ------------------------------------------------------------------ thread
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.queue.requeue_running()  # a job still "running" now was cut off by the last shutdown
        self._thread = threading.Thread(target=self._loop, name="escalation", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and the big model. The everyday model is left alone: this is for shutting down."""
        self._stop.set()
        self._wake.set()
        self.big_server.stop()  # also cuts short an answer that is still being generated
        thread, self._thread = self._thread, None
        if thread and thread is not threading.current_thread():
            thread.join(timeout=10)
        self.big_server.stop()  # in case the thread started it again before noticing
        self._small_was_stopped = False
        self._set_phase("idle")

    def wake(self) -> None:
        self._wake.set()

    def run_now(self) -> None:
        """Work through the whole queue, whatever the run mode says."""
        self._run_now = True
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001 - one bad pass must not kill the worker
                self._set_error(f"Big-model worker problem: {e}")
            self._wake.wait(POLL_SECONDS)

    def _tick(self) -> None:
        self.maybe_nightly_review()
        while not self._stop.is_set() and self.should_run() and self.run_one():
            pass
        if self._busy.acquire(blocking=False):
            try:
                self._settle()
            finally:
                self._busy.release()

    # ------------------------------------------------------------------ deciding
    def _due(self, cfg: dict, queued: int) -> bool:
        if not queued:
            return False
        if self._run_now:
            return True
        mode = cfg.get("big_run_mode")
        if mode == "immediate":
            return True
        return mode == "overnight" and in_window(self.clock().time(), cfg.get("overnight_start", ""),
                                                 cfg.get("overnight_end", ""))

    def should_run(self) -> bool:
        return self._due(self.get_settings(), self.queue.counts()["queued"])

    # ------------------------------------------------------------------ jobs
    def run_one(self) -> bool:
        """Answer the next queued job on the big model. False when there is nothing to do."""
        if not self._busy.acquire(blocking=False):
            return False
        try:
            job = self.queue.next_queued()
            if job is None:
                return False
            if self.queue.mark_running(job["id"]):
                self._execute(job)
                self._idle_since = self.clock()
                self._settle()
            return True
        finally:
            self._busy.release()

    def _execute(self, job: dict) -> None:
        self._set_job("answering", job["id"])
        try:
            cfg = self.get_settings()
            reply, text = self._ask_big(job, cfg)
            self._store(job, cfg, reply, text)
            self._set_error("")
        except Exception as e:  # noqa: BLE001 - whatever goes wrong must end up on the job, never leave it "running"
            if self._stop.is_set():  # shutting down is not the job's fault
                self.queue.requeue_running(job["id"])
            else:
                message = str(e) or type(e).__name__
                self.queue.fail(job["id"], message)
                self._set_error(message)
        finally:
            self._set_job("idle", None)

    def _ask_big(self, job: dict, cfg: dict) -> tuple[dict, str]:
        if not cfg.get("big_model_path"):
            raise RuntimeError("No big model is selected. Pick one in Settings -> Big model.")
        messages = json.loads(job["messages_json"])
        self._bring_up_big(cfg)
        self._set_phase("answering")
        max_tokens = int(cfg.get("big_max_tokens") or 3000)
        reply = self.complete(self.big_server, messages, max_tokens=max_tokens, temperature=TEACHER_TEMPERATURE,
                              effort=cfg.get("big_reasoning_effort", "high"), timeout=_answer_timeout(max_tokens))
        text = (reply.get("content") or "").strip()
        if not text:
            raise RuntimeError("The big model gave an empty answer. It may have spent every token thinking; "
                               "try raising 'Max tokens per big-model answer'.")
        return reply, text

    def _store(self, job: dict, cfg: dict, reply: dict, text: str) -> None:
        label = _model_label(cfg.get("big_model_path"), reply.get("model"))
        with self.db.transaction():  # the job is only "done" together with its teacher answer
            self.queue.finish(job["id"], text, tokens=reply.get("tokens", 0), tps=reply.get("tps", 0.0), model=label)
            if job["exchange_id"] is not None:
                self.exchanges.set_teacher(job["exchange_id"], text, label)

    # ------------------------------------------------------------------ servers
    @staticmethod
    def _active(server) -> bool:
        return server.status().get("state") in ACTIVE_STATES

    def _bring_up_big(self, cfg: dict) -> None:
        if self._stop.is_set():
            raise RuntimeError("Duckstoken is shutting down")
        if cfg.get("big_swap_models") and self._active(self.small_server):
            self._set_phase("swapping")
            self.small_server.stop()
            self._small_was_stopped = True
        if self.big_server.status().get("state") == "ready":
            return
        self._set_phase("loading_big")
        if not self._active(self.big_server):
            if int(cfg.get("big_port") or 0) == int(cfg.get("port") or 0) and self._active(self.small_server):
                raise RuntimeError(f"The big model's port ({cfg.get('big_port')}) is the everyday model's port. "
                                   "Change 'Big model server port' in Settings.")
            self.big_server.start(big_settings(cfg))
        self._wait_big_ready()

    def _wait_big_ready(self) -> None:
        waited = 0.0
        while waited < LOAD_TIMEOUT_SECONDS:
            if self._stop.is_set():
                raise RuntimeError("Duckstoken is shutting down")
            status = self.big_server.status()
            if status.get("state") == "ready":
                return
            if status.get("state") in ("error", "stopped"):
                raise RuntimeError(status.get("error") or "The big model server stopped before it finished loading.")
            self.sleep(LOAD_POLL_SECONDS)
            waited += LOAD_POLL_SECONDS
        self.big_server.stop()
        raise RuntimeError(f"The big model did not finish loading within {LOAD_TIMEOUT_SECONDS // 60} minutes.")

    def _settle(self) -> None:
        """Between jobs (caller holds `_busy`): forget a finished run-now, and give servers back when nothing is due.

        With the everyday model switched off for us it comes back as soon as nothing is due; otherwise the big
        model is kept for `big_idle_stop_minutes` in case more jobs arrive.
        """
        cfg = self.get_settings()
        queued = self.queue.counts()["queued"]
        if not queued:
            self._run_now = False
        if self._due(cfg, queued):
            self._idle_since = None
            return
        if not (self._small_was_stopped or self._active(self.big_server)):
            return
        now = self.clock()
        if self._idle_since is None:
            self._idle_since = now
        idle_limit = max(0, int(cfg.get("big_idle_stop_minutes") or 0)) * 60
        if self._small_was_stopped or (now - self._idle_since).total_seconds() >= idle_limit:
            self._release()

    def _release(self) -> None:
        self.big_server.stop()
        self._idle_since = None
        if not self._small_was_stopped:
            return
        self._small_was_stopped = False
        if self._active(self.small_server):  # the user already started it again
            return
        try:
            self.small_server.start(self.get_settings())
        except RuntimeError as e:
            self._set_error(f"Could not restart the everyday model: {e}")

    # ------------------------------------------------------------------ reviews
    def schedule_review(self, n: int) -> int:
        """Queue the newest `n` answers that have no big-model answer yet. Returns how many jobs were added."""
        n = max(0, min(MAX_REVIEW, int(n)))
        if n == 0:
            return 0
        max_queued = int(self.get_settings().get("queue_max") or 100)
        began, added = time.time(), 0
        for ex in self.exchanges.recent_without_teacher(n):
            try:
                job = self.queue.enqueue(ex["messages"], chat_id=ex["chat_id"], exchange_id=ex["id"],
                                         reason="review", priority=-1, max_queued=max_queued)
            except QueueFull:
                break
            except ValueError:  # an exchange that has no usable question
                continue
            if job["created"] >= began:  # an older job means this exchange was already waiting
                added += 1
        if added:
            self.wake()
        return added

    def maybe_nightly_review(self) -> int:
        """Queue the nightly review once per night, the first time it is called inside the night window."""
        cfg = self.get_settings()
        count = int(cfg.get("overnight_review_count") or 0)
        now = self.clock()
        start, end = cfg.get("overnight_start", ""), cfg.get("overnight_end", "")
        if count <= 0 or not cfg.get("big_model_path") or not in_window(now.time(), start, end):
            return 0
        night = _night_of(now, start, end)
        if night == self._reviewed_night:
            return 0
        self._reviewed_night = night
        return self.schedule_review(count)

    # ------------------------------------------------------------------ status
    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def _set_job(self, phase: str, job: int | None) -> None:
        with self._lock:
            self._phase, self._current_job = phase, job

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message

    def status(self) -> dict:
        cfg = self.get_settings()
        counts = self.queue.counts()
        with self._lock:
            phase, job, error = self._phase, self._current_job, self._last_error
        if phase == "idle" and counts["queued"] and not self._due(cfg, counts["queued"]):
            phase = "waiting"
        return {"phase": phase, "current_job": job, "run_now": self._run_now,
                "window_active": in_window(self.clock().time(), cfg.get("overnight_start", ""),
                                           cfg.get("overnight_end", "")),
                "big_state": self.big_server.status().get("state", "stopped"), "counts": counts,
                "small_was_stopped": self._small_was_stopped, "last_error": error}
