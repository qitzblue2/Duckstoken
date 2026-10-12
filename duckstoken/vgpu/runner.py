"""Single-slot background job runner.

Benchmarks and optimizer runs load multi-gigabyte models, so two of them must never overlap: `JobRunner` runs ONE job
at a time on a daemon thread and refuses (`Busy`) to start another while one is active.  The finished job's result
stays available until the next job starts.

A job function has the signature ``fn(cancel, progress, *args)``: ``cancel`` is a shared ``threading.Event`` that
`JobRunner.cancel()` sets (the job must poll it and stop promptly), ``progress(step, total, message)`` reports
progress (``total`` may be 0 when unknown) and ``progress.log(line)`` adds a free-form line to the job log.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import deque
from typing import Any, Callable

LOG_LINES = 200


class Busy(RuntimeError):
    """A job is already running."""


class JobCancelled(Exception):
    """A job function may raise this to finish as ``cancelled`` instead of ``failed``."""


class _Progress:
    """The `progress` callable handed to a job."""

    def __init__(self, runner: "JobRunner"):
        self._runner = runner

    def __call__(self, step: int = 0, total: int = 0, message: str = "") -> None:
        self._runner._set_progress(step, total, message)

    def log(self, line: str) -> None:
        self._runner._log(str(line))


class JobRunner:
    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._reset()

    def _reset(self) -> None:
        self._id: str | None = None
        self._kind = ""
        self._state = "idle"
        self._progress = {"step": 0, "total": 0, "message": "", "pct": None}
        self._result: Any = None
        self._error = ""
        self._started: float | None = None
        self._finished: float | None = None
        self._log_lines: deque[str] = deque(maxlen=LOG_LINES)

    # ------------------------------------------------------------------ control
    def start(self, kind: str, fn: Callable, *args) -> str:
        """Run ``fn(cancel, progress, *args)`` in the background and return the job id; raises `Busy` if one is active."""
        with self._lock:
            if self._state == "running":
                raise Busy(f"A {self._kind} job is already running. Wait for it or cancel it first.")
            self._reset()
            self._cancel = threading.Event()
            self._id = f"job-{uuid.uuid4().hex[:10]}"
            self._kind, self._state, self._started = str(kind), "running", self._clock()
            job_id, cancel = self._id, self._cancel
            self._log(f"started {kind}")
            self._thread = threading.Thread(target=self._run, args=(job_id, fn, args, cancel),
                                            name=f"vgpu-job-{kind}", daemon=True)
            self._thread.start()
            return job_id

    def _run(self, job_id: str, fn: Callable, args: tuple, cancel: threading.Event) -> None:
        state, result, error = "done", None, ""
        try:
            result = fn(cancel, _Progress(self), *args)
            if cancel.is_set():
                state = "cancelled"
        except JobCancelled as e:
            state, error = "cancelled", str(e)
        except BaseException as e:  # noqa: BLE001 - a job must never take the runner down
            state, error = "failed", f"{type(e).__name__}: {e}"
            self._log(traceback.format_exc(limit=8).rstrip())
        with self._lock:
            if self._id != job_id:
                return
            self._state, self._result, self._error, self._finished = state, result, error, self._clock()
            self._log(f"{state}" + (f": {error}" if error else ""))

    def cancel(self) -> bool:
        """Ask the running job to stop.  Returns False when nothing is running."""
        with self._lock:
            if self._state != "running":
                return False
            self._cancel.set()
            self._log("cancel requested")
            return True

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the job has finished; True if it has (or none was running), False on timeout."""
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def active(self) -> bool:
        with self._lock:
            return self._state == "running"

    # ------------------------------------------------------------------ job-side callbacks
    def _set_progress(self, step: int, total: int, message: str) -> None:
        try:
            step, total = max(0, int(step)), max(0, int(total))
        except (TypeError, ValueError):
            step = total = 0
        pct = min(100, round(100 * step / total)) if total else None
        with self._lock:
            if self._state != "running":
                return
            self._progress = {"step": step, "total": total, "message": str(message), "pct": pct}
            if message:
                self._log(f"[{step}/{total}] {message}" if total else str(message))

    def _log(self, line: str) -> None:
        with self._lock:
            stamp = time.strftime("%H:%M:%S", time.localtime(self._clock()))
            for part in str(line).splitlines() or [""]:
                self._log_lines.append(f"{stamp} {part}")

    # ------------------------------------------------------------------ view
    def status(self) -> dict:
        with self._lock:
            return {"id": self._id, "kind": self._kind, "state": self._state,
                    "progress": dict(self._progress), "result": self._result, "error": self._error,
                    "started": self._started, "finished": self._finished,
                    "cancel_requested": self._cancel.is_set() and self._state == "running",
                    "log": list(self._log_lines)}
