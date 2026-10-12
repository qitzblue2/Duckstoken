"""HTTP endpoints for app updates and the native shell (autostart). Mixed into the request handler in app.py."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from . import updater
from .errors import ApiError

APP = None


def bind(app) -> None:
    global APP
    APP = app


def post(fn):
    fn._method = "POST"
    return fn


class UpdateJob:
    """Progress of the one update download (the checker verifies size + SHA-256 before it reports success)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state, self.done, self.total, self.error = "idle", 0, 0, ""
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None

    def snapshot(self) -> dict:
        with self.lock:
            return {"state": self.state, "done": self.done, "total": self.total, "error": self.error}

    def start(self, checker: updater.UpdateChecker) -> None:
        with self.lock:
            if self.state == "downloading":
                raise ApiError("An update is already downloading", 409)
            self.state, self.done, self.total, self.error = "downloading", 0, 0, ""
            self.cancel = threading.Event()
        cancel = self.cancel

        def progress(done: int, total: int) -> None:
            with self.lock:
                self.done, self.total = done, total

        def work() -> None:
            try:
                checker.download_update(progress, cancel)
                state, err = "downloaded", ""
            except updater.UpdateError as e:
                state, err = ("cancelled", "") if cancel.is_set() else ("error", str(e))
            except Exception as e:  # noqa: BLE001 - never let a worker thread die silently
                state, err = "error", f"{e.__class__.__name__}: {e}"
            with self.lock:
                self.state, self.error = state, err

        self.thread = threading.Thread(target=work, daemon=True, name="update-download")
        self.thread.start()


def update_status() -> dict:
    info = updater.current_info()
    st = APP.updater.status()
    return {"current": info, "channel": APP.settings.get("update_channel"), "can_apply": bool(info["frozen"]) and _windows(),
            "auto_install": bool(APP.settings.get("update_auto_install")), "download": APP.update_job.snapshot(), **st}


def _windows() -> bool:
    return os.name == "nt"


class SystemApi:
    def api_update_status(self, q, b):
        return update_status()

    @post
    def api_update_check(self, q, b):
        if APP.settings.get("update_channel") == "off":
            raise ApiError("Update checks are turned off (Settings → App & updates → Update channel).")
        APP.updater.maybe_check(force=True)
        out = update_status()
        if out["last_error"]:
            raise ApiError(out["last_error"], 502)
        return out

    @post
    def api_update_download(self, q, b):
        res = APP.updater.status()["result"]
        if not res or not res.get("available"):
            raise ApiError("There is no update to download. Check for updates first.")
        APP.update_job.start(APP.updater)
        return update_status()

    @post
    def api_update_cancel(self, q, b):
        APP.update_job.cancel.set()
        return update_status()

    @post
    def api_update_apply(self, q, b):
        """Install the verified download and quit so the helper can swap the files. Needs an explicit `confirm`."""
        if b.get("confirm") is not True:
            raise ApiError("Applying an update restarts Duckstoken. Confirm to continue.")
        st = APP.updater.status()
        if not st["downloaded"]:
            raise ApiError("Download the update first.")
        try:
            result = updater.apply(Path(st["downloaded"]), restart=bool(b.get("restart", True)))
        except updater.UpdateError as e:
            raise ApiError(str(e)) from e
        APP.updater.mark_applied()  # shutdown() must not launch the installer a second time
        threading.Timer(0.8, APP.shutdown).start()  # let this response reach the browser first
        return {"applied": True, **result}

    def api_shell_info(self, q, b):
        from . import shell
        return {"autostart": shell.autostart_enabled(), "window_mode": APP.settings.get("window_mode")}


def background_update_check(app) -> None:
    """Startup check (at most once a day per channel). With `update_auto_install` the verified download is fetched too,
    and installed when the app exits (App.shutdown)."""
    try:
        res = app.updater.maybe_check()
        if res is None:
            res = (app.updater.status() or {}).get("result")
        if res and res.get("available") and app.settings.get("update_auto_install") and updater.current_info()["frozen"]:
            app.update_job.start(app.updater)
    except Exception:  # noqa: BLE001 - updates must never get in the way of starting
        pass


def apply_on_exit(app) -> None:
    """update_auto_install: swap in an already-verified download as the app quits (Windows packaged builds only)."""
    if not app.settings.get("update_auto_install"):
        return
    try:
        st = app.updater.status()
        if st["downloaded"] and updater.current_info()["frozen"]:
            app.updater.mark_applied()
            updater.apply(Path(st["downloaded"]), restart=False)
    except Exception:  # noqa: BLE001
        pass

