"""Self-update from the project's GitHub releases.

Two channels: `stable` (tagged releases `vX.Y.Z`) and `latest` (the rolling pre-release tagged `latest`, built from every
commit of the main branch and identified by a `Build: <sha>` line in its notes). Every release carries `Duckstoken.exe`
(portable), `Duckstoken-Setup.exe` (installer) and `SHA256SUMS.txt`.

Privacy and safety: the only requests made are to the GitHub API and to the release assets of REPO, with a constant
user agent and no credentials or identifiers. Asset URLs must point at https://github.com/<REPO>/releases/download/...,
redirects may only go to GitHub's own download hosts, and a download is installed only if its size and SHA-256 match
the published checksum file.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import ntpath
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from . import __version__, _build, bundle, paths

REPO = "qitzblue2/Duckstoken"
API_BASE = "https://api.github.com"
UA = "Duckstoken-Updater"

PORTABLE_ASSET = "Duckstoken.exe"
INSTALLER_ASSET = "Duckstoken-Setup.exe"
CHECKSUM_ASSET = "SHA256SUMS.txt"

CHANNELS = ("off", "stable", "latest")
CHECK_INTERVAL = 24 * 3600  # seconds between automatic checks per channel
ERROR_BACKOFF = 3600  # seconds before an automatic check is retried after a failure
STATE_FILE = "update_state.json"

NOTES_LIMIT = 2000
MAX_JSON_BYTES = 4 << 20
MAX_CHECKSUM_BYTES = 64 << 10
MAX_ASSET_BYTES = 2 << 30

_GITHUB_ORIGIN = "https://github.com"
_DOWNLOAD_ORIGINS = frozenset({_GITHUB_ORIGIN, "https://objects.githubusercontent.com",
                               "https://release-assets.githubusercontent.com"})
_BUILD_LINE = re.compile(r"^\s*Build:\s*([0-9a-fA-F]{40})\s*$", re.MULTILINE)
_CHECKSUM_LINE = re.compile(r"([0-9a-fA-F]{64})[ \t]+\*?(\S.*?)\s*")
_VERSION = re.compile(r"[vV]?(\d{1,6}(?:\.\d{1,6}){0,3})")

# Windows process creation flags (spelled out so this module imports everywhere)
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000


class UpdateError(Exception):
    """Anything that stops an update check, download or installation; the message is meant for the user."""


class _NotFound(UpdateError):
    pass


# ----------------------------------------------------------------------------------------------- versions & build
def parse_version(tag: str) -> tuple[int, ...]:
    """'v1.2.3' -> (1, 2, 3). Accepts 1-4 numeric parts with an optional leading v and pads to at least three."""
    m = _VERSION.fullmatch(tag.strip()) if isinstance(tag, str) else None
    if not m:
        raise ValueError(f"not a version number: {tag!r}")
    parts = tuple(int(p) for p in m[1].split("."))
    return parts + (0,) * (3 - len(parts))


def _sort_key(version: tuple[int, ...]) -> tuple[int, ...]:
    return version + (0,) * (4 - len(version))


def _format_version(version: tuple[int, ...]) -> str:
    return ".".join(str(p) for p in version)


def current_info() -> dict:
    frozen = bundle.frozen()
    return {"version": __version__, "build_sha": _build.BUILD_SHA, "build_time": _build.BUILD_TIME,
            "installer": bool(_build.INSTALLER), "frozen": frozen, "exe": sys.executable if frozen else None}


# --------------------------------------------------------------------------------------------------------- network
def _origin(url: str) -> str | None:
    """'scheme://host[:port]' of an http(s) URL without credentials, else None."""
    try:
        u = urllib.parse.urlsplit(url)
        port = u.port
    except ValueError:
        return None
    if u.scheme not in ("http", "https") or not u.hostname or u.username is not None or u.password is not None:
        return None
    return f"{u.scheme}://{u.hostname}" + (f":{port}" if port else "")


def _is_github(origin: str) -> bool:
    return origin == _GITHUB_ORIGIN


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    """Follows redirects only to origins `allowed` accepts."""

    def __init__(self, allowed: Callable[[str], bool]):
        self._allowed = allowed

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        origin = _origin(newurl)
        if origin is None or not self._allowed(origin):
            raise UpdateError("GitHub redirected the request to an address that is not trusted, so it was refused.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _asset_redirects(allow_host: Callable[[str], bool]) -> Callable[[str], bool]:
    """Release downloads may bounce between github.com and GitHub's asset hosts, nowhere else."""
    return lambda origin: allow_host(origin) or origin in _DOWNLOAD_ORIGINS


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass  # a leftover temporary file is harmless


def _describe(e: Exception) -> str:
    reason = e.reason if isinstance(e, urllib.error.URLError) else None
    return str(reason or getattr(e, "strerror", None) or e) or e.__class__.__name__


def _open_url(url: str, *, accept: str, timeout: float, redirect_ok: Callable[[str], bool]):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
    try:
        return urllib.request.build_opener(_GuardedRedirect(redirect_ok)).open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        e.close()
        if e.code == 404:
            raise _NotFound("GitHub has no such release or file (HTTP 404).") from e
        if e.code == 429 or (e.code == 403 and e.headers.get("X-RateLimit-Remaining") == "0"):
            raise UpdateError("GitHub is limiting requests from this network right now. Try again later.") from e
        raise UpdateError(f"GitHub answered with an error (HTTP {e.code}).") from e
    except (OSError, http.client.HTTPException) as e:
        raise UpdateError(f"Could not reach GitHub ({_describe(e)}). Check your internet connection.") from e


def _fetch(url: str, *, accept: str, timeout: float, limit: int, redirect_ok: Callable[[str], bool]) -> bytes:
    with _open_url(url, accept=accept, timeout=timeout, redirect_ok=redirect_ok) as resp:
        try:
            data = resp.read(limit + 1)
        except (OSError, http.client.HTTPException) as e:
            raise UpdateError(f"The connection to GitHub dropped ({_describe(e)}).") from e
    if len(data) > limit:
        raise UpdateError("GitHub sent an unexpectedly large response, so it was ignored.")
    return data


def _api_get(url: str, timeout: float) -> object:
    api_origin = _origin(url)
    data = _fetch(url, accept="application/vnd.github+json", timeout=timeout, limit=MAX_JSON_BYTES,
                  redirect_ok=lambda origin: origin == api_origin)
    try:
        return json.loads(data)
    except ValueError as e:
        raise UpdateError("GitHub sent a response that could not be understood.") from e


def _trusted_asset_url(url: object, allow_host: Callable[[str], bool]) -> str:
    """The asset URL if it is https://github.com/<REPO>/releases/download/...; otherwise UpdateError.

    `allow_host` receives the URL's 'scheme://host[:port]' and decides whether that origin is trusted (tests widen it)."""
    origin = _origin(url) if isinstance(url, str) else None
    if origin is not None and allow_host(origin):
        path = urllib.parse.unquote(urllib.parse.urlsplit(url).path)
        if path.lower().startswith(f"/{REPO}/releases/download/".lower()) and ".." not in path.split("/"):
            return url
    raise UpdateError("The release points at a download address that is not trusted, so the update was refused.")


def parse_checksums(text: str) -> dict[str, str]:
    """{file name: lowercase SHA-256} from `<hex>  <name>` lines; a name listed with two different hashes is an error."""
    sums: dict[str, str] = {}
    for line in text.splitlines():
        m = _CHECKSUM_LINE.fullmatch(line)
        if not m:
            continue
        digest, name = m[1].lower(), m[2]
        if sums.setdefault(name, digest) != digest:
            raise UpdateError(f"The checksum file lists {name} twice with different values, so it cannot be trusted.")
    return sums


# ---------------------------------------------------------------------------------------------------------- checking
def _result(channel: str, **fields) -> dict:
    result = {"available": False, "channel": channel, "version": None, "sha": None, "notes": "", "published": "",
              "asset": None, "sha256": None, "html_url": "", "reason": ""}
    result.update(fields)
    return result


def _release_asset(release: dict, name: str, allow_host: Callable[[str], bool]) -> dict | None:
    assets = release.get("assets")
    for a in assets if isinstance(assets, list) else []:
        if isinstance(a, dict) and a.get("name") == name and a.get("state", "uploaded") == "uploaded":
            size = a.get("size")
            if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_ASSET_BYTES:
                raise UpdateError(f"{name} in the release has an unexpected size, so the update was refused.")
            return {"name": name, "url": _trusted_asset_url(a.get("browser_download_url"), allow_host), "size": size}
    return None


def _release_result(channel: str, release: dict, available: bool, reason: str, **fields) -> dict:
    body, published = release.get("body"), release.get("published_at") or release.get("created_at")
    tag = urllib.parse.quote(str(release.get("tag_name") or ""))
    return _result(channel, available=available, reason=reason,
                   notes=(body.strip() if isinstance(body, str) else "")[:NOTES_LIMIT],
                   published=published if isinstance(published, str) else "",
                   html_url=f"{_GITHUB_ORIGIN}/{REPO}/releases/tag/{tag}", **fields)


def _attach_download(result: dict, release: dict, info: dict, timeout: float, allow_host: Callable[[str], bool]) -> dict:
    """Add the asset for this installation's flavour and its published checksum to the result of an available update."""
    name = INSTALLER_ASSET if info.get("installer") else PORTABLE_ASSET
    asset = result["asset"] = _release_asset(release, name, allow_host)
    if asset is None:
        result["reason"] += f" It has no {name} file yet, so it cannot be installed automatically."
        return result
    sums = _release_asset(release, CHECKSUM_ASSET, allow_host)
    if sums is not None:
        data = _fetch(sums["url"], accept="application/octet-stream", timeout=timeout, limit=MAX_CHECKSUM_BYTES,
                      redirect_ok=_asset_redirects(allow_host))
        result["sha256"] = parse_checksums(data.decode("utf-8-sig", "replace")).get(name)
    if result["sha256"] is None:
        result["reason"] += f" The release publishes no checksum for {name}, so it cannot be installed automatically."
    return result


def _check_stable(base: str, timeout: float, info: dict, allow_host: Callable[[str], bool]) -> dict:
    try:
        current = parse_version(info["version"])
    except ValueError as e:
        raise UpdateError(f"This copy of Duckstoken has an unrecognised version ({info['version']}).") from e
    releases = _api_get(f"{base}/repos/{REPO}/releases?per_page=30", timeout)
    if not isinstance(releases, list):
        raise UpdateError("GitHub sent a response that could not be understood.")
    candidates = []
    for rel in releases:
        if isinstance(rel, dict) and not rel.get("draft") and not rel.get("prerelease"):
            try:
                candidates.append((parse_version(str(rel.get("tag_name") or "")), rel))
            except ValueError:
                continue
    if not candidates:
        return _result("stable", reason="No stable release has been published yet.")
    version, release = max(candidates, key=lambda c: _sort_key(c[0]))
    newer = _sort_key(version) > _sort_key(current)
    text = _format_version(version)
    result = _release_result("stable", release, newer, f"Version {text} is available." if newer
                             else f"You are running the latest version ({info['version']}).", version=text)
    return _attach_download(result, release, info, timeout, allow_host) if newer else result


def _check_latest(base: str, timeout: float, info: dict, allow_host: Callable[[str], bool]) -> dict:
    running = str(info.get("build_sha") or "").lower()
    if not running:
        return _result("latest", reason="This copy was not built by the release pipeline, so there is no build to "
                                        "compare with. Use the stable channel or install a released build.")
    try:
        release = _api_get(f"{base}/repos/{REPO}/releases/tags/latest", timeout)
    except _NotFound:
        release = None
    if not isinstance(release, dict) or release.get("draft"):
        return _result("latest", reason="No rolling build has been published yet.")
    body = release.get("body")
    m = _BUILD_LINE.search(body) if isinstance(body, str) else None
    if not m:
        raise UpdateError("The latest build does not say which commit it was built from, so it cannot be compared.")
    sha = m[1].lower()
    newer = sha != running
    result = _release_result("latest", release, newer, f"A newer build ({sha[:7]}) is available." if newer
                             else f"You are running the latest build ({sha[:7]}).", sha=sha)
    return _attach_download(result, release, info, timeout, allow_host) if newer else result


def check(channel: str, *, api_base: str = API_BASE, timeout: float = 10, info: dict | None = None,
          allow_host: Callable[[str], bool] | None = None) -> dict:
    """Ask GitHub whether `channel` offers a newer build than the running one (see the module docstring).

    Returns {"available", "channel", "version", "sha", "notes", "published", "asset": {"name","url","size"} | None,
    "sha256", "html_url", "reason"}; `asset` and `sha256` are what `download` needs. `allow_host(origin)` decides which
    'scheme://host[:port]' asset URLs may use (default: exactly https://github.com). Raises UpdateError with a
    user-readable message when GitHub cannot be reached or answers unexpectedly.
    """
    if channel == "off":
        return _result("off", reason="off")
    if channel not in CHANNELS:
        raise UpdateError(f"Unknown update channel: {channel!r}.")
    info = info or current_info()
    base = api_base.rstrip("/")
    allow = allow_host or _is_github
    return (_check_stable if channel == "stable" else _check_latest)(base, timeout, info, allow)


# -------------------------------------------------------------------------------------------------------- downloading
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(plan: dict, dest_dir, progress: Callable[[int, int], None] = lambda done, total: None, cancel=None,
             *, timeout: float = 30, allow_host: Callable[[str], bool] | None = None) -> Path:
    """Download the asset of a `check()` result into `dest_dir` and return its path.

    Streams to `<name>.part`, then verifies the announced size and the published SHA-256; on any mismatch (or error, or
    when `cancel.is_set()`) the partial file is deleted and UpdateError is raised. A plan without a checksum is refused.
    """
    allow = allow_host or _is_github
    asset = plan.get("asset") if isinstance(plan, dict) else None
    digest = plan.get("sha256") if isinstance(plan, dict) else None
    if not isinstance(asset, dict) or asset.get("name") not in (PORTABLE_ASSET, INSTALLER_ASSET):
        raise UpdateError("There is no installable file in this update.")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise UpdateError("This update has no published checksum, so it cannot be installed automatically.")
    total = asset.get("size")
    if not isinstance(total, int) or isinstance(total, bool) or not 0 < total <= MAX_ASSET_BYTES:
        raise UpdateError("This update announces an unexpected file size, so it was refused.")
    url = _trusted_asset_url(asset.get("url"), allow)

    dest = Path(dest_dir)
    final = dest / asset["name"]
    part = final.with_name(final.name + ".part")
    sha, done = hashlib.sha256(), 0
    try:
        dest.mkdir(parents=True, exist_ok=True)
        if final.is_file() and final.stat().st_size == total and _sha256_file(final) == digest:
            progress(total, total)
            return final
        free = shutil.disk_usage(dest).free
        if free < total:
            raise UpdateError(f"Not enough free disk space for the update: {total / 2**20:.0f} MB needed, "
                              f"{free / 2**20:.0f} MB free.")
        with _open_url(url, accept="application/octet-stream", timeout=timeout,
                       redirect_ok=_asset_redirects(allow)) as resp, open(part, "wb") as out:
            announced = resp.headers.get("Content-Length")
            if announced is not None and announced.strip() != str(total):
                raise UpdateError("The download size does not match the release, so the update was refused.")
            while True:
                if cancel is not None and cancel.is_set():
                    raise UpdateError("The download was cancelled.")
                chunk = resp.read(1 << 18)
                if not chunk:
                    break
                done += len(chunk)
                if done > total:
                    raise UpdateError("The download is larger than announced, so the update was refused.")
                out.write(chunk)
                sha.update(chunk)
                progress(done, total)
        if done != total:
            raise UpdateError("The download ended early. Try again.")
        if sha.hexdigest() != digest:
            raise UpdateError("The downloaded file does not match its published checksum, so it was deleted.")
        os.replace(part, final)
    except (OSError, http.client.HTTPException) as e:
        _discard(part)
        raise UpdateError(f"The download failed ({_describe(e)}).") from e
    except BaseException:
        _discard(part)
        raise
    return final


# --------------------------------------------------------------------------------------------------------- installing
# Static on purpose: every path reaches the script through environment variables, so no path can inject commands and
# no non-ASCII character depends on the console code page. Uses only `goto` (no parenthesised blocks) because the
# paths may contain parentheses such as "Program Files (x86)".
_PORTABLE_HELPER = r'''@echo off
rem Duckstoken self-update helper, rewritten by duckstoken.updater for every update.
rem Environment: DUCK_PID    process to wait for (the running app)
rem              DUCK_TARGET the running exe, replaced by the update (the previous exe is kept as .old)
rem              DUCK_STAGED the new exe, already copied next to DUCK_TARGET
rem              DUCK_RESTART 1 to start the app again afterwards, DUCK_LOG log file
setlocal EnableExtensions
set /a n=0

:wait_exit
tasklist /FI "PID eq %DUCK_PID%" /NH 2>nul | find "%DUCK_PID%" >nul
if errorlevel 1 goto swap
set /a n+=1
if %n% gtr 120 goto still_running
ping -n 2 127.0.0.1 >nul
goto wait_exit

:swap
set /a n=0
:swap_try
if exist "%DUCK_TARGET%.old" del /F /Q "%DUCK_TARGET%.old" >nul 2>&1
move /Y "%DUCK_TARGET%" "%DUCK_TARGET%.old" >nul 2>&1
if not errorlevel 1 goto place
set /a n+=1
if %n% gtr 30 goto abort
ping -n 2 127.0.0.1 >nul
goto swap_try

:place
move /Y "%DUCK_STAGED%" "%DUCK_TARGET%" >nul 2>&1
if errorlevel 1 goto rollback
call :log updated, previous version kept as .old
goto launch

:rollback
call :log could not put the new version in place, restoring the previous one
del /F /Q "%DUCK_TARGET%" >nul 2>&1
move /Y "%DUCK_TARGET%.old" "%DUCK_TARGET%" >nul 2>&1

:abort
call :log update abandoned, the previous version stays in use
del /F /Q "%DUCK_STAGED%" >nul 2>&1
goto launch

:still_running
call :log the app did not exit in time, update abandoned
del /F /Q "%DUCK_STAGED%" >nul 2>&1
exit /b 1

:launch
if "%DUCK_RESTART%"=="1" start "" "%DUCK_TARGET%"
exit /b 0

:log
>>"%DUCK_LOG%" echo %date% %time% %*
exit /b 0
'''


def _is_windows() -> bool:
    return os.name == "nt"


def _detached_env(extra: dict | None = None) -> dict:
    """Environment for a process that must outlive us. PyInstaller's one-file bootloader would otherwise let a relaunched
    exe reuse (and then lose) our temporary extraction folder."""
    env = {**os.environ, **(extra or {}), "PYINSTALLER_RESET_ENVIRONMENT": "1"}
    env.pop("_MEIPASS2", None)
    return env


def _spawn(command: list[str] | str, env: dict, flags: int) -> None:
    try:
        subprocess.Popen(command, env=env, creationflags=flags, close_fds=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        raise UpdateError(f"Could not start the update ({_describe(e)}).") from e


def _cmd_exe() -> str:
    return os.environ.get("COMSPEC") or ntpath.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "cmd.exe")


def apply(downloaded: Path, *, info: dict | None = None, restart: bool = True) -> dict:
    """Start installing a file returned by `download()`; the caller must then exit the app promptly.

    Windows packaged builds only (UpdateError otherwise). Installer build: runs Duckstoken-Setup.exe silently, which
    closes this app and (with `restart`) starts the new one. Portable build: stages the new exe next to the running one
    and starts a hidden helper script that waits for this process to exit, keeps the old exe as `<name>.old`, puts the
    new one in place (restoring the old one if that fails) and relaunches it. Returns
    {"mode": "installer" | "portable", "restart": bool, "helper": path of the helper script | None}.
    """
    info = info or current_info()
    if not _is_windows():
        raise UpdateError("Automatic installation is only available in the Windows app. "
                          "Download the new version from the release page instead.")
    if not info.get("frozen") or not info.get("exe"):
        raise UpdateError("Only the packaged app can update itself. Update this source checkout with git instead.")
    installer = bool(info.get("installer"))
    file = Path(downloaded)
    if file.name != (INSTALLER_ASSET if installer else PORTABLE_ASSET) or file.is_symlink() or not file.is_file():
        raise UpdateError("The downloaded update does not match this installation. Download it again.")
    if installer:
        args = [str(file), "/SILENT", "/CLOSEAPPLICATIONS", "/NORESTART"] + (["/RESTARTAPPLICATIONS"] if restart else [])
        _spawn(args, _detached_env(), _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP)
        return {"mode": "installer", "restart": restart, "helper": None}

    target = Path(info["exe"])
    staged = target.with_name(target.name + ".new")
    updates = paths.data_dir() / "updates"
    helper = updates / "apply_update.cmd"
    try:
        updates.mkdir(parents=True, exist_ok=True)
        helper.write_text(_PORTABLE_HELPER, encoding="ascii", newline="\r\n")
    except OSError as e:
        raise UpdateError(f"Could not prepare the update ({_describe(e)}).") from e
    try:
        shutil.copyfile(file, staged)
    except OSError as e:
        _discard(staged)
        raise UpdateError(f"Duckstoken cannot write to {target.parent} ({_describe(e)}). Move the app to a folder "
                          "you own, or install it with the installer, to update automatically.") from e
    env = _detached_env({"DUCK_PID": str(os.getpid()), "DUCK_TARGET": str(target), "DUCK_STAGED": str(staged),
                         "DUCK_RESTART": "1" if restart else "0", "DUCK_LOG": str(updates / "update.log")})
    try:
        _spawn(f'"{_cmd_exe()}" /d /s /c ""{helper}""', env, _CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP)
    except UpdateError:
        _discard(staged)
        raise
    _discard(file)
    return {"mode": "portable", "restart": restart, "helper": str(helper)}


# ------------------------------------------------------------------------------------------------------ background use
class UpdateChecker:
    """Checks for updates at most once per 24 hours per channel and remembers the outcome across restarts.

    `get_settings()` returns the settings dict (key `update_channel`); state lives in `<data_dir>/update_state.json` and
    only counts for the build that wrote it, so installing an update starts a fresh cycle. `maybe_check` blocks while
    it talks to GitHub: call it from a background thread.
    """

    def __init__(self, get_settings: Callable[[], dict], data_dir, clock: Callable[[], float] = time.time,
                 check_fn: Callable[[str], dict] = check):
        self._get_settings, self._dir, self._clock, self._check_fn = get_settings, Path(data_dir), clock, check_fn
        self._lock = threading.Lock()
        self._checking = False
        self._retry_at: dict[str, float] = {}
        self._downloaded: tuple[str, str] | None = None  # (path, sha256 of the update it belongs to)
        self.last_error = ""
        self._state = self._load()

    def _channel(self) -> str:
        return str(self._get_settings().get("update_channel", "stable"))

    @staticmethod
    def _identity() -> list[str]:
        info = current_info()
        return [info["version"], info["build_sha"]]

    def _load(self) -> dict:
        try:
            raw = json.loads((self._dir / STATE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        channels = raw.get("channels") if isinstance(raw, dict) else None
        if not isinstance(channels, dict):
            return {}
        return {c: e for c, e in channels.items() if c in CHANNELS and isinstance(e, dict)
                and isinstance(e.get("checked"), (int, float)) and isinstance(e.get("result"), dict)}

    def _save(self) -> None:
        tmp = self._dir / (STATE_FILE + ".tmp")
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({"channels": self._state}), encoding="utf-8")
            os.replace(tmp, self._dir / STATE_FILE)
        except OSError:
            _discard(tmp)  # not being able to remember a check only means checking again next time

    def _entry(self, channel: str) -> dict:
        entry = self._state.get(channel) or {}
        return entry if entry.get("build") == self._identity() else {}

    def maybe_check(self, *, force: bool = False) -> dict | None:
        """The fresh result, or None when nothing was checked (channel off, not due yet, already running, or failed;
        a failure is kept in `last_error` and retried after an hour). `force` skips the 24 h wait (a manual check)."""
        channel = self._channel()
        if channel not in ("stable", "latest"):
            return None
        with self._lock:
            now = self._clock()
            due = max(self._entry(channel).get("checked", 0) + CHECK_INTERVAL, self._retry_at.get(channel, 0))
            if self._checking or (now < due and not force):
                return None
            self._checking = True
        try:
            result = self._check_fn(channel)
        except Exception as e:  # noqa: BLE001 - a background check must never take the app down
            with self._lock:
                self._checking = False
                self.last_error = str(e) or e.__class__.__name__
                self._retry_at[channel] = now + ERROR_BACKOFF
            return None
        with self._lock:
            self._checking = False
            self.last_error = ""
            self._retry_at.pop(channel, None)
            self._state[channel] = {"checked": now, "result": result, "build": self._identity()}
            self._save()
        return result

    def status(self) -> dict:
        with self._lock:
            entry = self._entry(self._channel())
            result = entry.get("result")
            kept = self._downloaded
            downloaded = kept[0] if (kept and result and result.get("sha256") == kept[1]
                                     and Path(kept[0]).is_file()) else None
            return {"checking": self._checking, "result": result, "last_checked": float(entry.get("checked", 0)),
                    "last_error": self.last_error, "downloaded": downloaded}

    def download_update(self, progress: Callable[[int, int], None] = lambda done, total: None, cancel=None) -> Path:
        """Download the update found by the last check into `<data_dir>/updates` and remember it for `status()`."""
        result = self.status()["result"]
        if not result or not result.get("available"):
            raise UpdateError("There is no update to download.")
        path = download(result, self._dir / "updates", progress, cancel)
        with self._lock:
            self._downloaded = (str(path), result["sha256"])
        return path
