"""Auto-updater: release discovery, verified downloads, install helpers and the throttled background checker.

Everything runs against a local fake of the GitHub API / release downloads; nothing touches the internet and nothing
is executed (subprocess.Popen is mocked wherever an installer or helper script would be launched).
"""
import hashlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DUCKSTOKEN_HOME", tempfile.mkdtemp(prefix="duck-upd-"))

from duckstoken import updater  # noqa: E402
from duckstoken.updater import UpdateError  # noqa: E402

REPO = "acme/Quack"
RUNNING_SHA = "a" * 40
NEW_SHA = "b" * 40
INFO = {"version": "0.1.0", "build_sha": "", "build_time": "", "installer": False, "frozen": False, "exe": None}

_proxy_patch = mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})


def setUpModule():
    _proxy_patch.start()  # a developer's HTTP proxy must not swallow the loopback fake


def tearDownModule():
    _proxy_patch.stop()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload(tag: str, kind: str = "portable", size: int = 4000) -> bytes:
    return (f"MZ-{tag}-{kind}-".encode() * size)[:size]


def closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):  # clients that give up on purpose
            super().handle_error(request, client_address)


class FakeGitHub:
    """Just enough of api.github.com and the release download hosts, served from 127.0.0.1."""

    def __init__(self, repo: str = REPO):
        self.repo = repo
        self.releases: list[dict] = []
        self.latest: dict | None = None
        self.files: dict[str, bytes] = {}
        self.status: dict[str, tuple[int, dict]] = {}
        self.redirects: dict[str, str] = {}
        self.custom: dict = {}
        self.requests: list[str] = []
        self.stall = 0.0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                fake.serve(self)

        self.httpd = QuietServer(("127.0.0.1", 0), Handler)
        self.origin = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def allow(self, origin: str) -> bool:
        return origin == self.origin

    @staticmethod
    def reply(h, code: int, body: bytes, headers: dict | None = None, content_type="application/json"):
        h.send_response(code)
        h.send_header("Content-Type", content_type)
        h.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            h.send_header(k, v)
        h.end_headers()
        h.wfile.write(body)

    def serve(self, h):
        path = urllib.parse.urlsplit(h.path).path
        self.requests.append(path)
        if self.stall:
            time.sleep(self.stall)
        if path in self.redirects:
            h.send_response(302)
            h.send_header("Location", self.redirects[path])
            h.send_header("Content-Length", "0")
            h.end_headers()
        elif path in self.status:
            code, headers = self.status[path]
            self.reply(h, code, b'{"message": "nope"}', headers)
        elif path in self.custom:
            self.custom[path](h)
        elif path == f"/repos/{self.repo}/releases":
            self.reply(h, 200, json.dumps(self.releases).encode())
        elif path == f"/repos/{self.repo}/releases/tags/latest":
            if self.latest is None:
                self.reply(h, 404, b'{"message": "Not Found"}')
            else:
                self.reply(h, 200, json.dumps(self.latest).encode())
        elif path in self.files:
            self.reply(h, 200, self.files[path], content_type="application/octet-stream")
        else:
            self.reply(h, 404, b'{"message": "Not Found"}')

    def asset_path(self, tag: str, name: str) -> str:
        return f"/{self.repo}/releases/download/{tag}/{name}"

    def release(self, tag: str, *, body: str = "", prerelease: bool = False, draft: bool = False,
                assets: dict | None = None, checksums=True, published: str = "2026-01-02T03:04:05Z") -> dict:
        """Register a release's files and return its API representation. `checksums` True = a proper SHA256SUMS.txt,
        False = none, a str = that exact text."""
        if assets is None:
            assets = {"Duckstoken.exe": payload(tag, "portable"), "Duckstoken-Setup.exe": payload(tag, "setup")}
        if checksums is True:
            checksums = "".join(f"{sha(data)}  {name}\r\n" for name, data in assets.items())
        files = dict(assets)
        if checksums:
            files["SHA256SUMS.txt"] = checksums.encode()
        listing = []
        for name, data in files.items():
            self.files[self.asset_path(tag, name)] = data
            listing.append({"name": name, "size": len(data), "state": "uploaded",
                            "browser_download_url": self.origin + self.asset_path(tag, name)})
        return {"tag_name": tag, "draft": draft, "prerelease": prerelease, "body": body, "published_at": published,
                "html_url": "https://example.invalid/ignored", "assets": listing}


class FakeTestCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(updater, "REPO", REPO)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.fake = FakeGitHub()
        self.addCleanup(self.fake.close)
        tmp = tempfile.TemporaryDirectory(prefix="duck-upd-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def check(self, channel, **info):
        return updater.check(channel, api_base=self.fake.origin, info={**INFO, **info}, allow_host=self.fake.allow,
                             timeout=5)

    def plan(self, tag="v0.2.0", **info) -> dict:
        self.fake.releases = [self.fake.release(tag)]
        plan = self.check("stable", **info)
        self.assertTrue(plan["available"], plan)
        return plan

    def download(self, plan, **kw):
        kw.setdefault("allow_host", self.fake.allow)
        return updater.download(plan, self.tmp / "dl", **kw)


# ============================================================================================ versions & build info
class VersionTests(unittest.TestCase):
    def test_parse_and_pad(self):
        self.assertEqual(updater.parse_version("v1.2.3"), (1, 2, 3))
        self.assertEqual(updater.parse_version("1.2"), (1, 2, 0))
        self.assertEqual(updater.parse_version(" V7 "), (7, 0, 0))
        self.assertEqual(updater.parse_version("v1.2.3.4"), (1, 2, 3, 4))

    def test_junk_is_rejected(self):
        for junk in ("", "latest", "v", "v1.x", "1..2", "1.2.3-rc1", "1.2.3.4.5", "v-1.0", "1.2.3\n4", None, 3):
            with self.assertRaises(ValueError, msg=repr(junk)):
                updater.parse_version(junk)

    def test_numeric_not_lexicographic_ordering(self):
        self.assertGreater(updater._sort_key(updater.parse_version("v0.10.0")), updater._sort_key(updater.parse_version("v0.9.9")))
        self.assertEqual(updater._sort_key(updater.parse_version("1.2")), updater._sort_key(updater.parse_version("1.2.0.0")))


class CurrentInfoTests(unittest.TestCase):
    def test_source_checkout_defaults(self):
        info = updater.current_info()
        self.assertEqual(set(info), {"version", "build_sha", "build_time", "installer", "frozen", "exe"})
        self.assertEqual(info["version"], updater.__version__)
        self.assertEqual((info["build_sha"], info["build_time"], info["installer"]), ("", "", False))
        self.assertFalse(info["frozen"])
        self.assertIsNone(info["exe"])

    def test_stamped_frozen_build(self):
        with mock.patch.object(updater._build, "BUILD_SHA", RUNNING_SHA), \
                mock.patch.object(updater._build, "BUILD_TIME", "2026-01-01T00:00:00Z"), \
                mock.patch.object(updater._build, "INSTALLER", True), \
                mock.patch.object(updater.bundle, "frozen", return_value=True), \
                mock.patch.object(updater.sys, "executable", r"C:\Apps\Duckstoken.exe"):
            info = updater.current_info()
        self.assertEqual(info, {"version": updater.__version__, "build_sha": RUNNING_SHA,
                                "build_time": "2026-01-01T00:00:00Z", "installer": True, "frozen": True,
                                "exe": r"C:\Apps\Duckstoken.exe"})


# ================================================================================================== trusting URLs
class TrustedUrlTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(updater, "REPO", REPO)
        patcher.start()
        self.addCleanup(patcher.stop)

    def trusted(self, url):
        return updater._trusted_asset_url(url, updater._is_github)

    def test_accepts_release_downloads_of_the_repo(self):
        url = "https://github.com/acme/Quack/releases/download/v1.0.0/Duckstoken.exe"
        self.assertEqual(self.trusted(url), url)
        self.assertEqual(self.trusted(url.replace("acme/Quack", "ACME/quack")), url.replace("acme/Quack", "ACME/quack"))

    def test_rejects_everything_else(self):
        good = "github.com/acme/Quack/releases/download/v1.0.0/Duckstoken.exe"
        for bad in (
            "http://" + good,                                                    # no TLS
            "https://github.com.evil.example/acme/Quack/releases/download/v1/x.exe",  # look-alike host
            "https://evil.example/acme/Quack/releases/download/v1/x.exe",
            "https://github.com@evil.example/acme/Quack/releases/download/v1/x.exe",  # credentials trick
            "https://user:pw@" + good,
            "https://github.com:8443/acme/Quack/releases/download/v1/x.exe",
            "https://github.com/other/repo/releases/download/v1/x.exe",          # another repository
            "https://github.com/acme/Quack/archive/main.zip",                    # not a release asset
            "https://github.com/acme/Quack/releases/download/../../../evil/x.exe",
            "https://github.com/acme/Quack/releases/download/%2e%2e/%2e%2e/x.exe",
            "https://objects.githubusercontent.com/acme/Quack/releases/download/v1/x.exe",
            "file:///C:/Windows/System32/cmd.exe", "ftp://github.com/acme/Quack/releases/download/v1/x.exe",
            "", None, 5,
        ):
            with self.assertRaises(UpdateError, msg=repr(bad)):
                self.trusted(bad)

    def test_allow_host_widens_only_the_origin(self):
        allow = lambda origin: origin == "http://127.0.0.1:9"  # noqa: E731
        ok = "http://127.0.0.1:9/acme/Quack/releases/download/v1/x.exe"
        self.assertEqual(updater._trusted_asset_url(ok, allow), ok)
        with self.assertRaises(UpdateError):
            updater._trusted_asset_url("http://127.0.0.1:9/other/repo/releases/download/v1/x.exe", allow)


class ChecksumParsingTests(unittest.TestCase):
    A, B = "a" * 64, "B" * 64

    def test_formats(self):
        text = f"{self.A}  Duckstoken.exe\r\n{self.B} *Duckstoken-Setup.exe\n\n# comment\nnot a checksum line\n"
        self.assertEqual(updater.parse_checksums(text), {"Duckstoken.exe": self.A, "Duckstoken-Setup.exe": "b" * 64})

    def test_short_or_non_hex_hashes_are_ignored(self):
        self.assertEqual(updater.parse_checksums(f"{'a' * 63}  x.exe\n{'g' * 64}  y.exe\n"), {})

    def test_conflicting_duplicates_are_an_error(self):
        with self.assertRaises(UpdateError):
            updater.parse_checksums(f"{self.A}  x.exe\n{'c' * 64}  x.exe\n")
        self.assertEqual(updater.parse_checksums(f"{self.A}  x.exe\n{self.A}  x.exe\n"), {"x.exe": self.A})


# =================================================================================================== stable channel
class CheckStableTests(FakeTestCase):
    def test_picks_the_newest_stable_release_numerically(self):
        f = self.fake
        f.releases = [f.release("v0.9.0"), f.release("v0.10.0"), f.release("v0.2.0"),
                      f.release("v9.0.0", prerelease=True), f.release("v8.0.0", draft=True),
                      f.release("latest", prerelease=True), f.release("nightly-build"), f.release("v0.10.0-rc1")]
        r = self.check("stable")
        self.assertTrue(r["available"])
        self.assertEqual((r["channel"], r["version"], r["sha"]), ("stable", "0.10.0", None))
        self.assertEqual(r["asset"]["name"], "Duckstoken.exe")
        self.assertEqual(r["html_url"], "https://github.com/acme/Quack/releases/tag/v0.10.0")
        self.assertIn("0.10.0", r["reason"])

    def test_up_to_date_and_older(self):
        f = self.fake
        f.releases = [f.release("v0.1.0"), f.release("v0.0.9")]
        r = self.check("stable")
        self.assertFalse(r["available"])
        self.assertEqual((r["version"], r["asset"], r["sha256"]), ("0.1.0", None, None))
        self.assertIn("latest version", r["reason"])
        self.assertFalse(self.check("stable", version="0.5.0")["available"])
        self.assertEqual(f.requests.count(f.asset_path("v0.1.0", "SHA256SUMS.txt")), 0)  # nothing to install, nothing fetched

    def test_two_and_four_part_versions(self):
        f = self.fake
        f.releases = [f.release("v0.1")]
        self.assertFalse(self.check("stable")["available"])  # 0.1 == 0.1.0
        f.releases = [f.release("v0.1.0.1")]
        self.assertTrue(self.check("stable")["available"])

    def test_no_releases_is_not_an_error(self):
        r = self.check("stable")
        self.assertFalse(r["available"])
        self.assertIn("No stable release", r["reason"])
        self.fake.releases = [self.fake.release("latest", prerelease=True)]
        self.assertFalse(self.check("stable")["available"])

    def test_result_shape_and_notes_trimmed(self):
        self.fake.releases = [self.fake.release("v1.0.0", body="  " + "x" * 5000, published="2026-02-03T04:05:06Z")]
        r = self.check("stable")
        self.assertEqual(set(r), {"available", "channel", "version", "sha", "notes", "published", "asset", "sha256",
                                  "html_url", "reason"})
        self.assertEqual(len(r["notes"]), updater.NOTES_LIMIT)
        self.assertEqual(r["published"], "2026-02-03T04:05:06Z")
        data = payload("v1.0.0")
        self.assertEqual(r["asset"], {"name": "Duckstoken.exe", "size": len(data),
                                      "url": self.fake.origin + self.fake.asset_path("v1.0.0", "Duckstoken.exe")})
        self.assertEqual(r["sha256"], sha(data))

    def test_missing_body_and_dates_are_tolerated(self):
        rel = self.fake.release("v1.0.0")
        rel["body"] = None
        del rel["published_at"]
        self.fake.releases = [rel]
        r = self.check("stable")
        self.assertEqual((r["notes"], r["published"]), ("", ""))

    def test_installer_build_gets_the_installer(self):
        self.fake.releases = [self.fake.release("v1.0.0")]
        r = self.check("stable", installer=True)
        self.assertEqual(r["asset"]["name"], "Duckstoken-Setup.exe")
        self.assertEqual(r["sha256"], sha(payload("v1.0.0", "setup")))

    def test_checksum_file_variants(self):
        f = self.fake
        exe = payload("v1.0.0")
        cases = {
            "no SHA256SUMS.txt": (False, None),
            "other file only": (f"{'a' * 64}  Something-Else.exe\n", None),
            "garbage": ("<html>rate limited</html>", None),
            "binary marker, LF": (f"{sha(exe)} *Duckstoken.exe\n", sha(exe)),
            "BOM and upper case": ("\ufeff" + f"{sha(exe).upper()}  Duckstoken.exe\r\n", sha(exe)),
        }
        for label, (checksums, expected) in cases.items():
            with self.subTest(label):
                f.releases = [f.release("v1.0.0", checksums=checksums)]
                r = self.check("stable")
                self.assertTrue(r["available"])
                self.assertEqual(r["sha256"], expected)
                if expected is None:
                    self.assertIn("checksum", r["reason"])
                    self.assertIsNotNone(r["asset"])

    def test_conflicting_checksum_file_is_an_error(self):
        self.fake.releases = [self.fake.release("v1.0.0", checksums=f"{'a' * 64}  Duckstoken.exe\n{'b' * 64}  Duckstoken.exe\n")]
        with self.assertRaises(UpdateError):
            self.check("stable")

    def test_release_without_the_needed_exe(self):
        f = self.fake
        f.releases = [f.release("v1.0.0", assets={"Duckstoken-Setup.exe": payload("v1.0.0", "setup")})]
        r = self.check("stable")
        self.assertTrue(r["available"])
        self.assertEqual((r["asset"], r["sha256"]), (None, None))
        self.assertIn("Duckstoken.exe", r["reason"])
        self.assertEqual(self.check("stable", installer=True)["asset"]["name"], "Duckstoken-Setup.exe")

    def test_assets_still_uploading_are_ignored(self):
        rel = self.fake.release("v1.0.0")
        for a in rel["assets"]:
            a["state"] = "open"
        self.fake.releases = [rel]
        self.assertIsNone(self.check("stable")["asset"])

    def test_bad_asset_sizes_are_refused(self):
        for size in (0, -5, "100", None, True, updater.MAX_ASSET_BYTES + 1):
            with self.subTest(size=size):
                rel = self.fake.release("v1.0.0")
                rel["assets"][0]["size"] = size
                self.fake.releases = [rel]
                with self.assertRaises(UpdateError):
                    self.check("stable")

    def test_untrusted_asset_urls_are_refused(self):
        for url in ("https://evil.example/acme/Quack/releases/download/v1.0.0/Duckstoken.exe",
                    "http://github.com/acme/Quack/releases/download/v1.0.0/Duckstoken.exe",
                    "https://github.com/other/repo/releases/download/v1.0.0/Duckstoken.exe", None):
            with self.subTest(url=url):
                rel = self.fake.release("v1.0.0")
                rel["assets"][0]["browser_download_url"] = url
                self.fake.releases = [rel]
                with self.assertRaises(UpdateError) as cm:
                    self.check("stable")
                self.assertIn("not trusted", str(cm.exception))

    def test_default_trust_rejects_a_local_server(self):
        self.fake.releases = [self.fake.release("v1.0.0")]
        with self.assertRaises(UpdateError):
            updater.check("stable", api_base=self.fake.origin, info=INFO, timeout=5)

    def test_untrusted_checksum_url_is_refused(self):
        rel = self.fake.release("v1.0.0")
        rel["assets"][-1]["browser_download_url"] = "https://evil.example/acme/Quack/releases/download/v1.0.0/SHA256SUMS.txt"
        self.fake.releases = [rel]
        with self.assertRaises(UpdateError):
            self.check("stable")

    def test_off_makes_no_request(self):
        r = updater.check("off", api_base=self.fake.origin)
        self.assertEqual((r["available"], r["reason"], r["channel"]), (False, "off", "off"))
        self.assertEqual(self.fake.requests, [])

    def test_unknown_channel(self):
        with self.assertRaises(UpdateError):
            updater.check("beta", api_base=self.fake.origin)
        self.assertEqual(self.fake.requests, [])

    def test_unparseable_running_version(self):
        self.fake.releases = [self.fake.release("v1.0.0")]
        with self.assertRaises(UpdateError):
            self.check("stable", version="0.1.0.dev1")


# ==================================================================================================== latest channel
class CheckLatestTests(FakeTestCase):
    def latest(self, sha_text=NEW_SHA, **kw):
        self.fake.latest = self.fake.release("latest", prerelease=True,
                                             body=f"Automatic build.\r\n\r\nBuild: {sha_text}\r\n", **kw)

    def test_new_build_is_available(self):
        self.latest()
        r = self.check("latest", build_sha=RUNNING_SHA)
        self.assertTrue(r["available"])
        self.assertEqual((r["channel"], r["sha"], r["version"]), ("latest", NEW_SHA, None))
        self.assertEqual(r["asset"]["name"], "Duckstoken.exe")
        self.assertEqual(r["sha256"], sha(payload("latest")))
        self.assertIn("bbbbbbb", r["reason"])
        self.assertEqual(self.check("latest", build_sha=RUNNING_SHA, installer=True)["asset"]["name"], "Duckstoken-Setup.exe")

    def test_same_build_is_not_available_regardless_of_case(self):
        self.latest(NEW_SHA.upper())
        r = self.check("latest", build_sha=NEW_SHA)
        self.assertFalse(r["available"])
        self.assertEqual((r["sha"], r["asset"]), (NEW_SHA, None))
        self.assertTrue(self.check("latest", build_sha=NEW_SHA.upper().replace("B", "C"))["available"])

    def test_source_checkout_is_never_offered_a_rolling_build_and_costs_no_request(self):
        self.latest()
        r = self.check("latest", build_sha="")
        self.assertFalse(r["available"])
        self.assertIn("release pipeline", r["reason"])
        self.assertEqual(self.fake.requests, [])

    def test_no_rolling_release_yet(self):
        r = self.check("latest", build_sha=RUNNING_SHA)
        self.assertFalse(r["available"])
        self.assertIn("No rolling build", r["reason"])
        self.latest(draft=True)
        self.assertFalse(self.check("latest", build_sha=RUNNING_SHA)["available"])

    def test_notes_without_a_build_line_are_an_error(self):
        for body in (None, "", "Build: abc123", f"Rebuild: {NEW_SHA}", f"see Build: {NEW_SHA}", f"Build: {NEW_SHA}0"):
            with self.subTest(body=body):
                self.fake.latest = self.fake.release("latest", prerelease=True, body=body)
                with self.assertRaises(UpdateError):
                    self.check("latest", build_sha=RUNNING_SHA)

    def test_build_line_anywhere_in_the_notes(self):
        self.fake.latest = self.fake.release("latest", prerelease=True, body=f"first line\n  Build: {NEW_SHA}  \nlast")
        self.assertEqual(self.check("latest", build_sha=RUNNING_SHA)["sha"], NEW_SHA)

    def test_stable_releases_are_ignored_on_this_channel(self):
        self.fake.releases = [self.fake.release("v9.0.0")]
        self.assertFalse(self.check("latest", build_sha=RUNNING_SHA)["available"])


# ============================================================================================ network problems
class CheckErrorTests(FakeTestCase):
    def assertFriendly(self, cm, *needles):
        text = str(cm.exception)
        for needle in needles:
            self.assertIn(needle, text)
        self.assertNotIn("Traceback", text)
        self.assertNotIn("urlopen", text)

    def api_path(self):
        return f"/repos/{REPO}/releases"

    def test_server_unreachable(self):
        with self.assertRaises(UpdateError) as cm:
            updater.check("stable", api_base=f"http://127.0.0.1:{closed_port()}", info=INFO, timeout=5)
        self.assertFriendly(cm, "Could not reach GitHub")

    def test_http_errors(self):
        self.fake.status[self.api_path()] = (500, {})
        with self.assertRaises(UpdateError) as cm:
            self.check("stable")
        self.assertFriendly(cm, "HTTP 500")

    def test_missing_repository_is_a_friendly_error_on_stable(self):
        self.fake.status[self.api_path()] = (404, {})
        with self.assertRaises(UpdateError) as cm:
            self.check("stable")
        self.assertFriendly(cm, "404")

    def test_rate_limit(self):
        for code, headers in ((403, {"X-RateLimit-Remaining": "0"}), (429, {})):
            with self.subTest(code=code):
                self.fake.status[self.api_path()] = (code, headers)
                with self.assertRaises(UpdateError) as cm:
                    self.check("stable")
                self.assertFriendly(cm, "limiting requests")

    def test_plain_403_is_not_called_a_rate_limit(self):
        self.fake.status[self.api_path()] = (403, {})
        with self.assertRaises(UpdateError) as cm:
            self.check("stable")
        self.assertFriendly(cm, "HTTP 403")
        self.assertNotIn("limiting", str(cm.exception))

    def test_garbage_responses(self):
        for label, body in (("not json", b"<html>oops</html>"), ("wrong shape", b'{"releases": []}'), ("empty", b"")):
            with self.subTest(label):
                self.fake.custom[self.api_path()] = partial(self.fake.reply, code=200, body=body)
                with self.assertRaises(UpdateError) as cm:
                    self.check("stable")
                self.assertFriendly(cm, "could not be understood")

    def test_junk_entries_in_the_release_list_are_skipped(self):
        good = self.fake.release("v1.0.0")
        self.fake.releases = [None, 7, "x", {"tag_name": None}, {"tag_name": ["v9.0.0"]}, {}, good]
        self.assertEqual(self.check("stable")["version"], "1.0.0")

    def test_oversized_response(self):
        self.fake.releases = [self.fake.release("v1.0.0", body="x" * 5000)]
        with mock.patch.object(updater, "MAX_JSON_BYTES", 1000), self.assertRaises(UpdateError) as cm:
            self.check("stable")
        self.assertFriendly(cm, "unexpectedly large")

    def test_timeout(self):
        self.fake.stall = 1.0
        with self.assertRaises(UpdateError) as cm:
            updater.check("stable", api_base=self.fake.origin, info=INFO, timeout=0.2)
        self.assertFriendly(cm, "Could not reach GitHub")

    def test_api_redirect_to_another_origin_is_refused(self):
        self.fake.redirects[self.api_path()] = "http://127.0.0.1:9/elsewhere"
        with self.assertRaises(UpdateError) as cm:
            self.check("stable")
        self.assertFriendly(cm, "not trusted")
        self.assertNotIn("/elsewhere", self.fake.requests)

    def test_api_redirect_within_the_origin_is_followed(self):
        self.fake.releases = [self.fake.release("v1.0.0")]
        self.fake.redirects["/old/releases"] = self.fake.origin + self.api_path()
        data = updater._api_get(self.fake.origin + "/old/releases", 5)
        self.assertEqual(data[0]["tag_name"], "v1.0.0")

    def test_checksum_download_failure(self):
        self.fake.releases = [self.fake.release("v1.0.0")]
        self.fake.status[self.fake.asset_path("v1.0.0", "SHA256SUMS.txt")] = (500, {})
        with self.assertRaises(UpdateError) as cm:
            self.check("stable")
        self.assertFriendly(cm, "HTTP 500")

    def test_oversized_checksum_file(self):
        self.fake.releases = [self.fake.release("v1.0.0", checksums="x" * 5000)]
        with mock.patch.object(updater, "MAX_CHECKSUM_BYTES", 1000), self.assertRaises(UpdateError):
            self.check("stable")


# ====================================================================================================== downloading
class DownloadTests(FakeTestCase):
    def part(self):
        return self.tmp / "dl" / "Duckstoken.exe.part"

    def final(self):
        return self.tmp / "dl" / "Duckstoken.exe"

    def assertNothingLeft(self):
        self.assertFalse(self.part().exists())
        self.assertFalse(self.final().exists())

    def test_downloads_verifies_and_reports_progress(self):
        data = bytes(range(256)) * 4096  # 1 MiB: several chunks
        self.fake.releases = [self.fake.release("v0.2.0", assets={"Duckstoken.exe": data})]
        plan = self.check("stable")
        calls = []
        path = self.download(plan, progress=lambda done, total: calls.append((done, total)))
        self.assertEqual(path, self.final())
        self.assertEqual(path.read_bytes(), data)
        self.assertFalse(self.part().exists())
        self.assertGreater(len(calls), 1)
        self.assertEqual([c[0] for c in calls], sorted(c[0] for c in calls))
        self.assertEqual(calls[-1], (len(data), len(data)))

    def test_installer_flavour_keeps_its_own_name(self):
        plan = self.plan(installer=True)
        self.assertEqual(self.download(plan).name, "Duckstoken-Setup.exe")

    def test_tampered_download_is_rejected_and_deleted(self):
        plan = self.plan()
        tampered = bytearray(self.fake.files[self.fake.asset_path("v0.2.0", "Duckstoken.exe")])
        tampered[10] ^= 0xFF
        self.fake.files[self.fake.asset_path("v0.2.0", "Duckstoken.exe")] = bytes(tampered)
        with self.assertRaises(UpdateError) as cm:
            self.download(plan)
        self.assertIn("checksum", str(cm.exception))
        self.assertNothingLeft()

    def test_stale_file_with_the_same_name_is_replaced(self):
        plan = self.plan()
        self.final().parent.mkdir(parents=True)
        self.final().write_bytes(b"stale")
        self.assertEqual(self.download(plan).read_bytes(), payload("v0.2.0"))

    def test_truncated_download(self):
        plan = self.plan()
        path = self.fake.asset_path("v0.2.0", "Duckstoken.exe")
        data = self.fake.files[path]

        def cut_short(h):
            h.send_response(200)
            h.send_header("Content-Length", str(len(data)))
            h.end_headers()
            h.wfile.write(data[: len(data) // 2])
            h.close_connection = True

        self.fake.custom[path] = cut_short
        with self.assertRaises(UpdateError) as cm:
            self.download(plan)
        self.assertIn("ended early", str(cm.exception))
        self.assertNothingLeft()

    def test_announced_length_must_match_the_release(self):
        plan = self.plan()
        self.fake.files[self.fake.asset_path("v0.2.0", "Duckstoken.exe")] += b"extra"
        with self.assertRaises(UpdateError) as cm:
            self.download(plan)
        self.assertIn("does not match", str(cm.exception))
        self.assertNothingLeft()

    def test_stream_longer_than_announced(self):
        plan = self.plan()
        path = self.fake.asset_path("v0.2.0", "Duckstoken.exe")
        data = self.fake.files[path]

        def too_much(h):  # no Content-Length: the body just runs on until the connection closes
            h.send_response(200)
            h.end_headers()
            h.wfile.write(data + b"surplus" * 100)
            h.close_connection = True

        self.fake.custom[path] = too_much
        with self.assertRaises(UpdateError) as cm:
            self.download(plan)
        self.assertIn("larger than announced", str(cm.exception))
        self.assertNothingLeft()

    def test_http_error_while_downloading(self):
        plan = self.plan()
        self.fake.status[self.fake.asset_path("v0.2.0", "Duckstoken.exe")] = (503, {})
        with self.assertRaises(UpdateError) as cm:
            self.download(plan)
        self.assertIn("HTTP 503", str(cm.exception))
        self.assertNothingLeft()

    def test_cancel_stops_and_cleans_up(self):
        data = bytes(range(256)) * 4096
        self.fake.releases = [self.fake.release("v0.2.0", assets={"Duckstoken.exe": data})]
        plan = self.check("stable")
        cancel = threading.Event()
        with self.assertRaises(UpdateError) as cm:
            self.download(plan, cancel=cancel, progress=lambda done, total: cancel.set())
        self.assertIn("cancelled", str(cm.exception))
        self.assertNothingLeft()

    def test_plan_without_checksum_is_refused_before_any_request(self):
        self.fake.releases = [self.fake.release("v0.2.0", checksums=False)]
        plan = self.check("stable")
        self.assertIsNone(plan["sha256"])
        requests = len(self.fake.requests)
        with self.assertRaises(UpdateError) as cm:
            self.download(plan)
        self.assertIn("checksum", str(cm.exception))
        self.assertEqual(len(self.fake.requests), requests)

    def test_malformed_plans_are_refused(self):
        good = self.plan()
        digest = good["sha256"]
        bad_plans = {
            "no asset": {**good, "asset": None},
            "not a dict": None,
            "foreign file name": {**good, "asset": {**good["asset"], "name": "evil.exe"}},
            "path traversal name": {**good, "asset": {**good["asset"], "name": "../Duckstoken.exe"}},
            "short hash": {**good, "sha256": digest[:-1]},
            "upper-case hash": {**good, "sha256": digest.upper()},
            "huge size": {**good, "asset": {**good["asset"], "size": updater.MAX_ASSET_BYTES + 1}},
            "zero size": {**good, "asset": {**good["asset"], "size": 0}},
            "foreign url": {**good, "asset": {**good["asset"], "url": "https://evil.example/acme/Quack/releases/download/v/Duckstoken.exe"}},
        }
        requests = len(self.fake.requests)
        for label, plan in bad_plans.items():
            with self.subTest(label), self.assertRaises(UpdateError):
                self.download(plan)
        self.assertEqual(len(self.fake.requests), requests)

    def test_default_trust_refuses_the_local_server(self):
        plan = self.plan()
        requests = len(self.fake.requests)
        with self.assertRaises(UpdateError) as cm:
            updater.download(plan, self.tmp / "dl")
        self.assertIn("not trusted", str(cm.exception))
        self.assertEqual(len(self.fake.requests), requests)

    def test_redirect_within_trusted_origins_is_followed(self):
        plan = self.plan()
        real = self.fake.asset_path("v0.2.0", "Duckstoken.exe")
        self.fake.files["/cdn/blob"] = self.fake.files[real]
        self.fake.redirects[real] = self.fake.origin + "/cdn/blob"
        self.assertEqual(self.download(plan).read_bytes(), self.fake.files["/cdn/blob"])

    def test_redirect_to_an_untrusted_origin_is_refused(self):
        plan = self.plan()
        self.fake.redirects[self.fake.asset_path("v0.2.0", "Duckstoken.exe")] = "http://127.0.0.1:9/evil.exe"
        with self.assertRaises(UpdateError) as cm:
            self.download(plan)
        self.assertIn("not trusted", str(cm.exception))
        self.assertNothingLeft()

    def test_github_download_hosts_are_followed_by_default(self):
        redirects = updater._asset_redirects(updater._is_github)
        for origin in ("https://github.com", "https://objects.githubusercontent.com",
                       "https://release-assets.githubusercontent.com"):
            self.assertTrue(redirects(origin), origin)
        for origin in ("http://github.com", "https://githubusercontent.com", "https://evil.example",
                       "https://objects.githubusercontent.com.evil.example", "http://127.0.0.1:9"):
            self.assertFalse(redirects(origin), origin)

    def test_verified_file_is_reused_without_a_request(self):
        plan = self.plan()
        self.download(plan)
        requests = len(self.fake.requests)
        calls = []
        self.assertEqual(self.download(plan, progress=lambda done, total: calls.append((done, total))), self.final())
        self.assertEqual(len(self.fake.requests), requests)
        self.assertEqual(calls, [(plan["asset"]["size"],) * 2])

    def test_not_enough_disk_space(self):
        plan = self.plan()
        with mock.patch.object(updater.shutil, "disk_usage", return_value=mock.Mock(free=10)):
            with self.assertRaises(UpdateError) as cm:
                self.download(plan)
        self.assertIn("disk space", str(cm.exception))
        self.assertNothingLeft()

    def test_unwritable_destination_is_a_friendly_error(self):
        plan = self.plan()
        blocker = self.tmp / "dl"
        blocker.write_text("a file where the folder should be")
        with self.assertRaises(UpdateError) as cm:
            self.download(plan)
        self.assertIn("The download failed", str(cm.exception))

    def test_connection_failure_while_downloading(self):
        plan = self.plan()
        origin = f"http://127.0.0.1:{closed_port()}"
        plan["asset"]["url"] = origin + self.fake.asset_path("v0.2.0", "Duckstoken.exe")
        with self.assertRaises(UpdateError) as cm:
            self.download(plan, timeout=2, allow_host=lambda o: o == origin)
        self.assertIn("Could not reach GitHub", str(cm.exception))
        self.assertNothingLeft()


# ================================================================================================== installing
class HelperScriptTests(unittest.TestCase):
    script = updater._PORTABLE_HELPER

    def test_waits_for_the_app_process(self):
        self.assertIn('tasklist /FI "PID eq %DUCK_PID%" /NH 2>nul | find "%DUCK_PID%"', self.script)
        self.assertLess(self.script.index(":wait_exit"), self.script.index(":swap"))
        self.assertIn("goto still_running", self.script)  # gives up instead of replacing a running app

    def test_keeps_a_backup_and_rolls_back(self):
        self.assertIn('move /Y "%DUCK_TARGET%" "%DUCK_TARGET%.old"', self.script)
        self.assertIn('move /Y "%DUCK_STAGED%" "%DUCK_TARGET%"', self.script)
        self.assertIn("if errorlevel 1 goto rollback", self.script)
        rollback = self.script[self.script.index(":rollback"):self.script.index(":abort")]
        self.assertIn('move /Y "%DUCK_TARGET%.old" "%DUCK_TARGET%"', rollback)

    def test_retries_while_the_old_exe_is_locked_and_restarts_on_request(self):
        self.assertIn(":swap_try", self.script)
        self.assertIn("if %n% gtr 30 goto abort", self.script)
        self.assertIn('if "%DUCK_RESTART%"=="1" start "" "%DUCK_TARGET%"', self.script)

    def test_every_exit_path_cleans_up_the_staged_file(self):
        for label in (":abort", ":still_running"):
            section = self.script[self.script.index(label):]
            self.assertIn('del /F /Q "%DUCK_STAGED%"', section.split("\n\n")[0], label)

    def test_script_is_static_ascii_and_free_of_parenthesised_blocks(self):
        self.script.encode("ascii")
        self.assertNotIn("%1", self.script)
        for line in self.script.splitlines():
            if not line.lower().startswith("rem "):
                self.assertNotIn("(", line)
                self.assertNotIn(")", line)
        for var in ("DUCK_PID", "DUCK_TARGET", "DUCK_STAGED", "DUCK_RESTART", "DUCK_LOG"):
            self.assertIn(f"%{var}%", self.script)


class ApplyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="duck-apply-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.home = self.tmp / "home"
        self.app = self.tmp / "Program Files (x86)" / "Duckstoken"
        self.app.mkdir(parents=True)
        self.exe = self.app / "Duckstoken.exe"
        self.exe.write_bytes(b"old build")
        self.download_dir = self.tmp / "dl"
        self.download_dir.mkdir()
        for patcher in (mock.patch.dict(os.environ, {"DUCKSTOKEN_HOME": str(self.home)}),
                        mock.patch.object(updater, "_is_windows", return_value=True),
                        mock.patch.dict(os.environ, {"COMSPEC": r"C:\Windows\System32\cmd.exe"})):
            patcher.start()
            self.addCleanup(patcher.stop)
        popen = mock.patch.object(updater.subprocess, "Popen")
        self.popen = popen.start()
        self.addCleanup(popen.stop)

    def info(self, **kw):
        return {**INFO, "frozen": True, "exe": str(self.exe), **kw}

    def downloaded(self, name="Duckstoken.exe", data=b"new build"):
        path = self.download_dir / name
        path.write_bytes(data)
        return path

    def test_refused_outside_windows(self):
        with mock.patch.object(updater, "_is_windows", return_value=False):
            with self.assertRaises(UpdateError) as cm:
                updater.apply(self.downloaded(), info=self.info())
        self.assertIn("Windows", str(cm.exception))
        self.popen.assert_not_called()

    def test_refused_when_not_a_packaged_app(self):
        for info in ({**INFO}, self.info(frozen=False), self.info(exe=None), self.info(exe="")):
            with self.subTest(info=info), self.assertRaises(UpdateError) as cm:
                updater.apply(self.downloaded(), info=info)
            self.assertIn("packaged", str(cm.exception))
        self.popen.assert_not_called()

    def test_refused_for_the_wrong_or_missing_file(self):
        with self.assertRaises(UpdateError):
            updater.apply(self.downloaded("Duckstoken-Setup.exe"), info=self.info())  # installer file, portable app
        with self.assertRaises(UpdateError):
            updater.apply(self.downloaded("Duckstoken.exe"), info=self.info(installer=True))
        with self.assertRaises(UpdateError):
            updater.apply(self.download_dir / "missing" / "Duckstoken.exe", info=self.info())
        with self.assertRaises(UpdateError):
            updater.apply(self.downloaded("evil.exe"), info=self.info())
        self.popen.assert_not_called()

    def test_installer_runs_silently_with_close_and_restart(self):
        setup = self.downloaded("Duckstoken-Setup.exe")
        out = updater.apply(setup, info=self.info(installer=True))
        self.assertEqual(out, {"mode": "installer", "restart": True, "helper": None})
        args, kwargs = self.popen.call_args
        self.assertEqual(args[0], [str(setup), "/SILENT", "/CLOSEAPPLICATIONS", "/NORESTART", "/RESTARTAPPLICATIONS"])
        self.assertEqual(kwargs["creationflags"], 0x00000008 | 0x00000200)  # detached, own process group
        self.assertEqual(kwargs["env"]["PYINSTALLER_RESET_ENVIRONMENT"], "1")
        self.assertNotIn("_MEIPASS2", kwargs["env"])
        self.assertTrue(setup.exists())  # the installer needs it after we are gone

    def test_installer_without_restart(self):
        out = updater.apply(self.downloaded("Duckstoken-Setup.exe"), info=self.info(installer=True), restart=False)
        self.assertFalse(out["restart"])
        self.assertNotIn("/RESTARTAPPLICATIONS", self.popen.call_args[0][0])
        self.assertIn("/SILENT", self.popen.call_args[0][0])

    def test_portable_stages_writes_the_helper_and_launches_it_hidden(self):
        new = self.downloaded(data=b"new build")
        with mock.patch.dict(os.environ, {"_MEIPASS2": "C:\\Temp\\_MEI1234"}):
            out = updater.apply(new, info=self.info())
        helper = self.home / "updates" / "apply_update.cmd"
        self.assertEqual(out, {"mode": "portable", "restart": True, "helper": str(helper)})
        self.assertEqual((self.app / "Duckstoken.exe.new").read_bytes(), b"new build")
        self.assertEqual(self.exe.read_bytes(), b"old build")  # the swap is the helper's job, after we exit
        self.assertFalse(new.exists())
        raw = helper.read_bytes()
        self.assertEqual(raw, updater._PORTABLE_HELPER.replace("\n", "\r\n").encode("ascii"))
        args, kwargs = self.popen.call_args
        self.assertEqual(args[0], f'"C:\\Windows\\System32\\cmd.exe" /d /s /c ""{helper}""')
        self.assertEqual(kwargs["creationflags"], 0x08000000 | 0x00000200)  # no console window
        env = kwargs["env"]
        self.assertEqual(env["DUCK_PID"], str(os.getpid()))
        self.assertEqual(env["DUCK_TARGET"], str(self.exe))
        self.assertEqual(env["DUCK_STAGED"], str(self.app / "Duckstoken.exe.new"))
        self.assertEqual(env["DUCK_RESTART"], "1")
        self.assertEqual(env["DUCK_LOG"], str(self.home / "updates" / "update.log"))
        self.assertEqual(env["PYINSTALLER_RESET_ENVIRONMENT"], "1")
        self.assertNotIn("_MEIPASS2", env)
        self.assertIs(kwargs["stdin"], updater.subprocess.DEVNULL)

    def test_portable_without_restart(self):
        out = updater.apply(self.downloaded(), info=self.info(), restart=False)
        self.assertFalse(out["restart"])
        self.assertEqual(self.popen.call_args[1]["env"]["DUCK_RESTART"], "0")

    def test_cmd_exe_falls_back_to_the_system_folder(self):
        with mock.patch.dict(os.environ, {"SystemRoot": r"D:\Win"}):
            os.environ.pop("COMSPEC")
            self.assertEqual(updater._cmd_exe(), r"D:\Win\System32\cmd.exe")

    def test_unwritable_app_folder_is_reported_before_anything_is_launched(self):
        info = self.info(exe=str(self.app / "missing" / "Duckstoken.exe"))
        new = self.downloaded()
        with self.assertRaises(UpdateError) as cm:
            updater.apply(new, info=info)
        self.assertIn("cannot write", str(cm.exception))
        self.popen.assert_not_called()
        self.assertTrue(new.exists())

    def test_launch_failure_removes_the_staged_file_and_keeps_the_download(self):
        self.popen.side_effect = OSError("no cmd")
        new = self.downloaded()
        with self.assertRaises(UpdateError) as cm:
            updater.apply(new, info=self.info())
        self.assertIn("Could not start", str(cm.exception))
        self.assertFalse((self.app / "Duckstoken.exe.new").exists())
        self.assertTrue(new.exists())
        self.assertEqual(self.exe.read_bytes(), b"old build")

    def test_data_folder_problem_is_reported(self):
        (self.home).mkdir(parents=True)
        (self.home / "updates").write_text("a file where the folder should be")
        with self.assertRaises(UpdateError) as cm:
            updater.apply(self.downloaded(), info=self.info())
        self.assertIn("Could not prepare", str(cm.exception))
        self.popen.assert_not_called()

    def test_symlinked_download_is_refused(self):
        real = self.downloaded("real.bin")
        link = self.download_dir / "Duckstoken.exe"
        try:
            link.symlink_to(real)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available")
        with self.assertRaises(UpdateError):
            updater.apply(link, info=self.info())


class ApplyPlatformTests(unittest.TestCase):
    """No mocking: what `apply` does on the machine running the tests."""

    def test_a_source_checkout_never_updates_itself(self):
        with self.assertRaises(UpdateError):
            updater.apply(Path("Duckstoken.exe"))

    @unittest.skipIf(os.name == "nt", "Windows is the one platform that can install updates")
    def test_refused_off_windows_even_for_a_packaged_app(self):
        with self.assertRaises(UpdateError) as cm:
            updater.apply(Path("Duckstoken.exe"), info={**INFO, "frozen": True, "exe": "/opt/Duckstoken.exe"})
        self.assertIn("Windows", str(cm.exception))


# ================================================================================================ background checker
class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class CheckerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="duck-chk-")
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.clock = Clock()
        self.settings = {"update_channel": "stable"}
        self.calls = []
        self.result = {"available": True, "channel": "stable", "version": "9.9.9", "sha": None, "notes": "", "published": "",
                       "asset": None, "sha256": None, "html_url": "", "reason": "Version 9.9.9 is available."}

    def fake_check(self, channel):
        self.calls.append(channel)
        return {**self.result, "channel": channel}

    def checker(self, check_fn=None):
        return updater.UpdateChecker(lambda: self.settings, self.dir, clock=self.clock, check_fn=check_fn or self.fake_check)

    def test_checks_once_per_day_per_channel(self):
        c = self.checker()
        first = c.maybe_check()
        self.assertEqual(first["version"], "9.9.9")
        self.assertIsNone(c.maybe_check())
        self.clock.now += 24 * 3600 - 1
        self.assertIsNone(c.maybe_check())
        self.assertEqual(self.calls, ["stable"])
        self.clock.now += 2
        self.assertIsNotNone(c.maybe_check())
        self.assertEqual(self.calls, ["stable", "stable"])

    def test_each_channel_has_its_own_clock(self):
        c = self.checker()
        c.maybe_check()
        self.settings["update_channel"] = "latest"
        self.assertEqual(c.status()["result"], None)
        self.assertIsNotNone(c.maybe_check())
        self.settings["update_channel"] = "stable"
        self.assertEqual(c.status()["result"]["channel"], "stable")
        self.assertIsNone(c.maybe_check())
        self.assertEqual(self.calls, ["stable", "latest"])

    def test_force_skips_the_wait(self):
        c = self.checker()
        c.maybe_check()
        self.assertIsNotNone(c.maybe_check(force=True))
        self.assertEqual(len(self.calls), 2)

    def test_off_and_unknown_channels_do_nothing(self):
        c = self.checker()
        for channel in ("off", "bogus", None):
            self.settings["update_channel"] = channel
            self.assertIsNone(c.maybe_check(force=True))
        self.assertEqual(self.calls, [])
        self.assertIsNone(c.status()["result"])

    def test_default_channel_is_stable(self):
        self.settings.clear()
        self.checker().maybe_check()
        self.assertEqual(self.calls, ["stable"])

    def test_state_survives_a_restart(self):
        self.checker().maybe_check()
        state = json.loads((self.dir / "update_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["channels"]["stable"]["checked"], self.clock.now)
        again = self.checker()
        self.assertIsNone(again.maybe_check())
        self.assertEqual(self.calls, ["stable"])
        status = again.status()
        self.assertEqual(status["result"]["version"], "9.9.9")
        self.assertEqual(status["last_checked"], self.clock.now)

    def test_a_new_build_starts_a_fresh_cycle(self):
        self.checker().maybe_check()
        with mock.patch.object(updater._build, "BUILD_SHA", RUNNING_SHA):
            again = self.checker()
            self.assertIsNone(again.status()["result"])
            self.assertEqual(again.status()["last_checked"], 0.0)
            self.assertIsNotNone(again.maybe_check())
        with mock.patch.object(updater, "__version__", "42.0.0"):
            self.assertIsNone(self.checker().status()["result"])

    def test_corrupt_or_odd_state_files_are_ignored(self):
        for text in ("{not json", "[]", '{"channels": 5}', '{"channels": {"stable": "x", "latest": {"checked": "now"}}}', ""):
            with self.subTest(text=text):
                (self.dir / "update_state.json").write_text(text, encoding="utf-8")
                c = self.checker()
                self.assertIsNone(c.status()["result"])
                self.assertIsNotNone(c.maybe_check())
        (self.dir / "update_state.json").unlink()
        self.assertIsNotNone(self.checker().maybe_check())

    def test_unwritable_state_folder_does_not_break_checking(self):
        blocker = self.dir / "blocker"
        blocker.write_text("file")
        c = updater.UpdateChecker(lambda: self.settings, blocker / "data", clock=self.clock, check_fn=self.fake_check)
        self.assertIsNotNone(c.maybe_check())
        self.assertIsNone(c.maybe_check())  # still throttled in memory

    def test_errors_are_swallowed_into_last_error_and_retried_after_an_hour(self):
        outcomes = [UpdateError("GitHub is down"), RuntimeError("bug"), ValueError(), None]

        def flaky(channel):
            self.calls.append(channel)
            outcome = outcomes.pop(0)
            if outcome is not None:
                raise outcome
            return self.result

        c = self.checker(flaky)
        self.assertIsNone(c.maybe_check())
        self.assertEqual(c.last_error, "GitHub is down")
        self.assertEqual(c.status()["last_error"], "GitHub is down")
        self.assertIsNone(c.status()["result"])
        self.clock.now += 3599
        self.assertIsNone(c.maybe_check())
        self.assertEqual(len(self.calls), 1)
        self.clock.now += 2
        self.assertIsNone(c.maybe_check())
        self.assertEqual(c.last_error, "bug")
        self.clock.now += 3601
        self.assertIsNone(c.maybe_check())
        self.assertEqual(c.last_error, "ValueError")
        self.clock.now += 3601
        self.assertIsNotNone(c.maybe_check())
        self.assertEqual(c.last_error, "")

    def test_failures_are_not_remembered_across_restarts_but_successes_are(self):
        def broken(channel):
            raise UpdateError("offline")

        self.assertIsNone(self.checker(broken).maybe_check())
        self.assertFalse((self.dir / "update_state.json").exists())
        self.assertIsNotNone(self.checker().maybe_check())

    def test_a_forced_check_after_a_failure_does_not_wait(self):
        outcomes = [UpdateError("offline"), None]

        def flaky(channel):
            outcome = outcomes.pop(0)
            if outcome is not None:
                raise outcome
            return self.result

        c = self.checker(flaky)
        self.assertIsNone(c.maybe_check())
        self.assertIsNotNone(c.maybe_check(force=True))

    def test_only_one_check_at_a_time(self):
        entered, release = threading.Event(), threading.Event()

        def slow(channel):
            entered.set()
            release.wait(5)
            return self.result

        c = self.checker(slow)
        out = []
        t = threading.Thread(target=lambda: out.append(c.maybe_check()))
        t.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertTrue(c.status()["checking"])
            self.assertIsNone(c.maybe_check(force=True))
        finally:
            release.set()
            t.join(5)
        self.assertEqual(len(out), 1)
        self.assertIsNotNone(out[0])
        self.assertFalse(c.status()["checking"])

    def test_status_shape(self):
        c = self.checker()
        self.assertEqual(c.status(), {"checking": False, "result": None, "last_checked": 0.0, "last_error": "", "downloaded": None})
        c.maybe_check()
        status = c.status()
        self.assertEqual(set(status), {"checking", "result", "last_checked", "last_error", "downloaded"})
        self.assertEqual(status["last_checked"], self.clock.now)


class CheckerDownloadTests(FakeTestCase):
    def setUp(self):
        super().setUp()
        self.data_dir = self.tmp / "data"
        self.settings = {"update_channel": "stable"}
        self.clock = Clock()
        self.checker = updater.UpdateChecker(
            lambda: self.settings, self.data_dir, clock=self.clock,
            check_fn=lambda channel: updater.check(channel, api_base=self.fake.origin, info=INFO,
                                                   allow_host=self.fake.allow, timeout=5))
        patcher = mock.patch.object(updater, "download", partial(updater.download, allow_host=self.fake.allow))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_download_update_records_the_file_for_status(self):
        with self.assertRaises(UpdateError):
            self.checker.download_update()  # nothing checked yet
        self.fake.releases = [self.fake.release("v0.2.0")]
        self.checker.maybe_check()
        self.assertIsNone(self.checker.status()["downloaded"])
        progress = []
        path = self.checker.download_update(progress=lambda done, total: progress.append(done))
        self.assertEqual(path, self.data_dir / "updates" / "Duckstoken.exe")
        self.assertEqual(path.read_bytes(), payload("v0.2.0"))
        self.assertEqual(self.checker.status()["downloaded"], str(path))
        self.assertTrue(progress)

    def test_downloaded_goes_away_with_the_file_or_a_newer_release(self):
        self.fake.releases = [self.fake.release("v0.2.0")]
        self.checker.maybe_check()
        path = self.checker.download_update()
        self.fake.releases = [self.fake.release("v0.3.0")]
        self.checker.maybe_check(force=True)
        self.assertIsNone(self.checker.status()["downloaded"])
        self.fake.releases = [self.fake.release("v0.2.0")]
        self.checker.maybe_check(force=True)
        self.assertEqual(self.checker.status()["downloaded"], str(path))
        path.unlink()
        self.assertIsNone(self.checker.status()["downloaded"])

    def test_nothing_to_download_when_up_to_date(self):
        self.fake.releases = [self.fake.release("v0.1.0")]
        self.checker.maybe_check()
        with self.assertRaises(UpdateError):
            self.checker.download_update()

    def test_cancel_is_passed_through(self):
        self.fake.releases = [self.fake.release("v0.2.0")]
        self.checker.maybe_check()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(UpdateError):
            self.checker.download_update(cancel=cancel)
        self.assertIsNone(self.checker.status()["downloaded"])


if __name__ == "__main__":
    unittest.main()
