import gzip
import html.parser
import ipaddress
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import unquote_plus

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["DUCKSTOKEN_HOME"] = tempfile.mkdtemp(prefix="duck-websearch-")

from duckstoken import websearch  # noqa: E402
from duckstoken.websearch import WebError  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
CFG = {"web_provider": "duckduckgo", "web_max_results": 5, "web_fetch_pages": 2, "web_page_chars": 3000,
       "web_cache_minutes": 30, "web_timeout": 5, "web_allow_private": False}
PUBLIC_IP = "93.184.216.34"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def reply(body, ctype="text/html", status=200, charset="utf-8", url="https://example.test/"):
    if isinstance(body, (dict, list)):
        body = json.dumps(body)
    return websearch._Reply(url, status, ctype, charset, body.encode("utf-8") if isinstance(body, str) else body,
                            False)


def dns(mapping):
    """A getaddrinfo replacement answering from `mapping` {host: [ip, ...]}; IP literals resolve to themselves."""
    def fake(host, port, *args, **kwargs):
        try:
            ipaddress.ip_address(host)
            ips = [host]
        except ValueError:
            if host not in mapping:
                raise socket.gaierror(-2, "Name or service not known") from None
            ips = mapping[host]
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]
    return fake


# ------------------------------------------------------------------------------------------------ local server
class Site:
    """A throwaway HTTP server on 127.0.0.1. `routes` maps a path to (status, headers, body) or a callable."""

    def __init__(self, routes):
        site = self
        self.hits: list[tuple[str, dict]] = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                path = self.path.split("?")[0]
                site.hits.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
                route = routes.get(path)
                if route is None:
                    route = (404, {"Content-Type": "text/plain"}, b"nope")
                if callable(route):
                    route = route(site)
                status, headers, body = route
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def paths(self):
        return [p.split("?")[0] for p, _ in self.hits]


def html_route(body, ctype="text/html; charset=utf-8", status=200, headers=None):
    return status, {"Content-Type": ctype, **(headers or {})}, body.encode("utf-8") if isinstance(body, str) else body


PRIVATE_CFG = dict(CFG, web_allow_private=True)


# ----------------------------------------------------------------------------------------------- html_to_text
class HtmlToTextTests(unittest.TestCase):
    ARTICLE = ("<html><head><title> My &amp; Page </title><style>p{color:red}</style>"
               "<script>var x = '<p>script text</p>';</script></head><body>"
               "<nav><a href='/'>Home</a><ul><li>Nav item</li></ul></nav><header>Site banner</header>"
               "<article><header><h1>Real Headline</h1></header>"
               "<p>First   para\n with <b>bold</b> and <a href='x'>a link</a>.&nbsp;Entity &lt;ok&gt; &#169; &eacute;</p>"
               "<ul><li>one</li><li></li><li>two</li></ul>"
               "<table><tr><th>Name</th><th>Qty</th></tr><tr><td>Apple</td><td></td><td>3</td></tr></table>"
               "<pre>  code line 1\n    indented\n</pre><p>after</p>x<br/>line<br>break</article>"
               "<aside>Ad block</aside><form><input value=x>Subscribe now</form><footer>(c) 2025</footer>"
               "<noscript>enable js</noscript><svg><title>icon</title><text>svg text</text></svg></body></html>")

    def test_extracts_title_and_readable_text(self):
        title, text = websearch.html_to_text(self.ARTICLE + "<p>" + "filler " * 40 + "</p>")
        self.assertEqual(title, "My & Page")
        for kept in ("Real Headline", "First para with bold and a link. Entity <ok> © é", "- one\n- two",
                     "Name | Qty\nApple | 3", "  code line 1\n    indented", "after", "x\nline\nbreak"):
            self.assertIn(kept, text)
        for dropped in ("script text", "color:red", "Home", "Nav item", "Site banner", "Ad block", "Subscribe",
                        "(c) 2025", "enable js", "svg text", "icon"):
            self.assertNotIn(dropped, text)
        self.assertNotIn("\n\n\n", text)
        self.assertNotIn("- \n", text)

    def test_page_with_everything_inside_one_form_still_yields_text(self):
        body = "<form id='aspnetForm'><div><h1>Council minutes</h1><p>" + "The council met. " * 5 + "</p></div></form>"
        title, text = websearch.html_to_text(f"<html><head><title>Minutes</title></head><body>{body}</body></html>")
        self.assertEqual(title, "Minutes")
        self.assertIn("The council met.", text)

    def test_short_page_with_a_form_that_is_only_chrome_keeps_the_rest(self):
        long_text = "Interesting sentence number one. " * 10
        _, text = websearch.html_to_text(f"<p>{long_text}</p><form><label>Email</label><input></form>")
        self.assertNotIn("Email", text)

    def test_messy_markup_does_not_raise(self):
        samples = ["", "plain text only", "<p>unclosed <b>bold <i>italic", "</div></div></p>text",
                   "<![if IE]>x<![endif]>", "<![foo[ bar", "<!-- unclosed comment", "<a href='x",
                   "<p class='a>b'>quoted > gt</p>", "<SCRIPT>alert(1)</SCRIPT><P>UPPER</P>", "<![CDATA[ data ]]>",
                   "<?php echo 1 ?>text", "<<<>>>", "&#xZZ; &bogus;", "<nav>never closed <p>swallowed",
                   "\x00\x01<p>ctrl</p>"]
        for sample in samples:
            with self.subTest(sample=sample):
                title, text = websearch.html_to_text(sample)
                self.assertIsInstance(title, str)
                self.assertIsInstance(text, str)
        self.assertEqual(websearch.html_to_text("<SCRIPT>alert(1)</SCRIPT><P>UPPER</P>")[1], "UPPER")
        self.assertEqual(websearch.html_to_text("plain text only")[1], "plain text only")

    def test_non_string_input(self):
        self.assertEqual(websearch.html_to_text(None), ("", ""))
        self.assertEqual(websearch.html_to_text(b"<p>bytes</p>"), ("", ""))

    def test_deeply_nested_markup(self):
        title, text = websearch.html_to_text("<div>" * 20000 + "deep" + "</div>" * 20000)
        self.assertEqual(text, "deep")

    def test_parser_assertion_error_keeps_text_read_so_far(self):
        real_feed = html.parser.HTMLParser.feed

        def broken(self, data):
            real_feed(self, data)
            raise AssertionError("malformed marked section")

        with mock.patch.object(websearch._TextExtractor, "feed", broken):
            self.assertEqual(websearch.html_to_text("<title>T</title><p>kept</p>"), ("T", "kept"))

    def test_title_only_from_the_document_not_svg(self):
        self.assertEqual(websearch.html_to_text("<svg><title>icon</title></svg><title>Real</title>x")[0], "Real")

    def test_whitespace_and_blank_lines_are_collapsed(self):
        _, text = websearch.html_to_text("<p>a</p>\n\n\n<p>   </p><p>b</p><div>c</div><div>d</div>")
        self.assertEqual(text, "a\n\nb\n\nc\nd")


# ----------------------------------------------------------------------------------------- is_public_url / SSRF
class PublicUrlTests(unittest.TestCase):
    def test_literal_addresses_that_must_be_refused(self):
        for url in ["http://127.0.0.1/", "http://127.0.0.1:8080/x", "http://10.1.2.3/", "http://172.16.0.1/",
                    "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data/", "http://[::1]/", "http://[::1]:8080/",
                    "http://[fe80::1]/", "http://[fc00::1]/", "http://[::ffff:127.0.0.1]/", "http://[::ffff:10.0.0.1]/",
                    "http://[0:0:0:0:0:ffff:7f00:1]/", "http://0.0.0.0/", "http://[::]/", "http://100.64.0.1/",
                    "http://224.0.0.1/", "http://255.255.255.255/", "http://240.0.0.1/", "http://192.0.2.1/",
                    "http://[2002:7f00:1::]/", "http://[64:ff9b::7f00:1]/", "http://[ff02::1]/", "http://198.18.0.1/"]:
            with self.subTest(url=url):
                self.assertFalse(websearch.is_public_url(url))

    def test_numeric_host_tricks_are_refused(self):
        for url in ["http://2130706433/", "http://0x7f000001/", "http://017700000001/", "http://127.1/", "http://0x7f.1/",
                    "http://0177.0.0.1/", "http://010.0.0.1/", "http://134744072/", "http://0x08.0x08.0x08.0x08/",
                    "http://１２７.０.０.１/", "http://127.0.0.1./", "http://127。0。0。1/"]:
            with self.subTest(url=url):
                self.assertFalse(websearch.is_public_url(url))

    def test_scheme_and_syntax_tricks_are_refused(self):
        for url in ["file:///etc/passwd", "ftp://example.com/", "gopher://example.com/", "javascript:alert(1)",
                    "data:text/html,hi", "//example.com/", "example.com", "", "   ", "http://", "http:///x",
                    "http://user:pass@example.com/", "http://example.com@127.0.0.1/", "http://127.0.0.1#@example.com/",
                    "http://example.com\\@127.0.0.1/", "http://exa mple.com/", "http://example.com/\r\nHost: evil",
                    "http://example.com:99999/", "http://[::1/", "http://%31%32%37.0.0.1/", "http://a..b/",
                    "http://-bad-.com/", "http://" + "a" * 70 + ".com/", "http://example.com/" + "a" * 3000]:
            with self.subTest(url=url):
                self.assertFalse(websearch.is_public_url(url))
        self.assertFalse(websearch.is_public_url(None))
        self.assertFalse(websearch.is_public_url(42))

    def test_name_resolution_decides(self):
        with mock.patch.object(socket, "getaddrinfo", dns({
                "localhost": ["127.0.0.1"], "localhost6": ["::1"], "internal.corp": ["10.0.0.5"],
                "mixed.example": [PUBLIC_IP, "192.168.0.9"], "mapped.example": ["::ffff:127.0.0.1"],
                "meta.example": ["169.254.169.254"], "ok.example": [PUBLIC_IP], "ok6.example": ["2606:4700:4700::1111"],
                "xn--bcher-kva.example": [PUBLIC_IP]})):
            for host in ["localhost", "LOCALHOST", "localhost.", "localhost6", "internal.corp", "mixed.example",
                         "mapped.example", "meta.example", "does-not-exist.example"]:
                with self.subTest(host=host):
                    self.assertFalse(websearch.is_public_url(f"http://{host}/"))
            for url in ["http://ok.example/", "https://ok.example:8443/a?b=c#d", "HTTP://OK.EXAMPLE/", "http://ok.example./",
                        "http://ok6.example/", "http://bücher.example/"]:
                with self.subTest(url=url):
                    self.assertTrue(websearch.is_public_url(url))

    def test_public_literals_need_no_dns(self):
        with mock.patch.object(socket, "getaddrinfo", side_effect=AssertionError("no lookup expected")):
            for url in ["http://93.184.216.34/", "https://8.8.8.8/dns", "http://[2606:4700:4700::1111]/"]:
                with self.subTest(url=url):
                    self.assertTrue(websearch.is_public_url(url))

    def test_allow_private_lifts_only_the_address_check(self):
        for url in ["http://127.0.0.1:8080/", "http://10.0.0.5/", "http://[::1]/", "http://169.254.169.254/"]:
            with self.subTest(url=url):
                self.assertFalse(websearch.is_public_url(url))
                self.assertTrue(websearch.is_public_url(url, allow_private=True))
        for url in ["file:///etc/passwd", "http://user@127.0.0.1/", "http://2130706433/", "ftp://127.0.0.1/"]:
            with self.subTest(url=url):
                self.assertFalse(websearch.is_public_url(url, allow_private=True))


# ------------------------------------------------------------------------------------------------- fetch_page
class FetchPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = ("<html><head><title>Hello</title></head><body><h1>Heading</h1><p>" + "Some sentence here. " * 40 +
                    "</p></body></html>")
        cls.site = Site({
            "/page": html_route(cls.page),
            "/plain": html_route("line1\r\n\r\n\r\n\r\nline2", "text/plain; charset=utf-8"),
            "/json": html_route('{"a": 1}', "application/json"),
            "/latin": html_route("<p>café</p>".encode("latin-1"), "text/html; charset=iso-8859-1"),
            "/meta": html_route(b"<meta charset='windows-1252'><p>\x93quoted\x94</p>", "text/html"),
            "/badcharset": html_route("<p>naïve</p>", "text/html; charset=nonsense-9"),
            "/bom": html_route(b"\xef\xbb\xbf<p>bom</p>", "text/html"),
            "/png": html_route(b"\x89PNG\r\n", "image/png"),
            "/pdf": html_route(b"%PDF-1.4", "application/pdf"),
            "/bin": (200, {}, b"abc\x00\x01\x02def"),
            "/notype": (200, {}, b"<title>No type</title><p>works</p>"),
            "/big": html_route("word " * 800000, "text/plain"),
            "/gzip": html_route(gzip.compress(b"<title>Zipped</title><p>compressed body</p>"), "text/html",
                                headers={"Content-Encoding": "gzip"}),
            "/bomb": html_route(gzip.compress(b"0" * 60_000_000), "text/plain", headers={"Content-Encoding": "gzip"}),
            "/badgzip": html_route(b"this is not gzip", "text/html", headers={"Content-Encoding": "gzip"}),
            "/brotli": html_route(b"xx", "text/html", headers={"Content-Encoding": "br"}),
            "/r1": (302, {"Location": "/r2"}, b""),
            "/r2": (301, {"Location": "/page"}, b""),
            "/loop": (302, {"Location": "/loop"}, b""),
            "/nolocation": (302, {}, b""),
            "/404": html_route("gone", "text/plain", status=404),
            "/403": html_route("no", "text/plain", status=403),
            "/429": html_route("slow down", "text/plain", status=429),
            "/500": html_route("boom", "text/plain", status=500),
            "/slow": lambda site: (time.sleep(3), html_route("late"))[1],
        })
        cls.cfg = dict(PRIVATE_CFG, web_timeout=1)
        cls.url = cls.site.base

    @classmethod
    def tearDownClass(cls):
        cls.site.close()

    def fetch(self, path, **kw):
        return websearch.fetch_page(self.url + path, self.cfg, **kw)

    def test_html_page(self):
        page = self.fetch("/page")
        self.assertEqual(page["url"], self.url + "/page")
        self.assertEqual(page["title"], "Hello")
        self.assertTrue(page["text"].startswith("Heading\n\nSome sentence here."))
        self.assertFalse(page["truncated"])

    def test_sends_a_descriptive_user_agent_and_no_cookies(self):
        self.fetch("/page")
        headers = self.site.hits[-1][1]
        self.assertIn("Duckstoken", headers["user-agent"])
        self.assertNotIn("cookie", headers)
        self.assertNotIn("authorization", headers)
        self.assertEqual(headers["host"], f"127.0.0.1:{self.site.port}")

    def test_text_is_cut_to_the_configured_size(self):
        page = websearch.fetch_page(self.url + "/page", dict(self.cfg, web_page_chars=300))
        self.assertTrue(page["truncated"])
        self.assertLessEqual(len(page["text"]), 300)
        self.assertNotIn("  ", page["text"])
        page = self.fetch("/page", max_chars=50)
        self.assertLessEqual(len(page["text"]), 50)
        self.assertTrue(page["truncated"])
        page = self.fetch("/page", max_chars=200_000)
        self.assertFalse(page["truncated"])

    def test_plain_text_and_json_are_returned_as_text(self):
        self.assertEqual(self.fetch("/plain")["text"], "line1\n\nline2")
        self.assertEqual(self.fetch("/json")["text"], '{"a": 1}')
        self.assertEqual(self.fetch("/json")["title"], "")

    def test_charset_handling(self):
        self.assertEqual(self.fetch("/latin")["text"], "café")
        self.assertEqual(self.fetch("/meta")["text"], "“quoted”")
        self.assertEqual(self.fetch("/badcharset")["text"], "naïve")
        self.assertEqual(self.fetch("/bom")["text"], "bom")

    def test_missing_content_type_is_sniffed(self):
        page = self.fetch("/notype")
        self.assertEqual((page["title"], page["text"]), ("No type", "works"))
        with self.assertRaisesRegex(WebError, "binary"):
            self.fetch("/bin")

    def test_unsupported_types_are_refused(self):
        for path, name in (("/png", "image/png"), ("/pdf", "application/pdf")):
            with self.subTest(path=path):
                with self.assertRaisesRegex(WebError, name):
                    self.fetch(path)

    def test_body_is_limited_to_two_megabytes(self):
        page = self.fetch("/big", max_chars=10_000_000)
        self.assertTrue(page["truncated"])
        self.assertLessEqual(len(page["text"]), websearch.MAX_BODY)
        self.assertGreater(len(page["text"]), websearch.MAX_BODY // 2)

    def test_gzip_is_supported_and_bombs_are_bounded(self):
        page = self.fetch("/gzip")
        self.assertEqual((page["title"], page["text"]), ("Zipped", "compressed body"))
        t0 = time.monotonic()
        bomb = self.fetch("/bomb", max_chars=10_000_000)
        self.assertTrue(bomb["truncated"])
        self.assertLessEqual(len(bomb["text"]), websearch.MAX_BODY)
        self.assertLess(time.monotonic() - t0, 10)
        with self.assertRaisesRegex(WebError, "Could not get a response"):
            self.fetch("/badgzip")
        with self.assertRaisesRegex(WebError, "compression"):
            self.fetch("/brotli")

    def test_redirects_are_followed_and_the_final_url_is_reported(self):
        page = self.fetch("/r1")
        self.assertEqual(page["url"], self.url + "/page")
        self.assertEqual(page["title"], "Hello")

    def test_redirect_loops_and_missing_locations(self):
        with self.assertRaisesRegex(WebError, "Too many redirects"):
            self.fetch("/loop")
        hits_before = len(self.site.hits)
        with self.assertRaisesRegex(WebError, "without a destination"):
            self.fetch("/nolocation")
        self.assertEqual(len(self.site.hits), hits_before + 1)

    def test_http_errors_become_readable_messages(self):
        for path, expected in (("/404", "does not exist"), ("/403", "refused access"), ("/429", "rate-limiting"),
                               ("/500", "HTTP 500")):
            with self.subTest(path=path):
                with self.assertRaisesRegex(WebError, expected) as ctx:
                    self.fetch(path)
                self.assertEqual(ctx.exception.status, int(path[1:]))

    def test_slow_servers_time_out(self):
        t0 = time.monotonic()
        with self.assertRaisesRegex(WebError, "did not answer in time"):
            self.fetch("/slow")
        self.assertLess(time.monotonic() - t0, 3)

    def test_unreachable_server(self):
        dead = socket.socket()
        dead.bind(("127.0.0.1", 0))
        port = dead.getsockname()[1]
        dead.close()
        # Linux refuses at once; Windows keeps retrying a closed port and reports a timeout instead.
        with self.assertRaisesRegex(WebError, "Could not get a response|did not answer in time"):
            websearch.fetch_page(f"http://127.0.0.1:{port}/", self.cfg)

    def test_private_addresses_are_refused_by_default(self):
        hits_before = len(self.site.hits)
        with self.assertRaisesRegex(WebError, "private or local"):
            websearch.fetch_page(self.url + "/page", dict(CFG))
        self.assertEqual(len(self.site.hits), hits_before)

    def test_bad_urls(self):
        for url in ["file:///etc/passwd", "ftp://example.com/x", "", "not a url", None, "http://2130706433:1/"]:
            with self.subTest(url=url):
                with self.assertRaises(WebError):
                    websearch.fetch_page(url, self.cfg)

    def test_surrounding_whitespace_in_the_url_is_tolerated(self):
        self.assertEqual(websearch.fetch_page(f"  {self.url}/page\n", self.cfg)["title"], "Hello")


class FetchPageRedirectSsrfTests(unittest.TestCase):
    """The first hop looks public (DNS is faked, the TCP connection goes to a local server); the redirects do not."""

    @classmethod
    def setUpClass(cls):
        cls.secret_hits = []
        cls.site = Site({
            "/ok": html_route("<title>public</title><p>fine</p>"),
            "/secret": lambda site: (cls.secret_hits.append(1), html_route("secret"))[1],
            "/to-loopback": lambda site: (302, {"Location": f"http://127.0.0.1:{site.port}/secret"}, b""),
            "/to-decimal": lambda site: (302, {"Location": f"http://2130706433:{site.port}/secret"}, b""),
            "/to-v6": lambda site: (302, {"Location": f"http://[::1]:{site.port}/secret"}, b""),
            "/to-name": lambda site: (302, {"Location": f"http://localhost:{site.port}/secret"}, b""),
            "/to-meta": (302, {"Location": "http://169.254.169.254/latest/meta-data/"}, b""),
            "/to-lan": (302, {"Location": "http://192.168.0.1/admin"}, b""),
            "/to-file": (302, {"Location": "file:///etc/passwd"}, b""),
            "/to-userinfo": (302, {"Location": "http://good.test@127.0.0.1/"}, b""),
            "/to-relative": (302, {"Location": "/ok"}, b""),
            "/to-other-public": (302, {"Location": "http://other.test/ok"}, b""),
            "/two-hops": (302, {"Location": "/to-loopback"}, b""),
        })

    @classmethod
    def tearDownClass(cls):
        cls.site.close()

    def setUp(self):
        self.secret_hits.clear()
        self.connected = []
        self.lookups = []
        self.refuse = set()
        names = dns({"public.test": [PUBLIC_IP], "other.test": ["93.184.216.35"], "localhost": ["127.0.0.1"]})
        real_connect = socket.create_connection

        def resolver(host, *args, **kwargs):
            self.lookups.append(host)
            return names(host, *args, **kwargs)

        def connect(address, timeout=None, *args, **kwargs):
            self.connected.append(address)
            if address[0] in self.refuse:
                raise OSError("unreachable")
            return real_connect(("127.0.0.1", self.site.port), timeout)

        for name, fake in (("getaddrinfo", resolver), ("create_connection", connect)):
            patcher = mock.patch.object(socket, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fetch(self, path):
        return websearch.fetch_page("http://public.test" + path, dict(CFG, web_timeout=2))

    def test_public_page_works_and_connects_to_the_validated_address(self):
        page = self.fetch("/ok")
        self.assertEqual((page["title"], page["text"]), ("public", "fine"))
        self.assertEqual(self.connected, [(PUBLIC_IP, 80)])

    def test_redirects_to_private_targets_are_blocked(self):
        for path in ["/to-loopback", "/to-decimal", "/to-v6", "/to-name", "/to-meta", "/to-lan", "/to-file",
                     "/to-userinfo", "/two-hops"]:
            with self.subTest(path=path):
                self.connected.clear()
                with self.assertRaises(WebError):
                    self.fetch(path)
                self.assertEqual(self.secret_hits, [])
                self.assertEqual(len(self.connected), 2 if path == "/two-hops" else 1, "connected to the blocked hop")

    def test_relative_and_public_redirects_are_followed(self):
        self.assertEqual(self.fetch("/to-relative")["url"], "http://public.test/ok")
        self.connected.clear()
        page = self.fetch("/to-other-public")
        self.assertEqual((page["url"], page["title"]), ("http://other.test/ok", "public"))
        self.assertEqual(self.connected, [(PUBLIC_IP, 80), ("93.184.216.35", 80)])

    def test_the_host_is_resolved_once_per_hop_and_never_again_at_connect_time(self):
        self.fetch("/ok")
        self.assertEqual(self.lookups.count("public.test"), 1)
        self.assertNotIn(PUBLIC_IP, self.lookups)

    def test_dns_rebinding_cannot_change_the_connection_target(self):
        answers = iter([PUBLIC_IP, "127.0.0.1", "127.0.0.1"])

        def rebinding(host, port, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), port))]

        with mock.patch.object(socket, "getaddrinfo", rebinding):
            self.fetch("/ok")
        self.assertEqual(self.connected, [(PUBLIC_IP, 80)])

    def test_every_resolved_address_is_tried_in_order(self):
        self.refuse.add(PUBLIC_IP)
        with mock.patch.object(socket, "getaddrinfo", dns({"public.test": [PUBLIC_IP, "2606:4700:4700::1111"]})):
            self.fetch("/ok")
        self.assertEqual(self.connected, [(PUBLIC_IP, 80), ("2606:4700:4700::1111", 80)])


class ConnectionTests(unittest.TestCase):
    def test_connect_tries_the_next_address_when_one_fails(self):
        attempts = []

        def create(address, timeout):
            attempts.append(address)
            if address[0] == "10.9.9.9":
                raise OSError("unreachable")
            return mock.sentinel.sock

        with mock.patch.object(socket, "create_connection", create):
            self.assertIs(websearch._connect(["10.9.9.9", "93.184.216.34"], 443, 3), mock.sentinel.sock)
        self.assertEqual(attempts, [("10.9.9.9", 443), ("93.184.216.34", 443)])
        with mock.patch.object(socket, "create_connection", side_effect=OSError("down")):
            with self.assertRaisesRegex(OSError, "down"):
                websearch._connect(["93.184.216.34"], 443, 3)

    def test_https_wraps_the_pinned_socket_using_the_host_name_for_verification(self):
        raw = mock.Mock()
        context = mock.Mock()
        context.wrap_socket.return_value = mock.sentinel.tls
        with mock.patch.object(websearch, "_connect", return_value=raw) as connect, \
                mock.patch.object(websearch, "_tls_context", return_value=context):
            conn = websearch._HTTPSConnection("example.test", 443, [PUBLIC_IP], 7)
            conn.connect()
        connect.assert_called_once_with([PUBLIC_IP], 443, 7)
        context.wrap_socket.assert_called_once_with(raw, server_hostname="example.test")
        self.assertIs(conn.sock, mock.sentinel.tls)

    def test_failed_handshake_closes_the_raw_socket(self):
        raw = mock.Mock()
        context = mock.Mock()
        context.wrap_socket.side_effect = OSError("handshake failed")
        with mock.patch.object(websearch, "_connect", return_value=raw), \
                mock.patch.object(websearch, "_tls_context", return_value=context):
            with self.assertRaises(OSError):
                websearch._HTTPSConnection("example.test", 443, [PUBLIC_IP], 7).connect()
        raw.close.assert_called_once_with()

    def test_certificate_errors_have_their_own_message(self):
        import ssl
        message = websearch._describe(ssl.SSLCertVerificationError("bad"), "example.test")
        self.assertIn("certificate", message)
        self.assertIn("example.test", message)

    def test_read_body_enforces_the_overall_deadline(self):
        class Resp:
            def getheader(self, name):
                return None

            def read(self, n):
                return b"x" * 10

        with self.assertRaisesRegex(WebError, "too slow"):
            websearch._read_body(Resp(), 1000, time.monotonic() - 1)

    def test_read_body_stops_at_the_limit(self):
        class Resp:
            def getheader(self, name):
                return None

            def read(self, n):
                return b"x" * n

        body, truncated = websearch._read_body(Resp(), 100_000, time.monotonic() + 30)
        self.assertEqual(len(body), 100_000)
        self.assertTrue(truncated)

    def test_secret_headers_are_not_forwarded_to_another_host(self):
        site = Site({"/hop": (302, {"Location": "http://b.test/final"}, b""),
                     "/final": html_route("{}", "application/json")})
        self.addCleanup(site.close)
        get = dict(timeout=3, allow_private=False, accept="application/json", types=("application/json",),
                   secret_headers={"X-Subscription-Token": "k"})
        with mock.patch.object(socket, "getaddrinfo", dns({"a.test": [PUBLIC_IP], "b.test": ["93.184.216.35"]})), \
                mock.patch.object(websearch, "_connect", lambda addrs, port, timeout: socket.create_connection(
                    ("127.0.0.1", site.port), timeout)):
            websearch._get("http://a.test/hop", **get)
            self.assertEqual([h.get("x-subscription-token") for _, h in site.hits], ["k", None])
            site.hits.clear()
            websearch._get("http://a.test/final", **get)
            self.assertEqual(site.hits[0][1]["x-subscription-token"], "k")


# -------------------------------------------------------------------------------------------------- parsers
class ParserTests(unittest.TestCase):
    def test_duckduckgo_real_page(self):
        results = websearch.parse_duckduckgo(fixture("ws_duckduckgo.html"))
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0], {
            "title": "Status of Python versions", "url": "https://devguide.python.org/versions/",
            "snippet": "Find out the current and future status of Python versions, from feature to end-of-life, and the "
                       "dates of their first release and end of life. See the full chart and the key for each phase of "
                       "the release cycle."})
        self.assertEqual(results[2]["url"], "https://en.wikipedia.org/wiki/Python_(programming_language)")
        self.assertEqual(results[2]["title"], "Python (programming language) - Wikipedia")
        self.assertEqual(results[3]["url"], "https://www.python.org/doc/versions/")
        for r in results:
            self.assertTrue(r["url"].startswith("https://"))
            self.assertNotIn("duckduckgo.com", r["url"])
            self.assertNotIn("<", r["snippet"] + r["title"])
            self.assertNotIn("  ", r["snippet"])

    def test_duckduckgo_ads_and_junk_are_dropped(self):
        results = websearch.parse_duckduckgo(fixture("ws_duckduckgo_ads.html"))
        self.assertEqual([r["url"] for r in results], ["https://www.example.org/running?a=1&b=2", "http://news.example.net/"])
        self.assertEqual(results[0]["snippet"], "How to pick running shoes & socks.")
        self.assertEqual(results[1]["snippet"], "No link snippet, multi-line text.")

    def test_duckduckgo_unwrap(self):
        unwrap = websearch._unwrap_duckduckgo
        self.assertEqual(unwrap("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.test%2Fx%3Fy%3D1&rut=z"), "https://a.test/x?y=1")
        self.assertEqual(unwrap("/l/?uddg=https%3A%2F%2Fa.test%2F"), "https://a.test/")
        self.assertEqual(unwrap("https://a.test/direct"), "https://a.test/direct")
        for bad in ("https://duckduckgo.com/y.js?ad_domain=x", "//duckduckgo.com/l/?rut=only",
                    "https://links.duckduckgo.com/d.js", "", None, "http://[::1"):
            self.assertEqual(unwrap(bad), "", bad)

    def test_duckduckgo_garbage_input(self):
        for page in ("", None, "<html></html>", "<a class='result__a'>no href</a>", "<div class='result--ad'>unclosed",
                     "<a class='result__a' href='//duckduckgo.com/l/?uddg=http%3A%2F%2F[bad'>x</a>"):
            with self.subTest(page=page):
                self.assertEqual(websearch.parse_duckduckgo(page), [])

    def test_duckduckgo_duplicates_collapse(self):
        link = "<a class='result__a' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.test%2F'>A</a>"
        self.assertEqual(len(websearch.parse_duckduckgo(link * 3)), 1)

    def test_wikipedia_real_response(self):
        results = websearch.parse_wikipedia(json.loads(fixture("ws_wikipedia.json")))
        self.assertEqual([r["title"] for r in results], ["Python (programming language)",
                                                         "Outline of the Python programming language",
                                                         "Mojo (programming language)"])
        self.assertEqual(results[0]["url"], "https://en.wikipedia.org/wiki/Python_(programming_language)")
        self.assertTrue(results[0]["snippet"].startswith("Python is a high-level, general-purpose programming language"))
        self.assertLessEqual(max(len(r["snippet"]) for r in results), 500)
        self.assertNotIn("\n", results[0]["snippet"])

    def test_wikipedia_tolerates_other_shapes(self):
        pages = [{"title": "B & C", "index": 2, "extract": ""}, {"title": "A?", "index": 1}, {"pageid": 5}, "junk",
                 {"title": 5}]
        results = websearch.parse_wikipedia({"query": {"pages": pages}})
        self.assertEqual([r["url"] for r in results],
                         ["https://en.wikipedia.org/wiki/A%3F", "https://en.wikipedia.org/wiki/B_%26_C"])
        legacy = websearch.parse_wikipedia({"query": {"pages": {"1": {"title": "X", "extract": "x"}}}})
        self.assertEqual(legacy[0]["title"], "X")
        for bad in (None, [], {}, {"query": None}, {"query": {"pages": None}}, "text", {"query": {"pages": 5}}):
            self.assertEqual(websearch.parse_wikipedia(bad), [], bad)

    def test_searxng_fixture(self):
        results = websearch.parse_searxng(json.loads(fixture("ws_searxng.json")))
        self.assertEqual(results, [
            {"title": "Duck - Wikipedia", "url": "https://en.wikipedia.org/wiki/Duck",
             "snippet": "Duck is the common name for numerous species of waterfowl & relatives."},
            {"title": "Ducks of the world", "url": "https://example.org/ducks?x=1&y=2", "snippet": "A field guide."}])

    def test_brave_fixture(self):
        results = websearch.parse_brave(json.loads(fixture("ws_brave.json")))
        self.assertEqual(results, [
            {"title": "Duck | Britannica", "url": "https://www.britannica.com/animal/duck",
             "snippet": "Duck, any of various web-footed birds & relatives."},
            {"title": "Ducks Unlimited", "url": "https://www.ducks.org/", "snippet": "Wetlands conservation."}])

    def test_json_parsers_survive_garbage(self):
        for parse in (websearch.parse_searxng, websearch.parse_brave):
            for bad in (None, [], {}, "x", 5, {"results": 5}, {"web": []}, {"web": {"results": "x"}},
                        {"results": [None, 1, []]}):
                with self.subTest(parse=parse.__name__, bad=bad):
                    self.assertEqual(parse(bad), [])

    def test_results_are_bounded_and_clean(self):
        data = {"results": [{"url": "https://a.test/", "title": "T" * 1000, "content": "<script>x</script>" + "s " * 1000},
                            {"url": "https://a.test/" + "p" * 3000, "title": "too long url"},
                            {"url": "https://a.test/\nx", "title": "control char"},
                            {"url": "http://[::1", "title": "unparsable"},
                            {"url": "https://b.test/", "title": ""}]}
        results = websearch.parse_searxng(data)
        self.assertEqual([r["url"] for r in results], ["https://a.test/", "https://b.test/"])
        self.assertLessEqual(len(results[0]["title"]), 200)
        self.assertLessEqual(len(results[0]["snippet"]), 500)
        self.assertEqual(results[1]["title"], "https://b.test/")


# --------------------------------------------------------------------------------------------------- search
class SearchTests(unittest.TestCase):
    def run_search(self, query, cfg, answer):
        calls = []

        def fake_get(url, **kw):
            calls.append((url, kw))
            return answer

        with mock.patch.object(websearch, "_get", fake_get):
            return websearch.search(query, cfg), calls

    def test_duckduckgo(self):
        results, calls = self.run_search("python  release\tdate", dict(CFG, web_max_results=2),
                                         reply(fixture("ws_duckduckgo.html")))
        self.assertEqual(len(results), 2)
        url, kw = calls[0]
        self.assertEqual(url, "https://html.duckduckgo.com/html/?q=python+release+date")
        self.assertFalse(kw["allow_private"])
        self.assertEqual(kw["timeout"], 5)

    def test_duckduckgo_rate_limit_page_is_an_error_not_an_empty_result(self):
        blocked = reply("<form action='//duckduckgo.com/anomaly.js?sv=html&cc=botnet'></form>", status=202)
        with self.assertRaisesRegex(WebError, "human check"):
            self.run_search("anything", CFG, blocked)
        with self.assertRaisesRegex(WebError, "human check"):
            self.run_search("anything", CFG, reply("<html>blocked</html>", status=202))

    def test_duckduckgo_without_results_is_an_empty_list(self):
        results, _ = self.run_search("zzzz", CFG, reply("<html><div class='no-results'>No results.</div></html>"))
        self.assertEqual(results, [])

    def test_duckduckgo_result_about_anomalies_is_not_mistaken_for_a_block(self):
        page = fixture("ws_duckduckgo.html").replace("Status of Python versions", "Anomaly detection")
        results, _ = self.run_search("anomaly detection", CFG, reply(page))
        self.assertEqual(results[0]["title"], "Anomaly detection")

    def test_wikipedia(self):
        results, calls = self.run_search("python", dict(CFG, web_provider="wikipedia", web_max_results=3),
                                         reply(fixture("ws_wikipedia.json"), "application/json"))
        self.assertEqual(len(results), 3)
        url, kw = calls[0]
        self.assertTrue(url.startswith("https://en.wikipedia.org/w/api.php?"))
        for part in ("action=query", "generator=search", "gsrsearch=python", "gsrlimit=3", "prop=extracts", "explaintext=1",
                     "format=json"):
            self.assertIn(part, url)
        self.assertEqual(kw["types"], ("application/json",))

    def test_wikipedia_api_error(self):
        with self.assertRaisesRegex(WebError, "Wikipedia: bad things"):
            self.run_search("x", dict(CFG, web_provider="wikipedia"),
                            reply({"error": {"code": "bad", "info": "bad things"}}, "application/json"))

    def test_wikipedia_non_json_answer(self):
        with self.assertRaisesRegex(WebError, "could not be understood"):
            self.run_search("x", dict(CFG, web_provider="wikipedia"), reply("<html>maintenance</html>", "application/json"))

    def test_brave(self):
        cfg = dict(CFG, web_provider="brave", brave_api_key="  secret-key  ", web_max_results=1)
        results, calls = self.run_search("duck", cfg, reply(fixture("ws_brave.json"), "application/json"))
        self.assertEqual(len(results), 1)
        url, kw = calls[0]
        self.assertTrue(url.startswith("https://api.search.brave.com/res/v1/web/search?"))
        self.assertIn("q=duck", url)
        self.assertIn("count=1", url)
        self.assertEqual(kw["secret_headers"], {"X-Subscription-Token": "secret-key"})

    def test_brave_needs_a_key(self):
        with self.assertRaisesRegex(WebError, "Brave Search API key"):
            self.run_search("duck", dict(CFG, web_provider="brave"), reply("{}"))

    def test_searxng_through_a_real_local_instance(self):
        site = Site({"/search": html_route(fixture("ws_searxng.json"), "application/json"),
                     "/forbidden/search": html_route("no", "text/plain", status=403)})
        self.addCleanup(site.close)
        cfg = dict(CFG, web_provider="searxng", searxng_url=site.base + "/")
        results = websearch.search("duck & geese", cfg)
        self.assertEqual(len(results), 2)
        path = site.hits[0][0]
        self.assertTrue(path.startswith("/search?"))
        self.assertIn("q=duck+%26+geese", path)
        self.assertIn("format=json", path)
        for variant in (f"127.0.0.1:{site.port}", site.base + "/search"):
            self.assertEqual(len(websearch.search("duck", dict(cfg, searxng_url=variant))), 2, variant)
        with self.assertRaisesRegex(WebError, "JSON output must be enabled"):
            websearch.search("duck", dict(cfg, searxng_url=site.base + "/forbidden"))

    def test_searxng_needs_an_address(self):
        with self.assertRaisesRegex(WebError, "SearXNG"):
            self.run_search("duck", dict(CFG, web_provider="searxng", searxng_url="  "), reply("{}"))

    def test_searxng_may_live_on_the_home_network(self):
        results, calls = self.run_search("duck", dict(CFG, web_provider="searxng", searxng_url="http://192.168.1.20:8080"),
                                         reply(fixture("ws_searxng.json"), "application/json"))
        self.assertTrue(calls[0][1]["allow_private"])
        self.assertEqual(len(results), 2)

    def test_unknown_provider_and_bad_queries(self):
        with self.assertRaisesRegex(WebError, "Unknown search provider"):
            self.run_search("x", dict(CFG, web_provider="altavista"), reply(""))
        for query in ("", "   ", None, "\x00\x01 \n"):
            with self.subTest(query=query):
                with self.assertRaisesRegex(WebError, "nothing to search"):
                    self.run_search(query, CFG, reply(""))

    def test_query_is_cleaned_and_bounded(self):
        _, calls = self.run_search("a\x00b\nc " + "x" * 2000, CFG, reply(""))
        query = unquote_plus(calls[0][0].split("q=")[1])
        self.assertEqual(query, "a b c " + "x" * 393 + "…")
        self.assertEqual(len(query), websearch.MAX_QUERY)

    def test_provider_failures_are_not_swallowed(self):
        with mock.patch.object(websearch, "_get", side_effect=WebError("html.duckduckgo.com did not answer in time")):
            with self.assertRaisesRegex(WebError, "did not answer"):
                websearch.search("x", CFG)

    def test_config_values_are_clamped_and_defaulted(self):
        results, calls = self.run_search("python", {"web_max_results": "1", "web_timeout": 9999},
                                         reply(fixture("ws_duckduckgo.html")))
        self.assertEqual(len(results), 1)
        self.assertEqual(calls[0][1]["timeout"], 120)
        results, _ = self.run_search("python", {"web_max_results": "many"}, reply(fixture("ws_duckduckgo.html")))
        self.assertEqual(len(results), 4)
        results, _ = self.run_search("python", {"web_max_results": 99}, reply(fixture("ws_duckduckgo.html")))
        self.assertEqual(len(results), 4)


# ------------------------------------------------------------------------------------------------ needs_web
class NeedsWebTests(unittest.TestCase):
    YES = [
        "What is the latest version of Python?", "who is the current CEO of Microsoft", "What's the weather in Paris tomorrow",
        "bitcoin price", "when did the Berlin wall fall", "Who won the 2025 Champions League final?",
        "what happened in 2025 in tech", "search for good ramen restaurants in Berlin", "look up the release date of GTA 6",
        "google the capital of Australia", "Summarize the latest news about OpenAI", "what is the score of the Lakers game",
        "see https://example.com/post and tell me what you think", "Is Python 3.14 out yet?", "what's the stock price of Apple",
        "What is the current time in Tokyo?", "Please search the web for the best laptops under 1000",
        "where is the nearest open pharmacy right now?", "who is Ada Lovelace", "how many people live in Tokyo?",
        "what is the exchange rate USD to EUR", "when was Python 3.12 released?", "latest news on tesla",
        "is the new iphone out yet", "What movies are coming out this month?", "who is the president of France",
        "will it rain tomorrow in Seattle?", "what is the population of Japan", "check online whether the store is open",
        "Release notes for Node 24", "www.example.com is it down?", "what is the latest on the Ukraine war",
        "what time is it in London", "Compare iPhone prices", "Find me hotels in Lisbon, look it up online",
    ]
    NO = [
        "hi", "hello there", "thanks a lot", "how are you today", "ok", "yes please", "Hey!",
        "Write a Python function that checks if a number is prime", "rewrite this email to sound more polite: hello bob",
        "translate good morning to French", "Summarize this text: The quick brown fox jumps over the lazy dog and keeps going.",
        "what is 15% of 200", "what is the square root of 144", "12 * 34", "how much is 5 plus 3", "solve for x in 2x + 3 = 7",
        "fix the bug in my code, it throws an error", "tell me a joke about cats", "what should I cook for dinner tonight?",
        "I recently started learning piano and love it", "what is the difference between a list and a tuple in python",
        "explain how recursion works", "how do I search for a file in Windows", "write a poem about the sea",
        "what is my favorite color", "how do I get the current directory in python", "Can you help me plan my day?",
        "what is the meaning of life", "Help me write a cover letter for a software job",
        "Create a function to search for a substring in a string", "newsletter ideas for my bakery",
        "```python\nprint('hi')\n``` why does this fail", "my latest project is a todo app, help me design the database",
        "which version of python should I use for learning", "best way to learn Spanish", "what do you think about Python",
        "Tell me about the history of Rome", "is it safe to eat raw eggs", "", "   ", None,
        "x" * 900 + " latest news",
    ]

    def test_queries_that_need_the_web(self):
        for q in self.YES:
            with self.subTest(q=q):
                self.assertTrue(websearch.needs_web(q))

    def test_queries_that_do_not(self):
        for q in self.NO:
            with self.subTest(q=q):
                self.assertFalse(websearch.needs_web(q))

    def test_is_deterministic_and_never_raises(self):
        for q in self.YES + self.NO + [123, [], {"a": 1}, "\x00" * 5, "ünïcödé latest news ✓"]:
            self.assertEqual(websearch.needs_web(q), websearch.needs_web(q))

    def test_years(self):
        self.assertTrue(websearch.needs_web("best smartphones of 2026"))
        self.assertTrue(websearch.needs_web("what changed in 2023 for tax law"))
        self.assertFalse(websearch.needs_web("what happened in 2019 in the world"))
        self.assertFalse(websearch.needs_web("part number 20250 in the catalog is faulty"))


# -------------------------------------------------------------------------------------------------- ground
def result(i, snippet="snippet"):
    return {"title": f"Title {i}", "url": f"https://site{i}.test/page", "snippet": f"{snippet} {i}"}


class GroundCase(unittest.TestCase):
    """Fixtures for the ground() tests: search and fetch_page are replaced for the whole test."""

    def setUp(self):
        websearch.clear_cache()
        self.addCleanup(websearch.clear_cache)
        self.searches = []
        self.fetches = []
        self.results = [result(i) for i in range(1, 6)]
        self.pages = {r["url"]: f"Page text about {r['title']}.\n\n\nSecond paragraph." for r in self.results}
        for name, fake in (("search", self.fake_search), ("fetch_page", self.fake_fetch)):
            patcher = mock.patch.object(websearch, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_search(self, query, cfg):
        self.searches.append(query)
        return list(self.results)

    def fake_fetch(self, url, cfg, max_chars=None):
        self.fetches.append((url, max_chars))
        if url not in self.pages:
            raise WebError("blocked")
        return {"url": url, "title": "t", "text": self.pages[url], "truncated": False}

    def ground(self, query="python 3.14 release date", cfg=None):
        return websearch.ground(query, dict(CFG, **(cfg or {})))


class GroundTests(GroundCase):
    def test_builds_a_cited_block(self):
        g = self.ground()
        lines = g["text"].split("\n")
        today = time.strftime("%Y-%m-%d", time.gmtime())
        self.assertEqual(lines[0], f"Web results for “python 3.14 release date” (retrieved {today} UTC). "
                                   "Cite sources by URL when you use them:")
        self.assertIn("untrusted", lines[1])
        self.assertIn("[1] Title 1 — https://site1.test/page\nPage text about Title 1.\nSecond paragraph.", g["text"])
        self.assertIn("[2] Title 2 — https://site2.test/page\nPage text about Title 2.", g["text"])
        self.assertIn("[3] Title 3 — https://site3.test/page\nsnippet 3", g["text"], "snippet beyond web_fetch_pages")
        self.assertEqual([s["url"] for s in g["sources"]], [r["url"] for r in self.results])
        self.assertEqual(g["sources"][0], {"title": "Title 1", "url": "https://site1.test/page", "snippet": "snippet 1"})
        self.assertNotIn("error", g)

    def test_only_the_configured_number_of_pages_is_read(self):
        self.ground(cfg={"web_fetch_pages": 1, "web_page_chars": 777})
        self.assertEqual(self.fetches, [("https://site1.test/page", 777)])
        self.fetches.clear()
        websearch.clear_cache()
        g = self.ground(cfg={"web_fetch_pages": 0})
        self.assertEqual(self.fetches, [])
        self.assertIn("[1] Title 1 — https://site1.test/page\nsnippet 1", g["text"])

    def test_failed_pages_fall_back_to_the_snippet(self):
        del self.pages["https://site1.test/page"]
        g = self.ground()
        self.assertIn("[1] Title 1 — https://site1.test/page\nsnippet 1", g["text"])
        self.assertIn("Page text about Title 2", g["text"])
        self.assertEqual(len(g["sources"]), 5)

    def test_pages_are_fetched_in_parallel(self):
        barrier = threading.Barrier(2, timeout=5)

        def meeting_fetch(url, cfg, max_chars=None):
            barrier.wait()
            return {"url": url, "title": "", "text": "ok", "truncated": False}

        with mock.patch.object(websearch, "fetch_page", meeting_fetch):
            g = self.ground("parallel fetching")
        self.assertIn("[2] Title 2 — https://site2.test/page\nok", g["text"])

    def test_hard_character_budget(self):
        self.results = [{"title": "T" * 150, "url": "https://x.test/" + "p" * 200, "snippet": "s " * 250} for _ in range(8)]
        for i, r in enumerate(self.results):
            r["url"] += str(i)
        self.pages = {r["url"]: "word " * 5000 for r in self.results}
        for cfg in ({}, {"web_max_results": 1, "web_fetch_pages": 0},
                    {"web_max_results": 10, "web_fetch_pages": 5, "web_page_chars": 30000},
                    {"web_max_results": 3, "web_fetch_pages": 3, "web_page_chars": 300},
                    {"web_max_results": 2, "web_fetch_pages": 0}):
            with self.subTest(cfg=cfg):
                websearch.clear_cache()
                merged = dict(CFG, **cfg)
                budget = max(600, merged["web_max_results"] * 400 + merged["web_fetch_pages"] * merged["web_page_chars"])
                g = self.ground(cfg=cfg)
                self.assertLessEqual(len(g["text"]), budget)
                self.assertGreater(len(g["text"]), 300)
                self.assertTrue(g["sources"])
                for n, source in enumerate(g["sources"], 1):
                    self.assertIn(f"[{n}] {source['title']} — {source['url']}", g["text"])

    def test_results_that_do_not_fit_are_left_out_and_numbering_stays_consistent(self):
        self.results = [{"title": "short", "url": "https://a.test/", "snippet": "a"},
                        {"title": "L" * 190, "url": "https://b.test/" + "q" * 1500, "snippet": "b"},
                        {"title": "last", "url": "https://c.test/", "snippet": "c"}]
        g = self.ground(cfg={"web_max_results": 1, "web_fetch_pages": 0})
        self.assertLessEqual(len(g["text"]), 600)
        self.assertEqual([s["title"] for s in g["sources"]], ["short", "last"])
        self.assertIn("[2] last — https://c.test/", g["text"])

    def test_query_is_shortened_in_the_header_and_cleaned(self):
        g = self.ground(query="x" * 380 + "\n\x00y")
        header = g["text"].split("\n")[0]
        self.assertIn("x" * 120 + "”", header)
        self.assertLess(len(header), 220)

    def test_errors_never_raise(self):
        def boom(query, cfg):
            raise WebError("DuckDuckGo is asking for a human check")

        with mock.patch.object(websearch, "search", boom):
            g = websearch.ground("q q", dict(CFG))
        self.assertEqual(g, {"text": "", "sources": [], "error": "DuckDuckGo is asking for a human check"})

        with mock.patch.object(websearch, "search", side_effect=RuntimeError("bug")):
            g = websearch.ground("q q", dict(CFG))
        self.assertEqual((g["text"], g["sources"]), ("", []))
        self.assertIn("RuntimeError", g["error"])

        self.results = []
        self.assertIn("no results", self.ground()["error"])
        for q in ("", None, "  \n "):
            self.assertIn("nothing to search", self.ground(query=q)["error"])

    def test_unusable_results_give_an_error(self):
        self.results = [{"title": "T" * 100, "url": "https://x.test/" + "p" * 2000, "snippet": "s"}]
        g = self.ground(cfg={"web_max_results": 1, "web_fetch_pages": 0})
        self.assertEqual(g["text"], "")
        self.assertIn("nothing that could be used", g["error"])

    def test_page_reading_crash_is_contained(self):
        with mock.patch.object(websearch, "fetch_page", side_effect=KeyError("x")):
            g = self.ground("some query")
        self.assertEqual(g["text"], "")
        self.assertIn("KeyError", g["error"])


class GroundCacheTests(GroundCase):
    def setUp(self):
        super().setUp()
        self.now = [1_000_000.0]
        patcher = mock.patch.object(time, "time", lambda: self.now[0])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cache_hit_within_ttl_and_expiry_after(self):
        first = self.ground()
        self.assertEqual(len(self.searches), 1)
        self.now[0] += 29 * 60
        self.assertEqual(self.ground(), first)
        self.assertEqual(len(self.searches), 1, "served from the cache")
        self.now[0] += 2 * 60
        self.ground()
        self.assertEqual(len(self.searches), 2, "expired after 30 minutes")

    def test_cache_key_ignores_case_and_spacing_but_not_provider_or_settings(self):
        self.ground("Python  Release")
        self.ground("python release")
        self.assertEqual(len(self.searches), 1)
        self.ground("python release", {"web_provider": "wikipedia"})
        self.assertEqual(len(self.searches), 2)
        self.ground("python release", {"web_max_results": 3})
        self.assertEqual(len(self.searches), 3)
        self.ground("something else")
        self.assertEqual(len(self.searches), 4)

    def test_zero_minutes_disables_the_cache(self):
        self.ground(cfg={"web_cache_minutes": 0})
        self.ground(cfg={"web_cache_minutes": 0})
        self.assertEqual(len(self.searches), 2)

    def test_failures_are_not_cached(self):
        self.results = []
        self.ground()
        self.results = [result(1)]
        self.assertTrue(self.ground()["text"])
        self.assertEqual(len(self.searches), 2)

    def test_cached_values_cannot_be_modified_by_callers(self):
        first = self.ground()
        first["text"] = "tampered"
        first["sources"].clear()
        again = self.ground()
        self.assertTrue(again["text"].startswith("Web results"))
        self.assertEqual(len(again["sources"]), 5)

    def test_clear_cache(self):
        self.ground()
        websearch.clear_cache()
        self.ground()
        self.assertEqual(len(self.searches), 2)

    def test_cache_is_bounded(self):
        for i in range(websearch._CACHE_ENTRIES + 20):
            self.ground(f"query number {i}")
        self.assertLessEqual(len(websearch._cache), websearch._CACHE_ENTRIES)
        self.ground(f"query number {websearch._CACHE_ENTRIES + 19}")
        self.assertEqual(len(self.searches), websearch._CACHE_ENTRIES + 20, "newest entry is still cached")

    def test_concurrent_use_is_safe(self):
        errors = []

        def worker(n):
            try:
                for i in range(30):
                    g = self.ground(f"query {(n + i) % 7} shared")
                    assert g["text"].startswith("Web results")
                    if i % 10 == 0:
                        websearch.clear_cache()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
        [t.start() for t in threads]
        [t.join(30) for t in threads]
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
