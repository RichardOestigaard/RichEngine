import http.server
import socket
import threading
import unittest
from unittest import mock

from server import webfetch
from server.errors import APIError


def public_addrinfo(host, port, *args, **kwargs):
    """A resolver answer that always passes the public-address check."""
    return [(socket.AF_INET, 0, 0, "", ("93.184.216.34", port))]


class CheckedUrlTest(unittest.TestCase):
    def test_rejects_missing_and_blank_urls(self):
        for url in (None, "", "   ", 42):
            with self.assertRaises(APIError) as caught:
                webfetch._checked_url(url)
            self.assertEqual(caught.exception.status, 400)

    def test_rejects_non_http_schemes(self):
        for url in ("file:///etc/passwd", "ftp://example.com/x", "data:text/plain,hi"):
            with self.assertRaises(APIError) as caught:
                webfetch._checked_url(url)
            self.assertEqual(caught.exception.status, 422)

    def test_rejects_urls_with_credentials(self):
        with self.assertRaises(APIError) as caught:
            webfetch._checked_url("https://user:pass@example.com/")
        self.assertEqual(caught.exception.status, 422)

    def test_rejects_private_and_loopback_addresses(self):
        for ip in ("127.0.0.1", "10.0.0.8", "192.168.1.1", "169.254.1.1", "::1"):
            with mock.patch.object(
                webfetch,
                "_resolve",
                return_value=[(socket.AF_INET, 1, 6, "", (ip, 443))],
            ):
                with self.assertRaises(APIError) as caught:
                    webfetch._checked_url("https://internal.example/")
                self.assertEqual(caught.exception.status, 422, ip)

    def test_rejects_unresolvable_hosts(self):
        with mock.patch.object(
            webfetch, "_resolve", side_effect=socket.gaierror("nope")
        ):
            with self.assertRaises(APIError) as caught:
                webfetch._checked_url("https://gone.invalid/")
            self.assertEqual(caught.exception.status, 502)

    def test_accepts_a_public_url(self):
        with mock.patch.object(webfetch, "_resolve", side_effect=public_addrinfo):
            self.assertEqual(
                webfetch._checked_url("https://example.com/path?q=1"),
                "https://example.com/path?q=1",
            )


class TextExtractorTest(unittest.TestCase):
    def extract(self, markup):
        extractor = webfetch._TextExtractor()
        extractor.feed(markup)
        extractor.close()
        return extractor.title, webfetch._collapse("".join(extractor.parts))

    def test_extracts_text_and_title(self):
        title, text = self.extract(
            "<html><head><title>  My Page </title></head>"
            "<body><h1>Hello</h1><p>one</p><p>two</p></body></html>"
        )
        self.assertEqual(title.strip(), "My Page")
        self.assertEqual(text, "Hello\n\none\n\ntwo")

    def test_skips_script_and_style(self):
        _, text = self.extract(
            "<body><style>.x{color:red}</style>"
            "<p>keep</p><script>var x = 'drop';</script></body>"
        )
        self.assertEqual(text, "keep")

    def test_unescapes_entities_and_collapses_space(self):
        _, text = self.extract("<p>a &amp; b</p><p>c<br>d</p>")
        self.assertEqual(text, "a & b\n\nc\nd")


class FetchTest(unittest.TestCase):
    """End-to-end through a real local HTTP server. The resolver is patched
    to a public answer so the loopback fixture passes the private-address
    check the tool exists for."""

    def setUp(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/gone":
                    self.send_error(404)
                    return
                if self.path == "/away":
                    self.send_response(302)
                    self.send_header("Location", "/landing")
                    self.end_headers()
                    return
                if self.path == "/binary":
                    body = b"\x89PNG\r\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = (
                    "<html><head><title>Fixture</title></head>"
                    "<body><p>hello fixture</p></body></html>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # Cleanups run in reverse: stop serving, join the thread, then close.
        self.addCleanup(self.httpd.server_close)
        self.thread = threading.Thread(target=self.httpd.serve_forever)
        self.thread.start()
        self.addCleanup(self.thread.join)
        self.addCleanup(self.httpd.shutdown)
        patch = mock.patch.object(webfetch, "_resolve", side_effect=public_addrinfo)
        patch.start()
        self.addCleanup(patch.stop)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def test_fetches_page_text(self):
        result = webfetch.fetch(f"{self.base}/landing")
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["title"], "Fixture")
        self.assertEqual(result["text"], "hello fixture")

    def test_follows_checked_redirects(self):
        result = webfetch.fetch(f"{self.base}/away")
        self.assertEqual(result["status"], 200)
        self.assertTrue(result["url"].endswith("/landing"))

    def test_http_error_returns_as_result(self):
        result = webfetch.fetch(f"{self.base}/gone")
        self.assertEqual(result["status"], 404)

    def test_a_parser_failure_keeps_the_raw_text(self):
        def boom(*args, **kwargs):
            raise ValueError("unparseable")

        with mock.patch.object(webfetch._TextExtractor, "feed", boom):
            result = webfetch.fetch(f"{self.base}/landing")
        self.assertEqual(result["status"], 200)
        self.assertIn("hello fixture", result["text"])
        self.assertNotIn("title", result)

    def test_binary_answers_metadata_only(self):
        result = webfetch.fetch(f"{self.base}/binary")
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["content_type"], "image/png")
        self.assertNotIn("text", result)

    def test_refused_address_raises(self):
        with mock.patch.object(
            webfetch,
            "_resolve",
            return_value=[(socket.AF_INET, 1, 6, "", ("127.0.0.1", 80))],
        ):
            with self.assertRaises(APIError) as caught:
                webfetch.fetch(f"{self.base}/landing")
            self.assertEqual(caught.exception.status, 422)


if __name__ == "__main__":
    unittest.main()
