"""Fetch a public web page as text for the web UI's built-in web_fetch tool.

Local serving has no hosted search or fetch; the chat page offers web_fetch
as a client-owned tool and this endpoint does the network part so the browser
is not bound by cross-origin rules. Only public http(s) addresses are
fetched: a tool request must never reach the LAN or the host itself. The
address check resolves the name once up front, so a hostname that swaps to a
local address between the check and the connect is a residual risk a local
tool accepts.
"""

import contextlib
import ipaddress
import re
import socket
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from .errors import APIError

MAX_URL_CHARS = 2048
MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_TEXT_CHARS = 60_000
FETCH_SECONDS = 30
USER_AGENT = "RichEngine/1.0 (+web_fetch)"

# Content decoded as readable text; anything else (images, archives, PDFs)
# answers with its metadata only.
_TEXT_TYPE = re.compile(
    r"^(text/|application/(json|.+\+json|.+\+xml|xhtml\+xml|javascript|ecmascript))",
    re.IGNORECASE,
)

_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "canvas", "iframe"}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "details",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Leave every redirect to the fetch loop, which re-checks the target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _TextExtractor(HTMLParser):
    """Visible page text: skips script/style payloads, keeps the title, and
    breaks lines where block elements do."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def _collapse(text):
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _resolve(host, port):
    """getaddrinfo behind one seam so tests can answer without patching the
    shared socket module urllib also resolves through."""
    return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)


def _checked_url(url):
    """The URL as a string once it is a public http(s) address, else an
    APIError the request reports as its own status."""
    if not isinstance(url, str) or not url.strip():
        raise APIError(400, '"url" must be a non-empty URL string')
    url = url.strip()
    if len(url) > MAX_URL_CHARS:
        raise APIError(400, '"url" exceeds the length limit')
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise APIError(400, '"url" is not a valid URL') from None
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise APIError(
            422, "web_fetch only retrieves public http(s) URLs", "unsupported"
        )
    if parsed.username is not None or parsed.password is not None:
        raise APIError(422, "URLs carrying credentials are refused", "unsupported")
    try:
        infos = _resolve(
            parsed.hostname,
            port or (443 if parsed.scheme == "https" else 80),
        )
    except socket.gaierror as error:
        raise APIError(502, f"cannot resolve {parsed.hostname}") from error
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise APIError(
                422,
                "web_fetch refuses private, loopback and link-local addresses",
                "unsupported",
            )
    return url


def _read(response):
    try:
        return response.read(MAX_RESPONSE_BYTES + 1)
    except (TimeoutError, OSError) as error:
        raise APIError(504, "fetch timed out", "request_timeout") from error


def _result(url, response):
    status = getattr(response, "status", None) or response.code
    content_type = response.headers.get_content_type()
    result = {"url": url, "status": status, "content_type": content_type}
    if not _TEXT_TYPE.match(content_type):
        result["note"] = "unsupported content type; body not returned"
        return result
    raw = _read(response)
    truncated = len(raw) > MAX_RESPONSE_BYTES
    charset = response.headers.get_content_charset() or "utf-8"
    try:
        text = raw[:MAX_RESPONSE_BYTES].decode(charset, "replace")
    except LookupError:
        text = raw[:MAX_RESPONSE_BYTES].decode("utf-8", "replace")
    if content_type in ("text/html", "application/xhtml+xml"):
        extractor = _TextExtractor()
        # A page too malformed to parse keeps its raw text.
        with contextlib.suppress(Exception):
            extractor.feed(text)
            extractor.close()
        if extractor.parts:
            title = " ".join(extractor.title.split())
            if title:
                result["title"] = title
            text = _collapse("".join(extractor.parts))
    truncated = truncated or len(text) > MAX_TEXT_CHARS
    result["text"] = text[:MAX_TEXT_CHARS]
    if truncated:
        result["truncated"] = True
    return result


def fetch(url, deadline=None):
    """GET a public URL, following checked redirects, and return its text and
    metadata as a JSON-ready dict. Transport failures raise APIError; HTTP
    error statuses come back as a normal result the tool caller can report."""
    if deadline is None:
        deadline = time.monotonic() + FETCH_SECONDS
    current = _checked_url(url)
    opener = urllib.request.build_opener(_NoRedirect())
    for _ in range(MAX_REDIRECTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise APIError(504, "fetch timed out", "request_timeout")
        request = urllib.request.Request(
            current,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,text/plain,"
                "application/json;q=0.9,*/*;q=0.5",
            },
        )
        try:
            response = opener.open(request, timeout=remaining)
        except urllib.error.HTTPError as error:
            if error.code in (301, 302, 303, 307, 308):
                location = error.headers.get("Location")
                if location:
                    current = _checked_url(urljoin(current, location))
                    error.close()
                    continue
            response = error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise APIError(
                502, f"fetch failed: {getattr(error, 'reason', error)}"
            ) from error
        try:
            return _result(current, response)
        finally:
            response.close()
    raise APIError(502, f"fetch exceeded {MAX_REDIRECTS} redirects")
