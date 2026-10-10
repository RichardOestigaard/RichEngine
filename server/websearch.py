"""Search the web for the web UI's built-in web_search tool.

The provider is the page's choice, sent per request: bing reads its public
RSS endpoint (no key), brave and tavily take the caller's API key, and
searxng queries the caller's self-hosted instance, which is the one provider
a private address is fine for — a local instance is the point.
"""

import contextlib
import json
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode, urlsplit
from xml.etree import ElementTree

from .errors import APIError

PROVIDERS = ("bing", "brave", "tavily", "searxng")
KEYED_PROVIDERS = ("brave", "tavily")

MAX_QUERY_CHARS = 1000
MAX_RESULTS = 10
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
SEARCH_SECONDS = 20
USER_AGENT = "RichEngine/1.0 (+web_search)"

_TAG = re.compile(r"<[^>]+>")


def _open(request, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise APIError(504, "search timed out", "request_timeout")
    try:
        return urllib.request.urlopen(request, timeout=remaining)
    except urllib.error.HTTPError as error:
        detail = ""
        with contextlib.suppress(Exception):
            detail = error.read(4096).decode("utf-8", "replace")
        raise APIError(
            502, f"search provider answered HTTP {error.code}: {detail[:200]}"
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise APIError(
            502, f"search failed: {getattr(error, 'reason', error)}"
        ) from error


def _get(url, deadline, headers=None):
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **(headers or {})}
    )
    with _open(request, deadline) as response:
        return response.read(MAX_RESPONSE_BYTES)


def _post(url, payload, deadline, headers=None):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            **(headers or {}),
        },
    )
    with _open(request, deadline) as response:
        return response.read(MAX_RESPONSE_BYTES)


def _clean(text):
    return re.sub(r"\s+", " ", _TAG.sub("", text or "")).strip()


def _bing(query, count, deadline):
    raw = _get(
        "https://www.bing.com/search?"
        + urlencode({"q": query, "format": "rss", "count": count}),
        deadline,
    )
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as error:
        raise APIError(502, "bing returned unreadable RSS") from error
    return [
        {
            "title": _clean(item.findtext("title")),
            "url": _clean(item.findtext("link")),
            "snippet": _clean(item.findtext("description")),
        }
        for item in root.iter("item")
    ]


def _brave(query, count, key, deadline):
    raw = _get(
        "https://api.search.brave.com/res/v1/web/search?"
        + urlencode({"q": query, "count": count}),
        deadline,
        headers={
            "X-Subscription-Token": key,
            "Accept": "application/json",
        },
    )
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise APIError(502, "brave returned invalid JSON") from error
    return [
        {
            "title": _clean(item.get("title")),
            "url": item.get("url", ""),
            "snippet": _clean(item.get("description")),
        }
        for item in data.get("web", {}).get("results", [])
        if isinstance(item, dict)
    ]


def _tavily(query, count, key, deadline):
    raw = _post(
        "https://api.tavily.com/search",
        {"query": query, "max_results": count, "api_key": key},
        deadline,
    )
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise APIError(502, "tavily returned invalid JSON") from error
    return [
        {
            "title": _clean(item.get("title")),
            "url": item.get("url", ""),
            "snippet": _clean(item.get("content")),
        }
        for item in data.get("results", [])
        if isinstance(item, dict)
    ]


def _searxng(query, count, instance, deadline):
    """A caller-chosen instance URL: self-hosted searxng typically runs on a
    private address, so the private-address refusal does not apply here —
    the model never names the host, only the query."""
    if not isinstance(instance, str) or not instance.strip():
        raise APIError(400, '"instance" must name the SearXNG base URL')
    instance = instance.strip()
    if len(instance) > MAX_QUERY_CHARS:
        raise APIError(400, '"instance" exceeds the length limit')
    try:
        parsed = urlsplit(instance)
    except ValueError:
        raise APIError(400, '"instance" is not a valid URL') from None
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise APIError(400, '"instance" must be an http(s) URL')
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
    raw = _get(
        f"{base}/search?" + urlencode({"q": query, "format": "json"}),
        deadline,
    )
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise APIError(
            502, "instance returned invalid JSON (is this a SearXNG server?)"
        ) from error
    return [
        {
            "title": _clean(item.get("title")),
            "url": item.get("url", ""),
            "snippet": _clean(item.get("content")),
        }
        for item in data.get("results", [])
        if isinstance(item, dict)
    ]


def search(query, provider="bing", api_key="", instance="", count=8):
    """Query `provider` for `query` and return a JSON-ready result:
    `{provider, results: [{title, url, snippet}]}`."""
    if not isinstance(query, str) or not query.strip():
        raise APIError(400, '"query" must be a non-empty string')
    query = query.strip()[:MAX_QUERY_CHARS]
    if provider not in PROVIDERS:
        raise APIError(
            400,
            f'"provider" must be one of {", ".join(PROVIDERS)}',
        )
    if not isinstance(api_key, str):
        raise APIError(400, '"api_key" must be a string')
    if not isinstance(count, int) or isinstance(count, bool):
        raise APIError(400, '"count" must be an integer')
    count = max(1, min(count, MAX_RESULTS))
    deadline = time.monotonic() + SEARCH_SECONDS
    if provider == "bing":
        results = _bing(query, count, deadline)
    elif provider == "searxng":
        results = _searxng(query, count, instance, deadline)
    else:
        if not api_key.strip():
            raise APIError(
                400, f'"{provider}" search requires an API key', "missing_key"
            )
        results = (_brave if provider == "brave" else _tavily)(
            query, count, api_key.strip(), deadline
        )
    return {"provider": provider, "results": results[:count]}
