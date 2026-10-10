import unittest
from unittest import mock

from server import websearch
from server.errors import APIError

BING_RSS = b"""<?xml version="1.0" encoding="utf-8" ?>
<rss version="2.0"><channel><title>Bing: q</title>
<item><title>First &amp; Result</title><link>https://a.example/</link>
<description>Snippet &lt;b&gt;one&lt;/b&gt; here</description></item>
<item><title>Second</title><link>https://b.example/</link>
<description>Snippet two</description></item>
</channel></rss>"""


class SearchValidationTest(unittest.TestCase):
    def test_rejects_missing_and_blank_queries(self):
        for query in (None, "", "   ", 42):
            with self.assertRaises(APIError) as caught:
                websearch.search(query)
            self.assertEqual(caught.exception.status, 400)

    def test_rejects_an_unknown_provider(self):
        with self.assertRaises(APIError) as caught:
            websearch.search("q", provider="altavista")
        self.assertEqual(caught.exception.status, 400)

    def test_keyed_providers_require_a_key(self):
        for provider in ("brave", "tavily"):
            with self.assertRaises(APIError) as caught:
                websearch.search("q", provider=provider)
            self.assertEqual(caught.exception.status, 400, provider)
            self.assertEqual(caught.exception.code, "missing_key")

    def test_searxng_requires_an_instance(self):
        with self.assertRaises(APIError) as caught:
            websearch.search("q", provider="searxng")
        self.assertEqual(caught.exception.status, 400)

    def test_searxng_rejects_non_http_instances(self):
        for instance in ("file:///etc", "not a url"):
            with self.assertRaises(APIError) as caught:
                websearch.search("q", provider="searxng", instance=instance)
            self.assertEqual(caught.exception.status, 400, instance)


class BingSearchTest(unittest.TestCase):
    def test_parses_rss_items(self):
        with mock.patch.object(websearch, "_get", return_value=BING_RSS) as get:
            result = websearch.search("apple silicon", provider="bing")
        url = get.call_args[0][0]
        self.assertTrue(url.startswith("https://www.bing.com/search?"))
        self.assertIn("q=apple+silicon", url)
        self.assertEqual(result["provider"], "bing")
        self.assertEqual(
            result["results"],
            [
                {
                    "title": "First & Result",
                    "url": "https://a.example/",
                    "snippet": "Snippet one here",
                },
                {
                    "title": "Second",
                    "url": "https://b.example/",
                    "snippet": "Snippet two",
                },
            ],
        )

    def test_bad_feed_is_a_502(self):
        with mock.patch.object(websearch, "_get", return_value=b"<html>"):
            with self.assertRaises(APIError) as caught:
                websearch.search("q", provider="bing")
            self.assertEqual(caught.exception.status, 502)


class KeyedSearchTest(unittest.TestCase):
    def test_brave_sends_the_key_and_maps_results(self):
        payload = {
            "web": {
                "results": [
                    {"title": "T", "url": "https://x.example/", "description": "D"}
                ]
            }
        }
        with mock.patch.object(
            websearch, "_get", return_value=__import__("json").dumps(payload).encode()
        ) as get:
            result = websearch.search("q", provider="brave", api_key="k")
        request_url = get.call_args[0][0]
        self.assertTrue(request_url.startswith("https://api.search.brave.com/"))
        self.assertEqual(get.call_args[1]["headers"]["X-Subscription-Token"], "k")
        self.assertEqual(
            result["results"],
            [{"title": "T", "url": "https://x.example/", "snippet": "D"}],
        )

    def test_tavily_posts_key_in_body(self):
        import json

        payload = {"results": [{"title": "T", "url": "u", "content": "C"}]}
        with mock.patch.object(
            websearch, "_post", return_value=json.dumps(payload).encode()
        ) as post:
            result = websearch.search("q", provider="tavily", api_key="k")
        self.assertEqual(post.call_args[0][0], "https://api.tavily.com/search")
        self.assertEqual(post.call_args[0][1]["api_key"], "k")
        self.assertEqual(
            result["results"], [{"title": "T", "url": "u", "snippet": "C"}]
        )

    def test_searxng_queries_the_instance(self):
        import json

        payload = {"results": [{"title": "T", "url": "u", "content": "C"}]}
        with mock.patch.object(
            websearch, "_get", return_value=json.dumps(payload).encode()
        ) as get:
            result = websearch.search(
                "q", provider="searxng", instance="http://localhost:8888/"
            )
        self.assertEqual(
            get.call_args[0][0], "http://localhost:8888/search?q=q&format=json"
        )
        self.assertEqual(
            result["results"], [{"title": "T", "url": "u", "snippet": "C"}]
        )


if __name__ == "__main__":
    unittest.main()
