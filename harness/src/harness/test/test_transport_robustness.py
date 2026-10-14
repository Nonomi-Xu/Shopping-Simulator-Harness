"""Transport-robustness regression tests for the shared HTTP provider.

These drive the provider through a real local socket against a server that
injects the exact transport faults observed in the Test1 audit, so the fix is
verified end to end rather than against a mocked opener:

* a response truncated mid-body (``http.client.IncompleteRead``), which used to
  escape as a non-provider ``runtime_error`` and kill the run;
* a cross-origin 307, which used to be refused, killing the run with
  ``provider_error``;
* an undecodable envelope, a 429 burst, and a 402 that must not be retried.
"""

from __future__ import annotations

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.request import ProxyHandler, build_opener

from src.harness.providers import DeepSeekAPIError, DeepSeekProvider
from src.harness.providers.base import ProviderError
from src.harness.providers.openai_compat import (
    OpenAICompatibleProvider,
    _is_loopback,
    _origin,
)

_redirect_refusal = OpenAICompatibleProvider._redirect_refusal

#: ``urlopen`` honours the macOS/Windows system proxy, which would forward these
#: loopback requests to an unrelated host and break the test hermetically.
_DIRECT_OPENER = build_opener(ProxyHandler({}))


def _direct_opener(request, *, timeout):
    return _DIRECT_OPENER.open(request, timeout=timeout)


def _completion(content: str = "ok") -> dict:
    return {
        "id": "response-1",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


def _send_json(handler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_truncated(handler, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    # A truthful length followed by fewer bytes: http.client raises IncompleteRead.
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body[: max(1, len(body) // 2)])
    handler.wfile.flush()
    handler.close_connection = True


def _send_redirect(handler, target: str | None, status: int = 307) -> None:
    handler.send_response(status)
    if target is not None:
        handler.send_header("Location", target)
    handler.send_header("Content-Length", "0")
    handler.end_headers()


class _Server:
    """Local HTTP server running a per-request script over real sockets."""

    def __init__(self, script) -> None:
        self.script = script
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:  # silence test output
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                outer.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": body,
                    }
                )
                outer.script(self, len(outer.requests))

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )

    def __enter__(self) -> "_Server":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
        return False


def _provider(base_url: str, **overrides):
    events: list[tuple[str, dict]] = []
    settings = {
        "api_key": "test-secret",
        "base_url": base_url,
        "max_retries": 1,
        "opener": _direct_opener,
        "sleeper": lambda _delay: None,
        "request_observer": lambda kind, payload: events.append((kind, dict(payload))),
    }
    settings.update(overrides)
    return DeepSeekProvider(**settings), events


def _complete(provider):
    from src.harness.providers import ChatMessage

    return provider.complete([ChatMessage(role="user", content="hi")])


class IncompleteReadTests(unittest.TestCase):
    def test_truncated_response_is_retried_and_recovers(self) -> None:
        def script(handler, attempt):
            if attempt == 1:
                _send_truncated(handler, _completion())
            else:
                _send_json(handler, _completion("recovered"))

        with _Server(script) as server:
            provider, events = _provider(server.url, max_retries=2)
            turn = _complete(provider)

        self.assertEqual(turn.content, "recovered")
        self.assertEqual(len(server.requests), 2)
        kinds = [kind for kind, _ in events]
        self.assertIn("error", kinds)
        # The retry must not be masked as a successful first response.
        self.assertEqual(kinds.count("response"), 1)
        error = next(p for k, p in events if k == "error")
        self.assertEqual(error["type"], "IncompleteRead")

    def test_exhausted_truncation_raises_a_provider_error(self) -> None:
        def script(handler, attempt):
            _send_truncated(handler, _completion())

        with _Server(script) as server:
            provider, _ = _provider(server.url, max_retries=0)
            with self.assertRaises(DeepSeekAPIError) as caught:
                _complete(provider)

        # A ProviderError is what lets the orchestrator report provider_error
        # instead of the generic runtime_error that used to lose these runs.
        self.assertIsInstance(caught.exception, ProviderError)
        self.assertIn("truncated", str(caught.exception))
        self.assertEqual(len(server.requests), 1)


class RedirectTests(unittest.TestCase):
    def test_cross_origin_redirect_drops_credentials_and_keeps_post_body(self) -> None:
        with _Server(lambda h, a: _send_json(h, _completion("final"))) as target:
            def script(handler, attempt):
                _send_redirect(handler, f"{target.url}/final")

            with _Server(script) as source:
                provider, events = _provider(source.url)
                turn = _complete(provider)

        self.assertEqual(turn.content, "final")
        self.assertEqual(len(source.requests), 1)
        self.assertEqual(len(target.requests), 1)
        # The POST body must be replayed unchanged for a 307.
        self.assertEqual(source.requests[0]["body"], target.requests[0]["body"])
        self.assertEqual(source.requests[0]["body"] != b"", True)
        # The credential must not follow the redirect off-origin.
        self.assertEqual(source.requests[0]["authorization"], "Bearer test-secret")
        self.assertIsNone(target.requests[0]["authorization"])
        redirects = [p for k, p in events if k == "redirect"]
        self.assertEqual(len(redirects), 1)
        self.assertTrue(redirects[0]["cross_origin"])
        self.assertTrue(redirects[0]["credentials_dropped"])

    def test_same_origin_redirect_keeps_credentials(self) -> None:
        def script(handler, attempt):
            if handler.path == "/chat/completions":
                _send_redirect(handler, "/moved")
            else:
                _send_json(handler, _completion("moved"))

        with _Server(script) as server:
            provider, events = _provider(server.url)
            turn = _complete(provider)

        self.assertEqual(turn.content, "moved")
        self.assertEqual(len(server.requests), 2)
        self.assertEqual(server.requests[1]["path"], "/moved")
        for request in server.requests:
            self.assertEqual(request["authorization"], "Bearer test-secret")
        redirects = [p for k, p in events if k == "redirect"]
        self.assertFalse(redirects[0]["cross_origin"])
        self.assertFalse(redirects[0]["credentials_dropped"])

    def test_missing_location_is_refused_and_recorded(self) -> None:
        with _Server(lambda h, a: _send_redirect(h, None)) as server:
            provider, events = _provider(server.url)
            with self.assertRaises(DeepSeekAPIError):
                _complete(provider)

        refusals = [p for k, p in events if k == "redirect_refused"]
        self.assertEqual(len(refusals), 1)
        self.assertEqual(refusals[0]["reason"], "missing_location")
        self.assertEqual(len(server.requests), 1)


class StatusCodeTests(unittest.TestCase):
    def test_repeated_429_is_retried_within_budget(self) -> None:
        def script(handler, attempt):
            if attempt <= 2:
                _send_json(handler, {"error": {"message": "slow down"}}, status=429)
            else:
                _send_json(handler, _completion("after-429"))

        with _Server(script) as server:
            provider, _ = _provider(server.url, max_retries=3)
            turn = _complete(provider)

        self.assertEqual(turn.content, "after-429")
        self.assertEqual(len(server.requests), 3)

    def test_402_is_not_retried(self) -> None:
        def script(handler, attempt):
            _send_json(
                handler, {"error": {"message": "Insufficient Balance"}}, status=402
            )

        with _Server(script) as server:
            provider, _ = _provider(server.url, max_retries=3)
            with self.assertRaises(DeepSeekAPIError) as caught:
                _complete(provider)

        self.assertEqual(caught.exception.status_code, 402)
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(len(server.requests), 1)


class EnvelopeDecodeTests(unittest.TestCase):
    def test_undecodable_envelope_is_retried(self) -> None:
        def script(handler, attempt):
            if attempt == 1:
                body = b"<html>502 Bad Gateway</html>"
                handler.send_response(200)
                handler.send_header("Content-Type", "text/html")
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)
            else:
                _send_json(handler, _completion("gateway-recovered"))

        with _Server(script) as server:
            provider, _ = _provider(server.url, max_retries=2)
            turn = _complete(provider)

        self.assertEqual(turn.content, "gateway-recovered")
        self.assertEqual(len(server.requests), 2)


class LoopbackProxyTests(unittest.TestCase):
    """A system proxy must never swallow traffic to a local model runtime."""

    def test_default_opener_reaches_loopback_despite_a_proxy(self) -> None:
        with _Server(lambda h, a: _send_json(h, _completion("local"))) as server:
            with patch.dict(
                os.environ,
                {"http_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9"},
            ):
                # No opener injected: the provider builds its own default.
                provider = DeepSeekProvider(
                    api_key="test-secret",
                    base_url=server.url,
                    max_retries=0,
                    sleeper=lambda _delay: None,
                )
                turn = _complete(provider)

        self.assertEqual(turn.content, "local")
        self.assertEqual(len(server.requests), 1)

    def test_remote_hosts_still_use_the_configured_proxy(self) -> None:
        captured: dict = {}

        def script(handler, attempt):
            captured["path"] = handler.path
            _send_json(handler, _completion("via-proxy"))

        with _Server(script) as proxy:
            with patch.dict(
                os.environ,
                {
                    "http_proxy": proxy.url,
                    "HTTP_PROXY": proxy.url,
                    "no_proxy": "",
                    "NO_PROXY": "",
                },
            ):
                provider = DeepSeekProvider(
                    api_key="test-secret",
                    base_url="http://remote.invalid",
                    max_retries=0,
                    sleeper=lambda _delay: None,
                )
                turn = _complete(provider)

        self.assertEqual(turn.content, "via-proxy")
        # A proxied request carries the absolute URI as its request target.
        self.assertEqual(captured["path"], "http://remote.invalid/chat/completions")


class OriginNormalizationTests(unittest.TestCase):
    def test_default_port_matches_explicit_default_port(self) -> None:
        # The old comparison treated these as different origins and refused a
        # legitimate redirect.
        self.assertEqual(
            _origin("https://api.deepseek.com/chat/completions"),
            _origin("https://api.deepseek.com:443/chat/completions"),
        )
        self.assertEqual(_origin("http://localhost:11434/v1"), ("http", "localhost", 11434))
        self.assertEqual(_origin("http://localhost/v1"), ("http", "localhost", 80))
        self.assertIsNone(_origin("ftp://example.com/x"))
        self.assertIsNone(_origin("not a url"))

    def test_host_case_is_normalized(self) -> None:
        self.assertEqual(
            _origin("https://API.DeepSeek.com/x"),
            _origin("https://api.deepseek.com/y"),
        )

    def test_loopback_detection(self) -> None:
        for host in ("localhost", "127.0.0.1", "127.1.2.3", "::1"):
            self.assertTrue(_is_loopback(host), host)
        for host in ("api.deepseek.com", "example.com", "128.0.0.1"):
            self.assertFalse(_is_loopback(host), host)

    def test_refusal_rules(self) -> None:
        https = "https://api.deepseek.com/chat/completions"
        self.assertEqual(
            _redirect_refusal(https, "/next", "https://api.deepseek.com/next",
                              _origin("https://api.deepseek.com/next"), 0),
            None,
        )
        self.assertEqual(
            _redirect_refusal(https, "http://api.deepseek.com/next",
                              "http://api.deepseek.com/next",
                              _origin("http://api.deepseek.com/next"), 0),
            "plaintext_downgrade",
        )
        self.assertEqual(
            _redirect_refusal(https, "ftp://api.deepseek.com/next",
                              "ftp://api.deepseek.com/next", None, 0),
            "unsupported_scheme",
        )
        self.assertEqual(
            _redirect_refusal(https, "https://u:p@api.deepseek.com/next",
                              "https://u:p@api.deepseek.com/next",
                              _origin("https://api.deepseek.com/next"), 0),
            "credentials_in_location",
        )
        self.assertEqual(
            _redirect_refusal(https, None, https, _origin(https), 0),
            "missing_location",
        )
        self.assertEqual(
            _redirect_refusal(https, "/next", "https://api.deepseek.com/next",
                              _origin("https://api.deepseek.com/next"), 3),
            "redirect_limit",
        )
        # Cross-origin plaintext is refused unless the target is loopback, so a
        # local Ollama-style hop still works. From an https source the more
        # specific plaintext-downgrade reason wins.
        self.assertEqual(
            _redirect_refusal(https, "http://other.example.com/x",
                              "http://other.example.com/x",
                              _origin("http://other.example.com/x"), 0),
            "plaintext_downgrade",
        )
        self.assertEqual(
            _redirect_refusal("http://localhost:11434/v1",
                              "http://other.example.com/x",
                              "http://other.example.com/x",
                              _origin("http://other.example.com/x"), 0),
            "cross_origin_plaintext",
        )
        # An https source may never fall back to plaintext, not even to loopback.
        self.assertEqual(
            _redirect_refusal(https, "http://127.0.0.1:11434/x",
                              "http://127.0.0.1:11434/x",
                              _origin("http://127.0.0.1:11434/x"), 0),
            "plaintext_downgrade",
        )
        # A loopback http target on another port stays reachable, so a local
        # Ollama runtime can redirect between localhost and 127.0.0.1.
        self.assertEqual(
            _redirect_refusal("http://localhost:11434/v1",
                              "http://127.0.0.1:11434/v1",
                              "http://127.0.0.1:11434/v1",
                              _origin("http://127.0.0.1:11434/v1"), 0),
            None,
        )


if __name__ == "__main__":
    unittest.main()
