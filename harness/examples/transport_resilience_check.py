"""End-to-end transport-resilience check: python -m examples.transport_resilience_check.

Reproduces the two transport faults that killed real Test1 tasks and drives the
full ``AgentRunner`` against them through real sockets, a real HTTP server, the
real provider transport and the real shopping harness:

* Scenario A - the first model response is truncated mid-body.  Before the fix
  ``http.client.IncompleteRead`` escaped the provider as a non-provider error and
  the run ended as ``runtime_error``.  Now it is retried and the task completes.
* Scenario B - truncation persists with no retries left.  The run must still end
  as ``provider_error``, i.e. a classified provider failure rather than the
  generic ``runtime_error`` that the batch layer cannot reason about.

The scripted policy is the same offline ``DemoProvider`` used by
``examples.runtime_demo``, so no inference is performed and no API key is needed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from examples.runtime_demo import DemoProvider
from src.data.search_bm25 import build_index, sha256_file
from src.harness.agent import AgentRunner, RuntimeConfig
from src.harness.harness import ShoppingHarness
from src.harness.providers import OllamaProvider


class _Message:
    """Adapts a wire-format message dict to the scripted provider's interface."""

    def __init__(self, payload: dict) -> None:
        self.content = payload.get("content")

    def as_provider_payload(self) -> dict:
        return {"role": "assistant", "content": self.content}


class _ScriptedModelServer:
    """Serves the scripted policy as an OpenAI-compatible endpoint."""

    def __init__(self, truncate_first: int = 0, truncate_always: bool = False) -> None:
        self.truncate_first = truncate_first
        self.truncate_always = truncate_always
        self.policy = DemoProvider()
        self.requests: list[dict] = []
        self.truncated = 0
        # Keyed by the number of completed assistant turns in the request, so a
        # retry of the same logical turn replays the same answer instead of
        # advancing the scripted policy.
        self._responses: dict[int, bytes] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "message_count": len(payload.get("messages") or []),
                    }
                )
                body = outer._response_body(payload)
                truncate = outer.truncate_always or (
                    len(outer.requests) <= outer.truncate_first
                )
                if truncate:
                    outer.truncated += 1
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body[: max(1, len(body) // 2)])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )

    @staticmethod
    def _request_key(payload: dict) -> str:
        """Fingerprint the exact request so a retry replays the same answer."""
        return hashlib.sha256(
            json.dumps(payload.get("messages"), sort_keys=True,
                       ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _response_body(self, payload: dict) -> bytes:
        # The harness truncates history to `recent_turns`, so the number of prior
        # turns cannot identify a logical step. A byte-identical request is a
        # transport retry and must replay; anything else advances the policy.
        key = self._request_key(payload)
        cached = self._responses.get(key)
        if cached is not None:
            return cached
        messages = [_Message(item) for item in payload.get("messages") or []]
        turn = self.policy.complete(messages)
        usage = turn.usage
        body = json.dumps(
            {
                "id": "scripted-1",
                "model": "scripted-policy",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": turn.finish_reason,
                        "message": turn.as_message().as_provider_payload(),
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                },
            }
        ).encode("utf-8")
        self._responses[key] = body
        return body

    def __enter__(self) -> "_ScriptedModelServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
        return False


def _product() -> dict:
    return {
        "asin": "000000000001",
        "tag": "eval",
        "title": "白色静音冰箱",
        "shop_name": "演示商店",
        "category": "家电›冰箱",
        "pricing": [799],
        "attribute": ["静音", "节能"],
        "customization_options": {
            "颜色": [{"value": "白色", "price": 799, "is_available": True}]
        },
        "instructions": [
            {
                "asin": "000000000001",
                "instruction": "寻找一款白色静音冰箱",
                "instruction_options": ["白色"],
                "attributes": ["静音"],
            }
        ],
    }


def _run_scenario(*, max_retries: int, truncate_first: int, truncate_always: bool) -> dict:
    product = _product()
    events: list[tuple[str, dict]] = []
    with tempfile.TemporaryDirectory(prefix="transport-check-") as temporary:
        products = Path(temporary) / "products.json"
        index = Path(temporary) / "catalog.sqlite3"
        products.write_text(json.dumps([product], ensure_ascii=False), encoding="utf-8")
        build_index([product], index, product_data_sha256=sha256_file(products))
        with _ScriptedModelServer(truncate_first, truncate_always) as server:
            provider = OllamaProvider(
                base_url=server.url,
                max_retries=max_retries,
                retry_backoff=0.0,
                sleeper=lambda _delay: None,
                request_observer=lambda kind, payload: events.append(
                    (kind, dict(payload))
                ),
            )
            with ShoppingHarness(products_path=products, index_path=index) as harness:
                task_id = harness.list_task_ids("all")[0]
                result = AgentRunner(
                    provider=provider,
                    harness=harness,
                    config=RuntimeConfig(max_actions=10, recent_turns=1),
                ).run(task_id)
            model_requests = len(server.requests)
            truncated = server.truncated
            loopback_ok = bool(server.requests) and all(
                request["path"] == "/chat/completions" for request in server.requests
            )
    error_types = [
        payload.get("type") for kind, payload in events if kind == "error"
    ]
    return {
        "status": result.status,
        "terminal_reason": result.terminal_reason,
        "strict_success": bool(
            result.session.evaluation and result.session.evaluation.strict_success
        ),
        "actions": result.action_count,
        "model_requests": model_requests,
        "truncated_responses": truncated,
        "error_event_types": error_types,
        "loopback_reached": loopback_ok,
    }


def main() -> int:
    recovered = _run_scenario(max_retries=2, truncate_first=1, truncate_always=False)
    exhausted = _run_scenario(max_retries=0, truncate_first=0, truncate_always=True)

    print(json.dumps({"recovered": recovered, "exhausted": exhausted},
                     ensure_ascii=False, indent=2))

    problems = []
    if recovered["truncated_responses"] != 1:
        problems.append("scenario A did not inject exactly one truncation")
    if "IncompleteRead" not in recovered["error_event_types"]:
        problems.append("scenario A did not observe IncompleteRead as a transport error")
    if not recovered["loopback_reached"]:
        problems.append("scenario A never reached the loopback model server")
    if recovered["status"] != "completed" or not recovered["strict_success"]:
        problems.append(
            "scenario A did not recover: "
            f"{recovered['status']}/{recovered['terminal_reason']}"
        )
    if exhausted["terminal_reason"] != "provider_error":
        problems.append(
            "scenario B was not classified as provider_error: "
            f"{exhausted['terminal_reason']}"
        )
    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        "\nOK: truncated response retried to completion; exhausted truncation "
        "classified as provider_error instead of runtime_error."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
