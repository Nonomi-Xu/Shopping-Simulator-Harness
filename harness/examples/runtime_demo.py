"""Offline six-action demonstration: python -m examples.runtime_demo."""

import argparse
import json
from pathlib import Path
from src.harness.paths import result_path
import tempfile

from src.data.search_bm25 import build_index, sha256_file
from src.harness.agent import (
    AgentRunner,
    RuntimeConfig,
    replay_run,
    write_run_artifacts,
)
from src.harness.harness import ShoppingHarness
from src.harness.providers import AssistantTurn, TokenUsage, ToolCall, canonical_json


class DemoProvider:
    """Scripted responses for a synthetic catalogue; makes no network requests."""

    name = "offline-demo"
    model = "scripted-policy"

    def __init__(self):
        self.stage = 0
        self.asin = None

    def complete(self, messages, *, tools=(), tool_choice=None, parameters=None):
        self.stage += 1
        if self.stage == 1:
            name, arguments = "update_plan", {
                "constraints": [
                    {
                        "constraint_id": "quiet",
                        "description": "需要静音",
                        "priority": "hard",
                        "source": "user_query",
                        "source_quote": "静音",
                    },
                    {
                        "constraint_id": "color",
                        "description": "需要白色",
                        "priority": "hard",
                        "source": "user_query",
                        "source_quote": "白色",
                    },
                ],
                "queries": ["白色静音冰箱"],
                "verification_targets": ["静音属性与白色规格"],
                "change_summary": "将公开需求整理成查询和核验计划",
                "evidence_observation_ids": [],
            }
        elif self.stage == 2:
            name, arguments = "search_products", {"query": "白色静音冰箱", "limit": 5}
        elif self.stage == 3:
            observation = json.loads(messages[-1].content)
            self.asin = observation["data"][0]["asin"]
            name, arguments = "get_product", {"asin": self.asin}
        elif self.stage == 4:
            name, arguments = "read_evidence", {
                "asin": self.asin,
                "field_paths": ["/attributes/0", "/customization_options/颜色/0/value"],
            }
        elif self.stage == 5:
            evidence = json.loads(messages[-1].content)["data"]
            name, arguments = "record_evidence_batch", {
                "assessments": [
                    {
                        "asin": self.asin,
                        "constraint_id": constraint,
                        "status": "satisfied",
                        "observation_id": evidence["source_observation_id"],
                        "field_path": fragment["field_path"],
                        "quote": fragment["value"],
                        "explanation": "公开详情支持该需求",
                    }
                    for constraint, fragment in zip(
                        ("quiet", "color"), evidence["items"]
                    )
                ]
            }
        else:
            name, arguments = "submit_recommendations", {
                "recommendations": [
                    {
                        "asin": self.asin,
                        "selected_options": {"颜色": "白色"},
                        "reason": "已检查公开属性中的静音说明，以及可用的白色规格。",
                    }
                ]
            }
        return AssistantTurn(
            provider=self.name,
            model=self.model,
            tool_calls=[
                ToolCall(
                    id=f"demo-{self.stage}",
                    name=name,
                    arguments=arguments,
                    raw_arguments=canonical_json(arguments),
                )
            ],
            finish_reason="tool_calls",
            # Synthetic usage, not a measurement of an actual model.
            usage=TokenUsage(input_tokens=10, output_tokens=4, total_tokens=14),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=result_path, default=result_path("outputs/runtime-demo"))
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new directory")
    product = {
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
    with tempfile.TemporaryDirectory(prefix="shopping-runtime-demo-") as temporary:
        products = Path(temporary) / "products.json"
        index = Path(temporary) / "catalog.sqlite3"
        products.write_text(json.dumps([product], ensure_ascii=False), encoding="utf-8")
        build_index([product], index, product_data_sha256=sha256_file(products))
        with ShoppingHarness(products_path=products, index_path=index) as harness:
            task_id = harness.list_task_ids("all")[0]
            result = AgentRunner(
                provider=DemoProvider(),
                harness=harness,
                config=RuntimeConfig(max_actions=10, recent_turns=1),
            ).run(task_id)
        directory = write_run_artifacts(result, args.output_dir)
        replayed = replay_run(directory)
    if result.status != "completed" or not result.session.evaluation.strict_success:
        raise RuntimeError(result.error_message or result.terminal_reason)
    print(
        json.dumps(
            {
                "output_dir": str(directory),
                "status": result.status,
                "strict_success": result.session.evaluation.strict_success,
                "actions": result.action_count,
                "environment_calls": result.tool_call_count,
                "searches": result.search_count,
                "events": len(result.events),
                "replay_verified": replayed["memory"]
                == result.memory.model_dump(mode="json"),
                "provider": "offline scripted demo; no real inference",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
