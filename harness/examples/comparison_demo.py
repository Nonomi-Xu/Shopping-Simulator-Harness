"""Offline phase-two demo: filtered discovery, field reads, comparison and selection."""

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
from src.harness.providers import AssistantTurn, ToolCall


class ComparisonProvider:
    name = "offline-comparison-demo"
    model = "scripted-policy"

    def __init__(self):
        self.stage = 0

    def complete(self, messages, **kwargs):
        self.stage += 1
        observation = (
            json.loads(messages[-1].content) if messages[-1].role == "tool" else None
        )
        if self.stage == 1:
            name, args = "update_plan", {
                "constraints": [
                    {
                        "constraint_id": cid,
                        "description": text,
                        "priority": "hard",
                        "source": "user_query",
                        "source_quote": text,
                    }
                    for cid, text in [
                        ("quiet", "静音"),
                        ("color", "白色"),
                        ("budget", "预算800以内"),
                    ]
                ],
                "queries": ["白色静音冰箱"],
                "change_summary": "核验属性、颜色和所选规格价格",
            }
        elif self.stage == 2:
            name, args = "search_products", {
                "query": "白色静音冰箱",
                "limit": 5,
                "filters": {"category": "冰箱", "max_price": 800},
            }
        elif self.stage == 3:
            self.asins = [row["asin"] for row in observation["data"]]
            name, args = "get_product", {
                "asin": self.asins[0],
                "fields": [
                    "attributes",
                    "customization_options",
                    "variant_combinations",
                    "price_min",
                    "price_max",
                ],
            }
        elif self.stage == 4:
            self.detail_id = observation["observation_id"]
            name, args = "get_product", {
                "asin": self.asins[1],
                "fields": [
                    "attributes",
                    "customization_options",
                    "variant_combinations",
                    "price_min",
                    "price_max",
                ],
            }
        elif self.stage == 5:
            name, args = "compare_products", {
                "asins": self.asins[:2],
                "fields": ["attributes", "price_min", "customization_options"],
            }
        elif self.stage == 6:
            name, args = "verify_selection", {
                "asin": self.asins[0],
                "selected_options": {"颜色": "白色"},
                "max_price": 800,
                "budget_constraint_id": "budget",
            }
        elif self.stage == 7:
            if observation["data"]["budget_status"] != "satisfied":
                raise RuntimeError("demo selection unexpectedly exceeds budget")
            name, args = "read_evidence", {
                "asin": self.asins[0],
                "observation_id": self.detail_id,
                "field_paths": [
                    "/attributes/0",
                    "/customization_options/颜色/0/value",
                    "/customization_options/颜色/0/price",
                ],
            }
        elif self.stage == 8:
            evidence = observation["data"]
            name, args = "record_evidence_batch", {
                "assessments": [
                    {
                        "asin": self.asins[0],
                        "constraint_id": cid,
                        "status": "satisfied",
                        "observation_id": evidence["source_observation_id"],
                        "field_path": item["field_path"],
                        "quote": (
                            item["value"]
                            if isinstance(item["value"], str)
                            else json.dumps(item["value"])
                        ),
                        "explanation": "公开详情与规格核验支持该需求",
                    }
                    for cid, item in zip(
                        ("quiet", "color", "budget"), evidence["items"]
                    )
                ]
            }
        else:
            name, args = "submit_recommendations", {
                "recommendations": [
                    {
                        "asin": self.asins[0],
                        "selected_options": {"颜色": "白色"},
                        "reason": "公开属性注明静音，白色规格可用，所选规格价格799，满足800以内预算。",
                    }
                ]
            }
        return AssistantTurn(
            provider=self.name,
            model=self.model,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id=f"comparison-{self.stage}",
                    name=name,
                    arguments=args,
                    raw_arguments=json.dumps(args, ensure_ascii=False),
                )
            ],
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=result_path, default=result_path("outputs/comparison-demo")
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory exists; select a new directory")
    products = [
        {
            "asin": "000000000001",
            "title": "白色静音冰箱",
            "category": "家电›冰箱",
            "pricing": [799],
            "attribute": ["静音", "节能"],
            "customization_options": {"颜色": [{"value": "白色", "price": 799}]},
            "instructions": [
                {
                    "asin": "000000000001",
                    "instruction": "寻找白色静音冰箱，预算800以内",
                    "instruction_options": ["白色"],
                    "attributes": ["静音"],
                }
            ],
        },
        {
            "asin": "000000000002",
            "title": "白色普通冰箱",
            "category": "家电›冰箱",
            "pricing": [699],
            "attribute": ["小容量"],
            "customization_options": {"颜色": [{"value": "白色", "price": 699}]},
            "instructions": [
                {
                    "asin": "000000000002",
                    "instruction": "寻找普通白色冰箱",
                    "instruction_options": ["白色"],
                    "attributes": [],
                }
            ],
        },
    ]
    with tempfile.TemporaryDirectory(prefix="comparison-demo-") as temporary:
        source, index = (
            Path(temporary) / "products.json",
            Path(temporary) / "catalog.sqlite3",
        )
        source.write_text(json.dumps(products, ensure_ascii=False), encoding="utf-8")
        build_index(products, index, product_data_sha256=sha256_file(source))
        with ShoppingHarness(products_path=source, index_path=index) as harness:
            result = AgentRunner(
                provider=ComparisonProvider(),
                harness=harness,
                config=RuntimeConfig(recent_turns=1),
            ).run(harness.list_task_ids()[0])
        if result.status != "completed" or not result.session.evaluation.strict_success:
            raise RuntimeError(result.error_message or result.terminal_reason)
        directory = write_run_artifacts(result, args.output_dir)
        replayed = replay_run(directory)
    print(
        json.dumps(
            {
                "output_dir": str(directory),
                "status": result.status,
                "strict_success": result.session.evaluation.strict_success,
                "actions": result.action_count,
                "environment_calls": result.tool_call_count,
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
