"""Offline interactive demo: pause, persist, restore and revise a recommendation."""

import argparse
import json
from pathlib import Path
from src.harness.paths import result_path

from src.harness.agent import (
    AgentRunner,
    AgentRunResult,
    RuntimeConfig,
    UserReply,
    replay_run,
    write_run_artifacts,
)
from src.harness.branch.interactive import InteractiveHarness
from src.harness.branch.preferences import PreferenceStore
from src.harness.sources import JsonProductSource
from src.harness.providers import AssistantTurn, ToolCall, TokenUsage, canonical_json


class DemoProvider:
    name = "interactive-offline-demo"
    model = "scripted-policy"

    def __init__(self, stage=0):
        self.stage = stage

    def complete(self, messages, *, tools=(), tool_choice=None, parameters=None):
        self.stage += 1
        color = "白色" if self.stage <= 5 else "黑色"
        asin = "white-headphones" if self.stage <= 5 else "black-headphones"
        if self.stage in {1, 6}:
            constraint = {
                "constraint_id": "color",
                "description": f"需要{color}",
                "priority": "hard",
                "source": "user_query" if self.stage == 1 else "user_message",
                "source_quote": color,
            }
            if self.stage == 6:
                constraint["source_message_id"] = "user_0001"
            name, args = "update_plan", {
                "constraints": [constraint],
                "queries": ["耳机"],
                "change_summary": "记录用户当前颜色要求",
            }
        elif self.stage == 2:
            name, args = "search_products", {"query": "耳机"}
        elif self.stage in {3, 7}:
            name, args = "get_product", {"asin": asin}
        elif self.stage in {4, 8}:
            observation = json.loads(messages[-1].content)
            name, args = "record_evidence", {
                "asin": asin,
                "constraint_id": "color",
                "status": "satisfied",
                "observation_id": observation["observation_id"],
                "field_path": "/attributes/0",
                "quote": color,
                "explanation": "公开详情注明颜色",
            }
        elif self.stage == 5:
            name, args = "ask_user", {
                "question": "颜色是否需要调整？",
                "information_gap": "演示用户修改最终颜色",
                "constraint_ids": ["color"],
                "options": ["保留白色", "改成黑色"],
            }
        else:
            name, args = "submit_recommendations", {
                "recommendations": [
                    {
                        "asin": asin,
                        "selected_options": {"颜色": color},
                        "reason": "按你最新要求选择黑色，可用规格的目录价格为199元。",
                    }
                ]
            }
        return AssistantTurn(
            provider=self.name,
            model=self.model,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id=f"demo-{self.stage}",
                    name=name,
                    arguments=args,
                    raw_arguments=canonical_json(args),
                )
            ],
            usage=TokenUsage(input_tokens=10, output_tokens=4, total_tokens=14),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=result_path, default=result_path("outputs/stage-three-demo")
    )
    args = parser.parse_args(argv)
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    products = Path(__file__).with_name("public-products.json")
    preferences = PreferenceStore(root / "preferences.sqlite3")
    preferences.set("demo-user", "颜色偏好", "通常喜欢白色", expected_revision=0)
    profile, _ = preferences.load("demo-user")
    config = RuntimeConfig(
        interaction_mode="interactive", recent_turns=1, max_actions=12
    )
    with InteractiveHarness(JsonProductSource(products, currency="CNY")) as harness:
        paused = AgentRunner(
            provider=DemoProvider(),
            harness=harness,
            user_profile=profile,
            config=config,
        ).run_query("寻找白色耳机", session_id="interactive-demo")
        if paused.status != "waiting_for_user":
            raise RuntimeError(paused.error_message or paused.terminal_reason)
        write_run_artifacts(paused, root / "waiting")
    replay_run(root / "waiting")
    # New environment and provider objects simulate a fresh process; no old live state is used.
    saved = AgentRunResult.model_validate_json(
        (root / "waiting/checkpoint.json").read_text()
    )
    with InteractiveHarness(JsonProductSource(products, currency="CNY")) as harness:
        result = AgentRunner(
            provider=DemoProvider(stage=5),
            harness=harness,
            user_profile=saved.user_profile,
            config=config,
        ).resume(saved, UserReply(text="这次改成黑色", replace_constraint_ids=["color"]))
        if result.status != "completed":
            raise RuntimeError(result.error_message or result.terminal_reason)
        write_run_artifacts(result, root / "completed")
    replayed = replay_run(root / "completed")
    print(
        json.dumps(
            {
                "output_dir": str(root),
                "status": result.status,
                "recommendations": result.session.history[-1].arguments[
                    "recommendations"
                ],
                "task_revision": result.task_profile.revision,
                "actions": result.action_count,
                "environment_calls": result.tool_call_count,
                "evaluation": None,
                "long_term_preference": preferences.load("demo-user")[0]
                .preferences[0]
                .value,
                "replay_verified": replayed["memory"]
                == result.memory.model_dump(mode="json"),
                "provider": "scripted; no external API or measured model quality",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
