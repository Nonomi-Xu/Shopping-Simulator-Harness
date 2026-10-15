"""Stateless, explicitly synthetic headphone demo for local Studio smoke tests."""
import json

from ..providers import AssistantTurn, ToolCall, TokenUsage, canonical_json


class StudioDemoProvider:
    name = "studio-demo"
    model = "scripted-headphone-demo"

    def complete(self, messages, *, tools=(), tool_choice=None, parameters=None):
        state = json.loads(messages[1].content)
        task = state["task_profile"]
        allowed = {t.name for t in tools}
        text = task["user_messages"][-1]["text"] if task["user_messages"] else task["task"]["query"]
        color = next((c for c in ("白色", "黑色") if c in text), None)
        candidates = state["memory"]["candidates"]
        candidate = next((c for c in candidates if color and color in c["summary"].get("title", "")), None)
        if "update_plan" in allowed:
            if color is None:
                name, args = "ask_user", {
                    "question": "演示耳机要白色还是黑色？", "options": ["白色", "黑色"],
                    "information_gap": "需要选择演示商品的颜色", "constraint_ids": [],
                }
            else:
                constraint = {"constraint_id": "color", "description": f"需要{color}",
                              "priority": "hard", "source_quote": color,
                              "source": "user_message" if task["user_messages"] else "user_query"}
                if task["user_messages"]:
                    constraint["source_message_id"] = task["user_messages"][-1]["message_id"]
                name, args = "update_plan", {"constraints": [constraint], "queries": ["耳机"],
                                              "change_summary": "离线演示：提取颜色需求"}
        elif "review_progress" in allowed:
            if not candidate or not candidate["inspected"]:
                name, args = "review_progress", {"next_step": "act", "summary": "检查候选详情"}
            elif not candidate["evidence_ready"]:
                name, args = "record_evidence", {
                    "asin": candidate["asin"], "constraint_id": "color", "status": "satisfied",
                    "observation_id": candidate["detail_observation_id"], "field_path": "/attributes/0",
                    "quote": color, "explanation": "演示商品公开详情注明颜色",
                }
            else:
                name, args = "review_progress", {"next_step": "recommend", "summary": "颜色证据已核验",
                                                 "reviewed_asins": [candidate["asin"]]}
        elif "submit_recommendations" in allowed:
            name, args = "submit_recommendations", {"recommendations": [{
                "asin": candidate["asin"], "selected_options": {"颜色": color},
                "reason": f"离线演示：公开详情证据支持{color}要求。",
            }]}
        elif not candidate:
            name, args = "search_products", {"query": "耳机", "method": "bm25"}
        else:
            name, args = "get_product", {"asin": candidate["asin"]}
        return AssistantTurn(provider=self.name, model=self.model, finish_reason="tool_calls",
                             tool_calls=[ToolCall(id=f"demo-{state['budget']['actions_remaining']}",
                                                  name=name, arguments=args, raw_arguments=canonical_json(args))],
                             usage=TokenUsage(input_tokens=10, output_tokens=4, total_tokens=14))
