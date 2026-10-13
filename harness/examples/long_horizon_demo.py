"""Offline paired benchmark using public-context-only scripted decisions and real BM25."""
import argparse
import json
import re
from pathlib import Path
from src.harness.paths import result_path
from src.data.search_bm25 import build_index, sha256_file
from src.harness.tasks import TaskStore
from src.harness.providers import AssistantTurn, ToolCall, TokenUsage
from src.harness.episodes.parser import RuleGoalParser
from src.harness.episodes.preprocess import prepare_episodes
from src.harness.episodes.store import EpisodeStore
from src.harness.episodes.runner import EpisodeRunner
from src.harness.episodes.baselines import run_comparison
from src.harness.episodes.artifacts import replay_episode


class DemoProvider:
    """Fixture policy only. Receives the same public inputs as an LLM, no goal or ID."""
    name = "offline-demo"
    model = "public-context-script-v1"

    def __init__(self):
        self.calls = 0
        self.search_revision = None
        self.hits = []
        self.details = {}
        self.compared = False
        self.verified = False
        self.requests = []

    def complete(self, messages, *, tools=(), tool_choice=None, parameters=None):
        self.requests.append([m.model_dump(mode="json") for m in messages])
        self.calls += 1
        payload = json.loads(messages[1].content)
        task = payload["task_profile"]
        if messages[-1].role == "tool":
            obs = json.loads(messages[-1].content)
            if obs["status"] == "success":
                if obs["tool_name"] == "search_products": self.hits = obs["data"]
                if obs["tool_name"] == "get_product": self.details[obs["data"]["asin"]] = (obs["data"], obs["observation_id"])
        name, arguments = self.decide(task, payload)
        return AssistantTurn(provider=self.name, model=self.model, tool_calls=[ToolCall(id=f"demo-{self.calls}", name=name, arguments=arguments, raw_arguments=json.dumps(arguments, ensure_ascii=False))],
            finish_reason="tool_calls", usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15))

    def decide(self, task, payload):
        messages = [("user_query", None, task["task"]["query"])] + [("user_message", m["message_id"], m["text"]) for m in task["user_messages"]]
        budget_source = next(((kind, mid, re.search(r"预算(\d+)元以内", text)) for kind, mid, text in messages if re.search(r"预算(\d+)元以内", text)), None)
        if task["requirements_status"] != "parsed":
            constraints = [{"constraint_id": "category", "description": "耳机", "source": "user_query", "source_quote": "耳机", "priority": "hard", "topic": "category",
                "predicate": {"field": "category", "op": "contains", "value": "耳机"}}]
            if budget_source:
                kind, mid, match = budget_source
                constraints.append({"constraint_id": "budget", "description": match[0], "source": kind, "source_message_id": mid,
                    "source_quote": match[0], "priority": "hard", "topic": "budget", "predicate": {"field": "price", "op": "le", "value": float(match[1]), "unit": "CNY"}})
            return "update_plan", {"constraints": constraints, "queries": ["耳机"], "change_summary": "根据可见请求更新需求", "uncertain_requirements": [] if budget_source else ["预算"]}
        if self.search_revision != task["revision"]:
            self.search_revision = task["revision"]
            return "search_products", {"query": "耳机", "limit": 5}
        if budget_source is None:
            return "ask_user", {"question": "你的预算是多少？", "information_gap": "需要预算筛选候选"}
        hits = sorted(self.hits, key=lambda p: (p["price_min"], p["asin"]))[:2]
        if not hits: raise ValueError("demo has no eligible products")
        for hit in hits:
            if hit["asin"] not in self.details: return "get_product", {"asin": hit["asin"]}
        if len(hits) > 1 and not self.compared:
            self.compared = True
            return "compare_products", {"asins": [p["asin"] for p in hits]}
        best = hits[0]["asin"]
        if not self.verified:
            self.verified = True
            return "verify_selection", {"asin": best, "selected_options": {}, "max_price": float(budget_source[2][1]), "budget_constraint_id": "budget"}
        candidate = next(c for c in payload["memory"]["candidates"] if c["asin"] == best)
        if not candidate["evidence_ready"]:
            product, observation = self.details[best]
            return "record_evidence_batch", {"assessments": [dict(asin=best, constraint_id=cid, status="satisfied", observation_id=observation,
                field_path="/"+field, quote=str(product[field]), explanation="公开字段支持需求") for cid, field in [("category", "category"), ("budget", "price_min")]]}
        return "submit_recommendations", {"recommendations": [{"asin": best, "selected_options": {}, "reason": "已比较公开价格，所选耳机在预算内。"}]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=result_path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = args.output_dir/"private-data"
    data.mkdir(mode=0o700)
    products = [dict(asin=asin, title="耳机", category="耳机", shop_name="演示商店", pricing=[price], attribute=["耳机"], customization_options={},
        instructions=[dict(asin=asin, instruction=f"我想买耳机，预算{budget}元以内", instruction_options=[], attributes=[])])
        for asin, price, budget in [("earphone-a", 80, 100), ("earphone-b", 90, 150), ("earphone-c", 200, 300)]]
    products_path, index_path = data/"products.json", data/"index.sqlite3"
    products_path.write_text(json.dumps(products, ensure_ascii=False))
    build_index(products, index_path, product_data_sha256=sha256_file(products_path))
    tasks = TaskStore(products_path)
    with EpisodeStore(data/"episodes.sqlite3") as store:
        preparation = prepare_episodes([next(tasks.iter_public_tasks("all"))], RuleGoalParser(), store, dataset_sha256=tasks.product_data_sha256)
        summary = run_comparison(list(store.episodes()), lambda: EpisodeRunner(products_path=products_path, index_path=index_path, provider=DemoProvider()), args.output_dir/"comparison")
    for path in (args.output_dir/"comparison").glob("episode_*/*"):
        replay_episode(path)
    if any(v["strict_success_rate"] != 1 for v in summary["baselines"].values()):
        raise RuntimeError("offline demo did not reach strict success")
    print(json.dumps({"preparation": preparation, "summary": summary, "replay_verified": True,
        "provider": "scripted fixture; no external model quality claim"}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
