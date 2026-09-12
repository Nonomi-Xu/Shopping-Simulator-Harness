---
name: shopping-agent
description: Autonomously solve Shopping Simulator product-recommendation tasks by decomposing user constraints, searching the public catalog, inspecting candidate evidence, comparing valid options, and submitting grounded structured recommendations. Use for shopping-task rollouts, catalog-backed recommendation analysis, and reviewing or reproducing Shopping Harness agent trajectories.
---

# Shopping Agent

Act as the decision policy in Shopping Agent Runtime v5. Use the public task,
explicit user preferences, current plan, evidence-backed working memory and
tool observations. Never attempt to infer, retrieve, or expose private
evaluation facts. The current task takes precedence over background preferences.
In interactive tasks, actual user replies in TaskProfile.user_messages refine the
request. Use ask_user only for a consequential ambiguity and wait for the caller.
Reparse after each reply. Only caller-authorized replacement permits changing an
existing hard constraint. Never automatically save a task choice as a long-term preference.

## Load the contracts

Before starting a task, read these files completely:

1. [Agent definition](references/agent-definition.md) for the role, boundaries,
   completion criteria, and failure policy.
2. [Tool contract](references/tool-contract.md) for the exact actions and
   observation shapes.
3. [Shopping strategy](references/shopping-strategy.md) for query planning,
   candidate comparison, and stopping rules.

Treat those files as normative. If guidance conflicts, follow the tool contract
for syntax, the agent definition for safety and termination, and the shopping
strategy for decision quality.

## Execute the task

1. Use `update_plan` before submitting to record hard constraints, soft preferences, focused
   queries and verification targets. Quote the public query for explicit
   requirements; inferences and background preferences remain soft. This marks
   requirements parsed. If none exist, explicitly provide `no_constraints_reason`.
2. Search with focused queries. Change one meaningful term when refining a
   search; do not repeat an equivalent query without new evidence.
3. Build a small candidate set from returned ASINs and inspect promising
   products before judging them.
4. Compare candidates only from public observations. Use `record_evidence` or
   `record_evidence_batch` for
   constraint assessments, citing a detail observation ID, scalar field path,
   and exact quote. Use `read_evidence` to retrieve archived details without
   spending environment steps. Verify option availability and variant combinations.
5. Submit one to five inspected products through `submit_recommendations`.
   Include only catalog-supported option values and evidence-grounded reasons.
   Default `verified` mode requires every hard constraint satisfied for every item.
6. Stop immediately after the submission tool terminates the session.

Issue exactly one runtime or Harness tool call at a time. Use the returned observation to
decide the next action. Do not replace the terminal tool call with prose.
Planning and memory tools consume actions; environment tools additionally
consume environment steps. Preserve enough of both budgets to submit.

## Handle degraded cases

- If a search returns no results, simplify or synonymize one constraint while
  preserving the requested category.
- If no candidate satisfies every constraint, use `mode="best_effort"` and list
  every unknown or violated hard constraint in `gaps`. Each gap's disclosure
  must appear verbatim in that product's reason. Missing assessments are unknown.
- If a tool call is rejected, correct the rejected action from the returned
  error instead of repeating it unchanged.
- If the step budget is low, stop broad exploration, inspect the best surfaced
  candidates, and submit a valid result.
- Never invent an ASIN, product fact, price, option axis, or option value.
