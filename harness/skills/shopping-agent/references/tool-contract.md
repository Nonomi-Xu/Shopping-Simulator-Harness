# Harness Tool Contract

Use exactly one tool call per assistant turn. Names and arguments are
case-sensitive; unknown fields are invalid. The provider receives only the
tools available in the current state. Every attempted tool call consumes one
runtime action. Only dispatched environment calls consume a Harness step.

## Runtime tools

### `update_plan`

Record a concise plan, not an internal reasoning transcript:

```json
{
  "constraints": [
    {
      "constraint_id": "quiet",
      "description": "商品需要具备静音属性",
      "priority": "hard",
      "source": "user_query",
      "source_quote": "静音"
    }
  ],
  "queries": ["白色静音冰箱"],
  "verification_targets": ["静音属性", "白色规格是否可用"],
  "change_summary": "根据当前需求制定查询与核验目标",
  "evidence_observation_ids": []
}
```

Sources are `user_query`, `user_profile`, `user_message`, or `inferred`. Query constraints must
quote the current public query. Profile constraints must quote a supplied
preference value. Profile and inferred constraints must be soft. Constraint IDs
are unique. An update replaces the constraint list, but existing hard
constraints must remain identical. Supply known observation IDs when results
motivate a replan. Changed soft constraints invalidate earlier assessments.
For user_message, source_message_id must identify an authentic TaskProfile user
reply and source_quote must occur in that reply. A caller may explicitly remove
constraints when resuming; preserve every remaining hard constraint. The original
query remains available alongside the revision history.

### `ask_user` (interactive mode only)

Pass a nonblank question and information_gap, optional constraint_ids, and up to
five suggested options. Ask only when the answer could change the recommendation.
This consumes one action, returns a pending_question with question_id, and pauses
the runtime. It does not consume an environment step. Do not answer your own
question or treat the first suggested option as accepted. The host resumes with
a real UserReply. Offline benchmark mode does not expose this tool.

After a reply, requirements become unparsed. Call update_plan again; quote replies
using source=user_message. Reuse archived public facts, but reassess constraints
the user replaced. Refreshed external details clear previous candidate judgments;
changed fields cannot be justified using superseded citations. Source metadata
identifies the observation time and currency, not a live checkout guarantee.

A bootstrap plan exists before the first call; use this tool to refine it.
Its requirements_status is initially unparsed. A successful update marks it
parsed. If constraints is empty, no_constraints_reason must be a nonblank
explanation. Parsing and cited assessments are model judgments, not semantic proof.
The runtime advances its phase from search to inspect, verify, and submit using
public state. Phase is guidance, not permission to skip tool guards.

### `record_evidence`

Record one model assessment with a verifiable citation:

```json
{
  "asin": "an-inspected-asin",
  "constraint_id": "quiet",
  "status": "satisfied",
  "observation_id": "the-exact-detail-observation-id",
  "field_path": "/attributes/0",
  "quote": "静音",
  "explanation": "商品公开属性包含静音说明"
}
```

Status is `satisfied`, `violated`, or `unknown`. The constraint must exist in the
current task profile. The source must be a successful get_product observation
for this exact ASIN. field_path is an RFC 6901 JSON Pointer into observation.data
and must resolve to a scalar field. The quote must occur in that field.
Container-level quotes and citations to another product are rejected.

The runtime verifies source identity and quotation, not semantic entailment.
Keep unknown facts unknown. Model assessments never become evaluator labels.

### `record_evidence_batch`

Pass `{"assessments": [...]}` containing 1-30 objects with the exact
`record_evidence` fields. Each ASIN/constraint pair must be unique. All citations
are validated before any assessment is changed; one invalid item rejects the
whole batch. This costs one runtime action and zero environment steps.

### `read_evidence`

```json
{"asin": "an-inspected-asin", "field_paths": ["/attributes", "/customization_options"], "offset": 0, "limit": 10}
```

Reads only details previously returned in this session. Optional observation_id
selects a particular successful original detail observation; otherwise the latest
detail is used. Omit field_paths to traverse all public fields. Each path may
identify a scalar or a container. The response flattens containers into scalar
fragments and returns source_observation_id, items, total_fragments and next_offset.

Each item has field_path and value. Long strings are split into exact fragments
of up to 1000 characters, with character_offset and total_characters. At most 20
fragments and 12000 serialized data characters are returned. Continue with the
same ASIN, observation_id, field_paths and next_offset as offset. A null
next_offset ends the selected traversal; it does not imply that unrelated fields
were examined. Cite source_observation_id and the fragment's field_path when
recording evidence. This costs one action and zero environment steps.

## Environment tools

### `compare_products` (runtime)

Pass 2-5 inspected asins, up to 8 public top-level fields and up to 10 constraint_ids.
Example: {"asins": ["A", "B"], "fields": ["attributes", "price_min", "customization_options"]}.
Returns aligned observed values, differences and existing model assessments with original
citations. Unobserved fields remain unknown. Large values are marked previews; recover
exact text with read_evidence. Default constraint selection is the first ten and reports omissions.
This costs one runtime action, no environment step, and never provides an authoritative ranking.

### `verify_selection` (runtime)

Pass asin, selected_options and optionally max_price and budget_constraint_id.
It checks observed option availability, combination compatibility and selected price.
It never substitutes product price_min for a multi-variant selection. Incomplete selections,
missing/conflicting prices, or independent multi-axis option prices return unknown price.
With no combination table, multi-axis combination availability remains unknown.
Price evidence identifies original observed scalar fields. max_price is your interpretation
in catalog price units; a constraint ID validates existence, not semantic correctness of the threshold.
The tool does not automatically record a semantic assessment. Budget checks attached to hard
constraints are rechecked at submission; unknown/violated results require repair or best_effort gaps.
This costs one action, no environment step. Selection fields not yet observed must be fetched first.

### `search_products`

The executable local contract currently accepts method="bm25" (also the default).
Typed filters.predicates are supported by the SQLite episode catalogue. Each predicate
has field, optional key, op, value and optional unit. They are evaluated before Top-K;
unknown facts do not pass. Episode execution automatically preserves the actor's typed
hard requirements in applied filters. See observation.projection.applied_filters for
the actual filter request. Legacy price ranges retain their documented overlap semantics.

```json
{"query": "白色静音冰箱", "limit": 20}
```

query is non-empty; limit is optional and ranges from 1 to 50. Returns an ordered
array of public summaries: asin, title, category, brand, price_min, price_max,
bm25_score, rank. An empty array is a successful result. Surfaced ASINs become
eligible for get_product. Price ranges are discovery evidence, not confirmed
prices of the user's selected variant.

The search budget limits accepted search attempts. Equivalent normalized query
repetition is bounded independently, including attempts to vary only limit.

Optional filters contains category (normalized substring), brand (normalized exact match),
min_price, max_price and include_unknown_price (default true). Price filters test product-range
overlap, not selected-option price. offset starts at zero; exclude_asins accepts up to 200 IDs.
Filtering occurs before limit. Keep query/filters/exclusions fixed while paging with projection.next_offset.
projection.has_more is explicit; total_results remains unknown. Only returned ASINs are surfaced.
Each result may include up to three match_snippets with field_path, text and truncated.
These are discovery excerpts, not detail citations. Repeat detection includes filters/offset/exclusions.

### `get_product`

```json
{"asin": "a-previously-surfaced-asin"}
```

Returns public detail fields: asin, title, brand, category, price_min, price_max,
pricing, description, attributes, customization_options, variant_combinations,
images. Option entries expose value, price, is_available and optional asin.
Variant entries expose options, price, asin and is_available.

Optional fields selects up to 11 public top-level fields; asin/title/category are always returned.
Omitting fields returns full detail. Unrequested fields are unobserved, never empty or absent.
Different field reads merge into the working fact view; facts.field_sources identifies each
original observation ID. read_evidence without an ID reads the latest observation only.
To check a selection, fetch customization_options and variant_combinations. Fetching only
attributes does not satisfy the option evidence required by submission preflight.

A successful call makes the ASIN eligible for evidence assessment and submission
preflight. Use exact catalog option axes and values. Independently available
options do not prove that their combination is available.

### `submit_recommendations`

```json
{
  "recommendations": [
    {
      "asin": "a-previously-inspected-asin",
      "selected_options": {"颜色": "白色"},
      "reason": "详情确认白色规格可用，商品属性注明静音。"
    }
  ]
}
```

Recommend one to five unique inspected products. Use {} for selected_options
when no selectable option is relevant. Runtime defaults to mode="verified":
requirements must be parsed and every hard constraint on every recommended item
must have a satisfied assessment. It also checks public option availability and,
when variant combinations exist, that the selected values extend to an available
combination. Use verify_selection for selected-variant price checks. Neither tool semantically
verifies the model's constraint interpretation.

If further investigation cannot resolve a mismatch or missing fact, pass
mode="best_effort" and gaps listing every unknown/violated hard constraint:

```json
{
  "mode": "best_effort",
  "recommendations": [{"asin": "an-inspected-asin", "selected_options": {}, "reason": "商品类型合适，但静音表现尚未确认。"}],
  "gaps": [{"asin": "an-inspected-asin", "constraint_id": "quiet", "status": "unknown", "disclosure": "静音表现尚未确认"}]
}
```

Each disclosure must occur verbatim in the corresponding reason. Gaps must match
the current hard-constraint statuses exactly; missing assessments count as unknown.
Best-effort mode does not bypass requirement parsing or public option validity.
Preflight rejection returns submission_not_ready with structured data.issues,
consumes an action but no environment step, and leaves the session open for repair.
Mode and gaps stay in the runtime audit; the environment receives only recommendations.
`shopping_tool_definitions()` retains the three original environment schemas;
the runtime ToolRegistry publishes the extended submit schema and nine tools.

Accepted submission seals the environment
session and returns only {"submitted": true}. Do not call another tool.
The private evaluator stores its result separately; it is never actor feedback.

## Observation envelope

Every executed or rejected action returns an AgentObservation:

- observation_id, action_id and tool_name identify the causal action.
- status is success, rejected or error; data contains the public result.
- error_code and message describe a rejected or failed action.
- environment_step_id is present only for a dispatched environment action.
- terminated and terminal_reason describe environment termination.
- allowed_next_tools lists structurally available tools; individual ASINs and
  parameters still require validation.
- budget records remaining runtime actions, searches, environment steps and
  model turns, plus reported total tokens and usage completeness.
- projection records the observation version and returned item count. A null
  total_results means the engine did not provide a total; do not infer it from
  the number of returned items.

Runtime rejections use action_rejected with a diagnostic message such as
invalid_arguments, tool_not_available or repeated_action. Environment rejections
include product_not_surfaced and product_not_inspected. Correct the action from
the current evidence. Rejections consume their applicable budget.

The orchestrator can also stop for action, model-turn, token, elapsed-time or
context limits. Runtime termination is recorded in run_finished. Do not use
ordinary text as a substitute for a terminal submission.

The context may include advisory action_proposals. Fill fields listed in requires_model_input
before executing a template. These suggestions use observed information gaps; you remain free
to select another legal action. Do not confuse a suggested action with one already executed.
