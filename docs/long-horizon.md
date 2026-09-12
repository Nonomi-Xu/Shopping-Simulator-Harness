# Long-Horizon Interactive Shopping Agent

The episode layer extends the existing Harness. SQLite FTS5/BM25, public product
projection, tool execution, evidence memory, ask/resume and Evaluator v1 remain in use.
No training dependency or alternative retrieval engine is required.

## Offline runnable example

From the `harness` directory, with Python 3.10+ and Pydantic 2:

```bash
python -m examples.long_horizon_demo --output-dir outputs/long-horizon-demo
```

This creates a small catalogue, preprocesses a query without network calls, runs
Full Query and Long-Horizon policies over the same SQLite index, evaluates and verifies
both event chains. The scripted policy receives public model messages only. It is a
fixture for engineering validation, not a measurement of an LLM's reasoning quality.
Output directories must be new. The example retains synthetic raw data in `private-data/`.

The long-horizon example performs:

```text
partial category → update_plan → search → ask budget → simulator reply
→ update_plan → filtered search → inspect two products → compare
→ verify selection → record evidence → recommend → private evaluation
```

## Prepare the real dataset separately

Keep the existing prepared `data/product_search.sqlite3`. No rebuild is necessary for
typed predicates: deterministic SQLite functions evaluate public fields before LIMIT,
using the same matching implementation as the episode metrics. Legacy queries without
typed predicates retain their previous BM25 weights, tokenizer, ordering and filter semantics.

```bash
python -m src.harness.cli episodes prepare \
  --products-path data/fine_items_eval_train_all.json \
  --cache data/episodes.sqlite3 --split test --count 20 \
  --parser structured --model deepseek-v4-flash
```

`structured` explicitly calls the configured provider using one Pydantic `parse_goal`
tool schema. It receives the full query, never a target product or instruction annotations.
Credentials use existing dotenv handling. `--provider-parameters '{"temperature":0}'`
and `--base-url` configure the model. This command can incur inference costs.

`--parser rule --no-env-file` runs entirely offline. The rule parser supports a small
documented vocabulary of categories, explicit price ceilings, RAM minimums, selected use
cases and preferences. Unhandled text is preserved for review. It is not a general
parser for all 235 categories; it never silently treats uncovered text as complete.

Ready episodes are committed one by one to an immutable SQLite cache. Cache identity
includes the original task/query, dataset hash, parser/model configuration, policy and
seed. A repeated preprocessing run reuses existing valid records. A `.prepare.json`
sidecar reports the selected tasks, prepared/cached/failure counts and parser manifest.
Extraction errors, ambiguous goals and invalid initial disclosures fail preparation for
that task and produce a nonzero command status. They are not silently scored as valid
interactive tasks. The full query is retained in the private cache.

For a full preprocessing pass use `--split all --count 0`. The input is streamed and
the ready cache is incrementally committed. The default is at most 10 tasks.
Rerunning retries failed tasks; successful tasks do not call the parser again. Selecting
different parser/policy versions into one cache requires explicit episode IDs at evaluation.

## Run or compare cached episodes

```bash
python -m src.harness.cli episodes run \
  --cache data/episodes.sqlite3 --count 1 \
  --config examples/long_horizon.json \
  --output-dir outputs/episode-01

python -m src.harness.cli episodes compare \
  --cache data/episodes.sqlite3 --count 20 \
  --config examples/long_horizon.json \
  --output-dir outputs/episode-comparison

python -m src.harness.cli episodes replay --run-dir outputs/episode-01
```

Installed environments may use `shopping-agent` instead of `python -m src.harness.cli`.
Use `--episode-id episode_...` to select a frozen episode explicitly; it may be repeated
for comparison. Run accepts exactly one selected episode. `--baseline full_query` selects
the complete-query baseline. Evaluation reads the cache and never invokes GoalParser.

Both baselines share task IDs, target facts, index hash, base model, provider parameters,
skill hash and resource configuration. The full-query baseline disables asking and starts
with the original query. Long-Horizon starts with the selected initial quotes. The existing
`BM25/` Top-5 baseline remains separate because it has a different action/recall budget.

The comparison writes each completed episode immediately and retains compact metrics for
aggregation. It does not implement distributed scheduling or resume an interrupted comparison;
use a fresh output directory and explicitly select remaining episode IDs.

## State, disclosure and boundaries

`ShoppingEpisode` is controller-private. It contains full_query, UserGoal, initial disclosure,
policy and dataset/parser provenance. The original `TaskStore` remains the sole owner of
target ASIN and gold options, referenced by the original task_id.

`UserGoal.requirements` contains atomic hard/soft requirements, a topic, exact query quote,
critical flag and optional typed predicate. Category and use cases are topic-tagged atoms;
budget has one predicate representation rather than duplicate mutable fields. Source quotes
are validated, including numeric thresholds and literal text values. Schema/provenance
validation does not prove semantic extraction accuracy; retain extraction manifests and
review ambiguous records before drawing dataset-wide conclusions.

Initial observation is made only from policy-selected exact quotes, not the whole query.
At least one requirement must remain undisclosed. Overlapping quotes that would expose an
unselected requirement are rejected. `instruction_simple` is not blindly trusted or injected.

`DeterministicSimulator` only receives UserGoal, DisclosurePolicy and initially revealed IDs.
It has no catalog, TaskStore, target product or evaluation handle. The versioned `topics.json`
maps synonymous questions to relevant topics. Each answer is capped by policy and copies
only matching source quotes. Repeated questions stay consistent; absent topics do not create
new constraints. Broad questions without a recognized topic request clarification.

Optional `ConstrainedLLMSimulator` accepts a separate provider through `simulator_factory` in
the Python API. It retains deterministic fact routing and allows the model only to choose
one of three wording styles. That provider sees only facts already selected for this reply.
It cannot invent new facts. Simulator failure seals the agent run without further disclosure.
This mode is deliberately more constrained than unconstrained natural-language roleplay.

The actor owns TaskProfile.constraints, uncertain_requirements, plan and WorkingMemory.
PreferenceState in each trajectory turn is a snapshot of that state, not a second mutable
requirement store. After each reply, update_plan must reparse public messages before search
or recommendation. Hard constraints cannot be erased or weakened autonomously. The simulator
does not directly inject its correct structured predicates into the actor's state.

`progressive_disclosure` prevents recommending before any clarification or while the actor
still reports unresolved requirements. Private critical fields are used only in metrics;
no hidden 'missing field' feedback is sent to the model.

## Structured search and its limits

Typed predicates support category/legacy-brand text comparisons, option axis/value tests,
CNY price bounds and labeled numeric public attributes (RAM, weight, power, battery runtime,
length, or an explicit label). Known actor hard predicates are automatically merged with
search filters. Their actual applied values appear in public observation metadata, making
the filter application deterministic. Explicit literal hard price ceilings and RAM minimums
are also normalized from the actor's source quotes when it omitted a typed predicate;
this uses only disclosed text and never supplies missing hidden requirements. This makes
model-requested versus environment-applied filters auditable.

The legacy `brand` field is `shop_name`, often a merchant name, not normalized manufacturer
identity. A query excluding Apple/Mac must not be automatically converted to a guessed
merchant-name filter. Such semantic constraints can retain a null predicate and are reported
unknown by deterministic metrics until supported public manufacturer facts exist.

Numeric values require labels and units. kg converts to g; numeric marketing claims such as
'up to 16GB' remain unknown. Generic natural-language similarity is not a filter. Missing
attributes, unknown prices and unavailable/unlisted multi-axis combinations do not pass strict
typed filters. Price and option constraints must hold in one common available selection.
Product minimum price cannot be combined with a different expensive RAM option to pass a budget.
Final recommendation checks use the selected options and independently recompute typed statuses.

Legacy min_price/max_price filters still mean range overlap, with unknown prices included by
default. The new strict predicates have explicitly different semantics; both baselines use
the same implementation. JSON and HTTP interactive sources do not silently claim support for
these SQLite predicates. Vector retrieval code remains present but is outside this benchmark;
the current public search contract advertises only executable BM25.

## Metrics

Original Evaluator v1 rewards and strict option matching are unchanged. Additional metrics
use a versioned `EpisodeMetric.compute(MetricContext)` interface:

| Metric | Definition |
|---|---|
| Success@1, @1–5 | Valid final recommendations contain the target at/before that rank |
| strict_success | Original target + gold-option success |
| Constraint satisfaction | Satisfied / all revealed hard-constraint × recommendation pairs |
| Verification coverage | (Satisfied + violated) / all such pairs; unknown stays visible |
| Turns to success | Total actor model calls when target is successfully recommended |
| Tool cost | Runtime action attempts plus separate environment/action-category counts |
| Redundant questions | Answered questions with no new requirement and an already known topic / answered questions |
| Forgetting | Previously tracked hard requirements absent at subsequent domain-action opportunities |
| Omission | Revealed hard requirements absent from final actor state |
| Violation | Revealed hard constraints violated by final recommendations |
| Premature recommendation | Accepted recommendation leaves a critical field undisclosed or untracked |
| Search efficiency | Search calls, calls per successful episode, and paired success |
| Information gain | (eligible_before − eligible_after) / eligible_before per answered question |

Information gain uses the same fixed public catalogue and only disclosed hard predicates
that can be executed. It is separate from changing lexical queries or Top-K pages. Candidate
reduction records include executable-constraint counts. It is not Shannon entropy, and it
does not claim to measure unsupported soft/semantic preferences. Zero denominators produce
null. Aggregates report macro averages and episode coverage. Failed/no-submission runs remain
in success denominators; there is no LLM judge substituting for missing deterministic facts.

## Budgets, artifacts and reproducibility

`EpisodeConfig.max_turns` maps to cumulative max_model_turns. max_tool_calls maps to
max_actions (including rejected/runtime actions); max_environment_calls maps to max_steps.
Question, search, token, time and context budgets reuse RuntimeConfig. A waiting checkpoint at
an exhausted actor budget terminates without disclosing another answer. Simulator calls are
separate from actor model-call accounting. For deterministic benchmark use the default simulator.

```text
episode-output/
├── public/
│   ├── trajectory.jsonl            # one complete typed episode per JSONL record
│   └── runtime/                    # public event chain, plans, memory, manifests
│       └── evaluation.json         # always null for episode actor runtime
└── private/
    ├── evaluation.json             # original evaluation + metrics + disclosure audit
    └── manifest.json               # episode/parser/config/source hashes
```

Comparison aggregates are in `comparison/private/summary.json` and `pairs.jsonl`.
Only the `public/` subtree of an individual Long-Horizon run is actor-safe. A Full Query
baseline intentionally exposes its original query. Never pass an EpisodeResult, private
cache, private manifest, evaluation sidecar or original raw dataset to the agent.

Every action has public preference snapshots, actual observations, returned/surfaced/inspected
candidate IDs and search counts where available. The private disclosure audit records fixed-pool
counts before/after asking. Manifests retain seed, model/config hashes, parser prompt hash,
skill hash, runtime instruction hash, actual system-prompt hashes in context events, index
file hash, dataset hash, predicate/disclosure/simulator/metric versions and event-chain hash.
Replay validates hashes and reconstructs public runtime state without calling any model.
External models need not reproduce outputs bit-for-bit even with temperature zero; saved
trajectories are the authoritative replay artifact.

Python process isolation is by capability and allowlisted serialization, not an OS security
sandbox. Raw catalog identifiers can legitimately include the target after normal retrieval;
tests prohibit answer labels, hidden fields and private-only canaries rather than censoring
all occurrences of a normally surfaced product ID.

## Validation

```bash
python -m unittest discover -s src/harness/test -p 'test_*.py'
python -m unittest src.harness.test.test_episode_leakage -v
```

Tests cover parsing/cache reuse, disclosure consistency, state updates, hard preservation,
SQL-before-Top-K filters, variant compatibility, budget termination across replies, private
canaries in every model input, metrics, JSONL serialization, replay and paired baselines.

Existing `run`, `rollout`, `chat`, `resume`, preference store and legacy evaluation remain.
CLI chat defaults to the tested single-agent runtime; `--agent-mode multi` is explicit.
SearchProductsAction now accepts explicit method=bm25 so the existing multi-role tool
contract no longer requires a field that its Pydantic schema rejects. Existing completed
event bundles remain replayable; resuming any checkpoint requires the original compatible
runtime configuration and prompt as enforced by the existing checkpoint contract.
