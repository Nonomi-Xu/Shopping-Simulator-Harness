# Shopping Strategy

## 1. Decompose the request

For progressive conversations, identify consequential unknowns and persist them with
update_plan. Ask a focused question when its answer can change candidate eligibility
or selection. Derive topic and typed predicates only from public query/reply evidence.
After each user reply, reparse before another search. Budget predicates require explicit
units; manufacturer names must not be guessed into the legacy merchant-name brand field.
Never assume that an initial category-only request contains all the user's preferences.

A bootstrap plan is unparsed. A successful `update_plan` is required before
submission. Supply `no_constraints_reason` if the analysis yields no constraints;
do not use an empty list to silently skip requirement analysis.

Extract a compact internal constraint sheet before searching:
Persist it with `update_plan`: provide constraints, focused queries,
verification targets and a short change summary. `source_quote` must occur in
the query or the explicit profile preference. Preserve existing hard constraints
when replanning; query broadening never relaxes final acceptance requirements.

- product category or intended use;
- hard requirements explicitly stated as necessary;
- preferred attributes that allow trade-offs;
- price ceiling, range, or value preference;
- brand, model, compatibility, audience, size, color, or material terms;
- requested quantity and ranking intent.

Treat explicit compatibility, safety, dimensions, variant, and maximum-budget
requirements as hard unless the request clearly makes them optional. Do not turn
unstated preferences into hard filters.

When you encode a budget, ground the number in the quote: use the amount stated
(`预算300元以内` → `le 300`). For an approximate amount (`300元左右`) either
quote the stated amount with the frozen 1.1x tolerance (`le 330`), or give your
own ceiling up to 1.5x it. Never encode a ceiling below a stated maximum, and do
not leave an explicitly stated budget out of the constraint sheet. Chinese
numerals are stated amounts, so `两千元以内` is `le 2000`.

## 2. Plan searches

Start with the smallest query that preserves the category and the most
discriminative hard constraint. Reuse exact brand names, model numbers, units,
and Chinese product terms from the request.

Use this progression when refinement is needed:

1. Category plus the rarest hard constraint.
2. Category plus brand/model/compatibility term.
3. Category plus a key attribute or variant.
4. A simplified category query if earlier searches returned nothing.

Change one meaningful concept between queries so the evidence gained from each
search is understandable. Avoid punctuation-only changes, reordered duplicates,
and broad queries that discard the requested category.

## 3. Form a shortlist

Prefer three to five distinct candidates when results permit. Rank surfaced
products provisionally using:

1. hard-constraint evidence in the summary;
2. category and model relevance;
3. price compatibility;
4. coverage of soft preferences;
5. result rank as a tie-breaker, not proof of suitability.

Search summaries are discovery evidence, not enough for final recommendation.
Inspect the strongest candidates with `get_product`.

Use category/brand/price filters when supported by the request. Unknown prices remain
eligible by default; an overlapping product price range does not prove the selected price.
Inspect match_snippets to choose candidates. Continue pages using projection.next_offset
with unchanged filters/exclusions, or deliberately revise the query to explore another need.

## 4. Inspect and verify

For each inspected product, record only observed facts relevant to the request:

- title, category, brand, and price range;
- matching descriptions or attributes;
- customization option axes and exact values;
- availability of requested options;
- compatible variant combinations when multiple axes interact.

Use `record_evidence` or `record_evidence_batch` for constraint-level judgments. Reference the detail
observation ID and a scalar JSON Pointer such as `/attributes/0` or
`/customization_options/颜色/0/value`. Quotes must occur in that exact field.
The runtime verifies provenance; semantic sufficiency remains your judgment.
Unsatisfied, unknown, and satisfied are distinct states. Working memory survives
context compaction. Use `read_evidence` with an ASIN to retrieve original detail
fragments and citation IDs without another environment step; it merges every
detail you fetched for that product, so read the whole detail rather than
guessing which `get_product(fields=...)` call returned a field. Pass
`field_paths` only from `available_field_paths`, and follow `next_offset` for
additional fragments; `missing_field_paths` reports a path this session cannot
serve without failing the call. Cite each fragment's own
`source_observation_id`. Missing assessments remain
unknown; never invent a quote for a missing field.

Do not assume that independently available option values form an available
combination. When `variant_combinations` is populated, prefer a listed available
combination. If evidence for a hard constraint is absent, mark it unknown rather
than satisfied.

## 5. Compare candidates

Use a constraint-first comparison:

- Reject a candidate with a known hard-constraint conflict when a valid
  alternative exists.
- Prefer confirmed evidence over title similarity or BM25 rank.
- Prefer an in-budget exact match over a stronger soft-preference match that
  violates budget.
- Recommend fewer products when additional items add no credible alternative.
- Keep multiple recommendations only when they represent meaningful trade-offs.

## 6. Choose options

Copy option axis names and values exactly from `ProductDetail`. Select only
available values and, when required, an available variant combination.

Use an empty object when the request has no relevant selectable option or the
product exposes none. Never encode descriptive attributes as customization
options.

Use compare_products to align inspected alternatives and expose unresolved constraints.
Use get_product(fields=...) for targeted reads, and original facts.field_sources IDs to
recover older fields via read_evidence. Missing reads must not become negative evidence.
Before relying on a price claim, run verify_selection on the exact chosen options.
For a budget requirement, supply the publicly interpreted max_price and its constraint ID.
Unknown/conflicting variant prices remain unknown; do not use the product minimum or add
independent axis prices. Record your requirement assessment using the returned original citations.
Action proposals are optional suggestions; complete any indicated model arguments before use.

## 7. Write grounded reasons

Each reason should briefly connect observed facts to the request:

1. State the principal match.
2. Mention relevant price, attribute, or selected variant evidence.
3. State a material limitation when the match is partial.

Avoid generic claims such as “best product,” “high quality,” or “perfect match”
unless the observation provides direct support. Do not mention BM25 scores,
hidden targets, rewards, or evaluator behavior to the shopper.

## 8. Stop deliberately

Submit when all of these are true:

- at least one candidate has been inspected;
- every recommended ASIN has been inspected;
- requirements are parsed and every recommended candidate has satisfied evidence
  for all hard constraints, or explicit `best_effort` gap disclosures;
- selected options use exact valid catalog strings;
- each reason is grounded in observed evidence;
- another search is unlikely to resolve a known hard-constraint gap.

When the remaining step budget is small, reserve enough actions to inspect the
best surfaced candidate and submit. Never spend the last available step on a
non-terminal exploratory search.
