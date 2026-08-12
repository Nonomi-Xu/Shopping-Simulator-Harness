# Shopping Strategy

## 1. Decompose the request

Extract a compact internal constraint sheet before searching:

- product category or intended use;
- hard requirements explicitly stated as necessary;
- preferred attributes that allow trade-offs;
- price ceiling, range, or value preference;
- brand, model, compatibility, audience, size, color, or material terms;
- requested quantity and ranking intent.

Treat explicit compatibility, safety, dimensions, variant, and maximum-budget
requirements as hard unless the request clearly makes them optional. Do not turn
unstated preferences into hard filters.

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

## 4. Inspect and verify

For each inspected product, record only observed facts relevant to the request:

- title, category, brand, and price range;
- matching descriptions or attributes;
- customization option axes and exact values;
- availability of requested options;
- compatible variant combinations when multiple axes interact.

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
- selected options use exact valid catalog strings;
- each reason is grounded in observed evidence;
- another search is unlikely to resolve a known hard-constraint gap.

When the remaining step budget is small, reserve enough actions to inspect the
best surfaced candidate and submit. Never spend the last available step on a
non-terminal exploratory search.
