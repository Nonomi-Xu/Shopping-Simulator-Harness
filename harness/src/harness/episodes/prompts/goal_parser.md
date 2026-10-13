Extract a shopping goal ONLY from the supplied query. Product identities, target products,
annotations and evaluator facts are unavailable and must never be inferred.
Call parse_goal exactly once with structured arguments. Keep atomic requirements with
short EXACT source_quote substrings. Separate category, use_case, budget, brand, color,
size, material, compatibility, ram, weight, battery and other named topics.
Do not reuse a whole multi-requirement sentence as the source for one atom.
Preserve Chinese category words; do not translate them into unrelated index vocabulary.
Explicit must/at least/at most/exclusions are hard; desires and approximate prices are soft.
Do not convert 'around 40' into a hard maximum or invent a tolerance.
critical=true for explicit hard requirements that affect the purchasing decision.
Use predicates only for literal facts that can be checked: category contains, price le/ge
(CNY), exact known brand, option axis/value, named numeric attribute. Available units:
GB, g, W, h, cm. Convert kg to g. Brand in the legacy catalog is often a store name;
do not invent a store-name match from a manufacturer's name. Leave predicate null when
semantics cannot reliably be represented. Do not invent an option axis from a hidden
product. An unknown technical value, missing category or ambiguous hard/soft distinction
should mark the goal needs_review with review_reasons. Never silently drop query needs.
UserGoal has no target product, full query or evaluator field. Only include requirements
explicitly supported by the query. Initial disclosure is generated separately.
