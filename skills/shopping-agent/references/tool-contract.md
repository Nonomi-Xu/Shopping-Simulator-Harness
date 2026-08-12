# Harness Tool Contract

Use exactly one tool call per assistant turn. Tool names and arguments are
case-sensitive. Unknown fields are invalid.

## `search_products`

Search the public catalog using deterministic multi-field BM25 ranking.

Arguments:

```json
{
  "query": "string, non-empty",
  "limit": "integer, optional, 1-50, default 20"
}
```

Successful observation: an ordered JSON array of summaries. Each summary may
contain:

```json
{
  "asin": "string",
  "title": "string",
  "category": "string",
  "brand": "string or null",
  "price_min": "number or null",
  "price_max": "number or null",
  "bm25_score": "number",
  "rank": "integer"
}
```

An empty array is a valid successful result. Any returned ASIN becomes eligible
for `get_product` in the same session.

## `get_product`

Read the public detail of one product previously surfaced by search.

Arguments:

```json
{
  "asin": "string, non-empty"
}
```

Successful observation:

```json
{
  "asin": "string",
  "title": "string",
  "brand": "string or null",
  "category": "string",
  "price_min": "number or null",
  "price_max": "number or null",
  "pricing": ["number"],
  "description": "string",
  "attributes": ["string"],
  "customization_options": {
    "axis": [
      {
        "value": "string",
        "price": "number or null",
        "is_available": "boolean",
        "asin": "string or null"
      }
    ]
  },
  "variant_combinations": [
    {
      "options": {"axis": "exact value"},
      "price": "number or null",
      "asin": "string or null",
      "is_available": "boolean"
    }
  ],
  "images": ["string"]
}
```

The call is rejected with `product_not_surfaced` when the ASIN has not appeared
in a prior search observation. A successful call makes the ASIN eligible for
terminal recommendation.

## `submit_recommendations`

Submit the final ranked recommendation list and terminate the normal session.

Arguments:

```json
{
  "recommendations": [
    {
      "asin": "string, previously inspected",
      "selected_options": {
        "exact option axis": "exact available option value"
      },
      "reason": "string, non-empty and evidence-grounded"
    }
  ]
}
```

Provide one to five unique recommendations. Every ASIN must have a successful
prior `get_product` observation. Use `{}` for `selected_options` when no relevant
catalog option is required. The call is rejected with `product_not_inspected`
when any recommended ASIN was not inspected.

On acceptance, the observation contains the deterministic evaluation and the
step has `terminated=true` with `terminal_reason="submitted"`. Do not call any
tool after this response.

## Common rejected and terminal responses

- `invalid_arguments`: correct the arguments to match the schema.
- `unknown_action`: use one of the three documented tool names.
- `product_not_surfaced`: search for the ASIN before reading its detail.
- `product_not_inspected`: inspect every recommendation before submitting.
- `max_steps`: the session is terminated; preserve the incomplete trace.
- `infrastructure_error`: the session is terminated; do not retry inside the
  sealed session.

Rejected calls consume a Harness step. Do not repeat a rejected call unchanged.
