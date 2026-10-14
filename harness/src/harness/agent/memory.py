"""Session-scoped, evidence-backed working memory; no implicit cross-task learning."""

import json
import unicodedata

from ..schemas import ProductDetail
from .contracts import (
    AgentObservation,
    CandidateMemory,
    EvidenceAssessment,
    ReadEvidence,
    TaskProfile,
    WorkingMemory,
)


def normalized_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


class MemoryStore:
    version = "shopping-memory-v3"

    def __init__(self, session_id: str):
        self.state = WorkingMemory(session_id=session_id)
        self._observations: dict[str, AgentObservation] = {}

    @property
    def observation_ids(self) -> set[str]:
        return set(self._observations)

    def observe(self, observation: AgentObservation) -> None:
        self._observations[observation.observation_id] = observation.model_copy(
            deep=True
        )
        candidates = dict(self.state.candidates)
        if observation.status != "success":
            self.state = self.state.model_copy(
                update={"last_error": observation.error_code}
            )
            return
        if observation.tool_name == "search_products" and isinstance(
            observation.data, list
        ):
            for summary in observation.data:
                asin = summary["asin"]
                previous = candidates.get(asin)
                candidates[asin] = CandidateMemory(
                    asin=asin,
                    summary=summary,
                    inspected=previous.inspected if previous else False,
                    observation_ids=[
                        *(previous.observation_ids if previous else []),
                        observation.observation_id,
                    ],
                    detail_observation_id=(
                        previous.detail_observation_id if previous else None
                    ),
                    assessments=previous.assessments if previous else {},
                )
        elif observation.tool_name == "get_product" and isinstance(
            observation.data, dict
        ):
            asin = observation.data["asin"]
            previous = candidates[asin]
            candidates[asin] = previous.model_copy(
                update={
                    "inspected": True,
                    "detail_observation_id": observation.observation_id,
                    "observation_ids": [
                        *previous.observation_ids,
                        observation.observation_id,
                    ],
                    "assessments": (
                        {} if observation.projection.source else previous.assessments
                    ),
                }
            )
            archive = dict(self.state.evidence_archive)
            archive[observation.observation_id] = ProductDetail.model_validate(
                observation.data
            )
            self.state = self.state.model_copy(
                update={
                    "evidence_archive": archive,
                    "detail_fields": {
                        **self.state.detail_fields,
                        observation.observation_id: sorted(observation.data),
                    },
                    "evidence_sources": {
                        **self.state.evidence_sources,
                        observation.observation_id: observation.projection.source or {},
                    },
                    "selection_checks": {
                        key: value
                        for key, value in self.state.selection_checks.items()
                        if not observation.projection.source
                        or value.get("asin") != asin
                    },
                },
                deep=True,
            )
        elif observation.tool_name == "verify_selection" and isinstance(
            observation.data, dict
        ):
            self.state = self.state.model_copy(
                update={
                    "selection_checks": {
                        **{
                            k: v
                            for k, v in self.state.selection_checks.items()
                            if (
                                v.get("asin"),
                                v.get("selected_options"),
                                v.get("budget_constraint_id"),
                            )
                            != (
                                observation.data["asin"],
                                observation.data["selected_options"],
                                observation.data["budget_constraint_id"],
                            )
                        },
                        observation.observation_id: observation.data,
                    }
                },
                deep=True,
            )
        self.state = self.state.model_copy(
            update={"candidates": candidates, "last_error": None}, deep=True
        )

    def record_query(
        self, query: str, request: dict | None = None, next_offset: int | None = None
    ) -> None:
        self.state = self.state.model_copy(
            update={
                "queries": [*self.state.queries, query],
                "search_requests": [
                    *self.state.search_requests,
                    {
                        "arguments": request or {"query": query},
                        "next_offset": next_offset,
                    },
                ],
            }
        )

    def assess(self, assessment: EvidenceAssessment, task: TaskProfile) -> None:
        self.assess_batch([assessment], task)

    def _validate_assessment(
        self, assessment: EvidenceAssessment, task: TaskProfile
    ) -> None:
        candidate = self.state.candidates.get(assessment.asin)
        if candidate is None or not candidate.inspected:
            raise ValueError("assessments require an inspected candidate")
        if assessment.constraint_id not in {c.constraint_id for c in task.constraints}:
            raise ValueError("unknown constraint")
        observation = self._observations.get(assessment.observation_id)
        if (
            observation is None
            or observation.status != "success"
            or observation.tool_name != "get_product"
            or not isinstance(observation.data, dict)
            or observation.data.get("asin") != assessment.asin
        ):
            raise ValueError(
                "evidence must reference this candidate's successful detail observation"
            )
        value = resolve_pointer(observation.data, assessment.field_path)
        if observation.projection.source:
            latest, sources = observed_product(self.state, assessment.asin)
            top_field = (
                assessment.field_path.split("/")[1]
                .replace("~1", "/")
                .replace("~0", "~")
            )
            if sources.get(top_field) != assessment.observation_id:
                if (
                    resolve_pointer(
                        latest.model_dump(mode="json"), assessment.field_path
                    )
                    != value
                ):
                    raise ValueError(
                        "evidence was superseded by a changed source observation"
                    )
        if isinstance(value, (dict, list)):
            raise ValueError("evidence must point to a scalar field")
        rendered = (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
        if assessment.quote not in rendered:
            raise ValueError("evidence quote does not occur in the observed field")

    def assess_batch(
        self, assessments: list[EvidenceAssessment], task: TaskProfile
    ) -> None:
        """Validate every citation before committing any assessment."""
        targets = [(a.asin, a.constraint_id) for a in assessments]
        if not 1 <= len(targets) <= 30 or len(targets) != len(set(targets)):
            raise ValueError(
                "batch requires 1-30 unique product/constraint assessments"
            )
        for assessment in assessments:
            self._validate_assessment(assessment, task)
        candidates = dict(self.state.candidates)
        for assessment in assessments:
            candidate = candidates[assessment.asin]
            candidates[assessment.asin] = candidate.model_copy(
                update={
                    "assessments": {
                        **candidate.assessments,
                        assessment.constraint_id: assessment,
                    }
                }
            )
        self.state = self.state.model_copy(update={"candidates": candidates}, deep=True)

    def read_evidence(self, request: ReadEvidence) -> dict:
        """Page exact scalar fragments from this run's already observed details.

        The read merges every detail observation recorded for the ASIN, so a field
        an earlier ``get_product`` returned stays readable even when the pinned
        observation did not request it. Each fragment reports the observation that
        actually returned its field, because ``record_evidence`` validates a
        citation against that exact observation. A path that cannot be resolved is
        reported in ``missing_field_paths`` instead of failing the whole call, and
        ``available_field_paths`` lists what this session can serve.
        """
        candidate = self.state.candidates.get(request.asin)
        if candidate is None or not candidate.inspected:
            raise ValueError("read_evidence requires an inspected candidate")
        if (
            request.observation_id is not None
            and request.observation_id not in candidate.observation_ids
        ):
            raise ValueError(
                "unknown detail observation for this candidate in this session"
            )
        merged, sources = observed_product(
            self.state, request.asin, prefer=request.observation_id
        )
        raw = merged.model_dump(mode="json")
        # ProductDetail fills defaults for fields this session never observed;
        # serving those would fabricate evidence, so keep only observed fields.
        data = {key: value for key, value in raw.items() if key in sources}
        available = sorted(f"/{key}" for key in data)
        paths = request.field_paths or available
        fragments: list[dict] = []
        missing: list[dict] = []
        seen = set()
        for path in paths:
            try:
                resolved = resolve_pointer(data, path)
            except ValueError as error:
                # An unobserved field is reported, never served, and never treated
                # as an empty value: "not observed" and "observed as empty" differ.
                missing.append(
                    {
                        "field_path": path,
                        "reason": "not observed in this session",
                        "detail": str(error),
                    }
                )
                continue
            top_field = (
                path.split("/")[1].replace("~1", "/").replace("~0", "~")
                if "/" in path[1:]
                else path[1:].replace("~1", "/").replace("~0", "~")
            )
            source_id = sources.get(top_field, candidate.detail_observation_id)
            for field_path, value in scalar_fields(resolved, path):
                if field_path in seen:
                    continue
                seen.add(field_path)
                if isinstance(value, str):
                    for start in range(0, max(1, len(value)), 1000):
                        fragments.append(
                            {
                                "field_path": field_path,
                                "value": value[start : start + 1000],
                                "character_offset": start,
                                "total_characters": len(value),
                                "source_observation_id": source_id,
                            }
                        )
                else:
                    fragments.append(
                        {
                            "field_path": field_path,
                            "value": value,
                            "source_observation_id": source_id,
                        }
                    )
        # Paging past the end is a normal end-of-traversal, not an error.
        offset_out_of_range = request.offset > len(fragments)
        items = []
        # Bound the serialized data, including metadata, independently of the context budget.
        response = {
            "asin": request.asin,
            "source": self.state.evidence_sources.get(
                candidate.detail_observation_id
            ),
            "source_observation_id": candidate.detail_observation_id,
            "field_sources": sources,
            "available_field_paths": available,
            "requested_field_paths": paths,
            "missing_field_paths": missing,
            "items": items,
            "total_fragments": len(fragments),
            "next_offset": None,
        }
        for fragment in fragments[request.offset : request.offset + request.limit]:
            trial = {
                **response,
                "items": [*items, fragment],
                "next_offset": len(fragments),
            }
            if len(json.dumps(trial, ensure_ascii=False)) > 12000:
                if not items:
                    raise ValueError(
                        "evidence metadata exceeds response budget; narrow field_paths"
                    )
                break
            items.append(fragment)
        end = request.offset + len(items)
        response["next_offset"] = end if end < len(fragments) else None
        response["offset_out_of_range"] = offset_out_of_range
        if len(json.dumps(response, ensure_ascii=False)) > 12000:
            raise ValueError(
                "evidence metadata exceeds response budget; narrow field_paths"
            )
        return response

    def invalidate_changed_constraints(
        self, before: TaskProfile, after: TaskProfile
    ) -> None:
        old = {c.constraint_id: c for c in before.constraints}
        unchanged = {
            c.constraint_id for c in after.constraints if old.get(c.constraint_id) == c
        }
        self.state = self.state.model_copy(
            update={
                "candidates": {
                    asin: candidate.model_copy(
                        update={
                            "assessments": {
                                key: value
                                for key, value in candidate.assessments.items()
                                if key in unchanged
                            }
                        }
                    )
                    for asin, candidate in self.state.candidates.items()
                },
                "selection_checks": {
                    k: v
                    for k, v in self.state.selection_checks.items()
                    if v.get("budget_constraint_id") is None
                    or v.get("budget_constraint_id") in unchanged
                },
            }
        )

    def compact(self, task: TaskProfile, limit: int) -> dict:
        hard = {c.constraint_id for c in task.constraints if c.priority == "hard"}
        soft = {c.constraint_id for c in task.constraints if c.priority == "soft"}

        def count(candidate, ids, status):
            return sum(
                cid in candidate.assessments
                and candidate.assessments[cid].status == status
                for cid in ids
            )

        candidates = sorted(
            self.state.candidates.values(),
            key=lambda c: (
                count(c, hard, "violated") > 0,
                -count(c, hard, "satisfied"),
                not c.inspected,
                -count(c, soft, "satisfied"),
                count(c, soft, "violated"),
                c.summary.get("rank", 999),
                c.asin,
            ),
        )
        constraints = {c.constraint_id for c in task.constraints}
        return {
            "queries": self.state.queries,
            "last_error": self.state.last_error,
            "total_candidates": len(candidates),
            "omitted_candidates": max(0, len(candidates) - limit),
            "hard_conflict_count": sum(
                count(c, hard, "violated") > 0 for c in candidates
            ),
            "requirements_status": task.requirements_status,
            "candidates": [
                {
                    **c.model_dump(mode="json"),
                    "facts": self._fact_view(c),
                    "constraint_matrix": {
                        cid: (
                            c.assessments[cid].status
                            if cid in c.assessments
                            else "unknown"
                        )
                        for cid in sorted(constraints)
                    },
                    "hard_conflicts": sorted(
                        cid
                        for cid in hard
                        if cid in c.assessments
                        and c.assessments[cid].status == "violated"
                    ),
                    "evidence_ready": (
                        task.requirements_status == "parsed"
                        and c.inspected
                        and count(c, hard, "satisfied") == len(hard)
                    ),
                    "unknown_constraints": sorted(
                        cid
                        for cid in constraints
                        if cid not in c.assessments
                        or c.assessments[cid].status == "unknown"
                    ),
                }
                for c in candidates[:limit]
            ],
        }

    def _fact_view(self, candidate: CandidateMemory) -> dict | None:
        if candidate.detail_observation_id is None:
            return None
        detail, sources = observed_product(self.state, candidate.asin)
        attributes = detail.attributes[:4]
        return {
            "source_observation_id": candidate.detail_observation_id,
            "source": self.state.evidence_sources.get(candidate.detail_observation_id),
            "field_sources": sources,
            "unobserved_fields": sorted(set(ProductDetail.model_fields) - set(sources)),
            # Exactly the pointers read_evidence can serve for this candidate.
            "readable_field_paths": sorted(f"/{key}" for key in sources),
            "price_min": detail.price_min,
            "price_max": detail.price_max,
            "price_scope": "product_range_not_selected_variant",
            "attribute_previews": [a[:180] for a in attributes],
            "attributes_truncated": len(detail.attributes) > 4
            or any(len(a) > 180 for a in attributes),
            "option_axes": list(detail.customization_options)[:8],
            "omitted_option_axes": max(0, len(detail.customization_options) - 8),
            "variant_count": len(detail.variant_combinations),
            "read_more_tool": "read_evidence",
            "selection_checks": [
                {**v, "stale": v.get("field_sources") != sources}
                for v in self.state.selection_checks.values()
                if v.get("asin") == candidate.asin
            ][-2:],
        }


def observed_product(
    memory: WorkingMemory, asin: str, *, prefer: str | None = None
) -> tuple[ProductDetail, dict[str, str]]:
    """Merge only fields actually returned, retaining original citation ownership.

    ``prefer`` names one observation whose fields win when several observations
    returned the same field. It is applied last so its provenance is the one
    reported, which keeps a citation valid when the caller pinned an observation.
    """
    candidate = memory.candidates.get(asin)
    if candidate is None or not candidate.inspected:
        raise ValueError("product must be inspected in this session")
    order = [oid for oid in candidate.observation_ids if oid != prefer]
    if prefer is not None and prefer in candidate.observation_ids:
        order.append(prefer)
    data, sources = {}, {}
    for oid in order:
        detail = memory.evidence_archive.get(oid)
        if detail is None or detail.asin != asin:
            continue
        raw = detail.model_dump(mode="json")
        for field in memory.detail_fields.get(oid, list(raw)):
            data[field] = raw[field]
            sources[field] = oid
    if not data:
        raise ValueError("missing observed product details")
    return ProductDetail.model_validate(data), sources


def resolve_pointer(data, path: str):
    if not path.startswith("/"):
        raise ValueError("evidence field path must be a JSON Pointer")
    value = data
    try:
        for raw in path[1:].split("/"):
            # Reject malformed escapes instead of silently accepting non-RFC pointers.
            if "~" in raw.replace("~0", "").replace("~1", ""):
                raise ValueError("invalid pointer escape")
            key = raw.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if (
                    not key.isascii()
                    or not key.isdecimal()
                    or (len(key) > 1 and key[0] == "0")
                ):
                    raise ValueError("invalid array index")
                value = value[int(key)]
            else:
                value = value[key]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("evidence field path does not exist") from exc
    return value


def scalar_fields(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            yield from scalar_fields(child, f"{path}/{escaped}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from scalar_fields(child, f"{path}/{index}")
    else:
        yield path, value
