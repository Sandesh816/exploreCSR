from __future__ import annotations

import json
from typing import Optional

from milestone1_analysis import BundleRecord, format_predicted_changes, summarize_canvas_delta
from milestone1_core import NamedRect, RelationRecord, changed_params


PROPOSAL_SYSTEM_PROMPT = """You propose plausible relation bundles for a geometric canvas editor.

Use only the provided relation IDs.
Do not invent new relations.
Prefer coherent geometric hypotheses that explain the observed edit.
Prefer smaller bundles unless a larger one is clearly more meaningful.
Return structured JSON only."""


RANKING_SYSTEM_PROMPT = """You rank verified relation bundles for likely designer intent.

Use the provided candidate metadata, census signals, and predicted C3 changes.
Prefer coherent, relevant, robust, and minimally-assumptive bundles.
Rank only the provided candidate IDs.
Return structured JSON only."""


def _format_canvas(rects: list[NamedRect], title: str) -> list[str]:
    lines = [title]
    for nr in rects:
        r = nr.rect
        lines.append(f"- {nr.name}: x={r.x}, y={r.y}, w={r.w}, h={r.h}")
    return lines


def _format_delta(c1: list[NamedRect], c2: list[NamedRect]) -> list[str]:
    delta = sorted(changed_params(c1, c2))
    if not delta:
        return ["Changed parameters (delta): none"]

    lines = ["Changed parameters (delta):"]
    for var, before, after in summarize_canvas_delta(c1, c2):
        lines.append(f"- {var}: {before} -> {after}")
    return lines


def build_bundle_proposal_messages(
    c1: list[NamedRect],
    c2: list[NamedRect],
    relation_pool: list[RelationRecord],
    *,
    max_bundles: int = 16,
    max_bundle_size: Optional[int] = None,
    previous_bundles: Optional[list[tuple[int, ...]]] = None,
) -> list[dict[str, str]]:
    lines: list[str] = []
    lines.extend(_format_canvas(c1, "Canvas C1:"))
    lines.append("")
    lines.extend(_format_canvas(c2, "Canvas C2:"))
    lines.append("")
    lines.extend(_format_delta(c1, c2))
    lines.append("")
    lines.append("Relation pool:")
    for relation in relation_pool:
        lines.append(
            f"- relation_id={relation.relation_id}; type={relation.relation_type}; "
            f"support_vars={relation.support_vars}; equation={relation.equation_text}"
        )

    lines.append("")
    lines.append("Proposal guidance:")
    lines.append("- Each proposed bundle must include at least one relation mentioning a changed parameter.")
    lines.append("- A bundle may include other relations that do not individually mention delta.")
    lines.append("- Prefer coherent geometric relationships over algebraically arbitrary combinations.")
    lines.append(f"- Propose at most {max_bundles} bundles.")
    if max_bundle_size is not None:
        lines.append(f"- Each bundle may include at most {max_bundle_size} relation IDs.")
    else:
        lines.append("- Prefer smaller sets of relations when they are equally plausible.")
    lines.append("- Prefer diversity across the proposed bundles.")
    lines.append("- Each bundle must have a unique non-empty candidate_id.")
    lines.append("- Do not repeat an exact relation-id bundle that already appeared earlier in this response.")
    lines.append("- If previous rounds are provided, do not repeat those exact bundles; use the remaining budget on distinct alternatives.")

    if previous_bundles:
        lines.append("")
        lines.append("Previously proposed bundles to avoid repeating exactly:")
        for bundle in previous_bundles:
            lines.append(f"- {bundle}")

    return [
        {"role": "system", "content": PROPOSAL_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def build_bundle_ranking_messages(
    c1: list[NamedRect],
    c2: list[NamedRect],
    records: list[BundleRecord],
    *,
    top_k: int = 5,
    user_history: Optional[str] = None,
    max_ties_per_bundle: int = 3,
    schema: Optional[dict] = None,
) -> list[dict[str, str]]:
    prompt_text, _ = build_bundle_ranking_prompt(
        c1,
        c2,
        records,
        top_k=top_k,
        user_history=user_history,
        max_ties_per_bundle=max_ties_per_bundle,
        schema=schema,
    )
    return [
        {"role": "system", "content": RANKING_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


def build_bundle_ranking_prompt(
    c1: list[NamedRect],
    c2: list[NamedRect],
    records: list[BundleRecord],
    *,
    top_k: int = 5,
    user_history: Optional[str] = None,
    max_ties_per_bundle: int = 3,
    schema: Optional[dict] = None,
) -> tuple[str, dict[int, BundleRecord]]:
    selected_records = records[:top_k]
    candidate_lookup = {
        candidate_id: record
        for candidate_id, record in enumerate(selected_records, start=1)
    }
    candidate_ids = tuple(candidate_lookup.keys())

    lines: list[str] = []
    lines.extend(_format_canvas(c1, "Canvas C1:"))
    lines.append("")
    lines.extend(_format_canvas(c2, "Canvas C2:"))
    lines.append("")
    lines.extend(_format_delta(c1, c2))
    lines.append("")
    lines.append("Ranking guidance:")
    lines.append("- Rank candidates by likely designer intent.")
    lines.append("- Prefer semantic coherence and relevance to the observed edit.")
    lines.append("- Prefer plausible C3 outcomes over algebraically arbitrary ones.")
    lines.append("- Prefer fewer equations and fewer extra fixed variables when the intent is equally plausible.")
    lines.append("- Use robustness as a signal: outcomes supported by more programs are more trustworthy.")
    lines.append("- Complexity is a preference, not a law.")
    lines.append(f"- Allowed candidate_ids: {candidate_ids}")
    lines.append(f"User history summary: {user_history or 'none provided'}")

    for candidate_id, record in candidate_lookup.items():
        lines.append("")
        lines.append(f"Candidate {candidate_id}:")
        lines.append(f"- candidate_id: {candidate_id}")
        lines.append(f"- eq_indices: {record.eq_indices}")
        lines.append(f"- equations: {list(record.equations)}")
        lines.append(f"- support_vars: {record.support_vars}")
        lines.append(f"- changed_vars_hit: {record.changed_vars_hit}")
        lines.append(f"- shared_variable: {record.has_shared_variable}")
        lines.append(f"- connected_support: {record.has_connected_support}")
        lines.append(f"- min_extra_fixed: {record.min_extra_fixed}")
        lines.append(f"- unique_c3_count: {record.unique_c3_count}")
        lines.append(f"- parameterization_conflict: {record.has_parameterization_conflict}")
        lines.append("- viable_parameterizations:")
        for option in record.viable_fixed_sets[:max_ties_per_bundle]:
            lines.append(
                "  - "
                f"fixed_vars={option.fixed_vars}; extra_fixed_vars={option.extra_fixed_vars}; "
                f"driven_vars={option.driven_vars}; "
                f"predicted_changes={format_predicted_changes(option.predicted_changes)}; "
                f"c3_census_count={option.c3_census_count}"
            )
        omitted_count = max(0, len(record.viable_fixed_sets) - max_ties_per_bundle)
        if omitted_count:
            lines.append(f"- omitted_tied_fixed_set_count: {omitted_count}")

    if schema is not None:
        lines.append("")
        lines.append("Output JSON schema:")
        lines.append(json.dumps(schema, indent=2, sort_keys=True))

    return "\n".join(lines), candidate_lookup
