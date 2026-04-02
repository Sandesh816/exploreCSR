from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Optional

from milestone1_analysis import (
    BundleRecord,
    build_verifier_context,
    format_predicted_changes,
    print_bundle_summary,
    review_proposed_bundle_indices,
    verify_and_materialize_candidate_bundles,
)
from milestone1_core import NamedRect, Rect, build_relation_records, detect_c1_equations
from milestone1_ollama import (
    BundleProposalResult,
    OllamaRankerConfig,
    RankingResult,
    propose_relation_bundles_with_ollama,
    rank_verified_bundles_with_ollama,
)


@dataclass(frozen=True)
class PipelineConfig:
    model_name: str
    output_json_path: str = "runtime/pipeline_last_run.json"
    max_system_size: int = 3
    max_extra_fixed: Optional[int] = 2
    proposal_rounds: int = 2
    bundles_per_round: int = 24
    global_bundle_cap: int = 128
    linear_only: bool = True
    include_offsets: bool = True
    include_pins: bool = True
    timeout_seconds: int = 120
    temperature: int | float = 0
    proposal_temperature: int | float | None = None
    ranking_temperature: int | float | None = None
    keep_alive: str | None = "5m"
    scene_name: str = "default"


@dataclass(frozen=True)
class ProposedBundleLog:
    round_index: int
    candidate_id: str
    relation_ids: tuple[int, ...]
    rationale: str


@dataclass(frozen=True)
class RejectedBundleLog:
    stage: str
    relation_ids: tuple[int, ...]
    reason: str
    normalized_relation_ids: Optional[tuple[int, ...]] = None
    round_index: Optional[int] = None
    candidate_id: Optional[str] = None
    rationale: Optional[str] = None


@dataclass(frozen=True)
class ProposalRoundTiming:
    round_index: int
    requested_max_bundles: int
    returned_bundle_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class PipelineRunResult:
    c1: list[NamedRect]
    c2: list[NamedRect]
    eq_pool_size: int
    proposal_results: list[BundleProposalResult]
    proposal_round_timings: list[ProposalRoundTiming]
    proposal_phase_seconds: float
    proposal_filter_phase_seconds: float
    verification_phase_seconds: float
    ranking_phase_seconds: float
    total_seconds: float
    all_proposed_bundles: list[ProposedBundleLog]
    rejected_bundles: list[RejectedBundleLog]
    candidate_bundle_indices: list[tuple[int, ...]]
    verified_records: list[BundleRecord]
    ranking_result: RankingResult
    candidate_lookup: dict[int, BundleRecord]


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _serialize_bundle_record(record: BundleRecord) -> dict:
    return {
        "eq_indices": list(record.eq_indices),
        "equations": list(record.equations),
        "support_vars": list(record.support_vars),
        "delta_hit": record.delta_hit,
        "changed_vars_hit": list(record.changed_vars_hit),
        "has_shared_variable": record.has_shared_variable,
        "has_connected_support": record.has_connected_support,
        "verification_passed": record.verification_passed,
        "min_extra_fixed": record.min_extra_fixed,
        "has_parameterization_conflict": record.has_parameterization_conflict,
        "unique_c3_count": record.unique_c3_count,
        "failure_reason": record.failure_reason,
        "viable_fixed_sets": [
            {
                "fixed_vars": list(option.fixed_vars),
                "extra_fixed_vars": list(option.extra_fixed_vars),
                "driven_vars": list(option.driven_vars),
                "predicted_changes": [
                    {
                        "var": name,
                        "before": _json_safe(before),
                        "after": _json_safe(after),
                    }
                    for name, before, after in option.predicted_changes
                ],
                "c3_key": option.c3_key,
                "c3_census_count": option.c3_census_count,
                "program_text": option.program_text,
            }
            for option in record.viable_fixed_sets
        ],
    }


def pipeline_result_to_dict(result: PipelineRunResult) -> dict:
    return {
        "eq_pool_size": result.eq_pool_size,
        "timings": {
            "proposal_rounds": [
                {
                    "round_index": timing.round_index,
                    "requested_max_bundles": timing.requested_max_bundles,
                    "returned_bundle_count": timing.returned_bundle_count,
                    "elapsed_seconds": timing.elapsed_seconds,
                }
                for timing in result.proposal_round_timings
            ],
            "proposal_phase_seconds": result.proposal_phase_seconds,
            "proposal_filter_phase_seconds": result.proposal_filter_phase_seconds,
            "verification_phase_seconds": result.verification_phase_seconds,
            "ranking_phase_seconds": result.ranking_phase_seconds,
            "total_seconds": result.total_seconds,
        },
        "all_proposed_bundles": [
            {
                "round_index": bundle.round_index,
                "candidate_id": bundle.candidate_id,
                "relation_ids": list(bundle.relation_ids),
                "rationale": bundle.rationale,
            }
            for bundle in result.all_proposed_bundles
        ],
        "rejected_bundles": [
            {
                "stage": bundle.stage,
                "relation_ids": list(bundle.relation_ids),
                "normalized_relation_ids": (
                    None
                    if bundle.normalized_relation_ids is None
                    else list(bundle.normalized_relation_ids)
                ),
                "reason": bundle.reason,
                "round_index": bundle.round_index,
                "candidate_id": bundle.candidate_id,
                "rationale": bundle.rationale,
            }
            for bundle in result.rejected_bundles
        ],
        "proposal_results": [
            {
                "bundles": [
                    {
                        "candidate_id": bundle.candidate_id,
                        "relation_ids": list(bundle.relation_ids),
                        "rationale": bundle.rationale,
                    }
                    for bundle in proposal_result.bundles
                ],
                "raw_prompt": proposal_result.raw_prompt,
                "raw_content": proposal_result.raw_content,
                "raw_response_json": proposal_result.raw_response_json,
            }
            for proposal_result in result.proposal_results
        ],
        "candidate_bundle_indices": [list(bundle) for bundle in result.candidate_bundle_indices],
        "verified_records": [_serialize_bundle_record(record) for record in result.verified_records],
        "ranking_result": {
            "summary": result.ranking_result.summary,
            "chosen_candidate_id": result.ranking_result.chosen_candidate_id,
            "ranked_candidates": [
                {
                    "candidate_id": ranked.candidate_id,
                    "bundle_rank": ranked.bundle_rank,
                    "score": ranked.score,
                    "keep": ranked.keep,
                    "rationale": ranked.rationale,
                }
                for ranked in result.ranking_result.ranked_candidates
            ],
            "raw_prompt": result.ranking_result.raw_prompt,
            "raw_content": result.ranking_result.raw_content,
            "raw_response_json": result.ranking_result.raw_response_json,
        },
    }


def save_pipeline_result(result: PipelineRunResult, output_json_path: str) -> Path:
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(pipeline_result_to_dict(result)), indent=2),
        encoding="utf-8",
    )
    return output_path


def _print_heading(title: str) -> None:
    print()
    print(f"=== {title} ===")


def _rejection_reason_counts(rejected_bundles: list[RejectedBundleLog]) -> list[tuple[str, int]]:
    counts = Counter(bundle.reason for bundle in rejected_bundles)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def build_demo_scene(scene_name: str) -> tuple[list[NamedRect], list[NamedRect]]:
    if scene_name == "default":
        pizza = NamedRect("pizza", Rect(-2, 3, 3, 4))
        cutter = NamedRect("cutter", Rect(-4, 3, 1, 6))
        c1 = [pizza, cutter]
        c2 = [pizza, NamedRect("cutter", Rect(-5, 3, 1, 6))]
        return c1, c2

    if scene_name == "three-object":
        pizza = NamedRect("pizza", Rect(-2, 3, 3, 4))
        cutter = NamedRect("cutter", Rect(-4, 3, 1, 6))
        plate = NamedRect("plate", Rect(-3, 0, 8, 1))
        c1 = [pizza, cutter, plate]
        c2 = [
            NamedRect("pizza", Rect(-2, 4, 3, 4)),
            cutter,
            NamedRect("plate", Rect(-3, 1, 8, 1)),
        ]
        return c1, c2

    raise ValueError(f"Unknown scene_name: {scene_name!r}")


def build_ollama_config(config: PipelineConfig) -> OllamaRankerConfig:
    return OllamaRankerConfig(
        model_name=config.model_name,
        timeout_seconds=config.timeout_seconds,
        temperature=config.temperature,
        proposal_temperature=config.proposal_temperature,
        ranking_temperature=config.ranking_temperature,
        keep_alive=config.keep_alive,
    )


def run_pipeline(
    config: PipelineConfig,
    *,
    c1: Optional[list[NamedRect]] = None,
    c2: Optional[list[NamedRect]] = None,
    user_history: Optional[str] = None,
) -> PipelineRunResult:
    pipeline_start = perf_counter()
    if c1 is None or c2 is None:
        c1, c2 = build_demo_scene(config.scene_name)

    ollama_config = build_ollama_config(config)
    eq_pool = detect_c1_equations(
        c1,
        include_offsets=config.include_offsets,
        include_pins=config.include_pins,
        linear_only=config.linear_only,
        print_list=False,
    )
    relation_pool = build_relation_records(eq_pool)

    proposal_results: list[BundleProposalResult] = []
    proposal_round_timings: list[ProposalRoundTiming] = []
    all_proposed_bundles: list[ProposedBundleLog] = []
    proposed_bundles: list[tuple[int, ...]] = []
    previous_bundles: list[tuple[int, ...]] = []
    proposal_phase_start = perf_counter()
    for round_index in range(1, config.proposal_rounds + 1):
        remaining_budget = config.global_bundle_cap - len(proposed_bundles)
        if remaining_budget <= 0:
            break
        requested_max_bundles = min(config.bundles_per_round, remaining_budget)
        round_start = perf_counter()
        proposal_result = propose_relation_bundles_with_ollama(
            c1,
            c2,
            relation_pool,
            ollama_config,
            max_bundles=requested_max_bundles,
            max_bundle_size=None,
            previous_bundles=previous_bundles,
        )
        round_elapsed = perf_counter() - round_start
        proposal_results.append(proposal_result)
        proposal_round_timings.append(
            ProposalRoundTiming(
                round_index=round_index,
                requested_max_bundles=requested_max_bundles,
                returned_bundle_count=len(proposal_result.bundles),
                elapsed_seconds=round_elapsed,
            )
        )
        round_bundles = [bundle.relation_ids for bundle in proposal_result.bundles]
        for bundle in proposal_result.bundles:
            all_proposed_bundles.append(
                ProposedBundleLog(
                    round_index=round_index,
                    candidate_id=bundle.candidate_id,
                    relation_ids=bundle.relation_ids,
                    rationale=bundle.rationale,
                )
            )
        proposed_bundles.extend(round_bundles)
        previous_bundles.extend(round_bundles)
    proposal_phase_seconds = perf_counter() - proposal_phase_start

    context = build_verifier_context(c1, c2, eq_pool)
    proposal_filter_start = perf_counter()
    proposal_reviews = review_proposed_bundle_indices(
        proposed_bundles,
        eq_pool,
        context=context,
        max_bundle_size=None,
        max_candidates=config.global_bundle_cap,
    )
    proposal_filter_phase_seconds = perf_counter() - proposal_filter_start
    candidate_bundle_indices = [
        review.normalized_relation_ids
        for review in proposal_reviews
        if review.accepted
    ]
    rejected_bundles = [
        RejectedBundleLog(
            stage="proposal_filter",
            relation_ids=review.original_relation_ids,
            normalized_relation_ids=review.normalized_relation_ids,
            reason=review.rejection_reason or "rejected",
            round_index=bundle.round_index,
            candidate_id=bundle.candidate_id,
            rationale=bundle.rationale,
        )
        for bundle, review in zip(all_proposed_bundles, proposal_reviews)
        if not review.accepted
    ]
    verification_start = perf_counter()
    verified_records, _census, verification_rejections = verify_and_materialize_candidate_bundles(
        c1,
        c2,
        eq_pool,
        candidate_bundle_indices,
        max_extra_fixed=config.max_extra_fixed,
    )
    verification_phase_seconds = perf_counter() - verification_start
    rejected_bundles.extend(
        RejectedBundleLog(
            stage="verification",
            relation_ids=record.eq_indices,
            normalized_relation_ids=record.eq_indices,
            reason=record.failure_reason or "verification failed",
        )
        for record in verification_rejections
    )
    if not verified_records:
        raise ValueError("No verified bundles were found for ranking.")

    ranking_start = perf_counter()
    ranking_result = rank_verified_bundles_with_ollama(
        verified_records,
        user_history=user_history,
        config=ollama_config,
        top_k=len(verified_records),
        c1=c1,
        c2=c2,
    )
    ranking_phase_seconds = perf_counter() - ranking_start
    candidate_lookup = {
        candidate_id: record
        for candidate_id, record in enumerate(verified_records, start=1)
    }

    return PipelineRunResult(
        c1=c1,
        c2=c2,
        eq_pool_size=len(eq_pool),
        proposal_results=proposal_results,
        proposal_round_timings=proposal_round_timings,
        proposal_phase_seconds=proposal_phase_seconds,
        proposal_filter_phase_seconds=proposal_filter_phase_seconds,
        verification_phase_seconds=verification_phase_seconds,
        ranking_phase_seconds=ranking_phase_seconds,
        total_seconds=perf_counter() - pipeline_start,
        all_proposed_bundles=all_proposed_bundles,
        rejected_bundles=rejected_bundles,
        candidate_bundle_indices=candidate_bundle_indices,
        verified_records=verified_records,
        ranking_result=ranking_result,
        candidate_lookup=candidate_lookup,
    )


def print_pipeline_result(result: PipelineRunResult) -> None:
    _print_heading("Timings")
    for timing in result.proposal_round_timings:
        print(
            f"- generator round {timing.round_index}: {timing.elapsed_seconds:.2f}s "
            f"(requested up to {timing.requested_max_bundles}, got {timing.returned_bundle_count})"
        )
    print(f"- generator total: {result.proposal_phase_seconds:.2f}s")
    print(f"- proposal filter total: {result.proposal_filter_phase_seconds:.2f}s")
    print(f"- verification total: {result.verification_phase_seconds:.2f}s")
    print(f"- ranker total: {result.ranking_phase_seconds:.2f}s")
    print(f"- pipeline total: {result.total_seconds:.2f}s")

    _print_heading("Overview")
    print(f"Equation pool size: {result.eq_pool_size}")
    print(f"Proposal rounds: {len(result.proposal_results)}")
    print(f"All proposed bundles: {len(result.all_proposed_bundles)}")
    print(f"Rejected bundles: {len(result.rejected_bundles)}")
    print(f"Surviving verified bundles: {len(result.verified_records)}")

    _print_heading("All Proposed Bundles")
    if not result.all_proposed_bundles:
        print("- none")
    for index, bundle in enumerate(result.all_proposed_bundles, start=1):
        print(
            f"{index:02d}. round={bundle.round_index} candidate_id={bundle.candidate_id!r} "
            f"relations={bundle.relation_ids}"
        )
        print(f"    rationale: {bundle.rationale}")

    _print_heading("Rejected Bundles")
    if not result.rejected_bundles:
        print("- none")
    else:
        print("Summary by reason:")
        for reason, count in _rejection_reason_counts(result.rejected_bundles):
            print(f"- {reason}: {count}")
        print()
    for bundle in result.rejected_bundles:
        print(
            f"- stage={bundle.stage} | relation_ids={bundle.relation_ids} | "
            f"normalized={bundle.normalized_relation_ids}"
        )
        print(f"  reason: {bundle.reason}")
        if bundle.candidate_id is not None:
            print(f"  candidate_id: {bundle.candidate_id!r}")
        if bundle.rationale:
            print(f"  rationale: {bundle.rationale}")

    _print_heading("Final Ranked Bundles")
    print(f"Ranking summary: {result.ranking_result.summary}")
    print(f"Chosen candidate: {result.ranking_result.chosen_candidate_id}")

    for ranked in result.ranking_result.ranked_candidates:
        record = result.candidate_lookup.get(ranked.candidate_id)
        if record is None:
            continue
        print()
        print(
            f"Rank {ranked.bundle_rank}: candidate {ranked.candidate_id} "
            f"(score={ranked.score}, keep={ranked.keep})"
        )
        print(f"Rationale: {ranked.rationale}")
        print(f"Bundle: {record.eq_indices}")
        if record.viable_fixed_sets:
            option = record.viable_fixed_sets[0]
            print(f"Predicted changes: {format_predicted_changes(option.predicted_changes)}")
            print(f"C3 census count: {option.c3_census_count}")
        print_bundle_summary(record)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Run the C1 -> ranked C3 prediction pipeline.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-json", default="runtime/pipeline_last_run.json")
    parser.add_argument("--scene-name", default="default", choices=["default", "three-object"])
    parser.add_argument("--max-system-size", type=int, default=3)
    parser.add_argument("--max-extra-fixed", type=int, default=2)
    parser.add_argument("--proposal-rounds", type=int, default=2)
    parser.add_argument("--bundles-per-round", type=int, default=24)
    parser.add_argument("--global-bundle-cap", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--proposal-temperature", type=float, default=None)
    parser.add_argument("--ranking-temperature", type=float, default=None)
    parser.add_argument("--disable-offsets", action="store_true")
    parser.add_argument("--disable-pins", action="store_true")
    parser.add_argument("--include-nonlinear", action="store_true")
    args = parser.parse_args()

    return PipelineConfig(
        model_name=args.model_name,
        output_json_path=args.output_json,
        scene_name=args.scene_name,
        max_system_size=args.max_system_size,
        max_extra_fixed=args.max_extra_fixed,
        proposal_rounds=args.proposal_rounds,
        bundles_per_round=args.bundles_per_round,
        global_bundle_cap=args.global_bundle_cap,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        proposal_temperature=args.proposal_temperature,
        ranking_temperature=args.ranking_temperature,
        include_offsets=not args.disable_offsets,
        include_pins=not args.disable_pins,
        linear_only=not args.include_nonlinear,
    )


if __name__ == "__main__":
    pipeline_config = parse_args()
    pipeline_result = run_pipeline(pipeline_config)
    output_path = save_pipeline_result(pipeline_result, pipeline_config.output_json_path)
    print(f"Saved pipeline result JSON: {output_path}")
    print_pipeline_result(pipeline_result)
