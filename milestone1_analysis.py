from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
import json
from typing import Dict, Iterable, Optional, Set, Tuple

from milestone1_core import (
    Equation,
    NamedRect,
    ProgramProvenance,
    build_env,
    build_lin_system_cached,
    changed_params,
    compile_repair_program,
    equation_support_vars,
    linear_relation_to_linexpr,
    system_is_connected,
)


@dataclass(frozen=True)
class ParameterizationRecord:
    """
    One viable program shape for a bundle.

    A single equation bundle can admit multiple parameterizations depending on
    which extra variables are fixed versus solved.
    """
    fixed_vars: Tuple[str, ...]
    extra_fixed_vars: Tuple[str, ...]
    driven_vars: Tuple[str, ...]
    predicted_changes: Optional[Tuple[Tuple[str, Fraction, Fraction], ...]] = None
    c3_key: Optional[Tuple[Tuple[str, Fraction], ...]] = None
    c3_census_count: Optional[int] = None
    program_text: Optional[str] = None


@dataclass(frozen=True)
class BundleRecord:
    """
    One candidate equation bundle plus its viable parameterizations.

    `eq_indices` identifies the bundle E. `viable_fixed_sets` contains the
    parameterization choices for that same bundle.
    """
    eq_indices: Tuple[int, ...]
    equations: Tuple[str, ...]
    support_vars: Tuple[str, ...]
    delta_hit: bool
    changed_vars_hit: Tuple[str, ...]
    has_shared_variable: bool
    has_connected_support: bool
    verification_passed: bool
    viable_fixed_sets: Tuple[ParameterizationRecord, ...]
    min_extra_fixed: Optional[int]
    has_parameterization_conflict: bool
    unique_c3_count: int
    failure_reason: Optional[str]


@dataclass(frozen=True)
class ProposedBundleReview:
    original_relation_ids: Tuple[int, ...]
    normalized_relation_ids: Tuple[int, ...]
    accepted: bool
    rejection_reason: Optional[str]


@dataclass(frozen=True)
class VerifierContext:
    delta: frozenset[str]
    const_env: Dict[str, Fraction]
    global_vars: Tuple[str, ...]
    global_index: Dict[str, int]
    support_by_index: Dict[int, frozenset[str]]
    equation_text_by_index: Dict[int, str]
    row_by_index: Dict[int, Optional[Tuple[Fraction, ...]]]
    rhs_by_index: Dict[int, Optional[Fraction]]


def build_verifier_context(
    c1: list[NamedRect],
    c2: list[NamedRect],
    eq_pool: list[Equation],
) -> VerifierContext:
    global_vars = tuple(sorted(build_env(c1).keys()))
    global_index = {var: idx for idx, var in enumerate(global_vars)}

    support_by_index: Dict[int, frozenset[str]] = {}
    equation_text_by_index: Dict[int, str] = {}
    row_by_index: Dict[int, Optional[Tuple[Fraction, ...]]] = {}
    rhs_by_index: Dict[int, Optional[Fraction]] = {}

    for idx, eq in enumerate(eq_pool):
        support_by_index[idx] = frozenset(equation_support_vars(eq))
        equation_text_by_index[idx] = eq.pretty()

        linexpr = linear_relation_to_linexpr(eq)
        if linexpr is None:
            row_by_index[idx] = None
            rhs_by_index[idx] = None
            continue

        row = [Fraction(0) for _ in global_vars]
        for var, coeff in linexpr.coeffs:
            row[global_index[var]] = coeff
        row_by_index[idx] = tuple(row)
        rhs_by_index[idx] = -linexpr.const

    return VerifierContext(
        delta=frozenset(changed_params(c1, c2)),
        const_env=build_env(c2),
        global_vars=global_vars,
        global_index=global_index,
        support_by_index=support_by_index,
        equation_text_by_index=equation_text_by_index,
        row_by_index=row_by_index,
        rhs_by_index=rhs_by_index,
    )


def format_fraction(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"


def format_predicted_changes(
    changes: Optional[Tuple[Tuple[str, Fraction, Fraction], ...]],
) -> str:
    if changes is None:
        return "not materialized"
    if not changes:
        return "none"
    return ", ".join(
        f"{var}: {format_fraction(before)} -> {format_fraction(after)}"
        for var, before, after in changes
    )


def summarize_canvas_delta(
    before: list[NamedRect],
    after: list[NamedRect],
) -> Tuple[Tuple[str, Fraction, Fraction], ...]:
    before_env = build_env(before)
    after_env = build_env(after)
    changes = []
    for var in sorted(before_env.keys()):
        if before_env[var] != after_env[var]:
            changes.append((var, before_env[var], after_env[var]))
    return tuple(changes)


def c3_outcome_key(canvas: list[NamedRect]) -> Tuple[Tuple[str, Fraction], ...]:
    env = build_env(canvas)
    return tuple(sorted(env.items()))


def matrix_rank(matrix: list[list[Fraction]]) -> int:
    if not matrix:
        return 0
    if not matrix[0]:
        return 0

    work = [row[:] for row in matrix]
    row_count = len(work)
    col_count = len(work[0])
    row = 0
    rank = 0

    for col in range(col_count):
        pivot = None
        for r in range(row, row_count):
            if work[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue

        work[row], work[pivot] = work[pivot], work[row]
        pivot_value = work[row][col]
        inv = Fraction(1, 1) / pivot_value
        work[row] = [value * inv for value in work[row]]

        for r in range(row_count):
            if r != row and work[r][col] != 0:
                factor = work[r][col]
                work[r] = [work[r][c] - factor * work[row][c] for c in range(col_count)]

        row += 1
        rank += 1
        if row == row_count:
            break

    return rank


def bundle_support_vars(context: VerifierContext, idxs: Tuple[int, ...]) -> Tuple[str, ...]:
    support: Set[str] = set()
    for idx in idxs:
        support |= context.support_by_index[idx]
    return tuple(sorted(support))


def bundle_has_shared_variable(context: VerifierContext, idxs: Tuple[int, ...]) -> bool:
    if len(idxs) <= 1:
        return False

    counts: Counter[str] = Counter()
    for idx in idxs:
        counts.update(context.support_by_index[idx])
    return any(count >= 2 for count in counts.values())


def bundle_has_individual_contradiction(
    idxs: Tuple[int, ...],
    const_vars: Set[str],
    context: VerifierContext,
) -> bool:
    """
    Fast pre-check: after substituting edited variables as constants from C2,
    does any single equation collapse to `0 = nonzero`?
    """
    const_cols = {context.global_index[var]: context.const_env[var] for var in const_vars}

    for idx in idxs:
        row_full = context.row_by_index[idx]
        rhs_value = context.rhs_by_index[idx]
        if row_full is None or rhs_value is None:
            continue

        adjusted_rhs = rhs_value
        has_nonconst_term = False
        for col, coeff in enumerate(row_full):
            if coeff == 0:
                continue
            if col in const_cols:
                adjusted_rhs -= coeff * const_cols[col]
            else:
                has_nonconst_term = True

        if not has_nonconst_term and adjusted_rhs != 0:
            return True

    return False


def phase3_filter_reason(
    idxs: Tuple[int, ...],
    context: VerifierContext,
) -> Optional[str]:
    support_vars = bundle_support_vars(context, idxs)
    changed_vars_hit = tuple(var for var in support_vars if var in context.delta)
    if not changed_vars_hit:
        return "filtered out by delta-overlap heuristic"
    if any(context.row_by_index[idx] is None for idx in idxs):
        return "no unique solution within current parameterization policy"
    if bundle_has_individual_contradiction(idxs, set(changed_vars_hit), context):
        return "contradiction after fixing edited variables"
    return None


def enumerate_phase3_candidate_bundles(
    c1: list[NamedRect],
    c2: list[NamedRect],
    eq_pool: list[Equation],
    max_system_size: int = 5,
    context: Optional[VerifierContext] = None,
) -> list[Tuple[int, ...]]:
    context = context or build_verifier_context(c1, c2, eq_pool)
    candidates: list[Tuple[int, ...]] = []
    for size in range(1, min(max_system_size, len(eq_pool)) + 1):
        for idxs in combinations(range(len(eq_pool)), size):
            idxs_tuple = tuple(idxs)
            if phase3_filter_reason(idxs_tuple, context) is None:
                candidates.append(idxs_tuple)
    return candidates


def normalize_candidate_bundle_indices(
    proposed_bundles: Iterable[Iterable[int]],
    eq_pool: list[Equation],
    context: VerifierContext,
    max_bundle_size: Optional[int],
    max_candidates: Optional[int] = None,
) -> list[Tuple[int, ...]]:
    reviews = review_proposed_bundle_indices(
        proposed_bundles,
        eq_pool,
        context,
        max_bundle_size=max_bundle_size,
        max_candidates=max_candidates,
    )
    return [
        review.normalized_relation_ids
        for review in reviews
        if review.accepted
    ]


def review_proposed_bundle_indices(
    proposed_bundles: Iterable[Iterable[int]],
    eq_pool: list[Equation],
    context: VerifierContext,
    *,
    max_bundle_size: Optional[int],
    max_candidates: Optional[int] = None,
) -> list[ProposedBundleReview]:
    if max_bundle_size is not None and max_bundle_size <= 0:
        raise ValueError("max_bundle_size must be positive.")

    valid_relation_ids = set(range(len(eq_pool)))
    seen: Set[Tuple[int, ...]] = set()
    accepted_count = 0
    reviews: list[ProposedBundleReview] = []

    for bundle in proposed_bundles:
        original_tuple = tuple(bundle)
        normalized_tuple = tuple(sorted(set(bundle)))
        rejection_reason: Optional[str] = None

        if not normalized_tuple:
            rejection_reason = "empty bundle"
        elif max_bundle_size is not None and len(normalized_tuple) > max_bundle_size:
            rejection_reason = "bundle exceeds configured maximum size"
        elif any(idx not in valid_relation_ids for idx in normalized_tuple):
            rejection_reason = "bundle contains an invalid relation id"
        elif normalized_tuple in seen:
            rejection_reason = "duplicate normalized bundle"
        else:
            filter_reason = phase3_filter_reason(normalized_tuple, context)
            if filter_reason is not None:
                rejection_reason = filter_reason
            elif max_candidates is not None and accepted_count >= max_candidates:
                rejection_reason = "exceeds configured proposal cap"

        accepted = rejection_reason is None
        if accepted:
            seen.add(normalized_tuple)
            accepted_count += 1

        reviews.append(
            ProposedBundleReview(
                original_relation_ids=original_tuple,
                normalized_relation_ids=normalized_tuple,
                accepted=accepted,
                rejection_reason=rejection_reason,
            )
        )

    return reviews


def merge_candidate_bundle_indices(
    symbolic_candidates: Iterable[Tuple[int, ...]],
    proposed_candidates: Iterable[Tuple[int, ...]],
    *,
    max_candidates: Optional[int] = None,
) -> list[Tuple[int, ...]]:
    merged: list[Tuple[int, ...]] = []
    seen: Set[Tuple[int, ...]] = set()
    for candidate in list(symbolic_candidates) + list(proposed_candidates):
        if candidate in seen:
            continue
        seen.add(candidate)
        merged.append(candidate)
        if max_candidates is not None and len(merged) >= max_candidates:
            break
    return merged


def collect_candidate_bundle_indices(
    c1: list[NamedRect],
    c2: list[NamedRect],
    eq_pool: list[Equation],
    *,
    proposed_bundles: Optional[Iterable[Iterable[int]]] = None,
    max_system_size: int = 3,
    max_bundle_size: int = 3,
    symbolic_pool_limit: int = 30,
    symbolic_system_size_limit: int = 3,
    max_candidates: int = 128,
    context: Optional[VerifierContext] = None,
) -> list[Tuple[int, ...]]:
    context = context or build_verifier_context(c1, c2, eq_pool)
    if len(eq_pool) <= symbolic_pool_limit and max_system_size <= symbolic_system_size_limit:
        symbolic_candidates = enumerate_phase3_candidate_bundles(
            c1,
            c2,
            eq_pool,
            max_system_size=max_system_size,
            context=context,
        )
    else:
        symbolic_candidates = []

    normalized_proposals = normalize_candidate_bundle_indices(
        proposed_bundles or [],
        eq_pool,
        context,
        max_bundle_size=max_bundle_size,
    )
    return merge_candidate_bundle_indices(
        symbolic_candidates,
        normalized_proposals,
        max_candidates=max_candidates,
    )


def rhs_in_column_space(matrix: list[list[Fraction]], rhs: list[Fraction]) -> bool:
    if not matrix:
        return all(value == 0 for value in rhs)
    augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    return matrix_rank(augmented) == matrix_rank(matrix)


def nonconst_system_for_bundle(
    idxs: Tuple[int, ...],
    support_vars: Tuple[str, ...],
    const_vars: Set[str],
    context: VerifierContext,
) -> Optional[Tuple[Tuple[str, ...], list[list[Fraction]], list[Fraction]]]:
    nonconst_vars = tuple(v for v in support_vars if v not in const_vars)
    const_partition = tuple(v for v in support_vars if v in const_vars)

    nonconst_cols = [context.global_index[v] for v in nonconst_vars]
    const_cols = [context.global_index[v] for v in const_partition]
    const_values = [context.const_env[v] for v in const_partition]

    a_nonconst: list[list[Fraction]] = []
    rhs_adjusted: list[Fraction] = []

    for idx in idxs:
        row_full = context.row_by_index[idx]
        rhs_value = context.rhs_by_index[idx]
        if row_full is None or rhs_value is None:
            return None

        value = rhs_value
        for col, const_value in zip(const_cols, const_values):
            value -= row_full[col] * const_value

        a_nonconst.append([row_full[col] for col in nonconst_cols])
        rhs_adjusted.append(value)

    return nonconst_vars, a_nonconst, rhs_adjusted


def minimal_parameterizations(
    idxs: Tuple[int, ...],
    support_vars: Tuple[str, ...],
    const_vars: Set[str],
    context: VerifierContext,
    max_extra_fixed: Optional[int] = None,
) -> Tuple[Optional[int], Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...], Optional[str]]:
    bundle_system = nonconst_system_for_bundle(idxs, support_vars, const_vars, context)
    if bundle_system is None:
        return None, (), "no unique solution within current parameterization policy"

    nonconst_vars, a_nonconst, rhs_adjusted = bundle_system

    if not nonconst_vars:
        return None, (), "not enough constraints unless too many variables are fixed"

    rank_nonconst = matrix_rank(a_nonconst)
    if not rhs_in_column_space(a_nonconst, rhs_adjusted):
        return None, (), "contradiction after fixing edited variables"

    if rank_nonconst == 0:
        return None, (), "not enough constraints unless too many variables are fixed"

    driven_size = rank_nonconst
    viable: list[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
    index_by_var = {var: pos for pos, var in enumerate(nonconst_vars)}

    for driven_vars in combinations(nonconst_vars, driven_size):
        driven_cols = [index_by_var[var] for var in driven_vars]
        a_driven = [[row[j] for j in driven_cols] for row in a_nonconst]
        if matrix_rank(a_driven) != driven_size:
            continue

        extra_fixed_vars = tuple(var for var in nonconst_vars if var not in driven_vars)
        if max_extra_fixed is not None and len(extra_fixed_vars) > max_extra_fixed:
            continue
        viable.append((tuple(driven_vars), extra_fixed_vars))

    if not viable:
        return None, (), "no unique solution within current parameterization policy"

    viable.sort(key=lambda item: (item[1], item[0]))
    min_extra_fixed = len(viable[0][1])
    return min_extra_fixed, tuple(viable), None


def materialize_parameterization(
    c2: list[NamedRect],
    eqs: list[Equation],
    prebuilt: tuple[Tuple[str, ...], Dict[str, int], list[list[Fraction]], list[Fraction]],
    const_vars_in_bundle: Tuple[str, ...],
    extra_fixed_vars: Tuple[str, ...],
    driven_vars: Tuple[str, ...],
    const_env: Dict[str, Fraction],
) -> ParameterizationRecord:
    fixed_vars = set(const_vars_in_bundle) | set(extra_fixed_vars)
    program = compile_repair_program(
        eqs=eqs,
        prebuilt=prebuilt,
        fixed_vars=fixed_vars,
        const_vars=set(const_vars_in_bundle),
        const_env=const_env,
    )
    if program is None:
        raise ValueError("Expected a valid repair program for a viable parameterization.")

    c3 = program.apply(c2)
    outcome_key = c3_outcome_key(c3)
    return ParameterizationRecord(
        fixed_vars=tuple(sorted(program.const_vars + program.fixed_vars)),
        extra_fixed_vars=tuple(sorted(program.fixed_vars)),
        driven_vars=tuple(sorted(program.driven_vars)),
        predicted_changes=summarize_canvas_delta(c2, c3),
        c3_key=outcome_key,
        program_text=program.pretty(),
    )


def verify_bundle(
    c1: list[NamedRect],
    c2: list[NamedRect],
    eq_pool: list[Equation],
    idxs: Tuple[int, ...],
    context: Optional[VerifierContext] = None,
    include_details: bool = False,
    max_extra_fixed: Optional[int] = None,
) -> BundleRecord:
    context = context or build_verifier_context(c1, c2, eq_pool)
    eqs = [eq_pool[i] for i in idxs]
    equations = tuple(context.equation_text_by_index[i] for i in idxs)
    support_vars = bundle_support_vars(context, idxs)
    changed_vars_hit = tuple(var for var in support_vars if var in context.delta)
    shared_variable = bundle_has_shared_variable(context, idxs)
    connected_support = system_is_connected(eqs)

    filter_reason = phase3_filter_reason(idxs, context)
    if filter_reason == "filtered out by delta-overlap heuristic":
        return BundleRecord(
            eq_indices=idxs,
            equations=equations,
            support_vars=support_vars,
            delta_hit=False,
            changed_vars_hit=(),
            has_shared_variable=shared_variable,
            has_connected_support=connected_support,
            verification_passed=False,
            viable_fixed_sets=(),
            min_extra_fixed=None,
            has_parameterization_conflict=False,
            unique_c3_count=0,
            failure_reason=filter_reason,
        )

    if filter_reason == "no unique solution within current parameterization policy":
        return BundleRecord(
            eq_indices=idxs,
            equations=equations,
            support_vars=support_vars,
            delta_hit=True,
            changed_vars_hit=changed_vars_hit,
            has_shared_variable=shared_variable,
            has_connected_support=connected_support,
            verification_passed=False,
            viable_fixed_sets=(),
            min_extra_fixed=None,
            has_parameterization_conflict=False,
            unique_c3_count=0,
            failure_reason=filter_reason,
        )

    if filter_reason == "contradiction after fixing edited variables":
        return BundleRecord(
            eq_indices=idxs,
            equations=equations,
            support_vars=support_vars,
            delta_hit=True,
            changed_vars_hit=changed_vars_hit,
            has_shared_variable=shared_variable,
            has_connected_support=connected_support,
            verification_passed=False,
            viable_fixed_sets=(),
            min_extra_fixed=None,
            has_parameterization_conflict=False,
            unique_c3_count=0,
            failure_reason=filter_reason,
        )

    min_extra_fixed, viable_pairs, failure_reason = minimal_parameterizations(
        idxs,
        support_vars,
        set(changed_vars_hit),
        context,
        max_extra_fixed=max_extra_fixed,
    )
    if not viable_pairs:
        return BundleRecord(
            eq_indices=idxs,
            equations=equations,
            support_vars=support_vars,
            delta_hit=True,
            changed_vars_hit=changed_vars_hit,
            has_shared_variable=shared_variable,
            has_connected_support=connected_support,
            verification_passed=False,
            viable_fixed_sets=(),
            min_extra_fixed=None,
            has_parameterization_conflict=False,
            unique_c3_count=0,
            failure_reason=failure_reason,
        )

    if include_details:
        prebuilt = build_lin_system_cached(eqs)
        if prebuilt is None:
            raise ValueError("Expected a cached linear system for a viable linear bundle.")
        viable_fixed_sets = tuple(
            materialize_parameterization(
                c2=c2,
                eqs=eqs,
                prebuilt=prebuilt,
                const_vars_in_bundle=changed_vars_hit,
                extra_fixed_vars=extra_fixed_vars,
                driven_vars=driven_vars,
                const_env=context.const_env,
            )
            for driven_vars, extra_fixed_vars in viable_pairs
        )
    else:
        viable_fixed_sets = tuple(
            ParameterizationRecord(
                fixed_vars=tuple(sorted(set(changed_vars_hit) | set(extra_fixed_vars))),
                extra_fixed_vars=tuple(sorted(extra_fixed_vars)),
                driven_vars=tuple(sorted(driven_vars)),
            )
            for driven_vars, extra_fixed_vars in viable_pairs
        )

    return BundleRecord(
        eq_indices=idxs,
        equations=equations,
        support_vars=support_vars,
        delta_hit=True,
        changed_vars_hit=changed_vars_hit,
        has_shared_variable=shared_variable,
        has_connected_support=connected_support,
        verification_passed=True,
        viable_fixed_sets=viable_fixed_sets,
        min_extra_fixed=min_extra_fixed,
        has_parameterization_conflict=False,
        unique_c3_count=0,
        failure_reason=None,
    )


def build_c3_census(
    records: list[BundleRecord],
) -> Dict[Tuple[Tuple[str, Fraction], ...], list[ProgramProvenance]]:
    census: Dict[Tuple[Tuple[str, Fraction], ...], list[ProgramProvenance]] = {}
    for record in records:
        for option in record.viable_fixed_sets:
            if option.c3_key is None:
                continue
            census.setdefault(option.c3_key, []).append(
                ProgramProvenance(
                    eq_indices=record.eq_indices,
                    fixed_vars=option.fixed_vars,
                    program_text=option.program_text or "",
                )
            )
    return census


def annotate_records_with_c3_census(
    records: list[BundleRecord],
    census: Dict[Tuple[Tuple[str, Fraction], ...], list[ProgramProvenance]],
) -> list[BundleRecord]:
    annotated_records: list[BundleRecord] = []
    for record in records:
        unique_c3_keys = {option.c3_key for option in record.viable_fixed_sets if option.c3_key is not None}
        annotated_options = tuple(
            replace(
                option,
                c3_census_count=(
                    len(census.get(option.c3_key, []))
                    if option.c3_key is not None
                    else option.c3_census_count
                ),
            )
            for option in record.viable_fixed_sets
        )
        annotated_records.append(
            replace(
                record,
                viable_fixed_sets=annotated_options,
                has_parameterization_conflict=len(unique_c3_keys) > 1,
                unique_c3_count=len(unique_c3_keys),
            )
        )
    return annotated_records


def analyze_relation_bundles(
    c1: list[NamedRect],
    c2: list[NamedRect],
    eq_pool: list[Equation],
    max_system_size: int = 5,
    context: Optional[VerifierContext] = None,
    max_extra_fixed: Optional[int] = None,
) -> Tuple[Dict[str, int], list[BundleRecord]]:
    context = context or build_verifier_context(c1, c2, eq_pool)
    stats = {
        "pool_size": len(eq_pool),
        "max_system_size": max_system_size,
        "naive_total": 0,
        "delta_overlap": 0,
        "shared_variable": 0,
        "verifier_passed": 0,
    }
    accepted_records: list[BundleRecord] = []

    for size in range(1, min(max_system_size, len(eq_pool)) + 1):
        for idxs in combinations(range(len(eq_pool)), size):
            stats["naive_total"] += 1
            record = verify_bundle(
                c1,
                c2,
                eq_pool,
                tuple(idxs),
                context=context,
                max_extra_fixed=max_extra_fixed,
            )
            if record.delta_hit:
                stats["delta_overlap"] += 1
            if record.delta_hit and record.has_shared_variable:
                stats["shared_variable"] += 1
            if record.verification_passed:
                stats["verifier_passed"] += 1
                accepted_records.append(record)

    accepted_records.sort(
        key=lambda item: (
            item.min_extra_fixed if item.min_extra_fixed is not None else 10**9,
            len(item.eq_indices),
            0 if item.has_shared_variable else 1,
            item.eq_indices,
        )
    )
    return stats, accepted_records


def materialize_bundle_records(
    c1: list[NamedRect],
    c2: list[NamedRect],
    eq_pool: list[Equation],
    records: list[BundleRecord],
    context: Optional[VerifierContext] = None,
    max_extra_fixed: Optional[int] = None,
) -> list[BundleRecord]:
    context = context or build_verifier_context(c1, c2, eq_pool)
    detailed_records = [
        verify_bundle(
            c1,
            c2,
            eq_pool,
            record.eq_indices,
            context=context,
            include_details=True,
            max_extra_fixed=max_extra_fixed,
        )
        for record in records
    ]
    census = build_c3_census([record for record in detailed_records if record.verification_passed])
    return annotate_records_with_c3_census(detailed_records, census)


def verify_and_materialize_candidate_bundles(
    c1: list[NamedRect],
    c2: list[NamedRect],
    eq_pool: list[Equation],
    candidate_bundle_indices: list[Tuple[int, ...]],
    context: Optional[VerifierContext] = None,
    max_extra_fixed: Optional[int] = None,
) -> tuple[
    list[BundleRecord],
    Dict[Tuple[Tuple[str, Fraction], ...], list[ProgramProvenance]],
    list[BundleRecord],
]:
    context = context or build_verifier_context(c1, c2, eq_pool)
    seen: Set[Tuple[int, ...]] = set()
    verified_records: list[BundleRecord] = []
    rejected_records: list[BundleRecord] = []
    for idxs in candidate_bundle_indices:
        idxs_tuple = tuple(idxs)
        if idxs_tuple in seen:
            continue
        seen.add(idxs_tuple)
        record = verify_bundle(
            c1,
            c2,
            eq_pool,
            idxs_tuple,
            context=context,
            include_details=True,
            max_extra_fixed=max_extra_fixed,
        )
        if record.verification_passed:
            verified_records.append(record)
        else:
            rejected_records.append(record)

    census = build_c3_census(verified_records)
    annotated_records = annotate_records_with_c3_census(verified_records, census)
    return annotated_records, census, rejected_records


def print_bundle_summary(record: BundleRecord) -> None:
    print(f"Bundle {record.eq_indices}")
    for eq_idx, eq_text in zip(record.eq_indices, record.equations):
        print(f"  {eq_idx:03d}: {eq_text}")
    print(f"  support vars: {record.support_vars}")
    print(f"  hits delta: {record.changed_vars_hit}")
    print(f"  shared-variable heuristic: {record.has_shared_variable}")
    print(f"  connected-support heuristic: {record.has_connected_support}")
    if record.verification_passed:
        print(f"  verifier: kept | min_extra_fixed={record.min_extra_fixed}")
        print(f"  unique_c3_count: {record.unique_c3_count}")
        print(f"  parameterization_conflict: {record.has_parameterization_conflict}")
        print("  tied viable fixed sets:")
        for option in record.viable_fixed_sets:
            print(
                "    "
                f"fixed={option.fixed_vars} | extra={option.extra_fixed_vars} | "
                f"driven={option.driven_vars} | "
                f"changes={format_predicted_changes(option.predicted_changes)} | "
                f"c3_census_count={option.c3_census_count}"
            )
    else:
        print(f"  verifier: rejected | reason={record.failure_reason}")


def print_search_report(stats: Dict[str, int], accepted_records: list[BundleRecord]) -> None:
    print(f"Equation pool size: {stats['pool_size']}")
    print(f"Naive subsets up to size {stats['max_system_size']}: {stats['naive_total']}")
    print()
    print("Primary funnel:")
    print(f"  naive -> {stats['naive_total']}")
    print(f"  delta-overlap -> {stats['delta_overlap']}")
    print(f"  verifier -> {stats['verifier_passed']}")
    print()
    print("Diagnostic view:")
    print(f"  naive -> {stats['naive_total']}")
    print(f"  delta-overlap -> {stats['delta_overlap']}")
    print(f"  shared-variable -> {stats['shared_variable']}")
    print()
    distribution = Counter(record.min_extra_fixed for record in accepted_records)
    print("Minimum-extra-fixed distribution:")
    for extra_count in sorted(distribution.keys()):
        print(f"  {extra_count}: {distribution[extra_count]}")


def print_surviving_bundle_ids(records: list[BundleRecord], limit: int = 30) -> None:
    ids = [record.eq_indices for record in records[:limit]]
    print(f"First {len(ids)} surviving bundle ids:")
    print(ids)


def print_top_verified_bundles(records: list[BundleRecord], top_k: int = 10) -> None:
    for record in records[:top_k]:
        print()
        print_bundle_summary(record)


def build_llm_ranking_prompt(
    c1: list[NamedRect],
    c2: list[NamedRect],
    records: list[BundleRecord],
    top_k: int = 5,
    user_history: Optional[str] = None,
    max_ties_per_bundle: int = 3,
    schema: Optional[dict] = None,
) -> tuple[str, dict[int, BundleRecord]]:
    from instructions import build_bundle_ranking_prompt

    return build_bundle_ranking_prompt(
        c1,
        c2,
        records,
        top_k=top_k,
        user_history=user_history,
        max_ties_per_bundle=max_ties_per_bundle,
        schema=schema,
    )


def build_llm_prompt(
    c1: list[NamedRect],
    c2: list[NamedRect],
    records: list[BundleRecord],
    top_k: int = 5,
    user_history: Optional[str] = None,
    max_ties_per_bundle: int = 3,
    schema: Optional[dict] = None,
) -> str:
    prompt_text, _ = build_llm_ranking_prompt(
        c1,
        c2,
        records,
        top_k=top_k,
        user_history=user_history,
        max_ties_per_bundle=max_ties_per_bundle,
        schema=schema,
    )
    return prompt_text
