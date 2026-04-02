from __future__ import annotations

import unittest

from instructions import build_bundle_proposal_messages, build_bundle_ranking_messages
from milestone1_analysis import (
    BundleRecord,
    ParameterizationRecord,
    collect_candidate_bundle_indices,
    merge_candidate_bundle_indices,
    normalize_candidate_bundle_indices,
    build_verifier_context,
    review_proposed_bundle_indices,
)
from milestone1_core import NamedRect, Rect, build_relation_records, detect_c1_equations


class ProposalPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.c1 = [
            NamedRect("pizza", Rect(-2, 3, 3, 4)),
            NamedRect("cutter", Rect(-4, 3, 1, 6)),
        ]
        self.c2 = [
            NamedRect("pizza", Rect(-2, 3, 3, 4)),
            NamedRect("cutter", Rect(-5, 3, 1, 6)),
        ]

    def test_proposal_messages_include_delta_and_relation_pool(self) -> None:
        eq_pool = detect_c1_equations(self.c1, linear_only=True, print_list=False)
        relation_pool = build_relation_records(eq_pool)

        messages = build_bundle_proposal_messages(
            self.c1,
            self.c2,
            relation_pool,
            max_bundles=8,
            max_bundle_size=3,
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        user_prompt = messages[-1]["content"]
        self.assertIn("Changed parameters (delta):", user_prompt)
        self.assertIn("cutter.x: -4 -> -5", user_prompt)
        self.assertIn("Relation pool:", user_prompt)
        self.assertIn("relation_id=14", user_prompt)
        self.assertIn("Propose at most 8 bundles.", user_prompt)
        self.assertIn("Each bundle must have a unique non-empty candidate_id.", user_prompt)
        self.assertIn("If previous rounds are provided, do not repeat those exact bundles", user_prompt)

    def test_ranking_messages_wrap_existing_prompt_builder(self) -> None:
        records = [
            BundleRecord(
                eq_indices=(14,),
                equations=("+ cutter.x - pizza.x + 2 = + 0",),
                support_vars=("cutter.x", "pizza.x"),
                delta_hit=True,
                changed_vars_hit=("cutter.x",),
                has_shared_variable=False,
                has_connected_support=True,
                verification_passed=True,
                viable_fixed_sets=(
                    ParameterizationRecord(
                        fixed_vars=("cutter.x",),
                        extra_fixed_vars=(),
                        driven_vars=("pizza.x",),
                        predicted_changes=(("pizza.x", -2, -3),),
                        c3_census_count=2,
                    ),
                ),
                min_extra_fixed=0,
                has_parameterization_conflict=False,
                unique_c3_count=1,
                failure_reason=None,
            )
        ]

        messages = build_bundle_ranking_messages(self.c1, self.c2, records, top_k=1)

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Candidate 1:", messages[-1]["content"])
        self.assertIn("Canvas C1:", messages[-1]["content"])
        self.assertIn("unique_c3_count: 1", messages[-1]["content"])
        self.assertIn("c3_census_count=2", messages[-1]["content"])


class ProposalNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.c1 = [
            NamedRect("pizza", Rect(-2, 3, 3, 4)),
            NamedRect("cutter", Rect(-4, 3, 1, 6)),
        ]
        self.c2 = [
            NamedRect("pizza", Rect(-2, 3, 3, 4)),
            NamedRect("cutter", Rect(-5, 3, 1, 6)),
        ]
        self.eq_pool = detect_c1_equations(self.c1, linear_only=True, print_list=False)
        self.context = build_verifier_context(self.c1, self.c2, self.eq_pool)

    def find_eq_index(self, text: str) -> int:
        for idx, eq in enumerate(self.eq_pool):
            if eq.pretty() == text:
                return idx
        raise AssertionError(f"Equation not found: {text}")

    def test_normalize_candidate_bundle_indices_applies_phase3_filters(self) -> None:
        cutter_height_pin = self.find_eq_index("+ cutter.h - 6 = + 0")
        cutter_x_pin = self.find_eq_index("+ cutter.x + 4 = + 0")
        cutter_left_of_pizza = self.find_eq_index("+ cutter.x - pizza.x + 2 = + 0")

        normalized = normalize_candidate_bundle_indices(
            [
                [cutter_height_pin],  # no delta overlap
                [cutter_x_pin],       # contradiction after fixing delta
                [cutter_left_of_pizza, cutter_height_pin],  # valid
                [cutter_left_of_pizza, cutter_height_pin],  # duplicate
            ],
            self.eq_pool,
            self.context,
            max_bundle_size=3,
        )

        self.assertEqual(normalized, [(cutter_height_pin, cutter_left_of_pizza)])

    def test_review_proposed_bundle_indices_records_rejection_reasons(self) -> None:
        cutter_height_pin = self.find_eq_index("+ cutter.h - 6 = + 0")
        cutter_x_pin = self.find_eq_index("+ cutter.x + 4 = + 0")
        cutter_left_of_pizza = self.find_eq_index("+ cutter.x - pizza.x + 2 = + 0")

        reviews = review_proposed_bundle_indices(
            [
                [cutter_height_pin],
                [cutter_x_pin],
                [cutter_left_of_pizza, cutter_height_pin],
                [cutter_left_of_pizza, cutter_height_pin],
            ],
            self.eq_pool,
            self.context,
            max_bundle_size=3,
        )

        self.assertEqual([review.accepted for review in reviews], [False, False, True, False])
        self.assertEqual(reviews[0].rejection_reason, "filtered out by delta-overlap heuristic")
        self.assertEqual(reviews[1].rejection_reason, "contradiction after fixing edited variables")
        self.assertIsNone(reviews[2].rejection_reason)
        self.assertEqual(reviews[3].rejection_reason, "duplicate normalized bundle")

    def test_merge_and_collect_candidate_bundle_indices_union_results(self) -> None:
        cutter_left_of_pizza = self.find_eq_index("+ cutter.x - pizza.x + 2 = + 0")
        merged = merge_candidate_bundle_indices([(cutter_left_of_pizza,)], [(cutter_left_of_pizza,), (0, 14)])
        self.assertEqual(merged[0], (cutter_left_of_pizza,))
        self.assertEqual(len(merged), 2)

        collected = collect_candidate_bundle_indices(
            self.c1,
            self.c2,
            self.eq_pool,
            proposed_bundles=[(0, 14), (cutter_left_of_pizza,)],
            max_system_size=1,
            max_bundle_size=2,
            max_candidates=20,
        )

        self.assertIn((cutter_left_of_pizza,), collected)
        self.assertIn((0, 14), collected)


if __name__ == "__main__":
    unittest.main()
