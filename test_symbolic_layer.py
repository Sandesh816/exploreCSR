from __future__ import annotations

import unittest

from milestone1_analysis import (
    BundleRecord,
    ParameterizationRecord,
    analyze_relation_bundles,
    verify_bundle,
)
from milestone1_core import (
    NamedRect,
    Rect,
    build_relation_records,
    changed_params,
    detect_c1_equations,
)


class ChangedParamsTests(unittest.TestCase):
    def test_changed_params_handles_multiple_changes_across_three_objects(self) -> None:
        c1 = [
            NamedRect("pizza", Rect(-2, 3, 3, 4)),
            NamedRect("cutter", Rect(-4, 3, 1, 6)),
            NamedRect("plate", Rect(-3, 0, 8, 1)),
        ]
        c2 = [
            NamedRect("pizza", Rect(-2, 4, 3, 4)),
            NamedRect("cutter", Rect(-4, 3, 2, 6)),
            NamedRect("plate", Rect(-3, 0, 8, 1)),
        ]

        self.assertEqual(changed_params(c1, c2), {"pizza.y", "cutter.w"})

    def test_changed_params_rejects_mismatched_variable_universe(self) -> None:
        c1 = [NamedRect("pizza", Rect(-2, 3, 3, 4))]
        c2 = [NamedRect("cutter", Rect(-2, 3, 3, 4))]

        with self.assertRaises(ValueError):
            changed_params(c1, c2)


class RelationRecordTests(unittest.TestCase):
    def test_build_relation_records_works_for_three_object_scene(self) -> None:
        c1 = [
            NamedRect("pizza", Rect(-2, 3, 3, 4)),
            NamedRect("cutter", Rect(-4, 3, 1, 6)),
            NamedRect("plate", Rect(-3, 0, 8, 1)),
        ]

        eq_pool = detect_c1_equations(c1, linear_only=True, print_list=False)
        records = build_relation_records(eq_pool)

        self.assertEqual([record.relation_id for record in records], list(range(len(records))))
        self.assertTrue(records)
        self.assertTrue(all(record.equation_text for record in records))
        self.assertTrue(all(isinstance(record.support_vars, tuple) for record in records))


class BundleModelTests(unittest.TestCase):
    def test_bundle_record_keeps_parameterizations_separate(self) -> None:
        c1 = [
            NamedRect("pizza", Rect(-2, 3, 3, 4)),
            NamedRect("cutter", Rect(-4, 3, 1, 6)),
        ]
        c2 = [
            NamedRect("pizza", Rect(-2, 3, 3, 4)),
            NamedRect("cutter", Rect(-5, 3, 1, 6)),
        ]

        eq_pool = detect_c1_equations(c1, linear_only=True, print_list=False)
        _, records = analyze_relation_bundles(c1, c2, eq_pool, max_system_size=2)

        self.assertTrue(records)
        record = records[0]

        self.assertIsInstance(record, BundleRecord)
        self.assertIsInstance(record.eq_indices, tuple)
        self.assertIsInstance(record.viable_fixed_sets, tuple)
        self.assertTrue(record.viable_fixed_sets)
        self.assertTrue(all(isinstance(option, ParameterizationRecord) for option in record.viable_fixed_sets))


class DeterministicFilterTests(unittest.TestCase):
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

    def find_eq_index(self, text: str) -> int:
        for idx, eq in enumerate(self.eq_pool):
            if eq.pretty() == text:
                return idx
        raise AssertionError(f"Equation not found: {text}")

    def test_delta_overlap_is_hard_gate(self) -> None:
        cutter_height_pin = self.find_eq_index("+ cutter.h - 6 = + 0")

        record = verify_bundle(self.c1, self.c2, self.eq_pool, (cutter_height_pin,))

        self.assertFalse(record.delta_hit)
        self.assertFalse(record.verification_passed)
        self.assertEqual(record.failure_reason, "filtered out by delta-overlap heuristic")

    def test_individual_equation_contradiction_is_hard_gate(self) -> None:
        cutter_x_pin = self.find_eq_index("+ cutter.x + 4 = + 0")

        record = verify_bundle(self.c1, self.c2, self.eq_pool, (cutter_x_pin,))

        self.assertTrue(record.delta_hit)
        self.assertFalse(record.verification_passed)
        self.assertEqual(record.failure_reason, "contradiction after fixing edited variables")

    def test_structural_heuristics_are_diagnostic_only(self) -> None:
        cutter_height_pin = self.find_eq_index("+ cutter.h - 6 = + 0")
        cutter_left_of_pizza = self.find_eq_index("+ cutter.x - pizza.x + 2 = + 0")

        record = verify_bundle(
            self.c1,
            self.c2,
            self.eq_pool,
            (cutter_height_pin, cutter_left_of_pizza),
        )

        self.assertTrue(record.delta_hit)
        self.assertFalse(record.has_shared_variable)
        self.assertFalse(record.has_connected_support)
        self.assertTrue(record.verification_passed)


if __name__ == "__main__":
    unittest.main()
