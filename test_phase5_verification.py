from __future__ import annotations

import unittest

from milestone1_analysis import verify_and_materialize_candidate_bundles
from milestone1_core import NamedRect, Rect, detect_c1_equations


class VerificationAndCensusTests(unittest.TestCase):
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

    def test_verify_and_materialize_candidate_bundles_builds_census(self) -> None:
        candidate_bundle_indices = [
            (14,),
            (0, 14),
            (12,),
            (9, 22),
        ]

        records, census, rejected_records = verify_and_materialize_candidate_bundles(
            self.c1,
            self.c2,
            self.eq_pool,
            candidate_bundle_indices,
        )

        record_lookup = {record.eq_indices: record for record in records}

        self.assertNotIn((12,), record_lookup)
        self.assertIn((14,), record_lookup)
        self.assertIn((0, 14), record_lookup)
        self.assertIn((9, 22), record_lookup)
        self.assertIn((12,), {record.eq_indices for record in rejected_records})

        for record in records:
            self.assertTrue(record.verification_passed)
            self.assertTrue(record.viable_fixed_sets)
            self.assertGreaterEqual(record.unique_c3_count, 1)
            for option in record.viable_fixed_sets:
                self.assertIsNotNone(option.predicted_changes)
                self.assertIsNotNone(option.c3_key)
                self.assertIsNotNone(option.c3_census_count)

        self.assertEqual(record_lookup[(14,)].viable_fixed_sets[0].c3_census_count, 2)
        self.assertEqual(record_lookup[(0, 14)].viable_fixed_sets[0].c3_census_count, 2)

        unstable_record = record_lookup[(9, 22)]
        self.assertTrue(unstable_record.has_parameterization_conflict)
        self.assertEqual(unstable_record.unique_c3_count, 2)
        self.assertEqual(sorted(option.c3_census_count for option in unstable_record.viable_fixed_sets), [1, 2, 2])

        shared_c3_bucket_sizes = sorted(len(provenance) for provenance in census.values())
        self.assertIn(2, shared_c3_bucket_sizes)


if __name__ == "__main__":
    unittest.main()
