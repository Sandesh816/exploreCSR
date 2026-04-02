from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from milestone1_analysis import ParameterizationRecord
from milestone1_ollama import BundleProposalResult, ProposedBundle, RankedCandidate, RankingResult
from run_pipeline import (
    PipelineConfig,
    build_demo_scene,
    run_pipeline,
    save_pipeline_result,
)


class RunPipelineTests(unittest.TestCase):
    def test_build_demo_scene_supports_default_and_three_object(self) -> None:
        default_c1, default_c2 = build_demo_scene("default")
        three_c1, three_c2 = build_demo_scene("three-object")

        self.assertEqual(len(default_c1), 2)
        self.assertEqual(len(default_c2), 2)
        self.assertEqual(len(three_c1), 3)
        self.assertEqual(len(three_c2), 3)

    @patch("run_pipeline.rank_verified_bundles_with_ollama")
    @patch("run_pipeline.propose_relation_bundles_with_ollama")
    def test_run_pipeline_smoke(self, mock_propose, mock_rank) -> None:
        mock_propose.return_value = BundleProposalResult(
            bundles=[
                ProposedBundle(
                    candidate_id="LLM-001",
                    relation_ids=(14,),
                    rationale="Keep the horizontal offset.",
                )
            ],
            raw_prompt="proposal prompt",
            raw_content="{}",
            raw_response_json={},
        )
        mock_rank.return_value = RankingResult(
            summary="Candidate 1 best matches the edit.",
            chosen_candidate_id=1,
            ranked_candidates=[
                RankedCandidate(
                    candidate_id=1,
                    bundle_rank=1,
                    score=90,
                    keep=True,
                    rationale="Best semantic fit.",
                )
            ],
            raw_prompt="ranking prompt",
            raw_content="{}",
            raw_response_json={},
        )

        result = run_pipeline(
            PipelineConfig(
                model_name="mock-model",
                proposal_rounds=1,
                bundles_per_round=4,
            )
        )

        self.assertGreater(result.eq_pool_size, 0)
        self.assertEqual(len(result.proposal_results), 1)
        self.assertEqual(len(result.all_proposed_bundles), 1)
        self.assertTrue(result.candidate_bundle_indices)
        self.assertTrue(result.verified_records)
        self.assertEqual(result.rejected_bundles, [])
        self.assertEqual(result.ranking_result.chosen_candidate_id, 1)
        self.assertIn(1, result.candidate_lookup)
        self.assertTrue(
            all(
                isinstance(option, ParameterizationRecord)
                for record in result.verified_records
                for option in record.viable_fixed_sets
            )
        )

    @patch("run_pipeline.rank_verified_bundles_with_ollama")
    @patch("run_pipeline.propose_relation_bundles_with_ollama")
    def test_save_pipeline_result_writes_llm_artifacts(self, mock_propose, mock_rank) -> None:
        mock_propose.return_value = BundleProposalResult(
            bundles=[
                ProposedBundle(
                    candidate_id="LLM-001",
                    relation_ids=(14,),
                    rationale="Keep the horizontal offset.",
                )
            ],
            raw_prompt="proposal prompt",
            raw_content='{"bundles": []}',
            raw_response_json={"message": {"content": '{"bundles": []}'}},
        )
        mock_rank.return_value = RankingResult(
            summary="Candidate 1 best matches the edit.",
            chosen_candidate_id=1,
            ranked_candidates=[
                RankedCandidate(
                    candidate_id=1,
                    bundle_rank=1,
                    score=90,
                    keep=True,
                    rationale="Best semantic fit.",
                )
            ],
            raw_prompt="ranking prompt",
            raw_content='{"summary": "ok"}',
            raw_response_json={"message": {"content": '{"summary": "ok"}'}},
        )

        result = run_pipeline(
            PipelineConfig(
                model_name="mock-model",
                proposal_rounds=1,
                bundles_per_round=4,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pipeline_result.json"
            saved_path = save_pipeline_result(result, str(output_path))
            payload = json.loads(saved_path.read_text(encoding="utf-8"))

        self.assertEqual(saved_path, output_path)
        self.assertEqual(payload["all_proposed_bundles"][0]["candidate_id"], "LLM-001")
        self.assertEqual(payload["rejected_bundles"], [])
        self.assertEqual(payload["proposal_results"][0]["raw_prompt"], "proposal prompt")
        self.assertEqual(payload["ranking_result"]["raw_prompt"], "ranking prompt")
        self.assertIn("raw_content", payload["ranking_result"])
        self.assertTrue(payload["verified_records"])


if __name__ == "__main__":
    unittest.main()
