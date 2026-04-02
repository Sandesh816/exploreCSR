from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from milestone1_analysis import (
    BundleRecord,
    ParameterizationRecord,
    build_llm_ranking_prompt,
)
from milestone1_core import NamedRect, Rect
from milestone1_ollama import (
    OllamaRankerConfig,
    build_bundle_proposal_schema,
    build_ranking_schema,
    call_ollama_chat,
    call_ollama_proposal,
    call_ollama_ranking,
    list_ollama_models,
    parse_bundle_proposal_response,
    parse_ranking_response,
    propose_and_collect_candidate_bundles_with_ollama,
    rank_verified_bundles_with_ollama,
)


def make_parameterization(tag: int) -> ParameterizationRecord:
    return ParameterizationRecord(
        fixed_vars=("cutter.x", f"pizza.x.{tag}"),
        extra_fixed_vars=(f"pizza.x.{tag}",),
        driven_vars=(f"pizza.h.{tag}",),
        predicted_changes=(("pizza.x", -2, -3 - tag),),
        c3_census_count=tag + 1,
        program_text=f"program {tag}",
    )


def make_bundle(eq_indices: tuple[int, ...], tie_count: int = 1) -> BundleRecord:
    return BundleRecord(
        eq_indices=eq_indices,
        equations=tuple(f"eq {idx}" for idx in eq_indices),
        support_vars=("cutter.x", "pizza.x"),
        delta_hit=True,
        changed_vars_hit=("cutter.x",),
        has_shared_variable=len(eq_indices) > 1,
        has_connected_support=True,
        verification_passed=True,
        viable_fixed_sets=tuple(make_parameterization(i) for i in range(tie_count)),
        min_extra_fixed=0,
        has_parameterization_conflict=False,
        unique_c3_count=1,
        failure_reason=None,
    )


class PromptBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.c1 = [
            NamedRect("pizza", Rect(-2, 3, 3, 4)),
            NamedRect("cutter", Rect(-4, 3, 1, 6)),
        ]
        self.c2 = [
            NamedRect("pizza", Rect(-2, 3, 3, 4)),
            NamedRect("cutter", Rect(-5, 3, 1, 6)),
        ]

    def test_prompt_contains_schema_and_candidate_ids(self) -> None:
        records = [make_bundle((7,), tie_count=4), make_bundle((14,), tie_count=1)]
        schema = build_ranking_schema(2)
        prompt_text, candidate_lookup = build_llm_ranking_prompt(
            self.c1,
            self.c2,
            records,
            top_k=2,
            user_history="The user moved cutter.x left.",
            max_ties_per_bundle=3,
            schema=schema,
        )

        self.assertIn("Canvas C1:", prompt_text)
        self.assertIn("Changed parameters (delta):", prompt_text)
        self.assertIn("cutter.x: -4 -> -5", prompt_text)
        self.assertIn("User history summary: The user moved cutter.x left.", prompt_text)
        self.assertIn("- candidate_id: 1", prompt_text)
        self.assertIn("Allowed candidate_ids: (1, 2)", prompt_text)
        self.assertIn("Output JSON schema:", prompt_text)
        self.assertIn('"chosen_candidate_id"', prompt_text)
        self.assertIn("c3_census_count=1", prompt_text)
        self.assertIn("- omitted_tied_fixed_set_count: 1", prompt_text)
        self.assertEqual(set(candidate_lookup.keys()), {1, 2})


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt_text = "prompt"
        self.candidate_ids = {1, 2}

    def valid_response(self) -> dict:
        content = {
            "summary": "Candidate 1 best matches the recent edit.",
            "chosen_candidate_id": 1,
            "ranked_candidates": [
                {
                    "candidate_id": 1,
                    "bundle_rank": 1,
                    "score": 91,
                    "keep": True,
                    "rationale": "Directly explains the moved cutter.x relation.",
                },
                {
                    "candidate_id": 2,
                    "bundle_rank": 2,
                    "score": 63,
                    "keep": False,
                    "rationale": "Plausible but less aligned with the recent edit.",
                },
            ],
        }
        return {"message": {"content": json.dumps(content)}}

    def test_parse_valid_ranking_response(self) -> None:
        result = parse_ranking_response(self.valid_response(), self.candidate_ids, self.prompt_text)
        self.assertEqual(result.summary, "Candidate 1 best matches the recent edit.")
        self.assertEqual(result.chosen_candidate_id, 1)
        self.assertEqual([item.bundle_rank for item in result.ranked_candidates], [1, 2])

    def test_parse_unknown_candidate_id_fails(self) -> None:
        response = self.valid_response()
        payload = json.loads(response["message"]["content"])
        payload["ranked_candidates"][0]["candidate_id"] = 99
        response["message"]["content"] = json.dumps(payload)

        with self.assertRaises(ValueError):
            parse_ranking_response(response, self.candidate_ids, self.prompt_text)

    def test_parse_duplicate_candidate_id_fails(self) -> None:
        response = self.valid_response()
        payload = json.loads(response["message"]["content"])
        payload["ranked_candidates"][1]["candidate_id"] = 1
        response["message"]["content"] = json.dumps(payload)

        with self.assertRaises(ValueError):
            parse_ranking_response(response, self.candidate_ids, self.prompt_text)

    def test_parse_missing_candidate_fails(self) -> None:
        response = self.valid_response()
        payload = json.loads(response["message"]["content"])
        payload["ranked_candidates"] = payload["ranked_candidates"][:1]
        response["message"]["content"] = json.dumps(payload)

        with self.assertRaises(ValueError):
            parse_ranking_response(response, self.candidate_ids, self.prompt_text)

    def test_parse_non_contiguous_bundle_rank_is_normalized(self) -> None:
        response = self.valid_response()
        payload = json.loads(response["message"]["content"])
        payload["ranked_candidates"][0]["bundle_rank"] = 7
        payload["ranked_candidates"][1]["bundle_rank"] = 3
        response["message"]["content"] = json.dumps(payload)

        result = parse_ranking_response(response, self.candidate_ids, self.prompt_text)
        self.assertEqual([item.candidate_id for item in result.ranked_candidates], [2, 1])
        self.assertEqual([item.bundle_rank for item in result.ranked_candidates], [1, 2])

    def test_parse_invalid_chosen_candidate_fails(self) -> None:
        response = self.valid_response()
        payload = json.loads(response["message"]["content"])
        payload["chosen_candidate_id"] = 99
        response["message"]["content"] = json.dumps(payload)

        with self.assertRaises(ValueError):
            parse_ranking_response(response, self.candidate_ids, self.prompt_text)

    def test_parse_bundle_proposal_response_normalizes_and_drops_invalid(self) -> None:
        response = {
            "message": {
                "content": json.dumps(
                    {
                        "bundles": [
                            {
                                "candidate_id": "LLM-001",
                                "relation_ids": [14, 3, 14],
                                "rationale": "Good bundle.",
                            },
                            {
                                "candidate_id": "LLM-002",
                                "relation_ids": [999],
                                "rationale": "Invalid relation id.",
                            },
                            {
                                "candidate_id": "LLM-003",
                                "relation_ids": [1, 2, 3, 4],
                                "rationale": "Too large.",
                            },
                            {
                                "candidate_id": "LLM-004",
                                "relation_ids": [3, 14],
                                "rationale": "Duplicate normalized bundle.",
                            },
                        ]
                    }
                )
            }
        }

        result = parse_bundle_proposal_response(
            response,
            valid_relation_ids=set(range(30)),
            max_bundle_size=3,
            prompt_text=self.prompt_text,
        )

        self.assertEqual(len(result.bundles), 1)
        self.assertEqual(result.bundles[0].relation_ids, (3, 14))

    def test_parse_bundle_proposal_response_allows_larger_bundles_when_unbounded(self) -> None:
        response = {
            "message": {
                "content": json.dumps(
                    {
                        "bundles": [
                            {
                                "candidate_id": "LLM-001",
                                "relation_ids": [1, 2, 3, 4],
                                "rationale": "Larger bundle remains allowed.",
                            }
                        ]
                    }
                )
            }
        }

        result = parse_bundle_proposal_response(
            response,
            valid_relation_ids=set(range(30)),
            max_bundle_size=None,
            prompt_text=self.prompt_text,
        )

        self.assertEqual(len(result.bundles), 1)
        self.assertEqual(result.bundles[0].relation_ids, (1, 2, 3, 4))


class HttpClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = OllamaRankerConfig(model_name="llama3.1")
        self.c1 = [
            NamedRect("pizza", Rect(-2, 3, 3, 4)),
            NamedRect("cutter", Rect(-4, 3, 1, 6)),
        ]
        self.c2 = [
            NamedRect("pizza", Rect(-2, 3, 3, 4)),
            NamedRect("cutter", Rect(-5, 3, 1, 6)),
        ]

    @patch("milestone1_ollama.requests.get")
    def test_list_ollama_models_reads_tags(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {
            "models": [
                {"name": "llama3.1"},
                {"name": "qwen2.5"},
            ]
        }
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        model_names = list_ollama_models(self.config)

        self.assertEqual(model_names, ["llama3.1", "qwen2.5"])
        mock_get.assert_called_once()
        self.assertEqual(mock_get.call_args.kwargs["timeout"], self.config.timeout_seconds)

    @patch("milestone1_ollama.requests.post")
    def test_call_ollama_chat_sends_schema_and_temperature(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"message": {"content": "{\"summary\": \"ok\"}"}}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        schema = build_ranking_schema(2)
        payload = call_ollama_chat("prompt text", schema, self.config)

        self.assertEqual(payload, {"message": {"content": "{\"summary\": \"ok\"}"}})
        mock_post.assert_called_once()
        request_json = mock_post.call_args.kwargs["json"]
        self.assertEqual(request_json["stream"], False)
        self.assertEqual(request_json["format"], schema)
        self.assertEqual(request_json["options"]["temperature"], 0)
        self.assertEqual(request_json["keep_alive"], "5m")

    @patch("milestone1_ollama.requests.post")
    def test_call_ollama_chat_accepts_message_list(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"message": {"content": "{\"bundles\": []}"}}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        schema = build_bundle_proposal_schema(4)
        payload = call_ollama_chat(
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
            schema,
            self.config,
        )

        self.assertEqual(payload, {"message": {"content": "{\"bundles\": []}"}})
        request_json = mock_post.call_args.kwargs["json"]
        self.assertEqual(len(request_json["messages"]), 2)
        self.assertEqual(request_json["messages"][0]["role"], "system")

    @patch("milestone1_ollama.requests.post")
    def test_call_ollama_proposal_uses_proposal_temperature_override(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"message": {"content": "{\"bundles\": []}"}}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        config = OllamaRankerConfig(
            model_name="llama3.1",
            temperature=0,
            proposal_temperature=0.7,
        )
        schema = build_bundle_proposal_schema(4)
        call_ollama_proposal("proposal prompt", schema, config)

        request_json = mock_post.call_args.kwargs["json"]
        self.assertEqual(request_json["options"]["temperature"], 0.7)

    @patch("milestone1_ollama.requests.post")
    def test_call_ollama_ranking_uses_ranking_temperature_override(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"message": {"content": "{\"summary\": \"ok\"}"}}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        config = OllamaRankerConfig(
            model_name="llama3.1",
            temperature=0.5,
            ranking_temperature=0,
        )
        schema = build_ranking_schema(2)
        call_ollama_ranking("ranking prompt", schema, config)

        request_json = mock_post.call_args.kwargs["json"]
        self.assertEqual(request_json["options"]["temperature"], 0)

    def test_extract_message_content_strips_think_tags(self) -> None:
        parsed = parse_bundle_proposal_response(
            {
                "message": {
                    "content": "<think>hidden</think>{\"bundles\": [{\"candidate_id\": \"LLM-001\", \"relation_ids\": [1], \"rationale\": \"ok\"}]}"
                }
            },
            valid_relation_ids={1},
            max_bundle_size=2,
            prompt_text="prompt",
        )

        self.assertEqual(parsed.bundles[0].relation_ids, (1,))

    @patch("milestone1_ollama.requests.post")
    def test_rank_verified_bundles_with_ollama_uses_message_list(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = self._ranking_response_json()
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        result = rank_verified_bundles_with_ollama(
            [make_bundle((14,), tie_count=1), make_bundle((7,), tie_count=1)],
            user_history="Designer is aligning objects.",
            config=self.config,
            c1=self.c1,
            c2=self.c2,
            top_k=2,
        )

        self.assertEqual(result.chosen_candidate_id, 1)
        request_json = mock_post.call_args.kwargs["json"]
        self.assertEqual(len(request_json["messages"]), 2)
        self.assertEqual(request_json["messages"][0]["role"], "system")
        self.assertIn("Canvas C1:", request_json["messages"][1]["content"])
        self.assertIn("c3_census_count=", request_json["messages"][1]["content"])
        self.assertIn("Designer is aligning objects.", request_json["messages"][1]["content"])

    @patch("milestone1_ollama.requests.post")
    def test_propose_and_collect_candidate_bundles_with_ollama(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {
                        "bundles": [
                            {
                                "candidate_id": "LLM-001",
                                "relation_ids": [14, 0],
                                "rationale": "Keep left-of relation and height pin.",
                            },
                            {
                                "candidate_id": "LLM-002",
                                "relation_ids": [12],
                                "rationale": "Invalid after the edit.",
                            },
                        ]
                    }
                )
            }
        }
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        from milestone1_core import detect_c1_equations

        eq_pool = detect_c1_equations(self.c1, linear_only=True, print_list=False)
        proposal_result, candidate_bundle_indices = propose_and_collect_candidate_bundles_with_ollama(
            self.c1,
            self.c2,
            eq_pool,
            self.config,
            max_bundles=8,
            max_bundle_size=2,
            max_system_size=1,
            max_candidates=20,
        )

        self.assertEqual(proposal_result.bundles[0].relation_ids, (0, 14))
        self.assertIn((0, 14), candidate_bundle_indices)
        self.assertNotIn((12,), candidate_bundle_indices)

    def _ranking_response_json(self) -> dict:
        return {
            "message": {
                "content": json.dumps(
                    {
                        "summary": "Candidate 1 best matches the recent edit.",
                        "chosen_candidate_id": 1,
                        "ranked_candidates": [
                            {
                                "candidate_id": 1,
                                "bundle_rank": 1,
                                "score": 91,
                                "keep": True,
                                "rationale": "Directly explains the moved cutter.x relation.",
                            },
                            {
                                "candidate_id": 2,
                                "bundle_rank": 2,
                                "score": 63,
                                "keep": False,
                                "rationale": "Plausible but less aligned with the recent edit.",
                            },
                        ],
                    }
                )
            }
        }


if __name__ == "__main__":
    unittest.main()
