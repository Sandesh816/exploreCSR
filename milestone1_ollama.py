from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Optional
import re

import requests

from instructions import build_bundle_ranking_messages, build_bundle_ranking_prompt
from milestone1_analysis import BundleRecord
from milestone1_core import Equation, NamedRect, RelationRecord, build_relation_records


@dataclass(frozen=True)
class OllamaRankerConfig:
    model_name: str
    base_url: str = "http://localhost:11434/api"
    timeout_seconds: int = 300
    temperature: int | float = 0
    proposal_temperature: int | float | None = None
    ranking_temperature: int | float | None = None
    keep_alive: str | None = "5m"


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: int
    bundle_rank: int
    score: int
    keep: bool
    rationale: str


@dataclass(frozen=True)
class RankingResult:
    summary: str
    chosen_candidate_id: int
    ranked_candidates: list[RankedCandidate]
    raw_prompt: str
    raw_content: str
    raw_response_json: dict


@dataclass(frozen=True)
class ProposedBundle:
    candidate_id: str
    relation_ids: tuple[int, ...]
    rationale: str


@dataclass(frozen=True)
class BundleProposalResult:
    bundles: list[ProposedBundle]
    raw_prompt: str
    raw_content: str
    raw_response_json: dict


def _api_url(config: OllamaRankerConfig, path: str) -> str:
    return f"{config.base_url.rstrip('/')}/{path.lstrip('/')}"


def list_ollama_models(config: OllamaRankerConfig) -> list[str]:
    try:
        response = requests.get(
            _api_url(config, "tags"),
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Ollama is not running or unreachable. "
            f"Expected a local API at {config.base_url!r}."
        ) from exc

    payload = response.json()
    models = payload.get("models", [])
    model_names = []
    for model in models:
        name = model.get("name")
        if isinstance(name, str) and name:
            model_names.append(name)
    return sorted(model_names)


def build_ranking_schema(candidate_count: int) -> dict[str, Any]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")

    candidate_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidate_id", "bundle_rank", "score", "keep", "rationale"],
        "properties": {
            "candidate_id": {"type": "integer"},
            "bundle_rank": {"type": "integer"},
            "score": {"type": "integer", "minimum": 0, "maximum": 100},
            "keep": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "chosen_candidate_id", "ranked_candidates"],
        "properties": {
            "summary": {"type": "string"},
            "chosen_candidate_id": {"type": "integer"},
            "ranked_candidates": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": candidate_schema,
            },
        },
    }


def build_bundle_proposal_schema(max_bundles: int) -> dict[str, Any]:
    if max_bundles <= 0:
        raise ValueError("max_bundles must be positive.")

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["bundles"],
        "properties": {
            "bundles": {
                "type": "array",
                "maxItems": max_bundles,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["candidate_id", "relation_ids", "rationale"],
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "relation_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "rationale": {"type": "string"},
                    },
                },
            },
        },
    }


def call_ollama_chat(
    prompt_text: str | list[dict[str, str]],
    schema: dict[str, Any],
    config: OllamaRankerConfig,
    *,
    temperature_override: int | float | None = None,
) -> dict[str, Any]:
    messages = prompt_text
    if isinstance(messages, str):
        messages = [
            {
                "role": "user",
                "content": messages,
            }
        ]

    payload: dict[str, Any] = {
        "model": config.model_name,
        "messages": messages,
        "stream": False,
        "format": schema,
        "options": {
            "temperature": (
                config.temperature
                if temperature_override is None
                else temperature_override
            ),
        },
    }
    if config.keep_alive is not None:
        payload["keep_alive"] = config.keep_alive

    try:
        response = requests.post(
            _api_url(config, "chat"),
            json=payload,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Ollama is not running or unreachable, or the chat request timed out. "
            f"Expected a local API at {config.base_url!r}."
        ) from exc

    return response.json()


def call_ollama_proposal(
    messages: str | list[dict[str, str]],
    schema: dict[str, Any],
    config: OllamaRankerConfig,
) -> dict[str, Any]:
    return call_ollama_chat(
        messages,
        schema,
        config,
        temperature_override=config.proposal_temperature,
    )


def call_ollama_ranking(
    messages: str | list[dict[str, str]],
    schema: dict[str, Any],
    config: OllamaRankerConfig,
) -> dict[str, Any]:
    return call_ollama_chat(
        messages,
        schema,
        config,
        temperature_override=config.ranking_temperature,
    )


def _extract_message_content(raw_response: dict[str, Any]) -> str:
    message = raw_response.get("message")
    if not isinstance(message, dict):
        raise ValueError("Ollama response is missing the 'message' object.")

    content = message.get("content")
    if isinstance(content, str):
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if not content.strip():
            raise ValueError("Ollama response message content is empty.")
        return content
    if isinstance(content, dict):
        return json.dumps(content)
    raise ValueError("Ollama response message content must be a JSON string.")


def parse_ranking_response(
    raw_response: dict[str, Any],
    candidate_ids: set[int],
    prompt_text: str,
) -> RankingResult:
    raw_content = _extract_message_content(raw_response)
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Ollama returned invalid JSON content. "
            f"raw_content={raw_content!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Ollama JSON content must be an object. raw_content={raw_content!r}")

    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(f"Ranking response must include a non-empty summary. raw_content={raw_content!r}")

    chosen_candidate_id = parsed.get("chosen_candidate_id")
    if not isinstance(chosen_candidate_id, int) or chosen_candidate_id not in candidate_ids:
        raise ValueError(
            "chosen_candidate_id must be an integer from the provided candidate_ids. "
            f"raw_content={raw_content!r}"
        )

    ranked_candidates_raw = parsed.get("ranked_candidates")
    if not isinstance(ranked_candidates_raw, list):
        raise ValueError(f"ranked_candidates must be a list. raw_content={raw_content!r}")
    if len(ranked_candidates_raw) != len(candidate_ids):
        raise ValueError(
            "ranked_candidates must contain every prompt candidate exactly once. "
            f"raw_content={raw_content!r}"
        )

    seen_candidate_ids: set[int] = set()
    ranked_candidates_raw_items: list[RankedCandidate] = []

    for item in ranked_candidates_raw:
        if not isinstance(item, dict):
            raise ValueError(f"Each ranked candidate must be an object. raw_content={raw_content!r}")

        candidate_id = item.get("candidate_id")
        bundle_rank = item.get("bundle_rank")
        score = item.get("score")
        keep = item.get("keep")
        rationale = item.get("rationale")

        if not isinstance(candidate_id, int) or candidate_id not in candidate_ids:
            raise ValueError(
                "Each candidate_id must be an integer from the provided candidate_ids. "
                f"raw_content={raw_content!r}"
            )
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"Duplicate candidate_id in response. raw_content={raw_content!r}")

        if not isinstance(bundle_rank, int):
            raise ValueError(f"bundle_rank must be an integer. raw_content={raw_content!r}")

        if not isinstance(score, int) or not (0 <= score <= 100):
            raise ValueError(f"score must be an integer between 0 and 100. raw_content={raw_content!r}")
        if not isinstance(keep, bool):
            raise ValueError(f"keep must be a boolean. raw_content={raw_content!r}")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"rationale must be a non-empty string. raw_content={raw_content!r}")

        seen_candidate_ids.add(candidate_id)
        ranked_candidates_raw_items.append(
            RankedCandidate(
                candidate_id=candidate_id,
                bundle_rank=bundle_rank,
                score=score,
                keep=keep,
                rationale=rationale,
            )
        )

    if seen_candidate_ids != candidate_ids:
        raise ValueError(
            "Response must include every prompt candidate exactly once. "
            f"raw_content={raw_content!r}"
        )

    ranked_candidates_raw_items.sort(key=lambda item: (item.bundle_rank, item.candidate_id))
    ranked_candidates = [
        RankedCandidate(
            candidate_id=item.candidate_id,
            bundle_rank=normalized_rank,
            score=item.score,
            keep=item.keep,
            rationale=item.rationale,
        )
        for normalized_rank, item in enumerate(ranked_candidates_raw_items, start=1)
    ]
    return RankingResult(
        summary=summary.strip(),
        chosen_candidate_id=chosen_candidate_id,
        ranked_candidates=ranked_candidates,
        raw_prompt=prompt_text,
        raw_content=raw_content,
        raw_response_json=raw_response,
    )


def parse_bundle_proposal_response(
    raw_response: dict[str, Any],
    valid_relation_ids: set[int],
    max_bundle_size: Optional[int],
    prompt_text: str,
) -> BundleProposalResult:
    if max_bundle_size is not None and max_bundle_size <= 0:
        raise ValueError("max_bundle_size must be positive.")

    raw_content = _extract_message_content(raw_response)
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Ollama returned invalid JSON content. "
            f"raw_content={raw_content!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Ollama JSON content must be an object. raw_content={raw_content!r}")

    bundles_raw = parsed.get("bundles")
    if not isinstance(bundles_raw, list):
        raise ValueError(f"bundles must be a list. raw_content={raw_content!r}")

    bundles: list[ProposedBundle] = []
    seen_relation_sets: set[tuple[int, ...]] = set()
    for item in bundles_raw:
        if not isinstance(item, dict):
            raise ValueError(f"Each bundle must be an object. raw_content={raw_content!r}")

        candidate_id = item.get("candidate_id")
        relation_ids = item.get("relation_ids")
        rationale = item.get("rationale")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"candidate_id must be a non-empty string. raw_content={raw_content!r}")
        if not isinstance(relation_ids, list) or not all(isinstance(value, int) for value in relation_ids):
            raise ValueError(f"relation_ids must be a list of integers. raw_content={raw_content!r}")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"rationale must be a non-empty string. raw_content={raw_content!r}")

        normalized_ids = tuple(sorted(set(relation_ids)))
        if not normalized_ids:
            continue
        if max_bundle_size is not None and len(normalized_ids) > max_bundle_size:
            continue
        if any(relation_id not in valid_relation_ids for relation_id in normalized_ids):
            continue
        if normalized_ids in seen_relation_sets:
            continue

        seen_relation_sets.add(normalized_ids)
        bundles.append(
            ProposedBundle(
                candidate_id=candidate_id.strip(),
                relation_ids=normalized_ids,
                rationale=rationale.strip(),
            )
        )

    return BundleProposalResult(
        bundles=bundles,
        raw_prompt=prompt_text,
        raw_content=raw_content,
        raw_response_json=raw_response,
    )


def rank_verified_bundles_with_ollama(
    records: list[BundleRecord],
    user_history: Optional[str],
    config: OllamaRankerConfig,
    top_k: int = 5,
    max_ties_per_bundle: int = 3,
    *,
    c1: Optional[list[NamedRect]] = None,
    c2: Optional[list[NamedRect]] = None,
    prompt_text: Optional[str] = None,
    candidate_lookup: Optional[dict[int, BundleRecord]] = None,
) -> RankingResult:
    selected_records = records[:top_k]
    if not selected_records:
        raise ValueError("No verified bundles were provided for Ollama ranking.")

    schema = build_ranking_schema(len(selected_records))
    prompt_messages: list[dict[str, str]]

    if prompt_text is None or candidate_lookup is None:
        if c1 is None or c2 is None:
            raise ValueError(
                "Provide c1 and c2 when prompt_text and candidate_lookup are not supplied."
            )
        prompt_text, candidate_lookup = build_bundle_ranking_prompt(
            c1,
            c2,
            selected_records,
            top_k=top_k,
            user_history=user_history,
            max_ties_per_bundle=max_ties_per_bundle,
            schema=schema,
        )
        prompt_messages = build_bundle_ranking_messages(
            c1,
            c2,
            selected_records,
            top_k=top_k,
            user_history=user_history,
            max_ties_per_bundle=max_ties_per_bundle,
            schema=schema,
        )
    else:
        prompt_messages = [{"role": "user", "content": prompt_text}]

    raw_response = call_ollama_ranking(prompt_messages, schema, config)
    return parse_ranking_response(raw_response, set(candidate_lookup.keys()), prompt_text)


def propose_relation_bundles_with_ollama(
    c1: list[NamedRect],
    c2: list[NamedRect],
    relation_pool: list[RelationRecord],
    config: OllamaRankerConfig,
    *,
    max_bundles: int = 16,
    max_bundle_size: Optional[int] = None,
    previous_bundles: Optional[list[tuple[int, ...]]] = None,
) -> BundleProposalResult:
    from instructions import build_bundle_proposal_messages

    messages = build_bundle_proposal_messages(
        c1,
        c2,
        relation_pool,
        max_bundles=max_bundles,
        max_bundle_size=max_bundle_size,
        previous_bundles=previous_bundles,
    )
    prompt_text = messages[-1]["content"]
    schema = build_bundle_proposal_schema(max_bundles)
    raw_response = call_ollama_proposal(messages, schema, config)
    valid_relation_ids = {relation.relation_id for relation in relation_pool}
    return parse_bundle_proposal_response(
        raw_response,
        valid_relation_ids=valid_relation_ids,
        max_bundle_size=max_bundle_size,
        prompt_text=prompt_text,
    )


def propose_and_collect_candidate_bundles_with_ollama(
    c1: list[NamedRect],
    c2: list[NamedRect],
    eq_pool: list[Equation],
    config: OllamaRankerConfig,
    *,
    max_bundles: int = 16,
    max_bundle_size: Optional[int] = None,
    max_system_size: int = 3,
    max_candidates: int = 128,
    previous_bundles: Optional[list[tuple[int, ...]]] = None,
) -> tuple[BundleProposalResult, list[tuple[int, ...]]]:
    from milestone1_analysis import collect_candidate_bundle_indices

    relation_pool = build_relation_records(eq_pool)
    proposal_result = propose_relation_bundles_with_ollama(
        c1,
        c2,
        relation_pool,
        config,
        max_bundles=max_bundles,
        max_bundle_size=max_bundle_size,
        previous_bundles=previous_bundles,
    )
    candidate_bundle_indices = collect_candidate_bundle_indices(
        c1,
        c2,
        eq_pool,
        proposed_bundles=[bundle.relation_ids for bundle in proposal_result.bundles],
        max_system_size=max_system_size,
        max_bundle_size=max_bundle_size,
        max_candidates=max_candidates,
    )
    return proposal_result, candidate_bundle_indices
