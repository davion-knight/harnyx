from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

import pytest

from harnyx_commons.domain.tool_usage_accounting import merge_tool_usage_summaries
from harnyx_commons.domain_tweak_generation import (
    DomainTweakAdkPhaseResult,
    DomainTweakAdkRunConfig,
    DomainTweakAdkRunner,
    DomainTweakAdkTurn,
    DomainTweakGenerationPipeline,
    DomainTweakReviewedQuestion,
)
from harnyx_commons.domain_tweak_generation.types import DomainTweakAdkEventSummary
from harnyx_commons.domain_tweak_generation.validation import (
    validate_form_review_output,
    validate_question_generation_output,
)
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.miner_task_generation import DomainTweakFormReview, DomainTweakPairInput

pytestmark = pytest.mark.anyio("asyncio")
_TASK_ID = UUID("00000000-0000-0000-0000-000000000123")


async def test_pipeline_generates_reviewed_question_without_a2() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            "not json",
            json.dumps(
                {
                    "question": "Which players meet all constraints?",
                    "short_answer": "Ada Example; Ben Example",
                    "solution_plan": "- Find candidate pool\n- Intersect constraints",
                }
            ),
            json.dumps(
                {
                    "form_match": True,
                    "false_premise_status": "none",
                    "reviewer_feedback": "Form preserved.",
                    "retry_recommended": False,
                }
            ),
        )
    )
    pipeline = DomainTweakGenerationPipeline(
        config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview", max_retries=1),
        runner=DomainTweakAdkRunner(turn_executor=executor),
    )

    result = await pipeline.generate_reviewed_question(_pair_input())

    assert result.question_generation.terminal_status == "validated"
    assert len(result.question_generation.attempts) == 2
    assert "failed deterministic validation" in executor.prompts[1]
    assert result.form_review is not None
    assert result.form_review.terminal_status == "validated"
    assert executor.phases == ["question_generation", "question_generation", "form_review"]
    assert result.tool_usage.llm.call_count == 3
    assert result.tool_usage.search_tool.call_count == 3


async def test_pipeline_finalizes_reviewed_question_with_a2() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            json.dumps(
                {
                    "question": "Which players meet all constraints?",
                    "premise_assessment": "The premise is supported by the cited table.",
                    "reference_answer": {
                        "text": "Ada Example and Ben Example meet all constraints.",
                        "citations": [
                            {
                                "url": " https://example.com/table ",
                                "title": " Official table ",
                                "note": " Lists both qualifying players. ",
                            }
                        ],
                    },
                }
            ),
        )
    )
    pipeline = DomainTweakGenerationPipeline(
        config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview"),
        runner=DomainTweakAdkRunner(turn_executor=executor),
    )

    finalization = await pipeline.finalize_task(
        _reviewed_question(),
        task_id_factory=lambda: _TASK_ID,
    )

    assert finalization.task.task_id == _TASK_ID
    assert finalization.task.query.text == "Which players meet all constraints?"
    assert finalization.task.reference_answer.text == "Ada Example and Ben Example meet all constraints."
    assert finalization.task.reference_answer.citations is not None
    assert finalization.task.reference_answer.citations[0].url == "https://example.com/table"
    assert finalization.task.reference_answer.citations[0].title == "Official table"
    assert finalization.task.reference_answer.citations[0].note == "Lists both qualifying players."
    assert finalization.tool_usage.llm.call_count == 1
    assert "Current timestamp: 2026-06-23T00:00:00+00:00" in executor.prompts[0]
    assert executor.phases == ["reference_answer"]


async def test_pipeline_retries_a2_when_citation_packet_fails_validation() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            json.dumps(
                {
                    "question": "Which players meet all constraints?",
                    "premise_assessment": "The premise is supported by the cited table.",
                    "reference_answer": {
                        "text": "Ada Example and Ben Example meet all constraints.",
                        "citations": [
                            {
                                "url": "https://example.com/table",
                                "title": "",
                                "note": "Lists both qualifying players.",
                            }
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "question": "Which players meet all constraints?",
                    "premise_assessment": "The premise is supported by the cited table.",
                    "reference_answer": {
                        "text": "Ada Example and Ben Example meet all constraints.",
                        "citations": [
                            {
                                "url": "https://example.com/table",
                                "title": "Official table",
                                "note": "Lists both qualifying players.",
                            }
                        ],
                    },
                }
            ),
        )
    )
    pipeline = DomainTweakGenerationPipeline(
        config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview", max_retries=1),
        runner=DomainTweakAdkRunner(turn_executor=executor),
    )

    finalization = await pipeline.finalize_task(
        _reviewed_question(),
        task_id_factory=lambda: _TASK_ID,
    )

    assert finalization.task.reference_answer.citations is not None
    assert finalization.task.reference_answer.citations[0].title == "Official table"
    assert "citation" in executor.prompts[1]
    assert "title" in executor.prompts[1]
    assert executor.phases == ["reference_answer", "reference_answer"]


async def test_pipeline_retries_a2_when_returned_question_does_not_match() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            json.dumps(
                {
                    "question": "Which managers meet all constraints?",
                    "premise_assessment": "The premise is supported by the cited table.",
                    "reference_answer": {
                        "text": "Cara Example meets all constraints.",
                        "citations": [
                            {
                                "url": "https://example.com/table",
                                "title": "Official table",
                                "note": "Lists Cara Example with all required constraints.",
                            }
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "question": "Which players meet all constraints?",
                    "premise_assessment": "The premise is supported by the cited table.",
                    "reference_answer": {
                        "text": "Ada Example and Ben Example meet all constraints.",
                        "citations": [
                            {
                                "url": "https://example.com/table",
                                "title": "Official table",
                                "note": "Lists both qualifying players.",
                            }
                        ],
                    },
                }
            ),
        )
    )
    pipeline = DomainTweakGenerationPipeline(
        config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview", max_retries=1),
        runner=DomainTweakAdkRunner(turn_executor=executor),
    )

    finalization = await pipeline.finalize_task(
        _reviewed_question(),
        task_id_factory=lambda: _TASK_ID,
    )

    assert finalization.task.query.text == "Which players meet all constraints?"
    assert finalization.task.reference_answer.text == "Ada Example and Ben Example meet all constraints."
    assert "reference answer question must exactly match" in executor.prompts[1]
    assert executor.phases == ["reference_answer", "reference_answer"]


async def test_pipeline_stops_when_question_generation_returns_no_generate() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            json.dumps(
                {
                    "no_generate": True,
                    "reason": "No grounded domain evidence supports the original form.",
                    "retry_recommended": False,
                }
            ),
        )
    )
    pipeline = DomainTweakGenerationPipeline(
        config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview"),
        runner=DomainTweakAdkRunner(turn_executor=executor),
    )

    result = await pipeline.generate_reviewed_question(_pair_input())

    assert result.question_generation.terminal_status == "no_generate"
    assert result.form_review is None
    assert len(executor.prompts) == 1


async def test_pipeline_repairs_form_review_retry_recommendations_with_same_pair() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            json.dumps(
                {
                    "question": "Which player is the direct lookup answer?",
                    "short_answer": "Ada Example",
                    "solution_plan": "- Look up one page",
                }
            ),
            json.dumps(
                {
                    "form_match": False,
                    "false_premise_status": "none",
                    "reviewer_feedback": "The generated question lost the aggregation step.",
                    "retry_recommended": True,
                }
            ),
            json.dumps(
                {
                    "question": "Which players meet all constraints?",
                    "short_answer": "Ada Example; Ben Example",
                    "solution_plan": "- Find candidate pool\n- Intersect constraints",
                }
            ),
            json.dumps(
                {
                    "form_match": True,
                    "false_premise_status": "none",
                    "reviewer_feedback": "Form preserved.",
                    "retry_recommended": False,
                }
            ),
        )
    )
    pipeline = DomainTweakGenerationPipeline(
        config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview", max_retries=0),
        runner=DomainTweakAdkRunner(turn_executor=executor),
        form_review_retries=1,
    )

    result = await pipeline.generate_reviewed_question(_pair_input())

    assert result.question_generation.terminal_status == "validated"
    assert result.form_review is not None
    assert result.form_review.terminal_status == "validated"
    assert result.question_generation.parsed_output.question == "Which players meet all constraints?"
    assert "The generated question lost the aggregation step." in executor.prompts[2]
    assert executor.phases == [
        "question_generation",
        "form_review",
        "question_generation",
        "form_review",
    ]
    assert result.tool_usage.llm.call_count == 4


async def test_pipeline_rejects_form_review_retry_recommendation_after_repair_budget() -> None:
    executor = _FakeTurnExecutor(
        responses=(
            json.dumps(
                {
                    "question": "Which player is the direct lookup answer?",
                    "short_answer": "Ada Example",
                    "solution_plan": "- Look up one page",
                }
            ),
            json.dumps(
                {
                    "form_match": False,
                    "false_premise_status": "none",
                    "reviewer_feedback": "The generated question lost the aggregation step.",
                    "retry_recommended": True,
                }
            ),
        )
    )
    pipeline = DomainTweakGenerationPipeline(
        config=DomainTweakAdkRunConfig(model="gemini-3-pro-preview", max_retries=0),
        runner=DomainTweakAdkRunner(turn_executor=executor),
        form_review_retries=0,
    )

    result = await pipeline.generate_reviewed_question(_pair_input())

    assert result.form_review is not None
    assert result.form_review.terminal_status == "form_rejected"
    assert executor.phases == ["question_generation", "form_review"]


class _FakeTurnExecutor:
    def __init__(self, *, responses: tuple[str, ...]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.phases: list[str] = []

    async def __call__(
        self,
        *,
        phase: str,
        prompt: str,
        attempt_index: int,
        config: DomainTweakAdkRunConfig,
        agent_instruction: str,
    ) -> DomainTweakAdkTurn:
        self.phases.append(phase)
        self.prompts.append(prompt)
        return DomainTweakAdkTurn(
            final_text=self._responses.pop(0),
            events=(
                DomainTweakAdkEventSummary(
                    is_final_response=True,
                    usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    function_call_names=("google_search_agent",),
                ),
            ),
        )


def _pair_input() -> DomainTweakPairInput:
    return DomainTweakPairInput(
        pair_id="pair-001",
        deepsearchqa_form_target="Which films meet all constraints?",
        deepresearch9k_domain_target="Football records question",
        timestamp=datetime(2026, 6, 23, tzinfo=UTC),
    )


def _reviewed_question() -> DomainTweakReviewedQuestion:
    pair_input = _pair_input()
    question_result = _phase_result(
        "question_generation",
        {
            "question": "Which players meet all constraints?",
            "short_answer": "Ada Example; Ben Example",
            "solution_plan": "- Find candidate pool\n- Intersect constraints",
        },
    )
    form_result = _phase_result(
        "form_review",
        {
            "form_match": True,
            "false_premise_status": "none",
            "reviewer_feedback": "Form preserved.",
            "retry_recommended": False,
        },
    )
    assert question_result.parsed_output is not None
    assert isinstance(form_result.parsed_output, DomainTweakFormReview)
    return DomainTweakReviewedQuestion(
        pair_input=pair_input,
        question_candidate=question_result.parsed_output,
        form_review=form_result.parsed_output,
        question_generation_result=question_result,
        form_review_result=form_result,
        tool_usage=merge_tool_usage_summaries(question_result.tool_usage, form_result.tool_usage),
    )


def _phase_result(phase: str, payload: dict[str, object]) -> DomainTweakAdkPhaseResult:
    validation = (
        validate_question_generation_output(json.dumps(payload))
        if phase == "question_generation"
        else validate_form_review_output(json.dumps(payload))
    )
    return DomainTweakAdkPhaseResult(
        phase=phase,
        terminal_status=validation.terminal_status,
        parsed_output=validation.parsed_output,
        tool_usage=_tool_usage(),
    )


def _tool_usage():
    from harnyx_commons.domain_tweak_generation.adk_events import tool_usage_from_adk_events

    return tool_usage_from_adk_events(
        (
            DomainTweakAdkEventSummary(
                usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                function_call_names=("google_search_agent",),
            ),
        ),
        provider="vertex",
        model="unknown-model",
    )
