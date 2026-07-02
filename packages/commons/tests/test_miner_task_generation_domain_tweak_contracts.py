from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from harnyx_commons.domain.miner_task import AnswerCitation, ReferenceAnswer
from harnyx_commons.miner_task_generation import (
    DomainTweakFormReview,
    DomainTweakPairInput,
    DomainTweakQuestionCandidate,
    DomainTweakReferenceAnswerCandidate,
)


def test_pair_input_records_one_form_and_domain_pair() -> None:
    pair = DomainTweakPairInput(
        pair_id="pair-001",
        deepsearchqa_form_target="Which films meet all of these constraints?",
        deepresearch9k_domain_target="Question about football records",
        timestamp=datetime(2026, 6, 21, tzinfo=UTC),
    )

    assert pair.pair_id == "pair-001"
    assert pair.deepsearchqa_form_target.startswith("Which films")


def test_question_candidate_has_short_answer_and_markdown_solution_plan_but_no_reference_answer() -> None:
    candidate = DomainTweakQuestionCandidate(
        question="Which players meet all of these constraints?",
        short_answer="Ada Example; Ben Example",
        solution_plan="- Find source rows\n  Continue the same step\n- Intersect constraints",
    )

    assert candidate.solution_plan.startswith("- Find")
    with pytest.raises(ValidationError, match="solution_plan"):
        DomainTweakQuestionCandidate(
            question="Which players meet all of these constraints?",
            short_answer="Ada Example; Ben Example",
            solution_plan="Find source rows, then intersect constraints.",
        )
    with pytest.raises(ValidationError, match="reference_answer"):
        DomainTweakQuestionCandidate(
            question="Which players meet all of these constraints?",
            short_answer="Ada Example; Ben Example",
            solution_plan="- Find source rows",
            reference_answer={"text": "Ada Example and Ben Example."},
        )


def test_form_review_contract_carries_retry_guidance() -> None:
    review = DomainTweakFormReview(
        form_match=False,
        false_premise_status="none",
        reviewer_feedback="The generated question lost the aggregation operation.",
        retry_recommended=True,
    )

    assert review.retry_recommended is True


def test_reference_answer_candidate_requires_premise_assessment_and_reference_answer() -> None:
    candidate = DomainTweakReferenceAnswerCandidate(
        question="Which players meet all of these constraints?",
        premise_assessment="The premise is supported by the cited official table.",
        reference_answer=ReferenceAnswer(
            text="Ada Example and Ben Example meet all constraints.",
            citations=(
                AnswerCitation(
                    url="https://example.com/table",
                    title="Official table",
                    note="Lists the relevant players and constraints.",
                ),
            ),
        ),
    )

    assert candidate.reference_answer.text.startswith("Ada Example")


def test_trace_and_finalization_contracts_are_deferred_from_this_slice() -> None:
    import harnyx_commons.miner_task_generation as generation

    assert not hasattr(generation, "DomainTweakGenerationTrace")
    assert not hasattr(generation, "DomainTweakFinalizedTask")
