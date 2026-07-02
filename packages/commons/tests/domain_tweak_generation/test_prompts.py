from __future__ import annotations

from datetime import UTC, datetime

from harnyx_commons.domain_tweak_generation.prompts import (
    form_review_prompt,
    phase_instruction,
    question_generation_prompt,
    question_generation_repair_prompt,
    reference_answer_prompt,
    soft_timeout_feedback_prompt,
)
from harnyx_commons.miner_task_generation import (
    DomainTweakFormReview,
    DomainTweakPairInput,
    DomainTweakQuestionCandidate,
)


def test_question_prompt_preserves_form_domain_split_and_excludes_reference_answer() -> None:
    prompt = question_generation_prompt(_pair_input())

    assert "Source form question" in prompt
    assert "Domain seed question" in prompt
    assert "Preserve this question's operation structure" in prompt
    assert "DeepSearchQA" not in prompt
    assert "DeepResearch-9K" not in prompt
    assert "no_generate" in prompt


def test_form_review_prompt_checks_operation_structure() -> None:
    prompt = form_review_prompt(_pair_input(), _candidate())

    assert "Source form question" in prompt
    assert "Compare operation structure" in prompt
    assert "filters, joins, aggregation/comparison" in prompt
    assert "false or unsupported premise" in prompt
    assert "DeepSearchQA" not in prompt


def test_question_repair_prompt_uses_reviewer_feedback_without_internal_context_terms() -> None:
    prompt = question_generation_repair_prompt(_pair_input(), _candidate(), _form_review())

    assert "Previous generated candidate" in prompt
    assert "Independent form-review feedback" in prompt
    assert "Generate a replacement question for the same source/domain pair" in prompt
    assert "The generated question lost the aggregation step." in prompt
    _assert_no_internal_context_terms(prompt)


def test_reference_answer_prompt_uses_a2_answer_strategy_without_internal_context_terms() -> None:
    prompt = reference_answer_prompt(_candidate(), timestamp=_pair_input().timestamp)

    assert "Current timestamp: 2026-06-23T00:00:00+00:00" in prompt
    assert "dedicated deep-research reference-answer writer" in prompt
    assert "false, partially supported, or unresolved premise" in prompt
    assert "Prefer bounded research" in prompt
    assert "complete final set near the top" in prompt
    assert "included-entity proof table or compact proof list" in prompt
    assert "For each included entity, bind every query predicate" in prompt
    assert "candidate-pool membership, inclusion criteria, exclusion criteria not triggered" in prompt
    assert "date/time basis" in prompt
    assert "values used in filters or calculations" in prompt
    assert "candidate pool, filters, and calculations" in prompt
    assert "Discuss excluded or borderline entities only when" in prompt
    assert "A citation note is scorer-visible evidence" in prompt
    assert "compact factual grounding snippet" in prompt
    assert "exactly which visible claim it supports" in prompt
    assert "If one citation does not support all major subclaims" in prompt
    assert "Do not claim broad all-others exclusions or universal negatives" in prompt
    assert "official, primary, canonical, database, or specialized sources" in prompt
    assert "Search separately only for filters" in prompt
    assert "Search procedure" in prompt
    assert "Compact exhaustive-list example" in prompt
    assert "Good:" in prompt
    assert "Proof:" in prompt
    assert "Completeness:" in prompt
    assert "Bad:" in prompt
    _assert_no_internal_context_terms(prompt)


def test_phase_instructions_do_not_name_source_datasets() -> None:
    for phase in ("question_generation", "form_review", "reference_answer"):
        prompt = phase_instruction(phase)
        _assert_no_internal_context_terms(prompt)


def test_soft_timeout_feedback_prompt_pushes_final_answer_without_broad_research() -> None:
    prompt = soft_timeout_feedback_prompt(elapsed_seconds=900.0)

    assert "ran too long" in prompt
    assert "Elapsed wall time: 15 minutes (900 seconds)." in prompt
    assert "Time is almost gone" in prompt
    assert "Do not restart broad research" in prompt
    assert "Return one corrected JSON object only" in prompt
    _assert_no_internal_context_terms(prompt)


def _pair_input() -> DomainTweakPairInput:
    return DomainTweakPairInput(
        pair_id="pair-001",
        deepsearchqa_form_target="Which films meet all constraints?",
        deepresearch9k_domain_target="Football records question",
        timestamp=datetime(2026, 6, 23, tzinfo=UTC),
    )


def _candidate() -> DomainTweakQuestionCandidate:
    return DomainTweakQuestionCandidate(
        question="Which players meet all constraints?",
        short_answer="Ada Example; Ben Example",
        solution_plan="- Find candidate pool\n- Intersect constraints",
    )


def _form_review() -> DomainTweakFormReview:
    return DomainTweakFormReview(
        form_match=False,
        false_premise_status="none",
        reviewer_feedback="The generated question lost the aggregation step.",
        retry_recommended=True,
    )


def _assert_no_internal_context_terms(prompt: str) -> None:
    forbidden_terms = (
        "DeepSearchQA",
        "DeepResearch-9K",
        "Short answer from question generation",
        "A1",
        "A2",
        "champion",
        "judge",
        "scoring",
        "generation trace",
    )
    for term in forbidden_terms:
        assert term not in prompt
